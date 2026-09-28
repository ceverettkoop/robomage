#include "filters.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <vector>

#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../ecs/coordinator.h"
#include "../object_ref.h"
#include "../str_util.h"
#include "../svar_eval.h"
#include "battlefield.h"
#include "characteristics.h"
#include "combat.h"
#include "counters.h"
#include "keywords.h"
#include "lki.h"
#include "spells.h"
#include "types.h"

static std::vector<std::pair<std::string, bool>> filter_tokens(const std::string &spec);

// ── Unified filter matcher (declared in filters.h) ─────────────────────
// A flattened view of the one object under test, populated either from a card's printed
// characteristics or a battlefield permanent's live components, then handed to one shared
// qualifier evaluator. This is the single place the "live for types & P/T, copiable-from-
// card for color & mana value" rule (CR 105/112.7/205/208) is encoded.
namespace {

struct CharView {
    Entity entity = 0;                       // 0 when matching a bare CardData (no identity quals)
    const std::set<Type> *types = nullptr;   // live (permanent) or printed (card) type line
    std::set<Colors> colors;                 // effective (permanent) or printed (card)
    int cmc = 0;                             // mana value (always from the card; CR 112.7)
    bool has_pt = false;                     // object has power/toughness
    int power = 0, toughness = 0;
    bool on_battlefield = false;
    bool is_token = false;
    Zone::Ownership controller = Zone::UNKNOWN;
    Zone::Ownership owner = Zone::UNKNOWN;     // card's owner (YouOwn/OppOwn), from its Zone
    long entered_on_turn = -1;               // -1 when not on the battlefield
    bool is_attacking = false;               // live combat state (battlefield creatures only)
    bool is_blocking = false;
    bool is_unblocked = false;               // an unblocked attacker (CR 509.1h)
    bool is_tapped = false;
    bool has_x_cost = false;                 // printed mana cost contains {X} (Gaddock Teeg's hasXCost)
    bool creature_suppressed = false;        // on battlefield, has "Creature" in its type line but
                                             // is NOT currently a creature (no live Creature component)
    bool entered_by_cast = false;            // a permanent that entered by being cast (wasCastByYou)
};

// The name an object has in its current zone: a permanent's (the face it shows, CR 712.8e),
// else its card's. Empty for an object with neither.
std::string object_name(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e))
        return global_coordinator.GetComponent<Permanent>(e).name;
    if (global_coordinator.entity_has_component<CardData>(e))
        return global_coordinator.GetComponent<CardData>(e).name;
    return "";
}

bool view_has_typeline(const CharView &v, const std::string &name) {
    if (!v.types) return false;
    // CR 702.151b / 702.175d: some battlefield permanents keep "Creature" in their printed type
    // line yet are not currently creatures — a Reconfigure equipment while attached, an Impending
    // permanent that still has time counters. The live Creature component is the source of truth
    // for creature-ness on the battlefield (it is what the static pass strips in those cases), so a
    // "Creature" type query consults it rather than the stale type line. Other type names, and
    // off-battlefield card views, read the type line unchanged.
    if (name == "Creature" && v.creature_suppressed) return false;
    for (const auto &t : *v.types)
        if (t.name == name) return true;
    return false;
}

// True when the object has a permanent card type (CR 110.4a). The "Permanent" filter head must
// require this so an off-battlefield card filter (e.g. Lion Sash's "if it was a permanent card",
// matched against a card now in exile) excludes instants/sorceries. Battlefield objects always
// carry a permanent type in their live type line, so this stays a no-op for permanent_view.
bool view_is_permanent(const CharView &v) {
    if (v.on_battlefield) return true;  // a battlefield object is by definition a permanent (CR 110.4a)
    return view_has_typeline(v, "Artifact") || view_has_typeline(v, "Battle") ||
           view_has_typeline(v, "Creature") || view_has_typeline(v, "Enchantment") ||
           view_has_typeline(v, "Land") || view_has_typeline(v, "Planeswalker");
}

bool color_token(const std::string &q, Colors &c) {
    if (q == "White") { c = WHITE; return true; }
    if (q == "Blue")  { c = BLUE;  return true; }
    if (q == "Black") { c = BLACK; return true; }
    if (q == "Red")   { c = RED;   return true; }
    if (q == "Green") { c = GREEN; return true; }
    return false;
}

// True when the object is one or more colors (CR 105.2a). Every color source (card_colors,
// effective_colors, last-known info) holds only the five colors, so this is non-emptiness.
// Single source for the Colorless / nonColorless qualifier pair.
bool view_has_any_color(const CharView &v) { return !v.colors.empty(); }

// A "power"/"toughness" comparator qualifier (e.g. "toughnessLE2", "powerGE5"): static
// characteristic compared against the object's P/T (CR 208.2 / 107.1). Returns true when `q`
// IS such a qualifier (and writes the result into `ok`); false when `q` is something else.
bool try_pt_qualifier(const CharView &v, const std::string &q, bool &ok) {
    const std::string lead = q.rfind("power", 0) == 0       ? "power"
                             : q.rfind("toughness", 0) == 0 ? "toughness"
                                                            : "";
    if (lead.empty()) return false;
    std::string rest = q.substr(lead.size());
    if (rest.size() < 3) return false;
    std::string op = rest.substr(0, 2), num = rest.substr(2);
    for (char c : num)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    if (!v.has_pt) { ok = false; return true; }  // a P/T filter never matches a P/T-less object
    int lhs = (lead == "power") ? v.power : v.toughness;
    ok = apply_svar_op(lhs, op, std::stoi(num));
    return true;
}

// A counter-count qualifier "counters_<OP><N>_<TYPE>" (e.g. counters_GE1_VOID — "with a void
// counter on it", CR 122.1): the number of TYPE counters on the object compared against N.
// Returns true when `q` IS such a qualifier (and writes the result into `ok`).
bool try_counters_qualifier(const CharView &v, const std::string &q, bool &ok) {
    static const std::string lead = "counters_";
    if (q.rfind(lead, 0) != 0) return false;
    std::string rest = q.substr(lead.size());
    size_t sep = rest.find('_', 2);
    if (rest.size() < 4 || sep == std::string::npos || sep == 2) return false;
    std::string op = rest.substr(0, 2), num = rest.substr(2, sep - 2), type = rest.substr(sep + 1);
    for (char c : num)
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    ok = v.entity != 0 && apply_svar_op(object_counters(v.entity, type), op, std::stoi(num));
    return true;
}

// Main card types, for the non<CardType> negation (CR 110.4a + the spell-only types).
const char *const kCardTypes[] = {"Land", "Creature", "Artifact", "Enchantment",
                                  "Planeswalker", "Battle", "Instant", "Sorcery", "Tribal"};

void warn_unknown_qualifier(const std::string &q) {
    static std::set<std::string> warned;  // once per distinct token — this is on the SBA/legality hot path
    if (warned.insert(q).second)
        printf("WARNING: Unrecognized filter qualifier '%s' (fail-closed, matched nothing)\n", q.c_str());
}

// Evaluate ONE '.'/'+'-joined qualifier token. Dynamic mana-value bounds are applied once by
// the caller, so cmc tokens here only honour the legacy "cmcLEX" (x_paid) form.
bool eval_qualifier(const CharView &v, const MatchCtx &ctx, const std::string &q) {
    if (q.empty()) return true;
    // "!<qualifier>" negates it (Doomsday's Card.!IsRemembered); "!token" is spelled out below.
    if (q[0] == '!' && q != "!token") return !eval_qualifier(v, ctx, q.substr(1));
    // identity / state keywords ------------------------------------------------
    if (q == "IsRemembered") return refs_contain(cur_game.remembered_entities, v.entity);
    // IsImprinted — one of the cards the resolving ability imprinted (Atraxa's revealed pile).
    if (q == "IsImprinted") return refs_contain(cur_game.imprinted_entities, v.entity);
    // NamedCard — the object has the name chosen by a preceding name-a-card effect (Cabal
    // Therapy's discard, CR 201.4); nothing matches when no name was chosen.
    if (q == "NamedCard")
        return v.entity != 0 && !cur_game.named_card.empty() && object_name(v.entity) == cur_game.named_card;
    if (q == "Other")        return ctx.source == 0 || v.entity != ctx.source;
    if (q == "Self")         return ctx.source != 0 && v.entity == ctx.source;
    if (q == "nonChosenCard") return !cur_game.chosen_cards.count(v.entity);
    // targetedBy — the object IS the card the resolving ability chain targeted (Cloak and
    // Dagger, Entwined's exile filter alternative "Card.targetedBy" matches exactly the
    // creature its DBPump sub chose). Fails closed when no chain target is in context.
    if (q == "targetedBy")   return ctx.chain_target != 0 && v.entity == ctx.chain_target;
    if (q == "YouCtrl")      return !v.on_battlefield || v.controller == ctx.controller;
    if (q == "OppCtrl")      return !v.on_battlefield || v.controller != ctx.controller;
    // "ControlledBy TriggeredDefendingPlayer" (Forge): the object is controlled by the defending
    // player of the attack that fired this trigger (Frenzied Trapbreaker's attack-trigger Destroy).
    // The trigger is controlled by the attacking player (ctx.controller); in the two-player engine
    // the defending player is that controller's sole opponent, so this is OppCtrl semantics.
    if (q == "ControlledBy TriggeredDefendingPlayer")
        return !v.on_battlefield || v.controller != ctx.controller;
    // "ControlledBy ParentTarget" (Forge): the object is controlled by the player the parent
    // ability targeted (Cloak and Dagger, Entwined: "up to one target creature they control").
    if (q == "ControlledBy ParentTarget")
        return !v.on_battlefield ||
               (ctx.targeted_player != Zone::UNKNOWN && v.controller == ctx.targeted_player);
    // Ownership (CR 108.3) — distinct from control; meaningful for cards in any zone
    // (e.g. Karn's -2 "an artifact card you own from outside the game or in exile").
    // Lenient when either side is unknown (bare CardData / no "you" supplied), mirroring
    // the YouCtrl off-battlefield convention.
    if (q == "YouOwn")  return ctx.controller == Zone::UNKNOWN || v.owner == Zone::UNKNOWN
                                || v.owner == ctx.controller;
    if (q == "OppOwn")  return ctx.controller == Zone::UNKNOWN || v.owner == Zone::UNKNOWN
                                || v.owner != ctx.controller;
    if (q == "token")        return v.is_token;
    if (q == "nonToken" || q == "!token") return !v.is_token;
    if (q == "ThisTurnEntered") return v.on_battlefield && entered_battlefield_this_turn(v.entered_on_turn);
    // wasCastByYou — the permanent entered by being cast (The One Ring's "if you cast it"); its
    // controller cast it, since a permanent spell enters under its caster's control (CR 608.3a).
    if (q == "wasCastByYou") return v.on_battlefield && v.entered_by_cast;
    // live combat / tap state (e.g. Guide of Souls' ValidTgts$ Creature.attacking) — only a
    // battlefield permanent can be in these states; a card view leaves them false.
    if (q == "attacking") return v.is_attacking;
    if (q == "blocking")  return v.is_blocking;
    if (q == "unblocked") return v.is_unblocked;
    if (q == "tapped")    return v.is_tapped;
    if (q == "untapped")  return !v.is_tapped;
    if (q == "Basic")        return v.types && has_basic_supertype(*v.types);
    if (q == "nonBasic")     return v.types && !has_basic_supertype(*v.types);
    if (q == "Colorless")    return !view_has_any_color(v);  // CR 105.2c
    if (q == "hasXCost")     return v.has_x_cost;       // {X} in the printed mana cost (Gaddock Teeg)
    // hasABasicLandType — the object's subtypes include one of the five basic land types
    // (CR 305.6; Boseiju, Who Endures' compensation search "a land card with a basic land type").
    if (q == "hasABasicLandType") return v.types && has_a_basic_land_type(*v.types);
    // mana-value family (dynamic bound applied once by the caller) --------------
    if (q.rfind("cmc", 0) == 0) {
        // cmcLEX reads the X paid, unless the caller resolved X itself and supplied it as the
        // dynamic bound (Birthing Ritual's X = 1 + the sacrificed creature's mana value).
        if (q == "cmcLEX") return ctx.cmc_bound >= 0 || v.cmc <= current_x_paid();
        return true;  // cmcEQX / cmcLE3 / … enforced via ctx.cmc_bound
    }
    // dynamic power/toughness vs SVar X (Ensnaring Bridge: Creature.powerGTX — "power greater
    // than the number of cards in your hand"). The X value is supplied by the caller in
    // ctx.x_bound (the static source's controller's hand size). A P/T-less object never matches,
    // and with no X provided the qualifier can't be evaluated, so it fails closed.
    {
        const std::string lead = q.rfind("power", 0) == 0       ? "power"
                                 : q.rfind("toughness", 0) == 0 ? "toughness"
                                                                : "";
        if (!lead.empty()) {
            std::string rest = q.substr(lead.size());
            if (rest.size() == 3 && rest[2] == 'X') {  // <attr> + 2-letter op + "X"
                if (!v.has_pt || ctx.x_bound == INT_MIN) return false;
                int lhs = (lead == "power") ? v.power : v.toughness;
                return apply_svar_op(lhs, rest.substr(0, 2), ctx.x_bound);
            }
        }
    }
    // power/toughness comparator ----------------------------------------------
    { bool ok = false; if (try_pt_qualifier(v, q, ok)) return ok; }
    { bool ok = false; if (try_counters_qualifier(v, q, ok)) return ok; }
    // positive color ----------------------------------------------------------
    { Colors c; if (color_token(q, c)) return v.colors.count(c) > 0; }
    // negations ---------------------------------------------------------------
    // The "non" prefix and the card-type word are matched ASCII case-insensitively:
    // Forge scripts vary the token's casing (Cityscape Leveler writes "nonland" where
    // Abrupt Decay writes "nonLand"), and a case-sensitive miss here used to fall
    // through to the non<subtype> arm — which matches nothing for "land", silently
    // turning the restriction into a pass for every permanent, lands included.
    if (q.size() > 3 && ascii_lower(q.substr(0, 3)) == "non") {
        std::string rest = q.substr(3);
        Colors c;
        if (color_token(rest, c)) return v.colors.count(c) == 0;       // non<Color>  (fixes the old bug)
        const std::string rest_lc = ascii_lower(rest);
        // nonColorless = "one or more colors" (Ugin's exiles, All Is Dust). "Colorless" is not a
        // color, so it must be handled here — falling through to the non<subtype> arm made the
        // qualifier match EVERY object (nothing has "Colorless" in its type line).
        if (rest_lc == "colorless") return view_has_any_color(v);
        for (const char *ct : kCardTypes)
            if (rest_lc == ascii_lower(ct)) return !view_has_typeline(v, ct);  // non<CardType>
        return !view_has_typeline(v, rest);                           // non<subtype>
    }
    // with<Keyword> — the object currently has the named keyword ability (Pick Your Poison's
    // SacValid$ Creature.withFlying). Forge writes multi-word keywords without spaces
    // (withFirstStrike), so re-insert a space before each interior capital to recover the stored
    // keyword string ("FirstStrike" → "First Strike"). permanent_has_keyword reads the effective
    // keyword list for a battlefield permanent and falls back to the printed CardData keywords
    // for an off-battlefield card view, so this is correct in either zone.
    if (q.rfind("with", 0) == 0 && q.size() > 4 &&
        std::isupper(static_cast<unsigned char>(q[4]))) {
        std::string kw;
        for (size_t i = 4; i < q.size(); i++) {
            if (i > 4 && std::isupper(static_cast<unsigned char>(q[i]))) kw += ' ';
            kw += q[i];
        }
        return v.entity != 0 && permanent_has_keyword(v.entity, kw.c_str());
    }
    // inZone<Zone> — the object currently sits in the named zone. A multi-zone search filter
    // uses it to restrict ONE OR-alternative to a single origin (Cloak and Dagger, Entwined's
    // "Card.nonLand+inZoneHand": only the hand alternative takes any nonland card; the
    // battlefield alternative is the targetedBy creature). Read from the live Zone component;
    // a bare CardData view (entity 0) fails closed.
    if (q.rfind("inZone", 0) == 0 && q.size() > 6) {
        if (v.entity == 0 || !global_coordinator.entity_has_component<Zone>(v.entity)) return false;
        Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(v.entity).location;
        const std::string zn = q.substr(6);
        return (zn == "Hand" && loc == Zone::HAND) ||
               (zn == "Battlefield" && loc == Zone::BATTLEFIELD) ||
               (zn == "Library" && loc == Zone::LIBRARY) ||
               (zn == "Graveyard" && loc == Zone::GRAVEYARD) ||
               (zn == "Exile" && loc == Zone::EXILE) ||
               (zn == "Stack" && loc == Zone::STACK);
    }
    // "ManaCostN" — the object's mana value equals exactly N (Urza's Saga's chapter III tutor:
    // Artifact.ManaCost0 / Artifact.ManaCost1 = an artifact with mana cost {0} or {1}). Distinct
    // from the cmc family above (which carries a comparator); ManaCostN is the equality form Forge
    // emits as a bare property. A P/T-less / costless object reads mana value 0.
    if (q.rfind("ManaCost", 0) == 0 && q.size() > 8) {
        const std::string num = q.substr(8);
        if (std::all_of(num.begin(), num.end(),
                        [](unsigned char ch) { return std::isdigit(ch); }))
            return v.cmc == std::stoi(num);
    }
    // leftover: a PascalCase token is a type-line (type/supertype/subtype) name to require;
    // anything else is a qualifier we don't implement → fail closed and warn once.
    if (std::isupper(static_cast<unsigned char>(q[0]))) return view_has_typeline(v, q);
    warn_unknown_qualifier(q);
    return false;
}

// Match the view against ONE ';'-free alternative: "head[.q][+q]…".
bool eval_alternative(const CharView &v, const MatchCtx &ctx, const std::string &alt) {
    size_t sep = alt.find_first_of(".+");
    std::string head = alt.substr(0, sep);
    // "Self"/"Other" as a BARE head are identity qualifiers, not type-line names — Forge writes
    // SacValid$ Self (Uro: "sacrifice it") and "Other" the same way. Treat them like Card.<head>:
    // no type-line requirement; the identity check is evaluated as a qualifier just below.
    bool head_is_identity = (head == "Self" || head == "Other");
    if (!(head.empty() || head == "Card" || head == "Permanent" || head == "Spell" || head_is_identity) &&
        !view_has_typeline(v, head))
        return false;
    // "Permanent" head (CR 110.4a): require an actual permanent card type. No-op for battlefield
    // objects (always permanents); excludes instants/sorceries when matching a card in another zone.
    if (head == "Permanent" && !view_is_permanent(v)) return false;
    if (head_is_identity && !eval_qualifier(v, ctx, head))
        return false;
    if (sep != std::string::npos) {
        std::string rest = alt.substr(sep + 1);
        size_t p = 0;
        while (p <= rest.size()) {
            size_t nx = rest.find_first_of(".+", p);
            if (nx == std::string::npos) nx = rest.size();
            std::string q = rest.substr(p, nx - p);
            p = nx + 1;
            if (!q.empty() && !eval_qualifier(v, ctx, q)) return false;
        }
    }
    // Dynamic mana-value bound supplied by the caller (Aether Vial: MV == charge count).
    if (ctx.cmc_bound >= 0 && !apply_svar_op(v.cmc, ctx.cmc_op, ctx.cmc_bound)) return false;
    return true;
}

bool match_filter_core(const CharView &v, const std::string &spec, const MatchCtx &ctx) {
    if (spec.empty()) return false;  // an empty spec matches nothing (existing convention)
    size_t pp = 0;
    while (pp <= spec.size()) {
        // ';' and ',' are both top-level OR separators between alternatives. Forge writes
        // "Creature,Planeswalker" (SacValid$ on Archon of Cruelty) with a comma; the engine
        // historically only split on ';', so a comma-joined list matched nothing. No qualifier
        // token contains a comma, so treating ',' as OR here is safe and additive.
        size_t sc = spec.find_first_of(";,", pp);
        if (sc == std::string::npos) sc = spec.size();
        std::string alt = spec.substr(pp, sc - pp);
        pp = sc + 1;
        if (!alt.empty() && eval_alternative(v, ctx, alt)) return true;  // ';'/',' is OR
    }
    return false;
}

CharView card_view(Entity e, const CardData &cd) {
    CharView v;
    v.entity = e;
    v.types = &cd.types;
    v.colors = card_colors(cd);
    v.cmc = object_mana_value(e, cd);
    v.has_pt = true;  // printed P/T (a head type guard keeps P/T filters scoped to creatures)
    v.power = static_cast<int>(cd.power);
    v.toughness = static_cast<int>(cd.toughness);
    v.has_x_cost = cd.has_x_cost;
    if (e != 0 && global_coordinator.entity_has_component<Zone>(e))
        v.owner = global_coordinator.GetComponent<Zone>(e).owner;
    return v;
}

// Mana value of a permanent (CR 112.7), read from the face mana_value_face picks. A token (no
// card) has the mana value it copied (Token::mana_value, CR 707.2), 0 for a scripted token.
void set_permanent_mana_value(CharView &v, Entity e, bool transformed) {
    if (!global_coordinator.entity_has_component<CardData>(e)) {
        if (global_coordinator.entity_has_component<Token>(e))
            v.cmc = global_coordinator.GetComponent<Token>(e).mana_value;
        return;
    }
    const auto &cd = global_coordinator.GetComponent<CardData>(e);
    const CardData &mv_face = mana_value_face(cd, transformed);
    v.cmc = card_mana_value(mv_face);
    v.has_x_cost = mv_face.has_x_cost;
}

CharView permanent_view(Entity e, const Permanent &perm) {
    CharView v;
    v.entity = e;
    v.types = &perm.types;                 // live type line (type-changing effects write here)
    v.colors = effective_colors(e);        // CR 105: color from the card (no live color layer)
    v.is_token = perm.is_token;
    v.controller = perm.controller;
    if (global_coordinator.entity_has_component<Zone>(e))
        v.owner = global_coordinator.GetComponent<Zone>(e).owner;
    v.entered_on_turn = static_cast<long>(perm.entered_on_turn);
    v.on_battlefield = true;
    v.is_tapped = perm.is_tapped;
    v.entered_by_cast = perm.entered_by_cast;
    if (global_coordinator.entity_has_component<Creature>(e)) {
        v.has_pt = true;
        v.power = effective_power(e);
        v.toughness = effective_toughness(e);
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        v.is_attacking = cr.is_attacking;
        v.is_blocking = cr.is_blocking;
        v.is_unblocked = is_unblocked_attacker(e);
    } else {
        // On the battlefield with no live Creature component: any "Creature" still in the type line
        // is suppressed (CR 702.151b reconfigure-while-attached / 702.175d impending). See
        // view_has_typeline — a "Creature" type query must read the component, not the stale line.
        v.creature_suppressed = true;
    }
    set_permanent_mana_value(v, e, perm.transformed);
    return v;
}

// The object `e` as it last existed on the battlefield (CR 603.10a / 608.2h): its last-known
// type line, controller, colors and P/T. Its owner and mana value come from the card, which
// keeps its identity (a vanished token is owned by its last controller and has mana value 0).
CharView lki_view(Entity e, const LastKnownInfo &lki) {
    CharView v;
    v.entity = e;
    v.types = &lki.types;
    v.colors = lki.colors;
    v.is_token = lki.is_token;
    v.controller = lki.controller;
    v.owner = global_coordinator.entity_has_component<Zone>(e)
                  ? global_coordinator.GetComponent<Zone>(e).owner
                  : lki.controller;
    v.on_battlefield = true;
    v.entered_by_cast = lki.entered_by_cast;
    v.has_pt = type_set_has(lki.types, "Creature");
    v.power = lki.power;
    v.toughness = lki.toughness;
    set_permanent_mana_value(v, e, lki.transformed);
    return v;
}

}  // namespace

bool card_matches_filter(const CardData &cd, const std::string &spec, const MatchCtx &ctx) {
    return match_filter_core(card_view(0, cd), spec, ctx);
}

bool card_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    return match_filter_core(card_view(e, global_coordinator.GetComponent<CardData>(e)), spec, ctx);
}

bool permanent_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx) {
    if (!is_battlefield_permanent(e)) return false;
    return match_filter_core(permanent_view(e, global_coordinator.GetComponent<Permanent>(e)), spec, ctx);
}

bool object_matches_filter(Entity e, const std::string &spec, const MatchCtx &ctx) {
    return is_battlefield_permanent(e) ? permanent_matches_filter(e, spec, ctx)
                                       : card_matches_filter(e, spec, ctx);
}

bool zone_change_object_matches(Entity e, Zone::ZoneValue origin, Zone::ZoneValue destination,
                                const std::string &spec, const MatchCtx &ctx) {
    bool left_battlefield = (origin == Zone::BATTLEFIELD);
    if (!left_battlefield && destination == Zone::BATTLEFIELD && is_battlefield_permanent(e))
        return permanent_matches_filter(e, spec, ctx);
    if (left_battlefield || destination == Zone::BATTLEFIELD) {
        if (const LastKnownInfo *lki = departed_lki_for(e))
            return match_filter_core(lki_view(e, *lki), spec, ctx);
    }
    return card_matches_filter(e, spec, ctx);
}

// The '.'/'+'/','/';'-delimited tokens of a filter spec, each flagged whether it is the head
// (first) token of its OR alternative.
static std::vector<std::pair<std::string, bool>> filter_tokens(const std::string &spec) {
    std::vector<std::pair<std::string, bool>> out;
    size_t p = 0;
    bool head = true;
    while (p <= spec.size()) {
        size_t nx = spec.find_first_of(".+,;", p);
        if (nx == std::string::npos) nx = spec.size();
        out.emplace_back(spec.substr(p, nx - p), head);
        head = (nx < spec.size() && (spec[nx] == ',' || spec[nx] == ';'));
        p = nx + 1;
    }
    return out;
}

bool filter_names_token(const std::string &spec, const std::string &token) {
    for (const auto &t : filter_tokens(spec))
        if (t.first == token) return true;
    return false;
}

std::string owner_relative_filter(const std::string &spec) {
    std::string out;
    size_t p = 0;
    while (p <= spec.size()) {
        size_t nx = spec.find_first_of(".+,;", p);
        if (nx == std::string::npos) nx = spec.size();
        std::string tok = spec.substr(p, nx - p);
        if (tok == "YouCtrl") tok = "YouOwn";
        else if (tok == "OppCtrl") tok = "OppOwn";
        out += tok;
        if (nx < spec.size()) out += spec[nx];
        p = nx + 1;
    }
    return out;
}

bool filter_has_head(const std::string &spec, const std::string &head) {
    for (const auto &t : filter_tokens(spec))
        if (t.second && t.first == head) return true;
    return false;
}

bool permanent_matches_any(Entity e, const std::string &comma_or_spec,
                           const MatchCtx &ctx) {
    for (const auto &alt : split(comma_or_spec, ',', /*skip_empty=*/true))
        if (permanent_matches_filter(e, alt, ctx)) return true;
    return false;
}

bool card_matches_any(Entity e, const std::string &comma_or_spec,
                      const MatchCtx &ctx) {
    for (const auto &alt : split(comma_or_spec, ',', /*skip_empty=*/true))
        if (card_matches_filter(e, alt, ctx)) return true;
    return false;
}

void extract_static_cmc_bound(const std::string &spec, MatchCtx &ctx) {
    size_t p = 0;
    while (p < spec.size()) {
        size_t pos = spec.find("cmc", p);
        if (pos == std::string::npos) return;
        // A numeric cmc qualifier is "cmc" + 2-letter op + at least one digit (cmcGE7).
        if (pos + 5 < spec.size() &&
            std::isdigit(static_cast<unsigned char>(spec[pos + 5]))) {
            size_t e = pos + 5;
            while (e < spec.size() && std::isdigit(static_cast<unsigned char>(spec[e]))) ++e;
            ctx.cmc_op = spec.substr(pos + 3, 2);
            ctx.cmc_bound = std::stoi(spec.substr(pos + 5, e - (pos + 5)));
            return;
        }
        p = pos + 3;
    }
}

bool color_set_passes(const std::string &vt, const std::set<Colors> &colors) {
    static const struct { const char *tok; Colors col; } table[] = {
        {".White", WHITE}, {".Blue", BLUE}, {".Black", BLACK}, {".Red", RED}, {".Green", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && !colors.count(e.col)) return false;
    return true;
}

bool color_set_passes_noncolor(const std::string &vt, const std::set<Colors> &colors) {
    static const struct { const char *tok; Colors col; } table[] = {
        {"nonWhite", WHITE}, {"nonBlue", BLUE}, {"nonBlack", BLACK}, {"nonRed", RED}, {"nonGreen", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && colors.count(e.col)) return false;
    return true;
}

bool type_set_passes_nontype(const std::string &spec, const std::set<Type> &types) {
    static const char *kCardTypes[] = {
        "Land", "Creature", "Artifact", "Enchantment", "Planeswalker",
        "Battle", "Instant", "Sorcery", "Tribal"};
    const std::string spec_lc = ascii_lower(spec);
    for (const char *ct : kCardTypes) {
        std::string tok = ascii_lower(std::string("non") + ct);
        if (spec_lc.find(tok) == std::string::npos) continue;
        for (const auto &t : types)
            if (t.kind == TYPE && t.name == ct) return false;
    }
    return true;
}
