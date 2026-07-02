#include "game_queries.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

#include "classes/game.h"
#include "svar_eval.h"

extern Coordinator global_coordinator;

// The effective_* accessors implement CR 608.2h: use the object's current information while it
// is in the zone it is expected to be in (the battlefield, for a permanent's continuous-effect-
// and counter-modified characteristics), otherwise its last-known information. They live here
// rather than inline in game_queries.h so the header does not need to depend on game.h (and the
// cur_game.last_known_info store).

// Look up a leaving-the-battlefield snapshot, if one was captured for `e`.
static const LastKnownInfo *lki_for(Entity e) {
    auto it = cur_game.last_known_info.find(e);
    return it == cur_game.last_known_info.end() ? nullptr : &it->second;
}

int effective_power(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).power);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->power;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).power);
    return 0;
}

int effective_toughness(Entity e) {
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<Creature>(e))
        return static_cast<int>(global_coordinator.GetComponent<Creature>(e).toughness);
    if (const LastKnownInfo *lki = lki_for(e)) return lki->toughness;
    if (global_coordinator.entity_has_component<CardData>(e))
        return static_cast<int>(global_coordinator.GetComponent<CardData>(e).toughness);
    return 0;
}

std::set<Colors> effective_colors(Entity e) {
    // Layer-5 (613.1e) global color-changing override (Mycosynth Lattice: every card colorless).
    // Checked first so it wins over the printed/last-known color in every zone the static reaches.
    std::set<Colors> override_colors;
    if (setcolor_override_for(e, override_colors)) return override_colors;
    // On the battlefield, effective color is the layer-5 (613.1e) result. With no active color-
    // changing static this reads the printed colors; the override above is the single seam such a
    // static plugs into without touching any consumer.
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    if (const LastKnownInfo *lki = lki_for(e)) return lki->colors;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    // A token (no CardData) is colored by its color indicator (Token::explicit_colors, CR 111.4),
    // not a mana cost. With no active SetColor override (checked first, above) this is its
    // effective color — so a colored token reads correctly to the .Red/.Blue filter qualifiers
    // (e.g. Blue Elemental Blast's "target red permanent" can see a red Warrior token). Filtered
    // to the five real colors so a stray COLORLESS/GENERIC sentinel never leaks in.
    if (global_coordinator.entity_has_component<Token>(e)) {
        std::set<Colors> cols;
        const auto &tk = global_coordinator.GetComponent<Token>(e).explicit_colors;
        for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
            if (tk.count(c)) cols.insert(c);
        return cols;
    }
    return {};
}

// Strip Permanent/Creature/Damage from a card no longer on the battlefield (or re-entering it as
// a new object, CR 400.7). Clears equipment/aura attachment links first so no dangling reference
// survives (704.5n): a creature leaving unattaches its equipment (which stays), an equipment
// leaving clears the host's back-link. Contract documented at the declaration in game_queries.h.
void strip_permanent_components(Entity entity) {
    if (global_coordinator.entity_has_component<Permanent>(entity)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.equipped_by != 0 &&
            global_coordinator.entity_has_component<Permanent>(perm.equipped_by)) {
            global_coordinator.GetComponent<Permanent>(perm.equipped_by).equipped_to = 0;
        }
        if (perm.equipped_to != 0 &&
            global_coordinator.entity_has_component<Permanent>(perm.equipped_to)) {
            global_coordinator.GetComponent<Permanent>(perm.equipped_to).equipped_by = 0;
        }
        global_coordinator.RemoveComponent<Permanent>(entity);
    }
    if (global_coordinator.entity_has_component<Creature>(entity))
        global_coordinator.RemoveComponent<Creature>(entity);
    if (global_coordinator.entity_has_component<Damage>(entity))
        global_coordinator.RemoveComponent<Damage>(entity);
}

// ── Unified filter matcher (declared in game_queries.h) ─────────────────────
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
    bool is_tapped = false;
    bool has_x_cost = false;                 // printed mana cost contains {X} (Gaddock Teeg's hasXCost)
    bool creature_suppressed = false;        // on battlefield, has "Creature" in its type line but
                                             // is NOT currently a creature (no live Creature component)
};

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

// True when the object is one or more colors (CR 105.2a) — membership in the five real colors
// only. The colors set can carry a COLORLESS sentinel (a Devoid card's explicit_colors is
// {COLORLESS}), which must read as "has no colors", so a bare empty() test is not equivalent.
// Single source for the Colorless / nonColorless qualifier pair.
bool view_has_any_color(const CharView &v) {
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
        if (v.colors.count(c)) return true;
    return false;
}

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
    // identity / state keywords ------------------------------------------------
    if (q == "IsRemembered") {
        for (auto re : cur_game.remembered_entities)
            if (re == v.entity) return true;
        return false;
    }
    if (q == "Other")        return ctx.source == 0 || v.entity != ctx.source;
    if (q == "Self")         return ctx.source != 0 && v.entity == ctx.source;
    if (q == "nonChosenCard") return !cur_game.chosen_cards.count(v.entity);
    if (q == "YouCtrl")      return !v.on_battlefield || v.controller == ctx.controller;
    if (q == "OppCtrl")      return !v.on_battlefield || v.controller != ctx.controller;
    // "ControlledBy TriggeredDefendingPlayer" (Forge): the object is controlled by the defending
    // player of the attack that fired this trigger (Frenzied Trapbreaker's attack-trigger Destroy).
    // The trigger is controlled by the attacking player (ctx.controller); in the two-player engine
    // the defending player is that controller's sole opponent, so this is OppCtrl semantics.
    if (q == "ControlledBy TriggeredDefendingPlayer")
        return !v.on_battlefield || v.controller != ctx.controller;
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
    if (q == "ThisTurnEntered") return v.on_battlefield && v.entered_on_turn == static_cast<long>(cur_game.turn);
    // live combat / tap state (e.g. Guide of Souls' ValidTgts$ Creature.attacking) — only a
    // battlefield permanent can be in these states; a card view leaves them false.
    if (q == "attacking") return v.is_attacking;
    if (q == "blocking")  return v.is_blocking;
    if (q == "tapped")    return v.is_tapped;
    if (q == "untapped")  return !v.is_tapped;
    if (q == "Basic")        return v.types && has_basic_supertype(*v.types);
    if (q == "nonBasic")     return v.types && !has_basic_supertype(*v.types);
    if (q == "Colorless")    return !view_has_any_color(v);  // CR 105.2c
    if (q == "hasXCost")     return v.has_x_cost;       // {X} in the printed mana cost (Gaddock Teeg)
    // mana-value family (dynamic bound applied once by the caller) --------------
    if (q.rfind("cmc", 0) == 0) {
        if (q == "cmcLEX") return v.cmc <= static_cast<int>(cur_game.x_paid);
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
        size_t sc = spec.find(';', pp);
        if (sc == std::string::npos) sc = spec.size();
        std::string alt = spec.substr(pp, sc - pp);
        pp = sc + 1;
        if (!alt.empty() && eval_alternative(v, ctx, alt)) return true;  // ';' is OR
    }
    return false;
}

CharView card_view(Entity e, const CardData &cd) {
    CharView v;
    v.entity = e;
    v.types = &cd.types;
    v.colors = card_colors(cd);
    v.cmc = card_mana_value(cd);
    v.has_pt = true;  // printed P/T (a head type guard keeps P/T filters scoped to creatures)
    v.power = static_cast<int>(cd.power);
    v.toughness = static_cast<int>(cd.toughness);
    v.has_x_cost = cd.has_x_cost;
    if (e != 0 && global_coordinator.entity_has_component<Zone>(e))
        v.owner = global_coordinator.GetComponent<Zone>(e).owner;
    return v;
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
    if (global_coordinator.entity_has_component<Creature>(e)) {
        v.has_pt = true;
        v.power = effective_power(e);
        v.toughness = effective_toughness(e);
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        v.is_attacking = cr.is_attacking;
        v.is_blocking = cr.is_blocking;
    } else {
        // On the battlefield with no live Creature component: any "Creature" still in the type line
        // is suppressed (CR 702.151b reconfigure-while-attached / 702.175d impending). See
        // view_has_typeline — a "Creature" type query must read the component, not the stale line.
        v.creature_suppressed = true;
    }
    if (global_coordinator.entity_has_component<CardData>(e)) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        v.cmc = card_mana_value(cd);  // CR 112.7
        v.has_x_cost = cd.has_x_cost;
    }
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

int count_battlefield_matching(const std::string &filter_spec, Zone::Ownership controller,
                               Entity source) {
    MatchCtx mctx;
    mctx.controller = controller;  // the "you" reference for YouCtrl/OppCtrl in the spec
    mctx.source = source;          // for source-relative qualifiers (e.g. +Other, sameName)
    int count = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e) {
        if (!is_battlefield_permanent(e)) continue;  // control is enforced by the filter
        if (permanent_matches_filter(e, filter_spec, mctx)) count++;
    }
    return count;
}

// ── Defined$ player resolution (declared in game_queries.h) ─────────────────
Zone::Ownership source_controller(Entity source) {
    if (global_coordinator.entity_has_component<Permanent>(source))
        return global_coordinator.GetComponent<Permanent>(source).controller;
    if (global_coordinator.entity_has_component<Zone>(source))
        return global_coordinator.GetComponent<Zone>(source).owner;
    return Zone::UNKNOWN;
}

bool player_protected_from_source(Entity player_entity, Entity /*source*/) {
    if (cur_game.player_protection_from_everything.empty()) return false;
    Zone::Ownership prot =
        (player_entity == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    for (const auto &p : cur_game.player_protection_from_everything) {
        if (p.player != prot) continue;
        // CR 702.16: "protection from everything" is protection from ALL sources, including the
        // protected player's OWN sources — not only an opponent's. So the grant prevents the
        // damage/targeting regardless of who controls the source.
        return true;
    }
    return false;
}

bool permanent_protected_from_colored_spell_source(Entity perm_target, Entity source) {
    if (!global_coordinator.entity_has_component<Creature>(perm_target)) return false;
    if (!has_protection_from_colored_spells(global_coordinator.GetComponent<Creature>(perm_target)))
        return false;
    // The source must be a spell on the stack (not a creature dealing combat damage, nor an
    // activated/triggered ability) and one or more colors (CR 702.16a: the quality is a colored
    // spell). A colorless spell's damage is not prevented.
    if (!global_coordinator.entity_has_component<Spell>(source)) return false;
    return !effective_colors(source).empty();
}

Zone::Ownership last_known_controller(Entity e) {
    if (global_coordinator.entity_has_component<Zone>(e)) {
        Zone::Ownership c = global_coordinator.GetComponent<Zone>(e).controller;
        if (c != Zone::UNKNOWN) return c;  // still on the battlefield (or wherever Zone records it)
    }
    if (global_coordinator.entity_has_component<Permanent>(e))
        return global_coordinator.GetComponent<Permanent>(e).controller;  // mid-resolution, pre-SBA strip
    if (const LastKnownInfo *lki = lki_for(e)) return lki->controller;     // already left the battlefield
    return Zone::UNKNOWN;
}

static Zone::Ownership opponent_of(Zone::Ownership p) {
    if (p == Zone::PLAYER_A) return Zone::PLAYER_B;
    if (p == Zone::PLAYER_B) return Zone::PLAYER_A;
    return Zone::UNKNOWN;
}

Zone::Ownership resolve_defined_player(const Ability &ab) {
    if (ab.defined_you)                 return source_controller(ab.source);
    if (ab.defined_each_opponent)       return opponent_of(source_controller(ab.source));
    if (ab.defined_targeted_controller) return ab.target != 0 ? last_known_controller(ab.target) : Zone::UNKNOWN;
    if (ab.defined_triggered_activator) return ab.triggered_activator;
    return Zone::UNKNOWN;
}
