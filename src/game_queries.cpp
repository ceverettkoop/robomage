#include "game_queries.h"

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
    // On the battlefield, effective color is the layer-5 (613.1e) result. No current-vocab card
    // changes color, so this reads the printed colors today; this is the single seam a future
    // layer-5 color-changing effect plugs into without touching any consumer.
    if (is_battlefield_permanent(e) && global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    if (const LastKnownInfo *lki = lki_for(e)) return lki->colors;
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_colors(global_coordinator.GetComponent<CardData>(e));
    return {};
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
    long entered_on_turn = -1;               // -1 when not on the battlefield
    bool is_attacking = false;               // live combat state (battlefield creatures only)
    bool is_blocking = false;
    bool is_tapped = false;
    bool has_x_cost = false;                 // printed mana cost contains {X} (Gaddock Teeg's hasXCost)
};

bool view_has_typeline(const CharView &v, const std::string &name) {
    if (!v.types) return false;
    for (const auto &t : *v.types)
        if (t.name == name) return true;
    return false;
}

bool color_token(const std::string &q, Colors &c) {
    if (q == "White") { c = WHITE; return true; }
    if (q == "Blue")  { c = BLUE;  return true; }
    if (q == "Black") { c = BLACK; return true; }
    if (q == "Red")   { c = RED;   return true; }
    if (q == "Green") { c = GREEN; return true; }
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
    if (q == "nonChosenCard") return !cur_game.chosen_cards.count(v.entity);
    if (q == "YouCtrl")      return !v.on_battlefield || v.controller == ctx.controller;
    if (q == "OppCtrl")      return !v.on_battlefield || v.controller != ctx.controller;
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
    if (q == "Colorless")    return v.colors.empty();  // CR 105.2c
    if (q == "hasXCost")     return v.has_x_cost;       // {X} in the printed mana cost (Gaddock Teeg)
    // mana-value family (dynamic bound applied once by the caller) --------------
    if (q.rfind("cmc", 0) == 0) {
        if (q == "cmcLEX") return v.cmc <= static_cast<int>(cur_game.x_paid);
        return true;  // cmcEQX / cmcLE3 / … enforced via ctx.cmc_bound
    }
    // power/toughness comparator ----------------------------------------------
    { bool ok = false; if (try_pt_qualifier(v, q, ok)) return ok; }
    // positive color ----------------------------------------------------------
    { Colors c; if (color_token(q, c)) return v.colors.count(c) > 0; }
    // negations ---------------------------------------------------------------
    if (q.size() > 3 && q.compare(0, 3, "non") == 0) {
        std::string rest = q.substr(3);
        Colors c;
        if (color_token(rest, c)) return v.colors.count(c) == 0;       // non<Color>  (fixes the old bug)
        for (const char *ct : kCardTypes)
            if (rest == ct) return !view_has_typeline(v, ct);          // non<CardType>
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
    if (!(head.empty() || head == "Card" || head == "Permanent" || head == "Spell") &&
        !view_has_typeline(v, head))
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
    v.cmc = static_cast<int>(cd.mana_cost.size());
    v.has_pt = true;  // printed P/T (a head type guard keeps P/T filters scoped to creatures)
    v.power = static_cast<int>(cd.power);
    v.toughness = static_cast<int>(cd.toughness);
    v.has_x_cost = cd.has_x_cost;
    return v;
}

CharView permanent_view(Entity e, const Permanent &perm) {
    CharView v;
    v.entity = e;
    v.types = &perm.types;                 // live type line (type-changing effects write here)
    v.colors = effective_colors(e);        // CR 105: color from the card (no live color layer)
    v.is_token = perm.is_token;
    v.controller = perm.controller;
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
    }
    if (global_coordinator.entity_has_component<CardData>(e)) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        v.cmc = static_cast<int>(cd.mana_cost.size());  // CR 112.7
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

// ── Defined$ player resolution (declared in game_queries.h) ─────────────────
Zone::Ownership source_controller(Entity source) {
    if (global_coordinator.entity_has_component<Permanent>(source))
        return global_coordinator.GetComponent<Permanent>(source).controller;
    if (global_coordinator.entity_has_component<Zone>(source))
        return global_coordinator.GetComponent<Zone>(source).owner;
    return Zone::UNKNOWN;
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
