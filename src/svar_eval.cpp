#include "svar_eval.h"

#include <cstring>

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/types.h"
#include "components/zone.h"
#include "coverage.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "queries/activation.h"
#include "queries/affected.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/counters.h"
#include "queries/filters.h"
#include "queries/lki.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "queries/spells.h"
#include "queries/types.h"
#include "queries/zones.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace {

// What one evaluation reads an expression against: `controller`'s view, the object it belongs to
// and its ability's chosen target.
struct SvarArgs {
    Zone::Ownership controller;
    Entity source;
    Entity target;
};

struct SvarExpr;

using ValueFn = int (*)(const SvarExpr &x, const SvarArgs &a);
using ConditionFn = bool (*)(const SvarArgs &a);
using ObjectsFn = void (*)(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
using PropertyFn = int (*)(Entity e);
// Reads what follows a row's name into `x`. False: the expression is not this row's form.
using ReadFn = bool (*)(const std::string &rest, SvarExpr &x);
using StepFn = int (*)(int value, int n);

// How a set's objects become one number.
enum class Aggregate : uint8_t {
    COUNT,                // how many there are
    FIRST,                // the property of the first; none reads 0
    SUM,                  // the sum of the property
    DISTINCT_CARD_TYPES,  // how many card types they have between them (CR 205.2a: kind TYPE only)
};

// One spelled expression: a value of its own, or a set, a property and an aggregate.
struct SvarRow {
    const char *name;
    bool prefix;            // `name` is how it starts; `read` takes the rest
    ReadFn read;            // null: there is nothing after the name
    ValueFn value;          // null: it reads as a set, a property and an aggregate
    ConditionFn condition;  // what a <high>.<low> row chooses between them by
    ObjectsFn objects;
    PropertyFn property;
    Aggregate aggregate;
    bool names_filter;      // what `read` takes is a filter over the set (svar_filter_spellings)
};

// A set a script names before '$', and a property it names after: <Set>$<Property> is a row of
// neither table, so a new set or property is one row and combines with the others.
struct NamedSet {
    const char *name;
    ObjectsFn objects;
};
struct NamedProperty {
    const char *name;
    PropertyFn property;
};

// A trailing /<name>[<N>] step on the rest of the expression.
struct SuffixRow {
    const char *name;
    bool takes_number;  // `name` is followed by N
    StepFn apply;
};

// Whose permanents a Count$Valid filter's YouCtrl refers to.
enum class ControllerRef : uint8_t { YOU, TARGETED_PLAYER, REMEMBERED_PLAYER };

// A base expression, compiled: its row, and what the row read.
struct SvarExpr {
    const SvarRow *row = nullptr;
    ValueFn value = nullptr;
    ObjectsFn objects = nullptr;
    PropertyFn property = nullptr;
    Aggregate aggregate = Aggregate::COUNT;
    FilterSpec filter;            // what the set's objects must match
    std::string text;             // a counter type, a card type, a cast-history spec
    Colors color = NO_COLOR;      // Count$Devotion's
    Zone::ZoneValue zone = Zone::HAND;
    ControllerRef controller = ControllerRef::YOU;
    int high = 0, low = 0;        // a <high>.<low> pair
    int literal = 0;
    bool you_own = false, opp_own = false;  // a graveyard count scoped to one player's
    bool both_players = false;    // PlayerCount's players: both seats, else the opponent
    bool lost_life = false;       // PlayerCount's property; else `text` compares spells cast
};

struct SvarStep {
    const SuffixRow *row;
    int n;
};

// An expression, compiled once: its base, then its steps, innermost first.
struct CompiledSvar {
    bool recognized = false;
    std::string base_text;
    SvarExpr base;
    std::vector<SvarStep> steps;
};

}  // namespace

static bool parse_int(const std::string &s, int &out);
static bool starts_with(const std::string &s, const char *prefix);
static bool ends_with(const std::string &s, const std::string &suffix);
static const CompiledSvar &compiled_svar(const std::string &expr);
static CompiledSvar compile_svar(const std::string &expr);
static bool read_suffix(const std::string &op, SvarStep &step);
static bool compile_base(const std::string &expr, SvarExpr &x);
static bool compile_row(const std::string &expr, const SvarRow &row, SvarExpr &x);
static bool compile_set_property(const std::string &expr, SvarExpr &x);
static int evaluate(const CompiledSvar &c, const SvarArgs &a);
static int unrecognized(const std::string &expr);
static const Player *player_of(Zone::Ownership seat);

static bool read_no_dollar(const std::string &rest, SvarExpr &x);
static bool read_devotion_color(const std::string &rest, SvarExpr &x);
static bool read_high_low(const std::string &rest, SvarExpr &x);
static bool read_valid_battlefield(const std::string &rest, SvarExpr &x);
static bool read_card_types_clause(const std::string &rest, SvarExpr &x);
static bool read_graveyard_card_types(const std::string &rest, SvarExpr &x);
static bool read_zone_filter(const std::string &rest, SvarExpr &x, Zone::ZoneValue zone);
static bool read_valid_hand(const std::string &rest, SvarExpr &x);
static bool read_valid_graveyard(const std::string &rest, SvarExpr &x);
static bool read_valid_library(const std::string &rest, SvarExpr &x);
static bool read_valid_stack(const std::string &rest, SvarExpr &x);
static bool read_player_count(const std::string &rest, SvarExpr &x);

static int ev_literal(const SvarExpr &x, const SvarArgs &a);
static int ev_aggregate(const SvarExpr &x, const SvarArgs &a);
static int ev_high_low(const SvarExpr &x, const SvarArgs &a);
static int ev_x_paid(const SvarExpr &x, const SvarArgs &a);
static int ev_converge(const SvarExpr &x, const SvarArgs &a);
static int ev_chosen_number(const SvarExpr &x, const SvarArgs &a);
static int ev_resolved_this_turn(const SvarExpr &x, const SvarArgs &a);
static int ev_card_counters(const SvarExpr &x, const SvarArgs &a);
static int ev_devotion(const SvarExpr &x, const SvarArgs &a);
static int ev_spells_cast_this_turn(const SvarExpr &x, const SvarArgs &a);
static int ev_players_with_property(const SvarExpr &x, const SvarArgs &a);

static bool cond_revolt(const SvarArgs &a);
static bool cond_gift_promised(const SvarArgs &a);
static bool cond_threshold(const SvarArgs &a);
static bool cond_urza_lands(const SvarArgs &a);

static void set_you(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_your_library(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_your_yard_of_type(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_remembered(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_target(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_targeted_player(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_exiled_with_card(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_cards_exiled_with(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_battlefield(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_stack_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_zone_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);
static void set_graveyard_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out);

static int prop_energy(Entity player);
static int prop_life_total(Entity player);
static int prop_cards_in_library(Entity player);
static int prop_life_gained(Entity player);
static int prop_spells_cast_this_game(Entity player);
static int prop_power(Entity e);
static int prop_mana_value(Entity e);
static int prop_printed_mana_value(Entity e);

static int step_half_up(int value, int n);
static int step_plus(int value, int n);
static int step_times(int value, int n);
static int step_limit_max(int value, int n);
static int step_divide_down(int value, int n);

static bool urza_lands_assembled(Zone::Ownership controller);

static constexpr SvarRow value_row(const char *name, ValueFn value) {
    return {name, false, nullptr, value, nullptr, nullptr, nullptr, Aggregate::COUNT, false};
}
static constexpr SvarRow read_row(const char *name, ReadFn read, ValueFn value) {
    return {name, true, read, value, nullptr, nullptr, nullptr, Aggregate::COUNT, false};
}
// A <name><high>.<low> row: `high` when `condition` holds, else `low`.
static constexpr SvarRow high_low_row(const char *name, ConditionFn condition) {
    return {name, true, read_high_low, ev_high_low, condition, nullptr, nullptr, Aggregate::COUNT, false};
}
static constexpr SvarRow set_row(const char *name, ObjectsFn objects, PropertyFn property, Aggregate aggregate) {
    return {name, false, nullptr, nullptr, nullptr, objects, property, aggregate, false};
}
static constexpr SvarRow set_read_row(const char *name, ReadFn read, ObjectsFn objects, Aggregate aggregate) {
    return {name, true, read, nullptr, nullptr, objects, nullptr, aggregate, false};
}
// A set named by a filter over a zone.
static constexpr SvarRow filter_row(const char *name, ReadFn read, ObjectsFn objects, Aggregate aggregate) {
    return {name, true, read, nullptr, nullptr, objects, nullptr, aggregate, true};
}

// Every Count$-family expression, in the order an expression is tried against them. The evaluator
// and --parse-report read the same rows (svar_expr_recognized), so a new expression is one row here.
static const SvarRow kSvarRows[] = {
    // Count$xPaid — the X paid for the X-cost spell or ability in flight or resolving (Green Sun's
    // Zenith: ChangeType$ Creature.Green+cmcLEX; Kozilek's Command's token count, scry count and
    // graveyard-exile cap), read through current_x_paid().
    value_row("Count$xPaid", ev_x_paid),
    // Count$Converge (CR 702.90) — the number of distinct colors of mana spent to cast the spell
    // currently resolving (Prismatic Ending: the cmcLEY exile threshold).
    value_row("Count$Converge", ev_converge),
    // Count$ChosenNumber — the integer chosen by the DB$ ChooseNumber effect earlier in this
    // resolution (Wrath of the Skies: the amount of energy to pay), read by a chained sub-ability
    // (the DestroyAll's cmc bound Y and its PayEnergy<Y> unless-cost).
    value_row("Count$ChosenNumber", ev_chosen_number),
    // Count$ResolvedThisTurn — the number of times the source's triggered abilities resolved this
    // turn (Scythecat Cub).
    value_row("Count$ResolvedThisTurn", ev_resolved_this_turn),
    // Count$RememberedSize / RememberedSize — the total number of currently-remembered objects,
    // regardless of type. Triumph of Saint Katherine's recursion gates its shuffle-back on
    // "RememberedSize GE7" — the self-exiled card plus the six milled cards.
    set_row("Count$RememberedSize", set_remembered, nullptr, Aggregate::COUNT),
    set_row("RememberedSize", set_remembered, nullptr, Aggregate::COUNT),
    // Count$YourCountersEnergy — the controller's current energy ({E}) total (CR 122.1c), stored
    // as an "ENERGY" counter on the Player (Wrath of the Skies: the cap on the energy you may pay).
    set_row("Count$YourCountersEnergy", set_you, prop_energy, Aggregate::FIRST),
    // Count$YourLifeTotal — the controller's life total, never below 0 (Doomsday's half your life).
    set_row("Count$YourLifeTotal", set_you, prop_life_total, Aggregate::FIRST),
    // Count$LifeYouGainedThisTurn — Ocelot Pride: "if you gained life this turn".
    set_row("Count$LifeYouGainedThisTurn", set_you, prop_life_gained, Aggregate::FIRST),
    // Count$YouCastThisGame — spells the controller has cast this game (Once Upon a Time: "if
    // this spell is the first spell you've cast this game").
    set_row("Count$YouCastThisGame", set_you, prop_spells_cast_this_game, Aggregate::FIRST),
    // Count$InYourLibrary — the number of cards in the controller's library.
    set_row("Count$InYourLibrary", set_your_library, nullptr, Aggregate::COUNT),
    read_row("Count$CardCounters.", read_no_dollar, ev_card_counters),
    read_row("Count$Devotion.", read_devotion_color, ev_devotion),
    read_row("Count$ThisTurnCast_", read_no_dollar, ev_spells_cast_this_turn),
    // Count$Revolt.high.low — high if a permanent the controller controlled left the battlefield
    // this turn (Fatal Push).
    high_low_row("Count$Revolt.", cond_revolt),
    // Count$PromisedGift.high.low — Gift (CR 702.176): high if the spell being cast/resolved
    // promised its gift to an opponent. Into the Flood Maw drives its two ChangeZone abilities'
    // TargetMin$/TargetMax$ off this (X = .0.1, Y = .1.0): not promised → the creature-bounce
    // targets 1 and the nonland-bounce targets 0; promised → the reverse. Read from the cast-time
    // pending flag (the target counts are evaluated as targets are chosen, before the Spell
    // component exists).
    high_low_row("Count$PromisedGift.", cond_gift_promised),
    // Count$Threshold.high.low — Threshold: high if the controller has seven or more cards in their
    // graveyard (Cabal Ritual: Count$Threshold.5.3 → 5 black mana with threshold, else 3).
    high_low_row("Count$Threshold.", cond_threshold),
    // Count$UrzaLands.high.low — high if the controller controls an Urza's Mine, an Urza's
    // Power-Plant and an Urza's Tower (the Tron lands; Mine/Power Plant: .2.1, Tower: .3.1).
    high_low_row("Count$UrzaLands.", cond_urza_lands),
    filter_row("Count$ValidStack ", read_valid_stack, set_stack_cards, Aggregate::COUNT),
    filter_row("Count$Valid ", read_valid_battlefield, set_battlefield, Aggregate::COUNT),
    // Count$ValidExile ... $CardTypes — distinct card types among the cards exiled with `source`
    // (Keen-Eyed Curator's exiled-with pile).
    filter_row("Count$ValidExile", read_card_types_clause, set_cards_exiled_with, Aggregate::DISTINCT_CARD_TYPES),
    // Count$TypeInYourYard.<TypeName> — count cards of that type in controller's graveyard
    set_read_row("Count$TypeInYourYard.", read_no_dollar, set_your_yard_of_type, Aggregate::COUNT),
    set_row("Count$CardTypesInAllGraveyards", set_graveyard_cards, nullptr, Aggregate::DISTINCT_CARD_TYPES),
    set_read_row("Count$ValidGraveyard Card", read_graveyard_card_types, set_graveyard_cards,
                 Aggregate::DISTINCT_CARD_TYPES),
    // Count$ValidHand / ValidGraveyard / ValidLibrary <filter> — the number of cards in hands /
    // graveyards / libraries matching <filter> (Ensnaring Bridge: "Count$ValidHand Card.YouOwn" =
    // the cards in your hand; Knight of the Reliquary: "Count$ValidGraveyard Land.YouOwn"; Jace,
    // Wielder of Mysteries: "Count$ValidLibrary Card.YouOwn"). Matched by the shared filter
    // matcher against each card, with the whole grammar (comma-OR alternatives,
    // type/color/token qualifiers, OppOwn) honoured.
    filter_row("Count$ValidHand", read_valid_hand, set_zone_cards, Aggregate::COUNT),
    filter_row("Count$ValidGraveyard", read_valid_graveyard, set_zone_cards, Aggregate::COUNT),
    filter_row("Count$ValidLibrary", read_valid_library, set_zone_cards, Aggregate::COUNT),
    read_row("PlayerCount", read_player_count, ev_players_with_property),
};

// The sets and properties of <Set>$<Property>, and <Set>$Valid <filter>.
static const NamedSet kNamedSets[] = {
    // Targeted$ — the ability's chosen target.
    {"Targeted", set_target},
    // TargetedPlayer$ — the player the ability targets, no one if its target is an object (Peer into
    // the Abyss: half the cards in that player's library, half their life).
    {"TargetedPlayer", set_targeted_player},
    // ExiledWith$ — the card the source Saga exiled face down (The Creation of Avacyn chapter II:
    // "you lose life equal to its mana value"). Resolved from the source's Permanent::exiled_with;
    // an absent/gone card contributes 0.
    {"ExiledWith", set_exiled_with_card},
    // Remembered$ — the remembered objects. The RememberedLKI$ variant reads the same remembered
    // entity, but is populated by a RememberLKI$ ChangeZone that snapshots the card as last-known
    // info once it has left its origin zone (Reanimate: you lose life equal to the reanimated
    // creature's mana value — CR 608.2h last-known-info, since the card left the graveyard as it
    // entered play). Both forms resolve identically here because the LKI snapshot is pushed to the
    // remembered set all the same.
    {"Remembered", set_remembered},
    {"RememberedLKI", set_remembered},
};

static const NamedProperty kNamedProperties[] = {
    // CR 608.2h: effective power, read live while the creature is in play (counters/buffs
    // included), else its last-known value once it has left (e.g. Swords to Plowshares
    // reads the power of the creature it just exiled). Single unified accessor.
    {"CardPower", prop_power},
    // The object's mana value (CR 202.3 / 107.14). Used by Karn, the Great Creator's +1
    // Animate (Power$/Toughness$ X, X = Targeted$CardManaCost): the animated permanent
    // becomes a creature whose P/T equal its own mana value, snapshotted at resolution.
    {"CardManaCost", prop_mana_value},
    {"CardsInLibrary", prop_cards_in_library},
    {"LifeTotal", prop_life_total},
};

// /Plus.N adds N (Birthing Ritual: 1 plus the sacrificed creature's mana value), /Times.N
// multiplies (Price of Progress: twice the nonbasic lands), /HalfUp halves rounding up (Doomsday:
// half your life; Tamiyo: half your library), /LimitMax.N caps at N (Flow State: "an instant card"
// = at most 1), /DivideEvenlyDown.N divides rounding down (Dark Matter Manipulator: each whole seven
// cards).
static const SuffixRow kSuffixRows[] = {
    {"HalfUp", false, step_half_up},
    {"Plus.", true, step_plus},
    {"Times.", true, step_times},
    {"LimitMax.", true, step_limit_max},
    {"DivideEvenlyDown.", true, step_divide_down},
};

// A whole-string integer, optionally negative.
static bool parse_int(const std::string &s, int &out) {
    size_t i = (!s.empty() && s[0] == '-') ? 1 : 0;
    if (i >= s.size()) return false;
    for (size_t j = i; j < s.size(); ++j)
        if (!std::isdigit(static_cast<unsigned char>(s[j]))) return false;
    out = std::stoi(s);
    return true;
}

static bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

SvarOp svar_op_of(const std::string &op2) {
    if (op2 == "EQ") return SvarOp::EQ;
    if (op2 == "NE") return SvarOp::NE;
    if (op2 == "GE") return SvarOp::GE;
    if (op2 == "LE") return SvarOp::LE;
    if (op2 == "GT") return SvarOp::GT;
    if (op2 == "LT") return SvarOp::LT;
    return SvarOp::UNKNOWN;
}

bool apply_svar_op(int lhs, const std::string &op2, int rhs) {
    const SvarOp op = svar_op_of(op2);
    if (op == SvarOp::UNKNOWN) non_fatal_error("unrecognized SVar comparison operator '" + op2 + "'");
    return apply_svar_op(lhs, op, rhs);
}

bool apply_svar_op(int lhs, SvarOp op, int rhs) {
    switch (op) {
    case SvarOp::EQ: return lhs == rhs;
    case SvarOp::NE: return lhs != rhs;
    case SvarOp::GE: return lhs >= rhs;
    case SvarOp::LE: return lhs <= rhs;
    case SvarOp::GT: return lhs >  rhs;
    case SvarOp::LT: return lhs <  rhs;
    case SvarOp::UNKNOWN: return false;
    }
    return false;
}

bool compare_svar(int value, const std::string &compare) {
    int rhs = 0;
    if (compare.size() < 3 || !parse_int(compare.substr(2), rhs)) {
        non_fatal_error("unrecognized SVar comparison '" + compare + "'");
        return false;
    }
    return apply_svar_op(value, compare.substr(0, 2), rhs);
}

// Per-permanent stored-SVar trigger gate. Read the latched scratch int off the SOURCE permanent
// and compare it; absent reads as 0. An empty name = no gate (passes).
bool stored_svar_gate_passes(Entity source, const std::string &name, const std::string &compare) {
    if (name.empty()) return true;
    int val = 0;
    if (source != 0 && global_coordinator.entity_has_component<Permanent>(source)) {
        const auto &perm = global_coordinator.GetComponent<Permanent>(source);
        auto it = perm.stored_svars.find(name);
        if (it != perm.stored_svars.end()) val = it->second;
    }
    return compare_svar(val, compare);
}

int evaluate_svar(const std::string &expr, const ResolveCtx &ctx) {
    return evaluate_svar(expr, ctx.controller, ctx.source.lki_entity(), ctx.ability ? ctx.ability->target.get() : 0);
}

int evaluate_svar(const std::string &expr, Zone::Ownership controller, Entity source, Entity target) {
    return evaluate(compiled_svar(expr), SvarArgs{controller, source, target});
}

bool svar_expr_recognized(const std::string &expr) { return compiled_svar(expr).recognized; }

std::vector<std::string> svar_filter_spellings() {
    std::vector<std::string> out;
    for (const SvarRow &row : kSvarRows) {
        if (!row.names_filter) continue;
        out.emplace_back(row.name);
        if (out.back().back() != ' ') out.back() += ' ';
    }
    for (const NamedSet &set : kNamedSets) out.push_back(std::string(set.name) + "$Valid ");
    return out;
}

size_t evaluate_amount(const std::string &expr, const ResolveCtx &ctx) {
    return evaluate_amount(expr, ctx.controller, ctx.source.lki_entity(), ctx.ability ? ctx.ability->target.get() : 0);
}

size_t evaluate_amount(const std::string &expr, Zone::Ownership controller, Entity source,
                       Entity target) {
    int value = evaluate_svar(expr, controller, source, target);
    return value > 0 ? static_cast<size_t>(value) : 0;
}

// The compiled form of `expr`, made the first time it is asked for. Holds nothing bound to a game,
// so like a const AbilityDef * it outlives snapshot restores and per-game resets.
static const CompiledSvar &compiled_svar(const std::string &expr) {
    static std::unordered_map<std::string, CompiledSvar> store;
    auto it = store.find(expr);
    if (it != store.end()) return it->second;
    return store.emplace(expr, compile_svar(expr)).first->second;
}

// Peels the trailing steps, then reads what is left. A '/' that is no step leaves no reading.
static CompiledSvar compile_svar(const std::string &expr) {
    CompiledSvar c;
    std::string base = expr;
    size_t slash;
    SvarStep step{nullptr, 0};
    while ((slash = base.rfind('/')) != std::string::npos && read_suffix(base.substr(slash + 1), step)) {
        c.steps.insert(c.steps.begin(), step);
        base.erase(slash);
    }
    c.base_text = base;
    c.recognized = base.find('/') == std::string::npos && compile_base(base, c.base);
    return c;
}

static bool read_suffix(const std::string &op, SvarStep &step) {
    for (const SuffixRow &row : kSuffixRows) {
        if (!row.takes_number) {
            if (op != row.name) continue;
        } else if (!starts_with(op, row.name) || !parse_int(op.substr(std::strlen(row.name)), step.n)) {
            continue;
        }
        step.row = &row;
        return true;
    }
    return false;
}

static bool compile_base(const std::string &expr, SvarExpr &x) {
    // A plain integer literal (e.g. Humility's SetPower$ 1 / SetToughness$ 1) evaluates to itself.
    if (parse_int(expr, x.literal)) {
        x.value = ev_literal;
        return true;
    }
    for (const SvarRow &row : kSvarRows)
        if (compile_row(expr, row, x)) return true;
    return compile_set_property(expr, x);
}

static bool compile_row(const std::string &expr, const SvarRow &row, SvarExpr &x) {
    if (row.prefix ? !starts_with(expr, row.name) : expr != row.name) return false;
    SvarExpr read;
    read.row = &row;
    read.value = row.objects ? ev_aggregate : row.value;
    read.objects = row.objects;
    read.property = row.property;
    read.aggregate = row.aggregate;
    if (row.read && !row.read(expr.substr(std::strlen(row.name)), read)) return false;
    x = std::move(read);
    return true;
}

// <Set>$<Property>, or <Set>$Valid <comma-OR-filter>: the number of the set's objects matching ANY
// of the filters (Canoptek Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each artifact or
// land card exiled this way"). These are now in their destination zone (e.g. exile), so match by
// printed characteristics via the shared card_matches_filter; control qualifiers resolve against
// the controller.
static bool compile_set_property(const std::string &expr, SvarExpr &x) {
    const size_t dollar = expr.find('$');
    if (dollar == std::string::npos) return false;
    const std::string name = expr.substr(0, dollar), rest = expr.substr(dollar + 1);
    for (const NamedSet &set : kNamedSets) {
        if (name != set.name) continue;
        x.value = ev_aggregate;
        x.objects = set.objects;
        if (starts_with(rest, "Valid ")) {
            x.aggregate = Aggregate::COUNT;
            x.filter = rest.substr(std::strlen("Valid "));
            return rest.find('$') == std::string::npos;
        }
        x.aggregate = Aggregate::FIRST;
        for (const NamedProperty &property : kNamedProperties) {
            if (rest != property.name) continue;
            x.property = property.property;
            return true;
        }
    }
    return false;
}

static int evaluate(const CompiledSvar &c, const SvarArgs &a) {
    int value = c.recognized ? c.base.value(c.base, a) : unrecognized(c.base_text);
    for (const SvarStep &step : c.steps) value = step.row->apply(value, step.n);
    return value;
}

static int unrecognized(const std::string &expr) {
    non_fatal_error("unrecognized SVar expression '" + expr + "'");
    return 0;
}

static const Player *player_of(Zone::Ownership seat) {
    Entity pe = get_player_entity(seat);
    return global_coordinator.entity_has_component<Player>(pe)
               ? &global_coordinator.GetComponent<Player>(pe)
               : nullptr;
}

static bool read_no_dollar(const std::string &rest, SvarExpr &x) {
    x.text = rest;
    return rest.find('$') == std::string::npos;
}

static bool read_devotion_color(const std::string &rest, SvarExpr &x) {
    x.color = rest == "Blue"    ? BLUE
              : rest == "Black" ? BLACK
              : rest == "Red"   ? RED
              : rest == "Green" ? GREEN
              : rest == "White" ? WHITE
                                : NO_COLOR;
    return x.color != NO_COLOR;
}

// "<high>.<low>"; a malformed pair has no reading.
static bool read_high_low(const std::string &rest, SvarExpr &x) {
    const size_t dot = rest.find('.');
    return dot != std::string::npos && parse_int(rest.substr(0, dot), x.high) && parse_int(rest.substr(dot + 1), x.low);
}

// Count$Valid <filter> — battlefield permanents matching the Forge filter spec, the whole spec
// (head type + '.'/'+'-joined qualifiers like Legendary/YouCtrl/colors) through the shared
// permanent_matches_filter (Eldrazi Linebreaker: "Count$Valid Eldrazi.YouCtrl"; Eiganjo's Channel
// ReduceCost: "Count$Valid Creature.Legendary+YouCtrl"). Forms:
//   <filter>$CardManaCost — the SUM of the matches' mana values rather than their count (Summon:
//     Bahamut's Mega Flare: Permanent.YouCtrl+Other$CardManaCost; +Other excludes `source`; a
//     token contributes 0).
//   TargetedPlayerCtrl — permanents the PLAYER this ability targets controls (Carpet of Flowers:
//     Islands the target opponent controls); `target` is that player.
//   RememberedPlayerCtrl — permanents the remembered player controls (Price of Progress, once
//     per player by the RepeatEach loop).
static bool read_valid_battlefield(const std::string &rest, SvarExpr &x) {
    if (rest.empty()) return false;
    const std::string mv_suffix = "$CardManaCost";
    if (ends_with(rest, mv_suffix)) {
        x.filter = rest.substr(0, rest.size() - mv_suffix.size());
        x.property = prop_printed_mana_value;
        x.aggregate = Aggregate::SUM;
        return true;
    }
    if (rest.find('$') != std::string::npos) return false;
    static const struct {
        const char *token;
        ControllerRef ref;
    } kControllers[] = {{"TargetedPlayerCtrl", ControllerRef::TARGETED_PLAYER},
                        {"RememberedPlayerCtrl", ControllerRef::REMEMBERED_PLAYER}};
    for (const auto &who : kControllers) {
        const size_t pos = rest.find(who.token);
        if (pos == std::string::npos) continue;
        std::string spec = rest;
        spec.replace(pos, std::strlen(who.token), "YouCtrl");
        x.filter = spec;
        x.controller = who.ref;
        return true;
    }
    x.filter = rest;
    return true;
}

static bool read_card_types_clause(const std::string &rest, SvarExpr &) {
    return rest.find("$CardTypes") != std::string::npos;
}

// Count$ValidGraveyard Card$CardTypes — count distinct card types (Creature, Instant, etc.)
// across both players' graveyards (Barrowgoyf). The Card.YouOwn variant (Nethergoyf) scopes
// the count to the controller's own graveyard only; we detect the YouOwn/YouCtrl restriction
// on the Card filter and, when present, skip cards owned by the other player. (CR 205.2 — the
// distinct card types among the matching cards.)
static bool read_graveyard_card_types(const std::string &rest, SvarExpr &x) {
    const size_t types_pos = rest.find("$CardTypes");
    if (types_pos == std::string::npos) return false;
    // Restriction lives between "Card" and "$CardTypes" (e.g. ".YouOwn" or
    // ".YouOwn+Creature"). Ownership (YouOwn/YouCtrl/OppOwn/OppCtrl) scopes the count to one
    // player's graveyard and is applied here — a graveyard card has no live controller, so the
    // shared evaluator can't read it. Any OTHER subfilter (type/CMC/color/supertype) in the
    // Card clause is honored by routing the remaining qualifiers through card_matches_filter on
    // the card's printed characteristics, instead of being silently ignored.
    std::string restriction = rest.substr(0, types_pos);
    x.you_own = restriction.find("YouOwn") != std::string::npos || restriction.find("YouCtrl") != std::string::npos;
    x.opp_own = restriction.find("OppOwn") != std::string::npos || restriction.find("OppCtrl") != std::string::npos;
    // Split the restriction into qualifier tokens; keep everything that is not an ownership
    // token as a real card subfilter.
    std::string sub;  // '+'-joined remaining qualifiers
    if (!restriction.empty() && restriction[0] == '.') restriction.erase(0, 1);
    size_t p = 0;
    while (p <= restriction.size()) {
        size_t nx = restriction.find_first_of(".+", p);
        if (nx == std::string::npos) nx = restriction.size();
        std::string tok = restriction.substr(p, nx - p);
        p = nx + 1;
        if (tok.empty()) continue;
        if (tok == "YouOwn" || tok == "YouCtrl" || tok == "OppOwn" || tok == "OppCtrl")
            continue;
        sub += (sub.empty() ? "" : "+") + tok;
    }
    if (!sub.empty()) x.filter = "Card+" + sub;
    return true;
}

// A Count$Valid<Zone> filter. A card there is controlled by its owner (CR 108.4a), so
// YouCtrl/OppCtrl read as YouOwn/OppOwn against `controller`; an empty filter counts every card.
// A $-suffixed aggregate (…$GreatestCardManaCost) is not a count of matching cards.
static bool read_zone_filter(const std::string &rest, SvarExpr &x, Zone::ZoneValue zone) {
    const std::string spec = !rest.empty() && rest[0] == ' ' ? rest.substr(1) : rest;
    if (spec.find('$') != std::string::npos) return false;
    x.zone = zone;
    x.filter = spec.empty() ? FilterSpec("Card") : owner_relative_filter(FilterSpec(spec));
    return true;
}

static bool read_valid_hand(const std::string &rest, SvarExpr &x) { return read_zone_filter(rest, x, Zone::HAND); }
static bool read_valid_graveyard(const std::string &rest, SvarExpr &x) {
    return read_zone_filter(rest, x, Zone::GRAVEYARD);
}
static bool read_valid_library(const std::string &rest, SvarExpr &x) {
    return read_zone_filter(rest, x, Zone::LIBRARY);
}

static bool read_valid_stack(const std::string &rest, SvarExpr &x) {
    x.filter = rest;
    return rest.find('$') == std::string::npos;
}

// PlayerCount<Players>$<Property> — the number of players among <Players> (relative to
// `controller`) that have <Property>. In the two-player game Opponents / RegisteredOpponents is
// the one opponent and Players is both seats. Properties:
//   HasPropertyLostLifeThisTurn — lost life this turn (Kaito, Bane of Nightmares: "each opponent
//     who lost life this turn").
//   Condition<OP><N> SpellsCastThisTurn — cast <OP> N spells this turn (Mindbreak Trap: "if an
//     opponent cast three or more spells this turn").
static bool read_player_count(const std::string &rest, SvarExpr &x) {
    const size_t dollar = rest.find('$');
    if (dollar == std::string::npos) return false;
    const std::string players = rest.substr(0, dollar), prop = rest.substr(dollar + 1);
    if (players == "Opponents" || players == "RegisteredOpponents") x.both_players = false;
    else if (players == "Players") x.both_players = true;
    else return false;
    x.lost_life = prop == "HasPropertyLostLifeThisTurn";
    const std::string spells_cond = "Condition", spells_metric = " SpellsCastThisTurn";
    const bool spells = starts_with(prop, "Condition") && ends_with(prop, spells_metric);
    if (spells)
        x.text = prop.substr(spells_cond.size(), prop.size() - spells_cond.size() - spells_metric.size());
    return x.lost_life || spells;
}

static int ev_literal(const SvarExpr &x, const SvarArgs &) { return x.literal; }

// The row's set, narrowed by a <Set>$Valid filter, as one number.
static int ev_aggregate(const SvarExpr &x, const SvarArgs &a) {
    std::vector<Entity> objects;
    x.objects(x, a, objects);
    if (!x.row && !x.filter.empty()) {
        MatchCtx mctx;
        mctx.controller = a.controller;  // "you" reference for YouCtrl/OppCtrl in any filter
        objects.erase(std::remove_if(objects.begin(), objects.end(),
                                     [&](Entity e) {
                                         return !global_coordinator.entity_has_component<CardData>(e) ||
                                                !card_matches_filter(e, x.filter, mctx);  // ',' = OR over the filters
                                     }),
                      objects.end());
    }
    switch (x.aggregate) {
    case Aggregate::COUNT: return static_cast<int>(objects.size());
    case Aggregate::FIRST: return objects.empty() ? 0 : x.property(objects[0]);
    case Aggregate::SUM: {
        int total = 0;
        for (Entity e : objects) total += x.property(e);
        return total;
    }
    case Aggregate::DISTINCT_CARD_TYPES: {
        std::set<std::string> type_names;
        for (Entity e : objects)
            for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
                if (t.kind == TYPE) type_names.insert(t.name);
        return static_cast<int>(type_names.size());
    }
    }
    return 0;
}

static int ev_high_low(const SvarExpr &x, const SvarArgs &a) { return x.row->condition(a) ? x.high : x.low; }
static int ev_x_paid(const SvarExpr &, const SvarArgs &) { return current_x_paid(); }
static int ev_converge(const SvarExpr &, const SvarArgs &) { return current_converge(); }
static int ev_chosen_number(const SvarExpr &, const SvarArgs &) { return cur_game.resolution.memory.chosen_number; }
static int ev_resolved_this_turn(const SvarExpr &, const SvarArgs &a) { return ability_resolutions_this_turn(a.source); }

static bool cond_revolt(const SvarArgs &a) { return revolt_this_turn(a.controller); }
static bool cond_gift_promised(const SvarArgs &) { return current_gift_promised(); }
static bool cond_threshold(const SvarArgs &a) {
    return zone_objects(zoned_entities(), Zone::GRAVEYARD, a.controller).size() >= 7;
}
static bool cond_urza_lands(const SvarArgs &a) { return urza_lands_assembled(a.controller); }

// Count$CardCounters.<TYPE> — the number of <TYPE> counters on the SOURCE permanent (The One Ring:
// X = Count$CardCounters.BURDEN; Aether Vial's charge-counter mana-value bound; Blast Zone). Once
// the source has left the battlefield — Blast Zone is sacrificed as part of its own activation
// cost before its DestroyAll bound resolves — the count it had as it left (CR 608.2h).
static int ev_card_counters(const SvarExpr &x, const SvarArgs &a) {
    std::string ctype = x.text;
    size_t end = ctype.find_first_of(".+ ");
    if (end != std::string::npos) ctype = ctype.substr(0, end);
    if (a.source == 0) return 0;
    if (global_coordinator.entity_has_component<Permanent>(a.source))
        return std::max(0, get_counters(a.source, ctype));
    if (const LastKnownInfo *lki = departed_lki_for(a.source)) {
        auto ci = lki->counters.find(ctype);
        if (ci != lki->counters.end() && ci->second > 0) return ci->second;
    }
    return 0;
}

// Count$Devotion.<Color> (CR 700.5): each mana symbol of that color in the mana costs of
// permanents the controller controls, a hybrid or Phyrexian symbol of that color included.
static int ev_devotion(const SvarExpr &x, const SvarArgs &a) {
    int count = 0;
    for (Entity e : battlefield_permanents(zoned_entities(), a.controller)) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        count += static_cast<int>(cd.mana_cost.count(x.color));
        for (const auto &pip : cd.hybrid_mana)
            if (std::find(pip.colors.begin(), pip.colors.end(), x.color) != pip.colors.end())
                count++;
        count += static_cast<int>(
            std::count(cd.phyrexian_mana.begin(), cd.phyrexian_mana.end(), x.color));
    }
    return count;
}

// Per CR 205.3i Urza's Mine / Power-Plant / Tower are LAND TYPES, so this reads each permanent's
// effective type line (types added by continuous effects included, e.g. Planar Nexus), not card
// names. One permanent with several of the subtypes (Nexus) satisfies each it carries.
static bool urza_lands_assembled(Zone::Ownership controller) {
    bool mine = false, plant = false, tower = false;
    for (Entity e : battlefield_permanents(zoned_entities(), controller)) {
        const auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (!permanent_has_type(perm, "Urza's")) continue;
        if (permanent_has_type(perm, "Mine")) mine = true;
        if (permanent_has_type(perm, "Power-Plant")) plant = true;
        if (permanent_has_type(perm, "Tower")) tower = true;
    }
    return mine && plant && tower;
}

// Count$ThisTurnCast_<Card>.<Ctrl>[+<Color>][,...] — spells a player (OppCtrl: the controller's
// opponent, else the controller) cast this turn:
//   Instant/Sorcery alternatives — how many instant and sorcery spells (Arclight Phoenix: "if
//     you've cast three or more instant and sorcery spells this turn");
//   color qualifiers — 1 if they cast a spell of any named color (Veil of Summer:
//     Card.OppCtrl+Blue,Card.OppCtrl+Black), else 0;
//   a bare Card.<Ctrl> — how many spells.
static int ev_spells_cast_this_turn(const SvarExpr &x, const SvarArgs &a) {
    const std::string &spec = x.text;
    const Player *pl = player_of(spec.find("OppCtrl") != std::string::npos ? opponent_of(a.controller) : a.controller);
    if (!pl) return 0;
    if (spec.find("Instant") != std::string::npos || spec.find("Sorcery") != std::string::npos)
        return static_cast<int>(pl->instant_sorcery_spells_cast_this_turn);
    static const std::pair<const char *, Colors> kColors[] = {
        {"Blue", BLUE}, {"Black", BLACK}, {"Red", RED}, {"Green", GREEN}, {"White", WHITE}};
    bool names_color = false;
    for (const auto &c : kColors) {
        if (spec.find(c.first) == std::string::npos) continue;
        names_color = true;
        if (pl->spell_colors_cast_this_turn.count(c.second)) return 1;
    }
    if (names_color) return 0;
    return static_cast<int>(pl->spells_cast_this_turn);
}

static int ev_players_with_property(const SvarExpr &x, const SvarArgs &a) {
    std::vector<Zone::Ownership> seats;
    if (x.both_players) seats = {Zone::PLAYER_A, Zone::PLAYER_B};
    else seats = {opponent_of(a.controller)};
    int count = 0;
    for (Zone::Ownership seat : seats) {
        const Player *pl = player_of(seat);
        if (!pl) continue;
        if (x.lost_life) {
            if (pl->life_lost_this_turn > 0) count++;
        } else if (compare_svar(static_cast<int>(pl->spells_cast_this_turn), x.text)) {
            count++;
        }
    }
    return count;
}

static void set_you(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    if (player_of(a.controller)) out.push_back(get_player_entity(a.controller));
}

static void set_your_library(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    out = zone_objects(zoned_entities(), Zone::LIBRARY, a.controller);
}

static void set_your_yard_of_type(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out) {
    for (Entity e : zone_objects(zoned_entities(), Zone::GRAVEYARD, a.controller)) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
            if (t.name == x.text) {
                out.push_back(e);
                break;
            }
    }
}

// Every remembered object, in order, as it last existed: Remembered$CardManaCost reads the first
// (Birthing Ritual: X = 1 plus the sacrificed creature's mana value).
static void set_remembered(const SvarExpr &, const SvarArgs &, std::vector<Entity> &out) {
    for (const ObjectRef &r : cur_game.resolution.memory.remembered) out.push_back(r.lki_entity());
}

static void set_target(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    if (a.target != 0) out.push_back(a.target);
}

static void set_targeted_player(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    coverage_row("TargetedPlayer");
    if (seat_of_player(a.target) != Zone::UNKNOWN) out.push_back(a.target);
}

static void set_exiled_with_card(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    if (Entity e = exiled_with_card(a.source)) out.push_back(e);
}

// The cards still exiled with `source` (Keen-Eyed Curator). exiled_with is append-only, so an entry
// counts only while it is still the object that was exiled, in exile: a card that has since left
// exile is a new object and no longer "exiled with" the source.
static void set_cards_exiled_with(const SvarExpr &, const SvarArgs &a, std::vector<Entity> &out) {
    if (a.source == 0 || !global_coordinator.entity_has_component<Permanent>(a.source)) return;
    for (Entity ex_e : live_entities(global_coordinator.GetComponent<Permanent>(a.source).exiled_with)) {
        if (global_coordinator.GetComponent<Zone>(ex_e).location != Zone::EXILE) continue;
        if (global_coordinator.entity_has_component<CardData>(ex_e)) out.push_back(ex_e);
    }
}

static void set_battlefield(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out) {
    Zone::Ownership seat = a.controller;
    if (x.controller == ControllerRef::TARGETED_PLAYER) {
        seat = seat_of_player(a.target);
        if (seat == Zone::UNKNOWN) return;
    } else if (x.controller == ControllerRef::REMEMBERED_PLAYER) {
        const auto &remembered = cur_game.resolution.memory.remembered;
        seat = remembered.empty() ? Zone::UNKNOWN : seat_of_player(remembered[0].get());
        if (seat == Zone::UNKNOWN) seat = a.controller;
    }
    const MatchCtx mctx{seat, a.source};
    for (Entity e : zoned_entities())
        if (is_battlefield_permanent(e) && permanent_matches_filter(e, x.filter, mctx)) out.push_back(e);
}

// Count$ValidStack <filter> — number of stack objects matching a card filter (Mindbreak Trap:
// TargetMax$ MaxTgts, MaxTgts = Count$ValidStack Card — the cap on "exile any number of target
// spells" is the number of spell cards on the stack). Card-shaped filters are matched through the
// shared comma-OR card filter against each spell's printed characteristics; standalone ability
// entities (no CardData) are not cards and don't count. The evaluating ability's own source is
// excluded — a spell can never target itself, so counting it would only inflate the cap past the
// real candidate pool.
static void set_stack_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out) {
    for (Entity e : zone_objects(zoned_entities(), Zone::STACK, Zone::UNKNOWN)) {
        if (e == a.source) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (card_matches_filter(e, x.filter, MatchCtx{a.controller, a.source})) out.push_back(e);
    }
}

// The cards in `x.zone` (either player's) matching its filter.
static void set_zone_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out) {
    MatchCtx ctx;
    ctx.controller = a.controller;
    for (Entity e : zone_objects(zoned_entities(), x.zone, Zone::UNKNOWN))
        if (card_matches_filter(e, x.filter, ctx)) out.push_back(e);
}

static void set_graveyard_cards(const SvarExpr &x, const SvarArgs &a, std::vector<Entity> &out) {
    MatchCtx sub_ctx;
    sub_ctx.controller = a.controller;
    for (Entity e : zone_objects(zoned_entities(), Zone::GRAVEYARD, Zone::UNKNOWN)) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (x.you_own && z.owner != a.controller) continue;
        if (x.opp_own && z.owner == a.controller) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!x.filter.empty() && !card_matches_filter(e, x.filter, sub_ctx)) continue;
        out.push_back(e);
    }
}

static int prop_energy(Entity player) { return player_energy(global_coordinator.GetComponent<Player>(player)); }

// An object has no life total or library: Targeted$ may name either.
static int prop_life_total(Entity player) {
    if (seat_of_player(player) == Zone::UNKNOWN) return 0;
    return std::max(0, static_cast<int>(global_coordinator.GetComponent<Player>(player).life_total));
}

static int prop_cards_in_library(Entity player) {
    const Zone::Ownership seat = seat_of_player(player);
    if (seat == Zone::UNKNOWN) return 0;
    return static_cast<int>(zone_objects(zoned_entities(), Zone::LIBRARY, seat).size());
}

static int prop_life_gained(Entity player) { return global_coordinator.GetComponent<Player>(player).life_gained_this_turn; }

static int prop_spells_cast_this_game(Entity player) {
    return static_cast<int>(global_coordinator.GetComponent<Player>(player).spells_cast_this_game);
}

static int prop_power(Entity e) { return std::max(0, effective_power(e)); }

static int prop_mana_value(Entity e) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return 0;
    return std::max(0, object_mana_value(e, global_coordinator.GetComponent<CardData>(e)));
}

// A token contributes 0.
static int prop_printed_mana_value(Entity e) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return 0;
    return card_mana_value(global_coordinator.GetComponent<CardData>(e));
}

static int step_half_up(int value, int) { return (value + 1) / 2; }
static int step_plus(int value, int n) { return value + n; }
static int step_times(int value, int n) { return value * n; }
static int step_limit_max(int value, int n) { return std::min(value, n); }
static int step_divide_down(int value, int n) {
    coverage_row("DivideEvenlyDown");
    return n > 0 ? value / n : 0;
}
