#include "svar_eval.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <string>
#include <vector>

#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/types.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "queries/activation.h"
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

static bool parse_int(const std::string &s, int &out);
static bool starts_with(const std::string &s, const char *prefix);
static bool apply_suffix_op(const std::string &expr, Zone::Ownership controller, Entity source,
                            Entity target, int &out);
static int evaluate_base(const std::string &expr, Zone::Ownership controller, Entity source,
                         Entity target);
static int unrecognized(const std::string &expr);
static const Player *player_of(Zone::Ownership seat);
static int count_zone_cards_matching(Zone::ZoneValue zone, std::string spec,
                                     Zone::Ownership controller);
static int count_players_with_property(const std::string &expr, Zone::Ownership controller);
static bool high_low(const std::string &expr, const char *prefix, bool condition, int &out);
static int card_counters(Entity source, const std::string &expr);
static int devotion(const std::string &expr, Zone::Ownership controller);
static bool urza_lands_assembled(Zone::Ownership controller);
static int spells_cast_this_turn(const std::string &expr, Zone::Ownership controller);
static int count_valid(std::string spec, Zone::Ownership controller, Entity source, Entity target);
static int count_valid_stack(const std::string &spec, Zone::Ownership controller, Entity source);
static int exiled_with_card_types(Entity source);
static int graveyard_card_types(const std::string &expr, Zone::Ownership controller);
static int remembered_mana_value();

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

bool apply_svar_op(int lhs, const std::string &op2, int rhs) {
    if (op2 == "EQ") return lhs == rhs;
    if (op2 == "NE") return lhs != rhs;
    if (op2 == "GE") return lhs >= rhs;
    if (op2 == "LE") return lhs <= rhs;
    if (op2 == "GT") return lhs >  rhs;
    if (op2 == "LT") return lhs <  rhs;
    non_fatal_error("unrecognized SVar comparison operator '" + op2 + "'");
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

// Per-permanent stored-SVar trigger gate (Carpet of Flowers). Read the latched scratch int off the
// SOURCE permanent and compare it; absent reads as 0. An empty name = no gate (passes).
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

int evaluate_svar(const std::string &expr, Zone::Ownership controller, Entity source, Entity target) {
    int value = 0;
    if (apply_suffix_op(expr, controller, source, target, value)) return value;
    return evaluate_base(expr, controller, source, target);
}

size_t evaluate_amount(const std::string &expr, Zone::Ownership controller, Entity source,
                       Entity target) {
    int value = evaluate_svar(expr, controller, source, target);
    return value > 0 ? static_cast<size_t>(value) : 0;
}

// A trailing arithmetic step on the rest of the expression: /Plus.N adds N (Birthing Ritual:
// 1 plus the sacrificed creature's mana value), /Times.N multiplies (Price of Progress: twice
// the nonbasic lands), /HalfUp halves rounding up (Doomsday: half your life; Tamiyo: half your
// library), /LimitMax.N caps at N (Flow State: "an instant card" = at most 1). Returns false when
// the expression ends in no such step.
static bool apply_suffix_op(const std::string &expr, Zone::Ownership controller, Entity source,
                            Entity target, int &out) {
    size_t slash = expr.rfind('/');
    if (slash == std::string::npos) return false;
    const std::string base = expr.substr(0, slash);
    const std::string op = expr.substr(slash + 1);
    int n = 0;
    if (op == "HalfUp") {
        int v = evaluate_svar(base, controller, source, target);
        out = (v + 1) / 2;
        return true;
    }
    if (starts_with(op, "Plus.") && parse_int(op.substr(5), n)) {
        out = evaluate_svar(base, controller, source, target) + n;
        return true;
    }
    if (starts_with(op, "Times.") && parse_int(op.substr(6), n)) {
        out = evaluate_svar(base, controller, source, target) * n;
        return true;
    }
    if (starts_with(op, "LimitMax.") && parse_int(op.substr(9), n)) {
        out = std::min(evaluate_svar(base, controller, source, target), n);
        return true;
    }
    return false;
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

static int evaluate_base(const std::string &expr, Zone::Ownership controller, Entity source,
                         Entity target) {
    // A plain integer literal (e.g. Humility's SetPower$ 1 / SetToughness$ 1) evaluates to itself.
    int literal = 0;
    if (parse_int(expr, literal)) return literal;

    // Count$xPaid — the X paid for the X-cost spell or ability in flight or resolving (Green Sun's
    // Zenith: ChangeType$ Creature.Green+cmcLEX; Kozilek's Command's token count, scry count and
    // graveyard-exile cap), read through current_x_paid().
    if (expr == "Count$xPaid") return current_x_paid();
    // Count$Converge (CR 702.90) — the number of distinct colors of mana spent to cast the spell
    // currently resolving (Prismatic Ending: the cmcLEY exile threshold).
    if (expr == "Count$Converge") return current_converge();
    // Count$ChosenNumber — the integer chosen by the DB$ ChooseNumber effect earlier in this
    // resolution (Wrath of the Skies: the amount of energy to pay), read by a chained sub-ability
    // (the DestroyAll's cmc bound Y and its PayEnergy<Y> unless-cost).
    if (expr == "Count$ChosenNumber") return cur_game.resolution.memory.chosen_number;
    // Count$ResolvedThisTurn — the number of times the source's triggered abilities resolved this
    // turn (Scythecat Cub).
    if (expr == "Count$ResolvedThisTurn") return ability_resolutions_this_turn(source);
    // Count$RememberedSize / RememberedSize — the total number of currently-remembered objects,
    // regardless of type. Triumph of Saint Katherine's recursion gates its shuffle-back on
    // "RememberedSize GE7" — the self-exiled card plus the six milled cards.
    if (expr == "Count$RememberedSize" || expr == "RememberedSize")
        return static_cast<int>(cur_game.resolution.memory.remembered.size());

    const Player *you = player_of(controller);
    // Count$YourCountersEnergy — the controller's current energy ({E}) total (CR 122.1c), stored
    // as an "ENERGY" counter on the Player (Wrath of the Skies: the cap on the energy you may pay).
    if (expr == "Count$YourCountersEnergy") return you ? player_energy(*you) : 0;
    // Count$YourLifeTotal — the controller's life total, never below 0 (Doomsday's half your life).
    if (expr == "Count$YourLifeTotal") return you ? std::max(0, static_cast<int>(you->life_total)) : 0;
    // Count$LifeYouGainedThisTurn — Ocelot Pride: "if you gained life this turn".
    if (expr == "Count$LifeYouGainedThisTurn") return you ? you->life_gained_this_turn : 0;
    // Count$YouCastThisGame — spells the controller has cast this game (Once Upon a Time: "if
    // this spell is the first spell you've cast this game").
    if (expr == "Count$YouCastThisGame") return you ? static_cast<int>(you->spells_cast_this_game) : 0;
    // Count$InYourLibrary — the number of cards in the controller's library.
    if (expr == "Count$InYourLibrary")
        return static_cast<int>(zone_objects(zoned_entities(), Zone::LIBRARY, controller).size());

    if (starts_with(expr, "Count$CardCounters.")) return card_counters(source, expr);
    if (starts_with(expr, "Count$Devotion.")) return devotion(expr, controller);
    if (starts_with(expr, "Count$ThisTurnCast_")) return spells_cast_this_turn(expr, controller);

    int high_or_low = 0;
    // Count$Revolt.high.low — high if a permanent the controller controlled left the battlefield
    // this turn (Fatal Push).
    if (high_low(expr, "Count$Revolt.", revolt_this_turn(controller), high_or_low)) return high_or_low;
    // Count$PromisedGift.high.low — Gift (CR 702.176): high if the spell being cast/resolved
    // promised its gift to an opponent. Into the Flood Maw drives its two ChangeZone abilities'
    // TargetMin$/TargetMax$ off this (X = .0.1, Y = .1.0): not promised → the creature-bounce
    // targets 1 and the nonland-bounce targets 0; promised → the reverse. Read from the cast-time
    // pending flag (the target counts are evaluated as targets are chosen, before the Spell
    // component exists).
    if (high_low(expr, "Count$PromisedGift.", current_gift_promised(), high_or_low)) return high_or_low;
    // Count$Threshold.high.low — Threshold: high if the controller has seven or more cards in their
    // graveyard (Cabal Ritual: Count$Threshold.5.3 → 5 black mana with threshold, else 3).
    if (high_low(expr, "Count$Threshold.",
                 zone_objects(zoned_entities(), Zone::GRAVEYARD, controller).size() >= 7, high_or_low))
        return high_or_low;
    // Count$UrzaLands.high.low — high if the controller controls an Urza's Mine, an Urza's
    // Power-Plant and an Urza's Tower (the Tron lands; Mine/Power Plant: .2.1, Tower: .3.1).
    if (high_low(expr, "Count$UrzaLands.", urza_lands_assembled(controller), high_or_low))
        return high_or_low;

    if (starts_with(expr, "Count$ValidStack "))
        return count_valid_stack(expr.substr(std::string("Count$ValidStack ").size()), controller, source);
    if (starts_with(expr, "Count$Valid "))
        return count_valid(expr.substr(std::string("Count$Valid ").size()), controller, source, target);

    // Count$ValidExile ... $CardTypes — distinct card types among the cards exiled with `source`
    // (Keen-Eyed Curator's exiled-with pile).
    if (starts_with(expr, "Count$ValidExile") && expr.find("$CardTypes") != std::string::npos)
        return exiled_with_card_types(source);

    // Count$TypeInYourYard.<TypeName> — count cards of that type in controller's graveyard
    if (starts_with(expr, "Count$TypeInYourYard.")) {
        std::string type_name = expr.substr(std::string("Count$TypeInYourYard.").size());
        int count = 0;
        for (Entity e : zone_objects(zoned_entities(), Zone::GRAVEYARD, controller)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
                if (t.name == type_name) { count++; break; }
        }
        return count;
    }

    if (expr == "Count$CardTypesInAllGraveyards" ||
        (starts_with(expr, "Count$ValidGraveyard Card") && expr.find("$CardTypes") != std::string::npos))
        return graveyard_card_types(expr, controller);

    // Count$ValidHand / ValidGraveyard / ValidLibrary <filter> — the number of cards in hands /
    // graveyards / libraries matching <filter> (Ensnaring Bridge: "Count$ValidHand Card.YouOwn" =
    // the cards in your hand; Knight of the Reliquary: "Count$ValidGraveyard Land.YouOwn"; Jace,
    // Wielder of Mysteries: "Count$ValidLibrary Card.YouOwn"). Matched by the shared filter
    // matcher against each card, with the whole grammar (comma-OR alternatives,
    // type/color/token qualifiers, OppOwn) honoured.
    if (starts_with(expr, "Count$ValidHand"))
        return count_zone_cards_matching(Zone::HAND, expr.substr(std::string("Count$ValidHand").size()),
                                         controller);
    if (starts_with(expr, "Count$ValidGraveyard"))
        return count_zone_cards_matching(
            Zone::GRAVEYARD, expr.substr(std::string("Count$ValidGraveyard").size()), controller);
    if (starts_with(expr, "Count$ValidLibrary"))
        return count_zone_cards_matching(
            Zone::LIBRARY, expr.substr(std::string("Count$ValidLibrary").size()), controller);

    if (starts_with(expr, "PlayerCount") && expr.find('$') != std::string::npos)
        return count_players_with_property(expr, controller);

    if (expr == "Targeted$CardPower") {
        // CR 608.2h: effective power, read live while the creature is in play (counters/buffs
        // included), else its last-known value once it has left (e.g. Swords to Plowshares
        // reads the power of the creature it just exiled). Single unified accessor.
        return std::max(0, effective_power(target));
    }
    if (expr == "Targeted$CardManaCost") {
        // The target's mana value (CR 202.3 / 107.14). Used by Karn, the Great Creator's +1
        // Animate (Power$/Toughness$ X, X = Targeted$CardManaCost): the animated permanent
        // becomes a creature whose P/T equal its own mana value, snapshotted at resolution.
        if (target == 0 || !global_coordinator.entity_has_component<CardData>(target)) return 0;
        return std::max(0, object_mana_value(target, global_coordinator.GetComponent<CardData>(target)));
    }
    // ExiledWith$CardManaCost — the mana value of the card the source Saga exiled face down (The
    // Creation of Avacyn chapter II: "you lose life equal to its mana value"). Resolved from the
    // source's Permanent::exiled_with; an absent/gone card contributes 0.
    if (expr == "ExiledWith$CardManaCost") {
        Entity ew = exiled_with_card(source);
        if (ew == 0 || !global_coordinator.entity_has_component<CardData>(ew)) return 0;
        return std::max(0, object_mana_value(ew, global_coordinator.GetComponent<CardData>(ew)));
    }
    // Remembered$Valid <comma-OR-filter> — number of remembered cards (e.g. cards just moved
    // by a RememberChanged$ ChangeZoneAll) matching ANY of the comma-separated filters (Canoptek
    // Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each artifact or land card exiled
    // this way"). These are now in their destination zone (e.g. exile), so match by printed
    // characteristics via the shared card_matches_filter; control qualifiers resolve against ctrl.
    if (starts_with(expr, "Remembered$Valid ")) {
        std::string filters = expr.substr(std::string("Remembered$Valid ").size());
        MatchCtx mctx;
        mctx.controller = controller;  // "you" reference for YouCtrl/OppCtrl in any filter
        int count = 0;
        for (Entity e : lki_entities(cur_game.resolution.memory.remembered)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (card_matches_any(e, filters, mctx)) count++;  // ',' = OR over the filters
        }
        return count;
    }
    // Remembered$CardManaCost / RememberedLKI$CardManaCost — see remembered_mana_value.
    if (expr == "Remembered$CardManaCost" || expr == "RememberedLKI$CardManaCost")
        return remembered_mana_value();

    return unrecognized(expr);
}

// The number of cards in `zone` (either player's) matching a Count$Valid<Zone> filter. A card
// there is controlled by its owner (CR 108.4a), so YouCtrl/OppCtrl read as YouOwn/OppOwn against
// `controller`; an empty filter counts every card.
static int count_zone_cards_matching(Zone::ZoneValue zone, std::string spec,
                                     Zone::Ownership controller) {
    if (!spec.empty() && spec[0] == ' ') spec.erase(0, 1);
    // A $-suffixed aggregate (…$GreatestCardManaCost) is not a count of matching cards.
    if (spec.find('$') != std::string::npos) return unrecognized(spec);
    const std::string filter = spec.empty() ? std::string("Card") : owner_relative_filter(spec);
    MatchCtx ctx;
    ctx.controller = controller;
    int count = 0;
    for (Entity e : zone_objects(zoned_entities(), zone, Zone::UNKNOWN))
        if (card_matches_filter(e, filter, ctx)) count++;
    return count;
}

// PlayerCount<Players>$<Property> — the number of players among <Players> (relative to
// `controller`) that have <Property>. In the two-player game Opponents / RegisteredOpponents is
// the one opponent and Players is both seats. Properties:
//   HasPropertyLostLifeThisTurn — lost life this turn (Kaito, Bane of Nightmares: "each opponent
//     who lost life this turn").
//   Condition<OP><N> SpellsCastThisTurn — cast <OP> N spells this turn (Mindbreak Trap: "if an
//     opponent cast three or more spells this turn").
static int count_players_with_property(const std::string &expr, Zone::Ownership controller) {
    size_t dollar = expr.find('$');
    const std::string players = expr.substr(std::string("PlayerCount").size(),
                                            dollar - std::string("PlayerCount").size());
    const std::string prop = expr.substr(dollar + 1);
    std::vector<Zone::Ownership> seats;
    if (players == "Opponents" || players == "RegisteredOpponents") seats = {opponent_of(controller)};
    else if (players == "Players") seats = {Zone::PLAYER_A, Zone::PLAYER_B};
    else return unrecognized(expr);
    const std::string spells_cond = "Condition";
    const std::string spells_metric = " SpellsCastThisTurn";
    bool lost_life = prop == "HasPropertyLostLifeThisTurn";
    bool spells = starts_with(prop, "Condition") && prop.size() > spells_metric.size() &&
                  prop.compare(prop.size() - spells_metric.size(), spells_metric.size(), spells_metric) == 0;
    if (!lost_life && !spells) return unrecognized(expr);
    const std::string compare =
        spells ? prop.substr(spells_cond.size(), prop.size() - spells_cond.size() - spells_metric.size())
               : std::string();
    int count = 0;
    for (Zone::Ownership seat : seats) {
        const Player *pl = player_of(seat);
        if (!pl) continue;
        if (lost_life && pl->life_lost_this_turn > 0) count++;
        if (spells && compare_svar(static_cast<int>(pl->spells_cast_this_turn), compare)) count++;
    }
    return count;
}

// "<prefix><high>.<low>" — `high` when `condition` holds, else `low`. Returns false when `expr`
// doesn't start with `prefix`; a malformed pair is reported and reads as 0.
static bool high_low(const std::string &expr, const char *prefix, bool condition, int &out) {
    if (!starts_with(expr, prefix)) return false;
    const std::string rest = expr.substr(std::string(prefix).size());
    size_t dot = rest.find('.');
    int high = 0, low = 0;
    if (dot == std::string::npos || !parse_int(rest.substr(0, dot), high) ||
        !parse_int(rest.substr(dot + 1), low)) {
        out = unrecognized(expr);
        return true;
    }
    out = condition ? high : low;
    return true;
}

// Count$CardCounters.<TYPE> — the number of <TYPE> counters on the SOURCE permanent (The One Ring:
// X = Count$CardCounters.BURDEN; Aether Vial's charge-counter mana-value bound; Blast Zone). Once
// the source has left the battlefield — Blast Zone is sacrificed as part of its own activation
// cost before its DestroyAll bound resolves — the count it had as it left (CR 608.2h).
static int card_counters(Entity source, const std::string &expr) {
    std::string ctype = expr.substr(std::string("Count$CardCounters.").size());
    size_t end = ctype.find_first_of(".+ ");
    if (end != std::string::npos) ctype = ctype.substr(0, end);
    if (source == 0) return 0;
    if (global_coordinator.entity_has_component<Permanent>(source))
        return std::max(0, get_counters(source, ctype));
    if (const LastKnownInfo *lki = departed_lki_for(source)) {
        auto ci = lki->counters.find(ctype);
        if (ci != lki->counters.end() && ci->second > 0) return ci->second;
    }
    return 0;
}

// Count$Devotion.<Color> (CR 700.5): each mana symbol of that color in the mana costs of
// permanents the controller controls, a hybrid or Phyrexian symbol of that color included.
static int devotion(const std::string &expr, Zone::Ownership controller) {
    const std::string color = expr.substr(std::string("Count$Devotion.").size());
    Colors devotion_color = color == "Blue"    ? BLUE
                            : color == "Black" ? BLACK
                            : color == "Red"   ? RED
                            : color == "Green" ? GREEN
                            : color == "White" ? WHITE
                                               : NO_COLOR;
    if (devotion_color == NO_COLOR) return unrecognized(expr);
    int count = 0;
    for (Entity e : battlefield_permanents(zoned_entities(), controller)) {
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        count += static_cast<int>(cd.mana_cost.count(devotion_color));
        for (const auto &pip : cd.hybrid_mana)
            if (std::find(pip.colors.begin(), pip.colors.end(), devotion_color) != pip.colors.end())
                count++;
        count += static_cast<int>(
            std::count(cd.phyrexian_mana.begin(), cd.phyrexian_mana.end(), devotion_color));
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
static int spells_cast_this_turn(const std::string &expr, Zone::Ownership controller) {
    const Player *pl = player_of(expr.find("OppCtrl") != std::string::npos ? opponent_of(controller)
                                                                             : controller);
    if (!pl) return 0;
    if (expr.find("Instant") != std::string::npos || expr.find("Sorcery") != std::string::npos)
        return static_cast<int>(pl->instant_sorcery_spells_cast_this_turn);
    static const std::pair<const char *, Colors> kColors[] = {
        {"Blue", BLUE}, {"Black", BLACK}, {"Red", RED}, {"Green", GREEN}, {"White", WHITE}};
    bool names_color = false;
    for (const auto &c : kColors) {
        if (expr.find(c.first) == std::string::npos) continue;
        names_color = true;
        if (pl->spell_colors_cast_this_turn.count(c.second)) return 1;
    }
    if (names_color) return 0;
    return static_cast<int>(pl->spells_cast_this_turn);
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
static int count_valid(std::string spec, Zone::Ownership controller, Entity source, Entity target) {
    if (spec.empty()) return unrecognized("Count$Valid " + spec);
    const std::string mv_suffix = "$CardManaCost";
    if (spec.size() > mv_suffix.size() &&
        spec.compare(spec.size() - mv_suffix.size(), mv_suffix.size(), mv_suffix) == 0) {
        spec.erase(spec.size() - mv_suffix.size());
        MatchCtx mctx;
        mctx.controller = controller;
        mctx.source = source;
        int total = 0;
        for (Entity e : battlefield_permanents(zoned_entities())) {
            if (!permanent_matches_filter(e, spec, mctx)) continue;
            if (global_coordinator.entity_has_component<CardData>(e))
                total += card_mana_value(global_coordinator.GetComponent<CardData>(e));
        }
        return total;
    }
    if (spec.find('$') != std::string::npos) return unrecognized("Count$Valid " + spec);
    for (const char *who : {"TargetedPlayerCtrl", "RememberedPlayerCtrl"}) {
        size_t pos = spec.find(who);
        if (pos == std::string::npos) continue;
        Zone::Ownership seat;
        if (std::string(who) == "TargetedPlayerCtrl") {
            seat = seat_of_player(target);
            if (seat == Zone::UNKNOWN) return 0;
        } else {
            const auto &remembered = cur_game.resolution.memory.remembered;
            seat = remembered.empty() ? Zone::UNKNOWN : seat_of_player(remembered[0].get());
            if (seat == Zone::UNKNOWN) seat = controller;
        }
        spec.replace(pos, std::string(who).size(), "YouCtrl");
        return count_battlefield_matching(spec, seat, source);
    }
    return count_battlefield_matching(spec, controller, source);
}

// Count$ValidStack <filter> — number of stack objects matching a card filter (Mindbreak Trap:
// TargetMax$ MaxTgts, MaxTgts = Count$ValidStack Card — the cap on "exile any number of target
// spells" is the number of spell cards on the stack). Card-shaped filters are matched through the
// shared comma-OR card filter against each spell's printed characteristics; standalone ability
// entities (no CardData) are not cards and don't count. The evaluating ability's own source is
// excluded — a spell can never target itself, so counting it would only inflate the cap past the
// real candidate pool.
static int count_valid_stack(const std::string &spec, Zone::Ownership controller, Entity source) {
    int count = 0;
    for (Entity e : zone_objects(zoned_entities(), Zone::STACK, Zone::UNKNOWN)) {
        if (e == source) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (card_matches_any(e, spec, MatchCtx{controller, source})) count++;
    }
    return count;
}

// Distinct card types (CR 205.2a — kind TYPE only, so Plains / Legendary are excluded) among the
// cards still exiled with `source` (Keen-Eyed Curator). exiled_with is append-only, so an entry
// counts only while it is still the object that was exiled, in exile: a card that has since left
// exile is a new object and no longer "exiled with" the source.
static int exiled_with_card_types(Entity source) {
    if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source)) return 0;
    std::set<std::string> type_names;
    for (Entity ex_e : live_entities(global_coordinator.GetComponent<Permanent>(source).exiled_with)) {
        if (global_coordinator.GetComponent<Zone>(ex_e).location != Zone::EXILE) continue;
        if (!global_coordinator.entity_has_component<CardData>(ex_e)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(ex_e).types)
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return static_cast<int>(type_names.size());
}

// Count$ValidGraveyard Card$CardTypes — count distinct card types (Creature, Instant, etc.)
// across both players' graveyards (Barrowgoyf). The Card.YouOwn variant (Nethergoyf) scopes
// the count to the controller's own graveyard only; we detect the YouOwn/YouCtrl restriction
// on the Card filter and, when present, skip cards owned by the other player. (CR 205.2 — the
// distinct card types among the matching cards.)
static int graveyard_card_types(const std::string &expr, Zone::Ownership controller) {
    std::string restriction;
    if (expr != "Count$CardTypesInAllGraveyards") {
        // Restriction lives between "Card" and "$CardTypes" (e.g. ".YouOwn" or
        // ".YouOwn+Creature"). Ownership (YouOwn/YouCtrl/OppOwn/OppCtrl) scopes the count to one
        // player's graveyard and is applied here — a graveyard card has no live controller, so the
        // shared evaluator can't read it. Any OTHER subfilter (type/CMC/color/supertype) in the
        // Card clause is honored by routing the remaining qualifiers through card_matches_filter on
        // the card's printed characteristics, instead of being silently ignored.
        size_t card_pos = expr.find("Card");
        size_t types_pos = expr.find("$CardTypes");
        restriction = expr.substr(card_pos + 4, types_pos - (card_pos + 4));
    }
    bool you_own = restriction.find("YouOwn") != std::string::npos ||
                   restriction.find("YouCtrl") != std::string::npos;
    bool opp_own = restriction.find("OppOwn") != std::string::npos ||
                   restriction.find("OppCtrl") != std::string::npos;
    // Split the restriction into qualifier tokens; keep everything that is not an ownership
    // token as a real card subfilter, and pull a numeric mana-value bound into the MatchCtx
    // (the evaluator defers cmcLE<n>/cmcGE<n>/… to ctx.cmc_bound).
    std::string sub;  // '+'-joined remaining qualifiers
    MatchCtx sub_ctx;
    sub_ctx.controller = controller;
    {
        std::string r = restriction;
        if (!r.empty() && r[0] == '.') r.erase(0, 1);
        size_t p = 0;
        while (p <= r.size()) {
            size_t nx = r.find_first_of(".+", p);
            if (nx == std::string::npos) nx = r.size();
            std::string tok = r.substr(p, nx - p);
            p = nx + 1;
            if (tok.empty()) continue;
            if (tok == "YouOwn" || tok == "YouCtrl" || tok == "OppOwn" || tok == "OppCtrl")
                continue;
            // cmc<OP><n> (e.g. cmcLE3): a numeric mana-value bound goes in the MatchCtx. A
            // non-numeric form like cmcLEX falls through and is handled inline by the evaluator.
            if (tok.rfind("cmc", 0) == 0 && tok.size() >= 6 &&
                std::isdigit(static_cast<unsigned char>(tok[5]))) {
                sub_ctx.cmc_op = tok.substr(3, 2);
                sub_ctx.cmc_bound = std::stoi(tok.substr(5));
                continue;
            }
            sub += (sub.empty() ? "" : "+") + tok;
        }
    }
    std::string sub_spec = (sub.empty() && sub_ctx.cmc_bound < 0) ? "" : ("Card+" + sub);
    std::set<std::string> type_names;
    for (Entity e : zone_objects(zoned_entities(), Zone::GRAVEYARD, Zone::UNKNOWN)) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (you_own && z.owner != controller) continue;
        if (opp_own && z.owner == controller) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        if (!sub_spec.empty() && !card_matches_filter(e, sub_spec, sub_ctx)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return static_cast<int>(type_names.size());
}

// Remembered$CardManaCost — mana value of the first remembered card (Birthing Ritual: X = 1 plus
// the sacrificed creature's mana value). The RememberedLKI$ variant reads the same remembered
// entity, but is populated by a RememberLKI$ ChangeZone that snapshots the card as last-known
// info once it has left its origin zone (Reanimate: you lose life equal to the reanimated
// creature's mana value — CR 608.2h last-known-info, since the card left the graveyard as it
// entered play). Both forms resolve identically here because the LKI snapshot is pushed to the
// remembered set all the same.
static int remembered_mana_value() {
    const auto &remembered = cur_game.resolution.memory.remembered;
    if (remembered.empty()) return 0;
    Entity r = remembered[0].lki_entity();
    if (!global_coordinator.entity_has_component<CardData>(r)) return 0;
    return std::max(0, object_mana_value(r, global_coordinator.GetComponent<CardData>(r)));
}
