#include "svar_eval.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <string>

#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/types.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "queries/battlefield.h"
#include "queries/counters.h"
#include "queries/filters.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "queries/characteristics.h"
#include "queries/lki.h"
#include "queries/spells.h"
#include "queries/types.h"
#include "queries/zones.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;

static int count_zone_cards_matching(Zone::ZoneValue zone, std::string spec,
                                     Zone::Ownership controller);
static int count_players_with_property(const std::string &expr, Zone::Ownership controller);

// The bare operator table — one home for the EQ/NE/GE/LE/GT/LT switch.
// The number of cards in `zone` (either player's) matching a Count$Valid<Zone> filter. A card
// there is controlled by its owner (CR 108.4a), so YouCtrl/OppCtrl read as YouOwn/OppOwn against
// `controller`; an empty filter counts every card.
static int count_zone_cards_matching(Zone::ZoneValue zone, std::string spec,
                                     Zone::Ownership controller) {
    if (!spec.empty() && spec[0] == ' ') spec.erase(0, 1);
    const std::string filter = spec.empty() ? std::string("Card") : owner_relative_filter(spec);
    MatchCtx ctx;
    ctx.controller = controller;
    int count = 0;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        if (global_coordinator.GetComponent<Zone>(e).location != zone) continue;
        if (card_matches_filter(e, filter, ctx)) count++;
    }
    return count;
}

// PlayerCount<Players>$HasProperty<Prop> — the number of players among <Players> (relative to
// `controller`) that have <Prop> (Kaito, Bane of Nightmares: PlayerCountRegisteredOpponents$
// HasPropertyLostLifeThisTurn = "each opponent who lost life this turn"). In the two-player game
// Opponents / RegisteredOpponents is the one opponent and Players is both seats. An unmodelled
// player set or property counts no player (CR 107.2).
static int count_players_with_property(const std::string &expr, Zone::Ownership controller) {
    size_t dollar = expr.find('$');
    const std::string players = expr.substr(std::string("PlayerCount").size(),
                                            dollar - std::string("PlayerCount").size());
    const std::string prop = expr.substr(dollar + 1);
    std::vector<Zone::Ownership> seats;
    if (players == "Opponents" || players == "RegisteredOpponents") seats = {opponent_of(controller)};
    else if (players == "Players") seats = {Zone::PLAYER_A, Zone::PLAYER_B};
    int count = 0;
    for (Zone::Ownership seat : seats) {
        Entity pe = get_player_entity(seat);
        if (!global_coordinator.entity_has_component<Player>(pe)) continue;
        const Player &pl = global_coordinator.GetComponent<Player>(pe);
        if (prop == "HasPropertyLostLifeThisTurn" && pl.life_lost_this_turn > 0) count++;
    }
    return count;
}

bool apply_svar_op(int lhs, const std::string &op2, int rhs) {
    if (op2 == "EQ") return lhs == rhs;
    if (op2 == "NE") return lhs != rhs;
    if (op2 == "GE") return lhs >= rhs;
    if (op2 == "LE") return lhs <= rhs;
    if (op2 == "GT") return lhs >  rhs;
    if (op2 == "LT") return lhs <  rhs;
    return false;
}

// Simple SVar comparison logic (shared between statics and alt costs).
// A leading two-letter operator followed by an integer; anything else is false.
bool compare_svar(int value, const std::string &compare) {
    if (compare.size() < 2) return false;
    std::string op = compare.substr(0, 2);
    if (op != "EQ" && op != "NE" && op != "GE" &&
        op != "LE" && op != "GT" && op != "LT") return false;
    return apply_svar_op(value, op, std::stoi(compare.substr(2)));
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

// Evaluate a StaticAbility SVar expression such as "Count$TypeInYourYard.Land".
// Returns the computed integer value.
int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller, Entity source) {
    // A plain integer literal (e.g. Humility's SetPower$ 1 / SetToughness$ 1) evaluates
    // to itself. Without this, a constant SetPower/SetToughness would fall through to the
    // Count$ handlers and return 0 (making the creature 0/0).
    if (!expr.empty()) {
        size_t i = (expr[0] == '-') ? 1 : 0;
        bool all_digits = i < expr.size();
        for (; i < expr.size(); ++i)
            if (!std::isdigit(static_cast<unsigned char>(expr[i]))) { all_digits = false; break; }
        if (all_digits) return std::stoi(expr);
    }

    // Handle /Plus.N suffix: strip it, evaluate the base, then add N
    size_t plus_pos = expr.find("/Plus.");
    if (plus_pos != std::string::npos) {
        std::string base = expr.substr(0, plus_pos);
        int offset = std::stoi(expr.substr(plus_pos + 6));
        return evaluate_sa_svar(base, controller, source) + offset;
    }

    // Handle /LimitMax.N suffix: strip it, evaluate the base, then cap at N
    // (e.g. "Count$ValidGraveyard Instant.YouOwn/LimitMax.1" → 0 or 1).
    size_t limit_pos = expr.find("/LimitMax.");
    if (limit_pos != std::string::npos) {
        std::string base = expr.substr(0, limit_pos);
        int cap = std::stoi(expr.substr(limit_pos + 10));
        int val = evaluate_sa_svar(base, controller, source);
        return val < cap ? val : cap;
    }

    // Count$Valid <filter> — number of battlefield permanents matching the Forge filter spec
    // (the Urza's Saga Construct token's "+1/+1 for each artifact you control": AddPower$ X,
    // X = Count$Valid Artifact.YouCtrl). Routes the whole spec through the shared
    // permanent_matches_filter so type/control/etc. qualifiers are honoured. Needed by the
    // static-buff path (gather_active_statics → evaluate_sa_svar); the spell/ability path keeps
    // its own copy in evaluate_dynamic_amount. The "$CardManaCost" sum form is handled there.
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("RememberedPlayerCtrl") == std::string::npos &&
        expr.find("$CardManaCost") == std::string::npos) {
        std::string spec = expr.substr(std::string("Count$Valid ").size());
        if (!spec.empty()) return count_battlefield_matching(spec, controller, source);
    }

    // Count$xPaid — the X value paid at cast time for the X-cost spell currently resolving
    // (Green Sun's Zenith: ChangeType$ Creature.Green+cmcLEX with SVar:X:Count$xPaid → the
    // search's mana-value bound is X): the resolving spell's own X (current_x_paid).
    if (expr == "Count$xPaid")
        return current_x_paid();

    // Count$ChosenNumber — the integer chosen by the most recent DB$ ChooseNumber effect this
    // resolution (Wrath of the Skies: the amount of energy to pay). Stored in cur_game by the
    // ChooseNumber handler so a chained sub-ability (the DestroyAll's cmc bound Y and its
    // PayEnergy<Y> unless-cost) can reference the chosen value.
    if (expr == "Count$ChosenNumber")
        return cur_game.resolution.memory.chosen_number;

    if (expr.rfind("PlayerCount", 0) == 0 && expr.find('$') != std::string::npos)
        return count_players_with_property(expr, controller);

    // Count$YourCountersEnergy — the controller's current energy ({E}) total (CR 122.1c),
    // stored as an "ENERGY" counter on the Player (Wrath of the Skies: the cap on the amount
    // of energy you may choose to pay). Reads the same counter map every {E} producer/consumer
    // uses (queries/player_resources.h player_energy / pay_energy).
    if (expr == "Count$YourCountersEnergy") {
        Entity ctrl_entity = get_player_entity(controller);
        if (!global_coordinator.entity_has_component<Player>(ctrl_entity)) return 0;
        return player_energy(global_coordinator.GetComponent<Player>(ctrl_entity));
    }

    // Count$ValidExile ... CardTypes — distinct card types among the cards exiled
    // with `source` (e.g. Keen-Eyed Curator's exiled-with pile). Scoped to the
    // source permanent, hence the `source` parameter. Only kind == TYPE entries
    // count (CR 205.2a card types) — subtypes/supertypes on the type line (Plains,
    // Legendary, ...) are excluded by the kind filter.
    if (expr.find("Count$ValidExile") != std::string::npos &&
        expr.find("CardTypes") != std::string::npos) {
        if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source))
            return 0;
        auto &eperm = global_coordinator.GetComponent<Permanent>(source);
        std::set<std::string> type_names;
        for (Entity ex_e : live_entities(eperm.exiled_with)) {
            // exiled_with is append-only, so require the entry to still be the object that was
            // exiled, sitting in the exile zone: a card that has since left exile is a new object
            // and no longer "exiled with" the source — counting it would overstate the type count.
            if (global_coordinator.GetComponent<Zone>(ex_e).location != Zone::EXILE) continue;
            if (!global_coordinator.entity_has_component<CardData>(ex_e)) continue;
            for (auto &t : global_coordinator.GetComponent<CardData>(ex_e).types)
                if (t.kind == TYPE) type_names.insert(t.name);
        }
        return static_cast<int>(type_names.size());
    }

    // Count$CardCounters.<CounterType> — number of counters of that kind on the SVar's
    // own source permanent (Aether Vial: "Count$CardCounters.CHARGE" for the charge-counter
    // count, used as the mana-value bound on the creature it can put onto the battlefield).
    // Scoped to `source`, hence the parameter (CR 122.1).
    if (expr.rfind("Count$CardCounters.", 0) == 0) {
        std::string counter_type = expr.substr(19);  // after "Count$CardCounters."
        if (source == 0) return 0;
        return get_counters(source, counter_type);
    }

    // Count$TypeInYourYard.<TypeName> — count cards of that type in controller's graveyard
    if (expr.rfind("Count$TypeInYourYard.", 0) == 0) {
        std::string type_name = expr.substr(21);  // after "Count$TypeInYourYard."
        int count = 0;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD || z.owner != controller) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.name == type_name) { count++; break; }
            }
        }
        return count;
    }

    // Count$ValidGraveyard Card$CardTypes — count distinct card types (Creature, Instant, etc.)
    // across both players' graveyards (Barrowgoyf). The Card.YouOwn variant (Nethergoyf) scopes
    // the count to the controller's own graveyard only; we detect the YouOwn/YouCtrl restriction
    // on the Card filter and, when present, skip cards owned by the other player. (CR 205.2 — the
    // distinct card types among the matching cards.)
    if (expr == "Count$CardTypesInAllGraveyards" ||
        expr == "Count$ValidGraveyard Card$CardTypes" ||
        (expr.rfind("Count$ValidGraveyard Card", 0) == 0 &&
         expr.find("$CardTypes") != std::string::npos)) {
        // Restriction lives between "Card" and "$CardTypes" (e.g. ".YouOwn" or
        // ".YouOwn+Creature"). Ownership (YouOwn/YouCtrl/OppOwn/OppCtrl) scopes the count to one
        // player's graveyard and is applied here — a graveyard card has no live controller, so the
        // shared evaluator can't read it. Any OTHER subfilter (type/CMC/color/supertype) in the
        // Card clause is honored by routing the remaining qualifiers through card_matches_filter on
        // the card's printed characteristics, instead of being silently ignored.
        size_t card_pos = expr.find("Card");
        size_t types_pos = expr.find("$CardTypes");
        std::string restriction = expr.substr(card_pos + 4, types_pos - (card_pos + 4));
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
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD) continue;
            if (you_own && z.owner != controller) continue;
            if (opp_own && z.owner == controller) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (!sub_spec.empty() && !card_matches_filter(e, sub_spec, sub_ctx)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.kind == TYPE) type_names.insert(t.name);
            }
        }
        return static_cast<int>(type_names.size());
    }

    // Count$ValidHand <filter> / Count$ValidGraveyard <filter> — the number of cards in hands /
    // graveyards matching <filter> (Ensnaring Bridge: "Count$ValidHand Card.YouOwn" = the cards in
    // your hand; Knight of the Reliquary: "Count$ValidGraveyard Land.YouOwn"). Matched by the
    // shared filter matcher against each card, with the whole grammar (comma-OR alternatives,
    // type/color/token qualifiers, OppOwn) honoured.
    if (expr.rfind("Count$ValidHand", 0) == 0)
        return count_zone_cards_matching(Zone::HAND, expr.substr(std::string("Count$ValidHand").size()),
                                         controller);
    if (expr.rfind("Count$ValidGraveyard", 0) == 0)
        return count_zone_cards_matching(
            Zone::GRAVEYARD, expr.substr(std::string("Count$ValidGraveyard").size()), controller);

    return 0;
}

// Evaluates a dynamic_amount_expr at runtime for the given controller.
// Supports: Count$InYourLibrary, Count$YourLifeTotal, Count$YourLifeTotal/HalfUp,
//           Count$Valid Creature.YouCtrl, Targeted$CardPower.
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target,
    Entity source) {
    // Count$CardCounters.<TYPE> — the number of <TYPE> counters on the ability's SOURCE permanent
    // (The One Ring: X = Count$CardCounters.BURDEN, read by its upkeep life-loss and its draw).
    // The counter type is the substring after the dot, up to any further qualifier delimiter.
    if (expr.rfind("Count$CardCounters.", 0) == 0 && source != 0) {
        std::string ctype = expr.substr(std::string("Count$CardCounters.").size());
        size_t end = ctype.find_first_of(".+ ");
        if (end != std::string::npos) ctype = ctype.substr(0, end);
        if (global_coordinator.entity_has_component<Permanent>(source)) {
            const auto &counters = global_coordinator.GetComponent<Permanent>(source).counters;
            auto it = counters.find(ctype);
            return (it != counters.end() && it->second > 0) ? static_cast<size_t>(it->second) : 0;
        }
        // LKI fallback (CR 608.2h): the source has left the battlefield (Blast Zone is sacrificed
        // as part of its own activation cost before this DestroyAll bound resolves) — use the
        // counter count snapshotted as it left play.
        // The ability refers to the departed Blast Zone even if the card has since moved again.
        if (const LastKnownInfo *lki = departed_lki_for(source)) {
            auto ci = lki->counters.find(ctype);
            if (ci != lki->counters.end() && ci->second > 0)
                return static_cast<size_t>(ci->second);
        }
        return 0;
    }
    if (expr.find("Count$Devotion.") != std::string::npos) {
        // Count mana symbols of a given color in mana costs of permanents you control
        Colors devotion_color = NO_COLOR;
        if (expr.find("Devotion.Blue") != std::string::npos)
            devotion_color = BLUE;
        else if (expr.find("Devotion.Black") != std::string::npos)
            devotion_color = BLACK;
        else if (expr.find("Devotion.Red") != std::string::npos)
            devotion_color = RED;
        else if (expr.find("Devotion.Green") != std::string::npos)
            devotion_color = GREEN;
        else if (expr.find("Devotion.White") != std::string::npos)
            devotion_color = WHITE;
        // CR 700.5: each mana symbol of that color in the mana costs of permanents you control,
        // a hybrid or Phyrexian symbol of that color included.
        size_t count = 0;
        for (auto e : battlefield_permanents(orderer->mEntities, ctrl)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            count += cd.mana_cost.count(devotion_color);
            for (const auto &pip : cd.hybrid_mana)
                if (std::find(pip.colors.begin(), pip.colors.end(), devotion_color) != pip.colors.end())
                    count++;
            count += static_cast<size_t>(
                std::count(cd.phyrexian_mana.begin(), cd.phyrexian_mana.end(), devotion_color));
        }
        return count;
    }
    // Count$xPaid — the value of X chosen for an X cost when this spell/ability was
    // cast/activated (Kozilek's Command: X = Count$xPaid feeds the token count, scry
    // count and graveyard-exile cap), read through current_x_paid().
    if (expr.find("xPaid") != std::string::npos) {
        return static_cast<size_t>(current_x_paid());
    }
    // Count$Converge (CR 702.90) — the number of distinct colors of mana spent to cast the spell
    // currently resolving (Prismatic Ending: the cmcLEY exile threshold), captured from its
    // Spell::colors_spent as it starts resolving.
    if (expr.find("Count$Converge") != std::string::npos) {
        return static_cast<size_t>(current_converge());
    }
    if (expr.find("Count$InYourLibrary") != std::string::npos ||
        expr.find("Count$ValidLibrary Card.YouOwn") != std::string::npos) {
        size_t lib = orderer->get_library_contents(ctrl).size();
        // /HalfUp — half the count, rounded up (Tamiyo, Seasoned Scholar's -7: "draw cards
        // equal to half the number of cards in your library, rounded up"). ceil(N/2).
        if (expr.find("/HalfUp") != std::string::npos) return (lib + 1) / 2;
        return lib;
    }
    if (expr.find("Count$YourLifeTotal") != std::string::npos) {
        Entity ctrl_entity = get_player_entity(ctrl);
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        int life = player.life_total;
        if (life < 0) life = 0;
        if (expr.find("/HalfUp") != std::string::npos) {
            return static_cast<size_t>((life + 1) / 2);
        }
        return static_cast<size_t>(life);
    }
    // Count$ValidStack <filter> — number of stack objects matching a card filter (Mindbreak
    // Trap: TargetMax$ MaxTgts, MaxTgts = Count$ValidStack Card — the cap on "exile any number
    // of target spells" is the number of spell cards on the stack). Card-shaped filters are
    // matched through the shared comma-OR card filter against each spell's printed
    // characteristics; standalone ability entities (no CardData) are not cards and don't count.
    // The evaluating ability's own source is excluded — a spell can never target itself, so
    // counting it would only inflate the cap past the real candidate pool.
    if (expr.rfind("Count$ValidStack ", 0) == 0) {
        std::string spec = expr.substr(std::string("Count$ValidStack ").size());
        size_t count = 0;
        for (auto e : orderer->get_stack()) {
            if (e == source) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (!card_matches_any(e, spec, MatchCtx{ctrl, source})) continue;
            count++;
        }
        return count;
    }
    // Count$Valid <filter>$CardManaCost — the SUM of mana values of battlefield permanents matching
    // the filter, rather than their count (Summon: Bahamut's Mega Flare: X = Count$Valid
    // Permanent.YouCtrl+Other$CardManaCost = the total mana value of OTHER permanents you control).
    // The "+Other" qualifier excludes the ability's own source (the Bahamut); `source` is threaded
    // into the match context for it. A token / costless permanent contributes mana value 0.
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("$CardManaCost") != std::string::npos) {
        std::string rest = expr.substr(std::string("Count$Valid ").size());
        size_t dollar = rest.rfind("$CardManaCost");
        std::string spec = rest.substr(0, dollar);  // the filter, e.g. "Permanent.YouCtrl+Other"
        if (!spec.empty()) {
            MatchCtx mctx;
            mctx.controller = ctrl;  // "you" reference for YouCtrl/OppCtrl
            mctx.source = source;    // for the +Other qualifier (exclude the source)
            size_t total = 0;
            for (auto e : orderer->mEntities) {
                if (!is_battlefield_permanent(e)) continue;
                if (!permanent_matches_filter(e, spec, mctx)) continue;
                if (global_coordinator.entity_has_component<CardData>(e))
                    total += static_cast<size_t>(
                        card_mana_value(global_coordinator.GetComponent<CardData>(e)));
            }
            return total;
        }
    }
    // Count$Valid <filter>.TargetedPlayerCtrl — number of battlefield permanents matching the
    // filter controlled by the PLAYER this ability targets (Carpet of Flowers: Islands the target
    // opponent controls, via the curse-Pump's ValidTgts$ Opponent inherited by the Mana sub).
    // `target` is that Player entity; evaluate the filter from its perspective by rewriting
    // TargetedPlayerCtrl → YouCtrl and counting with the target player's ownership. Handled before
    // the generic Count$Valid branch (which would treat TargetedPlayerCtrl as an unknown qualifier).
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("TargetedPlayerCtrl") != std::string::npos) {
        std::string spec = expr.substr(std::string("Count$Valid ").size());
        size_t pos = spec.find("TargetedPlayerCtrl");
        spec.replace(pos, std::string("TargetedPlayerCtrl").size(), "YouCtrl");
        Zone::Ownership tgt_ctrl = Zone::UNKNOWN;
        if (target == cur_game.player_a_entity)      tgt_ctrl = Zone::PLAYER_A;
        else if (target == cur_game.player_b_entity) tgt_ctrl = Zone::PLAYER_B;
        if (tgt_ctrl == Zone::UNKNOWN) return 0;
        return static_cast<size_t>(count_battlefield_matching(spec, tgt_ctrl, source));
    }
    // Count$Valid <Filter> — number of battlefield permanents matching the full Forge filter
    // spec (e.g. Eldrazi Linebreaker: "Count$Valid Eldrazi.YouCtrl"; Eiganjo's Channel
    // ReduceCost: "Count$Valid Creature.Legendary+YouCtrl" = legendary creatures you control).
    // It routes the whole spec (head type + '.'/'+'-joined qualifiers like Legendary/YouCtrl/
    // colors) through the shared permanent_matches_filter so supertype/color/etc. qualifiers
    // are honored, not just the head type. The RememberedPlayerCtrl form is excluded so it
    // falls through to its dedicated handler below (it needs the remembered-player reference
    // and the /Times multiplier, neither of which permanent_matches_filter understands).
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("RememberedPlayerCtrl") == std::string::npos &&
        expr.find("$CardManaCost") == std::string::npos) {
        std::string spec = expr.substr(std::string("Count$Valid ").size());  // full filter spec
        if (!spec.empty())
            return static_cast<size_t>(count_battlefield_matching(spec, ctrl, source));
    }
    // Count$Revolt.high.low — returns high if revolt active for controller, low otherwise
    if (expr.find("Count$Revolt.") != std::string::npos) {
        size_t dot1 = expr.find("Revolt.") + 7;
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool revolt = revolt_this_turn(ctrl);
        return static_cast<size_t>(revolt ? high_val : low_val);
    }
    // Count$PromisedGift.high.low — Gift (CR 702.176): returns high if the spell currently being
    // cast/resolved promised its gift to an opponent, low otherwise. Into the Flood Maw drives its
    // two ChangeZone abilities' TargetMin$/TargetMax$ off this (X = .0.1, Y = .1.0): not promised →
    // the creature-bounce targets 1 and the nonland-bounce targets 0; promised → the reverse, so
    // the spell instead bounces any nonland permanent. Read from the cast-time pending flag (the
    // target counts are evaluated as targets are chosen, before the Spell component exists).
    if (expr.find("Count$PromisedGift.") != std::string::npos) {
        size_t dot1 = expr.find("PromisedGift.") + std::string("PromisedGift.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        return static_cast<size_t>(current_gift_promised() ? high_val : low_val);
    }
    // Count$Threshold.high.low — Threshold (CR 702.27 historical keyword action; modern cards
    // spell the condition out): returns high if the controller has seven or more cards in their
    // graveyard, low otherwise (Cabal Ritual: Count$Threshold.5.3 → 5 black mana with threshold,
    // else 3). General for any card scaling a dynamic amount by the threshold condition.
    if (expr.find("Count$Threshold.") != std::string::npos) {
        size_t dot1 = expr.find("Threshold.") + std::string("Threshold.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool threshold = orderer->get_graveyard(ctrl).size() >= 7;
        return static_cast<size_t>(threshold ? high_val : low_val);
    }
    // Count$UrzaLands.high.low — the "Tron" mana lands (Urza's Mine/Power Plant/Tower): returns
    // high if the controller controls at least one Urza's Mine AND one Urza's Power-Plant AND one
    // Urza's Tower (a complete set), low otherwise. Per CR 205.3i these are LAND TYPES, so the
    // check reads each permanent's effective type line (Permanent::types — includes types added
    // by continuous effects, e.g. Planar Nexus's AllNonBasicLandType self-CDA), not card names.
    // One permanent with several of the subtypes (Nexus) satisfies each it carries. Each land's
    // own ability scales its colorless output (Mine/Power Plant: .2.1 → {C}{C} assembled / {C}
    // alone; Tower: .3.1 → {C}{C}{C} / {C}). General over any card scaling a dynamic amount by
    // Tron assembly.
    if (expr.find("Count$UrzaLands.") != std::string::npos) {
        size_t dot1 = expr.find("UrzaLands.") + std::string("UrzaLands.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool mine = false, plant = false, tower = false;
        for (auto e : battlefield_permanents(orderer->mEntities, ctrl)) {
            const auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (!permanent_has_type(perm, "Urza's")) continue;
            if (permanent_has_type(perm, "Mine")) mine = true;
            if (permanent_has_type(perm, "Power-Plant")) plant = true;
            if (permanent_has_type(perm, "Tower")) tower = true;
        }
        return static_cast<size_t>((mine && plant && tower) ? high_val : low_val);
    }
    if (expr.find("Targeted$CardPower") != std::string::npos) {
        // CR 608.2h: effective power, read live while the creature is in play (counters/buffs
        // included), else its last-known value once it has left (e.g. Swords to Plowshares
        // reads the power of the creature it just exiled). Single unified accessor.
        int p = effective_power(target);
        return static_cast<size_t>(p < 0 ? 0 : p);
    }
    // ExiledWith$CardManaCost — the mana value of the card the source Saga exiled face down (The
    // Creation of Avacyn chapter II: "you lose life equal to its mana value"). Resolved from the
    // source's Permanent::exiled_with; an absent/gone card contributes 0.
    if (expr.find("ExiledWith$CardManaCost") != std::string::npos) {
        int mv = 0;
        Entity ew = exiled_with_card(source);
        if (ew != 0 && global_coordinator.entity_has_component<CardData>(ew))
            mv = object_mana_value(ew, global_coordinator.GetComponent<CardData>(ew));
        return static_cast<size_t>(mv < 0 ? 0 : mv);
    }
    if (expr.find("Targeted$CardManaCost") != std::string::npos) {
        // The target's mana value (CR 202.3 / 107.14). Used by Karn, the Great Creator's +1
        // Animate (Power$/Toughness$ X, X = Targeted$CardManaCost): the animated permanent
        // becomes a creature whose P/T equal its own mana value, snapshotted at resolution.
        int mv = 0;
        if (target != 0 && global_coordinator.entity_has_component<CardData>(target))
            mv = object_mana_value(target, global_coordinator.GetComponent<CardData>(target));
        return static_cast<size_t>(mv < 0 ? 0 : mv);
    }
    // Count$RememberedSize / RememberedSize — the total number of currently-remembered objects
    // (cur_game.resolution.memory.remembered), regardless of type. Triumph of Saint Katherine's recursion
    // gates its shuffle-back on "RememberedSize GE7" — the self-exiled card plus the six milled
    // cards. Distinct from Remembered$Valid, which filters by card characteristics.
    if (expr == "Count$RememberedSize" || expr == "RememberedSize")
        return cur_game.resolution.memory.remembered.size();
    // Remembered$Valid <comma-OR-filter> — number of remembered cards (e.g. cards just moved
    // by a RememberChanged$ ChangeZoneAll) matching ANY of the comma-separated filters (Canoptek
    // Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each artifact or land card exiled
    // this way"). These are now in their destination zone (e.g. exile), so match by printed
    // characteristics via the shared card_matches_filter; control qualifiers resolve against ctrl.
    if (expr.rfind("Remembered$Valid ", 0) == 0) {
        std::string filters = expr.substr(std::string("Remembered$Valid ").size());
        MatchCtx mctx;
        mctx.controller = ctrl;  // "you" reference for YouCtrl/OppCtrl in any filter
        size_t count = 0;
        for (Entity e : lki_entities(cur_game.resolution.memory.remembered)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (card_matches_any(e, filters, mctx)) count++;  // ',' = OR over the filters
        }
        return count;
    }
    // Remembered$CardManaCost[/Plus.N] — mana value of the first remembered card (Birthing
    // Ritual: X = 1 plus the sacrificed creature's mana value). The RememberedLKI$ variant
    // reads the same remembered entity, but is populated by a RememberLKI$ ChangeZone that
    // snapshots the card as last-known info once it has left its origin zone (Reanimate: you
    // lose life equal to the reanimated creature's mana value — CR 608.2h last-known-info,
    // since the card left the graveyard as it entered play). Both forms resolve identically
    // here because the LKI snapshot is pushed to cur_game.resolution.memory.remembered all the same.
    if (expr.find("Remembered$CardManaCost") != std::string::npos ||
        expr.find("RememberedLKI$CardManaCost") != std::string::npos) {
        int base = 0;
        if (!cur_game.resolution.memory.remembered.empty()) {
            Entity r = cur_game.resolution.memory.remembered[0].lki_entity();
            if (global_coordinator.entity_has_component<CardData>(r))
                base = object_mana_value(r, global_coordinator.GetComponent<CardData>(r));
        }
        size_t plus = expr.find("/Plus.");
        if (plus != std::string::npos) base += std::stoi(expr.substr(plus + 6));
        return static_cast<size_t>(base < 0 ? 0 : base);
    }
    // Count$Valid Land.nonBasic+RememberedPlayerCtrl[/Times.N] — number of nonbasic
    // lands controlled by the remembered player (Price of Progress, evaluated once per
    // player by the RepeatEach loop), optionally multiplied by N. The remembered player
    // is cur_game.resolution.memory.remembered[0] (a Player entity set by the repeat_each handler).
    if (expr.find("Count$Valid Land.nonBasic+RememberedPlayerCtrl") != std::string::npos) {
        Zone::Ownership remembered_ctrl = ctrl;
        if (!cur_game.resolution.memory.remembered.empty()) {
            Entity rp = cur_game.resolution.memory.remembered[0].get();
            if (rp == cur_game.player_a_entity) remembered_ctrl = Zone::PLAYER_A;
            else if (rp == cur_game.player_b_entity) remembered_ctrl = Zone::PLAYER_B;
        }
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!is_battlefield_permanent(e, remembered_ctrl)) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool is_land = false;
            for (auto &t : cd.types)
                if (t.name == "Land") { is_land = true; break; }
            if (!is_land) continue;
            if (has_basic_supertype(cd.types)) continue;  // nonBasic only
            count++;
        }
        size_t mult = 1;
        size_t times_pos = expr.find("/Times.");
        if (times_pos != std::string::npos)
            mult = static_cast<size_t>(std::stoi(expr.substr(times_pos + 7)));
        return count * mult;
    }
    // Count$ThisTurnCast_Card.<Ctrl>+<Color>[,Card.<Ctrl>+<Color>...] — has a player (relative to
    // ctrl) cast a spell of one of the named colors this turn? (Veil of Summer:
    // Count$ThisTurnCast_Card.OppCtrl+Blue,Card.OppCtrl+Black, the gate for its conditional draw.)
    // Reads Player::spell_colors_cast_this_turn (presence-tracked per color); returns 1 if any
    // requested color was cast by the relevant player this turn, else 0 — sufficient for the GE1
    // conditions that consume it.
    if (expr.find("Count$ThisTurnCast_") != std::string::npos) {
        Zone::Ownership opp = opponent_of(ctrl);
        // The clause controller token is read per-expression (Veil uses OppCtrl); YouCtrl (or no
        // controller token) means the source's controller.
        Zone::Ownership who = (expr.find("OppCtrl") != std::string::npos) ? opp : ctrl;
        Entity pe = get_player_entity(who);
        if (global_coordinator.entity_has_component<Player>(pe)) {
            const auto &colors = global_coordinator.GetComponent<Player>(pe).spell_colors_cast_this_turn;
            bool hit = (expr.find("Blue") != std::string::npos && colors.count(BLUE)) ||
                       (expr.find("Black") != std::string::npos && colors.count(BLACK)) ||
                       (expr.find("Red") != std::string::npos && colors.count(RED)) ||
                       (expr.find("Green") != std::string::npos && colors.count(GREEN)) ||
                       (expr.find("White") != std::string::npos && colors.count(WHITE));
            return hit ? 1 : 0;
        }
        return 0;
    }
    // Fall back to the shared static-ability SVar evaluator for graveyard-count
    // expressions (Count$TypeInYourYard / Count$ValidGraveyard / CardTypes). It
    // returns 0 for anything it doesn't recognise, so this preserves the prior
    // default while making one set of Count$ handlers serve both paths.
    int sa_val = evaluate_sa_svar(expr, ctrl);
    return sa_val > 0 ? static_cast<size_t>(sa_val) : 0;
}
