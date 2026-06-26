#include "mana_system.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <tuple>

#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/static_ability.h"
#include "components/types.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "game_queries.h"
#include "input_logger.h"
#include "systems/orderer.h"
#include "systems/rules_modifying.h"
#include "systems/state_manager.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static size_t eval_mana_amount(const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);
static ManaValue pay_from_pool(ManaValue &pool, const ManaValue &cost);
static bool auto_pay_mana(Zone::Ownership controller, ManaValue &remaining,
                          Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve,
                          bool commit = true);
static bool restricted_mana_matches(Entity source_entity, Entity paid_for);
static bool is_delve_eligible(Entity e, Zone::Ownership controller);
static void delve_exile_one(Entity e, Zone::Ownership controller,
                            std::shared_ptr<Orderer> orderer, ManaValue &remaining);
static void activate_mana_source(Entity entity, const Ability &ab, Zone::Ownership controller,
                                 std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                 Player &player, bool commit);

Entity get_player_entity(Zone::Ownership player) {
    return (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
}

// Non-mutating affordability check: the read-only counterpart of pay_from_pool.
// Dry-runs the payment on a throwaway copy (this function already copied the pool)
// so the spend rule is never duplicated.
bool can_afford_pool(const std::multiset<Colors> &pool, const std::multiset<Colors> &cost) {
    auto copy = pool;
    return pay_from_pool(copy, cost).empty();
}

bool can_afford(Zone::Ownership player_owner, const std::multiset<Colors> &cost) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) {
        return false;
    }
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    return can_afford_pool(player.mana, cost);
}

void spend_mana(Zone::Ownership player_owner, const std::multiset<Colors> &cost, Entity paid_for) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    if (!can_afford_pool(player.mana, cost)) {
        non_fatal_error("spend_mana called with insufficient mana in pool");
        #ifndef NDEBUG
        dump_entity(paid_for);
        #endif
    }
    pay_from_pool(player.mana, cost);
}

void add_mana(Zone::Ownership player_owner, Colors mana_color, size_t amount) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    for (size_t i = 0; i < amount; i++) {
        player.mana.insert(mana_color);
    }
}

ManaValue pay_partial(Zone::Ownership player_owner, const ManaValue &cost) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    return pay_from_pool(player.mana, cost);
}

void empty_mana_pool(Zone::Ownership player_owner) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    if (!player.mana.empty() && InputLogger::instance().is_machine_mode()) {
        // Signal the Python env to apply a shaping penalty to the model if this
        // player's side is the one being trained.
        const char* side = (player_owner == Zone::PLAYER_A) ? "A" : "B";
        printf("MANA_WASTED: %s\n", side);
        fflush(stdout);
    }
    player.mana.clear();
}

// Check if a permanent's abilities are suppressed by a CantBeActivated static
// Evaluate the mana amount a source produces (handles dynamic amounts like Gaea's Cradle)
static size_t eval_mana_amount(const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer) {
    if (!ab.dynamic_amount_expr.empty() &&
        ab.dynamic_amount_expr.find("Count$Valid Creature.YouCtrl") != std::string::npos) {
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Creature>(e)) continue;
            auto &sz = global_coordinator.GetComponent<Zone>(e);
            if (sz.location != Zone::BATTLEFIELD) continue;
            if (global_coordinator.GetComponent<Permanent>(e).controller == controller) count++;
        }
        return count;
    }
    return ab.amount;
}

// Check if a restricted mana source (Cavern of Souls) can be used to pay for a spell.
// Returns true if the spell is a creature with the source permanent's chosen subtype.
static bool restricted_mana_matches(Entity source_entity, Entity paid_for) {
    if (paid_for == 0) return false;
    auto &source_perm = global_coordinator.GetComponent<Permanent>(source_entity);
    if (source_perm.chosen_type.empty()) return false;
    if (!global_coordinator.entity_has_component<CardData>(paid_for)) return false;
    auto &paid_cd = global_coordinator.GetComponent<CardData>(paid_for);
    bool is_creature = false, has_chosen_subtype = false;
    for (auto &t : paid_cd.types) {
        if (t.kind == TYPE && t.name == "Creature") is_creature = true;
        if (t.kind == SUBTYPE && t.name == source_perm.chosen_type) has_chosen_subtype = true;
    }
    return is_creature && has_chosen_subtype;
}

// Collect all mana abilities a player could activate.
// Checks physical activation requirements (untapped, controller, phased out, CantBeActivated,
// summoning sickness, activation limits) but NOT activation_mana_cost — callers handle that
// to avoid circularity with can_afford_with_sources.
static std::vector<std::pair<Entity, Ability>> collect_available_mana_sources(
    Zone::Ownership player, std::shared_ptr<Orderer> orderer, bool include_instant_speed = false) {
    std::vector<std::pair<Entity, Ability>> sources;
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, player)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (rules_mod::mana_activation_prohibited(entity)) continue;

        for (const auto &ab : permanent.abilities) {
            if (ab.category != "AddMana") continue;
            if (ab.ability_type != Ability::ACTIVATED) continue;
            // InstantSpeed$ mana abilities (e.g. LED) may only be activated at priority, not
            // mid-cost-payment. Callers listing actions for a player who holds priority pass
            // include_instant_speed; the affordability/payment callers leave it false.
            if (ab.instant_speed && !include_instant_speed) continue;
            if (ab.tap_cost && permanent.is_tapped) continue;
            if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) continue;
            // Summoning sickness check for creatures with tap cost
            if (ab.tap_cost && permanent.has_summoning_sickness &&
                global_coordinator.entity_has_component<Creature>(entity)) {
                auto &cr = global_coordinator.GetComponent<Creature>(entity);
                bool has_haste = false;
                for (const auto &kw : cr.keywords)
                    if (kw == "Haste") { has_haste = true; break; }
                if (!has_haste) continue;
            }
            if (!ab.mana_choices.empty()) {
                for (Colors choice_color : ab.mana_choices) {
                    Ability choice_ab = ab;
                    choice_ab.color = choice_color;
                    sources.push_back({entity, choice_ab});
                }
            } else {
                sources.push_back({entity, ab});
            }
        }
    }
    return sources;
}

ActionCategory mana_action_category(Colors color) {
    switch (color) {
        case WHITE: return ActionCategory::MANA_W;
        case BLUE:  return ActionCategory::MANA_U;
        case BLACK: return ActionCategory::MANA_B;
        case RED:   return ActionCategory::MANA_R;
        case GREEN: return ActionCategory::MANA_G;
        default:    return ActionCategory::MANA_C;
    }
}

std::vector<LegalAction> collect_mana_legal_actions(
    Zone::Ownership player, std::shared_ptr<Orderer> orderer, Entity paid_for, bool at_priority) {
    std::vector<LegalAction> actions;
    auto sources = collect_available_mana_sources(player, orderer, at_priority);
    for (auto &[entity, ab] : sources) {
        // Filter restricted mana (Cavern of Souls): hide from payment when spell doesn't match
        if (ab.restrict_to_chosen_type_creature && !restricted_mana_matches(entity, paid_for))
            continue;
        // Sources with activation mana cost: check affordability
        if (!ab.activation_mana_cost.empty()) {
            Entity exclude = ab.tap_cost ? entity : 0;
            if (!can_afford_with_sources(player, ab.activation_mana_cost, orderer, exclude))
                continue;
        }
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        std::string desc = "Tap " + perm.name + " for (" +
                           mana_symbol_str(ab.color) + ")";
        LegalAction la(ACTIVATE_ABILITY, entity, ab, desc);
        la.category = mana_action_category(ab.color);
        actions.push_back(la);
    }
    return actions;
}

bool can_afford_with_sources(Zone::Ownership player_owner, const std::multiset<Colors> &cost,
                             std::shared_ptr<Orderer> orderer, Entity exclude_entity,
                             Entity paid_for) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Fast path: pool alone is enough
    if (can_afford_pool(player.mana, cost)) return true;

    // Build a hypothetical pool: current mana + all available sources
    auto hypothetical = player.mana;
    std::set<Entity> counted_entities;
    size_t flexible_count = 0;

    auto sources = collect_available_mana_sources(player_owner, orderer);

    // Filter restricted mana sources (Cavern of Souls): exclude unless spell matches
    sources.erase(std::remove_if(sources.begin(), sources.end(),
        [&](const std::pair<Entity, Ability> &s) {
            return s.second.restrict_to_chosen_type_creature &&
                   !restricted_mana_matches(s.first, paid_for);
        }), sources.end());

    // First pass: add free sources (no activation mana cost)
    // Multi-color sources appear as multiple entries for the same entity
    // (one per color choice); count those as flexible mana.
    for (auto &[entity, ab] : sources) {
        if (entity == exclude_entity) continue;
        if (counted_entities.count(entity)) continue;
        if (!ab.activation_mana_cost.empty()) continue;
        size_t amount = eval_mana_amount(ab, player_owner, orderer);
        // Check if this entity has more entries (multi-color source)
        bool is_multi_color = false;
        for (auto &[e2, ab2] : sources) {
            if (e2 == entity && ab2.color != ab.color) {
                is_multi_color = true;
                break;
            }
        }
        if (is_multi_color) {
            flexible_count += amount;
        } else {
            for (size_t i = 0; i < amount; i++) hypothetical.insert(ab.color);
        }
        counted_entities.insert(entity);
    }

    // Second pass: sources with activation mana cost — count them only if
    // the hypothetical pool (from free sources) can cover the activation cost
    for (auto &[entity, ab] : sources) {
        if (entity == exclude_entity) continue;
        if (counted_entities.count(entity)) continue;
        if (ab.activation_mana_cost.empty()) continue;
        // Build pool snapshot to check activation affordability
        auto check_pool = hypothetical;
        for (size_t i = 0; i < flexible_count; i++) check_pool.insert(GENERIC);
        if (!can_afford_pool(check_pool, ab.activation_mana_cost)) continue;
        size_t amount = eval_mana_amount(ab, player_owner, orderer);
        bool is_multi_color = false;
        for (auto &[e2, ab2] : sources) {
            if (e2 == entity && ab2.color != ab.color) {
                is_multi_color = true;
                break;
            }
        }
        if (is_multi_color) {
            flexible_count += amount;
        } else {
            for (size_t i = 0; i < amount; i++) hypothetical.insert(ab.color);
        }
        // Subtract the activation cost from the hypothetical pool
        for (auto c : ab.activation_mana_cost) {
            if (c == GENERIC) continue;
            auto it = hypothetical.find(c);
            if (it != hypothetical.end()) hypothetical.erase(it);
            else if (flexible_count > 0) flexible_count--;
        }
        size_t generic_activation = ab.activation_mana_cost.count(GENERIC);
        for (size_t i = 0; i < generic_activation; i++) {
            if (!hypothetical.empty()) {
                hypothetical.erase(hypothetical.begin());
            } else if (flexible_count > 0) {
                flexible_count--;
            }
        }
        counted_entities.insert(entity);
    }

    // Try to pay colored costs from the hypothetical pool
    auto remaining = hypothetical;
    size_t flexible_used = 0;
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = remaining.find(color);
        if (it != remaining.end()) {
            remaining.erase(it);
        } else if (flexible_used < flexible_count) {
            flexible_used++;
        } else {
            return false;
        }
    }

    size_t generic_needed = cost.count(GENERIC);
    size_t available_for_generic = remaining.size() + (flexible_count - flexible_used);
    return available_for_generic >= generic_needed;
}

size_t max_available_mana(Zone::Ownership player_owner, const ManaValue &base_cost,
                          std::shared_ptr<Orderer> orderer) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return 0;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Count total mana available: pool + all sources
    size_t total = player.mana.size();
    std::set<Entity> counted;
    auto sources = collect_available_mana_sources(player_owner, orderer);
    for (auto &[entity, ab] : sources) {
        if (counted.count(entity)) continue;
        total += eval_mana_amount(ab, player_owner, orderer);
        counted.insert(entity);
    }

    // Subtract colored requirements from base cost (generic is what X adds to)
    size_t colored_obligations = 0;
    for (auto c : base_cost) {
        if (c != GENERIC) colored_obligations++;
    }
    size_t generic_in_base = base_cost.count(GENERIC);
    size_t fixed_cost = colored_obligations + generic_in_base;
    return (total > fixed_cost) ? total - fixed_cost : 0;
}

ManaPaymentSnapshot snapshot_mana_state(Zone::Ownership player, std::shared_ptr<Orderer> orderer) {
    ManaPaymentSnapshot snap;
    Entity player_entity = get_player_entity(player);
    snap.player_mana = global_coordinator.GetComponent<Player>(player_entity).mana;
    snap.delve_exiled = cur_game.delve_exiled;

    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != player) continue;

        snap.tapped_state.push_back({entity, permanent.is_tapped});
        for (size_t i = 0; i < permanent.abilities.size(); i++) {
            if (permanent.abilities[i].category == "AddMana") {
                snap.activation_counts.push_back({entity, i, permanent.abilities[i].activations_this_turn});
            }
        }
    }
    return snap;
}

void restore_mana_state(Zone::Ownership player, const ManaPaymentSnapshot &snap,
                        std::shared_ptr<Orderer> orderer) {
    Entity player_entity = get_player_entity(player);
    global_coordinator.GetComponent<Player>(player_entity).mana = snap.player_mana;

    for (auto &[entity, was_tapped] : snap.tapped_state) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        global_coordinator.GetComponent<Permanent>(entity).is_tapped = was_tapped;
    }
    for (auto &[entity, idx, old_count] : snap.activation_counts) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (idx < permanent.abilities.size()) {
            permanent.abilities[idx].activations_this_turn = old_count;
        }
    }
    // Undo delve exiles: move cards exiled since snapshot back to graveyard
    for (size_t i = snap.delve_exiled.size(); i < cur_game.delve_exiled.size(); i++) {
        Entity exiled = cur_game.delve_exiled[i];
        orderer->add_to_zone(false, exiled, Zone::GRAVEYARD);
    }
    cur_game.delve_exiled = snap.delve_exiled;
    cur_game.pending_cant_be_countered = false;
}

// The single "pay colored pips first, then generic from any remaining mana" primitive.
// Pays `cost` out of `pool` (mutating it) and returns the portion that could NOT be
// paid. can_afford_pool/spend_mana/pay_partial and the auto-payer all route through this,
// so the spend rule (including generic color preference) lives in exactly one place.
static ManaValue pay_from_pool(ManaValue &pool, const ManaValue &cost) {
    ManaValue remaining;
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = pool.find(color);
        if (it != pool.end()) pool.erase(it);
        else remaining.insert(color);
    }
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!pool.empty()) pool.erase(pool.begin());
        else remaining.insert(GENERIC);
    }
    return remaining;
}

// True if `e` is a graveyard instant/sorcery owned by `controller` — i.e. a card
// Delve can exile to pay a generic pip. Single source for "what can Delve eat",
// consumed by both the automatic payer and the interactive prompt.
static bool is_delve_eligible(Entity e, Zone::Ownership controller) {
    if (!global_coordinator.entity_has_component<Zone>(e)) return false;
    auto &ez = global_coordinator.GetComponent<Zone>(e);
    if (ez.location != Zone::GRAVEYARD || ez.owner != controller) return false;
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
        if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) return true;
    return false;
}

// Pay one generic pip via Delve: exile `e`, record it for the etbCounter count, and
// drop one GENERIC from `remaining`. Single source for the delve-exile action used by
// both payment paths.
static void delve_exile_one(Entity e, Zone::Ownership controller,
                            std::shared_ptr<Orderer> orderer, ManaValue &remaining) {
    auto &ecd = global_coordinator.GetComponent<CardData>(e);
    orderer->add_to_zone(false, e, Zone::EXILE);
    cur_game.delve_exiled.push_back(e);
    auto git = remaining.find(GENERIC);
    if (git != remaining.end()) remaining.erase(git);
    game_log("%s exiles %s via Delve.\n", player_name(controller).c_str(), ecd.name.c_str());
}

void increment_activation_count(Permanent &perm, const Ability &ability) {
    if (ability.activation_limit <= 0) return;
    for (auto &perm_ab : perm.abilities) {
        if (perm_ab.category != ability.category) continue;
        // Mana abilities are keyed by tap/color; other activated abilities by return cost.
        bool match = (ability.category == "AddMana")
                         ? (perm_ab.tap_cost == ability.tap_cost && perm_ab.color == ability.color)
                         : (perm_ab.return_cost_type == ability.return_cost_type);
        if (match) {
            perm_ab.activations_this_turn++;
            break;
        }
    }
}

// Activate one mana source: tap/sacrifice it, pay its activation mana + life cost, add the
// mana it produces to `pool`, and (when committing) flag uncounterability, log, and bump the
// activation counter. Pool changes (activation cost paid, mana produced) always apply to the
// working `pool`; the write-only ECS side effects are skipped when !commit (simulate mode).
// Shared by the auto-payer (commit toggles per simulate/real) and the interactive payer
// (always commit, with pool == the player's real mana pool).
static void activate_mana_source(Entity entity, const Ability &ab, Zone::Ownership controller,
                                 std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                 Player &player, bool commit) {
    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
    if (commit && ab.tap_cost) perm.is_tapped = true;
    if (commit && ab.sac_self) {
        game_log("%s sacrifices %s\n", player_name(controller).c_str(), perm.name.c_str());
        orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
    }
    if (!ab.activation_mana_cost.empty())
        pay_from_pool(pool, ab.activation_mana_cost);
    if (commit && ab.life_cost > 0) {
        player.life_total -= ab.life_cost;
        game_log("%s pays %d life\n", player_name(controller).c_str(), ab.life_cost);
    }
    size_t amount = eval_mana_amount(ab, controller, orderer);
    for (size_t i = 0; i < amount; i++) pool.insert(ab.color);
    if (commit && ab.adds_no_counter) cur_game.pending_cant_be_countered = true;
    if (commit)
        game_log("%s activated %s for %zu(%s)\n", player_name(controller).c_str(),
                 perm.name.c_str(), amount, mana_symbol_str(ab.color));
    if (commit) increment_activation_count(perm, ab);
}

// Greedily tap sources to cover the remaining cost. This is the single mana-payment
// algorithm used by BOTH the machine-mode payer (commit=true, mutates real ECS state)
// and the legality check via can_pay_mana (commit=false, operates on a copied pool with
// no side effects) — so "is this castable" and "pay for it" can never disagree.
//
// Strategy: for each colored pip in the remaining cost, find a source that produces
// exactly that color, preferring single-color sources over multi-color to preserve
// flexibility. Then pay generic costs with whatever is left. Delve: exile graveyard
// instants/sorceries to reduce generic costs first.
//
// When commit==false the write-only side effects (tap, sacrifice, life payment, delve
// zone moves, activation counters, uncounterable flag) are skipped. None of these are
// re-read to drive the payment decision — reuse is gated by the local `tapped_entities`
// set — so the decision sequence is identical to a real payment.
static bool auto_pay_mana(Zone::Ownership controller, ManaValue &remaining,
                          Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve,
                          bool commit) {
    Entity player_entity = get_player_entity(controller);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // In commit mode the working pool IS the real mana pool; in simulate mode it is a
    // throwaway copy so the algorithm can drain/refill it without touching real state.
    ManaValue pool_copy = player.mana;
    ManaValue &pool = commit ? player.mana : pool_copy;

    // Delve: exile graveyard instants/sorceries to reduce generic portion
    if (has_delve) {
        for (auto e : orderer->mEntities) {
            if (remaining.count(GENERIC) == 0) break;
            if (!is_delve_eligible(e, controller)) continue;
            if (commit) {
                delve_exile_one(e, controller, orderer, remaining);
            } else {
                auto git = remaining.find(GENERIC);
                if (git != remaining.end()) remaining.erase(git);
            }
        }
    }

    // Check if pool covers remaining after delve
    if (can_afford_pool(pool, remaining)) {
        pay_from_pool(pool, remaining);
        return true;
    }

    // Collect available sources with their color info
    auto sources = collect_available_mana_sources(controller, orderer);

    // Build per-entity info: which colors it can produce, how many entries
    struct SourceInfo {
        Entity entity;
        Ability ability;        // one representative ability (for tap/life/activation costs)
        Colors color;
        bool is_multi_color;
    };
    // Filter to actions valid for this payment (same as collect_mana_legal_actions)
    std::vector<SourceInfo> valid_sources;
    for (auto &[entity, ab] : sources) {
        // Restricted mana check (Cavern of Souls)
        if (ab.restrict_to_chosen_type_creature && !restricted_mana_matches(entity, paid_for))
            continue;
        if (!ab.activation_mana_cost.empty()) {
            if (!can_afford_with_sources(controller, ab.activation_mana_cost, orderer, ab.tap_cost ? entity : 0))
                continue;
        }
        bool is_multi = false;
        for (auto &[e2, ab2] : sources)
            if (e2 == entity && ab2.color != ab.color) { is_multi = true; break; }
        valid_sources.push_back({entity, ab, ab.color, is_multi});
    }

    // Helper: activate a source (tap, sacrifice, pay costs, add mana). si.color always
    // equals si.ability.color (set when the SourceInfo was built), so the shared
    // activate_mana_source reads the produced color straight off the ability.
    auto activate_source = [&](const SourceInfo &si) {
        activate_mana_source(si.entity, si.ability, controller, orderer, pool, player, commit);
    };

    std::set<Entity> tapped_entities;

    // Pay colored costs first — prefer single-color sources to preserve flexibility
    for (auto it = remaining.begin(); it != remaining.end(); ) {
        if (*it == GENERIC) { ++it; continue; }
        Colors needed = *it;

        // Check pool first
        if (pool.count(needed) > 0) {
            auto pit = pool.find(needed);
            pool.erase(pit);
            it = remaining.erase(it);
            continue;
        }

        // Try sources in priority order: adds_no_counter > single-color > multi-color
        auto try_source = [&](auto predicate) -> bool {
            for (auto &si : valid_sources) {
                if (tapped_entities.count(si.entity)) continue;
                if (si.color != needed) continue;
                if (!predicate(si)) continue;
                activate_source(si);
                tapped_entities.insert(si.entity);
                if (pool.count(needed) > 0) {
                    auto pit = pool.find(needed);
                    pool.erase(pit);
                }
                it = remaining.erase(it);
                return true;
            }
            return false;
        };
        if (try_source([](const SourceInfo &s) { return s.ability.adds_no_counter && !s.ability.sac_self; })) continue;
        if (try_source([](const SourceInfo &s) { return !s.is_multi_color && !s.ability.sac_self; })) continue;
        if (try_source([](const SourceInfo &s) { return !s.ability.sac_self; })) continue;
        if (try_source([](const SourceInfo &) { return true; })) continue;  // sac_self last resort
        return false;
    }

    // Pay generic costs with remaining sources — prefer adds_no_counter (Cavern)
    while (remaining.count(GENERIC) > 0) {
        if (pool.size() > 0) {
            pool.erase(pool.begin());
            auto git = remaining.find(GENERIC);
            remaining.erase(git);
            continue;
        }
        auto try_generic = [&](auto predicate) -> bool {
            for (auto &si : valid_sources) {
                if (tapped_entities.count(si.entity)) continue;
                if (!predicate(si)) continue;
                activate_source(si);
                tapped_entities.insert(si.entity);
                if (pool.size() > 0) {
                    pool.erase(pool.begin());
                    auto git = remaining.find(GENERIC);
                    remaining.erase(git);
                }
                return true;
            }
            return false;
        };
        if (try_generic([](const SourceInfo &s) { return s.ability.adds_no_counter && !s.ability.sac_self; })) continue;
        if (try_generic([](const SourceInfo &s) { return !s.ability.sac_self; })) continue;
        if (try_generic([](const SourceInfo &) { return true; })) continue;  // sac_self last resort
        return false;
    }

    return true;
}

bool can_pay_mana(Zone::Ownership controller, const ManaValue &cost,
                  Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve) {
    Entity player_entity = get_player_entity(controller);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    // Run the exact machine-mode payment algorithm in simulate mode (no side effects).
    // This is the single predicate behind both "is this castable" and "pay for it",
    // so a spell can never be offered as legal and then fail to pay (and vice versa).
    ManaValue remaining = cost;
    return auto_pay_mana(controller, remaining, paid_for, orderer, has_delve, /*commit=*/false);
}

bool prompt_mana_payment(Zone::Ownership controller, const ManaValue &cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve) {
    Entity player_entity = get_player_entity(controller);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // If pool already covers it, just spend
    if (can_afford_pool(player.mana, cost)) {
        spend_mana(controller, cost, paid_for);
        return true;
    }

    bool is_machine = InputLogger::instance().is_machine_mode();

    // Drain existing pool toward the cost first
    ManaValue remaining = pay_partial(controller, cost);

    // Machine mode: auto-select mana sources (no interactive prompts)
    if (is_machine) {
        return auto_pay_mana(controller, remaining, paid_for, orderer, has_delve);
    }

    // Payment loop: prompt player to tap sources or delve until cost is paid
    while (!remaining.empty()) {
        // Check if pool now covers remaining cost (from prior mana activations or delve)
        if (can_afford_pool(player.mana, remaining)) {
            spend_mana(controller, remaining, paid_for);
            return true;
        }

        // Build list of available payment options
        std::vector<LegalAction> pay_actions;

        // Mana abilities
        auto mana_actions = collect_mana_legal_actions(controller, orderer, paid_for);
        for (auto &la : mana_actions) {
            la.category = ActionCategory::PAYING_COSTS;
            pay_actions.push_back(la);
        }

        // Delve: exile instants/sorceries from graveyard to pay generic costs
        size_t delve_action_start = pay_actions.size();
        if (has_delve && remaining.count(GENERIC) > 0) {
            for (auto e : orderer->mEntities) {
                if (!is_delve_eligible(e, controller)) continue;
                auto &ecd = global_coordinator.GetComponent<CardData>(e);
                LegalAction la(PASS_PRIORITY, e, "Exile " + ecd.name + " (Delve)");
                la.category = ActionCategory::PAYING_COSTS;
                pay_actions.push_back(la);
            }
        }

        if (pay_actions.empty()) {
            return false;
        }

        // Add cancel option in non-machine mode
        if (!is_machine) {
            LegalAction cancel(PASS_PRIORITY, "Cancel casting");
            cancel.category = ActionCategory::PAYING_COSTS;
            pay_actions.push_back(cancel);
        }

        // Show remaining cost
        game_log("Pay mana costs (");
        bool first = true;
        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            if (!first) game_log(",");
            game_log("(%s)", mana_symbol_str(*it));
            first = false;
        }
        game_log(" remaining):\n");

        int choice = InputLogger::instance().get_input(pay_actions);

        // Check for cancel
        if (!is_machine && choice == static_cast<int>(pay_actions.size()) - 1) {
            return false;
        }

        auto &chosen = pay_actions[static_cast<size_t>(choice)];

        // Is this a delve exile action?
        if (static_cast<size_t>(choice) >= delve_action_start &&
            (!has_delve || chosen.type == PASS_PRIORITY)) {
            delve_exile_one(chosen.source_entity, controller, orderer, remaining);
        } else {
            // Mana ability activation — same core sequence as the auto-payer. The interactive
            // payer always commits to real state, so pool == the player's real mana pool.
            activate_mana_source(chosen.source_entity, chosen.ability, controller, orderer,
                                 player.mana, player, /*commit=*/true);
        }
    }

    return true;
}
