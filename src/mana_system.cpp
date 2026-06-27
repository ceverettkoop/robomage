#include "mana_system.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <tuple>

#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/static_ability.h"
#include "components/types.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "effects/effects.h"
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
                          bool commit = true, bool has_improvise = false);
static bool restricted_mana_matches(Entity source_entity, Entity paid_for);
static bool creature_restricted_mana_matches(Entity paid_for);
static bool colorless_eldrazi_restricted_mana_matches(Entity paid_for);
static bool mana_source_usable_for(const Ability &ab, Entity source_entity, Entity paid_for);
static bool is_delve_eligible(Entity e, Zone::Ownership controller);
static void delve_exile_one(Entity e, Zone::Ownership controller,
                            std::shared_ptr<Orderer> orderer, ManaValue &remaining);
static bool is_improvise_eligible(Entity e, Zone::Ownership controller, Entity paid_for);
static void improvise_tap_one(Entity e, Zone::Ownership controller, ManaValue &remaining);
static void activate_mana_source(Entity entity, const Ability &ab, Zone::Ownership controller,
                                 std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                 Player &player, bool commit);
static void fire_taps_for_mana_triggers(Entity tapped_source, Zone::Ownership controller,
                                        std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                        bool log);

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

// The activation mana cost of `ab` after applying its ReduceCost$ (CR 601.2f). General over the
// reduction source: a literal integer or a Count$/SVar expression resolved via the shared
// dynamic-amount evaluator (the same Count$Valid machinery used for dynamic NumDmg/amounts).
// Only the GENERIC portion is reduced (floored at 0); colored pips are untouched. Used by both
// the affordability gate and the payment so the two never diverge.
ManaValue effective_activation_mana_cost(const Ability &ab, Zone::Ownership controller,
                                         std::shared_ptr<Orderer> orderer) {
    if (ab.reduce_cost_expr.empty()) return ab.activation_mana_cost;
    size_t reduction = evaluate_dynamic_amount(ab.reduce_cost_expr, controller, orderer, ab.target);
    if (reduction == 0) return ab.activation_mana_cost;
    ManaValue cost = ab.activation_mana_cost;
    // Remove up to `reduction` generic ({1}) symbols; never below zero, never a colored pip.
    auto it = cost.find(GENERIC);
    while (reduction > 0 && it != cost.end()) {
        it = cost.erase(it);
        it = cost.find(GENERIC);
        --reduction;
    }
    return cost;
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

// Check whether a "spend only to cast a creature spell" mana source (Abundant
// Countryside) can pay for the given spell: true iff the spell is a creature.
static bool creature_restricted_mana_matches(Entity paid_for) {
    if (paid_for == 0) return false;
    if (!global_coordinator.entity_has_component<CardData>(paid_for)) return false;
    return is_creature_card(global_coordinator.GetComponent<CardData>(paid_for));
}

// Check whether a "spend only to cast colorless Eldrazi spells" mana source (Eldrazi
// Temple's {C}{C} ability) can pay for the given spell: true iff the spell is an Eldrazi
// (subtype) and colorless. A card is colorless when it has no colored mana symbols and no
// explicit color override other than COLORLESS (Devoid sets explicit_colors = {COLORLESS}).
// CR 106.7 mana spending restrictions.
static bool colorless_eldrazi_restricted_mana_matches(Entity paid_for) {
    if (paid_for == 0) return false;
    if (!global_coordinator.entity_has_component<CardData>(paid_for)) return false;
    auto &paid_cd = global_coordinator.GetComponent<CardData>(paid_for);
    bool is_eldrazi = false;
    for (auto &t : paid_cd.types)
        if (t.kind == SUBTYPE && t.name == "Eldrazi") { is_eldrazi = true; break; }
    if (!is_eldrazi) return false;
    // Colorless test (CR 105.2c) shared with the rest of the engine via game_queries.h, so an
    // Eldrazi Temple mana restriction and a color-targeting check can never disagree on whether
    // the same spell is colorless.
    return is_colorless_card(paid_cd);
}

// True if a mana source (its ability `ab`, on `source_entity`) may be spent to pay for
// `paid_for` under its mana spending restriction (CR 106.7). An unrestricted source is always
// usable; a restricted source is usable only when the spell being paid for matches. This is the
// single predicate behind all three payment paths — legal-action listing, the affordability
// gate, and the auto-payer — so they can never disagree on whether a source is spendable (the
// "offered legal then fails to pay" divergence can_pay_mana exists to prevent).
static bool mana_source_usable_for(const Ability &ab, Entity source_entity, Entity paid_for) {
    if (ab.restrict_to_chosen_type_creature && !restricted_mana_matches(source_entity, paid_for))
        return false;
    if (ab.restrict_to_creature && !creature_restricted_mana_matches(paid_for))
        return false;
    if (ab.restrict_to_colorless_eldrazi && !colorless_eldrazi_restricted_mana_matches(paid_for))
        return false;
    return true;
}

bool ability_is_mana(const Ability &ab) {
    if (ab.ability_type != Ability::ACTIVATED) return false;
    return ab.category == "AddMana" || ab.category == "ManaReflected";
}

// The producible color set of an AB$ ManaReflected ability (Mox Amber): the UNION of the
// effective colors of every battlefield permanent (controlled by `player`) that matches the
// ability's Valid$ filter (ReflectProperty$ Is — reflect the colors those permanents are).
// Colorless contributes no color (CR 105.2c), so a colorless-only board yields an empty set
// and the source produces nothing. Returned in WUBRG order for a stable choice menu.
static std::vector<Colors> reflected_color_set(const Ability &ab, Zone::Ownership player,
                                               std::shared_ptr<Orderer> orderer) {
    std::set<Colors> colors;
    MatchCtx ctx;
    ctx.controller = player;
    // ManaReflected's Valid$ lists its alternatives comma-separated (Forge Valid$ convention),
    // whereas permanent_matches_filter ORs on ';'. Normalize commas to ';' so each alternative
    // (legendary creature / legendary planeswalker) is matched independently.
    std::string filter = ab.reflected_mana_filter;
    for (char &ch : filter)
        if (ch == ',') ch = ';';
    for (auto entity : battlefield_permanents(orderer->mEntities, player)) {
        if (!permanent_matches_filter(entity, filter, ctx)) continue;
        for (Colors c : effective_colors(entity)) colors.insert(c);
    }
    std::vector<Colors> ordered;
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
        if (colors.count(c)) ordered.push_back(c);
    return ordered;
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
            if (!ability_is_mana(ab)) continue;
            // InstantSpeed$ mana abilities (e.g. LED) may only be activated at priority, not
            // mid-cost-payment. Callers listing actions for a player who holds priority pass
            // include_instant_speed; the affordability/payment callers leave it false.
            if (ab.instant_speed && !include_instant_speed) continue;
            // Activation$ gate (CR 602.5): e.g. Mox Opal's Metalcraft — illegal unless the
            // controller meets the named condition (here, controls 3+ artifacts).
            if (!activation_condition_met(ab, player, orderer->mEntities, entity)) continue;
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
            // AB$ ManaReflected (Mox Amber): producible colors are the union of the colors of
            // the Valid$-matching permanents you control, computed live. Expand into per-color
            // choices like mana_choices. An empty set (no/colorless legendaries) makes the
            // ability produce nothing, so it is not offered as a usable mana source.
            if (!ab.reflected_mana_filter.empty()) {
                for (Colors choice_color : reflected_color_set(ab, player, orderer)) {
                    Ability choice_ab = ab;
                    choice_ab.color = choice_color;
                    sources.push_back({entity, choice_ab});
                }
            } else if (!ab.mana_choices.empty()) {
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
        // Filter restricted mana (Cavern of Souls / Abundant Countryside / Eldrazi Temple):
        // hide a source whose spending restriction doesn't match the spell being paid for.
        if (!mana_source_usable_for(ab, entity, paid_for)) continue;
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

    // Filter restricted mana sources (Cavern of Souls etc.): drop any whose spending
    // restriction doesn't match the spell being paid for.
    sources.erase(std::remove_if(sources.begin(), sources.end(),
        [&](const std::pair<Entity, Ability> &s) {
            return !mana_source_usable_for(s.second, s.first, paid_for);
        }), sources.end());

    // First pass: add free sources (no activation mana cost)
    // Multi-color sources appear as multiple entries for the same entity
    // (one per color choice); count those as flexible mana.
    for (auto &[entity, ab] : sources) {
        if (entity == exclude_entity) continue;
        if (counted_entities.count(entity)) continue;
        if (!ab.activation_mana_cost.empty()) continue;
        size_t amount = eval_mana_amount(ab, player_owner, orderer);
        // A permanent can only be tapped once, so when it offers several same-color mana
        // abilities of differing yield (Eldrazi Temple: {C} vs {C}{C}), the player picks the
        // largest applicable one. Count the maximum amount among this entity's other free,
        // same-color abilities so affordability reflects the best single activation.
        for (auto &[e2, ab2] : sources) {
            if (e2 != entity || &ab2 == &ab) continue;
            if (!ab2.activation_mana_cost.empty()) continue;
            if (ab2.color != ab.color) continue;
            amount = std::max(amount, eval_mana_amount(ab2, player_owner, orderer));
        }
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
        // A creature tapping for mana fires any TapsForMana bonus (Badgermole Cub's extra {G}).
        // Mirror the real payment path (activate_mana_source) so nested mana-source affordability
        // counts the same mana — the helper is internally gated to creature sources, so a
        // non-creature source adds nothing.
        fire_taps_for_mana_triggers(entity, player_owner, orderer, hypothetical, false);
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
        // TapsForMana bonus for this (cost-bearing) source's tap, added after its own activation
        // cost is settled so the bonus is pure additional mana (see the first-pass note).
        fire_taps_for_mana_triggers(entity, player_owner, orderer, hypothetical, false);
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

// True if `e` is an untapped artifact `controller` controls on the battlefield — i.e. an
// artifact Improvise can tap to pay one generic pip (CR 702.126b/c). The spell being paid
// for (`paid_for`) is excluded: it is on the stack, not the battlefield, but guard anyway so
// the artifact creature can never tap itself to help pay its own Improvise cost.
static bool is_improvise_eligible(Entity e, Zone::Ownership controller, Entity paid_for) {
    if (e == paid_for) return false;
    if (!is_battlefield_permanent(e, controller)) return false;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    if (perm.is_tapped) return false;
    return permanent_has_type(perm, "Artifact");
}

// Pay one generic pip via Improvise: tap `e` and drop one GENERIC from `remaining`. Mirrors
// delve_exile_one for the tap-an-artifact cost. Tapping an artifact this way is not a mana
// ability and produces no mana — it directly satisfies a {1} (CR 702.126b).
static void improvise_tap_one(Entity e, Zone::Ownership controller, ManaValue &remaining) {
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    perm.is_tapped = true;
    auto git = remaining.find(GENERIC);
    if (git != remaining.end()) remaining.erase(git);
    game_log("%s taps %s for Improvise.\n", player_name(controller).c_str(), perm.name.c_str());
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
    // Mana-additional "whenever you tap a <permanent> for mana" triggers (Badgermole Cub's
    // TapsForMana, CR 605.1a): resolve immediately as part of the tap, adding their extra mana
    // to the working pool. Fired in BOTH commit and simulate modes so affordability/legality
    // (can_pay_mana) and the real payment agree on the available mana; the narrative line is
    // emitted only on the real activation (commit).
    fire_taps_for_mana_triggers(entity, controller, orderer, pool, commit);
    if (commit && ab.adds_no_counter) cur_game.pending_cant_be_countered = true;
    if (commit)
        game_log("%s activated %s for %zu(%s)\n", player_name(controller).c_str(),
                 perm.name.c_str(), amount, mana_symbol_str(ab.color));
    // A mana ability may carry a SubAbility$ rider that is part of the mana ability and
    // resolves off-stack with it — e.g. Ancient Tomb's "deals 2 damage to you" (CR 605.1a,
    // 606.3). Only fire it when committing the activation (not during legality simulation).
    if (commit) {
        for (auto sub_ab : ab.subabilities) {
            sub_ab.source = entity;
            sub_ab.controller = controller;
            sub_ab.resolve(orderer);
        }
    }
    if (commit) increment_activation_count(perm, ab);
}

// True if `e` is (currently) a creature on the battlefield — used to gate ValidCard$ Creature on
// a TapsForMana trigger. An earthbended land that became a creature counts (it has a Creature
// component while animated).
static bool tapped_source_is_creature(Entity e) {
    return global_coordinator.entity_has_component<Creature>(e) && on_battlefield(e);
}

// Resolve mana-additional "whenever you tap a <permanent> for mana" triggers (Mode$ TapsForMana
// | Static$ True) the instant a permanent taps for mana (CR 605.1a — these never use the stack).
// For each battlefield permanent the tapping player controls whose TapsForMana trigger matches
// the tapped source (ValidCard$ Creature here) and whose Activator$ You is satisfied, add its
// Execute$ AddMana (color/amount) directly to the working pool. General over the produced color.
static void fire_taps_for_mana_triggers(Entity tapped_source, Zone::Ownership controller,
                                        std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                        bool log) {
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, controller)) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(entity);
        for (const auto &ab : cd.abilities) {
            if (ab.ability_type != Ability::TRIGGERED) continue;
            if (!ab.trigger_taps_for_mana_static) continue;
            if (ab.trigger_on != Events::TAPPED_FOR_MANA) continue;
            // Activator$ You — only the source controller tapping their own permanent (we already
            // restricted the scan and the tap to `controller`, so this always holds here).
            // ValidCard$ Creature — the tapped source must be a creature (an animated
            // land-creature counts while it has a Creature component).
            if (ab.trigger_valid_card_is_creature && !tapped_source_is_creature(tapped_source))
                continue;
            // The Execute$ SVar's AddMana (Produced$/Amount$) is the resolved effect: the parser
            // folds Execute$ into the trigger, so ab.category == "AddMana" with the produced
            // color and amount. Add that mana directly to the pool.
            Colors produced = ab.color;
            size_t add_amt = ab.amount > 0 ? ab.amount : 1;
            for (size_t i = 0; i < add_amt; i++) pool.insert(produced);
            if (log)
                game_log("%s adds an additional %zu(%s).\n",
                         cd.name.c_str(), add_amt, mana_symbol_str(produced));
        }
    }
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
                          bool commit, bool has_improvise) {
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
        // Restricted mana check (Cavern of Souls / Abundant Countryside / Eldrazi Temple):
        // same gate as collect_mana_legal_actions so a listed source is always spendable here.
        if (!mana_source_usable_for(ab, entity, paid_for)) continue;
        if (!ab.activation_mana_cost.empty()) {
            if (!can_afford_with_sources(controller, ab.activation_mana_cost, orderer, ab.tap_cost ? entity : 0))
                continue;
        }
        bool is_multi = false;
        for (auto &[e2, ab2] : sources)
            if (e2 == entity && ab2.color != ab.color) { is_multi = true; break; }
        // A permanent taps only once. When it has several valid same-color mana abilities of
        // differing yield (Eldrazi Temple: {C} vs the restricted {C}{C}), keep only the
        // highest-yield one so the greedy payer realizes the affordability the gating check
        // (can_afford_with_sources, which counts the max) promised, instead of burning the
        // single tap on the smaller ability.
        bool replaced = false;
        for (auto &existing : valid_sources) {
            if (existing.entity != entity || existing.color != ab.color) continue;
            if (eval_mana_amount(ab, controller, orderer) >
                eval_mana_amount(existing.ability, controller, orderer)) {
                existing.ability = ab;
            }
            replaced = true;
            break;
        }
        if (replaced) continue;
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
                // Only satisfy the pip with mana the source ACTUALLY produced. A source
                // that produced no mana of `needed` (e.g. a 0-mana ManaReflected source
                // with no qualifying permanent) erases nothing — the pip stays unpaid and
                // we keep scanning. This mirrors the interactive payer, where the pip is
                // only spent once can_afford_pool sees the color in the real pool.
                if (pool.count(needed) > 0) {
                    auto pit = pool.find(needed);
                    pool.erase(pit);
                    it = remaining.erase(it);
                    return true;
                }
                // Source produced nothing usable; it is now tapped (tracked above) so it
                // won't be retried, but the pip is still owed — try the next candidate.
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
                // Only pay a generic pip with mana the source actually produced; a 0-mana
                // source erases nothing (consistent with the colored loop above).
                if (pool.size() > 0) {
                    pool.erase(pool.begin());
                    auto git = remaining.find(GENERIC);
                    remaining.erase(git);
                    return true;
                }
            }
            return false;
        };
        if (try_generic([](const SourceInfo &s) { return s.ability.adds_no_counter && !s.ability.sac_self; })) continue;
        if (try_generic([](const SourceInfo &s) { return !s.ability.sac_self; })) continue;
        if (try_generic([](const SourceInfo &) { return true; })) continue;  // sac_self last resort
        // Improvise: a {1} can also be paid by tapping an untapped artifact (CR 702.126).
        // Tried after mana sources so colored pips (paid above) keep their producers; an
        // artifact already tapped for mana is excluded via tapped_entities.
        if (has_improvise) {
            bool tapped_one = false;
            for (auto e : orderer->mEntities) {
                if (tapped_entities.count(e)) continue;
                if (!is_improvise_eligible(e, controller, paid_for)) continue;
                if (commit) {
                    improvise_tap_one(e, controller, remaining);
                } else {
                    auto git = remaining.find(GENERIC);
                    if (git != remaining.end()) remaining.erase(git);
                }
                tapped_entities.insert(e);
                tapped_one = true;
                break;
            }
            if (tapped_one) continue;
        }
        return false;
    }

    return true;
}

bool can_pay_mana(Zone::Ownership controller, const ManaValue &cost,
                  Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve,
                  bool has_improvise) {
    Entity player_entity = get_player_entity(controller);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    // Run the exact machine-mode payment algorithm in simulate mode (no side effects).
    // This is the single predicate behind both "is this castable" and "pay for it",
    // so a spell can never be offered as legal and then fail to pay (and vice versa).
    ManaValue remaining = cost;
    return auto_pay_mana(controller, remaining, paid_for, orderer, has_delve, /*commit=*/false,
                         has_improvise);
}

bool prompt_mana_payment(Zone::Ownership controller, const ManaValue &cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve, bool has_improvise) {
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
        return auto_pay_mana(controller, remaining, paid_for, orderer, has_delve, /*commit=*/true,
                             has_improvise);
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

        // Improvise: tap untapped artifacts to pay generic costs (CR 702.126)
        size_t improvise_action_start = pay_actions.size();
        if (has_improvise && remaining.count(GENERIC) > 0) {
            for (auto e : orderer->mEntities) {
                if (!is_improvise_eligible(e, controller, paid_for)) continue;
                auto &perm = global_coordinator.GetComponent<Permanent>(e);
                LegalAction la(PASS_PRIORITY, e, "Tap " + perm.name + " (Improvise)");
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

        size_t uchoice = static_cast<size_t>(choice);
        if (has_improvise && uchoice >= improvise_action_start &&
            uchoice < pay_actions.size() - (is_machine ? 0 : 1)) {
            // Improvise: tap the chosen artifact to pay one generic.
            improvise_tap_one(chosen.source_entity, controller, remaining);
        } else if (has_delve && uchoice >= delve_action_start &&
                   uchoice < improvise_action_start) {
            // Delve exile action.
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
