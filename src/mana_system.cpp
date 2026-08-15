#include "mana_system.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <map>
#include <set>
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
#include "systems/replacement_effects.h"
#include "systems/rules_modifying.h"
#include "systems/state_manager.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static size_t eval_mana_amount(const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);
// Converge spent-color sink (CR 702.90): while a cast's mana payment is in flight this points at
// the accumulator that records every color of mana actually removed from the REAL pool. Armed only
// inside prompt_mana_payment (via a RAII guard); dry-run affordability checks route through
// pay_from_pool on POOL COPIES with a null `spent`, so they never pollute it.
static ManaValue *s_mana_spent_sink = nullptr;

// `spent` (when non-null) collects each color erased from `pool` — so a REAL-pool consumer can
// report the colors it spent. Dry-run callers (can_afford_pool, activation-cost trials) pass
// nullptr and are unaffected.
static ManaValue pay_from_pool(ManaValue &pool, const ManaValue &cost, ManaValue *spent = nullptr);
static bool auto_pay_mana(Zone::Ownership controller, ManaValue &remaining,
                          Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve,
                          bool commit = true, bool has_improvise = false,
                          Entity exclude_entity = 0);
static bool auto_pay_mana_attempt(Zone::Ownership controller, ManaValue &remaining,
                                  Entity paid_for, std::shared_ptr<Orderer> orderer,
                                  bool has_delve, bool commit, bool has_improvise,
                                  Entity exclude_entity, bool max_yield_only);
static bool restricted_mana_matches(Entity source_entity, Entity paid_for);
static bool creature_restricted_mana_matches(Entity paid_for);
static bool colorless_eldrazi_restricted_mana_matches(Entity paid_for);
static bool mana_source_usable_for(const Ability &ab, Entity source_entity, Entity paid_for);
static bool is_improvise_eligible(Entity e, Zone::Ownership controller, Entity paid_for);
static void improvise_tap_one(Entity e, Zone::Ownership controller, ManaValue &remaining);
static void fire_taps_for_mana_triggers(Entity tapped_source, Zone::Ownership controller,
                                        std::shared_ptr<Orderer> orderer, ManaValue &pool,
                                        bool log);
static bool mana_ability_is_painful(const Ability &ab);
static bool has_nonmana_activated_ability(Entity entity);
static std::array<int, 6> hand_color_demand(Zone::Ownership controller, Entity paid_for,
                                            std::shared_ptr<Orderer> orderer);

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
    pay_from_pool(player.mana, cost, s_mana_spent_sink);
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
    return pay_from_pool(player.mana, cost, s_mana_spent_sink);
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
    // Dynamic mana amount of the form "Count$Valid <filter>" — count the controller's
    // battlefield permanents matching the Forge filter (Gaea's Cradle: Creature.YouCtrl;
    // Urza's Workshop: Urza's.Land+YouCtrl). Routed through the shared permanent filter so
    // the full qualifier grammar (subtype head, type/ownership qualifiers) is honored.
    const std::string kPrefix = "Count$Valid ";
    if (!ab.dynamic_amount_expr.empty() && ab.dynamic_amount_expr.rfind(kPrefix, 0) == 0) {
        std::string filter = ab.dynamic_amount_expr.substr(kPrefix.size());
        MatchCtx ctx;
        ctx.controller = controller;
        ctx.source = ab.source;
        size_t count = 0;
        for (auto e : orderer->mEntities)
            if (is_battlefield_permanent(e) && permanent_matches_filter(e, filter, ctx)) count++;
        return count;
    }
    // Any other dynamic mana amount (e.g. Cabal Ritual's Count$Threshold.5.3) routes through the
    // shared runtime-amount evaluator, so mana production scales by the same Count$/Targeted$
    // grammar used for dynamic damage/draw/token counts rather than re-implementing each form here.
    if (!ab.dynamic_amount_expr.empty())
        return evaluate_dynamic_amount(ab.dynamic_amount_expr, controller, orderer, ab.target);
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
    // CR 605.1a / 606.3: a loyalty ability (a planeswalker's activated ability with a loyalty
    // cost) is NEVER a mana ability, even when it produces mana — it uses the stack and is a
    // sorcery-speed, once-per-turn activation, not a repeatable off-stack mana source. Ugin, Eye
    // of the Storms [0]: "Add {C}{C}{C}". Excluding it here routes it through the normal loyalty-
    // gated activated-ability path (state_manager_actions / process_activate_ability), where it
    // resolves off the stack via the AddMana effect handler.
    if (ab.is_loyalty_ability) return false;
    return ab.category == "AddMana" || ab.category == "ManaReflected";
}

// The producible color set of an AB$ ManaReflected ability (Mox Amber): the UNION of the
// effective colors of every battlefield permanent (controlled by `player`) that matches the
// ability's Valid$ filter (ReflectProperty$ Is — reflect the colors those permanents are).
// Colorless contributes no color (CR 105.2c), so a colorless-only board yields an empty set
// and the source produces nothing. Returned in WUBRG order for a stable choice menu.
static std::vector<Colors> reflected_color_set(const Ability &ab, Zone::Ownership player,
                                               const std::set<Entity> &entities) {
    std::set<Colors> colors;
    MatchCtx ctx;
    ctx.controller = player;
    // ManaReflected's Valid$ lists its alternatives comma-separated (Forge Valid$ convention).
    // permanent_matches_any matches each alternative (legendary creature / legendary
    // planeswalker) independently, so the comma-OR is handled in one shared place.
    for (auto entity : battlefield_permanents(entities, player)) {
        if (!permanent_matches_any(entity, ab.reflected_mana_filter, ctx)) continue;
        for (Colors c : effective_colors(entity)) colors.insert(c);
    }
    std::vector<Colors> ordered;
    for (Colors c : {WHITE, BLUE, BLACK, RED, GREEN})
        if (colors.count(c)) ordered.push_back(c);
    return ordered;
}

// Can `ab` (a mana ability of `permanent`, entity `e`) be activated RIGHT NOW, ignoring its
// own activation mana cost? The physical gate — instant-speed window, Activation$ condition,
// tap state, activation limit, summoning sickness without haste. Shared by
// collect_available_mana_sources and mana_potential so the observation's "what could I
// produce" summary can never disagree with the menu about which sources are available.
static bool mana_ability_available_now(Entity e, const Permanent &permanent, const Ability &ab,
                                       Zone::Ownership player, const std::set<Entity> &entities,
                                       bool include_instant_speed) {
    // InstantSpeed$ mana abilities (e.g. LED) may only be activated at priority, not
    // mid-cost-payment. Callers listing actions for a player who holds priority pass
    // include_instant_speed; the affordability/payment callers leave it false.
    if (ab.instant_speed && !include_instant_speed) return false;
    // Activation$ gate (CR 602.5): e.g. Mox Opal's Metalcraft — illegal unless the
    // controller meets the named condition (here, controls 3+ artifacts).
    if (!activation_condition_met(ab, player, entities, e)) return false;
    if (ab.tap_cost && permanent.is_tapped) return false;
    if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) return false;
    // Summoning sickness check for creatures with tap cost
    if (ab.tap_cost && permanent.has_summoning_sickness &&
        global_coordinator.entity_has_component<Creature>(e)) {
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        bool has_haste = false;
        for (const auto &kw : cr.keywords)
            if (kw == "Haste") { has_haste = true; break; }
        if (!has_haste) return false;
    }
    return true;
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
            if (!mana_ability_available_now(entity, permanent, ab, player, orderer->mEntities,
                                            include_instant_speed))
                continue;
            // AB$ ManaReflected (Mox Amber): producible colors are the union of the colors of
            // the Valid$-matching permanents you control, computed live. Expand into per-color
            // choices like mana_choices. An empty set (no/colorless legendaries) makes the
            // ability produce nothing, so it is not offered as a usable mana source.
            if (!ab.reflected_mana_filter.empty()) {
                for (Colors choice_color : reflected_color_set(ab, player, orderer->mEntities)) {
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

// See mana_system.h for the contract. One pass over the player's live battlefield
// permanents: per permanent, the union of the colors its AVAILABLE mana abilities could
// produce (each color counted once for the permanent, per the "one unit per source per
// color" rule), whether it is a mana source at all, and whether it is a land. Deliberately
// mana-COST-free (no payer), so it is safe to call on the ML serialization path where no
// Orderer is in scope.
ManaPotential mana_potential(Zone::Ownership player, const std::set<Entity> &entities) {
    ManaPotential out;
    for (auto entity : battlefield_permanents(entities, player)) {
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (type_set_has(permanent.types, "Land")) out.lands++;
        if (rules_mod::mana_activation_prohibited(entity)) continue;

        bool colors[6] = {false, false, false, false, false, false};
        bool is_source = false;
        auto note_color = [&](Colors c) {
            int idx = static_cast<int>(c);
            if (idx < 0 || idx >= 6) return;  // NO_COLOR/GENERIC: produces nothing nameable
            colors[idx] = true;
            is_source = true;                 // a source only counts once it can make SOMETHING
        };
        for (const auto &ab : permanent.abilities) {
            if (!ability_is_mana(ab)) continue;
            // Instant-speed mana abilities (LED) ARE potential mana for their controller at
            // priority, which is the horizon this summary describes.
            if (!mana_ability_available_now(entity, permanent, ab, player, entities,
                                            /*include_instant_speed=*/true))
                continue;
            // A ManaReflected source whose color set is empty (Mox Amber with no legendary,
            // or only colorless ones) produces nothing, so note_color leaves it uncounted —
            // matching collect_available_mana_sources, which offers no action for it.
            if (!ab.reflected_mana_filter.empty()) {
                for (Colors c : reflected_color_set(ab, player, entities)) note_color(c);
            } else if (!ab.mana_choices.empty()) {
                for (Colors c : ab.mana_choices) note_color(c);
            } else {
                note_color(ab.color);
            }
        }
        if (!is_source) continue;
        out.sources++;
        for (int i = 0; i < 6; i++)
            if (colors[i]) out.by_color[i]++;
    }
    return out;
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

    // Try to pay colored costs from the hypothetical pool. Under ManaConvert (Mycosynth Lattice)
    // any mana pays any colored pip, so every pip is treated as generic against the total pool.
    bool any_color = any_mana_as_any_color_active();
    auto remaining = hypothetical;
    size_t flexible_used = 0;
    if (!any_color) {
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
    }

    size_t generic_needed = any_color ? cost.size() : cost.count(GENERIC);
    size_t available_for_generic = remaining.size() + (flexible_count - flexible_used);
    return available_for_generic >= generic_needed;
}

size_t max_available_mana(Zone::Ownership player_owner, const ManaValue &base_cost,
                          std::shared_ptr<Orderer> orderer, Entity exclude_entity) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return 0;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Count total mana available: pool + all sources
    size_t total = player.mana.size();
    std::set<Entity> counted;
    auto sources = collect_available_mana_sources(player_owner, orderer);
    for (auto &[entity, ab] : sources) {
        if (entity == exclude_entity && ab.tap_cost) continue;  // its tap is the ability's own cost
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
static ManaValue pay_from_pool(ManaValue &pool, const ManaValue &cost, ManaValue *spent) {
    ManaValue remaining;
    auto erase_recording = [&](std::multiset<Colors>::iterator it) {
        if (spent) spent->insert(*it);
        pool.erase(it);
    };
    // Mycosynth Lattice (ManaConvert AnyType->AnyColor, CR 609.4 / 106.6): while active, a colored
    // pip may be paid with mana of ANY type. Pay exact-color matches first (so on-color mana is
    // preferred and never wasted), then satisfy any still-unpaid colored pip from any remaining
    // mana — exactly the generic-pip rule applied to colored pips.
    bool any_color = any_mana_as_any_color_active();
    std::vector<Colors> unpaid_colored;
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = pool.find(color);
        if (it != pool.end()) erase_recording(it);
        else unpaid_colored.push_back(color);
    }
    for (Colors color : unpaid_colored) {
        if (any_color && !pool.empty()) erase_recording(pool.begin());
        else remaining.insert(color);
    }
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!pool.empty()) erase_recording(pool.begin());
        else remaining.insert(GENERIC);
    }
    return remaining;
}

// True if `e` is a card in `controller`'s graveyard — i.e. a card Delve can exile to
// pay a generic pip. CR 702.66a places no type restriction ("you may exile a card from
// your graveyard rather than pay that mana"); riders that care about the exiled cards'
// types (Murktide Regent's etbCounter) filter cur_game.delve_exiled themselves. Single
// source for "what can Delve eat", consumed by both the automatic payer and the cast
// flow's DELVE steps (run_cast_flow, action_processor.cpp). Exported via mana_system.h.
bool is_delve_eligible(Entity e, Zone::Ownership controller) {
    if (!global_coordinator.entity_has_component<Zone>(e)) return false;
    auto &ez = global_coordinator.GetComponent<Zone>(e);
    if (ez.location != Zone::GRAVEYARD || ez.owner != controller) return false;
    return global_coordinator.entity_has_component<CardData>(e);
}

// Pay one generic pip via Delve: exile `e`, record it for the etbCounter count, and
// drop one GENERIC from `remaining`. Single source for the delve-exile action used by
// both payment paths. Exported via mana_system.h.
void delve_exile_one(Entity e, Zone::Ownership controller,
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

// The production half of a mana-ability activation — everything AFTER costs are paid.
// See mana_system.h for the contract. Shared by activate_mana_source (payer paths and
// the pay-unless loop) and the priority-menu activation path, whose costs are paid by
// the generic activated-ability code before it produces.
void produce_mana_from_ability(Entity source, const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer, ManaValue &pool,
                               bool commit, ManaLogStyle log_style) {
    auto &perm = global_coordinator.GetComponent<Permanent>(source);
    size_t amount = eval_mana_amount(ab, controller, orderer);
    // ProduceMana replacement effects (CR 614.1, Damping Sphere): a land tapped for 2+ mana
    // produces that much {C} instead. Consult active replacements before the produced mana enters
    // the pool, so a colored producer can be rewritten to colorless. Runs in both commit and
    // simulate mode so affordability (can_pay_mana) and the real payment agree on the colors.
    Colors produced_color = ab.color;
    if (amount >= 2) {
        ReplacementEvent ev;
        ev.type = ReplacementEvent::PRODUCE_MANA;
        ev.entity = source;
        ev.affected_player = controller;
        ev.produced_color = ab.color;
        ev.produced_amount = amount;
        replacement::dispatch(ev);
        produced_color = ev.produced_color;
    }
    for (size_t i = 0; i < amount; i++) pool.insert(produced_color);
    // Mana-additional "whenever you tap a <permanent> for mana" triggers (Badgermole Cub's
    // TapsForMana, CR 605.1a): resolve immediately as part of the tap, adding their extra mana
    // to the working pool. Fired in BOTH commit and simulate modes so affordability/legality
    // (can_pay_mana) and the real payment agree on the available mana; the narrative line is
    // emitted only on the real activation (commit).
    fire_taps_for_mana_triggers(source, controller, orderer, pool, commit);
    if (commit && ab.adds_no_counter) cur_game.pending_cant_be_countered = true;
    if (commit) {
        switch (log_style) {
            case ManaLogStyle::ACTIVATED:
                game_log("%s activated %s for %zu(%s)\n", player_name(controller).c_str(),
                         perm.name.c_str(), amount, mana_symbol_str(produced_color));
                break;
            case ManaLogStyle::TAPPED_AMOUNT:
                game_log("%s tapped %s for %zu(%s)\n", player_name(controller).c_str(),
                         perm.name.c_str(), amount, mana_symbol_str(produced_color));
                break;
            case ManaLogStyle::TAPPED_SYMBOL:
                game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(),
                         perm.name.c_str(), mana_symbol(produced_color).c_str());
                break;
        }
    }
    // A mana ability may carry a SubAbility$ rider that is part of the mana ability and
    // resolves off-stack with it — e.g. Ancient Tomb's "deals 2 damage to you" (CR 605.1a,
    // 606.3). Only fire it when committing the activation (not during legality simulation).
    if (commit) {
        for (auto sub_ab : ab.subabilities) {
            sub_ab.source = source;
            sub_ab.controller = controller;
            sub_ab.resolve(orderer);
        }
    }
    if (commit) increment_activation_count(perm, ab);
}

// Activate one mana source: pay its costs, then produce (see mana_system.h). The cost half
// lives here; the production half is produce_mana_from_ability. A refusal (unpayable
// activation mana cost) cancels cleanly with no side effects; the auto-payer pre-covers the
// cost by tapping other sources into the pool (cover_activation_cost), and the interactive
// payer / pay-unless loop rely on this check to refuse.
bool activate_mana_source(Entity source, const Ability &ab, Zone::Ownership controller,
                          std::shared_ptr<Orderer> orderer, ManaValue &pool,
                          Player &player, bool commit, ManaLogStyle log_style) {
    auto &perm = global_coordinator.GetComponent<Permanent>(source);
    if (!ab.activation_mana_cost.empty()) {
        // pay_from_pool returns the unpayable remainder and drains the pool even on a
        // partial payment, so snapshot the pool and restore it when the cost bounces.
        ManaValue pool_before = pool;
        ManaValue unpaid = pay_from_pool(pool, ab.activation_mana_cost);
        if (!unpaid.empty()) {
            pool = pool_before;
            return false;
        }
    }
    if (commit && ab.tap_cost) perm.is_tapped = true;
    if (commit && ab.sac_self) {
        game_log("%s sacrifices %s\n", player_name(controller).c_str(), perm.name.c_str());
        orderer->add_to_zone(false, source, Zone::GRAVEYARD);
    }
    if (commit && ab.life_cost > 0) {
        player.life_total -= ab.life_cost;
        player.life_lost_this_turn += ab.life_cost;  // CR 119.4: paying life is losing life
        game_log("%s pays %d life\n", player_name(controller).c_str(), ab.life_cost);
    }
    produce_mana_from_ability(source, ab, controller, orderer, pool, commit, log_style);
    return true;
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

// A mana ability is "painful" when activating it costs its controller life beyond the tap:
// either an explicit PayLife activation cost (Horizon Canopy's "{T}, Pay 1 life") or a
// self-damage / life-loss rider that resolves with the mana ability (Ancient Tomb's "deals
// 2 damage to you", a DealDamage sub-ability with Defined$ You). Derived from the ability's
// structure, not a card-name list, so any pain source scripted the same way is covered.
static bool mana_ability_is_painful(const Ability &ab) {
    if (ab.life_cost > 0) return true;
    for (const auto &sub : ab.subabilities)
        if ((sub.category == "DealDamage" || sub.category == "LoseLife") && sub.defined_you)
            return true;
    return false;
}

// True if the battlefield permanent has a non-mana ACTIVATED ability (Karakas's bounce,
// Wasteland's destroy, a man-land's animation): tapping it for mana costs its controller
// the option to use that ability, so the auto-payer prefers plain sources when otherwise
// equal. Loyalty abilities count too (ability_is_mana excludes them), which only matters
// if a planeswalker ever taps for mana — also a "save it for its other ability" case.
static bool has_nonmana_activated_ability(Entity entity) {
    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
    for (const auto &ab : perm.abilities)
        if (ab.ability_type == Ability::ACTIVATED && !ability_is_mana(ab)) return true;
    return false;
}

// Colored-pip demand of the controller's remaining hand: how many pips of each color
// (indexed by the Colors enum, WHITE..COLORLESS) the OTHER cards in hand ask for. Used to
// steer the auto-payer away from tapping a source whose color those cards still need.
// Counts raw CardData::mana_cost pips plus each hybrid pip's colored options; the spell
// being paid for (`paid_for`) is excluded. Deliberately simple — no affordability filter
// on the counted cards and no weighting — so simulate (can_pay_mana) and the real payment
// derive the identical ordering from public, side-effect-free state.
static std::array<int, 6> hand_color_demand(Zone::Ownership controller, Entity paid_for,
                                            std::shared_ptr<Orderer> orderer) {
    std::array<int, 6> demand{};
    for (Entity e : orderer->get_hand(controller)) {
        if (e == paid_for) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        for (Colors c : cd.mana_cost)
            if (c >= WHITE && c <= COLORLESS) demand[static_cast<size_t>(c)]++;
        for (const auto &pip : cd.hybrid_mana)
            for (Colors c : pip.colors)
                if (c >= WHITE && c <= COLORLESS) demand[static_cast<size_t>(c)]++;
    }
    return demand;
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
// The greedy payer runs pain-averse by default: painless sources engage before painful
// ones (Ancient Tomb). That preference can strand a payment when one permanent carries
// BOTH a painless low-yield and a painful high-yield mana ability (a Tomb made a Forest
// by Yavimaya: painless {G} vs painful {C}{C}) — the painless tier burns the permanent's
// single tap on the small ability and the total then falls short. So auto_pay_mana first
// tries the normal strategy, and only if that fails retries with `max_yield_only`, which
// keeps just the highest-yield ability per permanent so the tap realizes the permanent's
// full potential. The retry strictly widens what is payable; a payment the pain-averse
// pass can complete is still paid painlessly. Simulate picks the strategy; a commit run
// replays the winning strategy so legality and payment stay in lockstep.
static bool auto_pay_mana(Zone::Ownership controller, ManaValue &remaining,
                          Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve,
                          bool commit, bool has_improvise, Entity exclude_entity) {
    ManaValue trial = remaining;
    if (auto_pay_mana_attempt(controller, trial, paid_for, orderer, has_delve,
                              /*commit=*/false, has_improvise, exclude_entity,
                              /*max_yield_only=*/false)) {
        if (!commit) {
            remaining = trial;
            return true;
        }
        return auto_pay_mana_attempt(controller, remaining, paid_for, orderer, has_delve,
                                     /*commit=*/true, has_improvise, exclude_entity,
                                     /*max_yield_only=*/false);
    }
    trial = remaining;
    if (auto_pay_mana_attempt(controller, trial, paid_for, orderer, has_delve,
                              /*commit=*/false, has_improvise, exclude_entity,
                              /*max_yield_only=*/true)) {
        if (!commit) {
            remaining = trial;
            return true;
        }
        return auto_pay_mana_attempt(controller, remaining, paid_for, orderer, has_delve,
                                     /*commit=*/true, has_improvise, exclude_entity,
                                     /*max_yield_only=*/true);
    }
    return false;
}

static bool auto_pay_mana_attempt(Zone::Ownership controller, ManaValue &remaining,
                                  Entity paid_for, std::shared_ptr<Orderer> orderer,
                                  bool has_delve, bool commit, bool has_improvise,
                                  Entity exclude_entity, bool max_yield_only) {
    Entity player_entity = get_player_entity(controller);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // In commit mode the working pool IS the real mana pool; in simulate mode it is a
    // throwaway copy so the algorithm can drain/refill it without touching real state.
    ManaValue pool_copy = player.mana;
    ManaValue &pool = commit ? player.mana : pool_copy;

    // Mycosynth Lattice (ManaConvert AnyType->AnyColor, CR 609.4 / 106.6): while active, any mana
    // can pay any colored pip. We keep the colored pips colored (so Delve/Improvise, which reduce
    // only generic costs, never touch them) and instead relax the colored source-matching below.
    bool any_color = any_mana_as_any_color_active();

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


    // In commit mode `pool` IS the real mana pool, so record the colors it spends into the Converge
    // sink; in simulate mode the sink stays null (no side effects, no spurious color counts).
    ManaValue *spent_sink = commit ? s_mana_spent_sink : nullptr;

    // Check if pool covers remaining after delve
    if (can_afford_pool(pool, remaining)) {
        pay_from_pool(pool, remaining, spent_sink);
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
        // The caller's ability taps this permanent as part of its own cost, so the
        // permanent's TAP-requiring mana abilities are already spoken for. A mana ability
        // on the same permanent that needs no tap is still usable, so scope the exclusion
        // to the tap (see can_pay_mana's exclude_entity).
        if (entity == exclude_entity && ab.tap_cost) continue;
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

    // max_yield_only (the retry strategy — see auto_pay_mana): a permanent taps once, so
    // collapse each entity to its single highest-yield entry ACROSS colors (the loop above
    // already collapsed within a color). First-seen wins a yield tie, keeping the pass
    // deterministic.
    if (max_yield_only) {
        std::vector<SourceInfo> best;
        for (auto &si : valid_sources) {
            size_t amt = eval_mana_amount(si.ability, controller, orderer);
            bool merged = false;
            for (auto &existing : best) {
                if (existing.entity != si.entity) continue;
                if (amt > eval_mana_amount(existing.ability, controller, orderer)) {
                    existing.ability = si.ability;
                    existing.color = si.color;
                }
                merged = true;
                break;
            }
            if (!merged) {
                si.is_multi_color = false;  // one entry per entity now
                best.push_back(si);
            }
        }
        valid_sources = std::move(best);
    }

    // Candidate ordering: within every preference tier below, engage first the source
    // whose tap gives up the least. Two ranked criteria, applied by a stable sort (ties
    // keep entity order), read uniformly by all tiers and cover_activation_cost's payer
    // scan, so simulate (can_pay_mana) and the real payment stay in lockstep:
    //   1. HAND COLOR DEMAND — prefer a source whose producible colors the other cards in
    //      hand need least (with Lightning Bolt in hand, a generic pip taps Wasteland's
    //      {C} before a Mountain, keeping red open). A multi-color source is scored by
    //      the MOST-demanded color it could produce — any of its entries taps the whole
    //      permanent, giving up its best option.
    //   2. UTILITY — prefer a source whose only activated abilities are mana abilities
    //      over one that also carries a utility ability (Karakas's bounce, Wasteland's
    //      destroy, a man-land's animation).
    {
        std::array<int, 6> demand = hand_color_demand(controller, paid_for, orderer);
        std::map<Entity, int> entity_demand;
        for (const auto &si : valid_sources) {
            int d = (si.color >= WHITE && si.color <= COLORLESS)
                        ? demand[static_cast<size_t>(si.color)] : 0;
            auto it = entity_demand.find(si.entity);
            if (it == entity_demand.end()) entity_demand[si.entity] = d;
            else it->second = std::max(it->second, d);
        }
        std::map<Entity, bool> entity_utility;
        for (const auto &si : valid_sources)
            if (!entity_utility.count(si.entity))
                entity_utility[si.entity] = has_nonmana_activated_ability(si.entity);
        std::stable_sort(valid_sources.begin(), valid_sources.end(),
            [&](const SourceInfo &a, const SourceInfo &b) {
                int da = entity_demand[a.entity], db = entity_demand[b.entity];
                if (da != db) return da < db;
                return entity_utility[a.entity] < entity_utility[b.entity];
            });
    }

    // Helper: activate a source (tap, sacrifice, pay costs, add mana). si.color always
    // equals si.ability.color (set when the SourceInfo was built), so the shared
    // activate_mana_source reads the produced color straight off the ability.
    auto activate_source = [&](const SourceInfo &si) {
        return activate_mana_source(si.entity, si.ability, controller, orderer, pool, player,
                                    commit, ManaLogStyle::ACTIVATED);
    };

    std::set<Entity> tapped_entities;

    // A cost-bearing mana source (Talon Gates' {1}{T}: add one mana of any color) must
    // actually PAY its activation cost before producing. Cover the cost from the working
    // pool, tapping additional cost-FREE sources into the pool when it falls short. The
    // cover is planned on copies first, so a failed cover taps nothing and the caller just
    // skips the source; on success the payer taps are committed here and the cost itself is
    // then paid inside activate_mana_source. This preserves the invariant that total mana
    // produced minus activation costs paid equals the amount credited toward `remaining`.
    auto cover_activation_cost = [&](const SourceInfo &si) -> bool {
        const ManaValue &act = si.ability.activation_mana_cost;
        if (act.empty() || can_afford_pool(pool, act)) return true;
        ManaValue trial = pool;
        std::set<Entity> planned = tapped_entities;
        planned.insert(si.entity);
        std::vector<size_t> plan;  // indices into valid_sources to engage as payers
        while (!can_afford_pool(trial, act)) {
            // The unpayable pips of the activation cost against the trial pool; pay a
            // colored pip first (it is the more constrained match), else a generic one.
            ManaValue trial_copy = trial;
            ManaValue need = pay_from_pool(trial_copy, act);
            Colors pip = GENERIC;
            for (Colors c : need)
                if (c != GENERIC) { pip = c; break; }
            bool found = false;
            // Painless payers first; a painful one (Ancient Tomb) is engaged only when no
            // painless payer can supply the pip — same preference as the main tiers below.
            for (int pass = 0; pass < 2 && !found; pass++) {
                for (size_t i = 0; i < valid_sources.size(); i++) {
                    auto &cand = valid_sources[i];
                    if (planned.count(cand.entity)) continue;
                    // Payers must themselves be cost-free — no recursive cost chains.
                    if (!cand.ability.activation_mana_cost.empty()) continue;
                    if (pass == 0 && mana_ability_is_painful(cand.ability)) continue;
                    if (pip != GENERIC && !any_color && cand.color != pip) continue;
                    size_t amt = eval_mana_amount(cand.ability, controller, orderer);
                    if (amt == 0) continue;
                    for (size_t k = 0; k < amt; k++) trial.insert(cand.color);
                    planned.insert(cand.entity);
                    plan.push_back(i);
                    found = true;
                    break;
                }
            }
            if (!found) return false;
        }
        for (size_t i : plan) {
            if (!activate_source(valid_sources[i])) return false;  // cost-free; can't bounce
            tapped_entities.insert(valid_sources[i].entity);
        }
        return can_afford_pool(pool, act);
    };

    // Cost-bearing sources net less mana than they produce (their activation cost consumes
    // other sources' output), so every priority tier below prefers cost-free candidates.
    auto cost_free = [](const SourceInfo &s) { return s.ability.activation_mana_cost.empty(); };

    // Life is a resource too: a painful source (Ancient Tomb's self-damage rider, a pain
    // land's PayLife cost) is engaged only when the cost cannot be covered painlessly —
    // every regular tier below requires painless, and painful sources sit in their own
    // tier just above the sac_self last resort. Efficiency (Tomb's 2-for-1 tap) never
    // outranks life; only necessity (no painless way to finish the payment, including a
    // colored pip only a painful source produces) does.
    auto painless = [](const SourceInfo &s) { return !mana_ability_is_painful(s.ability); };

    // Pay colored costs first. Pay the MOST color-constrained pip first: the greedy per-pip
    // payer below has no lookahead, so if it spends a shared dual (e.g. a Tropical Island that
    // taps for G or U) on a plentiful color, it can strand a later pip whose ONLY producer was
    // that dual — the spell then reads as unpayable even though a valid assignment exists
    // (a {1}{G}{U} spell over Tropical Island + Underground Sea + Wasteland is the canonical
    // case: paying the U pip with the Tropical leaves nothing to make G). Ordering the colored
    // pips by how many distinct untapped sources can produce each color (scarcest first, ties by
    // color enum for determinism) makes the scarce pip claim its producer before a flexible pip
    // can take it. `producer_count` is computed once from the pre-payment source set; that
    // static order is enough to reserve uniquely-sourced colors. (Under ManaConvert every source
    // pays every pip, so the scarcity ordering is a harmless no-op.)
    std::map<Colors, int> producer_count;
    {
        std::map<Colors, std::set<Entity>> by_color;
        for (const auto &si : valid_sources)
            if (si.color >= WHITE && si.color <= COLORLESS)
                by_color[si.color].insert(si.entity);
        for (const auto &[c, ents] : by_color) producer_count[c] = static_cast<int>(ents.size());
    }
    auto pip_scarcity = [&](Colors c) {
        auto pc = producer_count.find(c);
        return pc == producer_count.end() ? 0 : pc->second;
    };
    while (true) {
        // Select the remaining colored pip whose color has the fewest producers. `remaining`
        // is sorted ascending by color enum, so replacing only on a STRICTLY-smaller scarcity
        // makes ties resolve to the lowest color enum (the first equal-scarcity pip scanned).
        auto it = remaining.end();
        int best = 0;
        for (auto pit = remaining.begin(); pit != remaining.end(); ++pit) {
            if (*pit == GENERIC) continue;
            int sc = pip_scarcity(*pit);
            if (it == remaining.end() || sc < best) { best = sc; it = pit; }
        }
        if (it == remaining.end()) break;  // no colored pips left — pay generic below
        Colors needed = *it;

        // Check pool first: exact color, or — under ManaConvert — any mana already floating.
        if (pool.count(needed) > 0) {
            auto pit = pool.find(needed);
            if (spent_sink) spent_sink->insert(*pit);
            pool.erase(pit);
            it = remaining.erase(it);
            continue;
        }
        if (any_color && !pool.empty()) {
            if (spent_sink) spent_sink->insert(*pool.begin());
            pool.erase(pool.begin());
            it = remaining.erase(it);
            continue;
        }

        // Try sources in priority order: adds_no_counter > single-color > multi-color. Under
        // ManaConvert any source's mana can pay the pip, so the exact-color gate is dropped.
        auto try_source = [&](auto predicate) -> bool {
            for (auto &si : valid_sources) {
                if (tapped_entities.count(si.entity)) continue;
                if (!any_color && si.color != needed) continue;
                if (!predicate(si)) continue;
                // A cost-bearing source is engaged only once its activation cost is
                // covered (pre-tapping payer sources into the pool); if the cost cannot
                // be covered — or the activation itself bounces — the source is skipped
                // untouched and the scan moves on.
                if (!cover_activation_cost(si)) continue;
                if (!activate_source(si)) continue;
                tapped_entities.insert(si.entity);
                // Only satisfy the pip with mana the source ACTUALLY produced. A source
                // that produced no mana of `needed` (e.g. a 0-mana ManaReflected source
                // with no qualifying permanent) erases nothing — the pip stays unpaid and
                // we keep scanning. This mirrors the interactive payer, where the pip is
                // only spent once can_afford_pool sees the color in the real pool. Under
                // ManaConvert any produced mana counts, so spend the first available pip.
                if (any_color && !pool.empty()) {
                    if (spent_sink) spent_sink->insert(*pool.begin());
                    pool.erase(pool.begin());
                    it = remaining.erase(it);
                    return true;
                }
                if (!any_color && pool.count(needed) > 0) {
                    auto pit = pool.find(needed);
                    if (spent_sink) spent_sink->insert(*pit);
                    pool.erase(pit);
                    it = remaining.erase(it);
                    return true;
                }
                // Source produced nothing usable; it is now tapped (tracked above) so it
                // won't be retried, but the pip is still owed — try the next candidate.
            }
            return false;
        };
        if (try_source([&](const SourceInfo &s) { return s.ability.adds_no_counter && !s.ability.sac_self && painless(s); })) continue;
        if (try_source([&](const SourceInfo &s) { return cost_free(s) && !s.is_multi_color && !s.ability.sac_self && painless(s); })) continue;
        if (try_source([&](const SourceInfo &s) { return cost_free(s) && !s.ability.sac_self && painless(s); })) continue;
        if (try_source([&](const SourceInfo &s) { return !s.is_multi_color && !s.ability.sac_self && painless(s); })) continue;
        if (try_source([&](const SourceInfo &s) { return !s.ability.sac_self && painless(s); })) continue;
        if (try_source([](const SourceInfo &s) { return !s.ability.sac_self; })) continue;  // painful, if needed
        if (try_source([](const SourceInfo &) { return true; })) continue;  // sac_self last resort
        return false;
    }

    // Pay generic costs with remaining sources — prefer adds_no_counter (Cavern)
    while (remaining.count(GENERIC) > 0) {
        if (pool.size() > 0) {
            if (spent_sink) spent_sink->insert(*pool.begin());
            pool.erase(pool.begin());
            auto git = remaining.find(GENERIC);
            remaining.erase(git);
            continue;
        }
        auto try_generic = [&](auto predicate) -> bool {
            for (auto &si : valid_sources) {
                if (tapped_entities.count(si.entity)) continue;
                if (!predicate(si)) continue;
                // Same activation-cost gate as the colored loop above.
                if (!cover_activation_cost(si)) continue;
                if (!activate_source(si)) continue;
                tapped_entities.insert(si.entity);
                // Only pay a generic pip with mana the source actually produced; a 0-mana
                // source erases nothing (consistent with the colored loop above).
                if (pool.size() > 0) {
                    if (spent_sink) spent_sink->insert(*pool.begin());
                    pool.erase(pool.begin());
                    auto git = remaining.find(GENERIC);
                    remaining.erase(git);
                    return true;
                }
            }
            return false;
        };
        if (try_generic([&](const SourceInfo &s) { return s.ability.adds_no_counter && !s.ability.sac_self && painless(s); })) continue;
        if (try_generic([&](const SourceInfo &s) { return cost_free(s) && !s.ability.sac_self && painless(s); })) continue;
        if (try_generic([&](const SourceInfo &s) { return !s.ability.sac_self && painless(s); })) continue;
        if (try_generic([](const SourceInfo &s) { return !s.ability.sac_self; })) continue;  // painful, if needed
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
                  bool has_improvise, Entity exclude_entity) {
    Entity player_entity = get_player_entity(controller);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    // Run the exact machine-mode payment algorithm in simulate mode (no side effects).
    // This is the single predicate behind both "is this castable" and "pay for it",
    // so a spell can never be offered as legal and then fail to pay (and vice versa).
    ManaValue remaining = cost;
    return auto_pay_mana(controller, remaining, paid_for, orderer, has_delve, /*commit=*/false,
                         has_improvise, exclude_entity);
}

bool float_mana_before_cost_removal(Entity leaving, Zone::Ownership controller,
                                    std::shared_ptr<Orderer> orderer,
                                    const ManaValue &unpaid_cost, Entity paid_for) {
    if (unpaid_cost.empty()) return false;
    if (!is_battlefield_permanent(leaving, controller)) return false;
    Entity player_entity = get_player_entity(controller);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    // Floating mana the payment does not need would only empty away at end of step —
    // and a painful source would charge life for it. Nothing owed, nothing tapped.
    if (can_afford_pool(player.mana, unpaid_cost)) return false;

    // Candidates: this permanent's mana abilities that are usable for THIS payment (same
    // restricted-mana gate the payer applies) and whose activation costs nothing but the
    // tap. An ability that spends a SECOND resource — its own sacrifice, a discard (Lion's
    // Eye Diamond), a land bounce — is skipped: the permanent is already being spent as a
    // cost, so stacking another cost onto it is never the intent. A life cost is allowed
    // here and weighed as "painful" below.
    std::vector<Ability> candidates;
    for (auto &[entity, ab] : collect_available_mana_sources(controller, orderer)) {
        if (entity != leaving) continue;
        if (!mana_source_usable_for(ab, entity, paid_for)) continue;
        if (!ab.activation_mana_cost.empty()) continue;
        if (ab.sac_self || ab.discard_hand_cost || ab.discard_self_cost ||
            ab.return_cost_count > 0)
            continue;
        candidates.push_back(ab);
    }
    if (candidates.empty()) return false;

    // Rank: producing a color the cost actually asks for beats one it doesn't (a dual land
    // taps for the pip that is owed), and a painless ability beats a painful one. Ties keep
    // collect_available_mana_sources' order, so the pick is deterministic.
    auto score = [&](const Ability &ab) {
        return (unpaid_cost.count(ab.color) > 0 ? 2 : 0) + (mana_ability_is_painful(ab) ? 0 : 1);
    };
    const Ability *chosen = &candidates.front();
    for (const auto &ab : candidates)
        if (score(ab) > score(*chosen)) chosen = &ab;

    // Life is a resource: engage a painful source only when the cost cannot be paid at all
    // without this permanent (can_afford_with_sources' exclude_entity is exactly the
    // "this source is being consumed by the cost" question it exists for).
    if (mana_ability_is_painful(*chosen) &&
        can_afford_with_sources(controller, unpaid_cost, orderer, leaving, paid_for))
        return false;

    return activate_mana_source(leaving, *chosen, controller, orderer, player.mana, player,
                                /*commit=*/true, ManaLogStyle::ACTIVATED);
}

// Depth-first enumeration of hybrid-pip assignments (see resolve_hybrid_cost). `cur` carries the
// concrete cost so far; at the leaf it is tested via can_pay_mana. Colored options are tried
// before a twobrid's generic alternative so the cheapest/most-flexible payable assignment wins.
static bool resolve_hybrid_recurse(Zone::Ownership caster, ManaValue &cur,
                                   const std::vector<HybridPip> &hybrids, size_t idx,
                                   Entity paid_for, std::shared_ptr<Orderer> orderer,
                                   bool has_delve, bool has_improvise, ManaValue *out) {
    if (idx == hybrids.size()) {
        if (!can_pay_mana(caster, cur, paid_for, orderer, has_delve, has_improvise)) return false;
        if (out) *out = cur;
        return true;
    }
    const HybridPip &pip = hybrids[idx];
    for (Colors c : pip.colors) {
        cur.insert(c);
        if (resolve_hybrid_recurse(caster, cur, hybrids, idx + 1, paid_for, orderer,
                                   has_delve, has_improvise, out))
            return true;
        cur.erase(cur.find(c));
    }
    if (pip.generic_alt > 0) {
        for (int i = 0; i < pip.generic_alt; i++) cur.insert(GENERIC);
        if (resolve_hybrid_recurse(caster, cur, hybrids, idx + 1, paid_for, orderer,
                                   has_delve, has_improvise, out))
            return true;
        for (int i = 0; i < pip.generic_alt; i++) cur.erase(cur.find(GENERIC));
    }
    return false;
}

bool resolve_hybrid_cost(Zone::Ownership caster, const ManaValue &base_flat_cost,
                         const std::vector<HybridPip> &hybrids, Entity paid_for,
                         std::shared_ptr<Orderer> orderer, bool has_delve, bool has_improvise,
                         ManaValue *out_resolved) {
    ManaValue cur = base_flat_cost;
    return resolve_hybrid_recurse(caster, cur, hybrids, 0, paid_for, orderer, has_delve,
                                  has_improvise, out_resolved);
}

bool prompt_mana_payment(Zone::Ownership controller, const ManaValue &cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve, bool has_improvise, ManaValue *spent_out) {
    Entity player_entity = get_player_entity(controller);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Arm the Converge spent-color sink for the duration of this payment (CR 702.90). The RAII
    // guard restores the previous sink on every return path (nested payments — e.g. a mana-source
    // activation cost paid mid-payment — correctly re-target their own sink). Dry-run affordability
    // checks inside the payment route through pay_from_pool on pool COPIES, so the armed sink never
    // sees a simulated spend.
    struct SinkGuard {
        ManaValue *prev;
        ~SinkGuard() { s_mana_spent_sink = prev; }
    } sink_guard{s_mana_spent_sink};
    s_mana_spent_sink = spent_out;

    // If pool already covers it, just spend
    if (can_afford_pool(player.mana, cost)) {
        spend_mana(controller, cost, paid_for);
        return true;
    }

    bool is_machine = InputLogger::instance().is_machine_schedule();

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
            // A cost-bearing source (Talon Gates' {1}{T}) refuses to activate while the pool
            // can't cover its activation cost: nothing is tapped and no mana is produced, and
            // the loop re-prompts so the player can float mana from another source first.
            if (!activate_mana_source(chosen.source_entity, chosen.ability, controller, orderer,
                                      player.mana, player, /*commit=*/true,
                                      ManaLogStyle::ACTIVATED))
                game_log("Cannot pay that ability's activation cost — tap another source for mana first.\n");
        }
    }

    return true;
}

// (The old blocking prompt_delve_exiles lived here. Batch 11 moved the interactive
// delve count/pick prompts into run_cast_flow's DELVE_COUNT/DELVE_PICK steps
// (action_processor.cpp), where they suspend as loop-top pending decisions; the
// shared primitives is_delve_eligible/delve_exile_one above are what it consumes.)
