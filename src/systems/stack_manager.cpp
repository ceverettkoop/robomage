#include "stack_manager.h"

#include <cstddef>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../resolution_frame.h"
#include "../saga.h"
#include "orderer.h"

extern Game cur_game;

// Arm (first entry) or re-enter (resume after suspension) the persisted
// resolution frame for the stack object about to resolve. Forward-declared per
// CLAUDE.md; see definitions below.
static void frame_enter(Entity top_entity, const Ability &ab, bool count_triggered);
static void frame_finish();

// First entry: save the incoming priority, count a triggered ability's
// resolution once (Count$ResolvedThisTurn — guarded by counted_resolution so a
// resume never recounts), save-and-clear the remembered set (each top-level
// resolution gets its own clean Remembered$ scope, CR 608.2 — formerly the
// RememberedResolutionScope RAII in ability.cpp, moved here so a suspension
// keeps the mid-resolution accumulations instead of unwinding them), push the
// ROOT level, and repoint priority at the resolving controller exactly as the
// old locals did. Re-entry: verify the scanned top is still the suspended
// object and change nothing.
static void frame_enter(Entity top_entity, const Ability &ab, bool count_triggered) {
    ResolutionFrame &fr = cur_game.resolution;
    if (fr.active) {
        if (fr.stack_entity != top_entity)
            fatal_error("resolution frame resume: top of stack is not the suspended object");
        return;
    }
    fr = ResolutionFrame{};
    fr.active = true;
    fr.stack_entity = top_entity;
    fr.prev_priority = cur_game.player_a_has_priority;
    fr.saved_remembered = cur_game.remembered_entities;
    cur_game.remembered_entities.clear();
    if (count_triggered && ab.ability_type == Ability::TRIGGERED) {
        cur_game.ability_resolution_counts[ab.source]++;
        fr.counted_resolution = true;
    }
    FrameLevel root;
    root.kind = FrameLevel::ROOT;
    fr.levels.push_back(root);
    cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
}

// Completion epilogue shared by both resolve sites: restore the pre-resolution
// priority and remembered set, and clear the frame. The caller then runs its
// existing component-removal / zone-move / saga / DestroyEntity code unchanged.
static void frame_finish() {
    ResolutionFrame &fr = cur_game.resolution;
    cur_game.player_a_has_priority = fr.prev_priority;
    cur_game.remembered_entities = fr.saved_remembered;
    fr = ResolutionFrame{};
}

void StackManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StackManager>(signature);
}

bool StackManager::is_empty() {
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK) {
            return false;
        }
    }
    return true;
}

void StackManager::resolve_top(std::shared_ptr<Orderer> orderer) {
    Entity top_entity = 0;
    size_t min_distance = SIZE_MAX;
    bool found = false;

    // Find the entity on top of the stack (closest to top, distance_from_top == 0)
    for (auto &&entity : mEntities) {
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::STACK && zone.distance_from_top < min_distance) {
            top_entity = entity;
            min_distance = zone.distance_from_top;
            found = true;
        }
    }
    if (!found) return;

    // Check if it's a spell card (not just an ability)
    if (global_coordinator.entity_has_component<CardData>(top_entity)) {
        auto &card_data = global_coordinator.GetComponent<CardData>(top_entity);
        // Check if it's a permanent type (Creature, Artifact, Enchantment, Planeswalker)
        bool is_permanent = false;
        for (auto &type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Creature" || type.name == "Artifact" || type.name == "Enchantment" ||
                    type.name == "Planeswalker") {
                    is_permanent = true;
                }
            }
        }
        if (is_permanent) {
            // Move to battlefield; Permanent component added by apply_permanent_components on next SBA pass
            // Capture evoke status before the Spell component (which carries it) is removed;
            // apply_permanent_components consumes pending_evoked to set Permanent::evoked.
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_evoke)
                cur_game.pending_evoked.insert(top_entity);
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_offspring)
                cur_game.pending_offspring.insert(top_entity);
            // Spell was cast from the graveyard for its Escape cost: carry the "escaped" bit onto
            // the permanent (apply_permanent_components consumes pending_escaped → Permanent::
            // cast_with_escape) so Uro's "sacrifice it unless it escaped" reads it.
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_escape)
                cur_game.pending_escaped.insert(top_entity);
            // Spell was cast for its Impending alternate cost (CR 702.175): carry the impending
            // bit onto the permanent so apply_permanent_components puts its time counters on it
            // (consumes pending_impending) — it enters as a noncreature until they shed.
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).cast_with_impending)
                cur_game.pending_impending.insert(top_entity);
            // Carry the X paid for an X-cost permanent into the ETB so an "enters with X
            // counters" replacement can read it (Chalice of the Void).
            if (global_coordinator.entity_has_component<Spell>(top_entity) &&
                global_coordinator.GetComponent<Spell>(top_entity).x_paid > 0)
                cur_game.pending_etb_xpaid[top_entity] =
                    global_coordinator.GetComponent<Spell>(top_entity).x_paid;
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                global_coordinator.RemoveComponent<Spell>(top_entity);
            if (global_coordinator.entity_has_component<Ability>(top_entity))
                global_coordinator.RemoveComponent<Ability>(top_entity);
            // This permanent is entering the battlefield because it was cast (CR 614.12):
            // mark it so an ETB replacement that cares about "wasn't cast" (Containment Priest)
            // lets it through. Consumed when its Permanent component is created.
            cur_game.cast_to_battlefield.insert(top_entity);
            orderer->add_to_zone(false, top_entity, Zone::BATTLEFIELD);
            auto &top_zone = global_coordinator.GetComponent<Zone>(top_entity);
            top_zone.controller = top_zone.owner;
            // TODO ETB event here
            game_log("%s enters the battlefield\n", card_data.name.c_str());
        } else {
            // Instant/Sorcery - resolve the Ability component added at cast time, then go to graveyard
            bool was_flashback = spell_cast_with_flashback(top_entity);
            // Restore the X paid at cast time so a Count$xPaid amount in the resolving
            // ability (Kozilek's Command's token/scry/exile counts) reads the value this
            // spell was cast with, not a later cast's. cur_game.x_paid is global, so this must
            // run for every resolving spell — including one cast with X=0 (which still needs to
            // overwrite a stale nonzero value from an unrelated earlier cast) and a non-X spell
            // (x_paid == 0) — not only when x_paid > 0.
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                cur_game.x_paid = static_cast<size_t>(global_coordinator.GetComponent<Spell>(top_entity).x_paid);
            if (global_coordinator.entity_has_component<Ability>(top_entity)) {
                auto &ab = global_coordinator.GetComponent<Ability>(top_entity);
                frame_enter(top_entity, ab, /*count_triggered=*/false);
                // On suspension leave EVERYTHING in place (frame armed, spell on
                // the stack, priority at the chooser) — the next advance_step
                // re-enters here as the resume path.
                if (ab.resolve(orderer, FrameCtx::root()) == ResolveStatus::SUSPENDED) return;
                frame_finish();
                global_coordinator.RemoveComponent<Ability>(top_entity);
            }
            // A COPY of a spell (CR 707.10c) is not a card: once it resolves it ceases to exist
            // rather than going to any zone. Capture before the Spell component is removed.
            bool was_copy = global_coordinator.entity_has_component<Spell>(top_entity) &&
                            global_coordinator.GetComponent<Spell>(top_entity).is_copy;
            global_coordinator.RemoveComponent<Spell>(top_entity);
            if (was_copy) {
                game_log("%s (copy) ceases to exist\n", card_data.name.c_str());
                global_coordinator.DestroyEntity(top_entity);
                return;
            }
            // Shuffle into library instead of graveyard (e.g. Green Sun's Zenith)
            if (card_data.shuffle_into_library) {
                orderer->add_to_zone(false, top_entity, Zone::LIBRARY);
                orderer->shuffle_library(global_coordinator.GetComponent<Zone>(top_entity).owner);
                game_log("%s is shuffled into its owner's library\n", card_data.name.c_str());
            } else if (was_flashback) {
                orderer->add_to_zone(false, top_entity, Zone::EXILE);
                game_log("%s is exiled (flashback)\n", card_data.name.c_str());
            } else {
                orderer->add_to_zone(false, top_entity, Zone::GRAVEYARD);
            }
        }
    }
    // CASE FOR ABILITY ON STACK; not spell
    else if (global_coordinator.entity_has_component<Ability>(top_entity)) {
        auto &ability = global_coordinator.GetComponent<Ability>(top_entity);
        // Count$ResolvedThisTurn tracking (Scythecat Cub) happens inside
        // frame_enter's first-entry block so a resume never recounts.
        frame_enter(top_entity, ability, /*count_triggered=*/true);
        if (ability.resolve(orderer, FrameCtx::root()) == ResolveStatus::SUSPENDED) return;
        frame_finish();

        // CR 714.4: a Saga chapter ability has now left the stack — release the sacrifice gate so a
        // completed Saga can be sacrificed on the next state-based check.
        decrement_saga_in_flight(ability);

        // Destroy the standalone ability entity — it has no card zone to return to
        global_coordinator.DestroyEntity(top_entity);
    }
}
