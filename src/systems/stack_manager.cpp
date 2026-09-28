#include "stack_manager.h"

#include <cstddef>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/damage.h"
#include "../components/entry_info.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../action_processor.h"
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
    // The resolution is one effect: the objects it moves stay findable by the rest of it
    // (CR 400.7j) until frame_finish.
    open_follow_window();
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
    // A ChooseCard's chosen cards belong to the resolution that chose them (Ajani's kept
    // permanents, Dauthi Voidwalker's card), so they don't leak into a later nonChosenCard filter.
    cur_game.chosen_cards.clear();
    // The revealing effect is over, so the cards it revealed in a library stop being revealed
    // (CR 701.20a).
    cur_game.revealed_in_library.clear();
    fr = ResolutionFrame{};
    close_follow_window();
    // The effects that could read a card this resolution moved off the battlefield as the
    // departed object have run; from here on it is a new object (CR 400.7).
    supersede_departed_cards();
}

// True if `spell` is an Aura spell whose enchant target (chosen at cast, CR 303.4a) is no longer
// legal as it resolves (CR 608.3b / 608.2b): the object changed zones (CR 400.7) or no longer
// matches its enchant ability. An Aura spell with no recorded target is not judged here.
bool StackManager::aura_spell_target_illegal(Entity spell) {
    const auto &cd = global_coordinator.GetComponent<CardData>(spell);
    if (cd.enchant_filter.empty()) return false;
    const EntryInfo *entry = find_entry_info(spell);
    if (!entry || entry->aura_target.empty()) return false;
    return !pending_aura_target_legal(spell, source_controller(spell));
}

void StackManager::record_cast_entry(Entity spell_entity, const Spell &spell) {
    EntryInfo &entry = entry_info(spell_entity);
    // Evoke: its evoke self-sacrifice ETB trigger fires.
    if (spell.cast_with_evoke) entry.evoked = true;
    // Offspring: its token-copy ETB trigger fires.
    if (spell.cast_with_offspring) entry.offspring = true;
    // Escape: Uro's "sacrifice it unless it escaped" reads it.
    if (spell.cast_with_escape) entry.escaped = true;
    // Impending (CR 702.175): it enters with time counters, a noncreature until they shed.
    if (spell.cast_with_impending) entry.impending = true;
    // Warp: its "exile at the next end step" delayed trigger is registered as it enters.
    if (spell.cast_with_warp) entry.warp = true;
    // The X paid for an X-cost permanent, for an "enters with X counters" replacement (Chalice
    // of the Void).
    if (spell.x_paid > 0) entry.x_paid = spell.x_paid;
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

    // A suspended resolution must RESUME the object it suspended on — not whatever is
    // currently on top of the stack. A resolving object can place NEW objects on the
    // stack ABOVE itself before it suspends: a Storm/Replicate copy machine places copy
    // N (place_created_on_stack shifts it to distance 0), then suspends choosing copy
    // N+1's targets. Re-scanning for distance_from_top == 0 here would pick that
    // freshly-placed copy instead of the still-resolving storm ability, tripping
    // frame_enter's "top of stack is not the suspended object" identity check. While a
    // frame is active we always re-enter its own stack_entity (still on the stack, mid-
    // resolve); the objects placed above it resolve only after it finishes and leaves.
    if (cur_game.resolution.active) {
        top_entity = cur_game.resolution.stack_entity;
    } else {
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
    }

    // Check if it's a spell card (not just an ability)
    if (global_coordinator.entity_has_component<CardData>(top_entity)) {
        auto &card_data = global_coordinator.GetComponent<CardData>(top_entity);
        // A permanent spell (CR 110.4a, any permanent card type incl. Battle) of the face that
        // was cast (a modal back face / split half, CR 712.8f, 709.3) enters the battlefield.
        bool is_permanent = is_permanent_card(active_face(top_entity, card_data));
        if (is_permanent && aura_spell_target_illegal(top_entity)) {
            // CR 608.3b: an Aura spell whose target is illegal doesn't resolve; it is removed
            // from the stack and put into its owner's graveyard.
            game_log("%s doesn't resolve: the object it targets is no longer legal (CR 608.3b)\n",
                     card_data.name.c_str());
            orderer->remove_from_stack(top_entity, Zone::GRAVEYARD);
        } else if (is_permanent) {
            // Move to battlefield; Permanent component added by apply_permanent_components on next SBA pass
            // Carry how it was cast onto its entry before the Spell component (which records
            // it) is removed; apply_permanent_components hands each fact to the Permanent.
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                record_cast_entry(top_entity, global_coordinator.GetComponent<Spell>(top_entity));
            // A resolving permanent spell enters under the control of the spell's controller
            // (CR 608.3a), who need not be its owner. Read before the Spell component goes.
            Zone::Ownership entering_controller = source_controller(top_entity);
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                global_coordinator.RemoveComponent<Spell>(top_entity);
            if (global_coordinator.entity_has_component<Ability>(top_entity))
                global_coordinator.RemoveComponent<Ability>(top_entity);
            // This permanent is entering the battlefield because it was cast (CR 614.12):
            // mark it so an ETB replacement that cares about "wasn't cast" (Containment Priest)
            // lets it through. Consumed when its Permanent component is created.
            entry_info(top_entity).cast = true;
            orderer->add_to_zone(false, top_entity, Zone::BATTLEFIELD);
            auto &top_zone = global_coordinator.GetComponent<Zone>(top_entity);
            top_zone.controller = entering_controller;
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
            // Restore Converge (CR 702.90) — distinct colors of mana spent to cast this spell — the
            // same way as x_paid, so a resolving Count$Converge bound (Prismatic Ending's cmcLEY)
            // reads THIS spell's value. Set for every resolving spell (0 for a no-colored-mana cast)
            // so a stale value from an unrelated earlier cast is always overwritten.
            if (global_coordinator.entity_has_component<Spell>(top_entity))
                cur_game.converge =
                    static_cast<int>(global_coordinator.GetComponent<Spell>(top_entity).colors_spent.size());
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
        // An activated ability carries the X announced when it was activated (CR 107.3a):
        // restore it the way a resolving spell restores Spell::x_paid, so Count$xPaid / cmcLEX
        // read this ability's X and not a spell or ability that resolved in between (Pernicious
        // Deed answered by Lightning Bolt). An ability is not cast, so no mana spent casting it
        // counts for Converge (CR 702.90). A triggered ability carries the X it was put on the
        // stack with (CR 107.3m/n, else 0); an ability with none recorded (x_paid < 0) leaves
        // both as is.
        if (ability.x_paid >= 0) {
            cur_game.x_paid = static_cast<size_t>(ability.x_paid);
            cur_game.converge = 0;
        }
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
