// General "copy a spell on the stack" mechanism (CR 707.10 / 707.12).
//
// A copy of a spell is a new object created ON THE STACK that copies the original spell's
// copiable characteristics (its CardData, color, and resolving ability). A copy is NOT cast:
// it pays no costs, is put directly onto the stack, and does not trigger "when you cast"
// abilities (so a copy with Replicate replicates nothing further). The controller of a copy
// MAY choose new targets for it (CR 707.12); here each copy re-runs the normal target
// selection. A copy is not a card — when it leaves the stack (resolves or is countered) it
// ceases to exist instead of moving to a zone (handled in the stack manager / effects::counter
// via Spell::is_copy).
//
// This routine is reusable by any copy-spell effect (Replicate drives it today; storm/fork
// would reuse it). It deliberately reuses the existing entity/component construction, the
// Spell/Ability components, and the shared select_target targeting path rather than
// open-coding a parallel copy or targeting UI.

#include <memory>

#include "../action_processor.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

// Forward declarations
static void spawn_spell_copies(const CardData &orig_card, const ColorIdentity *orig_color,
                               const Ability *orig_ability, bool cant_be_countered, int x_paid,
                               int count, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);

void copy_spell_on_stack(Entity original, int count, Zone::Ownership controller,
                         std::shared_ptr<Orderer> orderer) {
    if (count <= 0) return;
    if (!global_coordinator.entity_has_component<CardData>(original)) return;

    // Copiable characteristics: the card's printed values and color identity (CR 707.2). These
    // survive on the card entity even after the spell has left the stack (a countered spell keeps
    // its CardData/ColorIdentity in the graveyard; only its Spell/Ability components are stripped).
    const auto &orig_card = global_coordinator.GetComponent<CardData>(original);
    const ColorIdentity *orig_color =
        global_coordinator.entity_has_component<ColorIdentity>(original)
            ? &global_coordinator.GetComponent<ColorIdentity>(original)
            : nullptr;

    if (global_coordinator.entity_has_component<Spell>(original)) {
        // The original is still a live spell on the stack (Replicate / an uncountered Storm
        // spell): copy its resolving Ability and Spell flags directly.
        const Spell &orig_spell = global_coordinator.GetComponent<Spell>(original);
        const Ability *orig_ability =
            global_coordinator.entity_has_component<Ability>(original)
                ? &global_coordinator.GetComponent<Ability>(original)
                : nullptr;
        spawn_spell_copies(orig_card, orig_color, orig_ability, orig_spell.cant_be_countered,
                           orig_spell.x_paid, count, controller, orderer);
    } else {
        // The original has already left the stack (e.g. a Storm spell countered before its Storm
        // triggered ability resolved — CR 702.40a / 113.7a: the Storm ability is a separate object,
        // independent of the spell that created it, so countering that spell doesn't stop the
        // copies). Rebuild the copies from the spell's last-known copiable characteristics: the
        // card's SPELL ability template reproduces the resolving ability, and the copies choose
        // their own new targets anyway, so the template's (absent) targets don't matter.
        const Ability *orig_ability = nullptr;
        for (const auto &tmpl : orig_card.abilities)
            if (tmpl.ability_type == Ability::SPELL) { orig_ability = &tmpl; break; }
        spawn_spell_copies(orig_card, orig_color, orig_ability, false, 0, count, controller,
                           orderer);
    }
}

// Create `count` copies of a spell on top of the stack from its copiable characteristics. Used
// for both the live-spell path (source components read directly) and the last-known-information
// path (source already off the stack). See copy_spell_on_stack for the CR references.
static void spawn_spell_copies(const CardData &orig_card, const ColorIdentity *orig_color,
                               const Ability *orig_ability, bool cant_be_countered, int x_paid,
                               int count, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer) {
    for (int i = 0; i < count; i++) {
        Entity copy = global_coordinator.CreateEntity();

        global_coordinator.AddComponent(copy, orig_card);
        if (orig_color) global_coordinator.AddComponent(copy, *orig_color);

        // The copy's Zone is added by place_created_on_stack() at the end (CR 707.10: the copy
        // is created on the stack, not moved there from any zone). Until then it has no Zone, so
        // it can't appear as a target candidate during its own targeting below, and target
        // perspective falls back to the ability's controller.

        Spell copy_spell;
        copy_spell.caster = controller;
        copy_spell.cant_be_countered = cant_be_countered;
        copy_spell.x_paid = x_paid;  // copies copy the X chosen for the original (CR 707.10b)
        copy_spell.is_copy = true;               // ceases to exist when it leaves the stack
        // A copy is not cast: it has no replicate count of its own and copies no further.
        global_coordinator.AddComponent(copy, copy_spell);

        // Copy the resolving spell ability and let the controller choose new targets (CR 707.12).
        if (orig_ability) {
            Ability ability = *orig_ability;
            ability.source = copy;
            ability.controller = controller;
            ability.target = 0;
            ability.targets.clear();
            for (auto &sub : ability.subabilities) {
                sub.source = copy;
                sub.controller = controller;
                sub.target = 0;
                sub.targets.clear();
            }
            if (ability.valid_tgts != "N_A") {
                // If no legal target remains for this copy, it can't be put on the stack
                // (CR 707.10c-ish: a copy that requires a target with none available is not
                // created). Skip it cleanly rather than placing an untargeted copy.
                if (!has_legal_targets(ability, orderer)) {
                    global_coordinator.DestroyEntity(copy);
                    game_log("A copy of %s has no legal target — not created\n",
                             orig_card.name.c_str());
                    continue;
                }
                select_target(ability, orderer, controller);
            }
            for (auto &sub : ability.subabilities) {
                // Guard subability targeting the same way the main ability is guarded above:
                // select_target on an empty candidate pool (target required, none legal) would
                // index an empty action list. A copy whose subability has no legal target simply
                // leaves it untargeted (it does nothing at resolution) rather than crashing.
                if (sub.valid_tgts != "N_A" && has_legal_targets(sub, orderer))
                    select_target(sub, orderer, controller);
            }
            // Modal spell copy (CR 707.10c): the copy has the SAME chosen modes — they can't
            // be changed — but its controller may choose new targets for each chosen mode.
            // Re-select a chosen mode's targets when a legal candidate exists for this copy;
            // otherwise keep the original's targets (re-verified at resolution, CR 608.2b,
            // fizzling that mode naturally if they're illegal).
            for (int idx : ability.charm_chosen) {
                if (idx < 0 || static_cast<size_t>(idx) >= ability.charm_choices.size()) continue;
                Ability &mode = ability.charm_choices[static_cast<size_t>(idx)];
                mode.source = copy;
                mode.controller = controller;
                if (mode.valid_tgts != "N_A" && has_legal_targets(mode, orderer)) {
                    mode.target = 0;
                    mode.targets.clear();
                    select_target(mode, orderer, controller);
                }
            }
            global_coordinator.AddComponent(copy, ability);
        }

        // Bring the copy into existence on top of the stack (above the original, so it resolves
        // first) without firing a zone-change event/replacement it never earned.
        orderer->place_created_on_stack(copy, controller);
        game_log("%s copies %s\n", player_name(controller).c_str(), orig_card.name.c_str());
    }
}
