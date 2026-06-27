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

void copy_spell_on_stack(Entity original, int count, Zone::Ownership controller,
                         std::shared_ptr<Orderer> orderer) {
    if (count <= 0) return;
    if (!global_coordinator.entity_has_component<CardData>(original)) return;
    if (!global_coordinator.entity_has_component<Spell>(original)) return;

    const auto &orig_card = global_coordinator.GetComponent<CardData>(original);
    const Spell &orig_spell = global_coordinator.GetComponent<Spell>(original);
    bool orig_has_ability = global_coordinator.entity_has_component<Ability>(original);

    for (int i = 0; i < count; i++) {
        Entity copy = global_coordinator.CreateEntity();

        // Copiable characteristics: the card's printed values and color identity (CR 707.2). A
        // spell on the stack always carries a ColorIdentity (added when its card entity was
        // created), so copy it directly.
        global_coordinator.AddComponent(copy, orig_card);
        if (global_coordinator.entity_has_component<ColorIdentity>(original))
            global_coordinator.AddComponent(
                copy, global_coordinator.GetComponent<ColorIdentity>(original));

        // The copy's Zone is added by place_created_on_stack() at the end (CR 707.10: the copy
        // is created on the stack, not moved there from any zone). Until then it has no Zone, so
        // it can't appear as a target candidate during its own targeting below, and target
        // perspective falls back to the ability's controller.

        Spell copy_spell;
        copy_spell.caster = controller;
        copy_spell.cant_be_countered = orig_spell.cant_be_countered;
        copy_spell.x_paid = orig_spell.x_paid;  // copies copy the X chosen for the original (CR 707.10b)
        copy_spell.is_copy = true;               // ceases to exist when it leaves the stack
        // A copy is not cast: it has no replicate count of its own and copies no further.
        global_coordinator.AddComponent(copy, copy_spell);

        // Copy the resolving spell ability and let the controller choose new targets (CR 707.12).
        if (orig_has_ability) {
            Ability ability = global_coordinator.GetComponent<Ability>(original);
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
            global_coordinator.AddComponent(copy, ability);
        }

        // Bring the copy into existence on top of the stack (above the original, so it resolves
        // first) without firing a zone-change event/replacement it never earned.
        orderer->place_created_on_stack(copy, controller);
        game_log("%s copies %s\n", player_name(controller).c_str(), orig_card.name.c_str());
    }
}
