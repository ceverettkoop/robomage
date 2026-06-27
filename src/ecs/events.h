#ifndef EVENTS_H
#define EVENTS_H

#include "event.h"

// Event IDs
// Each constant is annotated with the card-script phrase it corresponds to.
namespace Events {
    constexpr EventId CARD_CHANGED_ZONE      = 1;  // any card changes zone; Params: ENTITY=card, PLAYER=owner, ORIGIN=old zone, DESTINATION=new zone
    constexpr EventId PLAYER_DREW_CARD       = 2;  // "whenever a player draws a card"; fired per individual card drawn. Params: PLAYER=drawer, ENTITY=drawn card, FIRST_IN_STEP=1 if this is the first card drawn in the drawer's draw step
    constexpr EventId UPKEEP_BEGAN           = 4;  // "at the beginning of [your] upkeep" / Phase$ Upkeep
    constexpr EventId NONCREATURE_SPELL_CAST = 5;  // "whenever [you cast] a noncreature spell" / SpellCast ValidCard$nonCreature
    constexpr EventId END_STEP_BEGAN         = 6;  // "at the beginning of [your] end step" / Phase$ EndStep
    constexpr EventId DRAW_STEP_BEGAN        = 7;  // "at the beginning of [your] draw step" / Phase$ Draw
    constexpr EventId SPELL_CAST             = 9;  // every spell cast; Params: PLAYER=caster, ENTITY=spell on the stack
    constexpr EventId CREATURE_ATTACKED_ALONE = 10; // exactly one creature declared as attacker; Params: ENTITY=sole attacker, PLAYER=controller
    constexpr EventId COMBAT_DAMAGE_TO_PLAYER = 11; // creature dealt combat damage to a player; Params: ENTITY=source creature, PLAYER=damaged player entity, AMOUNT=damage
    constexpr EventId CREATURE_ATTACKED       = 12; // "whenever this creature attacks" — fired once per declared attacker; Params: ENTITY=attacker, PLAYER=controller
    constexpr EventId BEGIN_COMBAT_BEGAN      = 13; // "at the beginning of combat on your turn" / Phase$ BeginCombat; Params: PLAYER=active player
    constexpr EventId ATTACKERS_DECLARED      = 14; // "whenever you attack" — fired once when one or more attackers are declared (Mode$ AttackersDeclared); Params: PLAYER=attacking (active) player
    constexpr EventId TAPPED_FOR_MANA         = 15; // "whenever you tap a <permanent> for mana" (Mode$ TapsForMana). Static$ True triggers resolve immediately as a mana-additional effect (off-stack, CR 605.1a). Params: ENTITY=tapped source, PLAYER=controller who tapped it
    constexpr EventId BECAME_TARGET           = 16; // "whenever ~ becomes the target of a spell" (Mode$ BecomesTarget). Fired once per (targeting object, targeted permanent) pair as a spell/ability with chosen targets is placed on the stack (CR 603.2c). Params: ENTITY=targeting spell/ability, PLAYER=its controller, TARGET=the permanent that became a target
    constexpr EventId BECAME_MONSTROUS        = 17; // "when ~ becomes monstrous" (Mode$ BecomeMonstrous, CR 701.37). Fired by a resolving Monstrosity$ ability that turns a not-yet-monstrous permanent monstrous. Params: ENTITY=the permanent that became monstrous, PLAYER=its controller
}

// Param IDs used across events
namespace Params {
    constexpr ParamId ENTITY      = 1;  // The primary entity involved in an event
    constexpr ParamId PLAYER      = 2;  // The player entity involved in an event
    constexpr ParamId ORIGIN      = 3;  // Zone::ZoneValue before the move (CARD_CHANGED_ZONE)
    constexpr ParamId DESTINATION = 4;  // Zone::ZoneValue after the move (CARD_CHANGED_ZONE)
    constexpr ParamId AMOUNT      = 5;  // Numeric amount (e.g. damage dealt)
    constexpr ParamId FIRST_IN_STEP = 6;  // 1 if a PLAYER_DREW_CARD is the first card drawn in the drawer's draw step, else 0
    constexpr ParamId TARGET      = 7;  // The permanent that became a target (BECAME_TARGET), distinct from ENTITY (the targeting object)
}

#endif /* EVENTS_H */
