#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "ecs/entity.h"

// DFC transform subsystem (711 / 712). A double-faced permanent is a single entity
// whose front face is its CardData and whose back face is CardData::backside (a full
// second face: own name, types, P/T, loyalty, abilities). Transforming swaps which
// face is "up"; the permanent stays the same object (CR 712.18).
//
// set_permanent_face turns an ON-BATTLEFIELD permanent to the requested face: name, types,
// printed base P/T, and the face's activated + static abilities change; marked damage,
// combat status, counters (loyalty included) and effects already applied to it are kept.
// A face that isn't a creature loses the Creature/Damage components (CR 506.4). No-op if the
// entity is not a permanent, already shows that face, or show_back is requested but the card
// has no back face. Used by Delver-style in-place flips and day/night. A card that ENTERS
// transformed (Ajani's / Tamiyo's exile-and-return, "Transformed$ True", a modal back face,
// daybound at night) is instead built from its back face as its Permanent is created
// (apply_permanent_components), where it also gets its loyalty (CR 306.5b, 712.14a).
void set_permanent_face(Entity e, bool show_back);

// Flip to whichever face is not currently shown.
void transform_permanent(Entity e);

#endif /* TRANSFORM_H */
