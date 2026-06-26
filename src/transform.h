#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "ecs/entity.h"

// DFC transform subsystem (711 / 712). A double-faced permanent is a single entity
// whose front face is its CardData and whose back face is CardData::backside (a full
// second face: own name, types, P/T, loyalty, abilities). Transforming swaps which
// face is "up" by rebuilding the permanent's runtime state from the now-active face.
//
// set_permanent_face rebuilds name, types, Creature/Damage components, loyalty
// counters, and the active face's activated + static abilities from the requested
// face. No-op if the entity is not a permanent, or show_back is requested but the
// card has no back face. Used by Delver-style in-place flips, Ajani's
// exile-and-return-transformed, and the ChangeZone "Transformed$ True" entry path.
void set_permanent_face(Entity e, bool show_back);

// Flip to whichever face is not currently shown.
void transform_permanent(Entity e);

#endif /* TRANSFORM_H */
