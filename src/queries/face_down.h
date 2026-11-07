#ifndef QUERIES_FACE_DOWN_H
#define QUERIES_FACE_DOWN_H

#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

// ── Face-down objects (CR 708, 406.3) ─────────────────────────────────────────

// Whether `e` lies face down where it is: the one read of Zone::is_face_down.
inline bool is_face_down(Entity e) {
    return global_coordinator.entity_has_component<Zone>(e) && global_coordinator.GetComponent<Zone>(e).is_face_down;
}

#endif /* QUERIES_FACE_DOWN_H */
