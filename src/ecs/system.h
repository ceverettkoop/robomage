#ifndef SYSTEM_H
#define SYSTEM_H

// credit https://austinmorlan.com/posts/entity_component_system/
#include "entity.h"
#include "entity_set.h"

class System {
    public:
        EntitySet mEntities;
};

#endif /* SYSTEM_H */
