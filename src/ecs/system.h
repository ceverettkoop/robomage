#ifndef SYSTEM_H
#define SYSTEM_H

#include <set>

// credit https://austinmorlan.com/posts/entity_component_system/

#include "entity.h"

class System {
    public:
        std::set<Entity> mEntities;
};

#endif /* SYSTEM_H */
