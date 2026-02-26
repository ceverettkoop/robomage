#ifndef STACK_MANAGER_H
#define STACK_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include <memory>
#include <vector>

class Orderer;

class StackManager : public System {

public:
    static void init();
    bool is_empty();
    void resolve_top(std::shared_ptr<Orderer> orderer);
    std::vector<Entity> get_stack_contents();
};

#endif /* STACK_MANAGER_H */
