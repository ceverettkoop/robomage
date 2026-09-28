#ifndef STACK_MANAGER_H
#define STACK_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include <memory>
#include <vector>

class Orderer;
struct Spell;

class StackManager : public System {

public:
    static void init();
    bool is_empty();
    void resolve_top(std::shared_ptr<Orderer> orderer);

private:
    static bool aura_spell_target_illegal(Entity spell);
    // Record on the resolving permanent spell's entry how it was cast (EntryInfo): evoke,
    // offspring, escape, impending and warp, and the X it was cast with.
    static void record_cast_entry(Entity spell_entity, const Spell &spell);
};

#endif /* STACK_MANAGER_H */
