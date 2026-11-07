#ifndef STACK_MANAGER_H
#define STACK_MANAGER_H

#include "../ecs/system.h"
#include "../resolution_frame.h"
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
    // Resolves `ab` with nothing on the stack (what replaces a turn-based draw) in a frame of its
    // own, so a question it asks can park. SUSPENDED leaves the frame armed; a later call, with
    // the answer latched, carries on the frame's copy and ignores `ab`.
    static ResolveStatus resolve_off_stack(const Ability &ab, std::shared_ptr<Orderer> orderer);

private:
    static bool aura_spell_target_illegal(Entity spell);
    static bool spell_ceases_to_be_overlaid(Entity spell);
    // Record on the resolving permanent spell's entry how it was cast (EntryInfo): evoke,
    // offspring, escape, impending and warp, and the X it was cast with.
    static void record_cast_entry(Entity spell_entity, const Spell &spell);
};

#endif /* STACK_MANAGER_H */
