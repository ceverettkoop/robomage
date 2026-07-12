#ifndef COMPONENT_MANAGER_H
#define COMPONENT_MANAGER_H
#include <array>
#include <cassert>
#include <memory>
#include <vector>
#include "component.h"
#include "component_array.h"
#include "entity.h"

// credit https://austinmorlan.com/posts/entity_component_system/
//
// Component types are identified by a process-global integer id assigned on first
// use (component_type_id<T>()), NOT by a per-call typeid()+hash-map lookup. The id
// is a function-local static, so it is computed once and reused for the lifetime of
// the process — across every per-game Coordinator::Init() (init_ecs registers the
// same component set in the same order each game, so the ids are stable). This
// removes the unordered_map<const char*> lookup that previously dominated every
// entity_has_component / GetComponent call (it was the hottest engine symbol), and
// lets GetComponentArray return a raw pointer instead of copying a shared_ptr
// (atomic refcount churn) on the hot path.

// Monotonic counter assigning the next component type id. The first
// component_type_id<T>() call for each T claims the next slot.
inline ComponentType &component_type_counter() {
    static ComponentType next = 0;
    return next;
}
template <typename T>
inline ComponentType component_type_id() {
    static const ComponentType id = component_type_counter()++;
    return id;
}

class ComponentManager {
    public:
        template <typename T>
        void RegisterComponent() {
            ComponentType type = component_type_id<T>();
            assert(type < MAX_COMPONENTS && "More component types than MAX_COMPONENTS.");
            assert(!mComponentArrays[type] && "Registering component type more than once.");
            mComponentArrays[type] = std::make_shared<ComponentArray<T>>();
        }
        template <typename T>
        ComponentType GetComponentType() {
            return component_type_id<T>();
        }
        template <typename T>
        void AddComponent(Entity entity, T component) {
            GetComponentArray<T>()->InsertData(entity, component);
        }
        template <typename T>
        void RemoveComponent(Entity entity) {
            GetComponentArray<T>()->RemoveData(entity);
        }
        template <typename T>
        T &GetComponent(Entity entity) {
            return GetComponentArray<T>()->GetData(entity);
        }
        void EntityDestroyed(Entity entity) {
            for (auto const &component : mComponentArrays) {
                if (component) component->EntityDestroyed(entity);
            }
        }
        // Snapshot/restore for in-process game snapshots (see snapshot.h). The
        // type-id maps are process-constants after registration, so only the array
        // CONTENTS are copied; restore writes back INTO the existing arrays via
        // RestoreFrom (keeping array identity, never swapping pointers) so no code
        // that holds a raw array pointer is invalidated. Slot i lines up with the
        // stable component type id used everywhere else.
        std::vector<std::shared_ptr<IComponentArray>> SnapshotArrays() const {
            std::vector<std::shared_ptr<IComponentArray>> out;
            out.reserve(mComponentArrays.size());
            for (auto const &arr : mComponentArrays)
                out.push_back(arr ? arr->SnapshotClone() : nullptr);
            return out;
        }
        void RestoreArrays(const std::vector<std::shared_ptr<IComponentArray>> &snap) {
            for (size_t i = 0; i < mComponentArrays.size() && i < snap.size(); ++i) {
                if (mComponentArrays[i] && snap[i]) mComponentArrays[i]->RestoreFrom(*snap[i]);
            }
        }
    private:
        // Component arrays indexed by component type id (dense, no hashing).
        std::array<std::shared_ptr<IComponentArray>, MAX_COMPONENTS> mComponentArrays{};
        template <typename T>
        ComponentArray<T> *GetComponentArray() {
            ComponentType type = component_type_id<T>();
            assert(mComponentArrays[type] && "Component not registered before use.");
            // Raw pointer: the array outlives every call within a game, and is
            // re-fetched from the live manager each call (so it stays valid across
            // the per-game Init() that recreates the manager). No shared_ptr copy.
            return static_cast<ComponentArray<T> *>(mComponentArrays[type].get());
        }
};
#endif /* COMPONENT_MANAGER_H */
