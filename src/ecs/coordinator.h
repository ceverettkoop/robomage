#ifndef COORDINATOR_H
#define COORDINATOR_H

#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include "component.h"
#include "component_manager.h"
#include "entity_manager.h"
#include "event_manager.h"
#include "system_manager.h"

// credit https://austinmorlan.com/posts/entity_component_system/

// Deep-copied snapshot of the whole ECS (the four managers' state), taken/restored
// as a unit for in-process game snapshots. See snapshot.h; not used on any hot path.
struct EcsSnapshot {
    std::vector<std::shared_ptr<IComponentArray>> component_arrays;
    EntityManager::EntityManagerState entity_state;
    std::unordered_map<const char *, std::set<Entity>> system_entities;
    std::vector<Event> pending_events;
};

class Coordinator {
    public:
        Coordinator(){
        };

        static Coordinator& global(){
            return *singleton;
        }

        void Init() {
            singleton = this;
            mComponentManager = std::make_unique<ComponentManager>();
            mEntityManager = std::make_unique<EntityManager>();
            mEventManager = std::make_unique<EventManager>();
            mSystemManager = std::make_unique<SystemManager>();
        }
        // Entity methods
        Entity CreateEntity() { return mEntityManager->CreateEntity(); }
        void DestroyEntity(Entity entity) {
            mEntityManager->DestroyEntity(entity);
            mComponentManager->EntityDestroyed(entity);
            mSystemManager->EntityDestroyed(entity);
        }
        // Component methods
        template <typename T>
        void RegisterComponent() {
            mComponentManager->RegisterComponent<T>();
        }
        template <typename T>
        void AddComponent(Entity entity, T component) {
            mComponentManager->AddComponent<T>(entity, component);
            auto signature = mEntityManager->GetSignature(entity);
            signature.set(mComponentManager->GetComponentType<T>(), true);
            mEntityManager->SetSignature(entity, signature);
            mSystemManager->EntitySignatureChanged(entity, signature);
        }
        template <typename T>
        void RemoveComponent(Entity entity) {
            mComponentManager->RemoveComponent<T>(entity);
            auto signature = mEntityManager->GetSignature(entity);
            signature.set(mComponentManager->GetComponentType<T>(), false);
            mEntityManager->SetSignature(entity, signature);
            mSystemManager->EntitySignatureChanged(entity, signature);
        }
        template <typename T>
        T &GetComponent(Entity entity) {
            return mComponentManager->GetComponent<T>(entity);
        }
        template <typename T>
        ComponentType GetComponentType() {
            return mComponentManager->GetComponentType<T>();
        }
        
        // Upper bound for linear entity scans: iterate [0, GetMaxIssuedEntity()) instead of [0, MAX_ENTITIES).
        Entity GetMaxIssuedEntity() const { return mEntityManager->GetMaxIssuedEntity(); }

        template <typename T>
        bool entity_has_component(Entity entity){
            // component_type_id<T>() is a memoized integer (no typeid/hash lookup);
            // read the signature bitset by reference and test the bit directly.
            return mEntityManager->GetSignatureRef(entity).test(component_type_id<T>());
        }

        // System methods
        template <typename T>
        std::shared_ptr<T> RegisterSystem() {
            return mSystemManager->RegisterSystem<T>();
        }
        template <typename T>
        void SetSystemSignature(Signature signature) {
            mSystemManager->SetSignature<T>(signature);
        }
        // Event methods
        void AddEventListener(EventId eventId, std::function<void(Event &)> const &listener) {
            mEventManager->AddListener(eventId, listener);
        }
        void SendEvent(Event &event) { mEventManager->SendEvent(event); }
        void SendEvent(EventId eventId) { mEventManager->SendEvent(eventId); }
        std::vector<Event> drain_pending_events() { return mEventManager->drain_pending_events(); }

        // Snapshot/restore the whole ECS as a unit (see snapshot.h). Member functions
        // because the four managers are private unique_ptrs; each delegates to the
        // manager's own snapshot/restore. Restore writes back into the existing
        // managers (no re-registration), so component type ids stay valid.
        void snapshot_to(EcsSnapshot &out) const {
            out.component_arrays = mComponentManager->SnapshotArrays();
            out.entity_state = mEntityManager->snapshot_state();
            out.system_entities = mSystemManager->snapshot_systems();
            out.pending_events = mEventManager->snapshot_pending();
        }
        void restore_from(const EcsSnapshot &in) {
            mComponentManager->RestoreArrays(in.component_arrays);
            mEntityManager->restore_state(in.entity_state);
            mSystemManager->restore_systems(in.system_entities);
            mEventManager->restore_pending(in.pending_events);
        }

    private:
        std::unique_ptr<ComponentManager> mComponentManager;
        std::unique_ptr<EntityManager> mEntityManager;
        std::unique_ptr<EventManager> mEventManager;
        std::unique_ptr<SystemManager> mSystemManager;
        static Coordinator *singleton;
};

extern Coordinator global_coordinator;

#endif /* COORDINATOR_H */
