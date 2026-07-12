#ifndef ENTITY_MANAGER_H
#define ENTITY_MANAGER_H
#include <cassert>
#include <queue>
#include <array>
#include "component.h"
#include "entity.h"

// credit https://austinmorlan.com/posts/entity_component_system/

class EntityManager {
    public:
        EntityManager() {
            // Initialize the queue with all possible entity IDs. Entity 0 is reserved
            // as the universal "null entity" sentinel — many fields (ability targets,
            // attack_target, equipped_by, etc.) default to 0 to mean "none", and code
            // throughout treats `== 0` as "no entity". Issuing 0 to a real entity (it
            // would otherwise go to Player A, the first entity created) collides with
            // that sentinel and makes that entity untargetable. Start issuing at 1.
            for (Entity entity = 1; entity < MAX_ENTITIES; ++entity) {
                mAvailableEntities.push(entity);
            }
        }
        Entity CreateEntity() {
            assert(mLivingEntityCount < MAX_ENTITIES && "Too many entities in existence.");
            // Take an ID from the front of the queue
            Entity id = mAvailableEntities.front();
            mAvailableEntities.pop();
            ++mLivingEntityCount;
            if (id >= mMaxIssuedEntity) mMaxIssuedEntity = id + 1;
            return id;
        }
        Entity GetMaxIssuedEntity() const { return mMaxIssuedEntity; }
        void DestroyEntity(Entity entity) {
            assert(entity < MAX_ENTITIES && "Entity out of range.");
            // Invalidate the destroyed entity's signature
            mSignatures[entity].reset();
            // Put the destroyed ID at the back of the queue
            mAvailableEntities.push(entity);
            --mLivingEntityCount;
        }
        void SetSignature(Entity entity, Signature signature) {
            assert(entity < MAX_ENTITIES && "Entity out of range.");
            // Put this entity's signature into the array
            mSignatures[entity] = signature;
        }
        Signature GetSignature(Entity entity) {
            assert(entity < MAX_ENTITIES && "Entity out of range.");
            // Get this entity's signature from the array
            return mSignatures[entity];
        }
        // Read-only access without copying the bitset (hot path: entity_has_component).
        const Signature &GetSignatureRef(Entity entity) const {
            assert(entity < MAX_ENTITIES && "Entity out of range.");
            return mSignatures[entity];
        }
        // Snapshot/restore of the whole entity allocator for in-process game
        // snapshots (see snapshot.h). std::queue and std::array both copy by value.
        struct EntityManagerState {
            std::queue<Entity> availableEntities;
            std::array<Signature, MAX_ENTITIES> signatures;
            uint32_t livingEntityCount;
            Entity maxIssuedEntity;
        };
        EntityManagerState snapshot_state() const {
            return {mAvailableEntities, mSignatures, mLivingEntityCount, mMaxIssuedEntity};
        }
        void restore_state(const EntityManagerState &s) {
            mAvailableEntities = s.availableEntities;
            mSignatures = s.signatures;
            mLivingEntityCount = s.livingEntityCount;
            mMaxIssuedEntity = s.maxIssuedEntity;
        }
    private:
        // Queue of unused entity IDs
        std::queue<Entity> mAvailableEntities{};
        // Array of signatures where the index corresponds to the entity ID
        std::array<Signature, MAX_ENTITIES> mSignatures{};
        // Total living entities - used to keep limits on how many exist
        uint32_t mLivingEntityCount{};
        // Highest entity ID ever issued + 1; used to bound linear scans
        Entity mMaxIssuedEntity{};
};
#endif /* ENTITY_MANAGER_H */