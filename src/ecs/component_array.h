#ifndef COMPONENT_ARRAY_H
#define COMPONENT_ARRAY_H
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <typeinfo>
#include "entity.h"
// credit https://austinmorlan.com/posts/entity_component_system/
class IComponentArray {
    public:
        virtual ~IComponentArray() = default;
        virtual void EntityDestroyed(Entity entity) = 0;
};
template <typename T>
class ComponentArray : public IComponentArray {
    public:
        // Entity ids are dense and bounded by MAX_ENTITIES, so the entity->dense-index
        // and index->entity maps are flat arrays rather than unordered_maps. This turns
        // every GetData/has-component access into a single array index instead of a hash
        // lookup (the entity->index find was the hottest engine symbol after the
        // component-type lookup was removed). NO_INDEX marks an entity with no component.
        static constexpr uint32_t NO_INDEX = 0xFFFFFFFFu;
        ComponentArray() { mEntityToIndex.fill(NO_INDEX); }

        void InsertData(Entity entity, T component) {
            assert(mEntityToIndex[entity] == NO_INDEX &&
                   "Component added to same entity more than once.");
            // Put new entry at end
            uint32_t newIndex = static_cast<uint32_t>(mSize);
            mEntityToIndex[entity] = newIndex;
            mIndexToEntity[newIndex] = entity;
            mComponentArray[newIndex] = component;
            ++mSize;
        }
        void RemoveData(Entity entity) {
            assert(mEntityToIndex[entity] != NO_INDEX && "Removing non-existent component.");
            // Copy element at end into deleted element's place to maintain density
            uint32_t indexOfRemovedEntity = mEntityToIndex[entity];
            uint32_t indexOfLastElement = static_cast<uint32_t>(mSize - 1);
            mComponentArray[indexOfRemovedEntity] = mComponentArray[indexOfLastElement];
            // Update map to point to moved spot
            Entity entityOfLastElement = mIndexToEntity[indexOfLastElement];
            mEntityToIndex[entityOfLastElement] = indexOfRemovedEntity;
            mIndexToEntity[indexOfRemovedEntity] = entityOfLastElement;
            mEntityToIndex[entity] = NO_INDEX;
            --mSize;
        }
        T &GetData(Entity entity) {
            uint32_t idx = mEntityToIndex[entity];
            if (idx == NO_INDEX) {
                fprintf(stderr, "CRASH: GetData<%s> on entity %u\n",
                        typeid(T).name(), entity);
                void *bt[20];
                int n = backtrace(bt, 20);
                backtrace_symbols_fd(bt, n, 2);
                abort();
            }
            return mComponentArray[idx];
        }
        void EntityDestroyed(Entity entity) override {
            if (mEntityToIndex[entity] != NO_INDEX) {
                RemoveData(entity);
            }
        }

    private:
        std::array<T, MAX_ENTITIES> mComponentArray{};
        // entity id -> dense index (NO_INDEX = absent); dense index -> entity id.
        std::array<uint32_t, MAX_ENTITIES> mEntityToIndex{};
        std::array<Entity, MAX_ENTITIES> mIndexToEntity{};
        size_t mSize{};
};
#endif /* COMPONENT_ARRAY_H */
