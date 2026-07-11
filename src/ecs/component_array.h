#ifndef COMPONENT_ARRAY_H
#define COMPONENT_ARRAY_H
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <execinfo.h>
#include <memory>
#include <type_traits>
#include <typeinfo>
#include "entity.h"
// credit https://austinmorlan.com/posts/entity_component_system/
class IComponentArray {
    public:
        virtual ~IComponentArray() = default;
        virtual void EntityDestroyed(Entity entity) = 0;
        // Deep-copy this array into a fresh IComponentArray for a game snapshot, and
        // restore a previously-cloned array's contents back into this one (see snapshot.h).
        virtual std::shared_ptr<IComponentArray> SnapshotClone() const = 0;
        virtual void RestoreFrom(const IComponentArray &src) = 0;
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

        std::shared_ptr<IComponentArray> SnapshotClone() const override {
            auto clone = std::make_shared<ComponentArray<T>>();
            clone->mEntityToIndex = mEntityToIndex;  // flat POD maps: plain copy
            clone->mIndexToEntity = mIndexToEntity;
            clone->mSize = mSize;
            // Copy ONLY the live dense slots [0, mSize). Slots >= mSize hold stale,
            // heap-heavy leftovers from RemoveData's swap-with-last and are never
            // read, so copying all MAX_ENTITIES would dominate the snapshot cost for
            // nothing. memcpy is wrong here — components own strings/vectors/maps —
            // so this is an element-wise copy-assign.
            for (size_t i = 0; i < mSize; ++i) clone->mComponentArray[i] = mComponentArray[i];
            return clone;
        }
        void RestoreFrom(const IComponentArray &src) override {
            const auto &s = static_cast<const ComponentArray<T> &>(src);
            mEntityToIndex = s.mEntityToIndex;
            mIndexToEntity = s.mIndexToEntity;
            mSize = s.mSize;
            // Restore only the live slots. Restoring fewer live slots than this array
            // previously held is sound: slots beyond the restored mSize are never read
            // (mEntityToIndex is fully restored, and InsertData reassigns a slot before
            // any read), so leftover stale contents there are harmless.
            for (size_t i = 0; i < mSize; ++i) mComponentArray[i] = s.mComponentArray[i];
        }

    private:
        // A future non-copyable component member would make snapshot copies silently
        // wrong; fail the build loudly instead.
        static_assert(std::is_copy_assignable_v<T>,
                      "ComponentArray<T> snapshot/restore requires copy-assignable T");
        std::array<T, MAX_ENTITIES> mComponentArray{};
        // entity id -> dense index (NO_INDEX = absent); dense index -> entity id.
        std::array<uint32_t, MAX_ENTITIES> mEntityToIndex{};
        std::array<Entity, MAX_ENTITIES> mIndexToEntity{};
        size_t mSize{};
};
#endif /* COMPONENT_ARRAY_H */
