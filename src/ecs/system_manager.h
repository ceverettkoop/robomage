#ifndef SYSTEM_MANAGER_H
#define SYSTEM_MANAGER_H
#include "system.h"
#include "component.h"
#include "entity.h"
#include <cassert>
#include <memory>
#include <set>
#include <unordered_map>
// credit https://austinmorlan.com/posts/entity_component_system/
class SystemManager
{
public:
	template<typename T>
	std::shared_ptr<T> RegisterSystem()
	{
		const char* typeName = typeid(T).name();
		assert(mSystems.find(typeName) == mSystems.end() && "Registering system more than once.");
		auto system = std::make_shared<T>();
		mSystems.insert({typeName, system});
		return system;
	}
	template<typename T>
	void SetSignature(Signature signature)
	{
		const char* typeName = typeid(T).name();
		assert(mSystems.find(typeName) != mSystems.end() && "System used before registered.");
		mSignatures.insert({typeName, signature});
	}
	void EntityDestroyed(Entity entity)
	{
		for (auto const& pair : mSystems)
		{
			auto const& system = pair.second;
			system->mEntities.erase(entity);
		}
	}
	void EntitySignatureChanged(Entity entity, Signature entitySignature)
	{
		for (auto const& pair : mSystems)
		{
			auto const& type = pair.first;
			auto const& system = pair.second;
			auto const& systemSignature = mSignatures[type];
			if ((entitySignature & systemSignature) == systemSignature)
			{
				system->mEntities.insert(entity);
			}
			else
			{
				system->mEntities.erase(entity);
			}
		}
	}
	// Snapshot/restore each registered system's entity set for in-process game
	// snapshots (see snapshot.h). Keyed by the same const char* type-name pointer
	// the map uses (stable per process); the per-system sets are independent, so
	// iteration order is irrelevant. Only the sets are copied — the system objects
	// themselves stay in place.
	std::unordered_map<const char*, std::set<Entity>> snapshot_systems() const
	{
		std::unordered_map<const char*, std::set<Entity>> out;
		for (auto const& pair : mSystems) out[pair.first] = pair.second->mEntities;
		return out;
	}
	void restore_systems(const std::unordered_map<const char*, std::set<Entity>>& snap)
	{
		for (auto const& pair : mSystems)
		{
			auto it = snap.find(pair.first);
			if (it != snap.end()) pair.second->mEntities = it->second;
		}
	}
private:
	std::unordered_map<const char*, Signature> mSignatures{};
	std::unordered_map<const char*, std::shared_ptr<System>> mSystems{};
};
#endif /* SYSTEM_MANAGER_H */