#ifndef OBJECT_REF_H
#define OBJECT_REF_H

#include <cstdint>
#include <map>
#include <vector>

#include "ecs/entity.h"

// A reference to one game OBJECT across time (CR 400.7). An entity id survives zone changes and
// is eventually reissued to an unrelated object, so an Entity alone cannot say whether it still
// names the object it was taken from. ObjectRef pairs the id with the object's identity stamp —
// Zone::obj_gen, fresh on every zone entry (Orderer::add_to_zone) and never reused — taken when
// the reference is made:
//   - get() is the entity while it is still that object, else 0: it changed zones (a new object),
//     ceased to exist, or its id now belongs to something else.
//   - lki_entity() is the raw id, for last-known-information lookups (lki_for / departed_lki_for,
//     effective_* on a departed object), logging, and matching a reference against the entity it
//     was taken of (bookkeeping that clears links to an object) — never to act on the object.
// Every reference kept across a zone change, a resolution boundary or a turn is an ObjectRef;
// plain Entity is for values that live inside one uninterrupted step.
//
// Players (and other entities with no Zone) never become new objects: of() gives them gen 0 and
// get() returns them as long as they still have no Zone.
//
// CR 400.7j: within one resolution, an object the resolving effect moved can still be found by
// the rest of that effect. get() therefore also resolves a reference to an earlier incarnation
// of an object moved while a follow window is open (FollowWindowScope, the stack resolution
// frame) — see note_object_moved.
//
// Trivially copyable, so component and whole-Game snapshots copy it by value.
struct ObjectRef {
    Entity e = 0;
    uint64_t gen = 0;

    static ObjectRef of(Entity e);  // stamps `e`'s identity now; of(0) is the empty ref
    Entity get() const;             // e if still the same object, else 0
    Entity lki_entity() const { return e; }  // raw id: LKI lookups, logging, link bookkeeping
    bool empty() const { return e == 0; }    // no object was ever referenced
    explicit operator bool() const { return get() != 0; }
    bool operator==(const ObjectRef &o) const { return e == o.e && gen == o.gen; }
    bool operator!=(const ObjectRef &o) const { return !(*this == o); }
};

// Refs to each of `entities`, stamped now.
std::vector<ObjectRef> refs_of(const std::vector<Entity> &entities);
// The entities `refs` still name (get() != 0), in order.
std::vector<Entity> live_entities(const std::vector<ObjectRef> &refs);
// True if some ref in `refs` names `e` as the object it is now.
bool refs_contain(const std::vector<ObjectRef> &refs, Entity e);
// Remove every ref to entity id `e` (whatever object it named).
void erase_refs_to(std::vector<ObjectRef> &refs, Entity e);

// An ordered set of objects keyed by entity id (the std::set<Entity> iteration order), each
// member remembered with its identity stamp: a member that became a new object, or whose id was
// reissued, is no longer contained.
class ObjectSet {
    public:
        void insert(Entity e) { members[e] = ObjectRef::of(e); }
        void erase(Entity e) { members.erase(e); }
        void clear() { members.clear(); }
        size_t count(Entity e) const;
        bool empty() const { return live().empty(); }
        std::vector<Entity> live() const;  // members that are still the same object

    private:
        std::map<Entity, ObjectRef> members;
};

// A map from objects to T keyed by entity id, each key remembered with its identity stamp. An
// entry whose object is gone (it changed zones, or its id was reissued) is invisible to find()
// and live_keys(); operator[] replaces such an entry with a fresh default one.
template <class T>
class ObjectMap {
    public:
        T *find(Entity e) {
            auto it = entries.find(e);
            return it != entries.end() && it->second.ref.get() == e ? &it->second.value : nullptr;
        }
        const T *find(Entity e) const {
            auto it = entries.find(e);
            return it != entries.end() && it->second.ref.get() == e ? &it->second.value : nullptr;
        }
        size_t count(Entity e) const { return find(e) ? 1 : 0; }
        T &operator[](Entity e) {
            auto it = entries.find(e);
            if (it == entries.end() || it->second.ref.get() != e)
                it = entries.insert_or_assign(e, Entry{ObjectRef::of(e), T{}}).first;
            return it->second.value;
        }
        void erase(Entity e) { entries.erase(e); }
        void clear() { entries.clear(); }
        // Keys still naming the object they were recorded for, in entity-id order.
        std::vector<Entity> live_keys() const {
            std::vector<Entity> out;
            for (const auto &kv : entries)
                if (kv.second.ref.get() == kv.first) out.push_back(kv.first);
            return out;
        }
        // Erase every entry (live or not) for which pred(entity, value) is true.
        template <class Pred>
        void erase_if(Pred pred) {
            for (auto it = entries.begin(); it != entries.end();) {
                if (pred(it->first, it->second.value))
                    it = entries.erase(it);
                else
                    ++it;
            }
        }

    private:
        struct Entry {
            ObjectRef ref;
            T value;
        };
        std::map<Entity, Entry> entries;
};

// CR 400.7j follow window. While open, Orderer records each object it moves (its identity stamp
// before the first move in the window), and ObjectRef::get() still resolves a reference to any
// incarnation of that object from the window. Opened for one resolution: the stack resolution
// frame (StackManager) and every resolution driven outside it (FollowWindowScope).
void open_follow_window();
void close_follow_window();
bool follow_window_open();
// Called by Orderer as `e` enters a new zone, with the stamp it had before the move.
void note_object_moved(Entity e, uint64_t old_gen);

// Opens a follow window for its lifetime unless one is already open.
class FollowWindowScope {
    public:
        FollowWindowScope() : opened(!follow_window_open()) {
            if (opened) open_follow_window();
        }
        ~FollowWindowScope() {
            if (opened) close_follow_window();
        }
        FollowWindowScope(const FollowWindowScope &) = delete;
        FollowWindowScope &operator=(const FollowWindowScope &) = delete;

    private:
        bool opened;
};

#endif /* OBJECT_REF_H */
