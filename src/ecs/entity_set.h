#ifndef ENTITY_SET_H
#define ENTITY_SET_H

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include "entity.h"

// A set of entity ids below MAX_ENTITIES, one bit per id: a system's mEntities. Iteration visits
// the ids in ascending order, as a std::set<Entity> does, and reads the bits as it goes, so the set
// may change while it is iterated with std::set's outcome: an id added above the current one is
// visited, one added below is not, a removed id is not visited (removing the current id is allowed
// too). Insertion, removal and lookup are a bit operation; a scan skips empty words.
class EntitySet {
    public:
        static constexpr size_t WORDS = (MAX_ENTITIES + 63) / 64;

        class const_iterator {
            public:
                using iterator_category = std::forward_iterator_tag;
                using value_type = Entity;
                using difference_type = std::ptrdiff_t;
                using pointer = const Entity *;
                using reference = Entity;

                const_iterator() = default;
                const_iterator(const EntitySet *set, size_t at) : set_(set), at_(at) {}
                Entity operator*() const { return static_cast<Entity>(at_); }
                const_iterator &operator++() {
                    at_ = set_->next_from(at_ + 1);
                    return *this;
                }
                const_iterator operator++(int) {
                    const_iterator old = *this;
                    ++*this;
                    return old;
                }
                bool operator==(const const_iterator &o) const { return at_ == o.at_; }
                bool operator!=(const const_iterator &o) const { return at_ != o.at_; }

            private:
                const EntitySet *set_ = nullptr;
                size_t at_ = MAX_ENTITIES;
        };
        using iterator = const_iterator;
        using value_type = Entity;
        using key_type = Entity;
        using size_type = size_t;

        const_iterator begin() const { return const_iterator(this, next_from(0)); }
        const_iterator end() const { return const_iterator(this, MAX_ENTITIES); }
        size_t size() const { return size_; }
        bool empty() const { return size_ == 0; }
        size_t count(Entity e) const { return e < MAX_ENTITIES && ((bits_[e / 64] >> (e % 64)) & 1u) ? 1 : 0; }
        const_iterator find(Entity e) const { return count(e) ? const_iterator(this, e) : end(); }
        std::pair<const_iterator, bool> insert(Entity e) {
            const uint64_t bit = uint64_t{1} << (e % 64);
            const bool added = !(bits_[e / 64] & bit);
            if (added) {
                bits_[e / 64] |= bit;
                ++size_;
            }
            return std::make_pair(const_iterator(this, e), added);
        }
        size_t erase(Entity e) {
            if (!count(e)) return 0;
            bits_[e / 64] &= ~(uint64_t{1} << (e % 64));
            --size_;
            return 1;
        }
        void clear() {
            for (uint64_t &w : bits_) w = 0;
            size_ = 0;
        }
        bool operator==(const EntitySet &o) const {
            for (size_t i = 0; i < WORDS; ++i)
                if (bits_[i] != o.bits_[i]) return false;
            return true;
        }
        bool operator!=(const EntitySet &o) const { return !(*this == o); }

    private:
        // The first id at or above `from` in the set, else MAX_ENTITIES.
        size_t next_from(size_t from) const {
            if (from >= MAX_ENTITIES) return MAX_ENTITIES;
            size_t w = from / 64;
            uint64_t word = bits_[w] & (~uint64_t{0} << (from % 64));
            while (true) {
                if (word) {
                    const size_t at = w * 64 + static_cast<size_t>(__builtin_ctzll(word));
                    return at < MAX_ENTITIES ? at : MAX_ENTITIES;
                }
                if (++w == WORDS) return MAX_ENTITIES;
                word = bits_[w];
            }
        }
        uint64_t bits_[WORDS] = {};
        size_t size_ = 0;
};

#endif /* ENTITY_SET_H */
