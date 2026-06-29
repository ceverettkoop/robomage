#ifndef COUNTER_MAP_H
#define COUNTER_MAP_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

// Small flat map for typed counters (P1P1, M1M1, LOYALTY, POISON, ENERGY, keyword
// counters — in practice 0-4 live entries on a permanent or player). Drop-in for the
// subset of std::map<std::string,int> the counter stores use: operator[], find,
// erase(key), and range-for over std::pair<std::string,int>.
//
// Entries are kept sorted by key, so iteration order is identical to the std::map it
// replaces (behaviour is byte-for-byte unchanged). A linear scan over a handful of
// entries beats a red-black tree here and removes the std::_Rb_tree_increment cost the
// profiler showed in the per-pass continuous-effects / state-based-action sweeps, which
// walk every permanent's counters on every pass.
class CounterMap {
   public:
    using value_type = std::pair<std::string, int>;
    using iterator = std::vector<value_type>::iterator;
    using const_iterator = std::vector<value_type>::const_iterator;

    iterator begin() { return v_.begin(); }
    iterator end() { return v_.end(); }
    const_iterator begin() const { return v_.begin(); }
    const_iterator end() const { return v_.end(); }

    bool empty() const { return v_.empty(); }
    std::size_t size() const { return v_.size(); }
    void clear() { v_.clear(); }

    iterator find(const std::string &key) {
        for (auto it = v_.begin(); it != v_.end() && it->first <= key; ++it)
            if (it->first == key) return it;
        return v_.end();
    }
    const_iterator find(const std::string &key) const {
        for (auto it = v_.begin(); it != v_.end() && it->first <= key; ++it)
            if (it->first == key) return it;
        return v_.end();
    }

    // std::map::operator[] semantics: returns a reference to the value for `key`,
    // inserting a 0-valued entry (in sorted position) if absent.
    int &operator[](const std::string &key) {
        auto it = v_.begin();
        while (it != v_.end() && it->first < key) ++it;
        if (it != v_.end() && it->first == key) return it->second;
        it = v_.insert(it, value_type(key, 0));
        return it->second;
    }

    std::size_t erase(const std::string &key) {
        for (auto it = v_.begin(); it != v_.end() && it->first <= key; ++it) {
            if (it->first == key) {
                v_.erase(it);
                return 1;
            }
        }
        return 0;
    }

   private:
    std::vector<value_type> v_;
};

#endif /* COUNTER_MAP_H */
