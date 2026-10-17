#include "generative.h"

#include <algorithm>

namespace rmknow_test {

static std::string expand(const std::vector<std::pair<int, int>> &counts);
static void arrangements(std::string cards, std::vector<std::string> *out);

// The cards of `counts`, one byte per card (card ids fit a byte in the tests' small decks), sorted.
static std::string expand(const std::vector<std::pair<int, int>> &counts) {
    std::string s;
    for (const auto &c : counts) s.append(static_cast<size_t>(c.second), static_cast<char>(c.first & 0x7F));
    std::sort(s.begin(), s.end());
    return s;
}

// Every distinct order of `cards`.
static void arrangements(std::string cards, std::vector<std::string> *out) {
    std::sort(cards.begin(), cards.end());
    do {
        out->push_back(cards);
    } while (std::next_permutation(cards.begin(), cards.end()));
}

bool GenerativeModel::State::operator<(const State &o) const {
    if (library != o.library) return library < o.library;
    if (hand != o.hand) return hand < o.hand;
    if (side != o.side) return side < o.side;
    return aside < o.aside;
}

char GenerativeModel::code(int card) {
    return static_cast<char>(card & 0x7F);
}

std::string GenerativeModel::sorted_insert(const std::string &s, char c) {
    std::string out = s;
    out.insert(std::upper_bound(out.begin(), out.end(), c), c);
    return out;
}

bool GenerativeModel::remove_one(std::string &s, char c) {
    const size_t at = s.find(c);
    if (at == std::string::npos) return false;
    s.erase(at, 1);
    return true;
}

int GenerativeModel::count_of(const std::string &s, char c) {
    return static_cast<int>(std::count(s.begin(), s.end(), c));
}

void GenerativeModel::add(Dist &d, const State &s, double p) {
    if (p > 0) d[s] += p;
}

void GenerativeModel::check_cap(const Dist &d) {
    if (d.size() > cap_) over_ = true;
}

GenerativeModel::GenerativeModel(const std::vector<std::pair<int, int>> &main,
    const std::vector<std::pair<int, int>> &side, int library, int sideboard, uint64_t cap)
    : cap_(cap) {
    const std::string m = expand(main), sd = expand(side);
    if (static_cast<int>(m.size()) < library || static_cast<int>(sd.size()) < sideboard) return;
    std::vector<std::string> lib_orders, side_orders;
    arrangements(m, &lib_orders);
    arrangements(sd, &side_orders);
    const double p = 1.0 / static_cast<double>(lib_orders.size() * side_orders.size());
    for (const std::string &lo : lib_orders)
        for (const std::string &so : side_orders) {
            State s;
            s.library = lo.substr(0, static_cast<size_t>(library));
            s.side = so.substr(0, static_cast<size_t>(sideboard));
            std::sort(s.side.begin(), s.side.end());
            s.aside = lo.substr(static_cast<size_t>(library)) + so.substr(static_cast<size_t>(sideboard));
            std::sort(s.aside.begin(), s.aside.end());
            add(dist_, s, p);
            check_cap(dist_);
            if (over_) return;
        }
    lib_known_.assign(static_cast<size_t>(library), false);
}

void GenerativeModel::draw() {
    const bool known = !lib_known_.empty() && lib_known_[0];
    if (!lib_known_.empty()) lib_known_.erase(lib_known_.begin());
    Dist next;
    char top = 0;
    for (const auto &e : dist_) {
        State s = e.first;
        if (s.library.empty()) continue;
        top = s.library[0];
        if (!known) s.hand = sorted_insert(s.hand, s.library[0]);
        s.library.erase(0, 1);
        add(next, s, e.second);
    }
    dist_.swap(next);
    // A known top card is the same in every state left.
    if (known && top) pinned_ = sorted_insert(pinned_, top);
}

void GenerativeModel::cast_unseen(int card) {
    if (remove_one(pinned_, code(card))) return;
    Dist next;
    for (const auto &e : dist_) {
        State s = e.first;
        const int n = static_cast<int>(s.hand.size()), k = count_of(s.hand, code(card));
        if (n == 0 || k == 0) continue;
        remove_one(s.hand, code(card));
        add(next, s, e.second * k / n);
    }
    dist_.swap(next);
}

void GenerativeModel::cast_known(int card) {
    remove_one(known_, code(card));
}

void GenerativeModel::reveal_top(int card) {
    if (!lib_known_.empty()) lib_known_[0] = true;
    Dist next;
    for (const auto &e : dist_)
        if (!e.first.library.empty() && e.first.library[0] == code(card)) add(next, e.first, e.second);
    dist_.swap(next);
}

void GenerativeModel::reveal_hand(const std::vector<int> &cards) {
    std::string shown;
    for (int c : cards) shown.push_back(code(c));
    std::sort(shown.begin(), shown.end());
    Dist next;
    for (const auto &e : dist_) {
        std::string all = e.first.hand + pinned_ + known_;
        std::sort(all.begin(), all.end());
        if (all != shown) continue;
        State s = e.first;
        s.hand.clear();
        add(next, s, e.second);
    }
    dist_.swap(next);
    // Every hand card is known now: the shown ones without a handle.
    pinned_ = shown;
    for (char c : known_) remove_one(pinned_, c);
}

void GenerativeModel::put_back_unseen(bool bottom) {
    Dist next;
    for (const auto &e : dist_) {
        std::string pool = e.first.hand + pinned_;
        std::sort(pool.begin(), pool.end());
        const int n = static_cast<int>(pool.size());
        if (n == 0) continue;
        std::string kinds = pool;
        kinds.erase(std::unique(kinds.begin(), kinds.end()), kinds.end());
        for (char c : kinds) {
            State s = e.first;
            s.hand = pool;
            remove_one(s.hand, c);
            if (bottom)
                s.library.push_back(c);
            else
                s.library.insert(s.library.begin(), c);
            add(next, s, e.second * count_of(pool, c) / n);
        }
    }
    dist_.swap(next);
    pinned_.clear();
    if (bottom)
        lib_known_.push_back(false);
    else
        lib_known_.insert(lib_known_.begin(), false);
    check_cap(dist_);
}

void GenerativeModel::put_back_known(int card, bool bottom) {
    remove_one(known_, code(card));
    if (bottom)
        lib_known_.push_back(true);
    else
        lib_known_.insert(lib_known_.begin(), true);
    Dist next;
    for (const auto &e : dist_) {
        State s = e.first;
        if (bottom)
            s.library.push_back(code(card));
        else
            s.library.insert(s.library.begin(), code(card));
        add(next, s, e.second);
    }
    dist_.swap(next);
}

void GenerativeModel::scry_bottom_one() {
    // The viewer cannot tell the two apart: neither position stays known.
    if (lib_known_.size() >= 2) {
        lib_known_[0] = false;
        lib_known_.erase(lib_known_.begin() + 1);
        lib_known_.push_back(false);
    }
    Dist next;
    for (const auto &e : dist_) {
        if (e.first.library.size() < 2) continue;
        for (size_t at = 0; at < 2; ++at) {
            State s = e.first;
            const char c = s.library[at];
            s.library.erase(at, 1);
            s.library.push_back(c);
            add(next, s, e.second / 2);
        }
    }
    dist_.swap(next);
    check_cap(dist_);
}

void GenerativeModel::mill_top(int card) {
    if (!lib_known_.empty()) lib_known_.erase(lib_known_.begin());
    Dist next;
    for (const auto &e : dist_) {
        if (e.first.library.empty() || e.first.library[0] != code(card)) continue;
        State s = e.first;
        s.library.erase(0, 1);
        add(next, s, e.second);
    }
    dist_.swap(next);
}

void GenerativeModel::shuffle() {
    lib_known_.assign(lib_known_.size(), false);
    Dist next;
    for (const auto &e : dist_) {
        std::vector<std::string> orders;
        arrangements(e.first.library, &orders);
        for (const std::string &o : orders) {
            State s = e.first;
            s.library = o;
            add(next, s, e.second / static_cast<double>(orders.size()));
        }
        check_cap(next);
        if (over_) return;
    }
    dist_.swap(next);
}

void GenerativeModel::wish(int card) {
    Dist next;
    for (const auto &e : dist_) {
        State s = e.first;
        const int n = static_cast<int>(s.side.size()), k = count_of(s.side, code(card));
        if (n == 0 || k == 0) continue;
        remove_one(s.side, code(card));
        add(next, s, e.second * k / n);
    }
    dist_.swap(next);
    known_ += code(card);
}

void GenerativeModel::from_set_aside(int card) {
    Dist next;
    for (const auto &e : dist_) {
        State s = e.first;
        const int n = static_cast<int>(s.aside.size()), k = count_of(s.aside, code(card));
        if (n == 0 || k == 0) continue;
        remove_one(s.aside, code(card));
        add(next, s, e.second * k / n);
    }
    dist_.swap(next);
    known_ += code(card);
}

double GenerativeModel::probability() const {
    double p = 0;
    for (const auto &e : dist_) p += e.second;
    return p;
}

}  // namespace rmknow_test
