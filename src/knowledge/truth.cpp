#include "truth.h"

namespace knowledge {

static int count_of(const std::map<int, int> &m, int card);
static bool pool_contents(const KPoolRow &pool, const Truth &truth, std::map<int, int> *out, size_t *start);
static void violation(TruthCheck *out, const std::string &what);
static void check_lists(const KRows &rows, const Truth &truth, TruthCheck *out);

static int count_of(const std::map<int, int> &m, int card) {
    const auto it = m.find(card);
    return it == m.end() ? 0 : it->second;
}

static void violation(TruthCheck *out, const std::string &what) {
    out->errors.push_back(what);
}

bool truth_from_snapshot(const Snapshot &omni, Truth *out) {
    *out = Truth{};
    std::map<int, int> lib_pos[2];
    for (const SnapObject &o : omni.objects) {
        if (o.owner == NO_SEAT || o.token) continue;
        if (o.card != NO_CARD) ++out->owned[o.owner][o.card];
        if (o.zone == Zn::HAND) ++out->hand[o.owner][o.card];
        if (o.zone == Zn::SIDEBOARD) ++out->sideboard[o.owner][o.card];
        if (o.zone == Zn::LIBRARY) lib_pos[o.owner][o.pos] = o.card;
    }
    for (int p = 0; p < 2; ++p)
        for (const auto &e : lib_pos[p]) {
            if (e.first != static_cast<int>(out->library[p].size())) return false;
            out->library[p].push_back(e.second);
        }
    return true;
}

// The true contents of `pool` (and, for a library range, its first position from the top); false
// when the truth cannot say (a set-aside pool, a face-down card the truth does not know by handle, a
// range outside the library).
static bool pool_contents(const KPoolRow &pool, const Truth &truth, std::map<int, int> *out, size_t *start) {
    out->clear();
    *start = 0;
    if (pool.owner != 0 && pool.owner != 1) return false;
    const int p = pool.owner;
    switch (pool.kind) {
        case rmstate::PoolKind::HAND:
            *out = truth.hand[p];
            return true;
        case rmstate::PoolKind::SIDEBOARD:
            *out = truth.sideboard[p];
            return true;
        case rmstate::PoolKind::LIBRARY_RANGE: {
            const std::vector<int> &lib = truth.library[p];
            if (!pool.range.in_library() || static_cast<size_t>(pool.range.pos) + pool.size > lib.size()) return false;
            *start = pool.range.bottom() ? lib.size() - pool.range.pos - pool.size : pool.range.pos;
            for (size_t k = *start; k < *start + pool.size; ++k) ++(*out)[lib[k]];
            return true;
        }
        case rmstate::PoolKind::FACE_DOWN: {
            if (!truth.face_down_known) return false;
            const auto it = truth.face_down.find(pool.obj);
            if (it == truth.face_down.end()) return false;
            (*out)[it->second] = 1;
            return true;
        }
        case rmstate::PoolKind::SET_ASIDE:
            return false;
    }
    return false;
}

void check_truth(
    const KRows &rows, const Truth &truth, int viewer, const Universe *const universe[2], TruthCheck *out) {
    const size_t n_pools = rows.pools.size();
    std::vector<std::map<int, int>> contents(n_pools);
    std::vector<size_t> start(n_pools, 0);
    std::vector<bool> checkable(n_pools, false);
    for (size_t i = 0; i < n_pools; ++i) {
        const KPoolRow &pool = rows.pools[i];
        checkable[i] = pool_contents(pool, truth, &contents[i], &start[i]);
        if (!checkable[i]) {
            if (pool.kind == rmstate::PoolKind::LIBRARY_RANGE ||
                (pool.kind == rmstate::PoolKind::FACE_DOWN && truth.face_down_known)) {
                ++out->facts;
                violation(out, "pool " + std::to_string(i) + " (kind " + std::to_string(static_cast<int>(pool.kind)) +
                                   ") names no such place in the true world");
            }
            continue;
        }
        ++out->facts;
        size_t size = 0;
        for (const auto &c : contents[i]) size += static_cast<size_t>(c.second);
        if (size != pool.size)
            violation(out, "pool " + std::to_string(i) + " has size " + std::to_string(pool.size) + ", truly " +
                               std::to_string(size));
    }
    std::vector<std::map<int, std::pair<int, int>>> listed(n_pools);
    for (const KPoolCardRow &c : rows.pool_cards) {
        if (c.pool >= n_pools) {
            ++out->facts;
            violation(out, "a k_pool_card row names pool " + std::to_string(c.pool) + " of " + std::to_string(n_pools));
            continue;
        }
        listed[c.pool][c.card] = std::make_pair(c.min, c.max);
        if (!checkable[c.pool]) continue;
        ++out->facts;
        const int n = count_of(contents[c.pool], c.card);
        if (n < c.min || n > c.max)
            violation(out, "pool " + std::to_string(c.pool) + " holds " + std::to_string(n) + " of card " +
                               std::to_string(c.card) + ", reported " + std::to_string(c.min) + "-" +
                               std::to_string(c.max));
    }
    for (size_t i = 0; i < n_pools; ++i) {
        if (!checkable[i]) continue;
        ++out->facts;
        const Universe *uni = universe[rows.pools[i].owner];
        for (const auto &c : contents[i])
            if (listed[i].find(c.first) == listed[i].end() && uni && uni->contains(c.first))
                violation(out,
                    "pool " + std::to_string(i) + " holds card " + std::to_string(c.first) + ", reported impossible");
    }
    for (const KPositionRow &pos : rows.positions) {
        ++out->facts;
        if (pos.owner != 0 && pos.owner != 1) {
            violation(out, "a k_position row without its owner");
            continue;
        }
        const std::vector<int> &lib = truth.library[pos.owner];
        const size_t at = pos.at.bottom() ? lib.size() - 1 - pos.at.pos : pos.at.pos;
        if (!pos.at.in_library() || pos.at.span != 1 || pos.at.pos >= lib.size()) {
            violation(out, "library position " + std::to_string(pos.at.pos) + " is not in the true library of " +
                               std::to_string(lib.size()));
            continue;
        }
        const int card = lib[at];
        if (pos.card != NO_CARD) {
            if (card != pos.card)
                violation(out, "library position " + std::to_string(at) + " is reported card " +
                                   std::to_string(pos.card) + ", truly " + std::to_string(card));
            continue;
        }
        if (pos.pool >= n_pools || rows.pools[pos.pool].kind != rmstate::PoolKind::LIBRARY_RANGE ||
            rows.pools[pos.pool].owner != pos.owner || !checkable[pos.pool] || at < start[pos.pool] ||
            at >= start[pos.pool] + rows.pools[pos.pool].size) {
            violation(out, "library position " + std::to_string(at) + " names pool " + std::to_string(pos.pool) +
                               ", which does not cover it");
            continue;
        }
        const auto it = listed[pos.pool].find(card);
        const Universe *uni = universe[pos.owner];
        const bool candidate = it != listed[pos.pool].end() ? it->second.second > 0 : !(uni && uni->contains(card));
        if (!candidate)
            violation(out, "library position " + std::to_string(at) + " truly holds card " + std::to_string(card) +
                               ", not a candidate of pool " + std::to_string(pos.pool));
    }
    const int opp = viewer == 0 ? 1 : 0;
    for (const KKnownHandRow &k : rows.known_hand) {
        ++out->facts;
        if (count_of(truth.hand[opp], k.card) < k.copies)
            violation(out, "the opponent's hand is reported to hold " + std::to_string(k.copies) + " of card " +
                               std::to_string(k.card) + ", truly " + std::to_string(count_of(truth.hand[opp], k.card)));
    }
    for (const KSeenRow &s : rows.seen) {
        ++out->facts;
        if (s.player != 0 && s.player != 1) {
            violation(out, "a k_seen row without its player");
            continue;
        }
        if (s.game > count_of(truth.owned[s.player], s.card))
            violation(out, "seen " + std::to_string(s.game) + " of card " + std::to_string(s.card) +
                               " this game, the player has " + std::to_string(count_of(truth.owned[s.player], s.card)));
        if (s.match < s.game)
            violation(out, "seen " + std::to_string(s.game) + " of card " + std::to_string(s.card) + " this game but " +
                               std::to_string(s.match) + " this match");
    }
    check_lists(rows, truth, out);
}

void set_truth_lists(const Metagame &metagame, const Decklists &lists, const bool visible[2], Truth *truth) {
    truth->lists_known = true;
    for (int p = 0; p < 2; ++p) {
        truth->reference[p].clear();
        truth->classified[p] = -1;
        if (!lists.present[p]) continue;
        std::vector<std::pair<int, int>> main, side;
        registered_counts(lists, p, &main, &side);
        for (int l : matching_lists(metagame, main, side))
            truth->reference[p].emplace_back(l, metagame.lists[static_cast<size_t>(l)].archetype);
        if (visible[p]) truth->classified[p] = classify_list(metagame, lists.name[p], main, side);
    }
}

// The k_template and k_archetype facts against the true registered lists.
static void check_lists(const KRows &rows, const Truth &truth, TruthCheck *out) {
    if (!truth.lists_known || rows.templates.empty()) return;
    for (int p = 0; p < 2; ++p) {
        const std::string who = p == 0 ? "A" : "B";
        for (const auto &ref : truth.reference[p]) {
            ++out->facts;
            bool consistent = false;
            for (const KTemplateRow &t : rows.templates)
                if (t.player == p && t.list == ref.first) consistent = t.consistent;
            if (!consistent)
                violation(out, "player " + who + "'s registered list is reference list " + std::to_string(ref.first) +
                                   ", reported inconsistent");
            ++out->facts;
            for (const KArchetypeRow &a : rows.archetypes) {
                if (a.player != p) continue;
                if (a.archetype == ref.second && !a.possible)
                    violation(out,
                        "player " + who + "'s archetype " + std::to_string(ref.second) + " is reported impossible");
                if (a.archetype != ref.second && a.exact)
                    violation(out, "player " + who + "'s archetype is reported exactly " + std::to_string(a.archetype) +
                                       ", truly " + std::to_string(ref.second));
            }
        }
        ++out->facts;
        int registered = -1, n_registered = 0;
        for (const KTemplateRow &t : rows.templates)
            if (t.player == p && t.registered) {
                registered = t.list;
                ++n_registered;
            }
        if (n_registered > 1 || registered != truth.classified[p])
            violation(out, "player " + who + "'s registered list is classified as list " + std::to_string(registered) +
                               " (" + std::to_string(n_registered) + " row(s)), expected " +
                               std::to_string(truth.classified[p]));
    }
}

}  // namespace knowledge
