// Archetype inference in the sidecar: the reference lists against each player (knowledge.h,
// templates.h, docs/knowledge_store.md §10).

#include <algorithm>
#include <climits>

#include "belief.h"
#include "knowledge.h"

namespace knowledge {

// Steps one reference list's feasibility check may use against one template network before it
// reports the list consistent without proof (sound; k_template.exact 0).
static constexpr uint64_t TEMPLATE_BUDGET = 250000;
// Constraint evaluations the propagation seeding that check may use (still sound past it).
static constexpr uint64_t PROPAGATION_STEPS = 1000000;
// Leaves one list's likelihood sweep may visit against one template network before the belief
// layer uses the copies-seen fallback for it (belief.h).
static constexpr uint64_t LIKELIHOOD_BUDGET = 200000;

// Whether the opponent's lists are checked against their network: a live game at level expert (or
// expert+belief), a metagame with lists, and the opponent's registered list hidden
// (--closed-decklist).
bool Knowledge::templates_by_network() const {
    return level_ != rmstate::KnowledgeLevel::OFF && viewer_ != NO_SEAT && game_active_ && !metagame_.lists.empty() &&
           !lists_.present[1 - viewer_];
}

size_t Knowledge::opponent_pins() const {
    return viewer_ == NO_SEAT ? 0 : players_[1 - viewer_].net.pin_log().size();
}

size_t Knowledge::opponent_roots() const {
    return viewer_ == NO_SEAT ? 0 : players_[1 - viewer_].net.roots().size();
}

// The opponent's network, settled after an event that pinned a card or added a root, becomes the
// template network every list is checked against until the next such event.
void Knowledge::save_template_network() {
    tpl_dirty_ = false;
    tpl_active_ = templates_by_network();
    if (!tpl_active_) return;
    tpl_net_ = players_[1 - viewer_].net;
    tpl_done_.assign(metagame_.lists.size(), 0);
    tpl_consistent_.assign(metagame_.lists.size(), 0);
    tpl_exact_.assign(metagame_.lists.size(), 0);
    tpl_ll_.assign(metagame_.lists.size(), 0.0);
    tpl_ll_exact_.assign(metagame_.lists.size(), 0);
    tpl_seen_ = seen_game_[1 - viewer_];
}

// Reference list `l` against the template network, once per template network: a function of that
// network alone.
void Knowledge::check_template(size_t l) {
    if (tpl_done_[l]) return;
    tpl_done_[l] = 1;
    const MetaList &list = metagame_.lists[l];
    Network h;
    if (!hypothesis_network(tpl_net_, list.main, list.side, &h)) {
        tpl_consistent_[l] = 0;
        tpl_exact_[l] = 1;
        return;
    }
    // Sound bounds of the hypothesis network seed the search (and may find it has no world).
    BoundStore store(true);
    store.universe = network_universe(h);
    std::vector<NodeId> frontier;
    for (size_t n = 0; n < h.node_count(); ++n)
        if (h.node(static_cast<NodeId>(n)).out == NO_JUNCTION) frontier.push_back(static_cast<NodeId>(n));
    store.propagate(h, frontier, PROPAGATION_STEPS);
    const Feasibility f = solver_.feasible(h, store, TEMPLATE_BUDGET);
    tpl_consistent_[l] = f.feasible ? 1 : 0;
    tpl_exact_[l] = f.exact ? 1 : 0;
    if (belief_on() && f.feasible) check_likelihood(l, h, f.exact);
}

// Every list the copies seen do not already exclude, against the current template network (each
// checked once per template network).
void Knowledge::solve_templates() {
    if (!tpl_active_) return;
    for (size_t l = 0; l < metagame_.lists.size(); ++l)
        if (!list_excluded_[l] && seen_allows(l)) check_template(l);
}

// At game_end: the game's last answer for every list becomes match memory (the registered list is
// the same in every game of the match, and each game's worlds are independent of the others').
void Knowledge::close_templates() {
    if (!tpl_active_) return;
    for (size_t l = 0; l < metagame_.lists.size(); ++l) {
        if (list_excluded_[l] || !seen_allows(l)) continue;
        check_template(l);
        if (!tpl_consistent_[l]) list_excluded_[l] = 1;
        if (!tpl_exact_[l]) list_unproven_[l] = 1;
        if (belief_on() && tpl_consistent_[l]) {
            list_ll_[l] += tpl_ll_[l];
            if (!tpl_ll_exact_[l]) list_ll_approx_[l] = 1;
        }
    }
    tpl_active_ = false;
}

// Whether list `l` holds at least the copies of every card the opponent was seen to own in a game
// of the match (k_seen.match: the fewest they can own, so at most what their 75 holds).
bool Knowledge::seen_allows(size_t l) const {
    const MetaList &list = metagame_.lists[l];
    for (const auto &s : seen_match_[1 - viewer_]) {
        int copies = 0;
        for (const auto &c : list.main)
            if (c.first == s.first) copies += c.second;
        for (const auto &c : list.side)
            if (c.first == s.first) copies += c.second;
        if (copies < s.second) return false;
    }
    return true;
}

// The k_template row of player `p` and list `l`, without `registered`: a list the viewer sees
// (`main`, `side`: its counts) is consistent exactly when it is that list; the opponent's hidden
// list combines the match memory, the copies seen and, in a live game, the template network's
// answer.
void Knowledge::list_state(int p, size_t l, const std::vector<std::pair<int, int>> &main,
    const std::vector<std::pair<int, int>> &side, KTemplateRow *row) const {
    row->player = p;
    row->list = static_cast<uint16_t>(l);
    if (lists_.present[p]) {
        row->consistent = same_list(metagame_.lists[l], main, side);
        row->exact = true;
        return;
    }
    row->consistent = !list_excluded_[l] && seen_allows(l) && (!tpl_active_ || tpl_consistent_[l]);
    row->exact = !row->consistent || (!list_unproven_[l] && (!tpl_active_ || tpl_exact_[l]));
}

// k_template and k_archetype: per player (self first), every reference list, then every archetype
// label: possible when some consistent list has it, exact when every consistent list does.
void Knowledge::template_rows(KRows &out) const {
    if (metagame_.lists.empty()) return;
    const int order[2] = {viewer_, 1 - viewer_};
    for (int p : order) {
        int classified = -1;
        std::vector<std::pair<int, int>> main, side;
        if (lists_.present[p]) {
            registered_counts(lists_, p, &main, &side);
            classified = classify_list(metagame_, lists_.name[p], main, side);
        }
        std::vector<int> consistent_labels;
        for (size_t l = 0; l < metagame_.lists.size(); ++l) {
            KTemplateRow row;
            list_state(p, l, main, side, &row);
            row.registered = static_cast<int>(l) == classified;
            if (row.consistent) consistent_labels.push_back(metagame_.lists[l].archetype);
            out.templates.push_back(row);
        }
        for (size_t a = 0; a < metagame_.archetypes.size(); ++a) {
            KArchetypeRow row;
            row.player = p;
            row.archetype = static_cast<uint8_t>(a);
            const long n = std::count(consistent_labels.begin(), consistent_labels.end(), static_cast<int>(a));
            row.possible = n > 0;
            row.exact = n > 0 && n == static_cast<long>(consistent_labels.size());
            out.archetypes.push_back(row);
        }
    }
}

bool Knowledge::belief_on() const {
    return level_ == rmstate::KnowledgeLevel::EXPERT_BELIEF;
}

// The likelihood of list `l` (consistent: `h` is its hypothesis network of the template network,
// `proven` whether the consistency is): the network's weighted world count, else (past the budget,
// or a zero count the sound layer did not prove: the belief layer never excludes a list the k_* tables
// keep) the copies-seen fallback.
void Knowledge::check_likelihood(size_t l, const Network &h, bool proven) {
    const Likelihood lk = network_likelihood(h, LIKELIHOOD_BUDGET);
    if (lk.finished && lk.possible) {
        tpl_ll_[l] = lk.log_p;
        tpl_ll_exact_[l] = 1;
        return;
    }
    if (lk.finished && proven)
        error("belief: reference list " + metagame_.lists[l].name +
              " has a world, but its hypothesis network's weighted count is zero");
    tpl_ll_[l] = seen_log_likelihood(metagame_.lists[l], tpl_seen_);
    tpl_ll_exact_[l] = 0;
}

// List `l`'s log-likelihood for the hidden opponent over the match: the finished games' and, in a
// live game, the template network's; `exact` when every one of them is an exact weighted count.
double Knowledge::list_log_likelihood(size_t l, bool *exact) const {
    double ll = list_ll_[l];
    *exact = !list_ll_approx_[l];
    if (tpl_active_) {
        ll += tpl_ll_[l];
        *exact = *exact && tpl_ll_exact_[l];
    }
    return ll;
}

// Player `p`'s log weights over the reference lists, the support (the lists `rows`' k_template rows
// keep) and per list whether its weight is exact. A list the viewer sees is explicit: the lists equal
// to it share the probability.
void Knowledge::belief_weights(int p, const KRows &rows, std::vector<double> *log_w, std::vector<uint8_t> *support,
    std::vector<uint8_t> *exact) const {
    const size_t n = metagame_.lists.size();
    log_w->assign(n, 0.0);
    support->assign(n, 0);
    exact->assign(n, 1);
    for (const KTemplateRow &t : rows.templates)
        if (t.player == p && t.consistent && t.list < n) (*support)[t.list] = 1;
    if (lists_.present[p]) return;
    for (size_t l = 0; l < n; ++l) {
        if (!(*support)[l]) continue;
        bool ex = true;
        (*log_w)[l] = list_log_likelihood(l, &ex);
        (*exact)[l] = ex ? 1 : 0;
    }
}

// b_list and b_archetype: per player (self first), every reference list's posterior, then every
// label's (the sum of its lists').
void Knowledge::belief_rows(KRows &out) const {
    if (!belief_on() || metagame_.lists.empty()) return;
    const int order[2] = {viewer_, 1 - viewer_};
    for (int p : order) {
        std::vector<double> log_w;
        std::vector<uint8_t> support, exact;
        belief_weights(p, out, &log_w, &support, &exact);
        const std::vector<uint32_t> prob = fixed_point_posterior(log_w, support);
        std::vector<uint64_t> by_label(metagame_.archetypes.size(), 0);
        for (size_t l = 0; l < metagame_.lists.size(); ++l) {
            BListRow row;
            row.player = p;
            row.list = static_cast<uint16_t>(l);
            row.prob = prob[l];
            row.loglik = support[l] ? loglik_column(log_w[l]) : INT32_MIN;
            row.exact = exact[l] != 0;
            out.belief_lists.push_back(row);
            const size_t a = static_cast<size_t>(metagame_.lists[l].archetype);
            if (a < by_label.size()) by_label[a] += prob[l];
        }
        for (size_t a = 0; a < metagame_.archetypes.size(); ++a) {
            BArchetypeRow row;
            row.player = p;
            row.archetype = static_cast<uint8_t>(a);
            row.prob = static_cast<uint32_t>(by_label[a]);
            out.belief_archetypes.push_back(row);
        }
    }
}

std::vector<double> Knowledge::belief_posterior(int p) const {
    if (!belief_on() || metagame_.lists.empty() || viewer_ == NO_SEAT)
        return std::vector<double>(metagame_.lists.size(), 0.0);
    KRows rows;
    template_rows(rows);
    std::vector<double> log_w;
    std::vector<uint8_t> support, exact;
    belief_weights(p, rows, &log_w, &support, &exact);
    return posterior(log_w, support);
}

}  // namespace knowledge
