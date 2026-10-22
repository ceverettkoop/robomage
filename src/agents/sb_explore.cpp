#include "sb_explore.h"

#include <cctype>
#include <optional>

#include "decision_view.h"

namespace agents {

static double card_weight(const std::map<std::string, double> &weights, double fallback, const std::string &name);
static bool row_weights(const DecisionView &view, const std::vector<size_t> &rows,
    const std::map<std::string, double> &weights, double fallback, double mass, std::vector<double> *q,
    std::string *err);

std::string sb_card_key(const std::string &name) {
    std::string key;
    for (const char c : name) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (u < 0x80 && std::isalnum(u)) key += static_cast<char>(std::tolower(u));
    }
    return key;
}

// The weight of card `name` in `weights`, else `fallback`.
static double card_weight(const std::map<std::string, double> &weights, double fallback, const std::string &name) {
    const auto it = weights.find(sb_card_key(name));
    return it != weights.end() ? it->second : fallback;
}

// `mass` spread over `rows` in proportion to their cards' weights, into q.
static bool row_weights(const DecisionView &view, const std::vector<size_t> &rows,
    const std::map<std::string, double> &weights, double fallback, double mass, std::vector<double> *q,
    std::string *err) {
    std::vector<double> w(rows.size());
    double total = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const std::optional<std::string> card = view.card(rows[i]);
        if (!card) {
            *err = "sb_explore: sideboarding row " + std::to_string(rows[i]) + " has no card";
            return false;
        }
        w[i] = card_weight(weights, fallback, *card);
        total += w[i];
    }
    for (size_t i = 0; i < rows.size(); ++i) (*q)[rows[i]] = mass * w[i] / total;
    return true;
}

bool sb_proposal(const DecisionView &view, const SbExplore &cfg, std::vector<double> *q, std::string *err) {
    const size_t n = view.n();
    q->assign(n, 0.0);
    std::vector<size_t> in, out;
    int done = -1;
    for (size_t i = 0; i < n; ++i) {
        const std::string &c = view.category(i);
        if (c == "SIDEBOARD_IN")
            in.push_back(i);
        else if (c == "SIDEBOARD_OUT")
            out.push_back(i);
        else if (c == "SIDEBOARD_DONE" && done < 0)
            done = static_cast<int>(i);
    }
    if (done < 0 && in.empty() && out.empty()) {
        for (double &x : *q) x = 1.0 / static_cast<double>(n);
        return true;
    }
    if (done < 0) {
        const bool cut = !out.empty();
        return row_weights(
            view, cut ? out : in, cut ? cfg.weights.out : cfg.weights.in, cfg.default_weight, 1.0, q, err);
    }
    const Table match = view.table("match");
    if (match.rows() == 0) {
        *err = "sb_explore: the frame has no match row";
        return false;
    }
    const uint64_t s = match.u(0, match.col("sb_swaps"));
    double at_least = 0;
    for (size_t k = s; k < cfg.count_weights.size(); ++k) at_least += cfg.count_weights[k];
    const double stop = s < cfg.count_weights.size() ? cfg.count_weights[s] : 0.0;
    const double q_done = in.empty() || !(at_least > 0) ? 1.0 : stop / at_least;
    (*q)[static_cast<size_t>(done)] = q_done;
    if (q_done >= 1.0) return true;
    return row_weights(view, in, cfg.weights.in, cfg.default_weight, 1.0 - q_done, q, err);
}

}  // namespace agents
