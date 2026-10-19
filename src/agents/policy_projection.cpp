#include "policy_projection.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>

namespace {

// How far above an integer a row's scaled probability may lie and still round to it.
const double CEIL_SLACK = 1e-9;

}  // namespace

namespace agents {

void threshold_policy(std::vector<double> *mu, double threshold) {
    if (threshold <= 0 || mu->empty()) return;
    if (*std::max_element(mu->begin(), mu->end()) < threshold) return;
    double sum = 0;
    for (double &p : *mu) {
        if (p < threshold) p = 0;
        sum += p;
    }
    for (double &p : *mu) p /= sum;
}

void discretize_policy(std::vector<double> *mu, int n) {
    if (n <= 0 || mu->empty()) return;
    std::vector<double> &m = *mu;
    std::vector<size_t> order(m.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&m](size_t a, size_t b) { return m[a] > m[b]; });
    std::vector<int64_t> count(m.size(), 0);
    int64_t left = n;
    for (size_t row : order) {
        const double scaled = std::ceil(m[row] * n - CEIL_SLACK);
        const int64_t up = scaled > 0 ? static_cast<int64_t>(std::min(scaled, static_cast<double>(n))) : 0;
        const int64_t take = std::min(up, left);
        count[row] += take;
        left -= take;
    }
    count[order[0]] += left;
    for (size_t i = 0; i < m.size(); ++i) m[i] = static_cast<double>(count[i]) / n;
}

}  // namespace agents
