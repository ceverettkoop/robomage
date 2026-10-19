#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "../../src/agents/policy_projection.h"
#include "../../src/agents/py_random.h"
#include "../../src/featurize/rmtb.h"
#include "actor.h"
#include "util.h"

namespace {

// A known answer: `mu` thresholded at `thresh`, then discretised to `disc`, is `want`.
struct KnownProjection {
        const char *what;
        std::vector<double> mu;
        double thresh;
        int disc;
        std::vector<double> want;
};

// Tolerance of a sum or a renormalised probability (double arithmetic).
const double SUM_TOLERANCE = 1e-12;
// Tolerance of a stored (f32) behaviour probability against the recomputed one.
const double MU_TOLERANCE = 1e-6;

}  // namespace

static std::vector<double> softmax(const std::vector<double> &logits, double temp);
static size_t first_max(const std::vector<double> &mu);
static std::string check_thresholded(const std::vector<double> &p, const std::vector<double> &t, double thresh);
static std::string check_discretized(const std::vector<double> &t, const std::vector<double> &d, int n);
static std::string check_known(const KnownProjection &k);
static std::string check_random(agents::PyRandom *rng, size_t *distributions);

// softmax(logits / temp), in double, computed as the nn: agent computes it.
static std::vector<double> softmax(const std::vector<double> &logits, double temp) {
    const double mx = *std::max_element(logits.begin(), logits.end());
    std::vector<double> p(logits.size());
    double sum = 0;
    for (size_t i = 0; i < p.size(); ++i) sum += p[i] = std::exp((logits[i] - mx) / temp);
    for (double &x : p) x /= sum;
    return p;
}

// The first row (in row order) of the largest probability.
static size_t first_max(const std::vector<double> &mu) {
    return static_cast<size_t>(std::max_element(mu.begin(), mu.end()) - mu.begin());
}

// `t` is `p` thresholded at `thresh`: unchanged when every row is below it, else the rows below
// it 0 and the others in their old proportions, summing to 1. "" when it is.
static std::string check_thresholded(const std::vector<double> &p, const std::vector<double> &t, double thresh) {
    if (*std::max_element(p.begin(), p.end()) < thresh) return t == p ? "" : "every row below thresh, yet changed";
    double kept = 0, sum = 0;
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] >= thresh) kept += p[i];
        sum += t[i];
    }
    if (std::fabs(sum - 1) > SUM_TOLERANCE) return "thresholded sum " + std::to_string(sum);
    for (size_t i = 0; i < p.size(); ++i) {
        if (p[i] < thresh && t[i] != 0) return "row " + std::to_string(i) + " below thresh kept";
        if (p[i] >= thresh && std::fabs(t[i] * kept - p[i]) > SUM_TOLERANCE)
            return "row " + std::to_string(i) + " not renormalised";
    }
    return "";
}

// `d` is `t` discretised to `n`: multiples of 1/n whose counts sum to n (to exactly 1 for a power
// of two), `t`'s first most probable row still a most probable row, a row of probability 0 still
// 0, and a fixed point of the projection. "" when it is.
static std::string check_discretized(const std::vector<double> &t, const std::vector<double> &d, int n) {
    int64_t counts = 0;
    double sum = 0;
    for (size_t i = 0; i < d.size(); ++i) {
        const double c = d[i] * n;
        if (std::fabs(c - std::round(c)) > 1e-9 || c < 0) return "row " + std::to_string(i) + " not a multiple of 1/n";
        if (t[i] == 0 && d[i] != 0) return "row " + std::to_string(i) + " of probability 0 given mass";
        counts += static_cast<int64_t>(std::round(c));
        sum += d[i];
    }
    if (counts != n) return "counts sum to " + std::to_string(counts) + ", not " + std::to_string(n);
    if ((n & (n - 1)) == 0 && sum != 1.0) return "probabilities do not sum to exactly 1";
    const size_t top = first_max(t);
    if (d[top] == 0 || d[top] < *std::max_element(d.begin(), d.end())) return "the most probable row lost its place";
    std::vector<double> again = d;
    agents::discretize_policy(&again, n);
    if (again != d) return "not a fixed point";
    again = t;
    agents::discretize_policy(&again, n);
    if (again != d) return "not deterministic";
    return "";
}

static std::string check_known(const KnownProjection &k) {
    std::vector<double> mu = k.mu;
    agents::threshold_policy(&mu, k.thresh);
    agents::discretize_policy(&mu, k.disc);
    for (size_t i = 0; i < mu.size(); ++i)
        if (std::fabs(mu[i] - k.want[i]) > SUM_TOLERANCE)
            return std::string(k.what) + ": row " + std::to_string(i) + " is " + std::to_string(mu[i]) + ", want " +
                   std::to_string(k.want[i]);
    return "";
}

// Random menus (1 to 12 rows, some logits repeated) under several thresholds and step counts; a
// row drawn from each projection has mass. "" when every one holds.
static std::string check_random(agents::PyRandom *rng, size_t *distributions) {
    const double threshes[] = {0, 0.03, 0.2};
    const int discs[] = {0, 1, 10, 16, 32};
    for (int trial = 0; trial < 2000; ++trial) {
        const size_t rows = 1 + static_cast<size_t>(rng->randrange(12));
        std::vector<double> logits(rows);
        for (size_t r = 0; r < rows; ++r)
            logits[r] = r > 0 && rng->randrange(4) == 0 ? logits[0] : rng->random() * 10 - 5;
        const std::vector<double> p = softmax(logits, 1);
        for (double thresh : threshes)
            for (int n : discs) {
                std::vector<double> t = p;
                agents::threshold_policy(&t, thresh);
                std::string bad = check_thresholded(p, t, thresh);
                std::vector<double> d = t;
                agents::discretize_policy(&d, n);
                if (bad.empty() && n > 0) bad = check_discretized(t, d, n);
                if (bad.empty() && n == 0 && d != t) bad = "disc=0 changed the distribution";
                if (bad.empty() && d[rng->weighted_index(d.data(), d.size())] <= 0) bad = "drew a row of mass 0";
                if (!bad.empty())
                    return "trial " + std::to_string(trial) + " (" + std::to_string(rows) + " rows, thresh " +
                           std::to_string(thresh) + ", disc " + std::to_string(n) + "): " + bad;
                ++*distributions;
            }
    }
    return "";
}

namespace rmcheck {

void projection_checks(Report &rep) {
    const KnownProjection known[] = {
        {"threshold then 1/16", {0.5, 0.3, 0.15, 0.05}, 0.1, 16, {9 / 16.0, 6 / 16.0, 1 / 16.0, 0}},
        {"every row below thresh", {0.3, 0.3, 0.4}, 0.5, 0, {0.3, 0.3, 0.4}},
        {"ties in row order", {0.25, 0.25, 0.25, 0.25}, 0, 3, {1 / 3.0, 1 / 3.0, 1 / 3.0, 0}},
        {"exact multiples kept", {0.25, 0.75}, 0.03, 16, {0.25, 0.75}},
        {"one step", {0.2, 0.45, 0.35}, 0, 1, {0, 1, 0}},
    };
    for (const KnownProjection &k : known) {
        const std::string bad = check_known(k);
        if (!bad.empty()) rep.error("actor", "policy projection: " + bad);
    }
    agents::PyRandom rng(20261019);
    size_t distributions = 0;
    const std::string bad = check_random(&rng, &distributions);
    if (!bad.empty()) {
        rep.error("actor", "policy projection: " + bad);
        return;
    }
    rep.info("policy projection: " + std::to_string(sizeof(known) / sizeof(known[0])) + " known answers, " +
             std::to_string(distributions) +
             " random menus thresholded and discretised (sums, dropped rows, multiples of 1/n, the most probable "
             "row kept, fixed point, draws on rows with mass)");
}

void check_projected_evals(
    Report &rep, const std::string &path, const std::string &options, double temp, double thresh, int disc) {
    featurize::RmtbFile f;
    std::string err;
    std::vector<float> logits, mu;
    std::vector<int64_t> offsets, chosen;
    const featurize::RmtbArray *l = nullptr, *m = nullptr, *o = nullptr, *c = nullptr;
    if (featurize::rmtb_read_file(path, &f, &err)) {
        l = f.find("logits");
        m = f.find("mu");
        o = f.find("menu_offsets");
        c = f.find("chosen");
    }
    if (!l || !m || !o || !c || !featurize::rmtb_values(*l, &logits) || !featurize::rmtb_values(*m, &mu) ||
        !featurize::rmtb_values(*o, &offsets) || !featurize::rmtb_values(*c, &chosen) || offsets.empty() ||
        chosen.size() + 1 != offsets.size() || mu.size() != static_cast<size_t>(offsets.back()) ||
        logits.size() != mu.size()) {
        rep.error("actor", "projected: " + path + " is not an evaluations file with mu and chosen " + err);
        return;
    }
    size_t menus = 0, dropped = 0, moved = 0;
    for (size_t i = 0; i + 1 < offsets.size(); ++i) {
        const size_t begin = static_cast<size_t>(offsets[i]), rows = static_cast<size_t>(offsets[i + 1]) - begin;
        const std::vector<double> p =
            softmax(std::vector<double>(logits.begin() + begin, logits.begin() + begin + rows), temp);
        std::vector<double> t = p;
        agents::threshold_policy(&t, thresh);
        std::vector<double> want = t;
        agents::discretize_policy(&want, disc);
        std::string bad = check_thresholded(p, t, thresh);
        if (bad.empty() && disc > 0) bad = check_discretized(t, want, disc);
        double sum = 0;
        for (size_t r = 0; r < rows && bad.empty(); ++r) {
            const double got = mu[begin + r];
            sum += got;
            if (std::fabs(got - want[r]) > MU_TOLERANCE)
                bad = "row " + std::to_string(r) + " mu " + std::to_string(got) + ", the projection " +
                      std::to_string(want[r]);
            else if (disc > 0 && std::fabs(got * disc - std::round(got * disc)) > MU_TOLERANCE * disc)
                bad = "row " + std::to_string(r) + " mu " + std::to_string(got) + " is not a multiple of 1/n";
            dropped += rows > 1 && t[r] == 0 && p[r] > 0;
        }
        if (bad.empty() && std::fabs(sum - 1) > MU_TOLERANCE * static_cast<double>(rows))
            bad = "mu sums to " + std::to_string(sum);
        if (bad.empty() && (chosen[i] < 0 || static_cast<size_t>(chosen[i]) >= rows || mu[begin + chosen[i]] <= 0))
            bad = "the chosen row " + std::to_string(chosen[i]) + " has no mass";
        if (!bad.empty()) {
            rep.error("actor", "projected: decision " + std::to_string(i) + ": " + bad + " (" + path + ")");
            return;
        }
        menus += rows > 1;
        moved += rows > 1 && want != t;
    }
    rep.info("projected: " + options + ": " + std::to_string(chosen.size()) + " decisions (" + std::to_string(menus) +
             " with a choice), mu the projection of their logits, multiples of 1/" + std::to_string(disc) +
             " summing to 1, the chosen row with mass; " + std::to_string(dropped) + " rows dropped by thresh, " +
             std::to_string(moved) + " distributions moved by disc");
}

}  // namespace rmcheck
