#ifndef RMWORK_POLICY_COMPARE_H
#define RMWORK_POLICY_COMPARE_H

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "evaluations.h"
#include "match.h"

// ── Policy Compare's data: several policies' π and V against a reference (ImGui-free) ───────────
//
// docs/workbench_plan.md §2.10. The policies are a match's evaluation series (evaluations.h: the
// episodes' recorded ones and the re-evaluated ones, reevaluation.h). Every evaluation of a decision
// is its deciding seat's, over that decision's menu rows, so the policies' π at a decision are over the
// same rows. What a viewer sees follows §2.7: a seat view the decisions its own seat decided, the
// omniscient view every decision.
//
// At a decision each compared policy has its value (the deciding seat's scale), its π, its argmax
// (the first row of the largest π) and, against the reference, KL(ref‖p) = Σ π_ref (log π_ref − log π_p)
// in nats and TV = ½ Σ |π_ref − π_p|. Agreement counts the decisions both the reference and the policy
// evaluated, and those where their argmax rows are one row.

namespace rmwork {

enum class CompareMetric {
    KL,
    TV
};
// "KL", "TV".
const char *compare_metric_name(CompareMetric m);

// KL(p‖q) in nats over distributions given as log π of the same rows (a row of q's below e^-30 counts
// as e^-30, so a row p reaches and q all but excludes stays finite).
double kl_divergence(const std::vector<float> &log_p, const std::vector<float> &log_q);
// ½ Σ |p − q|.
double total_variation(const std::vector<float> &log_p, const std::vector<float> &log_q);
// The first row of the largest log π, -1 for none.
int argmax_row(const std::vector<float> &log_p);

// The policies viewer `view` can compare: those with a series of a seat it sees, ascending by id.
std::vector<int64_t> comparable_policies(const MatchValues &v, int view);
// The default reference: the last of `policies` (the highest id: a re-evaluated policy served last),
// -1 when there is none.
int64_t default_reference(const std::vector<int64_t> &policies);

// One policy at one decision.
struct PolicyAt {
        int64_t policy = 0;
        uint64_t version = 0;
        bool has = false;  // it evaluated the decision over its menu
        double value = 0;  // the deciding seat's scale
        std::vector<double> pi;
        int argmax = -1;
        // Against the reference (the reference itself: 0, 0, true); meaningful when `has`.
        double kl = 0, tv = 0;
        bool agrees = true;
};

struct DecisionCompare {
        size_t index = 0;  // the decision
        int seat = -1;     // its deciding seat
        int game = 0;
        bool sideboarding = false;
        int turn = 0;
        int chosen = -1;                // the decision log's reply when a row, else -1
        size_t rows = 0;                // the menu's rows
        std::vector<PolicyAt> at;       // per compared policy, in the comparison's order
        int ref = -1;                   // the reference's index in `at`
        double max_kl = 0, max_tv = 0;  // the largest of the other policies'
        bool differs = false;           // some other policy's argmax is not the reference's
};

// The policies `policies` at decision `index` of `m` against `ref`; false when the decision has no
// episode step or the reference did not evaluate it over its menu.
bool compare_decision(
    const Match &m, size_t index, const std::vector<int64_t> &policies, int64_t ref, DecisionCompare *out);

struct Agreement {
        size_t n = 0, agree = 0;
        double kl = 0, tv = 0;  // sums over the n decisions
        double rate() const { return n ? static_cast<double>(agree) / static_cast<double>(n) : 0; }
        double mean_kl() const { return n ? kl / static_cast<double>(n) : 0; }
        double mean_tv() const { return n ? tv / static_cast<double>(n) : 0; }
        void add(const PolicyAt &p);
};

// A policy against the reference over the decisions the viewer sees.
struct PolicyAgreement {
        int64_t policy = 0;
        Agreement all;
        std::array<Agreement, 2> seat;
        std::vector<Agreement> bands;  // per MatchComparison::bands entry
};

struct MatchComparison {
        const Match *match = nullptr;
        int view = VIEW_OMNI;
        std::vector<int64_t> policies;  // compared, in order (the reference among them)
        int64_t ref = -1;
        std::vector<DecisionCompare> decisions;  // the viewer's decisions the reference evaluated, in order
        std::vector<PolicyAgreement> agreement;  // per compared policy but the reference, in order
        std::vector<TurnBand> bands;             // the match's turn bands (evaluations.h)
        // The decision `index`'s entry, nullptr when it has none.
        const DecisionCompare *find(size_t index) const;
        // The index of `policy` in `policies`, -1 when it is not compared.
        int position(int64_t policy) const;
        bool ready() const { return ref >= 0 && policies.size() >= 2; }
};

// `policies` (two or more, `ref` among them) compared over the decisions of `m` that `view` sees.
MatchComparison compare_match(const Match &m, int view, const std::vector<int64_t> &policies, int64_t ref);

struct RankedDecision {
        size_t index = 0;    // the decision
        int64_t policy = 0;  // the policy that diverges most there (among those ranked)
        double score = 0;    // its KL or TV
        double kl = 0, tv = 0;
        bool differs = false;  // its argmax is not the reference's
        int ref_argmax = -1, argmax = -1;
        double shift = 0;  // π_p(its argmax) − π_ref(that row)
};

// The decisions where `policy` (-1: every compared policy but the reference, each decision ranked by
// the one that diverges most) diverges most from the reference by `metric`, largest first (ties in
// decision order), only those where its argmax differs when `argmax_only`; at most `n`.
std::vector<RankedDecision> rank_disagreements(
    const MatchComparison &c, int64_t policy, CompareMetric metric, bool argmax_only, size_t n);

// "0.031" (KL and TV, three decimals; "<0.001" for a positive value below it).
std::string metric_text(double v);
// "87% (41/47)", "—" for none.
std::string agreement_text(const Agreement &a);

}  // namespace rmwork

#endif /* RMWORK_POLICY_COMPARE_H */
