#ifndef RMWORK_EVALUATIONS_H
#define RMWORK_EVALUATIONS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "match.h"

// ── The recorded evaluations of a match's episodes: values, π, swings ───────────────────────────
//
// An episode step (match.h's EpisodeStepInfo) carries, per evaluated policy, its value of the
// decision from the deciding seat's view and its log π over the menu rows
// (docs/trajectory_format.md). A value is on the deciding seat's own scale; on Player A's scale a B
// decision's value is negated. Series, swings and disagreements are per (seat, policy id): a series
// is one seat's values by one policy, in decision order.
//
// Swings are the largest changes of a seat's value between its consecutive evaluated decisions of
// one game (sideboarding phases left out), and from its last one to the game's result (z, +1 a win,
// -1 a loss, 0 a draw). A disagreement compares adjacent decisions of one game decided by the two
// seats, both evaluated by the policy: A's value against the negated value of B's seat.
//
// The evaluations are the episodes' recorded ones and those a re-evaluation added (reevaluation.h):
// a policy re-evaluated through the inference server has its own id, REEVAL_ID_BASE plus its served
// id, apart from every recorded id (an episode's ids are 32-bit), so it is a series of its own.

namespace rmwork {

inline constexpr int64_t REEVAL_ID_BASE = int64_t(1) << 32;
// A re-evaluated policy's id.
inline bool is_reevaluated(int64_t id) {
    return id >= REEVAL_ID_BASE;
}
inline int64_t reevaluated_id(uint32_t served) {
    return REEVAL_ID_BASE + static_cast<int64_t>(served);
}
// The `file` of an episode step made for a re-evaluation's records alone (the decision had none):
// no μ, policy version or teacher label.
extern const char *const REEVAL_STEP_FILE;

// The step's record of policy `id`; the first record for id -1; nullptr when it has none.
const featurize::PolicyEvalRecord *find_policy(const EpisodeStepInfo &e, int64_t id);
// π over the menu rows: exp of the record's log π.
std::vector<double> policy_pi(const featurize::PolicyEvalRecord &r);
// −Σ p log p (nats).
double entropy(const std::vector<double> &p);
// A value of `seat`'s on Player A's scale.
inline double on_a_scale(int seat, double v) {
    return seat == 1 ? -v : v;
}
// "p0"; a re-evaluated policy "re p0" (its served id).
std::string policy_name(int64_t id);
// "p0 v9834", "re p0 v9834".
std::string policy_label(int64_t id, uint64_t version);
// "+0.42", "−0.17" (a minus sign), "0.00".
std::string signed_text(double v, int decimals = 2);

struct ValuePoint {
        size_t decision = 0;
        double value = 0;  // the seat's own scale
};

struct ValueSeries {
        int seat = 0;
        int64_t policy = 0;
        uint64_t version = 0;  // the policy's version at its last point
        std::vector<ValuePoint> points;
};

struct MatchValues {
        std::vector<ValueSeries> series;       // seat A's, then B's, each by policy id
        std::vector<int64_t> policies;         // every policy id evaluated, ascending
        std::map<int64_t, uint64_t> versions;  // per policy its last version
        bool any() const { return !series.empty(); }
        const ValueSeries *find(int seat, int64_t policy) const;
        // `id` when evaluated, else the first policy; -1 when none.
        int64_t resolve(int64_t id) const;
};

MatchValues match_values(const Match &m);

// A run of decisions of one turn (or of a game's opening hands, or a sideboarding phase).
struct TurnBand {
        size_t first = 0, last = 0;
        int game = 0;
        std::string label;  // "T3", "Pre", "SB"
};
std::vector<TurnBand> turn_bands(const Match &m);
// "T3", "Pre" (the opening hands), "SB" (sideboarding).
std::string turn_label(const DecisionInfo &d);

// The game's result on `seat`'s scale: +1 a win, -1 a loss, 0 a draw or unknown.
double game_result(const Match &m, int game, int seat);

struct Swing {
        int seat = 0;
        int64_t policy = 0;
        size_t from = 0;
        bool to_result = false;         // to the game's result
        size_t to = 0;                  // the next decision (unless to_result)
        double v0 = 0, v1 = 0, dv = 0;  // the seat's own scale
        int game = 0;
};
// The decision a swing shows: its later decision, or its last one for a swing to the result.
inline size_t swing_target(const Swing &s) {
    return s.to_result ? s.from : s.to;
}
// The largest swings of seat `seat` (-1: both) by policy `policy`, largest |ΔV| first (ties in
// decision order), at most `n`.
std::vector<Swing> top_swings(const Match &m, const MatchValues &v, int seat, int64_t policy, size_t n);
// The first swing target after `cursor`, -1 when none.
long next_swing_target(const std::vector<Swing> &swings, size_t cursor);

struct Disagreement {
        size_t a = 0, b = 0;  // seat A's decision and seat B's
        double va = 0;        // A's value
        double vb_on_a = 0;   // −(B's value)
        double gap = 0;       // |va − vb_on_a|
        int game = 0;
};
std::vector<Disagreement> top_disagreements(const Match &m, const MatchValues &v, int64_t policy, size_t n);

// What happened after decision `from` up to decision `to` (to_result: to the end of from's game) in
// `view`'s log: its event lines (and the omniscient log's chosen rows before `to`), at most
// `max_lines` of them joined with " · " ("… +n" for the rest); "" when none.
std::string between_text(const Match &m, int view, size_t from, bool to_result, size_t to, size_t max_lines);

}  // namespace rmwork

#endif /* RMWORK_EVALUATIONS_H */
