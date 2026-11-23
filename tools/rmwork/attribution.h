#ifndef RMWORK_ATTRIBUTION_H
#define RMWORK_ATTRIBUTION_H

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "../../src/featurize/perturb.h"
#include "../../src/featurize/setup.h"
#include "match.h"
#include "replay_evaluator.h"

// ── Attribution: which inputs move V and π at a decision (docs/workbench_plan.md §2.8) ───────────
//
// Occlusion through the Replay evaluator (replay_evaluator.h): one seat's frame at one decision
// (Replay: a seat view's own frame, the omniscient view the decider's; Play: the human's) is
// featurized as the evaluator featurizes it (its loaded inputs, featurize_traced for the token
// sources), its units listed (src/featurize/perturb.h) and each unit's change evaluated by one served
// policy, as evaluator jobs whose items carry the change, at a priority above re-evaluation's. Each
// unit's metrics against the unchanged evaluation are learn/rmlearn/ablation.py's: dv (the value on the
// frame's seat's scale, or Player A's), tv, kl, flip (the changed argmax a row the unchanged policy
// ranked below its own), dlogp (log π of the focus row), plus dmargin (log π(focus) − log π(rival)); a
// frame without a menu has dv only. The focus row is the chosen row in Replay and the argmax in Play
// unless set (a Choices click); the rival the row of the next-highest π unless set (Shift+click).
//
// Staged: the unchanged evaluation and the coarse units (zone groups, players, the stack, single-row
// tokens, cards) first; then every token of the groups that moved the target most (the top
// SECOND_PASS_GROUPS) and every slot and relation type of the decision, streamed as the requests
// return; a row's children (a group's tokens, a token's cells, a card's holders, a slot's cells) when
// it is expanded. The resample baseline evaluates a slot or cell RESAMPLE_DRAWS times and averages the
// metrics. The optional Shapley mode then samples SHAPLEY_PERMUTATIONS permutations of the top k units
// (k ≤ SHAPLEY_MAX_UNITS, by |target|), each coalition evaluated once, and checks efficiency: the
// values' sum against the change with all k units changed.
//
// A new decision, seat, policy, baseline or method cancels the work (start()); a request in flight
// finishes and its results are still cached. The cache: one file per (decision, seat, model, method)
//
//   $XDG_CACHE_HOME/robomage/workbench/attribution/<log hash>/g<game>[s]d<decision>-<a|b>-<model hash>-<method>.bin
//
// the decision named by its state-format key (game, sideboarding, decision), the model by its identity
// (eval_cache.h), the method "occ-<baseline>" or "shap-<baseline>-k<k>-m<M>". A live Play match has no
// log hash yet: its directory is live-<hash of its decision log's path>, which Replay reads after its
// own. An entry holds the server's version, the frame's payload hash and the spec's hash (an entry of
// another frame or spec is no entry), the unchanged evaluation and per unit the unit itself (so a cached
// decision shows with no server and no featurizer) and its evaluations. It is written (to a `.tmp`,
// renamed) whenever a job of its decision ends, and ends with a hash, so a damaged file reads as none;
// the oldest files go past ATTR_CACHE_MAX_BYTES. A cached entry is checked against the version the
// server serves (a handshake) before more is asked of it; another version starts it afresh.

namespace rmwork {

inline constexpr size_t SECOND_PASS_GROUPS = 3;
inline constexpr int RESAMPLE_DRAWS = 4;
inline constexpr int SHAPLEY_MAX_UNITS = 12;
inline constexpr int SHAPLEY_PERMUTATIONS = 8;
inline constexpr int ATTR_PRIORITY = 10;  // above re-evaluation's (0)
inline constexpr uint64_t ATTR_CACHE_MAX_BYTES = 256ull << 20;
inline constexpr double SHAPLEY_TOLERANCE = 1e-4;

// What the tool ranks and the Table tints by.
enum class AttrTarget {
    V,       // dv
    TV,      // tv
    LOGP,    // dlogp
    MARGIN,  // dmargin
};
inline constexpr int ATTR_TARGET_COUNT = 4;
// "V", "π: TV", "log π(focus)", "focus vs rival".
const char *attr_target_name(AttrTarget t);
// Signed (dv, dlogp, dmargin) or not (tv).
inline bool attr_target_signed(AttrTarget t) {
    return t != AttrTarget::TV;
}

enum class AttrMethod {
    OCCLUSION,
    SHAPLEY
};

// One evaluation by one policy: its value and log π over the menu (empty without one).
struct AttrEval {
        float value = 0;
        std::vector<float> log_pi;
};

struct AttrMetrics {
        bool valid = false;   // evaluated
        bool policy = false;  // the π metrics are defined (a menu of two rows or more)
        double dv = 0, tv = 0, kl = 0, flip = 0, dlogp = 0, dmargin = 0;
        double get(AttrTarget t) const;
};

// The metrics of `changed` (a draw each, averaged) against `base`; `negate` puts dv on the other seat's scale.
AttrMetrics attr_metrics(const AttrEval &base, const std::vector<AttrEval> &changed, int focus, int rival, bool negate);
// The row of the highest log π, and of the next highest (-1 without a menu / a second row).
int attr_argmax(const std::vector<float> &log_pi);
int attr_runner_up(const std::vector<float> &log_pi, int best);

// The attribution cache's root ($XDG_CACHE_HOME/robomage/workbench/attribution).
std::string attr_cache_root();
// The directory of a match's entries: its log hash's, or for a live match its decision log path's.
std::string attr_cache_dir(const std::string &root, uint64_t log_hash, bool live, const std::string &log_path);

// What one attribution is of.
struct AttrKey {
        std::string dir;  // its cache directory (attr_cache_dir)
        int game = 0;
        bool sideboarding = false;
        int64_t decision = 0;
        int seat = 0;
        uint32_t policy = 0;  // the served id
        std::string model;    // the model's identity (model_identity)
        featurize::Baseline baseline = featurize::Baseline::NONE;
        AttrMethod method = AttrMethod::OCCLUSION;
        int shapley_k = SHAPLEY_MAX_UNITS;
        bool operator==(const AttrKey &o) const;
        bool operator!=(const AttrKey &o) const { return !(*this == o); }
        // "occ-none", "shap-mean-k12-m8".
        std::string method_name() const;
        // The entry's path under its directory.
        std::string path() const;
};

// A decision's cached results (the file's contents).
struct AttrEntry {
        AttrKey key;
        uint64_t frame_hash = 0, spec_hash = 0, version = 0;
        int menu_rows = 0;
        bool have_base = false;
        AttrEval base;
        std::vector<featurize::Unit> units;                        // every unit asked, in the order asked
        std::map<std::string, std::vector<AttrEval>> evals;        // per unit key, per draw
        std::map<std::string, std::vector<std::string>> children;  // a unit's key → its children's, once asked
        // Shapley: the units in play (keys), per coalition (a bit per unit) its evaluation.
        std::vector<std::string> players;
        std::map<uint32_t, AttrEval> coalitions;
        const featurize::Unit *unit(const std::string &key) const;
};
std::string attr_entry_bytes(const AttrEntry &e);
bool parse_attr_entry(const std::string &bytes, AttrEntry *out, std::string *err);
bool write_attr_entry(const AttrEntry &e, const std::string &root, std::string *err);
bool read_attr_entry(const std::string &path, AttrEntry *out, std::string *err);
// Removes the oldest entries while `root` (its directories) holds more than `max_bytes`, keeping `keep`.
void prune_attr_cache(const std::string &root, uint64_t max_bytes, const std::string &keep);

// The Shapley permutations of `k` units: SHAPLEY_PERMUTATIONS orders drawn with a generator seeded by `seed`.
std::vector<std::vector<int>> shapley_permutations(int k, uint64_t seed);
struct ShapleyResult {
        bool ready = false;
        std::vector<double> phi;    // per player, of the target
        double sum = 0, total = 0;  // Σ φ, and f(all changed) − f(none)
        bool efficient() const { return ready && std::fabs(sum - total) <= SHAPLEY_TOLERANCE; }
};
// The target's Shapley values from the entry's coalitions (TV, not additive, has none).
ShapleyResult shapley_values(const AttrEntry &e, AttrTarget t, int focus, int rival, bool negate);

// One shown row: a unit instance in the tree, with its metrics.
struct AttrRow {
        std::string key;  // its unit's
        int parent = -1;
        int depth = 0;
        std::vector<int> children;
        bool expandable = false;
        bool asked = false;  // its children were asked for
        AttrMetrics m;
        double phi = 0;  // Shapley value, when `shapley`
        bool shapley = false;
};

// What the Table layer, the State Tree and the Tokens tool read: per object (handle in the viewer's
// frame), player and frame row the current target's value.
struct AttrTints {
        bool on = false;
        AttrTarget target = AttrTarget::V;
        double scale = 0;       // the largest |value|
        double cell_scale = 0;  // the largest |value| of a cell (the Tokens tool's heat)
        std::map<uint64_t, double> objects;
        std::array<bool, 2> has_player{{false, false}};
        std::array<double, 2> players{{0, 0}};
        bool same_frame = false;                                        // the rows below are the viewer's frame's
        std::map<std::pair<int, int64_t>, double> rows;                 // (table id, row) → a token's
        std::map<std::tuple<int, int64_t, std::string>, double> cells;  // (table id, row, column) → a cell's
        std::map<std::pair<int64_t, int>, double> token_cells;          // (token, slot) → a cell's (Tokens tool)
        std::string legend;                                             // "V · decision 41, seat A · re p0 v3"
        std::map<uint64_t, uint64_t> handles;  // the attributed frame's handles in the viewer's (other frames)
        // A handle of the attributed frame as the viewer's frame names it (0: it names none).
        uint64_t view_handle(uint64_t h) const {
            if (same_frame) return h;
            auto it = handles.find(h);
            return it == handles.end() ? 0 : it->second;
        }
};

class Attribution {
    public:
        enum class State {
            IDLE,
            RUNNING,
            DONE,
            CANCELLED,
            FAILED
        };
        struct Request {
                std::shared_ptr<const EvalSource> source;
                const Match *match = nullptr;  // its frames (the seat's frame is decoded from it)
                size_t index = 0;              // the match's decision
                int seat = 0;
                std::string log_path;
                uint32_t policy = 0;
                std::string model;  // the served model's path or reference (model_identity of it)
                featurize::Baseline baseline = featurize::Baseline::NONE;
                std::shared_ptr<const featurize::SlotStats> stats;  // MEAN, RESAMPLE
                AttrMethod method = AttrMethod::OCCLUSION;
                std::string cache_root;  // attr_cache_root() unless a check names another
                bool use_server = true;  // false: the cache alone
                // Asked for (Explain): a cancelled, failed or unevaluated attribution of the same key starts
                // again; otherwise the same key is left as it is.
                bool force = false;
        };

        Attribution() = default;
        Attribution(const Attribution &) = delete;
        Attribution &operator=(const Attribution &) = delete;
        ~Attribution();

        // Attributes `r`'s decision (the key in hand: nothing, unless forced or a server is there now that
        // was not); the running work is cancelled first, its in-flight request still cached.
        void start(ReplayEvaluator *ev, const Request &r);
        // Takes what the jobs returned, advances the stages, writes the cache. Call it every frame.
        void poll();
        // Cancels the work (in-flight results still cached).
        void cancel();
        // Asks for row `row`'s children.
        void expand(int row);
        // Forgets the decision (cancelling its work).
        void clear();

        State state() const { return state_; }
        bool running() const { return state_ == State::RUNNING; }
        // Jobs of this or an earlier decision still returning.
        bool busy() const;
        const std::string &error() const { return error_; }
        bool no_server() const { return no_server_; }
        bool has_key() const { return have_key_; }
        const AttrKey &key() const { return key_; }
        size_t index() const { return index_; }
        const AttrEntry *entry() const { return entry_.get(); }
        // The unchanged evaluation (nullptr before it returned).
        const AttrEval *base() const { return entry_ && entry_->have_base ? &entry_->base : nullptr; }
        // "Coarse units: 12 of 40", "Second pass: 80 of 213", "Shapley: 20 of 96".
        std::string phase() const;
        double fraction() const;
        // The stage running: 0 the unchanged evaluation, 1 the coarse units, 2 the second pass, 3 children,
        // 4 Shapley's coalitions, 5 the cached version's check; 0 when idle.
        int stage() const { return stage_; }
        // The entry came from the cache (whole, nothing evaluated since).
        bool from_cache() const { return from_cache_; }
        // Units evaluated since start (each draw counted), and the server's requests for them.
        size_t evaluated() const { return evaluated_; }

        // The focus and rival rows (-1: the default: the chosen row or argmax; the runner-up).
        void set_focus(int focus, int rival);
        int focus() const;
        int rival() const;
        // The focus and rival rows set (-1: the default).
        int focus_choice() const { return focus_; }
        int rival_choice() const { return rival_; }
        // The seat whose scale dv is on: the frame's (false) or Player A's (true).
        void set_scale_a(bool on);
        bool scale_a() const { return scale_a_; }
        // The rows, roots first in the order given by `sort` (a target, descending |value|; ties by
        // label), each root's children after it when expanded.
        const std::vector<AttrRow> &rows() const { return rows_; }
        const featurize::Unit *unit(const AttrRow &r) const;
        // The ranked roots (indices into rows()) by `t`.
        std::vector<int> ranked_roots(AttrTarget t) const;
        // The values the views tint by, `view_frame` the viewer's frame (the handles mapped onto it when
        // it is not the attributed seat's).
        AttrTints tints(AttrTarget t, const rmstate::DecodedFrame *view_frame, int view) const;
        // Shapley's result for `t` (empty when not in that mode).
        ShapleyResult shapley(AttrTarget t) const;
        // The decision's tensors and units as listed (nullptr before the featurizer's inputs came).
        const featurize::Tensors *tensors() const { return have_units_ ? &tensors_ : nullptr; }
        const featurize::UnitList *units() const { return have_units_ ? &list_ : nullptr; }
        const featurize::FeatureSpec *spec() const { return features_ ? &features_->spec : nullptr; }
        const std::vector<std::vector<featurize::SlotDef>> &slots() const { return slots_; }
        // The seat frame attributed (nullptr none).
        const rmstate::DecodedFrame *frame() const { return frame_.get(); }

    private:
        // A job: the evaluator's id, its entry, per item what it evaluates.
        struct ItemRef {
                std::string key;  // a unit's key; "" the unchanged decision; "#<mask>" a coalition
                int draw = 0;
        };
        struct Job {
                uint64_t id = 0;
                std::shared_ptr<AttrEntry> entry;
                std::vector<ItemRef> items;
                int stage = 0;  // 0 base (+ coarse), 1 coarse, 2 second pass, 3 children, 4 Shapley, 5 version check
                bool current = true;
                size_t done = 0;
        };
        void submit(int stage, const std::vector<const featurize::Unit *> &units, bool with_base);
        bool submit_shapley();
        void submit_version_check();
        bool build_units();
        void take_results(Job &j);
        void finish_job(Job &j, const EvalJobStatus &st);
        // Submits the first stage from `stage` on with something missing; DONE when none has.
        void advance(int stage);
        std::vector<const featurize::Unit *> missing(int stage);
        int draws(const featurize::Unit &u) const;
        void fail(const std::string &why, bool no_server);
        void rebuild_rows();
        void add_row(const std::string &key, int parent, int depth);
        void recompute();
        void write(const AttrEntry &e);
        bool negate() const;
        double row_value(const AttrRow &r, AttrTarget t) const;

        ReplayEvaluator *ev_ = nullptr;
        Request req_;
        AttrKey key_;
        bool have_key_ = false;
        size_t index_ = 0;
        State state_ = State::IDLE;
        std::string error_;
        bool no_server_ = false;
        bool from_cache_ = false;
        size_t evaluated_ = 0;
        std::shared_ptr<AttrEntry> entry_;
        std::vector<Job> jobs_;
        std::shared_ptr<const rmstate::DecodedFrame> frame_;
        const rmstate::Schema *schema_ = nullptr;
        uint64_t frame_hash_ = 0;
        int menu_rows_ = 0;
        int chosen_ = -1;
        std::shared_ptr<const featurize::Featurization> features_;
        bool have_units_ = false;
        featurize::Tensors tensors_;
        featurize::UnitList list_;
        std::vector<std::vector<featurize::SlotDef>> slots_;
        std::vector<AttrRow> rows_;
        int focus_ = -1, rival_ = -1;
        bool scale_a_ = false;
        bool second_done_ = false, shapley_asked_ = false;
        size_t stage_total_ = 0, stage_done_ = 0;
        int stage_ = 0;
};

inline constexpr size_t SLOT_STATS_LOGS = 6;

// The slot statistics the mean and resample baselines read (perturb.h): <dir>/<spec hash>.rmtb (`rmfeat
// --slot-stats`'s file), built on first use on a thread from the first SLOT_STATS_LOGS of the corpus's
// decision logs, each replayed by the engine with --rmstate-dump and every seat frame featurized.
class SlotStatsSource {
    public:
        SlotStatsSource() = default;
        SlotStatsSource(const SlotStatsSource &) = delete;
        SlotStatsSource &operator=(const SlotStatsSource &) = delete;
        ~SlotStatsSource();
        // The statistics of `f`'s spec: read at once when the file is there, else the build started and
        // nullptr until it has written the file.
        std::shared_ptr<const featurize::SlotStats> get(const std::shared_ptr<const featurize::Featurization> &f,
            const std::string &engine, const std::string &workdir, const std::string &corpus, const std::string &dir);
        bool building() const;
        // "Slot statistics: replaying the corpus (2 of 6)", the failure, or "".
        std::string note() const;

    private:
        void build(std::shared_ptr<const featurize::Featurization> f, std::string engine, std::string workdir,
            std::string corpus, std::string path);

        std::thread thread_;
        std::atomic<bool> running_{false};
        std::atomic<bool> stop_{false};
        mutable std::mutex mu_;
        std::string note_;
        uint64_t spec_ = 0;  // the spec of `stats_` (or of the build)
        std::shared_ptr<const featurize::SlotStats> stats_;
};

// Per object handle of `from` (a seat's frame) the handle of the same object in `to` (another viewer's
// frame of the same decision): objects matched by zone, owner, controller, card and timestamp where
// `from` sees the card, stack objects by position.
std::map<uint64_t, uint64_t> map_handles(const rmstate::DecodedFrame &from, const rmstate::DecodedFrame &to);

}  // namespace rmwork

#endif /* RMWORK_ATTRIBUTION_H */
