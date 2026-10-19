#ifndef FEATURIZE_EPISODE_H
#define FEATURIZE_EPISODE_H

#include <cstdint>
#include <string>
#include <vector>

#include "feature_spec.h"
#include "featurizer.h"
#include "rmtb.h"
#include "targets.h"

// ── Trajectory episodes: one game's recorded decisions (docs/trajectory_format.md) ─────────────
//
// Standard library only (with the featurizer's types). An episode is an RMTB file (rmtb.h) of kind
// `episode`: one game of a match together with the sideboarding phase before it, every decision of
// the recorded seats in decision order, each with the deciding seat's featurized tensors as the
// actor's featurizer produced them, stored compactly:
//
//   - numeric features as f16, sparse: per decision the nonzero values (an f16 other than +0) and
//     their positions in the decision's row-major [N, F] tok_num;
//   - a token's categorical, card and ability slots only up to its token type's widths
//     (`type_widths`); the slots past them are 0 (categorical) or -1 (card, ability) in the
//     featurizer's layout, so nothing is lost;
//   - categorical ids as u16 (i32 past 65536 ids), card ids as i16, ability rows as i16 (i32 past
//     32768 rows), token types as u8, relations and numeric positions as u16 (i32 when a value of
//     the file passes 65535);
//   - tok_mask and act_mask (all ones for one unpadded decision) are not stored; act_tok is the
//     decision's menu-type tokens in order (the featurizer emits menu row a as the a-th of them).
//
// Per decision: the seat, the state format's (game, sideboarding, decision) key, the reply (a menu
// row or a concession sentinel), the behaviour distribution μ over every menu row when the deciding
// agent has one (has_mu), the policy version it was sampled from, the teacher's label (-1 none), and
// optionally every policy's evaluation (value, log π over the menu rows) and the omniscient
// featurization with the learner targets (targets.h). The header and footer fields are the file's
// `attrs`: the writer's own (episode_version, F, C, K, Q, menu_type) first, then the caller's.
//
// EpisodeWriter builds a file decision by decision; read_episode expands one back into the
// featurizer's tensors (tok_num from its f16 values). python/rmtraj.py reads the same files.

namespace featurize {

const int EPISODE_VERSION = 1;
extern const char *const EPISODE_KIND;  // "episode"

// How the game of an episode ended (footer `end_reason`).
enum class EndReason : uint8_t {
    WIN,         // a player won (CR 104.2)
    CONCESSION,  // a player conceded (CR 104.3a; the game, or the match between games)
    DRAW,        // the game was a draw (CR 104.4a)
    CAP,         // the run stopped at --max-decisions, no result
    QUIT,        // the run ended with no result: a human seat quit, or the recorded log's QUIT
};
const char *end_reason_name(EndReason r);

// One policy's evaluation of a decision.
struct PolicyEvalRecord {
        int64_t id = 0;
        uint64_t version = 0;
        float value = 0;
        std::vector<float> log_pi;  // over the menu rows
};

// One recorded decision.
struct EpisodeStep {
        uint8_t seat = 0;  // rmstate::Viewer A or B
        int64_t game = 0, decision = 0;
        bool sideboarding = false;
        Tensors tensors;  // the seat's frame, featurize_seat
        bool has_mu = false;
        std::vector<float> mu;  // over the menu rows; empty unless has_mu
        int64_t chosen = 0;     // the reply: a menu row, or -2 / -3 (concession)
        int64_t policy_version = -1;
        int64_t teacher = -1;  // the teacher's label (a menu row), -1 none
        std::vector<PolicyEvalRecord> policies;
        Tensors omniscient;  // the omniscient frame of the same decision (an omniscient episode only)
        Targets targets;     // its learner targets (likewise)
};

// Builds one episode's arrays as decisions are added.
class EpisodeWriter {
    public:
        // `omniscient`: every step carries the omniscient featurization and targets.
        EpisodeWriter(const FeatureSpec &spec, bool omniscient);
        // Appends a step; false with `err` when it does not fit the spec (a token type out of range,
        // a tensor or μ not its shape).
        bool add(const EpisodeStep &step, std::string *err);
        size_t steps() const { return seat_.size(); }
        bool omniscient() const { return omniscient_; }
        // The file: kind `episode`, the spec's and vocab's hashes, `model` = `text_model`, the
        // writer's attrs then `attrs`, the arrays.
        RmtbFile file(const std::string &text_model, const std::vector<RmtbAttr> &attrs) const;

    private:
        // One featurization's compact arrays.
        struct Tokens {
                std::vector<int64_t> tok_offsets{0}, rel_offsets{0}, num_offsets{0};
                std::vector<int64_t> tok_type, num_index, cat, card, ability, rel;
                std::vector<float> num_value;  // each exactly an f16 value
        };
        // One decision's featurization as compact arrays (`out` fresh: offsets {0, n} and {0, r}).
        bool compact(const Tensors &t, Tokens *out, std::string *err) const;
        static void append(const Tokens &one, Tokens *to);
        static void append_list(
            const std::vector<int64_t> &one, std::vector<int64_t> *values, std::vector<int64_t> *offsets);
        void tokens_arrays(const Tokens &t, const std::string &prefix, RmtbFile *f) const;

        const FeatureSpec &spec_;
        bool omniscient_;
        std::vector<int64_t> seat_, game_, decision_, sideboarding_, chosen_, has_mu_, policy_version_, teacher_;
        std::vector<int64_t> menu_offsets_{0};
        std::vector<float> mu_;
        Tokens seat_tokens_, omni_tokens_;
        std::vector<std::vector<PolicyEvalRecord>> policies_;
        size_t max_policies_ = 0;
        std::vector<int64_t> opp_list_, opp_archetype_;
        std::vector<int64_t> opp_hand_, opp_hand_offsets_{0}, own_library_, own_library_offsets_{0}, opp_library_,
            opp_library_offsets_{0};
};

// An episode read back: its file (meta, attrs) and every step with the tensors expanded to the
// featurizer's layout (masks all ones, act_tok rebuilt, tok_num from f16).
struct Episode {
        RmtbFile file;
        bool omniscient = false;
        size_t policies = 0;  // P: the eval_* arrays' policy dimension (0: none)
        std::vector<EpisodeStep> steps;
};

// Reads an episode; false with `err` when the file is not an episode of this version or its
// arrays are inconsistent.
bool read_episode(const RmtbFile &file, Episode *out, std::string *err);

}  // namespace featurize

#endif /* FEATURIZE_EPISODE_H */
