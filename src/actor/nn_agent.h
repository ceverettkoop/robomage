#ifndef ACTOR_NN_AGENT_H
#define ACTOR_NN_AGENT_H

#include <memory>
#include <string>
#include <vector>

#include "../agents/agent.h"
#include "../featurize/featurizer.h"
#include "../featurize/setup.h"
#include "evaluator.h"

// ── The nn: agent: featurize the seat's frame, evaluate, choose a menu row (plan step 5d) ─────
//
// docs/featurizer.md §6. Spec nn:<model>[:<key>=<value>…] (agents/agent.h; registered by the actor
// binary, bin/<config>/robomage_nn, with register_nn_agent):
//
//   <model>      a TorchScript (.ts.pt) or AOTInductor (.pt2) export of the network contract
//                (evaluator.h), relative to the engine's working directory (bin/)
//   temp=<t>     sampling temperature, t > 0 (default 1): row a with probability
//                softmax(logits / t)[a], drawn with the agents' shared generator (agents/agent.h:
//                random.choices on the probabilities), so a game replays from its seed
//   argmax       take the row of the largest logit, no draw; rows within 1e-5 (absolute plus
//                relative) of it tie and the first is taken, so the float drift between two
//                exports of one model does not reorder rows that are equal in exact arithmetic
//   dump=<file>  write every decision this agent featurized, at process exit, as one RMTB tensors
//                file (featurizer.h tensors_file): rmfeat --featurize over a --rmstate-dump of a
//                --replay of the game's decision log writes the same bytes for this seat
//   stats        at process exit, an NN-STATS: line (decisions, seconds featurizing and
//                evaluating) on stdout
//
// An unknown key or a bad value fails loudly; so does nn:remote:… (the remote evaluator is plan
// step 5d part 2). The featurizer's inputs come from the engine in process (the whole-vocab CARDS
// frame and the committed text table: state_format/cards_file.h load_engine_featurization, the
// same bytes robomage --cards-frame writes and rmfeat reads), once, before the match; the model is
// refused unless its spec_hash, vocab_hash and text_model are theirs (check_model). At each
// decision the agent reads nothing but the seat's decoded STATE frame: featurize_seat on it, one
// evaluation (a batch of one), a row. It keeps the exact distribution it chose from over every
// menu row and policy 0's version (last_behaviour), one-hot under argmax. A menu of one row is
// taken with no draw. Any featurizer or evaluator failure ends the run as a fatal error.

namespace actor {

struct NnOptions {
        std::string model;
        double temp = 1.0;
        bool argmax = false;
        std::string dump;
        bool stats = false;
};

class NnAgent : public agents::Agent {
    public:
        NnAgent(NnOptions opt, agents::PyRandom *rng, std::unique_ptr<featurize::Featurization> f,
            std::unique_ptr<Evaluator> evaluator);
        int decide(const agents::Decision &d) override;
        std::string label() const override { return "nn"; }
        const agents::Behaviour *last_behaviour() const override { return have_behaviour_ ? &behaviour_ : nullptr; }
        const NnOptions &options() const { return opt_; }
        // The dump file and the stats line (at process exit).
        void finish();
        // Parses the spec's arguments (after "nn:"); false with `err`.
        static bool parse_options(const std::string &args, NnOptions *out, std::string *err);

    private:
        // The distribution over the menu rows the choice is drawn from, into behaviour_.mu.
        void distribution(const std::vector<float> &logits);
        int choose();

        NnOptions opt_;
        agents::PyRandom *rng_;
        std::unique_ptr<featurize::Featurization> featurization_;
        std::unique_ptr<Evaluator> evaluator_;
        featurize::Tensors tensors_;
        std::vector<Evaluation> evals_;
        agents::Behaviour behaviour_;
        bool have_behaviour_ = false;
        std::vector<featurize::Tensors> dumped_;
        int seat_ = -1;
        size_t decisions_ = 0;
        double featurize_s_ = 0, evaluate_s_ = 0;
};

// Adds the nn: kind to make_agent (agents/agent.h).
void register_nn_agent();

}  // namespace actor

#endif /* ACTOR_NN_AGENT_H */
