#ifndef AGENTS_AGENT_H
#define AGENTS_AGENT_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "decision_view.h"
#include "py_random.h"

// ── In-process agents (plan step 3a) ──────────────────────────────────────────────────────────
//
// An agent decides one seat's decisions inside the engine process, through InputLogger's
// in-process decision hook (agents/driver.h installs it). At each decision it receives the
// deciding seat's frames, decoded: exactly the RMSTATE frames the seat would receive on the
// stream (docs/state_format.md §9), knowledge tables included (the in-process feed,
// state_format/agent_feed.h). An agent reads nothing else of the engine: this header and every
// agent's own files include only the standard library, the state format's decoder and each
// other.
//
// Specs (--agent-a / --agent-b; make_agent):
//
//   first (auto)        always row 0 (pass priority, the first choice)
//   random              a uniform row: random.randrange(n) on the shared generator (its behaviour
//                       distribution: uniform)
//   actions:<i,i,...>   the listed replies for this seat's decisions in order (a concession
//                       sentinel -2 / -3 is sent as is), then row 0
//   play:<specs>        the --play resolver (agents/play_spec.h) on this seat's decisions only,
//                       then row 0
//   explore             the coverage fuzzer, python/explore_agent.py ported (agents/explore.h)
//   explore:patient     its big-mana profile
//   nn:<model>[:...]    a learned policy (docs/featurizer.md §6), or nn:remote:<socket>[:...]
//                       evaluated by an inference server: a kind the actor binary
//                       bin/<config>/robomage_nn registers (src/actor/nn_agent.h); this binary
//                       refuses it, naming `make actor`
//
// A binary that links the engine adds an agent kind with register_agent_kind before the engine
// starts (engine_main.h): its specs are `<prefix>` followed by what the kind parses.
//
// The shared generator: every agent of the process draws from one PyRandom, seeded once before
// the match with the game seed (--game-seed), else the engine seed (--seed), as the Python
// drivers seed their global `random` before each match. A seat's agent and the other seat's draw
// from the same stream in decision order, as the Python explore tiers of both seats do.

namespace featurize {
struct Tensors;
}

namespace agents {

// One decision of the seat an agent decides.
struct Decision {
        const DecisionView &view;  // the seat's decoded STATE frame (menu row i = reply i)
        size_t number;             // decisions read before this one in the run
};

// What decide() returns when the agent cannot decide: the run ends with failure() as a fatal
// error.
constexpr int AGENT_FAILED = -1000;
// What decide() returns to end the run at this decision with no result (a human seat's quit, or
// the end of its input): the decision log records the stop, so the log replays up to it, and the
// run exits with status 0.
constexpr int AGENT_QUIT = -1001;

// One policy's evaluation of a decision, as a policy agent received it (the nn: agent's, per
// requested policy: the first is the one sampled from).
struct PolicyEvaluation {
        int64_t id = 0;  // the server's policy id (0 in process)
        uint64_t version = 0;
        float value = 0;
        std::vector<float> log_pi;  // over the menu rows
};

// What a behaviour agent sampled its last decision from: the exact distribution over every menu
// row (a policy's after temperature and any projection, one-hot at the row an argmax took; the
// random agent's uniform one) and the version of the policy that gave it, for the trajectory writer
// (agents/trajectory.h) recording the behaviour policy. A policy agent also gives its evaluations
// and the tensors it featurized the decision's frame into (with the spec's hash), which the writer
// reuses instead of featurizing the frame again.
struct Behaviour {
        std::vector<double> mu;
        uint64_t policy_version = 0;
        std::vector<PolicyEvaluation> policies;
        const featurize::Tensors *tensors = nullptr;
        uint64_t spec_hash = 0;
};

class Agent {
    public:
        virtual ~Agent() = default;
        // A menu row, -1 (the last row: confirm), a concession sentinel (-2 game, -3 match),
        // AGENT_QUIT or AGENT_FAILED.
        virtual int decide(const Decision &d) = 0;
        // A short label for traces ("first", "explore", ...).
        virtual std::string label() const = 0;
        // The last decision's behaviour distribution (a policy agent), or nullptr.
        virtual const Behaviour *last_behaviour() const { return nullptr; }
        // Before each match the agent plays (agents_install; an actor's agents play many): drops
        // whatever it kept from an earlier match (an in-process nn: model reloads its file when the
        // file changed). False with `err` ends the run with a fatal error.
        virtual bool begin_match(std::string *err) {
            (void)err;
            return true;
        }
        // Why decide() returned AGENT_FAILED.
        const std::string &failure() const { return failure_; }

    protected:
        std::string failure_;
};

// The agent `spec` names, drawing from `rng`; nullptr (with `err`) for an unknown spec.
std::unique_ptr<Agent> make_agent(const std::string &spec, PyRandom *rng, std::string *err);

// Makes an agent of a registered kind from `args` (the spec after the kind's prefix); nullptr with
// `err` on bad arguments.
using AgentFactory = std::unique_ptr<Agent> (*)(const std::string &args, PyRandom *rng, std::string *err);
// Adds the agent kind of specs starting with `prefix` (matched case-insensitively, e.g. "nn:").
void register_agent_kind(const std::string &prefix, AgentFactory make);

// Plays fixed replies in order, then row 0 (the actions: spec, and the global --actions script
// when given players: the seats' agents decide once the replies run out).
class ActionsAgent : public Agent {
    public:
        explicit ActionsAgent(std::vector<int> actions, Agent *player_a = nullptr, Agent *player_b = nullptr);
        int decide(const Decision &d) override;
        std::string label() const override { return "actions"; }
        // The behaviour of the seat's agent when it decided the last decision, else nullptr.
        const Behaviour *last_behaviour() const override { return delegate_ ? delegate_->last_behaviour() : nullptr; }

    private:
        std::vector<int> actions_;
        size_t next_ = 0;
        Agent *players_[2];
        const Agent *delegate_ = nullptr;  // the seat's agent that decided the last decision
};

// Parses a comma-separated integer list ("0,1,-2"); false on a bad entry.
bool parse_int_list(const std::string &csv, std::vector<int> *out);

}  // namespace agents

#endif /* AGENTS_AGENT_H */
