#ifndef ROBOMAGE_CHECK_ACTOR_H
#define ROBOMAGE_CHECK_ACTOR_H

#include <string>
#include <vector>

#include "gate.h"
#include "proc.h"

// ── robomage_check: the actor binary and the reference policy (plan step 5d, docs/ci.md) ──────
//
// The `actor` tier and the `bench` tier's evaluation stage need bin/<config>/robomage_nn (`make
// actor`, beside the engine) and a model of the network contract. The model is the reference
// policy, built by tools/nn/make_reference_policy.py with the Python that $ROBOMAGE_TORCH_PYTHON
// names (torch, onnx, onnxscript, onnxruntime): the opt-in. Without either the tiers print a
// `skipped:` line, never a warning. Under <out>/actor/: the whole-vocab cards file
// (robomage --cards-frame), the exported tables (rmfeat --export-tables) and the exports ref.ts.pt,
// ref.pt2 and ref.onnx, rebuilt only when the stamp beside them (the spec hash and the script's
// FNV-1a 64) changes.

namespace rmcheck {

// The opt-in: the Python the reference policy is built and checked with.
extern const char *const TORCH_PYTHON_ENV;

struct ActorSetup {
        std::string nn_binary;  // bin/<config>/robomage_nn
        std::string python;     // $ROBOMAGE_TORCH_PYTHON
        std::string script;     // tools/nn/make_reference_policy.py
        std::string dir;        // <out>/actor
        std::string cards;      // the whole-vocab cards file
        std::string tables;     // the exported tables (spec.json among them)
        std::string prefix;     // the exports' path prefix
        std::string ts() const { return prefix + ".ts.pt"; }
        std::string pt2() const { return prefix + ".pt2"; }
};

// The actor binary and the reference policy, built when missing or stale. False with `skip` (why,
// for a `skipped:` line) when the binary or the opt-in is absent; false with `skip` empty after an
// error reported under `tier`.
bool actor_setup(const Options &opt, Report &rep, const char *tier, ActorSetup *out, std::string *skip);

// An inference server (rmserve, tools/nn/serve_check.py) run as a session.
struct ServerRun {
        Session session;
        std::string socket;
};

// Starts the server `argv` (in the engine's working directory) and waits for its "listening on"
// line; false with `err`.
bool start_server(const Options &opt, const std::vector<std::string> &argv, const std::string &socket, ServerRun *s,
    std::string *err);
// Sends the server `sig` and collects its output (its log is stderr); a server still running 30 s
// later is killed and the result says it timed out.
ProcResult stop_server(ServerRun *s, int sig);

// The policy projection (src/agents/policy_projection.h) on known answers and random menus; the
// actor tier runs them with or without its opt-in (projection.cpp).
void projection_checks(Report &rep);
// The evals= file of an nn: agent run with `options` (temp=<temp>:thresh=<thresh>:disc=<disc>):
// every decision's mu is the projection of softmax(logits / temp), a distribution of multiples of
// 1/disc, rows below thresh 0, the chosen row with mass.
void check_projected_evals(
    Report &rep, const std::string &path, const std::string &options, double temp, double thresh, int disc);

// The actor tier's remote evaluation checks (tier_actor_remote.cpp): bin/<config>/rmserve beside
// robomage_nn, the Python server tools/nn/serve_check.py, under <out>/actor/remote/.
void actor_remote_checks(const Options &opt, Report &rep, const ActorSetup &a);

// The actor tier's hot reload check (tier_actor_reload.cpp, agents/actor.h): robomage_nn --actor
// picking up a replaced in-process model between matches, under <out>/actor/reload/.
void actor_reload_check(const Options &opt, Report &rep, const ActorSetup &a);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_ACTOR_H */
