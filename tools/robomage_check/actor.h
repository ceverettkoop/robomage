#ifndef ROBOMAGE_CHECK_ACTOR_H
#define ROBOMAGE_CHECK_ACTOR_H

#include <string>

#include "gate.h"

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

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_ACTOR_H */
