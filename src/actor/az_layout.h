#ifndef ACTOR_AZ_LAYOUT_H
#define ACTOR_AZ_LAYOUT_H

// The observation width the TorchScript evaluator (az_evaluator.cpp) sizes its input tensors by
// and checks a net's `.ts.meta.json` handshake against: the BQUERY-era observation (state vector,
// six per-action blocks, matchup tail) the existing AZ checkpoints were exported with. That
// observation is no longer built anywhere (plan step 3b); the step-5 evaluator replaces this
// constant with its own input layout.
constexpr int ACTOR_OBS_SIZE = 6922;

#endif /* ACTOR_AZ_LAYOUT_H */
