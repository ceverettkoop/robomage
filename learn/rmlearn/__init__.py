"""rmlearn: the learner side of docs/rnad_training_plan.md (PyTorch; never imported by python/,
the engine or the gate). Modules: rmtb (RMTB files in numpy), spec (FEATURE_SPEC and the card
tables), batch (decisions, padding, targets), model (the network, auxiliary heads, smoothed
policy), export (TorchScript / AOTInductor exports), server (the RMIP v2 inference server)."""
