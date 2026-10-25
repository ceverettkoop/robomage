# Held-out decks

Playable decks the network is never trained on: the generalisation test (docs/open_vocab_plan.md
S7–S8). A deck reaches this directory from `decks/candidates/` through the onboarding pipeline's
stage 10, `python3 tools/onboard/onboard.py designate --deck <stem> --to heldout`, after a clean
fuzz campaign; its status file stays in `decks/candidates/<stem>.status.json`. At least one fair
and, when one is available, one combo deck stay here permanently.

`robomage --check-decks` and the gate's `vocab` tier cover these decks with `league/`'s, and the
`fuzz` and `smoke` tiers play them in their mirrors and ring (`tools/robomage_check/gate.h`,
`playable_decks`). `rmtrain`'s `league.decks` never names them.
