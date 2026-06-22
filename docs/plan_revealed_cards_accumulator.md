# Plan: Opponent revealed-cards accumulator (inferred decklist)

Goal: give the model a **persistent, perspective-relative multi-hot of every
opponent card revealed so far this match**, accumulated across the games of a
bo3. This is the engineered "belief state" — the engine computes the inference
deterministically and feeds it as observation, instead of asking a feedforward
policy to remember reveals across `reset()` (which it structurally cannot — only
4 match floats survive a game boundary today).

## Design decisions (and why)

- **Binary multi-hot, not counts.** Within one game, copies-seen can be deduped
  by entity ID, but across games a fresh ECS is created per game
  (`main.cpp:372 init_ecs()`), so entity IDs reset and "3 Bolts seen" could be the
  *same* Bolt replayed. Binary "have they ever shown card X this match" is
  unambiguous and matches how a player reasons about a closed list. Counts are an
  optional later refinement (track per-game max, take max across games).
- **Centralize the reveal hook in one chokepoint**, not the 4+ call sites the
  research found. Every zone move funnels through
  `Orderer::add_to_zone` (`src/systems/orderer.cpp:47`). Mark a reveal there when
  the destination is a **public zone** (BATTLEFIELD, STACK, GRAVEYARD, EXILE).
  Drawing to HAND/moving within LIBRARY is *not* public and is skipped. One extra
  hook covers **tutor reveals** (card made public while moving to a hidden zone),
  at `src/effects/effect_change_zone.cpp:88/121` where `card_is_public`/`reveal`
  is already computed. This honors the CLAUDE.md "consolidate iterations / reusable
  function" guidance.
- **Idempotent set, so no dedupe needed.** Setting a multi-hot bit on every
  public-zone entry is safe — a Bolt going stack→graveyard sets the same bit
  twice, harmless.
- **No realism gating required.** The accumulator is empty at game-1 turn-1 and
  fills *during* play (realistic in-game reveal), then persists across games
  (realistic memory). Unlike a "god-mode full decklist gated on `is_post_board`",
  this is honest partial information by construction.
- **Track per owner, emit opponent-of-viewer.** State is serialized
  perspective-relative (`populate_gamestate(gs, viewer)`), so store reveals keyed
  by card owner (A / B) and emit the non-viewer's array.

## C++ implementation

### 1. New match-level state (`src/classes/match_state.h` + `.cpp`)
Avoid putting data/functions in `main.cpp` (CLAUDE.md). New translation unit holds
the match-scoped accumulator and helpers:

```cpp
// match_state.h
#include "card_vocab.h"          // N_CARD_TYPES is in machine_io.h; include accordingly
constexpr int REVEALED_SIZE = N_CARD_TYPES;   // 128
extern uint8_t g_revealed_by_a[REVEALED_SIZE];
extern uint8_t g_revealed_by_b[REVEALED_SIZE];

void match_reset_revealed();                       // zero both arrays (match start)
void mark_card_revealed(Entity e, Zone::Ownership owner);  // set bit for owner
```

`mark_card_revealed` resolves the vocab index via the existing
`get_card_vocab_idx(e)` (`machine_io.cpp:31`) and sets the bit in the owner's
array (ignore idx < 0 sentinel and the TOKEN_SENTINEL slot 127).

### 2. Reveal hooks
- **`Orderer::add_to_zone` (`orderer.cpp:47`)** — after the destination is
  assigned (~line 129), if `destination ∈ {BATTLEFIELD, STACK, GRAVEYARD, EXILE}`,
  call `mark_card_revealed(target, zone.owner)`. This single hook covers casts
  (→STACK), ETB (→BATTLEFIELD), and deaths/discards (→GRAVEYARD).
- **Tutor reveal (`effect_change_zone.cpp:88/121`)** — when `reveal == true` and
  the card moves to a hidden zone (HAND/LIBRARY top), also call
  `mark_card_revealed(chosen, owner)` so a revealed-but-hidden tutored card counts.

### 3. Lifecycle
- **Match start:** call `match_reset_revealed()` before game 1 of the bo3 loop
  (`main.cpp:361`, before the `for game_num` loop) and also in the single-game
  path at game init (so the feature works without `--bo3`).
- **Between bo3 games:** do **not** reset — persistence across games is the point.
  The accumulator lives outside the per-game ECS, like `match_wins_a`
  (`main.cpp:70-74`), so `init_ecs()` doesn't touch it.

### 4. GameState + serialization
- Add `uint8_t opp_revealed[N_CARD_TYPES];` to `GameState` (`gamestate.h`).
- In `populate_gamestate(gs, viewer)` (`machine_io.cpp:86`): copy the
  **non-viewer's** match array into `gs->opp_revealed`
  (`viewer==PLAYER_A ? g_revealed_by_b : g_revealed_by_a`).
- In `serialize_state` (`machine_io.cpp`), append 128 floats right after the
  known-top-library block (`machine_io.cpp:476-483`), mirroring its encoding:
  `state.push_back(gs->opp_revealed[i] ? 1.0f : 0.0f)` for i in 0..127.
- Bump `STATE_SIZE` **33666 → 33794** (`machine_io.h:92`) and update the layout
  comment block (`machine_io.h:40-90`) to document `[33666:33794] opponent
  revealed-cards multi-hot (128, zeros = none seen yet)`.

### 5. Build
Editing headers (`machine_io.h`, `gamestate.h`) → **`make clean && make`**
(Makefile doesn't track header deps; stale `.o` → startup segfault).

## Python implementation

### 6. `train/env.py`
- `STATE_SIZE = 33794`. `OBS_SIZE` recomputes automatically.

### 7. `train/extractor.py`
- Add layout constants for the new block immediately after
  `_KNOWN_TOP_LIB_END` (33666): `_REVEALED_START = 33666`,
  `_REVEALED_SIZE = 128`, `_REVEALED_END = 33794`, and shift `_STATE_END` to
  33794. `action_extras` slicing (`obs[:, _STATE_END:]`) then stays correct.
- Add a dedicated small encoder (the block is a *dense multi-hot*, semantically
  unlike the one-hot-per-slot encoders, so don't reuse `entity_encoder`):
  ```python
  self.revealed_encoder = nn.Sequential(
      nn.Linear(_REVEALED_SIZE, embed_dim), nn.ReLU(),
  )
  ```
  Encode `obs[:, _REVEALED_START:_REVEALED_END]` → `embed_dim`, concat into the
  output, and add `+ embed_dim` to `features_dim`. (Cheaper alt: concatenate the
  raw 128 floats and add `+ _REVEALED_SIZE`; the encoder is preferred for
  representational capacity at trivial cost.)
- Update the module docstring layout map (lines 14-30).

### 8. Downstream offset consumers
- `train/analysis.py` hardcodes some layout offsets (per project memory) — audit
  and shift anything at/after 33666.
- `train/test_harness.py` decoding (if it prints the new section).
- `gen_card_costs.py` / `card_costs.py`: **unaffected** (cost matrices only).

### 9. Checkpoint impact
`features_dim` and `OBS_SIZE` both change → **existing checkpoints will not load**.
This requires retraining. (Sequence this with the embed_dim rework in the other
plan so there's a single retrain, not two.)

## Testing
- **Single-game reveal:** `test_harness.py` with opponent casting Lightning Bolt;
  decode and assert the Bolt vocab slot in the revealed block flips to 1.0 and
  stays 1.0 after it leaves the stack.
- **Cross-game persistence (bo3):** stacked decks, opponent shows a card in game 1;
  assert the slot is still 1.0 at the start of game 2 (this is the core
  correctness property — proves it survives `init_ecs()`).
- **Tutor reveal:** opponent resolves Personal Tutor (already in vocab per the
  research); assert the revealed-but-hidden card's slot is set.
- **Negative:** a card only ever in the opponent's hand/library is **not** marked.
- `train.py diag` / `watch` with `--deck`/`--opponent` for a non-fatal-error,
  no-draw smoke run.

## Optional refinements (not in v1)
- Copies-seen counts (per-game max → match max), normalized by 4.
- Feed the **viewer's own** full decklist (known info) as a second 128 block, if
  early-game self-sequencing benefits.
- A derived archetype posterior on top of the multi-hot.
