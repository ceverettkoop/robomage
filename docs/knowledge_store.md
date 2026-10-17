# Knowledge store: the actor-side memory and inference sidecar (design)

Status: step 1 of [`engine_fork_plan.md`](engine_fork_plan.md). **T9 is implemented**
(`src/knowledge/`, `tools/rmknow/`): the model, the update rules, bound propagation, the per-frame
consistency check, the `k_*` rows and serialisation; section 9 says how. The exact search and
compaction (T10), the engine integration (T11) and the gate checks (T12) are not yet. It builds on
[`state_format.md`](state_format.md): the engine
sends only what each viewer explicitly knows right now. This component supplies everything a
player with **perfect recall and perfect logical inference** would know on top of that, in one
consistent form for every kind of agent.

## 1. What it is

- **A C++ component, separate from the engine's rules state.** Its only input is the byte
  stream one viewer receives: `SCHEMA`, `CARDS`, `DECKLISTS`, then each frame's snapshot and
  observation events. Its output is a set of **knowledge tables** attached to that viewer's
  frames.
- **Level per seat:**
  - `off` — the knowledge tables are present but empty;
  - `expert` — full recall plus every sound logical deduction (section 4).

  Flags are `--knowledge-a off|expert` / `--knowledge-b …`; the default is `expert`.
- **The same output for every consumer:**
  - the C++ CLI renders it as a "you know / you can deduce" panel;
  - the scripted agent decodes it like any other table;
  - external `--machine` agents receive it inside the same `RMSTATE` frame (tables prefixed
    `k_`).
- **It cannot leak, by construction.** It never touches engine state; it is a pure function of
  the viewer's frame stream. Two rules enforce this:
  - `src/knowledge/` may include only `src/state_format/decode.h`, `schema.h` and the standard
    library; the gate checks the includes (the `rmstate` tier's isolation part);
  - a standalone tool `rmknow` replays a recorded frame stream into the same component; its
    output must equal the in-process output byte for byte.
- **Deterministic.** Its outputs are part of the frame bytes, so RMLOG replay reproduces them.
  Snapshot/restore covers it: its state is serialisable and saved with the engine snapshot, so
  a restored search root resumes with the same knowledge.

## 2. Placement in the code

| Piece | Location |
|---|---|
| core (model, update rules, solver) | `src/knowledge/` (C++17, `-fno-exceptions`), linked into `bin/robomage` and `rmknow` |
| engine integration | per-seat instance fed from that seat's outbound frame bytes, i.e. the same bytes `--machine` writes, decoded through `src/state_format/decode.*`; it appends `k_*` tables before emission |
| in-process agents and the CLI | read the decoded knowledge tables through the same decoder as everything else |
| `rmknow` | `tools/rmknow/` → `bin/<config>/rmknow` (`make rmknow`, part of `make`): `rmknow [--viewer A\|B] < frames.bin > knowledge.bin`; used by the isolation test and for offline analysis of recorded games (section 9) |

## 3. Model

### 3.1 Cards as anonymous units in pools

- Every physical card of a player is a unit. A unit whose identity the viewer cannot currently
  see is **anonymous**.
- The model never tracks *which* anonymous unit is which; it tracks **pools**. A pool is a set
  of anonymous units the viewer cannot tell apart: the opponent's hand, a stretch of library
  positions with no known order, the opponent's sideboard, a face-down exiled card.
- A **unit with known identity** sits in a pool as a fixed value: a revealed card still in the
  opponent's hand, or a known card at a known library position.

### 3.2 The constraint record

- Pools are created, merged and split by events: a draw moves units from library positions
  into the hand pool; a put-back splits units off the hand pool.
- The model keeps the **history of pool contents as a flow network over time**:
  - every pool at every moment is a node;
  - every movement of anonymous units between pools is an edge carrying a known number of
    units of unknown identity;
  - every identity seen (a reveal, a card leaving to a public zone, a look) pins units of a
    known card at that node.
- **The root of each player's network** is their registered list:
  - game 1: the deck node is exactly the main deck (open decklists);
  - games 2–3: the registered 75 split into a deck node of deck size and a sideboard node;
    the split is unknown to the opponent.
- **A world** is an assignment of a card identity to every unit that is consistent with all
  pinned observations and all flow edges. **Knowledge** is what is true in every world.

### 3.3 Viewer-specific simplifications

- The viewer's own hand and every public zone are fully visible, so only the viewer's library
  (order and contents) is anonymous on their side.
- On the opponent's side: hand, library, sideboard (games 2–3) and face-down cards.
- With `--closed-decklist` the opponent's root is "unknown cards from the vocab", so only
  recall-type deductions remain. The model is the same; the root is unconstrained.

## 4. What `expert` reports

### 4.1 Recall (always exact)

- every card seen, per game and per match, with where it went;
- known cards in the opponent's hand: units with known identity still in the hand pool;
- known library positions, per player: position k from the top or bottom holds card c;
- the viewer's own library contents: own registered list minus every own card seen elsewhere
  this game.

### 4.2 Deduction (sound always; exact when the solver finishes)

- **Accounting:** for each card and each hidden pool, the minimum and maximum number of copies
  that can be in it (e.g. "1–2 copies of Force of Will are in hand or library"; "0 in library"
  after both were seen).
- **Candidates:** for each hidden position or pool, the set of cards it can still be, e.g.
  "the top card of their library is one of {Force of Will, Brainstorm}" after a Brainstorm
  put-back from a hand known to hold those.
- **Implied facts:** a candidate set of size one means the identity is known by deduction. For
  example, they put back two cards from a hand known to be exactly {Force of Will, Daze, X},
  then cast X, so the top two are {Force of Will, Daze}.
- **Sideboarding (games 2–3):** each sideboard card seen this game proves a swap. The min/max
  per card covers deck versus sideboard.

**Excluded, and left to agent policy:** probabilities, opponent modelling, and any
"strategic" inference ("they didn't counter, so no Force of Will").

### 4.3 Solving

- **Hardness.** Exact deduction over the flow network is an integer multi-commodity flow
  problem: card identities are the commodities, pool sizes the shared capacities. It is
  NP-hard in general, but tiny in practice: at most 75 units per player, and only a handful
  of pools ever carry constraints, since the library bulk stays one unconstrained remainder.
- **Algorithm.**
  1. Propagate bounds (per-card min/max per pool through every split and merge).
  2. Run an exact feasibility search for each (pool, card, count) query that bounds do not
     settle, with memoisation, under a step budget per frame.
- **Exactness flag.** If the budget runs out, the answer is the propagated bounds. These are
  still **sound** (never exclude a possible world) but may be wider than exact. The
  `k_meta.exact` flag says which. Exceeding the budget in practice is a finding to investigate.
- **Compaction.** Once a pool's history no longer constrains anything (e.g. after a shuffle
  merges a library back into one remainder pool), the network is compacted, so its size stays
  bounded over a long game.

## 5. Update rules per event kind

Every rule must be computable from the event's fields alone. This table is the step-1 check
that the event schema is complete. Library positions are (anchor, pos, span) as in
state_format §5: the model tracks each library's size, so it moves position facts as cards
enter or leave above them and converts between top and bottom.

| Event (state_format §5) | Update |
|---|---|
| `game_start` | new networks: deck node from the root (section 3.2); library pool = the stated library count of units from it; the rest of the main deck is a set-aside pool (test-harness presets, which arrive as `zone_move` from `none`); sideboard pool = the stated sideboard count (games 2–3: the registered 75 split, unknown to the opponent). Hands start empty. Game 1 roots the main deck and the sideboard separately, each with its own set-aside rest (a companion comes from the sideboard's); the viewer's own root is the registered list in game 1 (it holds the harness presets, which `own_deck` does not) and `own_deck` in games 2–3 |
| `draw` (own) | known cards leave the top of own library one at a time; positions shift; known-position facts move up |
| `draw` (opponent) | an anonymous unit moves from the top of their library into the hand pool (each card drawn is its own `draw` event). A known top card becomes a known card in hand; a card in the viewer's live window arrives with its identity. Unless the opening-hand flag is set, the units also join the hand's **drawn-this-turn** sub-pool for the event's `turn`; a unit leaves it when it leaves the hand, and the sub-pool empties when `turn` advances |
| `zone_move` to a public zone | pin the identity on the source pool (at the stated library position or range; e.g. a card cast from hand proves the hand held it), then remove the unit. A card the viewer holds a handle for is exactly that unit. From a hand (or another set zone) without a handle: a unit already known to be that card if there is one (copies are interchangeable), else the sub-pools whose bounds allow the card, merged when there are several; handled units are never candidates (a handled card moves with its handle). The unit stays in the network as a public unit keyed by its handle, so a later move back to a hidden zone continues it |
| `zone_move` public → hidden with handle | a unit with known identity enters the destination at the stated position (a bounce to hand: known card in hand) |
| `zone_move` from a hand with subset `drawn_this_turn` | as the matching rule below, but the unit comes from the hand's drawn-this-turn sub-pool, so the rest of the hand (e.g. cards known from an earlier reveal) is untouched (Sylvan Library) |
| `zone_move` hidden → hidden, no identity | one anonymous unit leaves the source pool and enters the destination at the stated position (a Brainstorm put-back: the top k of their library are units from the hand pool). From a library range of span > 1, those positions first merge into one unordered pool (the opponent's scry: which looked-at card went where is unknown) |
| `zone_move` in a batch | as above, but the destination position waits for the batch's `library_place`. Every card the instruction put into that library is in the batch for the viewer, identified or not |
| `zone_move` of a token or copy | no unit; nothing changes |
| `zone_move` from `none` of a card | a set-aside unit (harness preset) is pinned and placed |
| `library_place` | the batch's units occupy the stated range; since their order is hidden, the range becomes one unordered pool holding them (known identities pinned as pool contents). The batch's cards are placed at the end the range names as they arrive (the `library_place` is in the same frame: one instruction, no decision in between), so later cards' source positions count them as the engine does |
| `reveal_start` | pin the identities at the stated zone and positions. A card with a draw link pins the unit that draw brought in, and so the library position it was drawn from (Miracle). A shown hand or sideboard card lands on a distinct unit: the one its handle already names, else an unhandled unit known to be that card, else one split off the sub-pools that can hold it |
| `reveal_end` | nothing to forget: recall keeps the pinned facts; visibility ends but knowledge stays |
| `look` (by the viewer) | pin identities at positions. A search's find is shown at its position in the already-shuffled library; when it leaves, the two parts of the run around it rejoin (they are interchangeable) |
| `look` (by the opponent) | no identity; marks that the opponent knows those positions (for agents; no constraint). A card's position is unknown (span 0) when the opponent looked at it by identity (a search's find after the shuffle) |
| `reorder` | the reordered range becomes one unordered pool, unless the viewer saw the result (then each position is pinned) |
| `shuffle` | all positions of that library merge into one unordered remainder pool; position facts are dropped; contents facts are kept |
| `search` | nothing by itself (a look at the whole zone, when there is one, comes as `look`); the cards leave as `zone_move`s, usually followed by a `shuffle` |
| `cast_from_hidden` / `play_from_hidden` | as `zone_move` to a public zone |
| `face_down` / `face_up` | a face-down card is an anonymous singleton pool (with its source constraint), keyed by its handle; face-up pins it. Exiling face down is a `zone_move` with the face-down flag, with the same pool |
| mulligan (hand to library, shuffle, draw 7, London bottoming) | composed of the rules above |
| `game_end` / `match_end` | match memory updated; networks reset at the next `game_start` |

Each frame's snapshot is also a check: the model's pool sizes must equal the snapshot's zone
counts, and every identity the snapshot shows must agree with the model. A mismatch is a
missing event or a bug, reported loudly.

## 6. Output tables (schema fixed for both levels)

| Table | Rows | Columns |
|---|---|---|
| `k_meta` | 1 | level, exact flag, solver steps used |
| `k_seen` | per (player, card) | copies seen this game, copies seen this match |
| `k_pool` | per hidden pool | owner, kind (hand / library range / sideboard / face-down / set-aside), size, position range if a library range, handle if face-down |
| `k_pool_card` | per (pool, card) with max > 0 | min copies, max copies |
| `k_position` | per library position with a known identity or a narrowed candidate set | owner, position (from top or bottom), card if exact, pool id otherwise |
| `k_known_hand` | per known card in the opponent's hand | card, copies known |

Positions refer to the snapshot's library positions; public objects use the snapshot's handles.
Columns: state_format Appendix A.6.

As built (T9, `Knowledge::rows`, `src/knowledge/output.cpp`):
- **`k_meta` is always one row**, at both levels: `off` writes it with level off and every other
  table empty (so omitted), so a consumer always finds the level. `exact` is 0 until T10's search.
- **Rows go self first, then the opponent**, each in a fixed order: hand, library runs top to
  bottom, sideboard, face-down cards (by handle), set-aside.
- **`k_pool`**: one row for the opponent's whole hand and one for their whole sideboard (the
  viewer's own hand and sideboard are snapshot objects, not pools); one per library run (a run of
  positions whose order is unknown, `range` = its positions from the top) whose card is not
  determined; one per face-down card; one for the set-aside units while any are left.
- **`k_pool_card`**: per pool, every card of the player's universe (the registered list, or the
  cards seen under `--closed-decklist`) with max > 0; an aggregate pool (hand, sideboard,
  set-aside) sums its sub-pools' minima and maxima (capped at its size). Under
  `--closed-decklist` a card never seen is possible too without a row.
- **`k_position`**: every library position whose run's card is determined (`card` set, a run whose
  minimum of one card equals its size, so a run of several known copies gives a row per
  position), and every one-position run that is not (`pool` = its `k_pool` row). `at` is always
  anchored at the top (a consumer converts with the snapshot's library count).
- **`k_known_hand`**: per card, the sum of the opponent's hand sub-pools' minima: recall and
  deduction alike ("definitely in hand").
- **`k_seen`**: per (player, card) with a nonzero count, the fewest copies the observations of
  the game imply (`game`, section 9) and the most of that over the match's games (`match`).

## 7. Verification

- **Soundness (zero tolerance):** at every decision of the corpus, smoke and fuzz games,
  compare against the omniscient viewer's truth. The true world must satisfy every reported
  fact: exact identities, candidate sets, min/max counts.
- **Exactness (the solver):** on small synthetic and sculpted cases (≤ 15-card decks), a
  naive enumerator over the same pool/flow model, written independently of the solver,
  lists every consistent world. With `k_meta.exact` set, the reported min/max and candidate
  sets must equal the enumeration.
- **Sharpness (the model):** hand-written regression scenarios with the deduction a strong
  player would make, asserted on the knowledge dump. Example: after a Brainstorm put-back
  from a hand known to hold {Force of Will, Daze, X}, and X then cast, the top two cards are
  exactly {Force of Will, Daze}.
- **Isolation:** `rmknow` on the recorded frame stream gives byte-identical knowledge tables
  to the in-process run.
- **Shadow comparison with today's engine memory:** while the old vector is still emitted, at
  every decision compare `expert`'s implied view (own exact positions 0–4; definitely-in-hand
  opponent cards; cards seen this match) with the engine's `known_top_library_*`,
  `opp_known_hand` and revealed bits. Every mismatch is classified as either:
  - a sidecar or event bug: fixed; or
  - an old-vector leak or quirk: documented, with a regression scenario.

  **One leak is already known.** `Zone::identity_known` is tracked per physical card. After an
  opponent's Brainstorm put-back, the old vector still lists a known Force of Will in hand only
  if that particular card stayed in hand, which tells the viewer which cards went back.
  `expert` reports "Force of Will: in hand or in the top two". The shadow comparison must
  classify every such case.
- **Snapshot/restore:** the knowledge state is saved and restored with the engine snapshot,
  and frames after a restore match the originals.

## 8. Effect on the scripted-agent port (plan step 2)

- **The agent reads knowledge from `expert`, not from an `engine-compat` level.** The
  read map's two knowledge reads come from the knowledge tables:
  - own known top 0–2 and the Thassa's Oracle check: `k_position` (own, exact);
  - `opp_known_hand`: `k_known_hand`.
- **Parity target:** byte-identical transcripts, except in games where a classified old-vector
  leak changed what the agent saw. Those games are re-recorded, each with its classification
  as the reason.

## 9. Implementation (step 1 T9)

Code: `src/knowledge/` (std-only; `Knowledge` in `knowledge.h` is the entry point) and the
standalone driver `tools/rmknow/`.

### 9.1 Data model (section 3)

| Struct (file) | What it is |
|---|---|
| `Node` (`network.h`) | a set of one player's units the viewer cannot tell apart, at one moment: size, pin (the known card of a one-unit node), the junction that made it and the one that consumed it, root index, and while live its location (`Loc`), handle, drawn-this-turn turn and draw seq |
| `Junction` | `in` units = `out` units (a split: one in; a merge: one out) |
| `Root` | a node no junction made: `EXACT` contents (a registered list), `LOWER` bounds (a closed decklist after cards were seen this match), or `FREE` |
| `Network` | one player's nodes, junctions, roots and the log of pins added to existing nodes; nodes never change, so the network is the flow-over-time record of section 3.2 |
| `PlayerModel` (`player_model.h`) | the network plus where its live nodes are: the library as a sequence of runs (top first; a run is one node filling that many consecutive positions in an unknown order; a one-unit run is an exact position), the hand, sideboard and set-aside as sets of sub-pools, face-down and public units (one each, keyed by handle) |
| `BoundStore` (`bounds.h`) | per node and per card class of the player's `Universe` (the cards tracked, plus one OTHER class), the least and most copies; two per player: `full` (roots constrain: the deductions) and `seen` (roots free: what was observed) |
| `Knowledge` (`knowledge.h`) | the seat's sidecar: match constants (schema, decklists), match memory (copies seen per match), both players' models, the viewer's handle → unit map, draw seq → unit map |

Every unit is in exactly one live node, so the live nodes always sum to the roots: the **current
cut**, which propagation treats as one more junction. It is what turns "one Force of Will was cast
from an unknown hand" into "at most three in the library".

When the viewer cannot tell which sub-pool of a set zone a card left (a known card cast from a
hand holding this turn's draws and older cards that could each be it), the candidates merge first:
sound and exact for the multiset, but which hand units were drawn this turn is then unknown for the
rest of the turn, so a later drawn-this-turn pick (Sylvan Library) falls back to the whole hand.

Library operations go through the network: taking the unit at a position splits it off its run,
inserting into a run cuts the run there, a position range the viewer cannot tell apart (a scry
pick, a reorder) merges into one run, a shuffle merges the whole library, a pinned position is a
three-way split of its run. Two runs that become adjacent again and are unpinned parts of one split
rejoin (exact: nothing else constrains them).

### 9.2 Bound propagation (T9) and what T10 adds

`BoundStore::propagate` narrows intervals to a fixpoint with a worklist: a node's size (its classes
sum to it), pins, roots, every junction (each member's interval from the other members' sums) and
the cut. Constraints only accumulate within a game, so the store keeps its bounds between frames
and only processes nodes, junctions and pins added since (a card entering the universe resets
it). Bounds are sound always; they are per-card intervals per node, so they miss joint facts
(after a put-back from {Force of Will, Daze, X} and X cast, each of the top two is one of {Force of
Will, Daze}, but "one of each" is a joint fact) and they lose strength through merges of chains.

T10 hooks:
- **Exact search**: a memoised feasibility search per (node set, card, count) over the same
  `Network`, seeded by the propagated bounds; `Knowledge::pool_rows` and `single_card` are where
  its answers replace the interval sums (aggregates and runs), under the per-frame budget, setting
  `k_meta.exact`.
- **Compaction**: eliminating a node with no pin is exact when it is the only output of the
  junction that made it or the only input of the one that consumed it (its non-negativity is then
  implied); a node that is a split's child and a merge's input cannot be dropped that way. After a
  shuffle the history behind the merged run is the main target. Stores reset after compaction.

### 9.3 Sizes, steps, serialisation

On the replay corpus and the explore fuzz a game grows to at most ~750 nodes per player and a
frame's propagation to ~3.5k constraint evaluations (budget 4M per store per frame).
`Knowledge::serialize` writes a self-contained little-endian blob (`RMKN`, version 1): level,
viewer, the SCHEMA payload (re-decoded on load), the decklists, match memory, both
`PlayerModel`s (network, both stores' bounds, zone lists) and the handle and draw maps; card names
are not included. `deserialize` refuses a malformed blob. A blob is ~150-200 KB late in a game
(the bounds dominate; T10's compaction shrinks it).

### 9.4 Copies seen

`k_seen.game` is the fewest copies of the card the player can own given only the observations:
the larger of the `seen` store's sums over the live nodes and over the roots (roots free, so
deck-list arithmetic never counts as seen). A card revealed, shuffled away and then cast counts
once; two copies on the battlefield count twice. `match` is the maximum over the match's games;
under `--closed-decklist` it becomes the next game's root lower bounds.

### 9.5 Verification so far

- **Consistency** (`Knowledge::check`, every STATE frame of a live game): hand, library and
  sideboard sizes; every public and face-down object is a model unit with the same card; the
  viewer's own hand is exactly the model's pinned units; the own sideboard's counts are within
  bounds; every reveal row's card is the model's card at that position or handle; and the
  propagation finds no contradiction. A mismatch is an `ERROR: knowledge[...]` line.
- **`rmknow --truth`** checks every reported fact against the same dump's omniscient frames
  (pool contents within min/max and no unlisted universe card, exact positions, known-hand
  counts, pool sizes, seen ≤ owned).
- **`rmknow --selftest`**: synthetic event streams (`tools/rmknow/selftest.cpp`): the section 7
  put-back example, cut accounting, a scry range, a batch, drawn-this-turn, a draw link, the
  viewer's own rearrangement, the search rejoin, a London mulligan, closed-decklist match memory,
  serialisation round trip.

