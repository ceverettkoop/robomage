# Knowledge store: the actor-side memory and inference sidecar (design)

Status: step 1 of [`engine_fork_plan.md`](engine_fork_plan.md). **T9, T10 and T11 are
implemented** (`src/knowledge/`, `tools/rmknow/`, `src/state_format/knowledge_feed.*`): the model,
the update rules, bound propagation, the per-frame consistency check, the `k_*` rows and
serialisation (T9), the exact solver, compaction and the enumerator tests (T10), and the engine
integration: per-seat instances appending `k_*` to every seat frame, the snapshot hook and
`rmknow --isolation` (T11); section 9 says how. The gate checks (T12) are not yet. It builds on
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
  - `off` — the knowledge tables are present but empty: the `k_meta` row alone, with level off;
  - `expert` — full recall plus every sound logical deduction (section 4).

  Flags are `--knowledge-a off|expert` / `--knowledge-b …`; the default is `expert`. The level
  changes only the seat's `k_*` tables, never play.
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
| engine integration | `src/state_format/knowledge_feed.{h,cpp}` (engine side): one instance per RMSTATE frame writer (`--rmstate-dump`, `--machine-format rmstate`, `--selfcheck-state`) and seat, fed with exactly the bytes that writer builds for that seat, decoded through `src/state_format/decode.*`; it appends the `k_*` tables to each of the seat's STATE frames before the writer emits it (section 9.6) |
| in-process agents and the CLI | read the decoded knowledge tables through the same decoder as everything else |
| `rmknow` | `tools/rmknow/` → `bin/<config>/rmknow` (`make rmknow`, part of `make`): `rmknow [--viewer A\|B] [--level off\|expert] < frames.bin > knowledge.bin`; `--isolation` recomputes every frame's `k_*` from the frames and compares them with the engine's (the isolation test); offline analysis of recorded games (section 9) |

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
- **Algorithm** (as built, `src/knowledge/solver.h`; section 9.2).
  1. Propagate bounds (per-card min/max per node through every junction and the cut).
  2. Per frame and player, for every pool the tables may report: compact a copy of the network
     down to those pools' nodes, decompose it into independent components, and settle each
     pool's least and most copies of each card per component by a search over interval states
     whose leaves are transportation problems with closed-form projections. Each answer the
     leaves have not reached is searched for with the state narrowed to it: a leaf reaching it
     settles it, an empty search proves the bound one step tighter. A per-card network-flow
     relaxation bounds every answer and prunes those searches. Components are memoised across
     frames.
- **Budget.** 1,000,000 steps (constraint evaluations, search nodes, flow augmentations and
  passes) per player and frame: at most about 12 ms in a release build (80 ms in a debug one).
  The most any frame of the corpus and fuzz games needed is 162,622 (about 2 ms release).
- **Exactness flag.** If the budget runs out, the answers not yet settled are the propagated
  bounds (narrowed by whatever the search proved). These are still **sound** (never exclude a
  possible world) but may be wider than exact. `k_meta.exact` is 1 when every `k_pool_card`,
  `k_position` and `k_known_hand` fact of the frame is the exact projection of the consistent
  worlds. Exceeding the budget in practice is a finding to investigate: on the replay corpus and
  the fuzz games no frame does (section 9.3). The hard cases are a hand mixed over many turns
  (each turn's draw merged into a hand partly known, a known card cast from it), a chain of
  junctions joined by hand sub-pools: conservation of a card along the chain is what bounds the
  answers, which per-card intervals cannot see but the flow relaxation does.
- **Compaction.** History that no longer constrains anything is eliminated from the network
  every frame it changed, exactly (the consistent worlds projected onto the live nodes stay the
  same), so its size stays bounded over a long game: after a shuffle the history behind the
  merged run collapses into one junction (section 9.2).

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
  table empty (so omitted), so a consumer always finds the level. `exact` is 1 when the solver
  settled every reported pool (section 4.3; also between games, with no pool to report), 0 at
  level off. `steps` counts the frame's propagation and solver steps.
- **Rows go self first, then the opponent**, each in a fixed order: hand, library runs top to
  bottom, sideboard, face-down cards (by handle), set-aside.
- **`k_pool`**: one row for the opponent's whole hand and one for their whole sideboard (the
  viewer's own hand and sideboard are snapshot objects, not pools); one per library run (a run of
  positions whose order is unknown, `range` = its positions from the top) whose card is not
  determined; one per **block** of runs from the top (runs 0 to j) or from the bottom (runs i to
  the last), short of the whole library, that says more than its parts: some card's least or most
  copies in it is sharper than the block one run shorter (or that run alone) plus the run it
  adds; one per face-down card; one for the set-aside units while any are left. Blocks carry the
  joint facts per-run rows cannot: after a put-back of two cards from a hand known to be {Force of
  Will, Daze, X} and X cast, each of the top two runs holds Force of Will 0-1 and Daze 0-1, and
  the block of the top two holds exactly one of each (a 2-unit pool with min = max = 1 for both).
- **`k_pool_card`**: per pool, every card of the player's universe (the registered list, or the
  cards seen under `--closed-decklist`) with max > 0: the solver's least and most copies of the
  card in the pool (an aggregate pool, such as the hand, as a whole). Under `--closed-decklist` a
  card never seen is possible too without a row.
- **`k_position`**: every library position whose run's card is determined (`card` set, a run whose
  minimum of one card equals its size, so a run of several known copies gives a row per
  position), and every one-position run that is not (`pool` = its `k_pool` row). `at` is always
  anchored at the top (a consumer converts with the snapshot's library count). A position's
  candidates are its run's cards with max > 0 (any order of a run is possible, so this is exact
  for every position of a run when the solver is).
- **`k_known_hand`**: per card, the opponent's hand pool's least copies: recall and deduction
  alike ("definitely in hand").
- **`k_seen`**: per (player, card) with a nonzero count, the fewest copies the observations of
  the game imply (`game`, section 9) and the most of that over the match's games (`match`).

## 7. Verification

- **Soundness (zero tolerance):** at every decision of the corpus, smoke and fuzz games,
  compare against the omniscient viewer's truth. The true world must satisfy every reported
  fact: exact identities, candidate sets, min/max counts.
- **Exactness (the solver):** on small synthetic and sculpted cases (≤ 15-card decks), a
  naive enumerator over the same pool/flow model, written independently of the solver,
  lists every consistent world. With `k_meta.exact` set, the reported min/max and candidate
  sets must equal the enumeration. Built as `rmknow --exactness N` (section 9.5).
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

## 9. Implementation (step 1 T9, T10, T11)

Code: `src/knowledge/` (std-only; `Knowledge` in `knowledge.h` is the entry point), the
standalone driver `tools/rmknow/` and the engine side `src/state_format/knowledge_feed.*`.

### 9.1 Data model (section 3)

| Struct (file) | What it is |
|---|---|
| `Node` (`network.h`) | a set of one player's units the viewer cannot tell apart, at one moment: size, pin (the known card of a one-unit node), the junction that made it and the one that consumed it, root index, and while live its location (`Loc`), handle, drawn-this-turn turn and draw seq |
| `Junction` | `in` units plus `in_fixed` cards = `out` units plus `out_fixed` cards (a split: one in; a merge: one out; after compaction any number on either side, the fixed cards being pinned units compaction replaced by their card) |
| `Root` | a node no junction made: `EXACT` contents (a registered list), `LOWER` bounds (a closed decklist after cards were seen this match), or `FREE` |
| `Network` | one player's nodes, junctions, roots and the log of pins added to existing nodes; nodes never change, so the network is the flow-over-time record of section 3.2 |
| `PlayerModel` (`player_model.h`) | the network plus where its live nodes are: the library as a sequence of runs (top first; a run is one node filling that many consecutive positions in an unknown order; a one-unit run is an exact position), the hand, sideboard and set-aside as sets of sub-pools, face-down and public units (one each, keyed by handle) |
| `BoundStore` (`bounds.h`) | per node and per card class of the player's `Universe` (the cards tracked, plus one OTHER class), the least and most copies; two per player: `full` (roots constrain: the deductions) and `seen` (roots free: what was observed) |
| `Solver` (`solver.h`) | the exact solver: per pool the tables may report, the least and most copies of each card over the consistent worlds; memoises components across frames |
| `compact_network` (`compact.h`) | the exactly knowledge-preserving compaction of a network |
| `Knowledge` (`knowledge.h`) | the seat's sidecar: match constants (schema, decklists), match memory (copies seen per match), both players' models, the viewer's handle → unit map, the draw links (each player's first non-opening draw of the current turn), the last frame's solutions |

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

### 9.2 Bound propagation, the exact solver, compaction

**Bounds.** `BoundStore::propagate` narrows intervals to a fixpoint with a worklist: a node's size
(its classes sum to it), pins, roots, every junction (each member's interval from the other
members' sums and the junction's fixed cards) and the cut (the roots plus every junction's in-side
fixed cards equal the live nodes plus every out-side fixed card). Constraints only accumulate
between compactions, so the store keeps its bounds between frames and only processes nodes,
junctions and pins added since (a card entering the universe, or a compaction, resets it). Bounds
are sound always; they are per-card intervals per node, so they miss joint facts (after a
put-back from {Force of Will, Daze, X} and X cast, each of the top two is one of {Force of Will,
Daze}, but "one of each" is a joint fact) and they lose strength through merges of chains. They
seed the solver, decide which sub-pools can hold a card in the update rules, and are the fallback
when the solver's budget runs out.

**The exact solver** (`Solver::solve`, `Knowledge::solve` at the end of every frame, per player,
over the node sets of `pool_specs`: the tables' pools and every candidate block of runs).
1. *Reduce.* A copy of the network is compacted (below) keeping only the pools' nodes, which is
   exact for their projections (a this-turn draw a draw link may name, which the stored network
   keeps, can go here). The system is then one variable per node whose bounds leave more than one
   content, seeded with the store's bounds (for nodes compaction left unchanged) met with pins and
   roots; a determined node is a constant folded into its junctions. A `LOWER` root becomes its
   lower bounds (constants) plus a free variable, so no variable keeps a constraint of its own.
2. *Decompose.* Variables joined by a junction form a component; the worlds are the product of
   the components' worlds, so a pool's least (most) copies of a card is its determined members'
   copies plus the sum of the components' least (most).
3. *Leaves.* A state is a box of per-(variable, card) intervals that propagation keeps sound (the
   search's own propagator, with each pool's sum as a virtual node so its interval is tight). It
   is a leaf when every junction with a free variable is a transportation problem: its free
   variables are untouched by branching and in no other such junction, and they are all on one
   side, or the other side's free variables are each in no pool and no other junction (a slack:
   only their sum matters, as for a `LOWER` root's free part or a closed decklist's free root).
   With `T` the fixed cards the free side holds besides the slack, `N` its size and `s` the size
   of a pool's part in it, the part holds between `max(0, s + T_c − N)` and `min(s, T_c)` copies of
   card `c` (with a slack, `T_c` is `max(0, T_c)` and the upper end grows by the slack's spare
   units, `N − Σ max(0, T)`). These ends are exact: the junction is a transportation polytope with
   free cell bounds (every interval a free variable has is implied by its junction and size), and
   a leaf's junctions combine freely. Most components are a leaf at the root: one junction from
   the registered list to the live nodes.
4. *Search.* Otherwise the search branches by halving one interval of a blocking variable (one
   already branched on, else one in two junctions with free variables, else one on the smaller
   side of a junction with free variables on both sides), so the children partition the worlds.
   A first search finds any leaf (none: the component has no world, an `ERROR` line). Then every
   answer (pool, card, least or most) not yet reached by a leaf gets a proven bound from the
   root's interval and the card's **flow relaxation**: per card, every junction conserves copies
   and every variable leaves at most one junction and enters at most one (the outside world is one
   more node), so the card's cells under the current intervals are a network flow; its min-cost
   optimum (augmenting paths for a feasible flow, then Bellman-Ford negative-cycle cancelling)
   bounds every world's, since only the node sizes, which couple the cards, are dropped. The
   answer is then searched for with the root's interval for that pool and card narrowed to its
   proven bound, cutting a subtree whose flow bound falls short, branching first on the searched
   card's interval and trying its half toward the answer first. The first leaf found that reaches
   it settles it, and every leaf updates every answer it improves; a narrowed search that finds
   no such leaf proves no world reaches the bound, which moves one step, and the search repeats.
   When every answer is settled they are the exact projections (each end is reached by a world of
   some leaf, and nothing beyond the proven bounds exists). Steps are constraint evaluations,
   search nodes, flow augmentations and Bellman-Ford passes; past the budget (1,000,000 per player
   and frame) the settled answers are still exact and the rest report their proven bounds, and
   `k_meta.exact` is 0.
5. *Memo.* A component's result is memoised by its full description (sizes, seed intervals,
   junctions, constants, pools) together with the budget it was given; an unchanged component
   costs only that lookup, and a memoised result reports the steps its search took, so outputs do
   not depend on the memo (it is not serialised).

**Compaction** (`compact_network`, run by `PlayerModel::compact` on every network that changed
since its last compaction, before the frame's propagation; the stores restart). Live nodes and
this turn's draw-link nodes are kept, with all their fields; the rest of the history is rewritten
by rules that are each exact for the projection onto every node they leave, applied to a fixpoint:
- a pinned node that is not kept (and no `EXACT`/`LOWER` root) becomes its card: a fixed card on
  the out side of the junction that made it and the in side of the one that consumed it;
- an unpinned, non-root, not-kept node between junctions P and C is removed, P and C becoming one
  junction, when it is the only member (fixed cards included) of P's out side or of C's in side:
  it is then the sum of the other side, so its non-negativity is implied, and contracting cannot
  create a cycle; on both sides of one junction it cancels;
- such nodes with the same P and C, and `FREE` roots with the same consumer, are one node (only
  their sum appears anywhere, and any sum splits into parts of the given sizes);
- a `FREE` root alone on its consumer's in side makes that junction vacuous: it drops, and the
  nodes it made become `FREE` roots;
- cards fixed on both sides of one junction cancel; a junction left empty drops.

No rule touches a kept node, an `EXACT` or `LOWER` root, or the pins of the observation store
beyond turning them into fixed cards, so both stores' systems are preserved. Before the network
rules, `PlayerModel::compact` also merges live nodes nothing tells apart (exact the same way:
adjacent library runs, or sub-pools of a set zone with the same drawn-this-turn turn, made by the
same junction, unpinned, without a handle and not protected), which generalises T9's rejoin. A
shuffle therefore folds the whole history behind the library into the junction that made the
merged run: the opponent's model is typically one junction from the registered list to the live
nodes, plus small junctions for hands known in part.

### 9.3 Sizes, steps, serialisation

Measured with `rmknow --stats` over both seats of the replay corpus (64 dumps), 14 explore fuzz
games (bo1 and bo3 with sideboarding), 4 random `--closed-decklist` matches (3 bo3) and 2 long bo3
fuzz matches; 168 runs, 31,532 frames:

| | T9 (no compaction) | T10 |
|---|---|---|
| nodes per player network, most / mean of the runs' maxima | 845 / 310 | 87 / 61 |
| junctions, most | 425 | 13 |
| serialised state, largest / mean | 223 KB / 85 KB | 77 KB / 46 KB |

The state is now mostly the SCHEMA payload (35 KB). Solver and propagation steps per frame
(`k_meta.steps`): 11.5% of frames at most 100, 79% at most 1,000, 4.3% at most 10^4, 5.0% at most
10^5 and 0.2% above (the most 162,622, against the budget of 1,000,000); `k_meta.exact` is 1 on
every frame. Of the 32,445 component solves, 79% are memo hits, 19% a leaf at the root and 2% need
the search. The sweep (debug build, 4 cores) takes as long as T9's bounds alone (31 s); a step
costs about 12 ns in a release build. `rmknow --truth` finds no error in any of the
31,532 frames.

`Knowledge::serialize` writes a self-contained little-endian blob (`RMKN`, version 2): level,
viewer, the SCHEMA payload (re-decoded on load), the decklists, match memory, both
`PlayerModel`s (network with its fixed cards, both stores' bounds, zone lists, the node count at
the last compaction), the handle and draw-link maps, the latest turn and the last frame's
solutions; card names and the solver's memo are not included. `deserialize` refuses a malformed
blob.

### 9.4 Copies seen

`k_seen.game` is the fewest copies of the card the player can own given only the observations:
the larger of the `seen` store's sums over the live nodes and over the roots (roots free, so
deck-list arithmetic never counts as seen). A card revealed, shuffled away and then cast counts
once; two copies on the battlefield count twice. `match` is the maximum over the match's games;
under `--closed-decklist` it becomes the next game's root lower bounds.

### 9.5 Verification

- **Consistency** (`Knowledge::check`, every STATE frame of a live game): hand, library and
  sideboard sizes; every public and face-down object is a model unit with the same card; the
  viewer's own hand is exactly the model's pinned units; the own sideboard's counts are within
  bounds; every reveal row's card is the model's card at that position or handle; and neither the
  propagation nor the solver finds a contradiction. A mismatch is an `ERROR: knowledge[...]` line.
- **`rmknow --truth`** checks every reported fact against the same dump's omniscient frames
  (pool contents within min/max and no unlisted universe card, block pools included, exact
  positions, known-hand counts, pool sizes, seen ≤ owned). On the 168 runs above: 0 errors.
- **`rmknow --selftest`**: synthetic event streams (`tools/rmknow/selftest.cpp`): the section 7
  put-back example (each of the top two is Force of Will or Daze, the block of the top two holds
  exactly one of each, exact), cut accounting, a scry range, a batch, drawn-this-turn, a draw link,
  the viewer's own rearrangement, the search rejoin, a London mulligan, closed-decklist match
  memory, serialisation round trip, and forty draw / reveal / put-back / shuffle rounds staying
  under 20 nodes.
- **`rmknow --exactness N`** (`tools/rmknow/exactness.cpp`): the solver and compaction against
  an enumerator of the worlds (`tools/rmknow/enumerate.h`) written independently of the solver. It
  takes a plain description of the constraints (node sizes and pins, roots, junctions with fixed
  cards) and sweeps the worlds forward through the junctions in dependency order, a state holding
  the value of every frontier node something still reads; each junction divides its in nodes'
  total among its out nodes in every way their sizes and pins allow, and interchangeable frontier
  nodes (same size, consumer and pools) are kept sorted so equal states merge. It shares no code
  with the solver and does no interval reasoning. Cases (≤ 15-card decks, deterministic seeds):
  - N synthetic histories of library and hand operations on one player's model (draws, reveals,
    casts and put-backs from a hand whose sub-pools the viewer cannot tell apart, shuffles, scries,
    mills) under `EXACT`, `LOWER` and `FREE` roots, dealt from a hidden true deck: the raw
    network's enumeration against the solver on the raw network, against the compacted network's
    enumeration (same units, grouped by library position and zone) and against the solver on the
    compacted network;
  - N random two-player event streams through `Knowledge` (the opponent's draws, hand reveals,
    put-backs, casts, top reveals, scries, mills, shuffles; the viewer's draws, looks and
    put-backs), plus every fourth seed with compaction off: after each frame, both players' pools
    against the enumeration of that instance's network and against the simulated truth;
  - sculpted: the section 7 example from a hand also holding two unknown cards, a game 2 under a
    closed decklist (a `LOWER` root), a game 2 with open decklists (the registered 15 split
    between deck and sideboard).

  A check passes when the solver's bounds contain the enumeration's and, when it reports exact,
  equal them on every card. `--exactness 50`: 1,756 solver checks (all exact and equal) and 268
  compaction checks in 8 s (debug build; 1 s release), with 164 of its 1,194 component solves
  going beyond the closed form into the search; `--exactness 400`: 13,606 and 2,102. Deliberately broken, each of the leaf
  formula, the series-elimination condition, the propagation, the flow bound and its cycle
  cancelling is caught.

### 9.6 Engine integration (T11)

**Placement and lifetime.** `src/state_format/knowledge_feed.{h,cpp}` (engine side; `src/knowledge/`
still sees no engine header) keeps one `Knowledge` per RMSTATE frame writer (`ObsConsumer`:
`--rmstate-dump`, `--machine-format rmstate`, `--selfcheck-state`) and seat, at most six. An
instance is made at its writer's first frame for that seat and lives outside `Game` for the whole
match, so match memory crosses the per-game reset (the instance resets its networks itself at
`game_start`). The writer feeds it the seat's SCHEMA, CARDS and DECKLISTS frames as it writes them,
then every STATE frame of the seat exactly as encoded, its events drained from that writer's own
queue (terminal frames included). `Knowledge::table_blocks` gives the `k_*` tables as table blocks
and `rmstate::append_frame_blocks` adds them to the frame before the writer emits it. Their ids
(90-95) are the highest and they have no text column, so the result is the frame one `FrameWriter`
would have written with those rows (the self-check's roundtrip confirms it at every frame). The
omniscient frames carry no `k_*` tables. With no writer on, no instance exists and nothing runs.

**Levels.** `--knowledge-a` / `--knowledge-b off|expert` (default expert) choose the level each
seat's instances are made with. At level off `Knowledge::feed` decodes only the match frames and the
STATE frames get the `k_meta` row alone (level off, exact 0, steps 0). The level changes only the
`k_*` tables: play, the RNG, BQUERY and every other table are the same. It is an output option like
the writers themselves: the RMLOG header does not record it and `--replay` honours it (a replay
with the same level reproduces the frames).

**Cadence and determinism: one instance per writer.** An instance's output at a frame depends on
the frames it was fed before, not only on the events and the current snapshot: `k_meta.steps`
counts propagation, which is incremental per frame, and compaction and the merging of runs nothing
tells apart run per frame, so the pools a frame reports (which library runs stay split) can differ
too. Fed the seat's frames at every decision (the dump) or only at the seat's own decisions (the
stream), two instances report the same facts in different shapes: over three stream sessions, 11-38%
of a seat's stream frames had another `k_meta` than the dump's frame for the same decision and up to
25% other pool rows (a mulligan's bottomed card kept as a run of its own beside the rest of the
library), each frame exact. Rather than make every output cadence-free, every writer feeds an
instance of its own with exactly the frames it emits, so:
- a writer's `k_*` tables are a pure function of the seat's frames from that writer:
  `rmknow --isolation` on those frames (a dump's seat frames, or a saved stream) recomputes them
  byte for byte;
- one writer's bytes never depend on another being on: the stream is byte-identical with and
  without `--rmstate-dump` and `--selfcheck-state`, as T5 made it for the rest of the frame;
- the dump's and the self-check's instances see the same frames (every decision, each from its own
  queue), so their tables agree; the self-check has its own so that it runs without a dump.

**Snapshots.** The first instance registers sidecar hooks with the engine snapshot
(`snapshot_set_sidecar_hooks`, `src/snapshot.h`): every `snapshot_save` stores every instance's
`Knowledge::serialize` blob (a presence byte, then length and state, per writer and seat) and every
restore puts them back with `deserialize` (an instance the blob does not hold is dropped). A restore
into the live instance finds the same SCHEMA payload and does not decode it again; the solver's memo
is not part of the state and outputs do not depend on it. Without a writer no hook is registered, so
snapshots cost nothing more. With one, a snapshot carries about 45-65 KB per instance (two to six),
mostly the SCHEMA payload, and serialising or restoring one instance takes about 20-35 us in a
release build (150-300 us debug). The stream is refused under `--search-server` and the self-check
is off there, so search pays this only with `--rmstate-dump` on.

**Self-checks.** The self-check's instance takes each real frame exactly once (a decision's, and the
terminal one); the scrambled and permuted builds are compared with the frame as encoded, before its
`k_*` tables, which are a function of the instance and those bytes alone (state_format §10). The
snapshot check saves after the match frames are fed and before the decision's frames are, re-feeds
the saved decision's re-encoded frames to the restored instances and compares the frames with their
`k_*` tables. The instance's consistency errors are `selfcheck-state knowledge` failures. The dump's
and stream's instances print theirs as `ERROR: knowledge[…] … (dump frames)` / `(stream frames)`.

**Isolation.** `rmknow --isolation [--viewer A|B] [--level off|expert]` splits each of the seat's
STATE frames at table 90 (`rmstate::split_frame_blocks`), feeds the rest, computes
`Knowledge::table_blocks` and compares them with the split-off tables byte for byte (an `ERROR:
rmknow isolation` line names the first differing table and row). `train/test_rmstate.py --isolation
FILE…` runs it for both seats (`check_knowledge_isolation`), and `test_rmstate.py` alone runs it on
its fuzz dumps and its stream sessions' dumps and saved streams (`stream_<seed>.stream.rms`); the
gate does not run it yet (T12). On five stream sessions (bo1 and bo3, one under
`--closed-decklist`, one with seat A at level off), each with its dump, a dump with the self-check
sampled at every decision, and the off variants: 0 differences over 14,514 seat frames.

**Cost.** Feeding a seat frame and building its tables takes 0.16-1.0 ms on average per game in a
release build (most in reanimator vs uw_control bo3: 805 decisions with a dump took 3.3 s at
expert, 1.7 s at off and 0.8 s with no dump, so about 1 ms per seat frame) and 1.5-9 ms in a debug
one (22.7 s, 8.5 s, 3.3 s). A dump pays it twice per decision, the stream once per decision of the
seat, the self-check twice more. The `rmstate` gate tier (corpus and fuzz under the self-check,
two fuzz dumps, two stream sessions) takes 2 min 40 s on 4 cores, against 2 min 30 s before the
sidecar was on.
