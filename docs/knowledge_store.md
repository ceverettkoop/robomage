# Knowledge store: the actor-side memory and inference sidecar

[`state_format.md`](state_format.md) sends each viewer only what it explicitly knows right now. The
knowledge sidecar adds everything a player with **perfect recall and perfect logical inference**
would know on top of that, in one form for every kind of agent. Code: `src/knowledge/`,
`tools/rmknow/`, `src/state_format/knowledge_feed.*`, `src/state_format/selfcheck_knowledge.*`.

## 1. What it is

- **A C++ component separate from the engine's rules state.** Its only input is the byte stream one
  viewer receives (`SCHEMA`, `CARDS`, `DECKLISTS`, then each frame's snapshot and events); its output
  is the `k_*` knowledge tables appended to that viewer's STATE frames.
- **Level per seat:** `off` (the `k_meta` row alone, level off) or `expert` (full recall plus every
  sound deduction, section 4); `--knowledge-a/-b off|expert`, default `expert`. The level changes only
  the seat's `k_*` tables, never play.
- **Cannot leak, by construction:** a pure function of the viewer's frame stream. `src/knowledge/`
  includes only `src/state_format/decode.h`, `schema.h` and the standard library (checked by the
  `state` gate tier's isolation part), and `rmknow` recomputes the same tables from a recorded
  stream byte for byte.
- **Independent of cadence:** a function of the viewer's events alone. Each event is applied and
  settled (compaction, propagation) on its own; a frame only triggers the snapshot check and the
  solve. A consumer fed every decision and one fed only at its own decisions report the same `k_*`
  tables at every frame both receive (section 9.6).
- **Deterministic and snapshot-safe:** its outputs are frame bytes, so RMLOG replay reproduces them;
  its state is serialised with the engine snapshot, so a restore resumes with the same knowledge.

## 2. Placement in the code

| Piece | Location |
|---|---|
| core (model, update rules, solver) | `src/knowledge/` (C++17, `-fno-exceptions`; `Knowledge` in `knowledge.h` is the entry point), linked into `bin/<config>/robomage` and `rmknow` |
| engine integration | `src/state_format/knowledge_feed.{h,cpp}`: one instance per RMSTATE frame writer (`--rmstate-dump`, the `--machine` stream, `--selfcheck-state`, the in-process agents' feed) and seat, fed exactly the bytes that writer builds for that seat; it appends the `k_*` tables to each of the seat's STATE frames before the writer emits it (section 9.6) |
| consumers | read the decoded `k_*` tables through the same decoder as every other table (section 8) |
| `rmknow` | `tools/rmknow/` → `bin/<config>/rmknow` (part of `make`; `make rmknow` alone): `rmknow [--viewer A\|B] [--level off\|expert] [--in FILE] [--out FILE] [--text \| --quiet] [--truth] [--isolation] [--stats] < frames > knowledge`, `rmknow --selftest`, `rmknow --exactness N` (sections 7, 9.5) |

## 3. Model

### 3.1 Cards as anonymous units in pools

- Every physical card of a player is a unit; one whose identity the viewer cannot currently see is
  **anonymous**.
- The model tracks **pools**, never which anonymous unit is which. A pool is a set of units the
  viewer cannot tell apart: the opponent's hand, a stretch of library positions in unknown order,
  the opponent's sideboard, a face-down card (exiled, a permanent or a spell).
- A **unit with known identity** sits in a pool as a fixed value (a revealed card still in the
  opponent's hand, a known card at a known library position).

### 3.2 The constraint record

- Events create, merge and split pools (a draw moves a unit from a library position into the hand
  pool; a put-back splits units off the hand pool).
- The history of pool contents is a **flow network over time**: every pool at every moment is a
  node; every movement of anonymous units is an edge carrying a known number of units of unknown
  identity; every identity seen (a reveal, a move to a public zone, a look) pins units of a known
  card at that node.
- **The root of each player's network** is their registered list: in game 1 the deck node is
  exactly the main deck (open decklists); in games 2–3 the registered 75 split into a deck node of
  deck size and a sideboard node, the split unknown to the opponent.
- **A world** assigns a card identity to every unit consistently with all pins and flow edges.
  **Knowledge** is what is true in every world.

### 3.3 Viewer-specific simplifications

- The viewer's own hand and every public zone are visible, so on the viewer's side only its library
  (order and contents) is anonymous.
- On the opponent's side: hand, library, sideboard (games 2–3) and face-down cards.
- Under `--closed-decklist` the opponent's root is unconstrained (lower bounds only: the copies seen
  earlier in the match), so only recall-type deductions remain.

## 4. What `expert` reports

### 4.1 Recall (always exact)

- every card seen, per game and per match;
- known cards in the opponent's hand (units with known identity still in the hand pool);
- known library positions, per player;
- the viewer's own library contents: own registered list minus every own card seen elsewhere this
  game.

### 4.2 Deduction (sound always; exact when the solver finishes)

- **Accounting:** per card and hidden pool, the least and most copies it can hold ("1–2 Force of
  Will in hand or library"; "0 in library" after both were seen).
- **Candidates:** per hidden position or pool, the cards it can still be ("their top card is one of
  {Force of Will, Brainstorm}" after a put-back from a hand known to hold those).
- **Implied facts:** a candidate set of size one is a known identity. Two cards put back from a
  hand known to be exactly {Force of Will, Daze, X}, then X cast: the top two are {Force of Will,
  Daze}.
- **Sideboarding (games 2–3):** each sideboard card seen this game proves a swap; the per-card
  min/max covers deck versus sideboard.

**Excluded, left to agent policy:** probabilities, opponent modelling and strategic inference
("they didn't counter, so no Force of Will").

### 4.3 Solving

- **Hardness.** Exact deduction is an integer multi-commodity flow problem (card identities the
  commodities, pool sizes the shared capacities): NP-hard in general, tiny in practice (at most 75
  units per player; few pools carry constraints, the library bulk being one unconstrained
  remainder).
- **Algorithm** (`src/knowledge/solver.h`, section 9.2):
  1. after every event, propagate per-card min/max bounds per node through every junction and the
     cut;
  2. per frame and player, for every pool the tables may report: compact a copy of the network down
     to those pools' nodes, split it into independent components, and settle each pool's least and
     most copies of each card by a search over interval states whose leaves are transportation
     problems with closed-form projections. A per-card network-flow relaxation bounds every answer
     and prunes the search; components are memoised across frames.
- **Budget.** 1,000,000 steps (constraint evaluations, search nodes, flow augmentations and passes)
  per player and frame, at most about 12 ms in a release build.
- **Exactness flag.** Past the budget, unsettled answers report the propagated bounds narrowed by
  what the search proved: still **sound**, possibly wider than exact. `k_meta.exact` is 1 when every
  `k_pool_card`, `k_position` and `k_known_hand` fact of the frame is the exact projection of the
  consistent worlds. A frame exceeding the budget is a finding to investigate. The hard cases are a
  hand mixed over many turns (draws merged into a partly known hand, known cards cast from it): a
  chain of junctions joined by hand sub-pools, bounded by per-card conservation along the chain,
  which per-card intervals miss and the flow relaxation sees.
- **Compaction.** After every event that changed the network, history that no longer constrains
  anything is eliminated exactly (the consistent worlds projected onto the live nodes stay the
  same), so the network stays bounded over a long game: a shuffle collapses the history behind the
  merged run into one junction (section 9.2).

## 5. Update rules per event kind

Every rule is computable from the event's fields alone. Library positions are (anchor, pos, span)
as in state_format §5; the model tracks each library's size, so it moves position facts as cards
enter or leave above them and converts between top and bottom.

| Event (state_format §5) | Update |
|---|---|
| `game_start` | new networks: deck node from the root (section 3.2); library pool = the stated library count of units from it; the rest of the main deck is a set-aside pool (test-harness presets, which arrive as `zone_move` from `none`); sideboard pool = the stated sideboard count (games 2–3: the registered 75 split, unknown to the opponent). Hands start empty. Game 1 roots the main deck and the sideboard separately, each with its own set-aside rest (a companion comes from the sideboard's); the viewer's own root is the registered list in game 1 (it holds the harness presets, which `own_deck` does not) and `own_deck` in games 2–3. Under `--closed-decklist` the opponent's roots are lower-bound roots of the stated sizes (game 1 one for the library and one for the sideboard, nothing below them; games 2–3 one for both, the copies seen earlier in the match below it), which compaction keeps; a card created from `none` is a free unit of its own |
| `draw` (own) | known cards leave the top of own library one at a time; positions shift; known-position facts move up |
| `draw` (opponent) | an anonymous unit moves from the top of their library into the hand pool (one `draw` event per card). A known top card becomes a known card in hand; a card in the viewer's live window arrives with its identity. Unless the opening-hand flag is set, the unit also joins the hand's **drawn-this-turn** sub-pool for the event's `turn`; it leaves it when it leaves the hand, and the sub-pool empties when `turn` advances |
| `zone_move` to a public zone | pin the identity on the source pool (at the stated library position or range; a card cast from hand proves the hand held it), then remove the unit. A card the viewer holds a handle for is exactly that unit. From a hand (or another set zone) without a handle: a unit already known to be that card if there is one (copies are interchangeable), else the sub-pools whose bounds allow the card, merged when there are several; handled units are never candidates. The unit stays in the network as a public unit keyed by its handle, so a later move back to a hidden zone continues it |
| `zone_move` public → hidden with handle | a unit with known identity enters the destination at the stated position (a bounce to hand: known card in hand) |
| `zone_move` from a hand with subset `drawn_this_turn` | as the matching rule, but the unit comes from the hand's drawn-this-turn sub-pool, so the rest of the hand is untouched (Sylvan Library) |
| `zone_move` hidden → hidden, no identity | one anonymous unit leaves the source pool and enters the destination at the stated position (a Brainstorm put-back). From a library range of span > 1 those positions first merge into one unordered pool (the opponent's scry) |
| `zone_move` in a batch | as above, but the destination position waits for the batch's `library_place`. Every card the instruction put into that library is in the batch for the viewer, identified or not |
| `zone_move` of a token or copy | no unit; nothing changes |
| `zone_move` from `none` of a card | a set-aside unit (harness preset) is pinned and placed |
| `library_place` | the batch's units occupy the stated range as one unordered pool (known identities pinned as its contents). They are placed at the named end as they arrive (the `library_place` is in the same frame), so later cards' source positions count them as the engine does |
| `reveal_start` | pin the identities at the stated zone and positions. A card with a draw link pins the unit that draw brought in, and so the library position it was drawn from (Miracle). A shown hand or sideboard card lands on a distinct unit: the one its handle names, else an unhandled unit known to be that card, else one split off the sub-pools that can hold it |
| `reveal_end` | nothing: visibility ends, knowledge stays |
| `look` (by the viewer) | pin identities at positions. A search's find is shown at its position in the already-shuffled library; when it leaves, the two parts of the run around it rejoin. A standing look (Mystic Forge) is a `look` at each new top card, so a card seen on top and then shuffled in is in the library's contents at no position |
| `look` (by the opponent) | no identity; marks that the opponent knows those positions (no constraint). A position is unknown (span 0) when the opponent looked by identity (a search's find after the shuffle). The opponent's standing look comes once, as it starts |
| `reorder` | the range becomes one unordered pool, unless the viewer saw the result (each position pinned) |
| `shuffle` | all positions of that library merge into one unordered pool; position facts drop, contents facts stay |
| `search` | nothing by itself (a look at the whole zone comes as `look`); the cards leave as `zone_move`s, usually followed by a `shuffle` |
| `cast_from_hidden` / `play_from_hidden` | as `zone_move` to a public zone, or with the face-down flag to the pool below; a reversed face-down cast is a hidden `zone_move` back, which teaches nothing. A cloak's `play_from_hidden` takes the unit at the named library position into the pool with what is known of it, so a seat that had seen the top keeps it pinned in `k_pool_card` while the `object` row shows no card |
| `face_down` / `face_up` | a face-down card (exile, battlefield or stack) is an anonymous singleton pool keyed by its handle; face-up pins it. Any entry face down is a `zone_move` with the face-down flag, with the same pool. A face-down permanent or spell leaving is a `zone_move` naming the card (CR 708.9), which pins the unit; so does the reveal at the end of a game |
| mulligan (hand to library, shuffle, draw 7, London bottoming) | composed of the rules above |
| `game_end` / `match_end` | match memory updated; networks reset at the next `game_start` |

The snapshot is only a check, never an input: the model's pool sizes must equal its zone counts and
every identity it shows must agree with the model; a mismatch is a missing event or a bug, reported
loudly. The one snapshot table read is `own_deck`, for the viewer's own root at a `game_start` in
games 2–3 (the live 75 changes only while sideboarding, so every frame of the game shows the list it
started with).

## 6. Output tables

Table ids 90–95, no text column; columns in state_format Appendix A.6. Built by `Knowledge::rows`
(`src/knowledge/output.cpp`).

| Table | Rows | Columns |
|---|---|---|
| `k_meta` | 1 | `level`, `exact`, `steps` |
| `k_seen` | per (player, card) with a nonzero count | `player`, `card`, `game`, `match` |
| `k_pool` | per hidden pool | `owner`, `kind` (hand / library / sideboard / face-down / set-aside), `size`, `range` (library), `obj` (face-down handle) |
| `k_pool_card` | per (pool, card) with max > 0 | `pool`, `card`, `min`, `max` |
| `k_position` | per library position with a known card or a one-position run | `owner`, `at`, `card` or `pool` |
| `k_known_hand` | per card definitely in the opponent's hand | `card`, `copies` |

- **`k_meta`** is always one row, so a consumer always finds the level; at level `off` every other
  table is empty. `exact` is 1 when the solver settled every reported pool (also between games, with
  no pool), 0 at level off. `steps` counts the solver's steps for the frame (both players; 0
  between games); propagation is not counted. Like every column, a function of the events.
- **Order:** self first, then the opponent, each hand, library runs top to bottom, sideboard,
  face-down cards (by handle), set-aside.
- **`k_pool`**: one row for the opponent's whole hand and one for their whole sideboard (the
  viewer's own are snapshot objects); one per library run (positions in unknown order, `range` from
  the top) whose card is not determined; one per **block** of runs from the top (runs 0..j) or from
  the bottom (runs i..last), short of the whole library, that is sharper than its parts (some card's
  least or most copies tighter than the block one run shorter plus the run it adds); one per
  face-down card; one for the set-aside units while any are left. Blocks carry joint facts: after the
  section 4.2 put-back, each of the top two runs holds Force of Will 0–1 and Daze 0–1, and the block
  of the top two holds exactly one of each.
- **`k_pool_card`**: per pool, every card of the player's universe (the registered list, or the cards
  seen under `--closed-decklist`) with max > 0: the least and most copies in the pool as a whole.
  Under `--closed-decklist` a never-seen card is also possible without a row.
- **`k_position`**: every position whose run's card is determined (`card` set; a run of several known
  copies gives a row per position), and every one-position run that is not (`pool` = its `k_pool`
  row). `at` is anchored at the top. A position's candidates are its run's cards with max > 0 (exact
  for every position of a run when the solver is, since any order of a run is possible).
- **`k_known_hand`**: per card, the opponent's hand pool's least copies.
- **`k_seen`**: `game`, the fewest copies the game's observations imply (section 9.4); `match`, the
  most of that over the match's games.

## 7. Verification

- **Soundness (zero tolerance):** the true world satisfies every reported fact (identities,
  candidate sets, min/max). `--selfcheck-knowledge` checks against the engine state, `rmknow --truth`
  against a dump's omniscient frames; both use `knowledge::check_truth` (section 9.5).
- **Exactness:** an enumerator of every consistent world, independent of the solver, on synthetic
  and sculpted cases (≤ 15-card decks); with `k_meta.exact` set the reported min/max and candidates
  must equal the enumeration. `rmknow --exactness N` (section 9.5).
- **Sharpness:** scenarios asserting the deduction a strong player makes, e.g. after a Brainstorm
  put-back from a hand known to hold {Force of Will, Daze, X} and X then cast, the top two are
  exactly {Force of Will, Daze}. `tests/scenarios/knowledge_*.json` with the `expect_dump*` keys
  (section 9.7).
- **Isolation:** `rmknow --isolation` on a writer's recorded seat frames reproduces their `k_*`
  tables byte for byte, and the stream's `k_*` tables equal the dump's at every seat and decision
  (section 9.6).
- **Snapshot/restore:** the knowledge state is saved and restored with the engine snapshot, and
  frames after a restore match the originals.

The `knowledge` gate tier runs `--selftest`, `--exactness`, explore games under
`--selfcheck-knowledge`, stream sessions against their dumps, `--isolation` and `--truth`; the
`state` tier replays the corpus under `--selfcheck-knowledge` (`docs/ci.md`).

## 8. Consumers

- The C++ CLI (`src/cli/board.cpp`) renders the tables as a knowledge panel (the `know` command).
- The scripted agent (`python/rm_agent.py`) reads `k_position`, `k_pool`, `k_pool_card` and
  `k_known_hand`: own known top cards (`k_position` with `card` set, e.g. the Thassa's Oracle
  check) and the cards definitely in the opponent's hand (`k_known_hand`, e.g. Cabal Therapy's
  name pick).
- The featurizer featurizes the `k_*` tables (`docs/featurizer.md`); external `--machine` agents
  receive them in the same STATE frame.

## 9. Implementation

### 9.1 Data model

| Struct (file) | What it is |
|---|---|
| `Node` (`network.h`) | a set of one player's units the viewer cannot tell apart, at one moment: size, pin (the known card of a one-unit node), the junction that made it and the one that consumed it, root index, and while live its location (`Loc`), handle, drawn-this-turn turn and draw seq |
| `Junction` | `in` units plus `in_fixed` cards = `out` units plus `out_fixed` cards (a split: one in; a merge: one out; after compaction any number on either side, the fixed cards being pinned units compaction replaced by their card) |
| `Root` | a node no junction made: `EXACT` contents (a registered list), `LOWER` bounds (a closed decklist's roots) or `FREE` (a closed decklist's card created from zone none); and the part of the registered list it stands for (`RootPart`: main deck, sideboard, the 75, none) |
| `Network` | one player's nodes, junctions, roots and the log of pins added to existing nodes; nodes never change, so the network is the record of section 3.2 |
| `PlayerModel` (`player_model.h`) | the network plus where its live nodes are: the library as a sequence of runs (top first; a run is one node filling that many consecutive positions in unknown order; a one-unit run is an exact position), the hand, sideboard and set-aside as sets of sub-pools, face-down and public units (one each, keyed by handle) |
| `BoundStore` (`bounds.h`) | per node and per card class of the player's `Universe` (the cards tracked, plus one OTHER class), the least and most copies; two per player: `full` (roots constrain: the deductions) and `seen` (roots free: what was observed) |
| `Solver` (`solver.h`) | the exact solver: per reportable pool, the least and most copies of each card over the consistent worlds; memoises components across frames |
| `compact_network` (`compact.h`) | the exactly knowledge-preserving compaction |
| `Knowledge` (`knowledge.h`) | the seat's sidecar: match constants (schema, decklists), match memory (copies seen per match), both players' models, the viewer's handle → unit map, the draw links (each player's first non-opening draw of the current turn), the last frame's solutions |

Every unit is in exactly one live node, so the live nodes always sum to the roots: the **current
cut**, which propagation treats as one more junction ("one Force of Will cast from an unknown hand"
→ "at most three in the library").

When the viewer cannot tell which sub-pool of a set zone a card left, the candidates merge first:
exact for the multiset, but which hand units were drawn this turn is then unknown for the rest of
the turn, so a later drawn-this-turn pick (Sylvan Library) falls back to the whole hand.

Library operations go through the network: taking the unit at a position splits it off its run,
inserting cuts the run, a range the viewer cannot tell apart (a scry pick, a reorder) merges into
one run, a shuffle merges the whole library, a pinned position is a three-way split of its run. Two
runs that become adjacent again and are unpinned outputs of one junction rejoin, also when
compaction has folded the history behind that junction into it (`PlayerModel::rejoin`).

### 9.2 Bound propagation, the exact solver, compaction

**Bounds.** `BoundStore::propagate` narrows intervals to a fixpoint with a worklist over node sizes
(classes sum to the size), pins, roots, every junction (each member's interval from the other
members' sums and the fixed cards) and the cut (roots plus every junction's in-side fixed cards
equal the live nodes plus every out-side fixed card). It runs after every event
(`Knowledge::settle`), so each update rule sees the fixpoint of the events before it. Between
compactions it processes only nodes, junctions and pins added since (a card entering the universe,
or a compaction, resets it). Bounds are always sound but per-card per node, so they miss joint
facts ("one of each" of Force of Will and Daze in the top two) and weaken through chains of merges.
They seed the solver, decide which sub-pools can hold a card in the update rules, and are the
fallback past the budget.

**The exact solver** (`Solver::solve`; `Knowledge::solve` once per frame after its events, per
player, over `pool_specs`: the tables' pools and every candidate block of runs).
1. *Reduce.* A copy of the network is compacted keeping only the pools' nodes (exact for their
   projections). The system is one variable per node with more than one possible content, seeded
   with the store's bounds (for nodes compaction left unchanged) met with pins and roots; a
   determined node is a constant folded into its junctions. A `LOWER` root becomes its lower bounds
   (constants) plus a free variable.
2. *Decompose.* Variables joined by a junction form a component; worlds are the product of the
   components' worlds, so a pool's least (most) copies is its determined members' copies plus the
   sum of the components' least (most).
3. *Leaves.* A state is a box of per-(variable, card) intervals kept sound by the search's own
   propagator (each pool's sum a virtual node). It is a leaf when every junction with a free
   variable is a transportation problem: its free variables are untouched by branching and in no
   other such junction, and all on one side, or the other side's free variables are each in no pool
   and no other junction (a slack: only their sum matters, as for a `LOWER` root's free part or a
   `FREE` root). With `T` the fixed cards the free side holds besides the slack, `N` its size and
   `s` the size of a pool's part in it, the part holds between `max(0, s + T_c − N)` and
   `min(s, T_c)` copies of card `c` (with a slack, `T_c` is `max(0, T_c)` and the upper end grows by
   the slack's spare units, `N − Σ max(0, T)`). These ends are exact (a transportation polytope with
   free cell bounds; a leaf's junctions combine freely). Most components are a leaf at the root: one
   junction from the registered list to the live nodes.
4. *Search.* Otherwise branch by halving one interval of a blocking variable (one already branched
   on, else one in two junctions with free variables, else one on the smaller side of a junction
   with free variables on both sides), so children partition the worlds. A first search finds any
   leaf (none: the component has no world, an `ERROR` line). Every answer (pool, card, least or
   most) not yet reached by a leaf gets a proven bound from the root interval and the card's **flow
   relaxation**: per card, junctions conserve copies and each variable leaves and enters at most one
   junction, so the card's cells are a network flow whose min-cost optimum (augmenting paths, then
   Bellman-Ford negative-cycle cancelling) bounds every world's, only the node sizes that couple
   cards being dropped. The answer is then searched for with the root interval narrowed to the
   proven bound, cutting subtrees whose flow bound falls short, branching first on the searched
   card toward the answer. A leaf reaching it settles it (every leaf updates every answer it
   improves); a search finding none proves the bound one step tighter, and repeats. Settled answers
   are the exact projections. Steps are constraint evaluations, search nodes, flow augmentations and
   Bellman-Ford passes; past the budget the settled answers stay exact, the rest report their proven
   bounds, and `k_meta.exact` is 0.
5. *Memo.* A component's result is memoised by its full description (sizes, seed intervals,
   junctions, constants, pools) and the budget given; a memoised result reports the steps its search
   took, so outputs do not depend on the memo (not serialised).

**Compaction** (`compact_network`, run by `PlayerModel::compact` after every event that changed the
network, before that event's propagation; the stores restart. While an order-hidden batch waits for
its `library_place` in the same frame, compaction waits too). Live nodes and this turn's draw-link
nodes are kept with all their fields; the rest of the history is rewritten to a fixpoint by rules
each exact for the projection onto every node they leave:
- a pinned node that is not kept (and no `EXACT`/`LOWER` root) becomes its card: fixed on the out
  side of the junction that made it and the in side of the one that consumed it;
- an unpinned, non-root, not-kept node between junctions P and C is removed, P and C becoming one
  junction, when it is the only member (fixed cards included) of P's out side or of C's in side (it
  is then the sum of the other side, so its non-negativity is implied, and no cycle can form); on
  both sides of one junction it cancels;
- such nodes with the same P and C, and `FREE` roots with the same consumer, become one node (only
  their sum appears anywhere);
- a `FREE` root alone on its consumer's in side makes that junction vacuous: it drops, and the nodes
  it made become `FREE` roots;
- cards fixed on both sides of one junction cancel; an empty junction drops.

No rule touches a kept node, an `EXACT` or `LOWER` root, or the observation store's pins beyond
turning them into fixed cards, so both stores' systems are preserved. Before the network rules,
`PlayerModel::compact` merges live nodes nothing tells apart (adjacent library runs, or sub-pools of
a set zone with the same drawn-this-turn turn, made by the same junction, unpinned, without a handle
and not protected). A shuffle therefore folds the whole history behind the library into one
junction: the opponent's model is typically one junction from the registered list to the live
nodes, plus small junctions for hands known in part.

### 9.3 Serialisation

`Knowledge::serialize` writes a self-contained little-endian blob (`RMKN`, version 5): level,
viewer, the SCHEMA payload (re-decoded on load), the decklists with their names, match memory,
both `PlayerModel`s (network with its fixed cards and root parts, both stores' bounds, zone lists,
the node count at the last compaction), the handle and draw-link maps, the latest turn and the last
frame's solutions. Card names and the solver's memos are not included. `deserialize` refuses a
malformed blob. `rmknow --stats` reports network sizes, steps and serialised sizes over a stream.

### 9.4 Copies seen

`k_seen.game` is the fewest copies of the card the player can own given only the observations: the
larger of the `seen` store's sums over the live nodes and over the roots (roots free, so decklist
arithmetic never counts as seen). A card revealed, shuffled away and then cast counts once; two
copies on the battlefield count twice. `match` is the maximum over the match's games; under
`--closed-decklist` it becomes the next game's root lower bounds.

### 9.5 Verification

- **Consistency** (`Knowledge::check`, every STATE frame of a live game): hand, library and
  sideboard sizes; every public and face-down object is a model unit with the same card; the
  viewer's own hand is exactly the model's pinned units; the own sideboard's counts are within
  bounds; every reveal row's card is the model's card at that position or handle; neither
  propagation nor the solver finds a contradiction. A mismatch is an `ERROR: knowledge[...]` line.
- **Soundness** (`knowledge::check_truth`, `src/knowledge/truth.h`): every reported fact against a
  true world: each `k_pool` row's size, each `k_pool_card` range and no unlisted universe card in a
  pool, each `k_position` card (or, not exact, the true card among its run's candidates), each
  `k_known_hand` count, each `k_seen` row (game at most the copies owned, match at least game).
  Set-aside pools have no true counterpart. Two drivers:
  - **`--selfcheck-knowledge`** (`src/state_format/selfcheck_knowledge.h`; implies
    `--selfcheck-state`): at every decision, each seat's `k_*` tables read back from the
    self-check's frame bytes (`read_k_tables`) against the true world built from the engine state
    (identities by `card_identity`; hand, sideboard, library order; face-down objects by the seat's
    handle, so face-down pools are checked too), and at the snapshot check's restore every
    instance's serialised state against the one saved. Failures are `ERROR: selfcheck-knowledge
    soundness|snapshot: …` lines; `SELFCHECK-KNOWLEDGE: decisions=… facts=… errors=…` at match end;
  - **`rmknow --truth`** against the same dump's omniscient frames (face-down pools excluded: the
    omniscient handles are not the seat's).
- **`rmknow --selftest`** (`tools/rmknow/selftest.cpp`): synthetic event streams: the section 7
  put-back example (the two put-back units are one run of the top two holding exactly one Force of
  Will and one Daze, exact), cut accounting, a scry range, a batch, drawn-this-turn, a draw link,
  the viewer's own rearrangement, the search rejoin, a London mulligan, closed-decklist match memory,
  a serialisation round trip and forty draw / reveal / put-back / shuffle rounds staying under 20
  nodes.
- **`rmknow --exactness N`** (`tools/rmknow/exactness.cpp`): the solver and compaction against an
  enumerator of the worlds (`tools/rmknow/enumerate.h`) sharing no code with the solver and doing no
  interval reasoning. It takes a plain description of the constraints (node sizes and pins, roots,
  junctions with fixed cards) and sweeps the worlds forward through the junctions in dependency
  order, a state holding every frontier node something still reads; each junction divides its in
  nodes' total among its out nodes in every way sizes and pins allow, interchangeable frontier nodes
  kept sorted so equal states merge. Cases (≤ 15-card decks, deterministic seeds):
  - N synthetic histories of library and hand operations on one player's model (draws, reveals,
    casts and put-backs from indistinguishable hand sub-pools, shuffles, scries, mills) under
    `EXACT`, `LOWER` and `FREE` roots, dealt from a hidden true deck: the raw network's enumeration
    against the solver on it, against the compacted network's enumeration and against the solver on
    the compacted network;
  - N random two-player event streams through `Knowledge` (the opponent's draws, hand reveals,
    put-backs, casts, top reveals, scries, mills, shuffles; the viewer's draws, looks, shuffles and
    put-backs, and in even seeds a standing look), every fourth seed also with compaction off: after
    each frame, both players' pools against the enumeration of that network and against the
    simulated truth;
  - sculpted: the section 7 example from a hand also holding two unknown cards, a game 2 under a
    closed decklist (a `LOWER` root), a game 2 with open decklists (the registered 15 split between
    deck and sideboard), a look that outlives a shuffle.

  A check passes when the solver's bounds contain the enumeration's and, when it reports exact,
  equal them on every card.

### 9.6 Engine integration

**Placement and lifetime.** `src/state_format/knowledge_feed.{h,cpp}` (`src/knowledge/` sees no
engine header) keeps one `Knowledge` per RMSTATE frame writer (`ObsConsumer`) and seat, made at the
writer's first frame for that seat and living outside `Game` for the whole match, so match memory
crosses the per-game reset (the instance resets its networks at `game_start`). The writer feeds it
the seat's SCHEMA, CARDS and DECKLISTS frames, then every STATE frame of the seat as encoded, events
drained from that writer's own queue (terminal frames included). `Knowledge::table_blocks` gives the
`k_*` tables as table blocks, which `rmstate::append_frame_blocks` adds to the frame before it is
emitted; their ids are the highest and they have no text column, so the result is the frame one
`FrameWriter` would have written with those rows. Omniscient frames carry no `k_*` tables. With no
writer on, no instance exists.

**Levels.** At level `off`, `Knowledge::feed` decodes only the match frames and STATE frames get the
`k_meta` row alone (level off, exact 0, steps 0). The level is an output option: the RMLOG header
does not record it and `--replay` honours it (a replay at the same level reproduces the frames).

**Cadence independence.** Each writer's instance gets exactly the frames that writer emits, and its
output at a frame is a function of the seat's events up to that frame: each event is applied and
settled at once (compaction, propagation, copies seen), the snapshot only checked, and the frame's
solve and `k_meta.steps` read the settled model alone. Every writer sees the same event sequence for
a seat; only the drain times differ (the dump's at every decision, the stream's at the seat's own).
So:
- the stream's and the dump's frames for one seat and decision carry the same `k_*` tables byte for
  byte;
- a writer's `k_*` tables are a pure function of its seat frames (`rmknow --isolation` recomputes
  them);
- one writer's bytes never depend on another being on: the stream is byte-identical with and without
  `--rmstate-dump` and `--selfcheck-state`.

**Snapshots.** The first instance registers sidecar hooks with the engine snapshot
(`snapshot_set_sidecar_hooks`, `src/snapshot.h`): every `snapshot_save` stores every instance's
`Knowledge::serialize` blob (a presence byte, then length and state, per writer and seat) and every
restore puts them back with `deserialize` (an instance the blob does not hold is dropped). A restore
into the live instance finds the same SCHEMA payload and does not decode it again. Without a writer
no hook is registered.

**Self-checks.** The self-check's instance takes each real frame exactly once (a decision's, and the
terminal one); the scrambled and permuted builds are compared with the frame as encoded, before its
`k_*` tables (state_format §10). The snapshot check saves after the match frames are fed and before
the decision's frames, re-feeds the saved decision's re-encoded frames to the restored instances and
compares the frames with their `k_*` tables. The instance's consistency errors are
`selfcheck-state knowledge` failures; the dump's and stream's instances print theirs as
`ERROR: knowledge[…] … (dump frames)` / `(stream frames)`.

**Isolation.** `rmknow --isolation [--viewer A|B] [--level off|expert]` splits each of the seat's
STATE frames at table 90 (`rmstate::split_frame_blocks`), feeds the rest, computes
`Knowledge::table_blocks` and compares them with the split-off tables byte for byte (`ERROR: rmknow
isolation` names the first differing table and row). The `knowledge` gate tier runs it for both
seats on every dump and on each writer's frames of its stream sessions, after checking that every
stream frame equals the dump's frame for that seat and decision, `k_*` included.

### 9.7 Sharpness scenarios

A scenario with an `expect_dump`, `expect_dump_not`, `expect_dump_final` or `expect_dump_final_not`
key runs with `--rmstate-dump FILE`; the runner (`tools/robomage_check/knowledge_lines.h`, grammar
in its header) renders every seat STATE frame's `k_*` tables as one line per fact (`<seat>
d=<decision> g<game> t<turn> pool opp library top0+2 size=2: Daze 1-1, Force of Will 1-1`,
`position self top0 = Daze`, `known_hand Island x5`, …; `sb` for the turn between games), which the
regexes match (any frame, or each seat's last frame). `robomage_check --tier scenarios --scenario
NAME --show dump` prints them.

| Scenario | The deduction |
|---|---|
| `knowledge_section7_put_back_pool` | section 7's example: a hand known through Cabal Therapy, a Brainstorm put-back, the Bolt cast, then a Thoughtseize showing only Islands: A's top two are exactly one Force of Will and one Daze |
| `knowledge_own_brainstorm_put_back` | own Brainstorm put-back: own top two known exactly; the opponent knows neither position |
| `knowledge_shuffle_keeps_contents` | a shuffle drops every position fact and keeps the contents (both viewers: exactly one Force of Will and one Daze in the library) |
| `knowledge_sylvan_library_keeps_known_hand` | Sylvan Library's put-backs come from this turn's draws, so the known hand cards stay definitely in hand |
| `knowledge_miracle_draw_link` | the miracle reveal's draw link pins the drawn card: with one Triumph already known in hand, both are known |
| `knowledge_opponent_ponder_unordered` | the opponent's Ponder leaves the looked-at range unordered: a known top card becomes one of three positions, no `k_position` |
| `knowledge_bo3_sideboard_swap` | game 2: the Leyline of the Void the opponent starts with proves a swap (sideboard Leyline 4-4 in game 1, 0-4 in game 2, 0-3 once seen) |
| `knowledge_divergence_put_back_leak` | after the opponent's hidden Brainstorm put-back, a Force of Will known in hand is "in hand or in the top two" (0-1 each), not definitely in hand |
| `knowledge_divergence_copy_leak` | a second unseen copy drawn, then one copy cast: no copy is definitely in hand (copies are interchangeable) |
| `knowledge_divergence_bauble_look` | a private look (Mishra's Bauble) counts in `k_seen`, and the looked-at top card stays known in hand once drawn |
| `knowledge_divergence_companion_to_hand` | a companion put into hand is known to both players |
| `knowledge_divergence_between_games` | sideboarding frames report no pool, position or known-hand rows; `k_seen` keeps the match count |

With `--knowledge-a off --knowledge-b off` all of them fail; with the draw link ignored the miracle
scenario fails, and with the drawn-this-turn subset ignored the Sylvan Library one does.
