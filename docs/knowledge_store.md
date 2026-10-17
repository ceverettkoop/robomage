# Knowledge store: the actor-side memory and inference sidecar (design)

Status: step 1 of [`engine_fork_plan.md`](engine_fork_plan.md), **done**. T9-T12 are implemented
(`src/knowledge/`, `tools/rmknow/`, `src/state_format/knowledge_feed.*`,
`src/state_format/selfcheck_knowledge.*`, `src/state_format/shadow_knowledge.*`): the model, the
update rules, bound propagation, the per-frame consistency check, the `k_*` rows and serialisation
(T9), the exact solver, compaction and the enumerator tests (T10), the engine integration (T11),
and the verification of T12: the cadence-independent model (section 9.6), `--selfcheck-knowledge`
(soundness, the snapshot check and the shadow comparison with the old vector, sections 9.5 and
9.7), the sharpness scenarios (section 9.7) and the `knowledge` gate tier (`docs/ci.md`); section 9
says how. Archetype inference's sound layer (the reference metagame, `k_template`, `k_archetype`) is
built on it, and the belief layer (a posterior over the reference lists, `b_*` tables at level
`expert+belief`) on that: section 10.
It builds on
[`state_format.md`](state_format.md): the engine
sends only what each viewer explicitly knows right now. This component supplies everything a
player with **perfect recall and perfect logical inference** would know on top of that, in one
consistent form for every kind of agent.

## 1. What it is

- **A C++ component, separate from the engine's rules state.** Its only input is the byte
  stream one viewer receives: `SCHEMA`, `CARDS`, `DECKLISTS`, `METAGAME`, then each frame's
  snapshot and observation events. Its output is a set of **knowledge tables** attached to that viewer's
  frames.
- **Level per seat:**
  - `off` — the knowledge tables are present but empty: the `k_meta` row alone, with level off;
  - `expert` — full recall plus every sound logical deduction (section 4);
  - `expert+belief` — expert's `k_*` tables plus the belief layer's `b_*` tables: probabilities
    under a stated model, kept out of `k_*` (section 10.6).

  Flags are `--knowledge-a off|expert|expert+belief` / `--knowledge-b …`; the default is `expert`.
  The level changes only the seat's `k_*` and `b_*` tables, never play.
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
- **Independent of cadence.** The model is a function of the viewer's **events** alone: every
  event is applied and settled (compaction, propagation) on its own, and a frame only triggers the
  output (the snapshot check and the solve). A consumer fed every decision and one fed only at its
  own decisions report the same `k_*` tables at every frame both receive (the dump's and the
  stream's frames for one seat and decision are byte-identical, `k_*` included; section 9.6).
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
  recall-type deductions remain. The model is the same; the root is unconstrained (lower bounds
  only: the copies seen earlier in the match). Which reference lists of the metagame the opponent
  can still be playing is section 10.

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
"strategic" inference ("they didn't counter, so no Force of Will"). (The belief layer, section 10.6,
is the exception for the reference lists: a posterior under a stated model at a level of its own,
never in the `k_*` facts.)

### 4.3 Solving

- **Hardness.** Exact deduction over the flow network is an integer multi-commodity flow
  problem: card identities are the commodities, pool sizes the shared capacities. It is
  NP-hard in general, but tiny in practice: at most 75 units per player, and only a handful
  of pools ever carry constraints, since the library bulk stays one unconstrained remainder.
- **Algorithm** (as built, `src/knowledge/solver.h`; section 9.2).
  1. Propagate bounds (per-card min/max per node through every junction and the cut), after every
     event.
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
  after every event that changed it, exactly (the consistent worlds projected onto the live nodes stay the
  same), so its size stays bounded over a long game: after a shuffle the history behind the
  merged run collapses into one junction (section 9.2).

## 5. Update rules per event kind

Every rule must be computable from the event's fields alone. This table is the step-1 check
that the event schema is complete. Library positions are (anchor, pos, span) as in
state_format §5: the model tracks each library's size, so it moves position facts as cards
enter or leave above them and converts between top and bottom.

| Event (state_format §5) | Update |
|---|---|
| `game_start` | new networks: deck node from the root (section 3.2); library pool = the stated library count of units from it; the rest of the main deck is a set-aside pool (test-harness presets, which arrive as `zone_move` from `none`); sideboard pool = the stated sideboard count (games 2–3: the registered 75 split, unknown to the opponent). Hands start empty. Game 1 roots the main deck and the sideboard separately, each with its own set-aside rest (a companion comes from the sideboard's); the viewer's own root is the registered list in game 1 (it holds the harness presets, which `own_deck` does not) and `own_deck` in games 2–3. Under `--closed-decklist` the opponent's roots are lower-bound roots of the stated sizes (game 1 one for the library and one for the sideboard, nothing below them; games 2-3 one for both, the copies seen earlier in the match below it), which compaction keeps (section 10); a card created from `none` is a free unit of its own |
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

Each frame's snapshot is only a check, never an input: the model's pool sizes must equal the
snapshot's zone counts, and every identity the snapshot shows must agree with the model. A mismatch
is a missing event or a bug, reported loudly. The one snapshot table the model reads is `own_deck`,
for the viewer's own root at a `game_start` in the frame (games 2-3): the live 75 changes only while
sideboarding, before `game_start`, so every frame of the game shows the list it started with (the
check confirms it), whichever frame the `game_start` arrives in.

## 6. Output tables (schema fixed for both levels)

| Table | Rows | Columns |
|---|---|---|
| `k_meta` | 1 | level, exact flag, solver steps used |
| `k_seen` | per (player, card) | copies seen this game, copies seen this match |
| `k_pool` | per hidden pool | owner, kind (hand / library range / sideboard / face-down / set-aside), size, position range if a library range, handle if face-down |
| `k_pool_card` | per (pool, card) with max > 0 | min copies, max copies |
| `k_position` | per library position with a known identity or a narrowed candidate set | owner, position (from top or bottom), card if exact, pool id otherwise |
| `k_known_hand` | per known card in the opponent's hand | card, copies known |
| `k_template` | per (player, reference list of the METAGAME frame) | consistent, exact, registered (section 10) |
| `k_archetype` | per (player, archetype label) | possible, exact (section 10) |
| `b_list` | per (player, reference list); level expert+belief only | probability (10^-9 units), log-likelihood, exact (section 10.6) |
| `b_archetype` | per (player, archetype label); level expert+belief only | probability (section 10.6) |

Positions refer to the snapshot's library positions; public objects use the snapshot's handles.
Columns: state_format Appendix A.6.

As built (T9, `Knowledge::rows`, `src/knowledge/output.cpp`):
- **`k_meta` is always one row**, at both levels: `off` writes it with level off and every other
  table empty (so omitted), so a consumer always finds the level. `exact` is 1 when the solver
  settled every reported pool (section 4.3; also between games, with no pool to report), 0 at
  level off. `steps` counts the exact solver's steps for the frame's state (both players; 0 between
  games): a function of the model, so of the events, like every other column. Propagation runs per
  event and is not counted.
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
  fact: exact identities, candidate sets, min/max counts. Built as the engine's
  `--selfcheck-knowledge` (against the engine state itself) and `rmknow --truth` (against a dump's
  omniscient frames), sharing `knowledge::check_truth` (section 9.5).
- **Exactness (the solver):** on small synthetic and sculpted cases (≤ 15-card decks), a
  naive enumerator over the same pool/flow model, written independently of the solver,
  lists every consistent world. With `k_meta.exact` set, the reported min/max and candidate
  sets must equal the enumeration. Built as `rmknow --exactness N` (section 9.5).
- **Sharpness (the model):** hand-written regression scenarios with the deduction a strong
  player would make, asserted on the knowledge dump. Example: after a Brainstorm put-back
  from a hand known to hold {Force of Will, Daze, X}, and X then cast, the top two cards are
  exactly {Force of Will, Daze}. Built as `train/regression/scenarios/knowledge_*.json` with the
  `expect_dump*` keys of `train/test_scenarios.py` (section 9.7).
- **Isolation:** `rmknow` on the recorded frame stream gives byte-identical knowledge tables
  to the in-process run, per writer (the dump's seat frames, a saved stream); and the stream's
  `k_*` tables equal the dump's at every seat and decision (section 9.6).
- **Shadow comparison with today's engine memory:** while the old vector is still emitted, at
  every decision compare `expert`'s implied view (own exact positions 0–4; definitely-in-hand
  opponent cards; cards seen this match) with the engine's `known_top_library_*`,
  `opp_known_hand` and revealed bits. Every mismatch is classified as either:
  - a sidecar or event bug: fixed; or
  - an old-vector leak or quirk: documented, with a regression scenario.

  **One leak was known in advance.** `Zone::identity_known` is tracked per physical card. After an
  opponent's Brainstorm put-back, the old vector still lists a known Force of Will in hand only
  if that particular card stayed in hand, which tells the viewer which cards went back.
  `expert` reports "Force of Will: in hand or in the top two". Built as part of
  `--selfcheck-knowledge`, with every mismatch classified in
  `train/regression/knowledge_divergences.json` (section 9.7): three leaks (this one, its copy
  variant and a library-position one), the rest quirks; no sidecar bug remained.
- **Snapshot/restore:** the knowledge state is saved and restored with the engine snapshot,
  and frames after a restore match the originals.
- **Archetype inference** (section 10): soundness (the true registered list, when it is a
  reference list, is never excluded, its archetype is possible, no other archetype exact), exactness
  (the consistency answers against an enumerator of the hypothesis networks' worlds and against the
  sidecar given that list as an open decklist) and sharpness (at least as sharp as counting the
  cards seen) are section 10.5.

## 8. Effect on the scripted-agent port (plan step 2)

- **The agent reads knowledge from `expert`, not from an `engine-compat` level.** The
  read map's two knowledge reads come from the knowledge tables, both built and shadow-compared
  in step 1:
  - own known top 0–2 and the Thassa's Oracle check: the `k_position` rows of the viewer's own
    library with `card` set (`at` from the top);
  - `opp_known_hand`: `k_known_hand` (per card, the copies definitely in the opponent's hand).
- **Parity target:** byte-identical transcripts, except in games where a classified divergence
  (`train/regression/knowledge_divergences.json`, section 9.7) changed what the agent saw. Those
  games are re-recorded, each with its divergence as the reason. What the divergences imply:
  - `opp_known_hand` → `k_known_hand`: expert knows **less** after an opponent's hidden put-back
    (`put_back_leak`) and after a copy of a known card leaves the hand (`copy_leak`), and **more**
    after the opponent draws a card the viewer knew on top (`known_card_drawn`), after a companion
    is put into hand (`public_entry_untracked`) and by deduction (`deduction`). The agent's
    name-a-card pick (Cabal Therapy) is the reader, so these change what it names;
  - own top → `k_position`: expert knows **more** during a look (`pending_look`: the looked-at
    cards before the effect puts them back) and by deduction (positions known deeper than five,
    a library left with one card's copies); never less in a game. Between games the old vector
    still shows the finished game (`between_games`); the agent's sideboarding reads neither;
  - the opponent's known library top and the decklist revealed bits are not read by the agent
    (`state_format_agent_map.md`), so `position_leak`, `private_look_seen` and
    `harness_preset_seen` change nothing in the port.
- **As found** (step 2 pass 1, `state_format_agent_map.md` section 7): the corpus and smoke games
  are identical on both protocols; the differing games of a 300-match sweep are Cabal Therapy name
  picks after `put_back_leak` (five) and `public_entry_untracked` (one), listed in
  `train/regression/agent_parity_divergences.json`. The own-top differences (`pending_look`,
  `deduction`) changed no choice in any game played.

## 9. Implementation (step 1 T9-T12)

Code: `src/knowledge/` (std-only; `Knowledge` in `knowledge.h` is the entry point), the
standalone driver `tools/rmknow/` and the engine side `src/state_format/knowledge_feed.*`.

### 9.1 Data model (section 3)

| Struct (file) | What it is |
|---|---|
| `Node` (`network.h`) | a set of one player's units the viewer cannot tell apart, at one moment: size, pin (the known card of a one-unit node), the junction that made it and the one that consumed it, root index, and while live its location (`Loc`), handle, drawn-this-turn turn and draw seq |
| `Junction` | `in` units plus `in_fixed` cards = `out` units plus `out_fixed` cards (a split: one in; a merge: one out; after compaction any number on either side, the fixed cards being pinned units compaction replaced by their card) |
| `Root` | a node no junction made: `EXACT` contents (a registered list), `LOWER` bounds (a closed decklist's roots), or `FREE` (a closed decklist's card created from zone none); and the part of the registered list it stands for (`RootPart`: main deck, sideboard, the 75, none), which section 10's hypotheses replace |
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
three-way split of its run. Two runs that become adjacent again and are unpinned outputs of one
junction rejoin (exact: nothing else constrains them), also when compaction has folded the history
behind that junction into it (a searched card located in the shuffled library, then gone).

### 9.2 Bound propagation, the exact solver, compaction

**Bounds.** `BoundStore::propagate` narrows intervals to a fixpoint with a worklist: a node's size
(its classes sum to it), pins, roots, every junction (each member's interval from the other
members' sums and the junction's fixed cards) and the cut (the roots plus every junction's in-side
fixed cards equal the live nodes plus every out-side fixed card). It runs after every event
(`Knowledge::settle`), so each update rule that consults the bounds sees their fixpoint for the
events before it. Constraints only accumulate between compactions, so the store keeps its bounds
between events and only processes nodes, junctions and pins added since (a card entering the
universe, or a compaction, resets it). Bounds
are sound always; they are per-card intervals per node, so they miss joint facts (after a
put-back from {Force of Will, Daze, X} and X cast, each of the top two is one of {Force of Will,
Daze}, but "one of each" is a joint fact) and they lose strength through merges of chains. They
seed the solver, decide which sub-pools can hold a card in the update rules, and are the fallback
when the solver's budget runs out.

**The exact solver** (`Solver::solve`, `Knowledge::solve` once per frame, after its events, per player,
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

**Compaction** (`compact_network`, run by `PlayerModel::compact` after every event that changed the
network, before that event's propagation; the stores restart. While an order-hidden batch waits for
its `library_place`, which is in the same frame, compaction waits too: the batch's units are named
by id until then). Live nodes and
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
(what `k_meta.steps` counted at T10; it now counts the solver's alone, section 6): 11.5% of frames at most 100, 79% at most 1,000, 4.3% at most 10^4, 5.0% at most
10^5 and 0.2% above (the most 162,622, against the budget of 1,000,000); `k_meta.exact` is 1 on
every frame. Of the 32,445 component solves, 79% are memo hits, 19% a leaf at the root and 2% need
the search. The sweep (debug build, 4 cores) takes as long as T9's bounds alone (31 s); a step
costs about 12 ns in a release build. `rmknow --truth` finds no error in any of the
31,532 frames.

`Knowledge::serialize` writes a self-contained little-endian blob (`RMKN`, version 4): level,
viewer, the SCHEMA payload (re-decoded on load), the decklists with their names, the metagame,
match memory (copies seen, the reference lists excluded or left unproven by a finished game), both
`PlayerModel`s (network with its fixed cards and root parts, both stores' bounds, zone lists, the
node count at the last compaction), the handle and draw-link maps, the latest turn, the last
frame's solutions and the template network with its per-list answers and the belief layer's
log-likelihoods and match memory (section 10); card names and
the solver's memos are not included. `deserialize` refuses a malformed
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
- **Soundness** (`knowledge::check_truth`, `src/knowledge/truth.h`): every reported fact against a
  true world: each `k_pool` row's size (the opponent's hand, a sideboard, a library run or block,
  a face-down card), each `k_pool_card` range and no unlisted universe card in a pool, each
  `k_position` card (or, not exact, its true card among the candidates of the run it names), each
  `k_known_hand` count, and each `k_seen` row (game at most the copies the player owns, match at least
  game). Set-aside pools have no counterpart in the true world. Two drivers:
  - **`--selfcheck-knowledge`** (engine; `src/state_format/selfcheck_knowledge.h`; implies
    `--selfcheck-state`): at every decision, each seat's `k_*` tables read back from the
    self-check's frame bytes (`read_k_tables`) against the true world built from the engine state
    (every card object's identity by `card_identity`, hand, sideboard, library order, face-down
    objects by the seat's handle, so face-down pools are checked too), and at the self-check's
    snapshot restore every instance's serialised state against the one saved. Failures are `ERROR:
    selfcheck-knowledge soundness|snapshot: …` lines; `SELFCHECK-KNOWLEDGE: decisions=… facts=…
    errors=…` at match end. It also runs the shadow comparison (section 9.7). A reanimator vs uw_control bo3 fuzz match checks 179,079 facts over
    588 decisions. Proved with temporary bugs: a pool's max one too low (10,859 errors), a known
    position reported one place off (12) and a deserialise that dropped one player's copies seen
    (28 snapshot errors, besides the frame comparison's) were each caught. Over the replay corpus
    (the `rmstate` tier's corpus part): 9,382 decisions, 2,493,424 facts, 0 errors; the `knowledge`
    tier adds a targeted fuzz and three stream sessions (about a million facts more);
  - **`rmknow --truth`** against the same dump's omniscient frames (face-down pools excluded: the
    omniscient handles are not the seat's). On the 168 runs above: 0 errors.
- **`rmknow --selftest`**: synthetic event streams (`tools/rmknow/selftest.cpp`): the section 7
  put-back example (the two put-back units, which nothing tells apart, are one run of the top two
  holding exactly one Force of Will and one Daze, exact), cut accounting, a scry range, a batch, drawn-this-turn, a draw link,
  the viewer's own rearrangement, the search rejoin, a London mulligan, closed-decklist match
  memory, serialisation round trip, forty draw / reveal / put-back / shuffle rounds staying
  under 20 nodes, and the reference lists of section 10.
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
`game_start`). The writer feeds it the seat's SCHEMA, CARDS, DECKLISTS and METAGAME frames as it writes them,
then every STATE frame of the seat exactly as encoded, its events drained from that writer's own
queue (terminal frames included). `Knowledge::table_blocks` gives the `k_*` tables as table blocks
and `rmstate::append_frame_blocks` adds them to the frame before the writer emits it. Their ids
(90-97, and the belief layer's 98-99 at expert+belief) are the highest and they have no text
column, so the result is the frame one `FrameWriter` would have written with those rows (the self-check's roundtrip confirms it at every frame). The
omniscient frames carry no `k_*` tables. With no writer on, no instance exists and nothing runs.

**Levels.** `--knowledge-a` / `--knowledge-b off|expert` (default expert) choose the level each
seat's instances are made with. At level off `Knowledge::feed` decodes only the match frames and the
STATE frames get the `k_meta` row alone (level off, exact 0, steps 0). The level changes only the
`k_*` tables: play, the RNG, BQUERY and every other table are the same. It is an output option like
the writers themselves: the RMLOG header does not record it and `--replay` honours it (a replay
with the same level reproduces the frames).

**Cadence independence: one instance per writer, one output.** Every writer feeds an instance of
its own with exactly the frames it emits, and an instance's output at a frame is a function of the
seat's events up to that frame, not of how they were split into frames: each event is applied and
settled at once (compaction of a network it changed, propagation, copies seen), the snapshot is only
checked, and the frame's solve and `k_meta.steps` read the settled model alone. The event sequence
of a seat is the same for every writer (each drains its own copy of the queue); only the drain
times differ: the dump's at every decision, the stream's at the seat's own decisions. So:
- the stream's frame and the dump's frame for one seat and decision carry the same `k_*` tables
  byte for byte (`test_rmstate.check_stream`; over five stream sessions, bo1 and bo3 with
  sideboarding, two under `--closed-decklist`, 1,858 seat frames, no frame differs);
- a writer's `k_*` tables are a pure function of the seat's frames from that writer:
  `rmknow --isolation` on those frames (a dump's seat frames, or a saved stream) recomputes them
  byte for byte;
- one writer's bytes never depend on another being on: the stream is byte-identical with and
  without `--rmstate-dump` and `--selfcheck-state`, as T5 made it for the rest of the frame.

Settling per event is cheap: a release `rmknow` recomputes a seat's knowledge over a 673-decision
bo3 dump in 0.17 s and over that seat's 270-frame stream in 0.10 s (cost per engine run: below).

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
`knowledge` gate tier (`train/test_knowledge.py`, `docs/ci.md`) runs it on each writer's frames of
its stream sessions (the dump's and the saved stream), after checking that every stream frame equals
the dump's frame for that seat and decision, `k_*` included. At T11, on five stream sessions (bo1 and bo3, one under
`--closed-decklist`, one with seat A at level off), each with its dump, a dump with the self-check
sampled at every decision, and the off variants: 0 differences over 14,514 seat frames.

**Cost.** Feeding a seat frame and building its tables takes 0.16-1.0 ms on average per game in a
release build (most in reanimator vs uw_control bo3: 805 decisions with a dump took 3.3 s at
expert, 1.7 s at off and 0.8 s with no dump, so about 1 ms per seat frame) and 1.5-9 ms in a debug
one (22.7 s, 8.5 s, 3.3 s). A dump pays it twice per decision, the stream once per decision of the
seat, the self-check twice more. The `rmstate` gate tier (corpus under `--selfcheck-knowledge`,
fuzz under the self-check, two fuzz dumps, two stream sessions) takes about 3 min 10 s on 4 cores
(2 min 30 s without the sidecar), the `knowledge` tier about 90 s (its shadow part about 2 s: the
divergence scenarios, the corpus and fuzz comparisons riding on engines that run anyway; its
archetype part about 12 s, section 10.5), and the whole `make check` about 8 min.

### 9.7 Shadow comparison and sharpness (T12)

**Shadow comparison** (`src/state_format/shadow_knowledge.{h,cpp}`, part of
`--selfcheck-knowledge`: it reads the same frames the soundness check reads). At every decision,
for each seat, the seat's `k_*` rows are compared with the old BQUERY vector's memory for that
seat, read through the serializer's own sources (`populate_gamestate`, and `opp_known_hand_cards`
for the whole set behind the ten `opp_known_hand` slots, `src/machine_io.h`):

| Kind | `k_*` side | old side |
|---|---|---|
| `top` | `k_position` rows with a card, positions 0-4 from the top, of each library | `known_top_library_self` / `_opp` |
| `hand` | `k_known_hand` copies, per card | the opponent-hand cards with `Zone::identity_known`, per card |
| `seen` | `k_seen` rows of the opponent with `match` > 0 | the opponent's registered-decklist revealed bits (match_state) |

A mismatch is no engine error. It is printed to stderr when it appears (a mismatch of the same
kind, direction, owner and card at the seat's previous decision is only counted) as one `SHADOW:`
line naming its kind, direction (`old_more`, `new_more`, `conflict`), viewer, owner, card, both
sides' values, whether the old side's claim holds in the true world, the game, turn and decision,
and the frame's events with their source cards (the cause to classify by); a `SHADOW-KNOWLEDGE:`
summary closes the match. The Python tiers classify every line against
`train/regression/knowledge_divergences.json` (first matching signature: kind, direction, owner,
a regex over the line; `train/test_knowledge.py`): the `rmstate` tier's corpus part and the
`knowledge` tier's fuzz and stream sessions, and the `knowledge` tier runs each signature's
regression scenario to show it prints a line of that class. An unclassified line fails the tier.

**What it found** (every class investigated on sample games; counts are the lines printed over
the replay corpus, 9,382 decisions and 371,764 comparisons, and over 800 release-build explore
fuzz matches, half of them bo3 with sideboarding, all league pairings):

| Signature | Kind, direction | Classification | Lines (corpus / fuzz) | Scenario |
|---|---|---|---|---|
| `between_games` | hand, top; old_more | quirk: between games BQUERY still describes the finished game | 0 / 765 | `knowledge_divergence_between_games` |
| `put_back_leak` | hand; old_more | **leak**: the known Brainstorm put-back leak (section 7) | 19 / 387 | `knowledge_divergence_put_back_leak` |
| `copy_leak` | hand; old_more | **leak**: known-ness per physical copy tells which copy of a known card left the hand | 5 / 254 | `knowledge_divergence_copy_leak` |
| `position_leak` | top (opponent's library); old_more | **leak**: the old record follows a card out of a range of positions the viewer cannot tell apart by its true position (Ponder's or Doomsday's rearrange, a pick among looked-at cards) | 2 / 139 | `knowledge_opponent_ponder_unordered` |
| `pending_look` | top; new_more | quirk: during a look the old record does not have the looked-at cards yet (a search never) | 1,736 / 48,795 | `knowledge_opponent_ponder_unordered` |
| `private_look_seen` | seen; new_more | quirk: the old revealed bits count public reveals only (Mishra's Bauble's look is private) | 22 / 99 | `knowledge_divergence_bauble_look` |
| `harness_preset_seen` | seen; new_more | quirk: test-harness presets bypass the old reveal marking | 0 / 0 (harness only) | `knowledge_section7_put_back_pool` |
| `known_card_drawn` | hand; new_more | quirk: a card known on top of the library stays known when drawn | 48 / 727 | `knowledge_divergence_bauble_look` |
| `public_entry_untracked` | hand; new_more | quirk: a companion put into hand is known to both players | 3 / 120 | `knowledge_divergence_companion_to_hand` |
| `deduction` | top, hand; new_more | quirk: a sound deduction the old vector does not make (a library or hand whose cards the list minus the cards seen pins, a hand reveal that pins a put-back pool, positions deeper than five) | 1 / 153 | `knowledge_section7_put_back_pool` |

The 800 fuzz matches made 223,002 decisions, 12.3 million comparisons and 57.9 million soundness
facts (no soundness error). Most lines are `pending_look` (a search shows the searcher the whole
library, up to five lines per look). No `conflict` (both sides naming different cards) and no
old-side claim false in the true world occurred. Within a game expert never knows less than the
old vector except through the three leaks. The comparison also turned up three bugs, fixed: Doomsday's one-at-a-time search looked again at the
cards it had already chosen and put on top, so the other player's events could not tell a pick from
them and the sidecar lost the positions Doomsday's own rules give away (`src/zone_search.cpp`
now looks only at the cards the search may still pick; the old record had them, an `old_more` with
a search in the frame, which the `position_leak` signature therefore excludes); a card put on the
bottom of an empty library got position 1 (`Orderer::move_object`; the soundness check's "positions
are not 0..n-1"); and two library runs from one junction did not rejoin after compaction had folded
the history behind it into it (`PlayerModel::rejoin`; a run split around a searched card stayed
split after the card left).

**Sharpness scenarios** (`train/regression/scenarios/knowledge_*.json`). A scenario with an
`expect_dump`, `expect_dump_not`, `expect_dump_final` or `expect_dump_final_not` key runs with
`--rmstate-dump FILE --state-dump json`, and `train/test_scenarios.py` renders every seat STATE
frame's `k_*` tables as one text line per fact (`<seat> d=<decision> g<game> t<turn> pool opp
library top0+2 size=2: Daze 1-1, Force of Will 1-1`, `position self top0 = Daze`, `known_hand
Island x5`, …; `sb` for the turn between games; grammar in the module docstring), which the keys'
regexes match (any frame, or each seat's last frame). `test_scenarios.py --dump NAME` prints them.
Scenarios may now be bo3 (`format`). The sharpness set:

| Scenario | The deduction |
|---|---|
| `knowledge_section7_put_back_pool` | section 7's example: a hand known through Cabal Therapy, a Brainstorm put-back, the Bolt cast, then a Thoughtseize showing only Islands: A's top two are exactly one Force of Will and one Daze |
| `knowledge_own_brainstorm_put_back` | own Brainstorm put-back: own top two known exactly (`k_position`); the opponent knows neither position |
| `knowledge_shuffle_keeps_contents` | a shuffle drops every position fact and keeps the contents (both viewers: the library holds exactly one Force of Will and one Daze) |
| `knowledge_sylvan_library_keeps_known_hand` | Sylvan Library's put-backs come from this turn's draws, so the known hand cards stay definitely in hand |
| `knowledge_miracle_draw_link` | the miracle reveal's draw link pins the drawn card: with one Triumph already known in hand, both are known |
| `knowledge_opponent_ponder_unordered` | the opponent's Ponder leaves the looked-at range unordered: a known top card becomes one of three positions, no `k_position` |
| `knowledge_bo3_sideboard_swap` | game 2: the Leyline of the Void the opponent starts with proves a swap (sideboard Leyline 4-4 in game 1, 0-4 in game 2, 0-3 once seen) |

plus the divergence scenarios above, which assert expert's side of each difference. With
`--knowledge-a off --knowledge-b off` all twelve fail; with the draw link ignored the miracle
scenario fails, and with the drawn-this-turn subset ignored the Sylvan Library one does.

## 10. Archetype inference: the sound layer

What a strong player can say for certain about which deck the opponent is on: among a set of known
decklists (the metagame), which ones the opponent could still be playing, and so which archetypes.
It is the same kind of fact as everything else in `k_*` (true in every world); the probabilistic
layer that weighs the consistent lists is kept apart, in `b_*` tables of a level of its own
(section 10.6).

### 10.1 The reference metagame (input)

- **`METAGAME` frame** (state_format §4, A.4), sent to every viewer once per match after
  `DECKLISTS`, the same bytes for every viewer (it is public): the archetype labels (row index =
  archetype id), the reference lists (row index = list id: deck name, archetype, main and side
  counts) and their cards (physical card ids, main and side copies). The sidecar never reads a file;
  the metagame reaches it only through this frame, so it stays a pure function of the viewer's
  stream.
- **Source** (`src/state_format/metagame.{h,cpp}`, read once per process): `bin/resources/decks/archetypes.json`.
  Its `_archetypes` key is the canonical label list, in index order (also what `train/archetypes.py`
  reads as `ARCHETYPES`; append only), and `unknown` is appended last. Every tagged deck stem whose
  `.dk` file exists, whose label is listed and whose every card has a vocab entry is a reference
  list, in ascending stem order. Today all 13 tagged decks are: the ten league decks plus `delver`,
  `doomsday` and `mav` (none is left out; the `knowledge` tier fails when a tagged deck is missing).
- **The registered decks are not added.** The metagame is the same for every match; a match's deck
  is a reference list only when it is a tagged deck (a test-harness or temp deck usually is not).
  CARDS carries a row for every metagame card, so every card id of the frame resolves.

### 10.2 Consistency (the sound layer)

For each player and reference list L, `k_template.consistent` says whether L can be the player's
registered list given everything the viewer knows.

- **A list the viewer sees** (their own; the opponent's with open decklists, register 11) is
  explicit: L is consistent exactly when it has the same counts (main and side, per physical card)
  as that list, `exact` always 1. The model's own solve of that root (section 9.2) is the network
  check of the registered list: the network has a world with it, or an `ERROR` line says otherwise.
- **The opponent's hidden list** (`--closed-decklist`): L is consistent when a world of the
  opponent's network exists with L as the root: game 1 the main deck exactly L's main deck and the
  sideboard exactly L's sideboard; games 2-3 the 75 exactly L's 75, split between deck and sideboard
  in any way (test-harness presets and a companion, outside the zones at `game_start`, come from L's
  set-aside rest). This is the solver's feasibility question with the root swapped:
  `hypothesis_network` (`src/knowledge/templates.h`) rebuilds the network with each list root
  (`Root::part`: the closed decklist's lower-bound roots of the stated library and sideboard
  sizes, game 1 one each, games 2-3 one for both) made the output of a junction from a new `EXACT`
  root of L's part, beside a set-aside node of the units L has beyond the stated size (fewer: no
  world); every card created from zone `none` (free roots, and the pinned ones compaction turned
  into fixed cards: the network's net fixed cards) comes out of those set-aside nodes, with a
  remainder for the set-aside cards not seen yet (negative: no world). Nothing else changes, so every
  pin, junction and size constrains L as it constrains the opponent's own root. `Solver::feasible`
  (`src/knowledge/solver.h`) answers it: the system of section 9.2 step 1 with no pool, compacted
  keeping no node (exact for whether a world exists), seeded with the hypothesis network's own
  propagated bounds (a contradiction there: no world), each component searched for any leaf with
  worlds. Cards nothing pins or fixes share the OTHER class (they are interchangeable, so this is
  exact).
- **Exact when proven.** A finished search (or a size or propagation contradiction) proves the
  answer, `exact` = 1. Past the budget (250,000 steps per list and template network) the list is
  reported consistent with `exact` = 0: sound, a list is never excluded without proof. On the
  gate's sessions no row is unproven.
- **Copies seen.** A list holding fewer copies of a card than the opponent was seen to own in any
  game of the match (`k_seen.match`) is excluded without a search.
- **Match memory.** The registered list is the same in every game, and the games' worlds are
  independent, so at `game_end` the game's answer for every list becomes match memory: a list proven
  inconsistent stays so for the match (in game 2 a Force of Will cast from the hand no longer
  says it was in the main deck, but game 1 did), and one left unproven keeps `exact` 0. Between
  games the rows are the match memory and the copies seen.

### 10.3 Cost and cadence: the template network

Only an event that pins one of the opponent's cards or adds a root (or starts a game) can make a
list inconsistent: a split or merge of nodes (a draw, a shuffle, a put-back) always has a solution.
So the lists are checked against the **template network**, the opponent's network as it was after
the last such event (settled), each list once per template network, at the first frame after it
(or at `game_end`). Every answer is then a function of the events, whichever frames computed it, so
the dump's and the stream's frames carry the same rows and a restored snapshot recomputes the same
(the template network and its answers are serialised). A frame with no new pin costs nothing; the
steps are not in `k_meta.steps` (section 6).

Measured with a release `rmknow` over saved `--closed-decklist` streams, with and without their
METAGAME frame: the archetype work adds 6-18 ms per seat over a whole match (bo1 and bo3 streams of
250-370 frames; about 0.03-0.06 ms per frame on average), and about as much over a dump's every
decision. With open decklists it is the rows alone. Unseeded, the feasibility search of a game-3
template network ran past its budget on the true list (31 unproven rows in one bo3 session);
seeding it with the hypothesis network's propagated bounds settles every check of the gate's
sessions. The CARDS frame's metagame cards cost the engine about 4 ms (debug build) once per writer.

### 10.4 Output tables (`k_template`, `k_archetype`; state_format A.6)

- **`k_template`**: per player (self first, then the opponent) and reference list (list id order),
  always every list, at level expert, between games too: `consistent`, `exact` (the answer is
  proven) and `registered` (the list is the classification of the player's registered list, when
  the viewer sees it: the reference list of the same deck name if there is one, else the one with
  the most cards in common, Σ over cards of the lesser main count plus the lesser side count, ties
  to the lower id, and none when nothing is in common). `registered` is a deterministic
  classification of explicit information, not a deduction; it is never set for a hidden list.
- **`k_archetype`**: per player and archetype label (label order, `unknown` included): `possible`
  (some consistent list has this label) and `exact` (possible, and every consistent list has it).
  The hypothesis space is the metagame: with no consistent list (an off-metagame deck) no label is
  possible. An unproven list counts as consistent, so `exact` stays sound: the true list, when it is
  a reference list, is among the consistent ones.
- Level off: no rows (k_meta alone). No METAGAME frame (or no lists): no rows.

### 10.5 Verification

- **Soundness** (`knowledge::check_truth`, zero tolerance; `--selfcheck-knowledge` against the
  engine's registered lists, `rmknow --truth` against the omniscient DECKLISTS frame): when a
  player's registered list equals a reference list, its row is consistent, its archetype possible
  and no other archetype exact; the rows marked registered are exactly the classification of the
  true list when the viewer sees it, none otherwise.
- **Exactness** (`rmknow --exactness N`, the `templates` cases): simulated closed-decklist games of
  small decks (3-4 reference lists, main 8-10 and sideboard 2-3 cards of 4 kinds; draws, casts, top
  and hand reveals, put-backs, scries, mills, shuffles, a wish from the sideboard, sometimes a
  companion from zone none): after every frame each list's row against two independent answers, the
  enumeration of its hypothesis network's worlds (`tools/rmknow/enumerate.h`) and a sidecar fed the
  same events with that list as the opponent's open registered list (no error and its network's
  enumeration finds a world). `--exactness 40`: 2,576 template checks, all exact and equal;
  `--exactness 400`: 25,104 (11,916 of them lists with no world) in 1.3 s (release).
  Deliberately broken, the sideboard root standing for the 75 (192 failures), the cards from zone
  none left free (3) and a feasibility search ignoring an infeasible component (132) are caught.
- **Sharpness** (the `knowledge` tier's `archetype` part, `--machine-format rmstate` sessions; and
  the scenario `knowledge_archetype_open_registered`): with open decklists every list a seat sees is
  exactly consistent and classified registered from the first frame; under `--closed-decklist`,
  before any of the opponent's cards is seen every label of a list that can hold the library and
  sideboard the game started with is possible (all seven against a 60/15 deck; only slow_midrange
  against bw_dnt's 80 cards), no consistent list lacks a card the opponent was seen to own, the
  archetype is exact once the cards seen leave one label (Delver of Secrets is only in `delver`;
  Daze and Lightning Bolt only in the two tempo Delver lists), and it becomes exact in each session.
  `rmknow --selftest`'s `reference_lists` case covers the main-deck attribution (a Force of Will
  cast in game 1 excludes a list holding it only in the sideboard), the copies-seen exclusion, the
  match memory in game 2, the classification by cards in common and serialisation.

### 10.6 The belief layer (`b_*`, level `expert+belief`)

The sound layer says which reference lists the opponent can still be on; the belief layer says how
likely each is. Its numbers are probabilities under a stated model, not facts, so they live in their
own tables (`b_list`, `b_archetype`, ids 98-99) behind a level of their own: `--knowledge-a/-b
expert+belief` (state_format register 38). At that level the `k_*` tables are expert's, byte for
byte except `k_meta.level`; at `off` and `expert` the `b_*` tables are empty (so absent from the
frame; the schema is the same). Code: `src/knowledge/belief.{h,cpp}` (the likelihood and the
posterior), the integration in `src/knowledge/archetype.cpp`.

**The model** (one game of a player whose registered list L is hidden):
- the prior is uniform over the METAGAME frame's reference lists;
- every junction of L's hypothesis network (section 10.2, built on the same template network)
  distributes the units it takes in uniformly at random among its out nodes and out-side fixed
  cards. For the random parts of the game this is exact: a shuffled library is a uniformly random
  order, a draw takes the top of a run, a mill or a top reveal shows it. The opponent's own choices
  among cards the viewer cannot tell apart are modelled the same way, as uninformative: which
  unknown hand card is cast, discarded or put back, which of several looked-at cards goes where, is
  a uniformly random one ("the cards seen are a uniformly random sample of the cards that could be
  them"). A card the viewer already knew to be in the hand that is played is that card (the network
  takes the known unit, section 9.1), which says nothing more; a card set aside beyond the stated
  library and sideboard (a companion, a preset) is a uniformly random card of the list part;
- the evidence is every pin and fixed card of the network (each identity the viewer saw), and the
  likelihood P(evidence | L) is the probability that all of them hold. A junction with in-side
  composition y (n units) gives its out nodes compositions x_o (sizes s_o) with probability
  Π_c y_c! / Π_o Π_c x_oc! × Π_o s_o! / n! (multivariate hypergeometric), so the likelihood is the
  sum over the network's consistent worlds of the product of its junctions' probabilities: the
  sound layer's worlds, weighted. The weight counts arrangements of the physical (distinguishable)
  cards, not multisets: a one-card draw that turns out to be Lightning Bolt has probability 4/60
  under a list with four Bolts and 1/60 under one with one, where the number of consistent multiset
  worlds (one each) would not tell the lists apart;
- games are independent shuffles, so a match's likelihood is the product over its games: at each
  `game_end` the game's log-likelihood joins the match memory (beside the sound layer's), games 2-3
  adding theirs under the 75 hypothesis (the deck a uniformly random part of the 75, the sideboard
  the rest: the opponent's sideboarding is a choice, modelled as uninformative like the others).
  Main deck and sideboard are told apart only in game 1, as in the sound layer;
- the posterior: p(L) ∝ P(evidence | L) over the lists the `k_template` rows keep (consistent,
  proven or not), zero for the others. A list the sound layer could not exclude never gets zero:
  if its weighted count is zero (it then has no world, but the sound layer did not prove it) or runs
  past its budget, the copies-seen fallback below stands in. A list the viewer sees (their own, the
  opponent's with open decklists) is explicit: the reference lists equal to it share probability 1
  (`loglik` 0), so there is nothing to infer under open decklists.

**The weighted count** (`network_likelihood`): exact, swept backwards over the junctions. A
junction's uniform distribution is a uniformly random bijection from its out slots to its in slots,
so a pinned slot holds the card of the in slot it maps to: a known one (a pinned in node, an
in-side fixed card), a slot of the EXACT root (a shuffle of the list part: Π_c (T_c)_d over the
demanded slots) or a slot of an unpinned in node, which passes the demand for that card on to the
junction that made the node. The state is the demand (a count per named card) on the nodes whose
consumer has been swept and whose producer not yet, summed per producer (a junction reads only the
sum of its outs' demands), so its size is bounded by the pins, not by the nodes' sizes (a 60-card
library run costs nothing); a node's slots are exchangeable, so the demand's multiset is all that
matters. Per junction, with D the demand on its out side and n in
slots, P = Π_c D_c! / (a_c! Π_t d_tc!) × Π_c (K_c)_a_c × Π_t (s_t)_|d_t| / (n)_|D|, summed over the
ways a_c of each card's demand land on its K_c known slots and d_t on each other in node t of s_t
slots ((a)_k the falling factorial). A forward sweep over the nodes' full compositions, the obvious
alternative, ran past any budget on the first real game (a 60-unit node between the root and the
rest of the network). Budget: 200,000 leaves per list and template network; the fallback past it
(`seen_log_likelihood`, the row's `exact` 0): the cards seen this game as a uniformly random ordered
sample of the list's 75, Σ_c log (T_c)_k_c − log (N)_m (how often it is needed: Cost, below).

**Arithmetic and determinism.** Doubles in a fixed evaluation order (ordered maps, a log-factorial
table built by cumulative sums), states rescaled after every junction with the scale kept as a log;
log-likelihoods are summed per list. The posterior is turned into integers of 10^-9
(`fixed_point_posterior`): one unit for each kept list, the remaining 10^9 − k units by largest
remainder (ties to the lower list id), so a player's rows sum to exactly 10^9 (0 when no list is
consistent), a kept list is never 0 and an excluded one always is. `b_archetype` is the sum of its
lists' units. `b_list.loglik` is the match log-likelihood in milli-nats (i32 minimum for an excluded
list). Like the sound layer, the likelihood is computed once per list and template network (at the
first frame after it, or at `game_end`), so the rows are a function of the seat's events: the stream's
and the dump's frames carry the same bytes, `rmknow --isolation --level expert+belief` reproduces
them, and the serialised state (version 4) holds the per-list template log-likelihoods, the copies
seen they were taken at (the fallback's input) and the match memory, so a restored snapshot
continues exactly.

**Verification** (the `knowledge` tier's `belief` part, `docs/ci.md`):
- *structure*: two `--machine-format rmstate` sessions at expert+belief, each with a dump (one
  under `--closed-decklist` and `--selfcheck-knowledge`): stream frames equal the dump's (`b_*`
  included), `rmknow --isolation` on each writer's frames, `rmknow --truth` (and the self-check's
  soundness pass, through `knowledge::check_truth`): per player a list has probability zero exactly
  when `k_template` excludes it, the true list (a reference list) is positive, the rows sum to the
  full scale, each label is the sum of its lists; the self-check's snapshot restores compare the
  serialised state. A bo3 session under `--closed-decklist` with sideboarding (776 frames per seat,
  288,735 self-checked facts) shows no difference and no error;
- *exactness* (`rmknow --belief-exactness N`, `tools/rmknow/exactness.cpp`): the simulated
  closed-decklist games of section 10.5 with the viewer at expert+belief, after every frame against
  two independent answers: the weighted forward enumeration of each consistent list's hypothesis
  network over the lists' full card classes (`weighted_world_probability`, `tools/rmknow/enumerate.h`:
  the same probability, so the log-likelihoods must agree to 10^-9), and each list's game played out
  directly under the model (`tools/rmknow/generative.h`: an exact distribution over the true states,
  library order included, that every observation reweighs; nothing in common with the network): the
  posteriors must agree to 10^-9 and the fixed-point rows round them; and the serialised state must
  restore to the same bytes and posterior. `--belief-exactness 100`
  (release, 22 s): 1,622 enumeration and 873 model checks, all equal (the gate runs 10 cases, 179
  and 95 checks, about 5 s in a debug build). Deliberately broken, multiset instead of arrangement
  counting (a root slot holding any card the list has, unweighted) fails 327 checks in 10 cases, and
  the known slots' (K_c)_a factor dropped fails 23 in 100 (the first at case 40: two known copies of
  one card on one in side are rare);
- *calibration* (not soundness): the probability the seat's `b_list` / `b_archetype` rows give the
  opponent's true list and archetype at every seat frame of a live game, closed-decklist sessions
  over all 13 reference lists (each list against others; release build). Figures: mean log-loss of
  the true list (uniform prior: ln 13 = 2.565; "sound layer" is uniform over the consistent lists,
  `k_template` alone), the same on the frames where more than one list is still consistent (where
  the weighting matters), the archetype's log-loss (the uniform prior over lists gives a label its
  share of the lists), Brier score, expected calibration error over 10 bins of every (frame, list)
  pair, and the true archetype's mean probability at a game's first and last frame:

  | Sessions | Frames | List log-loss (uniform / sound layer) | Several lists consistent: frames, log-loss (sound layer) | Archetype log-loss (uniform) | Brier list | ECE | True archetype first → last |
  |---|---|---|---|---|---|---|---|
  | random replies (`--calibration 3`: 78 sessions, bo1 and bo3, 256 games) | 21,887 | 0.125 (2.565 / 0.128) | 1,712: 1.601 (1.638) | 0.080 (1.745) | 0.055 | 0.0005 | 0.568 → 1.000 |
  | explore agents, bo1 (26 sessions) | 7,166 | 0.217 (2.565 / 0.227) | 1,053: 1.479 (1.545) | 0.139 (1.773) | 0.096 | 0.0011 | 0.282 → 1.000 |
  | scripted (hard) agents, bo1 (26) | 8,098 | 0.176 (2.565 / 0.168) | 1,121: 1.274 (1.213) | 0.078 (1.716) | 0.097 | 0.0030 | 0.282 → 1.000 |
  | scripted agents, bo3 (13 matches, 58 games) | 9,380 | 0.122 (2.565 / 0.094) | 918: 1.247 (0.963) | 0.029 (1.659) | 0.080 | 0.0050 | 0.678 → 1.000 |

  The agents need BQUERY, which `--closed-decklist` refuses (its name-a-card menus are uncapped),
  so their games are played with open decklists and replayed through `rmknow --belief
  --hide-lists`, which feeds the seat's DECKLISTS frame without the opponent's list (the events are
  the same either way). Under random and explore play the weighting beats the sound layer's
  uniform answer where it matters (1.601 against 1.638 and 1.479 against 1.545 nats) and the
  reliability bins track the frequencies (random play: predicted 0.13 / 0.25 / 0.34 / 0.44 / 0.52 /
  0.65 against 0.16 / 0.17 / 0.36 / 0.52 / 0.46 / 0.58, bins of a few hundred correlated frames).
  Against the scripted agent it is worse than uniform over the consistent lists (1.274 against
  1.213 in bo1, 1.247 against 0.963 in bo3) and overconfident (bo3: pairs predicted 0.6-0.9 are the
  true list 0-9% of the time): its choices are not random (a smart mulligan keeps hands with the
  cards it wants, tutors fetch them), so the cards it shows over-represent what it looks for, and
  the lists it confuses are near-copies within one archetype (archetype log-loss 0.029). In games
  2-3 the match memory usually leaves one list (random play: no frame with several). The gate's
  light version (13 random-reply bo1 sessions, every list against the one five on: 2,401 frames)
  checks only that the list and archetype log-losses beat the uniform prior's and that the true
  archetype's mean probability rises from a game's first frame to its last (it printed list
  log-loss 0.253 against 2.565, 1.175 against the sound layer's 1.195 on its 517 frames with several
  consistent lists, archetype 0.122 against 1.725, true archetype 0.282 → 1.000).

**Cost.** The likelihood runs once per consistent list and template network (an opponent pin or
root event), beside the consistency check. Release `rmknow` over saved closed-decklist streams, both
seats: bug vs wubg_doomsday bo3 (785 records) 213 ms at expert, 232 ms at expert+belief; reanimator
vs uw_control bo3 338 / 360 ms; so about 10 ms per seat and match. The worst case is a hand merged
turn after turn with that turn's draws (gw_maverick's Sylvan Library): the demand on the library's
draws spreads over many states (up to 2.5 million leaves for one list), the sweep gives up at its
budget (about 0.15 s each in a release build) and the fallback stands in: gw_maverick vs lands bo1
620 / 1,213 ms. Over the random-reply calibration sessions 310 of 21,887 frames (1.4%) carry a
fallback likelihood, 299 of them against gw_maverick and 11 against reanimator; none in the agents'
sessions. The rows themselves (an exp and a rounding per list) cost nothing measurable. The gate's
`belief` part takes about 15 s on 4 cores (debug build).

**Not built: `b_card`** (expected copies per hidden pool and card under the posterior). It needs,
per list, the expected composition of each pool given the evidence, a forward pass of the same
sweep per pool; left for later.
