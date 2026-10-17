# RMSTATE: the engine's state format (draft spec)

Status: **step-1a spec, decisions closed** (register in section 11). Companions:
[`state_format_agent_map.md`](state_format_agent_map.md) (agent read map) and
[`knowledge_store.md`](knowledge_store.md) (recall/inference sidecar). Nothing here is
implemented yet.

RMSTATE replaces the float state vector (BQUERY). The engine sends it to whoever makes a
decision: an in-process agent, the C++ CLI, or an external machine agent over `--machine`.

## 1. Principle: the engine conveys explicit information; the actor remembers and infers

The engine tells each viewer **only what that viewer explicitly knows at that instant**:

- the open decklists;
- their own hand;
- the public zones;
- the counts of hidden zones;
- cards being revealed right now, or that the viewer may look at right now.

The engine does **not** remember on a viewer's behalf. Examples it no longer carries: "you saw
Force of Will in their hand last turn", "the card you put back is second from the top", "this
card was revealed once this match".

Perfect recall and inference belong to the **actor**, the agent playing the seat. They are
built from what the engine reports, through the C++ knowledge sidecar that sits beside the
agent (section 7). How much an agent infers is agent policy and can grow without touching the
engine.

Two channels per viewer, because a snapshot alone cannot carry information that was explicit
only for a moment:

1. **Snapshot**: the current explicit state, at each of the viewer's decisions (section 4).
2. **Observation events**: every piece of explicit information since the viewer's previous
   frame, in order, each as known at the instant it happened (section 5).

The events channel is required for three reasons:

- **No frame while a reveal is live.** The engine never asks a seat whose only legal action is
  a pass, so a viewer may get no frame while a card is revealed. Examples: Gitaxian Probe cast
  on them, an upkeep reveal, a reveal that starts and ends inside one resolution.
- **Moves into hidden zones are explicit only at the moment they happen.** "The card you saw
  goes on top of their library" is visible then and nowhere afterwards.
- **A shuffle invalidates positional knowledge,** and the actor has to be told it happened.

Snapshot plus event history is everything the player saw. An actor with no memory can still
play legally from the snapshot and menu alone.

## 2. Viewers

| Viewer | Gets | Used by |
|---|---|---|
| `A`, `B` | that seat's explicit information (sections 3–5) | seat agents, the CLI human |
| `omniscient` | everything, with no filtering | explicit flag only: debugging, spectator, the reference side of the leak tests. Never sent to a seat |

Two-player only (CLAUDE.md scope): "the opponent" is always the other seat.

## 3. Visibility rules (what is explicit right now)

| Zone / object | Viewer = owner/controller | Viewer = opponent | Basis |
|---|---|---|---|
| Battlefield, stack, graveyard | full | full | CR 400.2 |
| Exile, face up | full | full | CR 406.3 |
| Exile, face down | identity visible to each player in that card's **look window**: a player who looked at the card before it was exiled face down, or whom an instruction allows to look, may keep looking until it leaves exile. In practice this is usually the owner | same rule, so usually hidden | CR 406.3 |
| Face-down permanent or spell | full (controller may look) | face-down characteristics only; the object is still distinguishable | CR 708.5, 708.6 |
| Hand | full | count, plus identities of cards in a current reveal window | CR 402.3 |
| Library | count, plus cards in a current look/reveal window | count, plus cards in a current reveal window | CR 401.2, 401.3 |
| Sideboard (bo3) | own contents | count only | registered list is open (below); post-board configuration is hidden |
| Decklists | own registered list; the opponent's too unless `--closed-decklist` | same | open by default (decision 11), match header |

**Reveal windows.**
- A reveal shows a card to all players for as long as the revealing effect needs it (CR
  701.20a). A look is the same, shown only to the specified player (CR 701.20e).
- The engine keeps one per-viewer **reveal-window** set. It generalises today's
  `Game::revealed_in_library` to hands and looks.
- A window ends when the effect finishes with the card, when the card changes zones, or when
  its library is shuffled or reordered (CR 701.20d).
- Continuous reveals (e.g. "play with the top card of your library revealed") are windows that
  last as long as the effect.
- The snapshot shows exactly the cards in the viewer's windows.

## 4. Snapshot

A snapshot is a full description at one decision (no deltas). Tables are listed here; their
columns are fixed in the schema (section 9) and listed in Appendix A, which also adds a few
child tables.

| Table | Rows | Notes |
|---|---|---|
| `header` | 1 | format version, schema hash, viewer, game number, decision number |
| `turn` | 1 | turn, step, active player, priority holder, pass flags, extra turns queued, land drops remaining |
| `player` | 2 | life, poison, energy, mana pool, designations (monarch, city's blessing), day/night, per-turn counters, hand count, library count, mulligans taken |
| `object` | one per identifiable object | handle, zone, owner and controller (each both as absolute seat and relative to the viewer: self/opponent), card id (or "face-down"), face/transformed state, tapped/sick/phased, damage, base and post-layer characteristics, attached-to, chosen name/colour/type, linked exiled cards |
| `object_keyword`, `object_counter`, `object_ability`, `object_provenance` | children of `object` | every keyword with its parameters; counters by kind; abilities by reference (decision 2); which sources currently affect the object, what kind of effect, and when it expires (decision 1) |
| `combat` | one per attacker/blocker pairing | attack target, blockers, damage-assignment order |
| `stack` | one per stack object | kind (spell / activated / triggered / copy), source handle, controller, ability reference, modes, X, kicked/alternative costs, plus child table `stack_target` |
| `delayed_trigger`, `emblem`, `player_effect`, `play_permission`, `shield` | as present | full lists, no slot caps |
| `lki` | departed objects still referenced | last-known characteristics for a stack or effect reference |
| `reveal` | one per object in a current window for this viewer | handle, zone, position if in a library, card id, who it is revealed to |
| `pending` | 0 or 1 | the paused flow (`PendingQuery::Tag`), deciding seat, source handle |
| `pending_cost` | 0 or 1 during a cost-payment decision (manual mana payment under `--no-auto-mana`, pitch/alternative costs, unless-payments, X choices) | what is being paid for (spell/ability handle and which cost: mana, alternative, unless, additional), the **remaining cost** by pip type (W/U/B/R/G/C, generic, each hybrid and Phyrexian pip, snow, X), and the amount already paid. Child table `pending_cost_pip` for non-plain pips |
| `menu` | one per legal action | category, action type, source/target handles, ability reference, ordinal, cast-variant flags, card-is-public flag, description. No length cap; one integer answers the decision |
| `match` | 1 | game number, wins, who played first this game, sideboard phase, own current deck/sideboard during sideboarding |
| `events` | the drained event queue | section 5 |

Sent once per match (not per snapshot):

- `CARDS`: the card table for every implemented card and token in either registered list, plus
  tokens they can make. Each entry has the vocab id, name, uid, printed characteristics, the
  **oracle text** (always: `CardData::oracle_text`, parsed from the script's `Oracle:` line;
  token scripts carry one too), an ability list (category, costs, timing, script line text)
  and, optionally, the raw script.
- `DECKLISTS`: both registered lists with their deck names (only the viewer's own under
  `--closed-decklist`).

Not in the snapshot:

- known library positions;
- "seen in hand earlier";
- match-scoped revealed bits;
- the opponent's hand beyond its count and its current windows;
- the contents of the viewer's own library.

All of these are actor-side (section 7).

## 5. Observation events

**Queueing.**
- The engine queues events per viewer in `Game`, so snapshots and RMLOG replay cover them.
- A frame for a viewer drains that viewer's queue in order.
- Game start, game end and match end are events too, so actors keep memory across bo3 games.
- Events still queued when a game ends are delivered in that viewer's next frame, after a
  game-boundary event.
- Every event carries `seq`: per viewer, 1 at match start, +1 per event, never reset within
  the match. A gap means a lost event; decoders refuse it.

Each event carries **only what was explicit to that viewer at that instant**, projected through
the same visibility rules as the snapshot.

**Library positions.** A position in a library is (anchor, pos, span):
- anchor: top or bottom; pos counts from that end, 0 = the end card;
- span: how many consecutive positions from pos the card may be in: 1 = exact, n = one of n,
  0 = unknown (anywhere);
- a source position is read before the move, a destination position after it;
- cards move one at a time, so each event shifts the positions after it. The actor tracks
  library sizes from `game_start` and the moves, so it can convert between anchors;
- a pick from cards the viewer cannot tell apart is a range. Example: the opponent scries 2
  and bottoms one: "a card from top positions 0–1 → bottom 0". The range depends only on
  public counts.

**Order-hidden batches.** One instruction may put several cards into a library in an order
this viewer did not see: "in any order" chosen by the other player, or "in a random order"
(hidden from everyone). If the viewer can identify any of those cards, each card's
`zone_move` carries the same `batch` and no destination position, and a `library_place`
closes the batch with the range.

| Event | Fields (viewer's projection) |
|---|---|
| every event | `seq`; `turn` (the game turn it happened in); source object of the effect (handle, else card id) where there is one |
| `zone_move` | one card. From zone/owner, plus its position if from a library, plus its hand subset if from a hand (`drawn_this_turn` when the instruction may only choose among cards drawn this turn, e.g. Sylvan Library's put-back; else `any`); to zone/owner, plus its position (or `batch`) if into a library; handle before/after if identifiable (a face-down public object keeps a handle); card id if identifiable; face-down flag (exiled face down); token/copy flag. Zone `none` = created (token, spell copy, harness preset) or ceased to exist |
| `draw` | player, count, opening-hand flag (opening-hand and mulligan draws are not draws for "drawn this turn"); per card: handle after and card id for the drawer; for another viewer, handle before and card id only if the card was in that viewer's live window |
| `library_place` | closes an order-hidden batch: batch, library owner, range (anchor, pos, count), order known (false) |
| `reveal_start` / `reveal_end` | cards (handles, zone, owner, library positions, card ids), revealed to whom, source effect; per card, a draw link (`seq` of the `draw` event and the card's index in it) when the reveal shows a card as the one just drawn (e.g. Miracle). `reveal_end` also closes a look window. A window that ends because its card moved has no `reveal_end`; the move ends it |
| `look` | looker, zone/owner, count; per card: library position; identities and handles only for the looker. A search of another player's hidden zone is also a `look` at all of it |
| `shuffle` | library owner |
| `reorder` | library owner, range (anchor, pos, count), whether this viewer saw the result and, if so, the card at each position. No engine site today: rearrangements (Ponder, scry, surveil) are library → library `zone_move`s |
| `search` | searcher, zone searched (library, sideboard, graveyard, hand), its owner, cards found. The cards that leave follow as `zone_move`s, a shuffle as `shuffle` |
| `cast_from_hidden` / `play_from_hidden` | a `zone_move` (same fields) from a hidden zone to the stack or battlefield; the card becomes identifiable here. A reversed proposal (CR 733.1) returns it with a `zone_move` that keeps its identity |
| `face_down` / `face_up` | handle; identity at face-up. Exiling face down is a `zone_move` with the face-down flag; `face_down` is a public object turned face down where it is |
| `game_start` | game number, starting player; per player library and sideboard counts. Emitted after the libraries are built and shuffled (after sideboarding in games 2–3), before the opening draws. Opening hands and mulligans arrive as `draw`, `zone_move` (hand → library, London bottoming to the bottom) and `shuffle` |
| `game_end` / `match_end` | game number, winner (or draw); match wins per player |

**Identity references in events.**
- An event uses a handle when the viewer can identify the object at that instant.
- Otherwise it describes the object by zone and position: "an unknown card from B's hand",
  "the top card of B's library".
- When an identifiable object becomes unidentifiable, the event that hides it carries the
  handle, the card id and the destination, e.g. `zone_move h7 (Force of Will) → B library,
  position 0`. The handle then retires.
- A hidden-to-hidden move with no reveal never carries identity. An example is the opponent's
  Brainstorm put-back: two unknown cards from B's hand go to the top of B's library, order
  hidden. The opponent's hand is unordered for the viewer, so the event cannot say *which* hand
  cards went back.

## 6. Handles

- A handle is a per-viewer opaque id with no relation to entity ids. Entity ids are allocated
  in deck-list order (`src/systems/orderer.cpp:611-615`) and survive shuffles, so they would
  reveal identity.
- A handle exists **only while the viewer can individually identify the object**: it is in a
  public zone, in the viewer's own hand, or in one of the viewer's reveal windows.
- A public-to-public move keeps the old handle in the event (`h5 → graveyard as h9`), so it is
  linkable while still being a new object (CR 400.7).
- When an object stops being identifiable, the event that hides it carries the handle (section
  5) and the handle retires. If the object becomes identifiable again, it gets a fresh handle.
- New handles are issued in an order the viewer could derive from their own information:
  snapshot table order, then zone and position order. Never entity order.
- The opponent's face-down permanents are public-zone objects and have handles (CR 708.6: they
  must be distinguishable); only their identity is withheld.
- The handle table is per game, lives in `Game` (snapshot-covered), and resets at game start.

## 7. Actor side: the knowledge sidecar

Recall and inference are done by a C++ **knowledge sidecar**, designed in
[`knowledge_store.md`](knowledge_store.md):

- **Input:** only the viewer's own frame stream.
- **Output:** `k_*` knowledge tables attached to that viewer's frames, the same for every kind
  of agent.
- **Levels:** `off` (tables empty) and `expert` (perfect recall plus every sound logical
  deduction), selected by `--knowledge-a/-b off|expert`, default `expert`.
- **Checks:**
  - soundness against the omniscient truth;
  - exactness against brute-force enumeration;
  - isolation via the standalone `rmknow` tool;
  - a shadow comparison with the engine's current memory, in which every mismatch is
    classified as a bug or an old-vector leak.

## 8. Engine changes this requires

- **Event emission at the existing choke points:** `Orderer::add_to_zone`, `shuffle_library`,
  draw, library placement (top/bottom/position), the reveal/look/dig/peek/surveil/scry effects
  (`src/effects/effect_reveal*.cpp`, `effect_peek_and_reveal.cpp`, `effect_dig*.cpp`, …),
  face-down/face-up transitions, cast/play from hidden zones.
- **Typed event constructors** that take already-filtered data: a constructor receives the
  viewer and the visibility rules' answer, never a raw entity.
- **Reveal windows:** generalise `Game::revealed_in_library` into the per-viewer window set, and
  the snapshot's `reveal` table reads it.
- **Delete in the plan's step 3b**, after the sidecar's shadow comparison is fully classified,
  the persistent knowledge bookkeeping that existed only for the observation:
  `Game::KnownLibraryTop`, `Zone::identity_known`, the match revealed bits, and the
  `opp_known_hand` pass. The search server's determinizer also reads `identity_known`; it is
  deleted in the same step.
- **Menu collapse.** `interchangeable_cards` (`src/queries/zones.cpp:123-128`) refuses to merge
  two identical library cards when one is at a position the other player knows. That rule must
  be redefined on currently explicit information (a card in a live reveal window is
  distinguishable; memory is not). It changes the decision schedule, so it belongs with step 4a.

## 9. Encoding and protocol

- **Relational tables.** A frame is a sequence of tables; each table is fixed-width
  little-endian rows. Variable-length data goes in child tables keyed by handle, or by the
  parent's row index where the parent has no handle (events). Text is an offset into a
  per-frame string pool. C++ reads rows directly; Python reads them with
  `struct.iter_unpack`; arrays can later be built from them directly. Layout and columns:
  Appendix A.
- **Schema frame.** Sent once at startup, and available via `--print-schema`. It lists:
  - tables;
  - columns: name, storage type, width, and a **semantic tag**: `enum:<table>`, `count`,
    `bool`, `signed`, `handle`, `card`, `player-rel`, `text`, `index`, `row:<table>`,
    `mask:<enum>` (Appendix A.1), plus a value range where one is defined;
  - string tables for every enum: action categories, keywords, steps, zones, card types,
    event kinds. Counter kinds have no engine enum (they are script strings), so they travel
    as `text`.

  Decoders and feature builders build themselves from it, so a generic featurizer knows what
  to embed, what to scale and what is a relation, with no per-column code. They copy no
  constants, and there is no codegen.
- **Framing on `--machine`.** `RMSTATE <nbytes>\n` followed by the payload on stdout. The reply
  is one line: an action index, `-1` (confirm), `-2` (concede game) or `-3` (concede match).
  Narrative goes to stderr or its own file descriptor, never interleaved with frames.
- **Versioning.** Every frame header carries the format version and schema hash; a decoder
  refuses a mismatch.
- **Cadence.** A frame at each of the viewer's decisions. Pushed passive frames (a spectator or
  GUI stream) are deferred; batching events into the next frame is enough for agents.
- **Debug view.** `--state-dump json`, rendered from the decoded binary (not a second encoder).

## 10. Verification

- **roundtrip:** encode → decode → re-encode is byte-identical, at every decision.
- **snapshot leak:** because the snapshot keeps no memory, the test may permute *everything*
  hidden from the viewer at that instant. That means the opponent's hand and library contents
  together, and both libraries' order, keeping only the viewer's live windows fixed. The
  viewer's snapshot bytes must be identical.
- **event leak:** at each event, a test hook rebuilds its bytes in a world with the viewer's
  hidden information permuted (same rule); they must be identical. The typed constructors make
  this a structural guarantee; the hook checks it.
- **complete:** every public object appears, counts match zone sizes, every handle reference
  resolves, every menu handle exists, and the knowledge sidecar's pool sizes match the snapshot.
- **snapshot:** save → K actions → restore; both viewers' frames match the pre-save bytes.
- **knowledge:** the sidecar's soundness, exactness, isolation and shadow checks
  (`knowledge_store.md` §7).

## 11. Decision register

| # | Decision | Status |
|---|---|---|
| 1 | Post-layer results are authoritative, plus coarse per-object provenance. No dump of a continuous-effect registry: only layer 7 has a `ContinuousEffect` record today (`src/systems/continuous_effects.h`); the other layers are applied directly in `state_manager_statics.cpp` | decided |
| 2 | Abilities by reference: card id + script ability index (keyed definitions by their key string), a short header (category, costs, timing, limit state), script line text. Definition pointers and intern order are not stable across processes | decided |
| 3 | Relational fixed-width tables (section 9) | decided |
| 4 | Schema frame with enum string tables; no copied constants, no codegen | decided |
| 5 | Full snapshot per decision **plus the observation event stream** (section 5) | decided |
| 6 | Viewers A / B / omniscient (flag only) | decided |
| 7 | Handles exist only while the viewer can individually identify the object (section 6) | decided |
| 8 | The engine models **no** knowledge or memory; recall and inference are actor-side (section 7) | decided |
| 9 | Own library: count only; contents and order are actor-side | decided |
| 10 | Opponent hand: count plus live reveal windows | decided |
| 11 | Decklists are open by default: both registered lists (main + side) are explicit, sent in `DECKLISTS`. `--closed-decklist` turns this off: the viewer then gets only their own list, and the leak tests treat the opponent's list as hidden. The setting is recorded in the RMLOG header | decided |
| 11a | Name-a-card candidate set follows decision 11. With open decklists (default) it stays as today: the distinct vocab cards of the relevant registered list, filtered by `ValidCards$`. The list is explicit information, so this no longer leaks, but it remains the documented CR 201.4 deviation. Under `--closed-decklist`: every vocab card passing `ValidCards$` | decided |
| 12 | Descriptions included and covered by the leak tests; the full-information narrative never goes to a seat | decided |
| 13 | `CARDS` table once per match, replacing `card_props` / `card_costs` / script reading for the agent; card id stays the vocab index | decided |
| 14 | Full keyword lists with parameters; counters as kind → count | decided |
| 15 | Rules-derived convenience fields only where the engine already computes them (can attack now, effective sickness, land drops left, mana potential) | decided |
| 16 | Pending context = `PendingQuery::Tag`, deciding seat, source, menu category | decided |
| 17 | One integer per decision; keep the interchangeable-duplicate collapse until step 4a | decided |
| 18 | Rules-state tables (delayed triggers, emblems, player effects, play permissions, LKI, shields); the rest through provenance | decided |
| 19 | `RMSTATE <nbytes>\n` framing; narrative off stdout; version + hash in every frame | decided |
| 20 | Frames at the viewer's decisions only; push streams deferred | decided |
| 21 | Menu collapse redefined on currently explicit information (section 8), in step 4a | decided |
| 22 | Future search agents install a hidden world sampled from the actor's beliefs (a later engine command); the engine no longer determinizes from its own knowledge | deferred |
| 23 | `DECKLISTS` carries each registered list's deck name (the opponent's withheld under `--closed-decklist`); from the agent read map | decided |
| 24 | `object.is_token`, and token entries in `CARDS` with their real types; from the agent read map | decided |
| 25 | Ability references carry a keyword key for keyword-built abilities (`Ninjutsu`, `Equip`, `Reconfigure:attach`, …); from the agent read map | decided |
| 26 | Recall and inference live in a C++ sidecar fed only by the viewer's frame stream (`knowledge_store.md`), not in the engine and not in Python | decided |
| 27 | Knowledge levels `off` and `expert` only; default `expert`; per seat (`--knowledge-a/-b`); no `engine-compat` level (the agent port reads `expert`) | decided |
| 28 | Knowledge tables (`k_*`) travel inside the viewer's RMSTATE frame, with a schema fixed across levels | decided |
| 29 | `CARDS` always carries oracle text, for cards and tokens; text embeddings for learned models are built from it | decided |
| 30 | Every schema column carries a semantic tag and, where defined, a value range (section 9) | decided |
| 31 | Every row with an owner/controller/player column also carries it relative to the viewer (self/opponent) | decided |
| 32 | `auto-mana` is on by default for every consumer; `--no-auto-mana` turns it off (per seat with `-a`/`-b` if needed); recorded in the RMLOG header. Lands with the other schedule changes in plan step 4a; until then the existing machine/interactive split stands | decided |
| 33 | Every cost-payment decision's frame carries `pending_cost`: what is being paid for and the remaining cost by pip type (section 4) | decided |
| 34 | Counter kinds travel as `text` (no engine enum; they are script strings) | decided |
| 35 | Face-down exiled cards follow CR 406.3: a per-player look window, opened for whoever looked at the card before it was exiled face down or is allowed to look, lasting until the card leaves exile | decided |
| 36 | Events carry draw provenance: `turn` on every event, an opening-hand flag on `draw`, a hand subset (`drawn_this_turn`) on a `zone_move` out of a hand whose instruction is restricted to cards drawn this turn (Sylvan Library), and a draw link on a `reveal_start` that shows the card just drawn (Miracle). Both restrictions are explicit to both players, so the sidecar may use them | decided |

## Appendix A: column definitions

The column list T3 codes in `src/state_format/schema.h`; the `SCHEMA` frame is generated from
that header. Anything changed during implementation is updated here (step-1 exit).

### A.1 Conventions

| Item | Rule |
|---|---|
| Storage | little-endian `u8 u16 u32 u64 i8 i16 i32 f32`; rows packed, no padding; `text` = u32 offset into the frame's string pool |
| String pool | table `strings` (row width 1, one row per byte): UTF-8, each string NUL-terminated; offset 0 is the empty string (the pool starts with a NUL). Each distinct string is stored once, at its first use in table, row, column order, so re-encoding is byte-identical |
| None values | handle 0; text 0 (empty); card, `index`, `row:` and nullable enum columns: all ones of their width (0xFF, 0xFFFF, …) |
| Enums | the C++ value (`Step`, `ActionCategory`, `Zone::ZoneValue`, …), string tables in `SCHEMA`; enums that have no C++ type yet are listed in A.7 |
| Seat pair | type `seat` below = two u8 columns: `<name>` (`enum:seat` = `Zone::Ownership`: 0 none, 1 A, 2 B) and `<name>_rel` (`player-rel`: 0 none, 1 self, 2 opponent; for `omniscient`, self = A). Register 31 |
| Library position | type `libpos` = `<name>_anchor` u8 `enum:lib_anchor` (top, bottom) + `<name>_pos` u16 `index` (0 = the end card) + `<name>_span` u16 `count` (1 exact, n = one of n from pos, 0 unknown); section 5 |
| Ability reference | type `abref` = `ab_card` u16 `card` + `ab_index` u16 `index` (row of that card in `card_ability`) + `ab_key` `text` (keyword or keyed-definition key: `Ninjutsu`, `Equip`, `Reconfigure:attach`; empty for a plain script ability). Registers 2, 25 |
| Parent links | a child row names its parent by handle when the parent has one (`object`, `stack`), else by the parent's row index in the same frame (`row:<table>`) |
| Masks | bit i = value i of the enum (`mask:color` over `Colors`, `mask:card_type` over the engine's type list, …) |

**New semantic tags** (beyond section 9's list):
- `index`: a 0-based position or ordinal (library or stack position, ability index, mode), not
  a quantity;
- `row:<table>`: a row index into another table of the same frame;
- `mask:<enum>`: a bitset over an enum.

No `player-abs` or `zone` tag is needed: absolute seats are `enum:seat`, zones `enum:zone`.

### A.2 Frame layout

| Part | Layout |
|---|---|
| Framing | `RMSTATE <nbytes>\n`, then `nbytes` of payload (section 9) |
| Frame header (24 B) | `magic` u32 (`RMSF`), `version` u16, `kind` u8 (`enum:frame_kind`), `viewer` u8 (`enum:viewer`), `schema_hash` u64, `n_tables` u32, reserved u32 (0) |
| Table block | header (16 B): `table_id` u16, reserved u16, `row_width` u32, `row_count` u32, reserved u32; then `row_count × row_width` bytes, zero-padded to a multiple of 8 |
| Order | blocks in ascending table id (so `strings`, id 1, comes first); a table with no rows is omitted |
| Unknown data | a decoder skips a block whose id it does not know; a known id with another row width is a schema mismatch and is refused |
| Schema hash | 64-bit FNV-1a over the `SCHEMA` frame's table blocks (header excluded) |
| Frame kinds | `SCHEMA` (ids 2–4), `CARDS` (10–14), `DECKLISTS` (15–16), `STATE` (snapshot 20–48, events 80–82, knowledge 90–95) |

**Events channel.** One `event` table, one row per event in queue order, with a kind enum and
the scalar fields of every kind (unused ones are none/0). Per-card parts go in `event_card`,
per-player parts in `event_player`; both point at their `event` row.

### A.3 `SCHEMA` frame

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `schema_table` (2) | `id` | u16 | `index` | table id |
| | `name` | text | `text` | |
| | `frame_kind` | u8 | `enum:frame_kind` | |
| | `row_width` | u32 | `count` | bytes |
| `schema_column` (3) | `table` | u16 | `index` | table id |
| | `col` | u16 | `index` | column order |
| | `name` | text | `text` | |
| | `storage` | u8 | `enum:storage` | u8 … f32, text |
| | `offset` | u16 | `count` | byte offset in the row |
| | `tag` | text | `text` | e.g. `enum:zone`, `row:event` |
| | `has_range`, `min`, `max` | u8, i64, i64 | `bool`, `signed`, `signed` | |
| `schema_enum` (4) | `enum` | text | `text` | enum name (`zone`, `step`, …) |
| | `value` | u16 | `index` | |
| | `name` | text | `text` | |

### A.4 `CARDS` and `DECKLISTS` frames

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `card` (10): one row per face | `card` | u16 | `card` | vocab id (tokens in the token band) |
| | `face` | u8 | `enum:face` | front, back |
| | `name`, `uid` | text | `text` | |
| | `token` | u8 | `bool` | register 24 |
| | `layout` | u8 | `enum:layout` | normal, transform, modal DFC, split |
| | `mana_cost` | text | `text` | printed, e.g. `{2}{U}` |
| | `mana_value` | u8 | `count` | |
| | `cost_w` … `cost_c`, `cost_generic`, `cost_x` | 8 × u8 | `count` | plain pips; X pips |
| | `colors` | u8 | `mask:color` | |
| | `types` | u32 | `mask:card_type` | |
| | `supertypes` | u8 | `mask:supertype` | |
| | `power`, `toughness` | i16 | `signed` | printed |
| | `loyalty` | u8 | `count` | |
| | `oracle` | text | `text` | `CardData::oracle_text` (register 29) |
| | `script` | text | `text` | raw script; empty unless requested |
| `card_subtype` (11) | `card`, `face`, `subtype` | u16, u8, u16 | `card`, `enum:face`, `enum:subtype` | |
| `card_keyword` (12) | `card`, `face` | u16, u8 | `card`, `enum:face` | |
| | `kw` | u8 | `enum:keyword` | 0xFF = not in the parser's keyword table |
| | `text` | text | `text` | the keyword as held, e.g. `Ward:2` |
| `card_ability` (13) | `card`, `index` | u16, u16 | `card`, `index` | index: per card, script order over both faces (`abref.ab_index`) |
| | `face` | u8 | `enum:face` | |
| | `ability_kind` | u8 | `enum:ability_kind` | spell, activated, triggered, static, replacement, chapter, opening hand |
| | `category` | u8 | `enum:effect_kind` | `effect_kinds.def`; 0xFF for statics and replacements |
| | `key` | text | `text` | keyword key / keyed definition (register 25) |
| | `cost` | text | `text` | cost as printed |
| | `tap` | u8 | `bool` | `{T}` in the cost |
| | `timing` | u8 | `enum:timing` | any, sorcery speed, instant speed |
| | `limit` | u8 | `count` | activations per turn; 0 = none |
| | `line` | text | `text` | script line |
| `card_pip` (14) | `card`, `face` | u16, u8 | `card`, `enum:face` | hybrid and Phyrexian pips of the printed cost |
| | `kind` | u8 | `enum:pip_kind` | hybrid, twobrid, Phyrexian |
| | `colors` | u8 | `mask:color` | |
| | `generic_alt` | u8 | `count` | twobrid's 2 |
| `decklist` (15) | `player` | seat | | only the viewer's under `--closed-decklist` |
| | `name` | text | `text` | deck name (register 23) |
| | `main`, `side` | u16 | `count` | |
| `decklist_card` (16) | `player` | seat | | |
| | `card` | u16 | `card` | |
| | `main`, `side` | u8 | `count` | registered counts |

### A.5 Snapshot (`STATE` frame)

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `header` (20) | `game` | u8 | `count` | 0-based game of the match; version, schema hash and viewer are in the frame header |
| | `decision` | u32 | `count` | decision number in the game |
| | `knowledge` | u8 | `enum:knowledge_level` | this viewer's level |
| `turn` (21) | `turn` | u16 | `count` | |
| | `step` | u8 | `enum:step` | |
| | `active`, `priority` | seat | | |
| | `priority_window` | u8 | `bool` | an ordinary priority window |
| | `mandatory` | u8 | `enum:mandatory_choice` | |
| | `attackers_declared`, `blockers_declared` | u8 | `bool` | |
| | `day_night` | u8 | `enum:day_night` | |
| | `next_extra_turn` | seat | | none = no extra turn queued |
| `player` (22): self row first | `player` | seat | | |
| | `life` | i32 | `signed` | |
| | `mana_w` … `mana_c` | 6 × u16 | `count` | floating pool |
| | `hand`, `library`, `graveyard`, `exile`, `sideboard` | 5 × u16 | `count` | zone sizes |
| | `mulligans`, `to_bottom` | u8 | `count` | London bottoming still owed |
| | `passed` | u8 | `bool` | pass flag |
| | `monarch`, `city_blessing` | u8 | `bool` | |
| | `extra_turns` | u8 | `count` | queued |
| | `lands_played`, `land_drops_left` | u8 | `count` | register 15 |
| | `potential_w` … `potential_c`, `potential_total` | 7 × u8 | `count` | mana potential (register 15) |
| | `spells_turn`, `noncreature_turn`, `instant_sorcery_turn`, `spells_game`, `drawn_turn` | 5 × u16 | `count` | per-turn / per-game counters |
| | `life_gained_turn`, `life_lost_turn` | u16 | `count` | |
| | `colors_cast_turn` | u8 | `mask:color` | |
| | `revolt`, `drew_from_empty`, `companion_used` | u8 | `bool` | |
| | `match_wins` | u8 | `count` | |
| `player_counter` (23) | `player` | seat | | |
| | `kind` | text | `text` | `POISON`, `ENERGY`, … |
| | `count` | u16 | `count` | |
| `object` (24) | `handle` | u32 | `handle` | |
| | `zone` | u8 | `enum:zone` | |
| | `owner`, `controller` | seat | | |
| | `pos` | u16 | `index` | stack / graveyard / exile from the top (0 = newest); library position for a card in a window |
| | `card` | u16 | `card` | 0xFFFF = face down to this viewer |
| | `token`, `face_down`, `transformed` | u8 | `bool` | register 24 |
| | `copy_of` | u16 | `card` | card a copy effect copies |
| | `tapped`, `sick`, `phased_out`, `can_attack` | u8 | `bool` | engine flags (register 15) |
| | `attacking`, `blocking`, `blocked` | u8 | `bool` | details in `combat` |
| | `damage` | u16 | `count` | |
| | `deathtouch_damage` | u8 | `bool` | |
| | `power`, `toughness` | i16 | `signed` | post-layer |
| | `types` | u32 | `mask:card_type` | post-layer |
| | `supertypes` | u8 | `mask:supertype` | |
| | `colors` | u8 | `mask:color` | `effective_colors` |
| | `mana_value` | u8 | `count` | |
| | `attached_to` | u32 | `handle` | |
| | `timestamp` | u32 | `index` | entered-battlefield order |
| | `entered_turn`, `cant_be_blocked`, `damage_shielded`, `delayed_subject` | u8 | `bool` | as BQUERY's per-turn statuses |
| | `activations_turn`, `resolutions_turn` | u8 | `count` | |
| | `chosen_name` | u16 | `card` | |
| | `chosen_type` | u16 | `enum:subtype` | 0xFFFF none |
| | `chosen_color` | u8 | `enum:color` | |
| | `entered_x` | u16 | `count` | |
| | `monstrous`, `evoked`, `escaped`, `unearthed`, `cast_from_hand` | u8 | `bool` | |
| `object_subtype` (25) | `obj`, `subtype` | u32, u16 | `handle`, `enum:subtype` | post-layer |
| `object_keyword` (26) | `obj` | u32 | `handle` | register 14 |
| | `kw` | u8 | `enum:keyword` | as `card_keyword` |
| | `text` | text | `text` | |
| `object_counter` (27) | `obj` | u32 | `handle` | permanent counters, or a card's in exile |
| | `kind` | text | `text` | `P1P1`, `LOYALTY`, `TIME`, … |
| | `count` | u16 | `count` | |
| `object_ability` (28) | `obj` | u32 | `handle` | |
| | ability | abref | | |
| | `granted` | u8 | `bool` | not printed on the object |
| | `activations_turn` | u8 | `count` | limit state |
| | `exhausted` | u8 | `bool` | its limit is reached this turn |
| `object_provenance` (29) | `obj`, `src` | u32 | `handle` | register 1 |
| | `src_card` | u16 | `card` | when `src` is gone |
| | `layer` | u8 | `enum:layer` | CR 613 layer / sublayer |
| | `duration` | u8 | `enum:duration` | |
| `object_link` (30) | `obj`, `linked` | u32 | `handle` | linked exiled cards |
| | `kind` | u8 | `enum:link_kind` | exiled with, returnable |
| `combat` (31) | `attacker` | u32 | `handle` | |
| | `target_kind` | u8 | `enum:attack_target_kind` | player, planeswalker |
| | `target_player` | seat | | |
| | `target` | u32 | `handle` | planeswalker |
| | `blocker` | u32 | `handle` | 0 = unblocked row |
| | `order` | u8 | `index` | damage-assignment order |
| | `assigned` | u16 | `count` | damage assigned so far |
| `stack` (32) | `handle` | u32 | `handle` | |
| | `pos` | u16 | `index` | 0 = top |
| | `kind` | u8 | `enum:stack_kind` | spell, activated, triggered, copy |
| | `controller` | seat | | |
| | `source` | u32 | `handle` | an ability's source; 0 if gone |
| | `source_card` | u16 | `card` | |
| | ability | abref | | |
| | `x` | i16 | `signed` | −1 = no X |
| | `amount` | u16 | `count` | `Ability::amount` |
| | `kicked`, `replicate` | u8 | `count` | |
| | `variant` | u16 | `mask:cast_variant` | flashback, evoke, escape, offspring, impending, warp, back face, alt cost, impulse |
| | `copy`, `cant_be_countered`, `gift_promised` | u8 | `bool` | |
| | `mana_spent` | u8 | `count` | |
| | `colors_spent` | u8 | `mask:color` | |
| | `delayed` | u32 | `index` | `delayed_trigger.seq`; 0 none |
| `stack_mode` (33) | `stack`, `mode` | u32, u8 | `handle`, `index` | chosen modes |
| `stack_target` (34) | `stack` | u32 | `handle` | |
| | `order` | u8 | `index` | announcement order |
| | `kind` | u8 | `enum:target_kind` | object, player |
| | `target` | u32 | `handle` | |
| | `target_player` | seat | | |
| | `lki` | u16 | `row:lki` | the target left its zone |
| `delayed_trigger` (35) | `seq` | u32 | `index` | registration order |
| | `controller` | seat | | |
| | `state` | u8 | `enum:delayed_state` | waiting, on the stack |
| | `stack`, `creator` | u32 | `handle` | |
| | `creator_card` | u16 | `card` | |
| | `fire` | u8 | `enum:fire_kind` | `DelayedTriggerLink::FireKind` |
| | `fire_turn` | u16 | `count` | |
| | `fires_this_turn`, `expires_eot` | u8 | `bool` | |
| | ability | abref | | |
| `delayed_subject` (36) | `trigger` | u16 | `row:delayed_trigger` | |
| | `subject` | u32 | `handle` | |
| | `subject_card` | u16 | `card` | |
| | `lki` | u16 | `row:lki` | |
| `emblem` (37) | `controller` | seat | | |
| | `source_card` | u16 | `card` | |
| `player_effect` (38) | `player` | seat | | |
| | `kind` | u8 | `enum:player_effect_kind` | as BQUERY's player-effect block |
| | `colors` | u8 | `mask:color` | hexproof-from colours |
| | `duration` | u8 | `enum:duration` | |
| | `source_card` | u16 | `card` | |
| | `filter` | text | `text` | e.g. a flash permission's filter |
| `play_permission` (39) | `obj` | u32 | `handle` | |
| | `player` | seat | | who may play it |
| | `kind` | u8 | `enum:permission_kind` | flashback, escape, cast this turn, impulse, warp, suspend, land from graveyard |
| | `resource` | u8 | `enum:permission_resource` | mana, energy, life, free |
| | `amount` | u16 | `count` | |
| | `expires_turn` | u8 | `bool` | lapses at this turn's cleanup |
| `shield` (40) | `kind` | u8 | `enum:shield_kind` | combat-damage prevention |
| | `obj` | u32 | `handle` | |
| | `as_source`, `as_target` | u8 | `bool` | |
| `lki` (41) | `card` | u16 | `card` | |
| | `token` | u8 | `bool` | |
| | `controller` | seat | | |
| | `power`, `toughness` | i16 | `signed` | |
| | `types` | u32 | `mask:card_type` | |
| | `colors` | u8 | `mask:color` | |
| | `mana_value` | u8 | `count` | |
| `reveal` (42) | `handle` | u32 | `handle` | |
| | `zone` | u8 | `enum:zone` | |
| | `owner` | seat | | |
| | `pos` | u16 | `index` | library position from the top |
| | `card` | u16 | `card` | |
| | `window` | u8 | `enum:window_kind` | reveal, look |
| | `to` | seat | | none = all players |
| `pending` (43) | `tag` | u8 | `enum:pending_tag` | `PendingQuery::Tag` |
| | `chooser` | seat | | |
| | `source` | u32 | `handle` | |
| | `source_card` | u16 | `card` | |
| | `source_controller` | seat | | |
| `pending_cost` (44) | `for` | u32 | `handle` | spell or ability paid for |
| | `for_card` | u16 | `card` | |
| | `part` | u8 | `enum:cost_part` | mana, alternative, additional, unless, X, activation |
| | `rem_w` … `rem_c`, `rem_generic`, `rem_x` | 8 × u8 | `count` | remaining plain pips |
| | `rem_life` | u16 | `count` | |
| | `paid` | u8 | `count` | mana already paid |
| `pending_cost_pip` (45) | `cost` | u16 | `row:pending_cost` | remaining non-plain pips |
| | `kind`, `colors`, `generic_alt` | u8 | `enum:pip_kind`, `mask:color`, `count` | as `card_pip` (plus snow) |
| `menu` (46): row = action index | `category` | u8 | `enum:action_category` | |
| | `type` | u8 | `enum:action_type` | |
| | `source`, `target` | u32 | `handle` | |
| | `source_card` | u16 | `card` | when `source` has no handle |
| | `target_player` | seat | | |
| | ability | abref | | |
| | `ordinal` | i16 | `signed` | `option_ordinal`; −1 = n/a |
| | `variant` | u16 | `mask:cast_variant` | plus companion, suspend, cancel |
| | `card_public` | u8 | `bool` | |
| | `description` | text | `text` | covered by the leak checks (register 12) |
| `match` (47) | `game` | u8 | `count` | |
| | `plays_first` | seat | | this game, or the upcoming one while sideboarding |
| | `sideboard_phase` | u8 | `bool` | |
| | `sb_swaps` | u8 | `count` | |
| | `sb_drift` | i8 | `signed` | |
| | `closed_decklist` | u8 | `bool` | |
| `own_deck` (48) | `card` | u16 | `card` | the viewer's current 75 |
| | `main`, `side` | u8 | `count` | |

### A.6 Events and knowledge (`STATE` frame)

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `event` (80) | `seq` | u32 | `index` | section 5 |
| | `kind` | u8 | `enum:event_kind` | |
| | `player` | seat | | drawer, looker, searcher, shuffler, starting player, winner (none = draw) |
| | `zone`, `to_zone` | u8 | `enum:zone` | from / looked / searched zone; destination. 0xFF = `none` (created, ceased) |
| | `zone_owner`, `to_owner` | seat | | |
| | `count` | u16 | `count` | cards drawn / looked at / found; batch size |
| | `range` | libpos | | look, `library_place`, `reorder` |
| | `order_known` | u8 | `bool` | `library_place`, `reorder` |
| | `batch` | u16 | `index` | 0 = none |
| | `to` | seat | | audience of a reveal or look; none = all |
| | `source` | u32 | `handle` | |
| | `source_card` | u16 | `card` | |
| | `turn` | u16 | `count` | game turn of the event |
| | `opening` | u8 | `bool` | `draw`: opening hand or mulligan draw |
| | `game` | u8 | `count` | game events |
| `event_card` (81) | `event` | u32 | `row:event` | |
| | `handle_before`, `handle_after` | u32 | `handle` | |
| | `card` | u16 | `card` | |
| | `from` | libpos | | from a library |
| | `to` | libpos | | into a library; `to_pos` 0xFFFF = in a batch |
| | `face_down`, `token` | u8 | `bool` | token = token or spell copy |
| | `hand_subset` | u8 | `enum:hand_subset` | `zone_move` from a hand: any, drawn_this_turn |
| | `draw_seq` | u32 | `index` | `reveal_start`: `seq` of the `draw` that brought this card; 0 = none |
| | `draw_idx` | u16 | `index` | its index among that draw's cards |
| `event_player` (82) | `event` | u32 | `row:event` | `game_start`, `game_end`, `match_end` |
| | `player` | seat | | |
| | `library`, `sideboard` | u16 | `count` | `game_start` |
| | `wins` | u8 | `count` | match wins |
| `k_meta` (90) | `level` | u8 | `enum:knowledge_level` | off, expert |
| | `exact` | u8 | `bool` | |
| | `steps` | u32 | `count` | solver steps used |
| `k_seen` (91) | `player` | seat | | |
| | `card` | u16 | `card` | |
| | `game`, `match` | u8 | `count` | copies seen |
| `k_pool` (92): row = pool id | `owner` | seat | | |
| | `kind` | u8 | `enum:pool_kind` | hand, library range, sideboard, face-down, set aside |
| | `size` | u16 | `count` | |
| | `range` | libpos | | library range |
| | `obj` | u32 | `handle` | face-down object |
| `k_pool_card` (93) | `pool` | u16 | `row:k_pool` | rows with max > 0 |
| | `card` | u16 | `card` | |
| | `min`, `max` | u8 | `count` | |
| `k_position` (94) | `owner` | seat | | |
| | `at` | libpos | | span 1 |
| | `card` | u16 | `card` | 0xFFFF = not exact |
| | `pool` | u16 | `row:k_pool` | candidates when not exact |
| `k_known_hand` (95) | `card` | u16 | `card` | |
| | `copies` | u8 | `count` | |

### A.7 Enums with no C++ type yet

T3/T4 add these as C++ enums so their string tables are generated like the others.

| Enum | Values |
|---|---|
| `frame_kind` | SCHEMA, CARDS, DECKLISTS, STATE |
| `viewer` | A, B, OMNISCIENT |
| `storage` | U8, U16, U32, U64, I8, I16, I32, I64, F32, TEXT |
| `hand_subset` | ANY, DRAWN_THIS_TURN |
| `event_kind` | ZONE_MOVE, DRAW, LIBRARY_PLACE, REVEAL_START, REVEAL_END, LOOK, SHUFFLE, REORDER, SEARCH, CAST_FROM_HIDDEN, PLAY_FROM_HIDDEN, FACE_DOWN, FACE_UP, GAME_START, GAME_END, MATCH_END |
| `lib_anchor` | TOP, BOTTOM |
| `face` | FRONT, BACK |
| `layout` | NORMAL, TRANSFORM, MODAL_DFC, SPLIT |
| `ability_kind` | SPELL, ACTIVATED, TRIGGERED, STATIC, REPLACEMENT, CHAPTER, OPENING_HAND |
| `timing` | ANY, SORCERY, INSTANT |
| `pip_kind` | HYBRID, TWOBRID, PHYREXIAN, SNOW |
| `stack_kind` | SPELL, ACTIVATED, TRIGGERED, COPY |
| `target_kind` | OBJECT, PLAYER |
| `attack_target_kind` | PLAYER, PLANESWALKER |
| `delayed_state` | WAITING, ON_STACK |
| `window_kind` | REVEAL, LOOK |
| `cost_part` | MANA, ALTERNATIVE, ADDITIONAL, UNLESS, X, ACTIVATION |
| `layer` | COPY, CONTROL, TEXT, TYPE, COLOR, ABILITY, PT_CDA, PT_SET, PT_MODIFY, PT_COUNTERS, PT_SWITCH |
| `duration` | STATIC, END_OF_TURN, UNTIL_YOUR_NEXT_TURN, UNTIL_HOST_LEAVES, PERMANENT |
| `link_kind` | EXILED_WITH, RETURNABLE |
| `player_effect_kind`, `permission_kind`, `permission_resource`, `shield_kind`, `cast_variant` | the lists in the table notes above |
| `knowledge_level` | OFF, EXPERT |
| `pool_kind` | HAND, LIBRARY_RANGE, SIDEBOARD, FACE_DOWN, SET_ASIDE |

Existing C++ enums used: `Step` (`step`), `ActionCategory` (`action_category`), `ActionType`
(`action_type`), `Zone::ZoneValue` (`zone`), `Zone::Ownership` (`seat`), `Colors` (`color`),
`MandatoryChoice` (`mandatory_choice`), `Game::DayNight` (`day_night`), `PendingQuery::Tag`
(`pending_tag`), `DelayedTriggerLink::FireKind` (`fire_kind`), `EffectKind` (`effect_kind`,
from `effect_kinds.def`). The type lists `all_types`, `all_subtypes`, `all_supertypes`
(`src/constants.cpp`) give `card_type`, `subtype`, `supertype`; the parser's keyword table
(`kKeywordTable`, `src/parse.cpp`) gives `keyword`.
