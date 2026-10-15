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
| Exile, face down | identity only while an instruction lets them look | same | CR 406.3 |
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
columns are fixed in the schema (section 9).

| Table | Rows | Notes |
|---|---|---|
| `header` | 1 | format version, schema hash, viewer, game number, decision number |
| `turn` | 1 | turn, step, active player, priority holder, pass flags, extra turns queued, land drops remaining |
| `player` | 2 | life, poison, energy, mana pool, designations (monarch, city's blessing), day/night, per-turn counters, hand count, library count, mulligans taken |
| `object` | one per identifiable object | handle, zone, owner, controller, card id (or "face-down"), face/transformed state, tapped/sick/phased, damage, base and post-layer characteristics, attached-to, chosen name/colour/type, linked exiled cards |
| `object_keyword`, `object_counter`, `object_ability`, `object_provenance` | children of `object` | every keyword with its parameters; counters by kind; abilities by reference (decision 2); which sources currently affect the object, what kind of effect, and when it expires (decision 1) |
| `combat` | one per attacker/blocker pairing | attack target, blockers, damage-assignment order |
| `stack` | one per stack object | kind (spell / activated / triggered / copy), source handle, controller, ability reference, modes, X, kicked/alternative costs, plus child table `stack_target` |
| `delayed_trigger`, `emblem`, `player_effect`, `play_permission`, `shield` | as present | full lists, no slot caps |
| `lki` | departed objects still referenced | last-known characteristics for a stack or effect reference |
| `reveal` | one per object in a current window for this viewer | handle, zone, position if in a library, card id, who it is revealed to |
| `pending` | 0 or 1 | the paused flow (`PendingQuery::Tag`), deciding seat, source handle |
| `menu` | one per legal action | category, action type, source/target handles, ability reference, ordinal, cast-variant flags, card-is-public flag, description. No length cap; one integer answers the decision |
| `match` | 1 | game number, wins, who played first this game, sideboard phase, own current deck/sideboard during sideboarding |
| `events` | the drained event queue | section 5 |

Sent once per match (not per snapshot):

- `CARDS`: the card table for every implemented card and token in either registered list, plus
  tokens they can make. Each entry has the vocab id, name, uid, printed characteristics, an
  ability list (category, costs, timing, script line text) and, optionally, the raw script.
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

Each event carries **only what was explicit to that viewer at that instant**, projected through
the same visibility rules as the snapshot.

| Event | Fields (viewer's projection) |
|---|---|
| `zone_move` | from zone/owner, to zone/owner, handle before/after if identifiable, card id if identifiable, position if into a library (top / bottom / k from top) |
| `draw` | player, count; card ids for the drawer only |
| `library_place` | library owner, positions, identities only for cards visible at that moment, whether the order among them is known to this viewer |
| `reveal_start` / `reveal_end` | cards (handles, zone, positions), revealed to whom, source effect |
| `look` | looker, zone, positions or count; identities only for the looker |
| `shuffle` | library (or face-down pile) owner |
| `reorder` | library owner, the range of positions reordered, whether this viewer saw the result |
| `search` | searcher, library owner; cards leaving with identities only if revealed or going to a public zone |
| `cast_from_hidden` / `play_from_hidden` | origin zone; the card becomes identifiable here (it enters a public zone) |
| `face_down` / `face_up` | handle; identity at face-up |
| `game_start` / `game_end` / `match_end` | game number, result |

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
  little-endian rows. Variable-length data goes in child tables keyed by handle. C++ reads rows
  directly; Python reads them with `struct.iter_unpack`; arrays can later be built from them
  directly.
- **Schema frame.** Sent once at startup, and available via `--print-schema`. It lists tables,
  columns (name, type, width), and string tables for every enum: action categories, keywords,
  counter kinds, steps, zones, card types, event kinds. Decoders build themselves from it and
  copy no constants. There is no codegen.
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
