# RMSTATE: the engine's state format

RMSTATE is the engine's only machine protocol (`--machine`) and the frames every decision consumer
reads: an in-process agent (`--agent-a/-b`, decoded in memory), the C++ CLI, or an external agent
over `--machine`. Recall and inference over it: [`knowledge_store.md`](knowledge_store.md).

| Concern | Code |
|---|---|
| visibility, reveal windows | `src/queries/visibility.*` |
| handles | `src/state_format/handles.*` |
| schema, enum tables, encoder | `src/state_format/schema.*`, `enum_names.*`, `designations.def`, `encode.*` |
| observation events | `src/state_format/events.*` |
| decoders, JSON | `src/state_format/decode.*`, `json.*`, `recode.*`; `python/rmstate.py` |
| frame writers | `src/state_format/dump.*`, `stream.*`, `agent_feed.*` |
| self-checks (§10) | `src/state_format/selfcheck.*`, `selfcheck_knowledge.*` |
| knowledge sidecar (§7) | `src/knowledge/`, `src/state_format/knowledge_feed.*` |

## 1. Principle: the engine conveys explicit information; the actor remembers and infers

The engine tells each viewer **only what that viewer explicitly knows at that instant**: the open
decklists, their own hand, the public zones, the counts of hidden zones, and cards revealed to or
looked at by the viewer right now. It does **not** remember on a viewer's behalf ("you saw Force of
Will in their hand last turn", "the card you put back is second from the top"). Perfect recall and
inference belong to the actor, through the C++ knowledge sidecar beside it (section 7).

Two channels per viewer:

1. **Snapshot**: the current explicit state, at each of the viewer's decisions (section 4).
2. **Observation events**: every piece of explicit information since the viewer's previous frame,
   in order, each as known at the instant it happened (section 5).

Events are needed because a snapshot cannot carry what was explicit only for a moment: a seat gets
no frame at a one-row decision (section 9), so a reveal may start and end between its frames; a
move into a hidden zone is visible only as it happens; and a shuffle invalidates positional
knowledge. Snapshot plus event history is everything the player saw; the snapshot and menu alone
suffice to play legally.

## 2. Viewers

| Viewer | Gets | Used by |
|---|---|---|
| `A`, `B` | that seat's explicit information (sections 3–5) | seat agents, the CLI human |
| `omniscient` | everything, unfiltered | explicit flag only: debugging, spectator, the reference side of the leak tests. Never sent to a seat |

Two-player only: "the opponent" is always the other seat.

## 3. Visibility rules (what is explicit right now)

| Zone / object | Viewer = owner/controller | Viewer = opponent | Basis |
|---|---|---|---|
| Battlefield, stack, graveyard | full | full | CR 400.2 |
| Exile, face up | full | full | CR 406.3 |
| Exile, face down | identity visible to each player in that card's **look window**: a player who looked at the card before it was exiled face down, or whom an instruction allows to look, until it leaves exile. Usually the owner | same rule, so usually hidden | CR 406.3 |
| Face-down permanent or spell | its card (the controller may look) and its face-down characteristics | its face-down characteristics, what effects give it and its public status, never its card; the object is still distinguishable | CR 708.2, 708.5, 708.6 |
| Hand | full | count, plus cards in a current reveal window | CR 402.3 |
| Library | count, plus cards in a current look/reveal window | count, plus cards in a current reveal window | CR 401.2, 401.3 |
| Sideboard (bo3) | own contents | count only | the registered list is open; the post-board configuration is hidden |
| Decklists | own registered list; the opponent's too unless `--closed-decklist` | same | register 11 |

**Reveal windows** (`src/queries/visibility.{h,cpp}`: `identity_visible`, `order_visible`,
`reveal_windows_for` over `Game::reveal_windows`; `--trace-visibility` narrates each opening and
closing). One per-viewer window set covers reveals and looks in libraries, hands, the sideboard and
face-down exile. The snapshot shows exactly the cards in the viewer's windows.

- A reveal shows a card to all players for as long as the revealing effect needs it (CR 701.20a);
  a look is the same, to the specified player only (CR 701.20e).
- A window ends when the effect finishes with the card (the end of the resolution that opened it;
  a mid-resolution decision is still inside it), when the card changes zones, or when its library
  is shuffled or reordered (CR 701.20d). A card put back by Ponder or a scry stops being looked at
  as it is put back.
- A search is a look at every card in the searched hidden zones (CR 701.23a). A search that picks
  one card at a time (Doomsday) looks again only at the cards it may still pick, so the other
  player can tell the next pick apart from those already put back (`src/zone_search.cpp`).
- A miracle card stays revealed until its trigger leaves the stack (CR 701.20a, 702.94b).
- Continuous reveals last as long as their effect; the only one today is the chosen companion,
  revealed outside the game while it stays there.
- **Standing look** (`STANDING_LOOK`): the window a `MayLookAt$` static gives its controller on a
  library's top card (Mystic Forge). Derived, not kept: `sync_standing_looks` aligns the windows
  with the statics at every decision, before its menu and frames, so the window is on the live top
  card. A card that stops being on top without moving (a put-back over it) loses it with a
  `reveal_end`; a moved card or a shuffle ends it as any window. No new top card gets a window
  while a spell is cast, an ability activated or a special action taken (CR 401.5), so a card
  under one cast off the top is seen only once the cast is done (never, for a reversed cast).
  Within a resolution it lags: a card on top only mid-resolution is not looked at. An effect's own
  look at the same card is a separate window, so a frame may list both.
- **Face-down exile** (`FACE_DOWN_LOOK`, `open_face_down_windows`): a card exiled face down gets a
  window for each player who could see its identity in the hidden zone just before (its owner's
  hand, or a look or reveal window). One exiled face down from a public zone gets none. The window
  ends when the card leaves exile or is turned face up.
- A face-down permanent or spell needs no window: its controller sees the card by rule, the other
  player doesn't, and a change of controller changes who sees it. It is turned up before it leaves
  for any zone but the battlefield (CR 708.9), so that `zone_move` shows the card to every viewer,
  a move to a hidden zone included. At game end every face-down permanent and spell left is
  revealed (`reveal_face_down_objects`: a reveal to all, before `game_end`).
- A card cloaked out of a library (CR 701.58a) opens no window (CR 406.3's look is a rule of exile;
  CR 708.5 forbids looking at another player's face-down permanent). Its controller sees it once it
  is on the battlefield. A seat that knew the top card keeps knowing it only by its own record: the
  other player's `object` row shows no card, and its sidecar moves the top-of-library unit to the
  face-down pool (`knowledge_store.md`). A window open at that moment names the card in the move's
  event for that seat and ends with the move (in practice a standing look, whose looker controls
  the permanent).

## 4. Snapshot

A full description at one decision (no deltas). Columns: schema (section 9) and Appendix A, which
also lists child tables.

| Table | Rows | Notes |
|---|---|---|
| `header` | 1 | format version, schema hash, viewer, game number, decision number |
| `turn` | 1 | turn, step, active player, priority holder, pass flags, extra turns queued, land drops remaining |
| `pool_mana` | one per color, restriction and rider of a player's floating mana that has one | what the `player` row's counts can't say; none for plain mana |
| `player` | 2 | life, poison, energy, mana pool, designations (monarch, city's blessing, speed), day/night, per-turn counters, hand count, library count, mulligans taken |
| `object` | one per identifiable object | handle, zone, owner and controller (absolute and relative to the viewer), card id (or face-down), face/transformed state, tapped/sick/phased, damage, base and post-layer characteristics, attached-to, chosen name/colour/type, linked exiled cards |
| `object_keyword`, `object_counter`, `object_ability`, `object_provenance` | children of `object` | keywords with parameters; counters by kind; abilities by reference (register 2); which sources affect the object, the kind of effect and its expiry (register 1) |
| `combat` | one per attacker/blocker pairing | attack target, blockers, damage assigned |
| `stack` | one per stack object | kind (spell / activated / triggered / copy), source handle, controller, ability reference, modes, X, kicked/alternative costs; child `stack_target` |
| `delayed_trigger`, `emblem`, `player_effect`, `play_permission`, `shield` | as present | full lists, no slot caps |
| `lki` | departed objects still referenced | last-known characteristics for a stack or effect reference |
| `reveal` | one per object in a current window for this viewer | handle, zone, library position, card id, audience |
| `pending` | 0 or 1 | the paused flow (`PendingQuery::Tag`), deciding seat, source handle |
| `pending_cost` | 0 or 1 during a cost-payment decision (manual mana payment under `--no-auto-mana`, pitch/alternative costs, unless-payments, X choices) | what is paid for (spell/ability and which cost: mana, alternative, unless, additional), the **remaining cost** by pip type (W/U/B/R/G/C, generic, hybrid, Phyrexian, snow, X) and the amount paid. In the interactive payment loop, mana floating in the payer's pool counts as paid and is left out of the remaining cost (the engine spends the pool once it covers the whole cost, CR 601.2g-h). Child `pending_cost_pip` for non-plain pips |
| `menu` | one per legal action | category, action type, source/target handles, ability reference, ordinal, cast-variant flags, card-is-public flag, description. No length cap; one integer answers the decision |
| `match` | 1 | game number, wins, who plays first, sideboard phase, own current deck/sideboard during sideboarding |
| `events` | the drained event queue | section 5 |

Sent once per match:

- `CARDS`: every implemented card and token in either registered list (the whole vocab under
  `--closed-decklist`), plus the tokens they can make: vocab id, name, uid, printed
  characteristics, **oracle text** (always; `CardData::oracle_text` from the script's `Oracle:`
  line, token scripts included), the ability list (category, costs, timing, script line, structured
  parameters) and, under `--rmstate-card-scripts`, the raw script.
- `DECKLISTS`: both registered lists with their deck names (only the viewer's own under
  `--closed-decklist`).

Not in the snapshot (actor-side, section 7): known library positions, cards seen earlier, the
opponent's hand beyond its count and current windows, and the contents of the viewer's own library.

## 5. Observation events

**Queueing.**
- Per viewer (A, B, omniscient) in `Game::obs_events`, so snapshots and RMLOG replay cover them. A
  frame drains its viewer's queue in order.
- Game start, game end and match end are events, so actors keep memory across bo3 games. The
  queues and `seq` counters carry over the per-game `Game` reset (a match's first game starts
  fresh): undelivered events reach the viewer's next frame (a sideboarding frame or the next game's
  first), after a game-boundary event. Nothing follows the final `game_end` / `match_end`, so they
  stay queued unless a writer emits a frame at match end (the terminal frame, section 9).
- `seq`: per viewer, 1 at match start, +1 per event, never reset within the match. A gap means a
  lost event; decoders refuse it.
- Events are recorded only while something consumes them (a frame writer, `--selfcheck-state`,
  `--trace-events`); play and the RNG are the same either way.
- Each event carries **only what was explicit to that viewer at that instant**, projected through
  the snapshot's visibility rules.

**Library positions** (type `libpos`): (anchor, pos, span). Anchor top or bottom; pos from that end,
0 = the end card; span = how many consecutive positions from pos the card may be in (1 exact, n one
of n, 0 anywhere). "Anywhere" is anchor top, pos 0, span 0; "not in a library" is anchor 0xFF, pos
0xFFFF, span 0. A source position is read before the move, a destination after it; cards move one
at a time, so each event shifts the positions after it (the actor tracks library sizes from
`game_start` and the moves to convert between anchors).
- A destination is exact (the instruction says top, bottom or a depth).
- A source is exact for a viewer who sees the card's identity, and when nobody has a window on the
  card (a draw, a mill: the instruction names the position).
- A card in a window another player has is one of the cards in such windows in its library
  (`library_position_seen_by`, `src/queries/visibility.h`): the span covers their positions, known
  because those windows opened on known positions and every move since was reported. A card in a
  window opened on it by identity (`look_at_found_card`) is anywhere.
- A pick from cards the viewer cannot tell apart is a range, which depends only on public counts:
  the opponent scries 2 and bottoms one: "a card from top positions 0–1 → bottom 0".

**Order-hidden batches** (`ObsLibraryBatch`). An instruction that puts several cards into a library
in an order some seat did not see opens a batch: Dig's unchosen rest, `ChangeZoneAll` into a
library, `DigUntil`'s revealed cards. Every seat other than the player who orders them is batched
(both seats for a random order; the omniscient viewer never). For a batched viewer each card the
instruction puts there carries the `batch` and no destination, identifiable or not, and a
`library_place` closes the batch with the range they occupy.

| Event | Fields (viewer's projection) |
|---|---|
| every event | `seq`; `turn` (`Game::turn_state.turn`: 0 during the pregame and the first turn); the source: the resolving spell, or the source of a resolving ability while it is still that object (handle and card id, else card id only); none outside a resolution |
| `zone_move` | one card (a card, token or spell copy; an ability's stack object is not reported), count 1. From zone/owner, its position if from a library, its hand subset if from a hand (`drawn_this_turn` when the instruction may only choose among cards drawn this turn, e.g. Sylvan Library's put-back; else `any`); to zone/owner, its position (or `batch`) if into a library; `handle_before` (the viewer's handle, 0 if none: a handle is never issued just to be retired) and `handle_after` (for every viewer who can identify the new object); card id if the viewer saw the identity before or after; face-down flag; token/copy flag. Zone `none` (0xFF): as origin, created (a token, built in its controller's hand zone and moved onto the battlefield; a spell copy on the stack; a harness preset, after `game_start`); as destination, ceased to exist |
| `draw` | player, count (always 1: one event per card, so a draw link's index is 0), opening-hand flag (opening-hand and mulligan draws are not draws for "drawn this turn"); handle after and card id for the drawer; for another viewer, handle before and card id only if the card was in that viewer's live window. Always from top 0 |
| `library_place` | closes a batch: batch, library owner, range (anchor, pos, count), order known (false) |
| `reveal_start` / `reveal_end` | cards (handles, zone, owner, library positions, card ids), audience, source effect; per card a draw link (`seq` of the `draw` and the card's index in it) when the reveal shows the card just drawn (Miracle). `reveal_end` also closes a look. A window that ends because its card moved or was turned face up has no `reveal_end` |
| `look` | looker, zone/owner, count; per card its library position; identities and handles for the looker only. A search of another player's hidden zone is a `look` at all of it. A non-looker gets each card's position when the look is at known positions (the top N, the whole library), unknown when the looker picked the card by identity (a search's find, kept in view after the shuffle). A standing look is a `look` to the looker at each new top card, and to the other seat once, as the look at that library starts (a reordering it can't see must not tell it the top card changed) |
| `shuffle` | library owner |
| `reorder` | library owner, range, whether this viewer saw the result and, if so, the card at each position. No engine site: rearrangements (Ponder, scry, surveil) are library → library `zone_move`s |
| `search` | searcher, zone searched (library, sideboard, graveyard, hand), its owner, cards found; once the pick is made, per searched hidden zone (a pick from the searcher's own hand, such as a put-back, is no search), after the `look` at it. The cards that leave follow as `zone_move`s, a shuffle as `shuffle` |
| `cast_from_hidden` / `play_from_hidden` | a `zone_move` (same fields) from a hidden zone (hand, library, sideboard, face-down exile) to the stack / battlefield; the card becomes identifiable here. A reversed proposal (CR 733.1) returns it with a `zone_move` keeping its identity; a card cast off a library's top returns to its exact position, the other seat keeping what it saw. A card cast face down (CR 708.4) moves with the face-down flag and no identity for anyone who couldn't see it, and its reversal shows none. A cloak from a library (CR 701.58a) is a `play_from_hidden` with the flag: its position for every viewer, its card for the controller and a seat that saw it as it moved |
| `face_down` / `face_up` | handle; identity at face-up. `face_down` is a public object turned face down where it is (`SetState`'s `TurnFaceDown`, which no vocab card uses); entries face down are `zone_move`s with the flag (exile face down, the `--face-down-a/-b` presets, a morph cast as `cast_from_hidden` then `play_from_hidden` from the stack, a cloak). `face_up` is `SetState`'s `TurnFaceUp` or the special action turning a morph or cloak up (CR 702.37e, 701.58b): same handle, the card to both seats |
| `game_start` | game number, starting player; per player library and sideboard counts. After the libraries are built and shuffled (after sideboarding in games 2–3), before the opening draws; opening hands and mulligans arrive as `draw`, `zone_move` (hand → library bottom) and `shuffle` |
| `game_end` / `match_end` | game number, winner (or draw); match wins per player. `game_end` after the game's last events; `match_end` after the last `game_end` (a single game is a match of one) |

**Identity references.** An event uses a handle when the viewer can identify the object at that
instant, else describes it by zone and position ("an unknown card from B's hand"). The event that
makes an identifiable object unidentifiable carries its handle, card id and destination
(`zone_move h7 (Force of Will) → B library, position 0`); the handle then retires. A
hidden-to-hidden move with no reveal never carries identity (the opponent's Brainstorm put-back:
two unknown cards from B's hand to the top of B's library, order hidden).

**Implementation** (`src/state_format/events.{h,cpp}`). The engine calls `obs_*` hooks at its choke
points: `Orderer`'s zone move (`move_object`, behind `add_to_zone`, `put_in_library_at_depth`,
`perform_draw`), the cast move and its rewind, `shuffle_library`, object creation and cessation,
the window opens and closes in `src/queries/visibility.cpp`, `search_zones`, `SetState`'s face
turns and the game loop's boundaries. Each hook asks the visibility accessors once per viewer and
queues that viewer's projected `ObsEvent`, which holds only those answers, never an entity id. The
next STATE frame drains the queue into `event` / `event_card` / `event_player`. `--trace-events`
prints each event as queued (`EVENT:`).
- Reveals: each opening call reports its new windows as one `reveal_start` (or `look`) per zone; an
  already open window is not reported again. `reveal_end` is reported per zone and audience, to
  each viewer shown the windows, for windows ending at a resolution's end or a shuffle.
- Handle columns: `reveal_start` / `look` carry the card's handle in `handle_after`, `reveal_end` in
  `handle_before`, `face_up` / `face_down` in both.

## 6. Handles

- A handle is a per-viewer opaque id unrelated to entity ids (entity ids are allocated in
  deck-list order and survive shuffles, so they would reveal identity).
- A handle exists **only while the viewer can individually identify the object**: in a public zone,
  in the viewer's own hand, or in one of the viewer's reveal windows (`individually_identifiable`,
  `src/queries/visibility.h`).
- A public-to-public move links the old handle to the new one in the event (`h5 → graveyard as
  h9`): linkable, yet a new object (CR 400.7).
- When an object stops being identifiable, the event that hides it carries the handle (section 5)
  and the handle retires; if it becomes identifiable again it gets a fresh one.
- Face-down permanents and cards exiled face down are public-zone objects with handles (CR 708.6);
  only identity is withheld. Turning a public object face up or down keeps its handle. Phased-out
  permanents keep theirs (included, flagged).
- Tables are per game in `Game::handle_tables` (A, B and omniscient; snapshot-covered), reset at
  game start. `--trace-handles` prints each viewer's table at every decision (`HANDLES:`) and each
  issue, link and retirement (`HANDLE:`).

**Implementation** (`src/state_format/handles.{h,cpp}`).
- **Keyed by object**: an entry holds the object's identity stamp (`Zone::obj_gen` and the id's
  issue count), so a zone change, a token ceasing to exist or a reissued id ends it.
- **Retired where identifiability ends**: at every zone change (`Orderer::add_to_zone`, the cast
  move and its rewind: the rewound card returns with a fresh handle linked from its stack handle),
  as the window that showed a card closes (resolution end, shuffle, the close hook in
  `visibility.cpp`), and as an ability or spell copy leaves the stack. Any lookup and every
  canonical issue also retire stale entries.
- **Links**: a zone change issues the new object's handle at once, in move order, for every viewer
  who held one and can still identify it; `handles_begin_move` / `handles_end_move` return both
  handles per viewer (`HandleMove`) for the move event.
- **Issue order**: lazily, by links in move order, by `handle_of` in its caller's (event) order,
  and at frame start by `issue_handles_canonical` in `canonical_object_order`, which uses only what
  the viewer sees: zone (battlefield, stack, graveyards, exile, hands, libraries, sideboards); seat,
  the viewer's own first (controller on the battlefield, owner elsewhere, none on the stack);
  position where the zone's order is visible (stack, graveyard and exile from the top; a library
  card in a window by position); name when the identity is visible (a face-down object first);
  battlefield timestamp; arrival order where the viewer saw the arrival (a public zone or own
  hand). Never entity order.
- **With an event consumer** on (a frame writer, `--selfcheck-state`, `--trace-events`) a handle is
  usually issued by the event that made the object identifiable, so numbers follow the viewer's
  event stream; with none, every handle is issued canonically at the next frame. Either way they
  are functions of what the viewer saw: an event showing several cards lists them in a derivable
  order (`reveal_cards` / `look_at_cards`: owner, zone, library position, name, then the tie
  below). The scenarios `state_format_handles_deck_order_independent*.json` build one game from
  decklists in another order and require identical handles for the viewer who cannot see the
  reordered cards, with and without an event consumer.
- **Ties**: objects equal on every key are copies of one card in the same unordered place, with
  identical visible rows; they are tied by entity id, whose order among copies is fixed at deck
  build and unrelated to their history. Arrival order is not used for an opponent's hand (it would
  say which copy was drawn when). A face-down object never reaches the tie.
- **Caveat**: permanents entering together get their battlefield timestamps in entity order
  (`apply_permanent_components` walks entity ids). Harmless for public identities; it would leak
  for face-down permanents entering together. None do today (Morph resolves alone; cards are
  cloaked one at a time, CR 701.58e): a card that puts two face down at once must order their
  timestamps without the entity id.

## 7. Actor side: the knowledge sidecar

Recall and inference are done by the C++ knowledge sidecar ([`knowledge_store.md`](knowledge_store.md)).

- **Input:** only the viewer's own frame stream. **Output:** `k_*` tables (ids 90–95) appended as
  the last blocks of each of that seat's STATE frames, terminal frames included; the omniscient
  frames carry none.
- **Levels:** `off` (the `k_meta` row alone) and `expert` (perfect recall plus every sound
  deduction), per seat by `--knowledge-a/-b off|expert`, default `expert`. An output option: it
  changes only that seat's `k_*` tables, never play; the RMLOG header does not record it and
  `--replay` honours it.
- **Instances** (`src/state_format/knowledge_feed.h`, `knowledge_store.md` §9.6): every frame
  writer (`--rmstate-dump`, the `--machine` stream, `--selfcheck-state`, the in-process agents'
  feed) has its own per seat, fed exactly the frames that writer builds for the seat (SCHEMA,
  CARDS, DECKLISTS, then each STATE frame as encoded). The model is a function of the seat's events
  (each applied and settled on its own; the snapshot is only checked), not of how they were
  batched into frames, so every writer's frame for one seat and decision carries the same `k_*`
  tables; they are a pure function of that writer's frames for the seat (`rmknow --isolation`
  recomputes them byte for byte); one writer's bytes never depend on whether another is on.
- **Checks:** soundness against the omniscient truth, exactness against brute-force enumeration,
  isolation via `rmknow` (section 10).

## 8. Engine integration

- The engine models no per-viewer knowledge or memory; everything a frame says is derived from the
  current state, the live reveal windows and the event queues.
- Events are built by typed constructors that take already-filtered data: the viewer and the
  visibility rules' answer, never a raw entity (section 5).
- **Menu collapse**: `interchangeable_cards` (`src/queries/zones.cpp`) merges two identical cards in
  one library only when each seat sees both or neither right now (`identity_visible`: a live
  window); what a seat remembers of a closed window is not read.

## 9. Encoding and protocol

- **Relational tables.** A frame is a sequence of tables of fixed-width little-endian rows.
  Variable-length data goes in child tables keyed by handle, or by the parent's row index where the
  parent has no handle (events). Text is an offset into a per-frame string pool. Python reads rows
  with `struct.iter_unpack`. Layout and columns: Appendix A.
- **Schema frame.** Sent to each viewer once per match, before its other frames: tables; columns
  (name, storage type, width, a **semantic tag** — `enum:<table>`, `count`, `bool`, `signed`,
  `handle`, `card`, `player-rel`, `text`, `index`, `row:<table>`, `mask:<enum>` (Appendix A.1) —
  and a value range where defined); and string tables for every enum (action categories,
  keywords, steps, zones, card types, event kinds, …; counter kinds are script strings and travel
  as `text`). Decoders and featurizers build themselves from it: no copied constants, no codegen.
- **`--machine`** (also `--machine-format rmstate`; `src/state_format/stream.h`) is this protocol
  for each seat with no in-process agent that is not a human seat (a run with a human seat has no
  stream seat; under `--replay` the log supplies every decision and no stream is written).
  `--broadcast-steps` and `--search-server` are refused.
  - stdout carries only `RMSTATE <nbytes>\n` + payload records: the engine moves stdout's
    descriptor aside at startup and points stdout at stderr, so everything else (narrative under
    `--narrative`, `GAME_RESULT` / `MATCH_RESULT`) goes to stderr, flushed before each frame, so a
    reader of both pipes sees a decision's narrative ahead of its frame (`python/rm_driver.py`).
  - At each decision the deciding seat (the sideboarding player between games, else the priority
    holder) gets, the first time it decides in the match, its `SCHEMA`, `CARDS` and `DECKLISTS`
    frames, then its `STATE` frame with the menu, carrying every event queued for it since its
    previous frame.
  - The reply is one line: menu row *i* (the menu as offered, after the interchangeable-choice
    collapse), `-1` (the last row: an attacker/blocker confirm), `-2` (concede game) or `-3`
    (concede match). End of input or an unparseable line reads as `-1`. The menu is unbounded
    (`--closed-decklist`'s name-a-card menus offer every vocab card, register 11a).
  - **One-row menus are not asked.** A decision whose menu has one row after the collapse (a lone
    pass, a confirm with nothing left to declare, a single legal target, a search that can only
    fail, a sideboarding phase with no sideboard, …) is taken by the engine for every consumer (the
    stream, in-process agents and their teachers, the CLI, `--play` / `--actions`, `--replay`): no
    frame, no decision number (`header.decision` counts asked decisions), not in the decision log,
    not counted by the decision caps. Its events reach the seat's next frame. A seat can concede
    only at an asked decision.
  - **Terminal frame.** After the `match_end` event each seat, A then B, gets one more `STATE`
    frame with `header.terminal = 1`, no menu and no reply, carrying its last events; the engine
    then exits. `--rmstate-dump` ends with the same terminal frame for each of its three viewers.
  - The frames equal those `--rmstate-dump` writes for that seat and decision except for event
    batching (each writer drains its own queue copy, and feeds its own knowledge instance, whose
    output does not depend on batching). While either writer is on, every viewer's handles are
    issued canonically before any frame is built, so the stream's bytes do not depend on whether a
    dump is written.
- **Versioning.** Every frame header carries the format version (`FORMAT_VERSION`, `schema.h`;
  currently 2) and schema hash; a decoder refuses a mismatch.
- **Cadence.** A frame at each of the viewer's decisions only; there are no pushed passive frames.
- **Debug view.** `--state-dump json` renders the decoded binary (no second encoder): every record
  `--rmstate-dump FILE` writes becomes one JSON line in `FILE.jsonl`, and `--rmstate-read FILE
  --state-dump json` prints a stream's lines on stdout. Renderer `src/state_format/json.h`;
  `python/rmstate.py` renders the same bytes. One object per frame, no whitespace, keys in fixed
  order: `{"frame":…,"viewer":…,"version":…,"schema_hash":"<16 hex>","tables":{"<table>":[{row},…],…}}`
  (plus `"skipped":[ids]` for unknown block ids). Tables in ascending id, string pool left out;
  columns in schema order. Text as a string; `enum:<e>` as its name (`null` for all ones);
  `mask:<e>` as the list of set bits' names; `bool` as `true`/`false`; `card`, `index`, `row:` as a
  number or `null` for all ones; everything else (handles included) as a number.
- **Decoding.** `src/state_format/decode.h` (std and `schema.h` only) builds a schema from a
  `SCHEMA` frame (reading tables 1–4 with `schema.h`'s layout, which the frame must repeat, and
  checking its hash) and decodes any frame into tables of typed values; `encode_frame` re-encodes
  them, byte-identical for the engine's frames. It refuses a version or schema-hash mismatch, an
  unknown frame kind or viewer, a known table with another row width or in a frame of another kind,
  and malformed framing; it skips unknown table ids. `--rmstate-read FILE`
  (`src/state_format/recode.h`) decodes, re-encodes and compares every record of a stream (`ERROR:`
  lines, exit 1 on any difference). `python/rmstate.py` is built from the `SCHEMA` frame the same
  way (bootstrap: the frame and block headers and the layout of tables 1–4).

## 10. Verification

`--selfcheck-state` (`src/state_format/selfcheck.{h,cpp}`) runs these at every decision, building
every viewer's STATE frame (A, B, omniscient) from its own queue copy (`ObsConsumer::SELFCHECK`), so
it needs no frame writer. Failures print `ERROR: selfcheck-state <check>: viewer …, decision …
(game, turn, step): <first differing table, row, card and column>`; a `SELFCHECK-STATE: decisions=…
frames=… events=… permutations=… snapshots=… errors=…` summary goes to stderr at match end. Play,
the RNG and the `--rmstate-dump` bytes are the same with it on. It skips the decisions of an
in-process provider that also answers simulated ones (a search agent's).
`--selfcheck-knowledge` (`selfcheck_knowledge.h`, `knowledge_store.md` §9.5; implies
`--selfcheck-state`) adds every `k_*` fact of both seats against the true world, and the knowledge
state across the snapshot check's restore; `rmknow --isolation` checks isolation.

- **Hidden, decided independently.** Both leak checks decide what a seat cannot see from section
  3's table directly (zone, owner, controller, face-down state, the seat's windows in
  `Game::reveal_windows`), not through `queries/visibility.h`, so a wrong visibility rule is caught
  as well as an encoder read that bypasses the rules.
- **roundtrip:** every frame (and once per match each viewer's `SCHEMA`, `CARDS`, `DECKLISTS`)
  through `decode_frame` → `encode_frame`, byte-identical.
- **scramble** (event leak, and the snapshot at every decision): each seat's frame is encoded again
  inside `ScrambledIdentities`, which gives every card hidden from the seat another identity: the
  hidden cards' `CardData` rotate among them (sorted by name, shifted by the largest copy count) by
  remapping component slots (`Coordinator::PermuteComponents`; nothing is copied); a hidden object
  without `CardData` gets another id from `card_identity`'s test-mode scrambler. The bytes must
  match. Every event is built twice the same way as it is constructed (`emit` in `events.cpp`, the
  move capture in `obs_begin_move`) and the two builds' rows (`encode_events_frame`) must match. A
  menu entry's card made public by the choice itself (`card_is_public`) keeps its identity.
- **permutation** (snapshot leak, every `--selfcheck-state-sample N`th decision, default 8): per
  seat, the opponent's hidden hand, library and sideboard cards are dealt at random among those
  cards' places (entity ids move, as a determinizer's would), the seat's own hidden library cards
  among their positions, then the identities of all cards hidden from the seat are shuffled per
  owner (a card in a window shown only to the other player, or a face-down object, keeps its place
  with another identity). Cards in live windows keep their places; owners, zone sizes, identity
  stamps and handle tables are unchanged. The seat's frame must be the same bytes. Everything is
  put back exactly (saved `Zone`s, the inverse slot permutation, the handle table) without a
  snapshot restore, so it runs at any decision, a blocking prompt mid-flow included.
- **complete:** object handles unique and nonzero; every object the viewer can identify has a row,
  and every object of a public zone (plus the seat's own hand and sideboard; everything for
  omniscient) is counted per zone and owner against the object rows; player zone counts against the
  zones; the reveal table against the viewer's windows; every `handle` column of a snapshot table
  names an object row (event tables exempt); every `row:` reference in range; the menu has one row
  per action for the deciding seat and omniscient, none otherwise; the sidecar's pool sizes match
  the snapshot.
- **snapshot** (sampled): at a loop-safe decision outside sideboarding the game is saved
  (`snapshot_save`, the continuous-effects registry copied; the knowledge instances through the
  snapshot's sidecar hooks, after the match frames are fed and before the decision's). At the first
  loop-safe decision at least 3 decisions later the current state is saved, the earlier restored,
  every viewer's frame encoded there (menu, decider, priority-window flag, drained events) and fed
  to the restored knowledge instance, compared with the saved decision's bytes (`k_*` included),
  and the current state restored. Both restores use `snapshot_restore_exact` (registry put back
  from the copies): a plain `snapshot_restore` re-derives continuous effects, which can differ
  where the derived ones lag the state (`docs/rules_deviations.md` 2.3).
- **names** (every seat frame): a STATE frame's text columns (above all the menu's `description`;
  also keyword texts, counter kinds, player-effect filters) may name only cards the frame shows.
  Each vocab card name found as whole words (longest name at each word start, case-sensitive:
  `Island` is not found in `Islandwalk`) must be held by some `card` column of the frame (objects,
  stack, LKI, events, the menu's `source_card`, chosen names, …; not `own_deck`), be a face of such
  a card, or be named in such a card's oracle text. Descriptions are built before encoding, so the
  scramble and permutation cannot see them; this check can. It cannot tell copies apart.
- **Knowledge tables.** The self-check's own knowledge instance per seat is fed the match frames
  and, once, each real STATE frame it builds (decisions and terminal); its `k_*` tables are
  appended, and roundtrip, complete (`row:k_pool` references) and snapshot run on that frame. The
  scramble and permutation builds are never fed: they are compared with the frame before its `k_*`
  tables, which are a function of the instance's state and those bytes alone. The names check does
  not count `k_*` card columns as shown. The instance's consistency errors are `selfcheck-state
  knowledge` failures.
- **Bypasses of `card_identity`.** Three identity reads skip `visible_card_id`: the card a
  `card_is_public` menu choice makes public (`public_card_id`; the name-a-card menus take their
  candidates from the registered lists, public by default, or the whole vocab under
  `--closed-decklist`, with no object behind a candidate: `LegalAction::named_card`); captured
  public ids (last-known information of an object that left the battlefield, a delayed or floating
  trigger's creator), public when captured; and a zone-less sideboard template card, shown only to
  the sideboarding player. The encoder redacts a menu `description` whose source or target is
  hidden (`(hidden card)`); the names check covers the rest.

## 11. Decision register

Format rules cited by number from code and other docs.

| # | Rule |
|---|---|
| 1 | Post-layer results are authoritative, plus coarse per-object provenance (`object_provenance`); no dump of the continuous-effect registry (only layer 7 has `ContinuousEffect` payloads, `src/systems/continuous_effects.h`) |
| 2 | Abilities by reference: card id + script ability index (keyed definitions by key string), a short header (category, costs, timing, limit state), the script line. Definition pointers and intern order are not stable across processes |
| 3 | Relational fixed-width tables (section 9) |
| 4 | Schema frame with enum string tables; no copied constants, no codegen |
| 5 | Full snapshot per decision plus the observation event stream (section 5) |
| 6 | Viewers A / B / omniscient (flag only) |
| 7 | Handles exist only while the viewer can individually identify the object (section 6) |
| 8 | The engine models no knowledge or memory; recall and inference are actor-side (section 7) |
| 9 | Own library: count only; contents and order are actor-side |
| 10 | Opponent hand: count plus live reveal windows |
| 11 | Decklists are open by default: both registered lists (main + side) in `DECKLISTS`. `--closed-decklist` sends only the viewer's own, and the leak tests treat the opponent's as hidden; recorded in the RMLOG header |
| 11a | Name-a-card candidates follow 11: with open decklists, the distinct vocab cards of the relevant registered list(s) filtered by `ValidCards$` (the CR 201.4 deviation, `docs/rules_deviations.md`); under `--closed-decklist`, every vocab card passing `ValidCards$`, in name order, uncapped (`build_name_card_choices`) |
| 12 | Menu descriptions are included and covered by the leak tests; the full-information narrative never goes to a seat |
| 13 | `CARDS` once per match is the agent's card knowledge; card id is the vocab index |
| 14 | Full keyword lists with parameters; counters as kind → count |
| 15 | Rules-derived convenience fields only where the engine already computes them (effective sickness, land drops left, mana potential) |
| 16 | Pending context = `PendingQuery::Tag`, deciding seat, source, menu category |
| 17 | One integer per decision, over the menu after the interchangeable-choice collapse |
| 18 | Rules-state tables (delayed triggers, emblems, player effects, play permissions, LKI, shields); the rest through provenance |
| 19 | `RMSTATE <nbytes>\n` framing; narrative off stdout; version + hash in every frame |
| 20 | Frames at the viewer's decisions only; no push stream |
| 21 | The menu collapse reads only currently explicit information (section 8) |
| 22 | Deferred: search agents install a hidden world sampled from the actor's beliefs (a later engine command); the engine never determinizes from its own knowledge |
| 23 | `DECKLISTS` carries each registered list's deck name (the opponent's withheld under `--closed-decklist`) |
| 24 | `object.token`, and token entries in `CARDS` with their real types |
| 25 | Ability references carry a keyword key for keyword-built abilities (`Ninjutsu`, `Equip`, `Reconfigure:attach`, …) |
| 26 | Recall and inference live in a C++ sidecar fed only by the viewer's frame stream (`knowledge_store.md`), not in the engine |
| 27 | Knowledge levels `off` and `expert` only; default `expert`; per seat (`--knowledge-a/-b`) |
| 28 | Knowledge tables (`k_*`) travel inside the viewer's STATE frame, with a schema fixed across levels |
| 29 | `CARDS` always carries oracle text, for cards and tokens (the source of the text embeddings) |
| 30 | Every schema column carries a semantic tag and, where defined, a value range (section 9) |
| 31 | Every row with an owner/controller/player column also carries it relative to the viewer (self/opponent) |
| 32 | Auto-mana is on by default for every consumer; `--no-auto-mana` turns it off match-wide (no per-seat variant); recorded in the RMLOG header |
| 33 | Every cost-payment decision's frame carries `pending_cost`: what is paid for and the remaining cost by pip type (section 4) |
| 34 | Counter kinds travel as `text` (no engine enum) |
| 35 | Face-down exiled cards follow CR 406.3: a per-player look window for whoever looked at the card before it was exiled face down or is allowed to look, until it leaves exile |
| 36 | Events carry draw provenance: `turn` on every event, an opening-hand flag on `draw`, a hand subset (`drawn_this_turn`) on a `zone_move` out of a hand restricted to cards drawn this turn (Sylvan Library), and a draw link on a `reveal_start` showing the card just drawn (Miracle). Both restrictions are explicit to both players, so the sidecar may use them |
| 37–38 | Withdrawn (numbers not reused) |
| 39 | `--rmstate-card-scripts` adds the raw script to `CARDS` (the `script` column of a card's front row and a token's row) |
| 40 | A stack object's target that has left the stack has an `lki` row too: its card, token flag, controller, printed P/T, types and colours as it left the stack, as each viewer saw them (recorded at the zone-move choke point while another stack object can refer to it, dropped once the stack empties); scenario `format_stack_target_lki` |
| 41 | Structured ability parameters in `CARDS`: `card_ability`'s amount and its kind (fixed, variable, none), Pump modifiers, a target-class mask and bounds, `n_parts`, `charm_num`; child `card_ability_part` (9), a row per sub-ability and mode, recursively. Read from the parsed `AbilityDef` and its typed `params`, generic per category and params type, never per card; the `line` text stays |

## Appendix A: column definitions

The column list coded in `src/state_format/schema.cpp` (`table_defs`); the `SCHEMA` frame is
generated from it; keep this appendix in step with it. The encoder (`src/state_format/encode.cpp`)
checks every row it writes against these definitions in debug builds (storage type per value,
column count per row, row width per table).

### A.1 Conventions

| Item | Rule |
|---|---|
| Storage | little-endian `u8 u16 u32 u64 i8 i16 i32 f32`; rows packed, no padding; `text` = u32 offset into the frame's string pool |
| String pool | table `strings` (id 1; one column `byte`, u8, tag `text`; one row per byte): UTF-8, each string NUL-terminated; offset 0 is the empty string (the pool starts with a NUL). Each distinct string is stored once, at its first use in table, row, column order, so re-encoding is byte-identical. The pool is always present |
| None values | handle 0; text 0 (empty); card, `index`, `row:` and nullable enum columns: all ones of their width (0xFF, 0xFFFF, …) |
| Enums | the C++ value (`Step`, `ActionCategory`, `Zone::ZoneValue`, …), string tables in `SCHEMA`; the format's own enums are listed in A.7 |
| Seat pair | type `seat` below = two u8 columns: `<name>` (`enum:seat` = `Zone::Ownership`: 0 none, 1 A, 2 B; range 0-2) and `<name>_rel` (`player-rel`: 0 none, 1 self, 2 opponent; for `omniscient`, self = A; range 0-2). Register 31 |
| Library position | type `libpos` = `<name>_anchor` u8 `enum:lib_anchor` (top, bottom) + `<name>_pos` u16 `index` (0 = the end card) + `<name>_span` u16 `count` (1 exact, n = one of n from pos, 0 unknown); section 5 |
| Ability reference | type `abref` = `ab_card` u16 `card` + `ab_index` u16 `index` (row of that card in `card_ability`) + `ab_key` `text`. Registers 2, 25. `ab_key` is `AbilityDef::ref_key`: for a keyword-built ability the keyword table entry's name, plus `:<label>` when the ability's label differs (`Equip`, `Ninjutsu`, `Reconfigure`, `Reconfigure:unattach`); for an engine-built definition its `keyed_ability_def` key (`subtype_mana:1`, `keyword:Prowess`); empty for a plain script ability. An ability that is not in its card's list (granted, engine-built) has `ab_card` and `ab_index` none; its key is `ref_key`, else its keyword label, else the keyword it derives from, else `LandType` for a land-type mana ability. The lookup reads the card's loaded definition by card id, never an object the viewer cannot see |
| Card ids | a `card` column holds the vocab id of the physical card: a double-faced card's front face, whichever face is up and whichever face a deck line, a last-known name or a captured creator names (`physical_card_of`, `src/queries/visibility.h`); a copy's printed card; a token's token-band id. A back face's own vocab id appears only as `card.face_card` |
| Parent links | a child row names its parent by handle when the parent has one (`object`, `stack`), else by the parent's row index in the same frame (`row:<table>`) |
| Masks | bit i = value i of the enum (`mask:color` over `Colors`, `mask:card_type` over the engine's type list, …) |
| Ranges | `bool` columns 0-1; seat pairs 0-2; no other column carries a range yet (enums are bounded by their string tables) |

**Tags:** `index` is a 0-based position or ordinal (library or stack position, ability index,
mode), not a quantity; `row:<table>` a row index into another table of the same frame;
`mask:<enum>` a bitset over an enum. Absolute seats are `enum:seat`, zones `enum:zone`.

### A.2 Frame layout

| Part | Layout |
|---|---|
| Framing | `RMSTATE <nbytes>\n`, then `nbytes` of payload (section 9) |
| Frame header (24 B) | `magic` u32 (`RMSF`), `version` u16, `kind` u8 (`enum:frame_kind`), `viewer` u8 (`enum:viewer`), `schema_hash` u64, `n_tables` u32, reserved u32 (0) |
| Table block | header (16 B): `table_id` u16, reserved u16, `row_width` u32, `row_count` u32, reserved u32; then `row_count × row_width` bytes, zero-padded to a multiple of 8 |
| Order | blocks in ascending table id (so `strings`, id 1, comes first); a table with no rows is omitted; `n_tables` counts the blocks present |
| Stream | `--rmstate-dump FILE` writes `RMSTATE <nbytes>\n<payload>` records: per viewer (A, B, omniscient, in that order), at its first decision of the match `SCHEMA`, `CARDS`, `DECKLISTS`, then a `STATE` frame at every decision, and at match end a terminal `STATE` frame (`header.terminal`); a seat's `STATE` frames end with its `k_*` tables (section 7). `%p` in FILE is replaced by the process id. `--machine` writes the deciding seat's records on stdout (section 9) |
| Unknown data | a decoder skips a block whose id it does not know; a known id with another row width is a schema mismatch and is refused |
| Schema hash | 64-bit FNV-1a over the `SCHEMA` frame's table blocks (header excluded) |
| Frame kinds | `SCHEMA` (ids 2–4), `CARDS` (9–14), `DECKLISTS` (15–16), `STATE` (snapshot 20–48, events 80–82, knowledge 90–95) |

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
| `card` (10): one row per face | `card` | u16 | `card` | vocab id of the physical card (a double-faced card's front face; tokens in the token band) |
| | `face` | u8 | `enum:face` | front, back |
| | `face_card` | u16 | `card` | the face's own vocab id (a back face has its own: Delver of Secrets // Insectile Aberration), none when it has none |
| | `name`, `uid` | text | `text` | |
| | `token` | u8 | `bool` | register 24; a token's `uid` is its script stem |
| | `layout` | u8 | `enum:layout` | normal, transform, modal DFC, split |
| | `mana_cost` | text | `text` | printed: `{X}` pips, plain pips generic first (`{2}{U}`, `{0}`), then hybrid (`{W/U}`, `{2/W}`) and Phyrexian (`{B/P}`) pips; empty for a land |
| | `mana_value` | u8 | `count` | |
| | `cost_w`, `cost_u`, `cost_b`, `cost_r`, `cost_g`, `cost_c`, `cost_generic`, `cost_x` | 8 × u8 | `count` | plain pips; X pips |
| | `colors` | u8 | `mask:color` | |
| | `types` | u32 | `mask:card_type` | |
| | `supertypes` | u8 | `mask:supertype` | |
| | `power`, `toughness` | i16 | `signed` | printed |
| | `loyalty` | u8 | `count` | |
| | `oracle` | text | `text` | `CardData::oracle_text` (register 29) |
| | `script` | text | `text` | the raw script text under `--rmstate-card-scripts` (register 39): a card's file (a double-faced card's combined script) on its front row, a token's script on its row; empty otherwise |
| `card_subtype` (11) | `card`, `face`, `subtype` | u16, u8, u16 | `card`, `enum:face`, `enum:subtype` | |
| `card_keyword` (12) | `card`, `face` | u16, u8 | `card`, `enum:face` | |
| | `kw` | u8 | `enum:keyword` | the longest `keyword` value the text starts with, followed by its end, `:` or a space (`Ward:2` → Ward, `Protection from blue` → Protection); 0xFF = none |
| | `text` | text | `text` | the keyword as held, e.g. `Ward:2` |
| `card_ability` (13); statics and replacements have none / 0 in `amount` … `charm_num` | `card`, `index` | u16, u16 | `card`, `index` | index: per card over both faces, front then back; per face its spell, activated and triggered abilities (`CardData::abilities` order: A: lines, T: lines, then keyword-built), gift, chapter and opening-hand abilities, then statics (S:), then replacements (R:) (`abref.ab_index`). A token's list: its abilities, statics, replacements |
| | `face` | u8 | `enum:face` | |
| | `ability_kind` | u8 | `enum:ability_kind` | spell, activated, triggered, static, replacement, chapter, opening hand |
| | `category` | u8 | `enum:effect_kind` | `effect_kinds.def`; 0xFF for statics and replacements |
| | `key` | text | `text` | `AbilityDef::ref_key` (register 25; see `abref`) |
| | `cost` | text | `text` | the script's cost: the line's `Cost$` (`T Sac<1/CARDNAME>`), or a keyword line's argument (`K:Equip:1 R` → `1 R`) |
| | `tap` | u8 | `bool` | `{T}` in the cost |
| | `timing` | u8 | `enum:timing` | spell: instant when an instant or with Flash, else sorcery; activated: sorcery for a loyalty, `SorcerySpeed$`, Equip/Reconfigure/Unattach ability, else instant; anything else any |
| | `limit` | u8 | `count` | activations per turn; 0 = none |
| | `line` | text | `text` | the script line with its key (`A:`, `T:`, `S:`, `R:`, `K:`) |
| | `amount` | u16 | `count` | the effect's number (register 41): `AbilityDef::amount` (`NumDmg$`, `NumCards$`, `LifeAmount$`, `Amount$`, `ChangeNum$` …), else the effect's own count (`CounterNum$`, `PeekAmount$`, `DigNum$`); 0 when variable or none |
| | `amount_kind` | u8 | `enum:amount_kind` | variable: the parser kept an expression (an X or count-SVar, `dynamic_amount_expr`; a conditional, delirium-scaled or damage-dealt amount; a `CounterNum$` / `DigNum$` expression); fixed: a nonzero number; else none (a `Draw` with no `NumCards$` is none, as parsed) |
| | `power_mod`, `toughness_mod` | i16 | `signed` | a Pump / PumpAll's `NumAtt$` / `NumDef$`; a side given as an expression holds its sign (+1 for `+X`, -1 for `-X`) |
| | `pt_kind` | u8 | `enum:amount_kind` | variable when either side is an expression, fixed when either is nonzero, else none |
| | `targets` | u16 | `mask:target_class` | what may be targeted (`target_classes`, `src/targeting.h`, on `is_legal_target`'s branches): `TargetType$` → spell / ability; `TgtZone$ Graveyard` or a graveyard-origin ChangeZone → card in a graveyard; else per `ValidTgts$` alternative a spell for an `inZoneStack` one under `TgtZone$ Stack`, a player, a permanent card type, every permanent type a `Permanent` / `Card` head's `non<Type>` qualifiers leave, or permanent other (a subtype head); `Any` = creature, planeswalker, battle, player (CR 115.4). `card_other` (hand, library, exile) is never set: no ability targets there. 0 for no targeting (`ValidTgts$` absent) |
| | `target_min`, `target_max` | u8 | `count` | `TargetMin$` / `TargetMax$` (default 1/1); 0xFF when the bound is X or a count expression; 0/0 for no targeting |
| | `n_parts` | u8 | `count` | the ability's `card_ability_part` rows |
| | `charm_num` | u8 | `count` | modes to choose (`CharmNum$`, default 1) for a `Charm`; 0 otherwise |
| `card_ability_part` (9): a `card_ability` row's sub-abilities and modes | `card` | u16 | `card` | |
| | `ability` | u16 | `index` | the `card_ability` row's `index` |
| | `part` | u16 | `index` | 0-based per ability: depth-first in definition order, a definition's modes (`charm_choices`) before its sub-abilities (`subabilities`: `SubAbility$`, `RepeatSubAbility$`, `VoteSubAbility$`, `Execute$`), each part followed by its own parts |
| | `parent` | u16 | `index` | the enclosing part's `part`; none when it hangs off the ability |
| | `kind` | u8 | `enum:part_kind` | sub, mode |
| | `depth` | u8 | `count` | 1 for the ability's own parts |
| | `category` | u8 | `enum:effect_kind` | |
| | `amount` … `target_max` | | | as in `card_ability` |
| `card_pip` (14) | `card`, `face` | u16, u8 | `card`, `enum:face` | hybrid and Phyrexian pips of the printed cost |
| | `kind` | u8 | `enum:pip_kind` | hybrid, twobrid, Phyrexian |
| | `colors` | u8 | `mask:color` | |
| | `generic_alt` | u8 | `count` | twobrid's 2 |
| `decklist` (15) | `player` | seat | | only the viewer's under `--closed-decklist`; the viewer's own first |
| | `name` | text | `text` | deck name (register 23) |
| | `main`, `side` | u16 | `count` | |
| `decklist_card` (16) | `player` | seat | | |
| | `card` | u16 | `card` | a line naming a double-faced card's back face counts as that card (its front face) |
| | `main`, `side` | u8 | `count` | registered counts |

`CARDS` holds every card of both registered lists, in ascending card id, then every registered token those cards can make, in ascending card id. Under `--closed-decklist` it holds every vocab
card instead (by front face; a back face's own vocab entry is that card's back row), for every
viewer: the opponent's cards need rows, and listing only the seat's own would show which are the
opponent's (the engine's `--cards-frame FILE` writes this whole-vocab frame, after the `SCHEMA` frame, for
the featurizer's card tables, `docs/featurizer.md` §4). A card no deck holds is read from the parse cache (`card_data_by_name`,
`src/card_db.h`), which creates no entity. A card's tokens are found from its parsed definitions: each `TokenScript$` of its abilities (sub-abilities, modes,
floating and granted abilities included), the engine-synthesized ones (Investigate → Clue,
Amass → the subtype's Army, Mobilize → Warrior, `src/effects/effects.h`), then the tokens those
tokens make. A token script with no token-band vocab entry or no script file is left out.

### A.5 Snapshot (`STATE` frame)

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `header` (20) | `game` | u8 | `count` | 0-based game of the match (between games, the upcoming one); version, schema hash and viewer are in the frame header |
| | `decision` | u32 | `count` | decision number in the game, from 1 (`Game::state_decisions`); in a sideboarding frame, in the sideboarding phase before `game`, from 1 (`Game::state_sideboard_decisions`). (`game`, `sideboarding`, `decision`) names a decision of the match: unique per viewer among non-terminal frames (`rmstate_decision_number`, `src/state_format/stream.h`) |
| | `knowledge` | u8 | `enum:knowledge_level` | this viewer's level (`--knowledge-a/-b`); OFF for the omniscient viewer |
| | `terminal` | u8 | `bool` | the frame written once the match is over: no menu, no reply; it carries the last events (section 9) |
| | `sideboarding` | u8 | `bool` | a frame of the sideboarding phase between games (`match.sideboard_phase`) |
| `turn` (21) | `turn` | u16 | `count` | |
| | `step` | u8 | `enum:step` | |
| | `active`, `priority` | seat | | |
| | `priority_window` | u8 | `bool` | an ordinary priority window |
| | `mandatory` | u8 | `enum:mandatory_choice` | |
| | `attackers_declared`, `blockers_declared` | u8 | `bool` | |
| | `day_night` | u8 | `enum:day_night` | |
| | `next_extra_turn` | seat | | none = no extra turn queued |
| `player` (22): self row first (A first for omniscient) | `player` | seat | | |
| | `life` | i32 | `signed` | |
| | `mana_w`, `mana_u`, `mana_b`, `mana_r`, `mana_g`, `mana_c` | 6 × u16 | `count` | floating pool, all of it: restricted mana counts here and has its `pool_mana` row too |
| | `hand`, `library`, `graveyard`, `exile`, `sideboard` | 5 × u16 | `count` | zone sizes (by owner) |
| | `mulligans`, `to_bottom` | u8 | `count` | London bottoming still owed |
| | `passed` | u8 | `bool` | pass flag |
| | `monarch`, `city_blessing` | u8 | `bool` | |
| | `speed` | u8 | `count` | 1 + the player's speed (CR 702.179); 0 no speed, which is not speed 0 (CR 702.179a, c). A count for what `chosen_color` is one |
| | `speed_triggered` | u8 | `bool` | the ability speed brings has triggered this turn (CR 702.179d: it triggers only once each turn) |
| | `extra_turns` | u8 | `count` | queued |
| | `lands_played`, `land_drops_left` | u8 | `count` | register 15 |
| | `potential_w`, `potential_u`, `potential_b`, `potential_r`, `potential_g`, `potential_c`, `potential_total` | 7 × u8 | `count` | mana potential (register 15; `mana_potential`); total = untapped sources plus the floating pool |
| | `spells_turn`, `noncreature_turn`, `instant_sorcery_turn`, `spells_game`, `drawn_turn` | 5 × u16 | `count` | per-turn / per-game counters |
| | `life_gained_turn`, `life_lost_turn` | u16 | `count` | |
| | `colors_cast_turn` | u8 | `mask:color` | |
| | `revolt`, `drew_from_empty`, `companion_used` | u8 | `bool` | |
| | `match_wins` | u8 | `count` | |
| `player_counter` (23) | `player` | seat | | |
| | `kind` | text | `text` | `POISON`, `ENERGY`, … |
| | `count` | u16 | `count` | |
| `object` (24): one row per object the viewer can identify, in handle order | `handle` | u32 | `handle` | |
| | `zone` | u8 | `enum:zone` | |
| | `owner`, `controller` | seat | | controller: on the battlefield and the stack the object's controller, elsewhere its owner |
| | `pos` | u16 | `index` | stack from the top (0 = top); graveyard / exile from the top (0 = newest); library position for a card in a window; none elsewhere |
| | `card` | u16 | `card` | `visible_card_id`: 0xFFFF = face down to this viewer; an ability on the stack shows its source's card (else its last-known card, else a delayed or floating trigger's creator) |
| | `token`, `face_down`, `transformed` | u8 | `bool` | register 24; transformed also for a spell cast as its back face |
| | `copy_of` | u16 | `card` | the card or token a copy effect made a permanent a copy of (its printed card is `card`); a token copy of a card has the token sentinel as `card`, and this column names the card it copied, whose rows its `object_ability` rows reference |
| | `tapped`, `sick`, `phased_out` | u8 | `bool` | engine flags (register 15); `sick` is effective summoning sickness (`is_summoning_sick`: a creature, haste considered) |
| | `attacking`, `blocking`, `blocked` | u8 | `bool` | details in `combat` |
| | `damage` | u16 | `count` | |
| | `deathtouch_damage` | u8 | `bool` | |
| | `power`, `toughness` | i16 | `signed` | post-layer for a battlefield creature (a phased-out one's too); elsewhere printed (active face), or what a characteristic-defining ability counts there (CR 604.3: Nethergoyf in a graveyard) |
| | `types` | u32 | `mask:card_type` | post-layer on the battlefield, where Creature is set exactly while the permanent is a creature (an impending permanent with time counters and an attached reconfigure permanent keep the type in their type set but are not, CR 702.175d, 702.151b) |
| | `supertypes` | u8 | `mask:supertype` | |
| | `colors` | u8 | `mask:color` | `effective_colors` |
| | `mana_value` | u8 | `count` | |
| | `attached_to` | u32 | `handle` | |
| | `timestamp` | u32 | `index` | entered-battlefield order; none off the battlefield |
| | `entered_turn`, `cant_be_blocked`, `damage_shielded`, `delayed_subject` | u8 | `bool` | per-turn statuses: entered this turn, can't be blocked this turn, damage shielded, the subject of a pending delayed trigger |
| | `activations_turn`, `resolutions_turn` | u8 | `count` | this turn's activations of its activated abilities; this turn's resolutions from the stack of its triggered abilities (one that did nothing — declined, its intervening if false, its targets all illegal, CR 603.4 / 608.2b — did not resolve) |
| | `chosen_name` | u16 | `card` | |
| | `chosen_type` | u16 | `enum:subtype` | 0xFFFF none |
| | `entered_x` | u16 | `count` | |
| | `monstrous`, `evoked`, `escaped`, `unearthed`, `cast_from_hand` | u8 | `bool` | |
| | `exerted` | u8 | `bool` | it won't untap in the next untap step of the player who exerted it (CR 701.43) |
| | `bestowed` | u8 | `bool` | a bestowed Aura: it goes on as a creature if it falls off (CR 702.103f) |
| | `prepared` | u8 | `bool` | it has the prepared designation, a copy of its prepare spell in exile (CR 722.3) |
| | `chosen_color` | u8 | `count` | the color chosen as it entered, 1 + its `color` enum value; 0 none. A count, not an enum, because a categorical column moves every object's tensors |
| | `attacked` | u8 | `bool` | it was declared as an attacker this turn (CR 508.1k), not put onto the battlefield attacking (CR 508.4) |
| | `spent_w`, `spent_u`, `spent_b`, `spent_r`, `spent_g` | 5 × u8 | `count` | mana of each color spent to cast the spell that became it; 0 for a permanent not cast |
| `object_subtype` (25) | `obj`, `subtype` | u32, u16 | `handle`, `enum:subtype` | post-layer |
| `object_keyword` (26) | `obj` | u32 | `handle` | register 14; a battlefield permanent's `permanent_keywords`, else the active face's printed keywords |
| | `kw` | u8 | `enum:keyword` | as `card_keyword` |
| | `text` | text | `text` | |
| `object_counter` (27) | `obj` | u32 | `handle` | a permanent's counters, or a card's off the battlefield (`Zone::counters`); also for a face-down object |
| | `kind` | text | `text` | `P1P1`, `LOYALTY`, `TIME`, … |
| | `count` | u16 | `count` | |
| `object_ability` (28): a battlefield permanent's abilities as it has them now (`Permanent::abilities`: activated and mana abilities, keyword-derived triggers), and a printed triggered ability limited each turn (`ActivationLimit$`) once it has triggered, with its count and `exhausted` at the limit; printed abilities of other objects are in `CARDS` | `obj` | u32 | `handle` | |
| | ability | abref | | |
| | `granted` | u8 | `bool` | not in its card's own list, or granted by a static |
| | `activations_turn` | u8 | `count` | limit state |
| | `exhausted` | u8 | `bool` | a limit of its own is reached: each turn, or for as long as the object exists (exhaust, CR 702.177a) |
| `object_provenance` (29) | `obj`, `src` | u32 | `handle` | register 1, coarse: one row per kind of continuous effect the permanent records it is under (a copy; Animate types, keywords, abilities, set P/T; end-of-turn keyword grants and removals; ability removal; static, end-of-turn and counter P/T changes; P/T switch). Static abilities' contributions keep no source, so `src` is 0 |
| | `src_card` | u16 | `card` | none (no source recorded) |
| | `layer` | u8 | `enum:layer` | CR 613 layer / sublayer |
| | `duration` | u8 | `enum:duration` | |
| `object_link` (30) | `obj`, `linked` | u32 | `handle` | linked exiled cards, a card imprinted by exile among them (Chrome Mox) |
| | `kind` | u8 | `enum:link_kind` | exiled with, returnable |
| `combat` (31): one row per blocker of each attacker, attackers then blockers in handle order | `attacker` | u32 | `handle` | |
| | `target_kind` | u8 | `enum:attack_target_kind` | player, planeswalker |
| | `target_player` | seat | | |
| | `target` | u32 | `handle` | planeswalker |
| | `blocker` | u32 | `handle` | 0 = unblocked row |
| | `assigned` | u16 | `count` | damage assigned so far (`Game::combat.damage_assignment`). No damage-assignment order column: the engine keeps no order (CR 510.1c-d assignment is chosen as damage is dealt) |
| `stack` (32): top first | `handle` | u32 | `handle` | |
| | `pos` | u16 | `index` | 0 = top |
| | `kind` | u8 | `enum:stack_kind` | spell, activated, triggered, copy; a card on the stack is a spell (also a resolving permanent spell whose entry is waiting on a choice, Mox Diamond's discard) |
| | `controller` | seat | | |
| | `source` | u32 | `handle` | an ability's source; 0 if gone, and for a spell |
| | `source_card` | u16 | `card` | as `object.card` (a spell's own card) |
| | ability | abref | | |
| | `x` | i16 | `signed` | −1 = no X |
| | `amount` | u16 | `count` | `Ability::amount` |
| | `kicked`, `replicate` | u8 | `count` | |
| | `variant` | u32 | `mask:cast_variant` | the `Spell` flags: flashback, evoke, escape, offspring, impending, warp, back face, fused, and its overlay (overload, adventure, bestow, prepared, face down) |
| | `copy`, `cant_be_countered`, `gift_promised` | u8 | `bool` | |
| | `mana_spent` | u8 | `count` | the mana that left the pool: a pip a delve exile or a tap paid is none |
| | `colors_spent` | u8 | `mask:color` | |
| | `delayed` | u32 | `index` | `delayed_trigger.seq`; 0 none |
| `stack_mode` (33) | `stack`, `mode` | u32, u8 | `handle`, `index` | chosen modes |
| `stack_target` (34) | `stack` | u32 | `handle` | |
| | `order` | u8 | `index` | announcement order: the primary ability's targets, its sub-abilities', then each chosen mode's |
| | `kind` | u8 | `enum:target_kind` | object, player |
| | `target` | u32 | `handle` | |
| | `target_player` | seat | | |
| | `lki` | u16 | `row:lki` | the target left its zone (target 0): its last-known row (register 40); none when it has no last-known information |
| | `share` | u16 | | what it receives of what the object divides (CR 601.2d); 0 when nothing is divided |
| `delayed_trigger` (35): waiting and fired ones, in `seq` order | `seq` | u32 | `index` | registration order |
| | `controller` | seat | | |
| | `state` | u8 | `enum:delayed_state` | waiting, on the stack |
| | `stack`, `creator` | u32 | `handle` | |
| | `creator_card` | u16 | `card` | the live creator's `visible_card_id`, else the card captured at registration |
| | `fire` | u8 | `enum:fire_kind` | `DelayedTriggerLink::FireKind` (UPKEEP … LEAVES_BATTLEFIELD); FIRE_OTHER travels as none |
| | `fire_turn` | u16 | `count` | |
| | `fires_this_turn`, `expires_eot` | u8 | `bool` | |
| | ability | abref | | |
| `delayed_subject` (36): subjects (and a waiting trigger's watched object) the viewer can identify | `trigger` | u16 | `row:delayed_trigger` | |
| | `subject` | u32 | `handle` | |
| | `subject_card` | u16 | `card` | `visible_card_id` |
| `emblem` (37) | `controller` | seat | | |
| | `source_card` | u16 | `card` | |
| `player_effect` (38) | `player` | seat | | |
| | `kind` | u8 | `enum:player_effect_kind` | the player effects of `src/queries/player_effects.h` plus floating triggers, and RESOLVED_EFFECT: an effect object no other kind says, named by `source_card` alone |
| | `colors` | u8 | `mask:color` | hexproof-from colours |
| | `duration` | u8 | `enum:duration` | from the resolved effect behind it (end of turn, until your next turn); STATIC when a battlefield static gives it |
| | `source_card` | u16 | `card` | a floating trigger's creator |
| | `filter` | text | `text` | a cast-as-though-it-had-flash permission's filter |
| `play_permission` (39): per graveyard, exile or looked-at library object and player | `obj` | u32 | `handle` | |
| | `player` | seat | | who may play it |
| | `kind` | u8 | `enum:permission_kind` | one row per source of `play_permissions` (`Permission`) |
| | `resource` | u8 | `enum:permission_resource` | an exile grant's resource (`ImpulseCastPermission`: NORMAL → MANA); MANA otherwise |
| | `amount` | u16 | `count` | an exile grant's amount |
| | `expires_turn` | u8 | `bool` | lapses at this turn's cleanup |
| `shield` (40) | `kind` | u8 | `enum:shield_kind` | combat-damage prevention |
| | `obj` | u32 | `handle` | |
| | `as_source`, `as_target` | u8 | `bool` | |
| `lki` (41): rows for departed stack targets, as referenced: a permanent's characteristics as it left the battlefield (`LastKnownInfo`), else, for an object that left the stack while another stack object could refer to it (a target spell countered or resolved first), its characteristics as it left the stack, as the viewer saw it go (`ObsStackDeparture`, `src/state_format/events.h`; no row when the viewer did not see its identity; register 40) | `card` | u16 | `card` | |
| | `token` | u8 | `bool` | |
| | `controller` | seat | | |
| | `power`, `toughness` | i16 | `signed` | |
| | `types` | u32 | `mask:card_type` | |
| | `colors` | u8 | `mask:color` | (no mana value: `LastKnownInfo` keeps none) |
| `reveal` (42): `reveal_windows_for(viewer)`, in opening order | `handle` | u32 | `handle` | |
| | `zone` | u8 | `enum:zone` | |
| | `owner` | seat | | |
| | `pos` | u16 | `index` | library position from the top; none outside a library |
| | `card` | u16 | `card` | |
| | `window` | u8 | `enum:window_kind` | reveal, look, face-down look (CR 406.3), continuous reveal |
| | `until` | u8 | `enum:window_end` | end of the resolution, its stack object leaving the stack, or only the card leaving its zone |
| | `to` | seat | | none = all players |
| `pending` (43): when a query is parked or a decision source is set | `tag` | u8 | `enum:pending_tag` | `PendingQuery::Tag`; NONE for a blocking prompt and for a step of a mana payment (a row of the interactive payment, a hybrid pip) |
| | `chooser` | seat | | the parked query's chooser, else the deciding seat |
| | `source` | u32 | `handle` | `Game::pending.decision_source` (else the query's) |
| | `source_card` | u16 | `card` | |
| | `source_controller` | seat | | |
| `pending_cost` (44): `pending_cost` (`src/queries/pending_cost.h`): the cast flow at a cost step (not while choosing modes or targets), the activation flow at its X or cost steps, the interactive mana payment loop, an unless-cost offer, a triggered ability's offered cost | `for` | u32 | `handle` | spell or ability paid for (an unless offer: the spell it governs, else the asking source) |
| | `for_card` | u16 | `card` | |
| | `part` | u8 | `enum:cost_part` | mana, alternative, additional, unless, X, activation, a triggered ability's cost |
| | `rem_w`, `rem_u`, `rem_b`, `rem_r`, `rem_g`, `rem_c`, `rem_generic`, `rem_x` | 8 × u8 | `count` | remaining plain pips; `rem_x` = {X} pips still to announce |
| | `rem_life` | u16 | `count` | |
| | `rem_energy`, `rem_discard` | u8 | `count` | an energy cost; cards to discard (unless offers, a spell's escalate, a triggered ability's cost; a whole hand's, the hand's size) |
| | `paid` | u8 | `count` | mana already paid (interactive payment; the pool's mana that applies to the cost counts as paid, and the `rem_*` columns leave it out) |
| `pending_cost_pip` (45) | `cost` | u16 | `row:pending_cost` | remaining non-plain pips |
| | `kind`, `colors`, `generic_alt` | u8 | `enum:pip_kind`, `mask:color`, `count` | as `card_pip` (plus snow) |
| `menu` (46): row = action index of the menu as offered (after the interchangeable-choice collapse); only in the deciding seat's and the omniscient frame | `category` | u8 | `enum:action_category` | |
| | `type` | u8 | `enum:action_type` | |
| | `source`, `target` | u32 | `handle` | |
| | `source_card` | u16 | `card` | `visible_card_id` of the source, or for a `card_public` choice the card it makes public (a name in a name-a-card menu) |
| | `source_player`, `target_player` | seat | | when the source / target is a player |
| | ability | abref | | an ACTIVATE_ABILITY action's ability |
| | `ordinal` | i16 | `signed` | `option_ordinal`; −1 = n/a |
| | `variant` | u32 | `mask:cast_variant` | the action's flags: flashback, escape, offspring, back face (cast or play), alt cost, impulse, companion, suspend, cancel, without paying its mana cost (a free-cast static), fused, and its overlay (overload, adventure, bestow, prepared, face down) |
| | `card_public` | u8 | `bool` | |
| | `description` | text | `text` | covered by the leak checks (register 12). For a seat, an entry whose source or target's identity the viewer cannot see (and that is not `card_public`) gets `(hidden card)` instead |
| `match` (47) | `game` | u8 | `count` | as `header.game` |
| | `plays_first` | seat | | this game, or the upcoming one while sideboarding |
| | `sideboard_phase` | u8 | `bool` | |
| | `sb_swaps` | u8 | `count` | |
| | `sb_drift` | i8 | `signed` | |
| | `closed_decklist` | u8 | `bool` | |
| `own_deck` (48): the viewer's current 75 (the omniscient frame: both players') | `player` | seat | | |
| | `card` | u16 | `card` | as `decklist_card.card` |
| | `main`, `side` | u8 | `count` | |
| `pool_mana` (49): floating mana with a restriction or a rider (CR 106.6), both players', the viewer's first, each in the pool's order; a pool of plain mana has no row | `player` | seat | | |
| | `color` | u8 | `enum:color` | |
| | `count` | u16 | `count` | units alike in color, restriction and rider |
| | `restriction` | u8 | `enum:mana_restriction` | what it may be spent on; NONE for mana that only carries a rider |
| | `chosen_type` | text | `text` | the type a CHOSEN_TYPE_CREATURE restriction names, as its source stood when the mana was made |
| | `activated` | u8 | `bool` | the restriction also allows activating abilities |
| | `activated_on_battlefield` | u8 | `bool` | of permanents only |
| | `cant_be_countered` | u8 | `bool` | rider: a spell it is spent on can't be countered |
| | `keyword` | u8 | `enum:keyword` | rider: a spell it is spent on gains it, none without one |
| | `keyword_valid` | text | `text` | the spells that keyword goes to |
| | `keyword_until` | u8 | `enum:duration` | how long the spell has it |

The designation columns (`day_night`; `monarch` … `speed_triggered`; `chosen_name` … `attacked`) are one block per table, declared with the state they read in `src/state_format/designations.def`: a new one is a row there, at its block's end. A table's order in this appendix is not always the block's: the row order of that file is the columns'.

A row's characteristic columns (`power` … `mana_value`) and its `object_subtype`, `object_keyword`,
`object_ability`, `object_provenance` and `object_link` rows are filled only when the viewer can see
them (`characteristics_visible`): whatever identity they can see, and a face-down permanent or spell,
which every viewer sees as what it is face down: a 2/2 with what effects gave it. `transformed`,
`copy_of`, `chosen_name`, `chosen_type`, `chosen_color` and `spent_*` say something of the card (a
permanent keeps the mana spent on it only if its card reads that), so they follow the identity. A
face-down exiled card shows its zone, seats, position, public status and counters. Phased-out
permanents are included and flagged (the encoder reads `Permanent::is_phased_out` directly). No
`can_attack` column: the engine computes it only in its legal-action enumeration.

### A.6 Events and knowledge (`STATE` frame)

The encoder emits the event tables (`src/state_format/events.*`, §5). The knowledge tables are
appended to each seat's STATE frame by that writer's knowledge instance
(`src/state_format/knowledge_feed.h`, §7): `k_meta` always (one row), the others at level
`expert` (`knowledge_store.md` §6). They have no text column, so the frame's string pool is the
encoder's.

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `event` (80) | `seq` | u32 | `index` | section 5 |
| | `kind` | u8 | `enum:event_kind` | |
| | `player` | seat | | drawer, looker, searcher, shuffler, starting player, winner (none = draw) |
| | `zone`, `to_zone` | u8 | `enum:zone` | from / looked / searched zone; destination. 0xFF = `none` (created, ceased) |
| | `zone_owner`, `to_owner` | seat | | |
| | `count` | u16 | `count` | cards drawn (1) / looked at / found; batch size; 1 for a `zone_move` |
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
| | `from` | libpos | | from a library; for `reveal_start`, `look`, `reveal_end` the card's library position |
| | `to` | libpos | | into a library; anchor 0xFF, `to_pos` 0xFFFF = none (not into a library, or in a batch: the event's `batch`) |
| | `face_down`, `token` | u8 | `bool` | token = token or spell copy |
| | `hand_subset` | u8 | `enum:hand_subset` | `zone_move` from a hand: any, drawn_this_turn; 0xFF for any other card |
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

### A.7 Format enums

The format's own enums, X-macro lists in `schema.h` (each C++ enum and its string table from one
list):

| Enum | Values |
|---|---|
| `frame_kind` | SCHEMA, CARDS, DECKLISTS, STATE |
| `viewer` | A, B, OMNISCIENT |
| `storage` | U8, U16, U32, U64, I8, I16, I32, I64, F32, TEXT |
| `hand_subset` | ANY, DRAWN_THIS_TURN |
| `event_kind` | ZONE_MOVE, DRAW, LIBRARY_PLACE, REVEAL_START, REVEAL_END, LOOK, SHUFFLE, REORDER, SEARCH, CAST_FROM_HIDDEN, PLAY_FROM_HIDDEN, FACE_DOWN, FACE_UP, GAME_START, GAME_END, MATCH_END |
| `lib_anchor` | TOP, BOTTOM |
| `face` | FRONT, BACK |
| `layout` | NORMAL, TRANSFORM, MODAL_DFC, SPLIT, ADVENTURE, PREPARE |
| `ability_kind` | SPELL, ACTIVATED, TRIGGERED, STATIC, REPLACEMENT, CHAPTER, OPENING_HAND |
| `timing` | ANY, SORCERY, INSTANT |
| `pip_kind` | HYBRID, TWOBRID, PHYREXIAN, SNOW |
| `stack_kind` | SPELL, ACTIVATED, TRIGGERED, COPY |
| `target_kind` | OBJECT, PLAYER |
| `attack_target_kind` | PLAYER, PLANESWALKER |
| `delayed_state` | WAITING, ON_STACK |
| `cost_part` | MANA, ALTERNATIVE, ADDITIONAL, UNLESS, X, ACTIVATION, TRIGGER |
| `layer` | COPY, CONTROL, TEXT, TYPE, COLOR, ABILITY, PT_CDA, PT_SET, PT_MODIFY, PT_COUNTERS, PT_SWITCH |
| `duration` | STATIC, END_OF_TURN, UNTIL_YOUR_NEXT_TURN, UNTIL_HOST_LEAVES, PERMANENT |
| `link_kind` | EXILED_WITH, RETURNABLE |
| `player_effect_kind` | PROTECTION_FROM_EVERYTHING, CANT_GAIN_LIFE, HEXPROOF_FROM, SPELLS_CANT_BE_COUNTERED, CAST_AS_FLASH, SORCERY_SPEED_ONLY, FLOATING_TRIGGER, RESOLVED_EFFECT |
| `permission_kind` | FLASHBACK, ESCAPE, GRAVEYARD_CAST, GRAVEYARD_LAND, EXILE_GRANT, PREPARED, GRAVEYARD_SELF, LIBRARY (the `Permission` bits, in bit order) |
| `permission_resource` | MANA, ENERGY, LIFE, FREE |
| `shield_kind` | COMBAT_DAMAGE |
| `cast_variant` | FLASHBACK, EVOKE, ESCAPE, OFFSPRING, IMPENDING, WARP, BACK_FACE, ALT_COST, IMPULSE, COMPANION, SUSPEND, CANCEL, WITHOUT_MANA_COST, OVERLOAD, FUSED, ADVENTURE, BESTOW, PREPARED, GRAVEYARD_REPLACED (what cast it sends it elsewhere if it would go to a graveyard), BARGAINED, FACE_DOWN |
| `knowledge_level` | OFF, EXPERT |
| `pool_kind` | HAND, LIBRARY_RANGE, SIDEBOARD, FACE_DOWN, SET_ASIDE |
| `amount_kind` | NONE, FIXED, VARIABLE |
| `mana_restriction` | NONE, CHOSEN_TYPE_CREATURE, CREATURE, COLORLESS_ELDRAZI, NO_NONARTIFACT_SPELLS |
| `target_class` | CREATURE, PLANESWALKER, ARTIFACT, ENCHANTMENT, LAND, BATTLE, PERMANENT_OTHER, PLAYER, SPELL, ABILITY, CARD_GRAVEYARD, CARD_OTHER |
| `part_kind` | SUB, MODE |

Engine enums used (string tables built by `src/state_format/enum_names.cpp` from the
engine's own definitions, each a name function that switches over every enumerator with
`-Wswitch` as an error, so an enumerator without a name fails the build): `Step` (`step`),
`ActionCategory` (`action_category`), `ActionType` (`action_type`), `Zone::ZoneValue`
(`zone`), `Zone::Ownership` (`seat`: UNKNOWN, PLAYER_A, PLAYER_B), `Colors` (`color`, NO_COLOR
included), `MandatoryChoice` (`mandatory_choice`), `Game::DayNight` (`day_night`),
`PendingQuery::Tag` (`pending_tag`), `Game::RevealWindow::Kind` (`window_kind`: REVEAL, LOOK,
FACE_DOWN_LOOK, CONTINUOUS_REVEAL, STANDING_LOOK), `Game::RevealWindow::End` (`window_end`: RESOLUTION,
STACK_OBJECT, LEAVES_ZONE), `DelayedTriggerLink::FireKind` (`fire_kind`: FIRE_UPKEEP …
FIRE_LEAVES_BATTLEFIELD), `EffectKind` (`effect_kind`: None, then the categories of
`effect_kinds.def`, included as an X-macro). The type lists `all_types`, `all_subtypes`,
`all_supertypes` (`src/constants.cpp`, sorted) give `card_type`, `subtype`, `supertype`; the
parser's keyword table (`kKeywordTable`, `src/parse.cpp`, via `parser_keyword_names`) followed
by the observation's evergreen keywords it lacks (`EVERGREEN_KEYWORDS`, `src/state_format/enum_names.cpp`) gives
`keyword`. The SCHEMA frame lists the format's enums first, then these.
