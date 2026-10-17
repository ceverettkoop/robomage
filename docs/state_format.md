# RMSTATE: the engine's state format (draft spec)

Status: **step-1a spec, decisions closed** (register in section 11). Companions:
[`state_format_agent_map.md`](state_format_agent_map.md) (agent read map) and
[`knowledge_store.md`](knowledge_store.md) (recall/inference sidecar). Implemented so far:
visibility and reveal windows (`src/queries/visibility.*`), handles
(`src/state_format/handles.*`), the schema and the encoder for the snapshot and once-per-match
frames (`src/state_format/schema.*`, `enum_names.*`, `encode.*`) with `--rmstate-dump`
(`dump.*`), and the observation events (`src/state_format/events.*`). The decoders and the
knowledge sidecar are not yet.

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
  `Game::revealed_in_library` to hands and looks. Implemented in `src/queries/visibility.{h,cpp}`
  (`identity_visible`, `order_visible`, `reveal_windows_for`) over `Game::reveal_windows`;
  `--trace-visibility` narrates each window opening and closing.
- A window ends when the effect finishes with the card, when the card changes zones, or when
  its library is shuffled or reordered (CR 701.20d). "The effect finishes" is the end of the
  resolution that opened it (a mid-resolution decision is still inside it), so a card put back
  in a library by Ponder or a scry stops being looked at as it is put back.
- A search is a look at every card in the searched hidden zones (CR 701.23a).
- A miracle card stays revealed until its trigger leaves the stack (CR 701.20a, 702.94b).
- Continuous reveals (e.g. "play with the top card of your library revealed") are windows that
  last as long as the effect. No card has one today except the chosen companion, revealed
  outside the game while it stays there.
- A card exiled face down gets a face-down look window for each player who could see it in its
  hidden zone just before (its owner's hand, or a look or reveal window). A card exiled face
  down from a public zone gets none (CR 406.3 covers a card looked at and then exiled). The
  window ends when the card leaves exile or is turned face up (no engine effect shuffles a
  pile in exile).
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

- `CARDS`: the card table for every implemented card and token in either registered list (the
  whole vocab under `--closed-decklist`), plus tokens they can make. Each entry has the vocab id,
  name, uid, printed characteristics, the **oracle text** (always: `CardData::oracle_text`,
  parsed from the script's `Oracle:` line; token scripts carry one too), an ability list
  (category, costs, timing, script line text) and, optionally, the raw script.
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
- Events are recorded only while someone consumes them (a frame writer such as `--rmstate-dump`,
  or `--trace-events`); otherwise nothing is queued and play, the RNG and BQUERY are the same
  either way.

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
| `draw` | player, count, opening-hand flag (opening-hand and mulligan draws are not draws for "drawn this turn"); per card: handle after and card id for the drawer; for another viewer, handle before and card id only if the card was in that viewer's live window. Each card drawn is its own `draw` event (count 1, so a draw link's index is 0), always from top 0 |
| `library_place` | closes an order-hidden batch: batch, library owner, range (anchor, pos, count), order known (false) |
| `reveal_start` / `reveal_end` | cards (handles, zone, owner, library positions, card ids), revealed to whom, source effect; per card, a draw link (`seq` of the `draw` event and the card's index in it) when the reveal shows a card as the one just drawn (e.g. Miracle). `reveal_end` also closes a look window. A window that ends because its card moved has no `reveal_end`; the move ends it |
| `look` | looker, zone/owner, count; per card: library position; identities and handles only for the looker. A search of another player's hidden zone is also a `look` at all of it. A non-looker gets each card's position when the look is at known positions (the top N, the whole library), and position unknown when the looker picked the card out by identity (a search's find, kept in view after the shuffle) |
| `shuffle` | library owner |
| `reorder` | library owner, range (anchor, pos, count), whether this viewer saw the result and, if so, the card at each position. No engine site today: rearrangements (Ponder, scry, surveil) are library → library `zone_move`s |
| `search` | searcher, zone searched (library, sideboard, graveyard, hand), its owner, cards found. The cards that leave follow as `zone_move`s, a shuffle as `shuffle`. Emitted once the pick is made, per searched hidden zone (a library, a sideboard, another player's hand; a pick from the searcher's own hand, such as a put-back, is no search), with the cards found in that zone; the `look` at the searched zone comes first |
| `cast_from_hidden` / `play_from_hidden` | a `zone_move` (same fields) from a hidden zone to the stack or battlefield; the card becomes identifiable here. A reversed proposal (CR 733.1) returns it with a `zone_move` that keeps its identity |
| `face_down` / `face_up` | handle; identity at face-up. Exiling face down is a `zone_move` with the face-down flag; `face_down` is a public object turned face down where it is |
| `game_start` | game number, starting player; per player library and sideboard counts. Emitted after the libraries are built and shuffled (after sideboarding in games 2–3), before the opening draws. Opening hands and mulligans arrive as `draw`, `zone_move` (hand → library, London bottoming to the bottom) and `shuffle` |
| `game_end` / `match_end` | game number, winner (or draw); match wins per player. `game_end` is queued when the game's loop ends, after the game's last events; `match_end` after the last `game_end` (a single game is a match of one game) |

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

**Implementation** (`src/state_format/events.{h,cpp}`; queues in `Game::obs_events`, one per
viewer: A, B and omniscient). The engine calls `obs_*` hooks at its choke points: `Orderer`'s
zone move (`move_object`, behind `add_to_zone`, `put_in_library_at_depth` and `perform_draw`), the
cast move and its rewind, `shuffle_library`, object creation and cessation, the reveal-window opens
and closes in `src/queries/visibility.cpp`, `search_zones`, `SetState`'s face turns and the game
loop's boundaries. Each hook asks the visibility accessors once per viewer and queues that
viewer's projected `ObsEvent`, which holds only those answers (no entity id). The next STATE frame
built for the viewer drains its queue into `event` / `event_card` / `event_player`.
`--trace-events` narrates each event as it is queued (`EVENT:` lines).
- **Lifetime.** The queues and seq counters are carried over the per-game `Game` reset into the
  next game of a bo3 match (the first game of a match starts fresh), so events not yet delivered
  reach the viewer's next frame: a sideboarding frame, or the next game's first frame. Nothing
  follows the final `game_end` / `match_end`, so they stay queued unless a writer emits a frame at
  match end.
- **Zone moves.** A move of a card, token or spell copy (an ability's stack object is not
  reported). Kind `cast_from_hidden` for a move from a hidden zone (hand, library, sideboard, face
  down in exile) to the stack, `play_from_hidden` to the battlefield, else `zone_move`; count 1.
  `handle_before` is the handle the viewer held for the object (0 if none: a handle is never
  issued just to be retired), `handle_after` the new object's handle for every viewer who can
  identify it; the card id is given when the viewer saw the identity before or after the move.
  Zone `none` (0xFF) as the origin: a token (built in its controller's hand zone and moved onto the
  battlefield), a spell copy created on the stack, a test-harness preset (after `game_start`); as
  the destination, a token or copy ceasing to exist.
- **Library positions.** A destination is exact (the instruction says top, bottom or a depth). A
  source is exact for a viewer who sees the card's identity, and when nobody has a window on the
  card (a draw, a mill: the instruction names the position). A card in a window another player
  has is one of the cards in such windows in its library (`library_position_seen_by`,
  `src/queries/visibility.h`): the span covers their positions, which the viewer knows because
  those windows opened on known positions and every move since was reported; a card in a window
  opened on it by identity (`look_at_found_card`) is anywhere. "Anywhere" is anchor top, pos 0,
  span 0; "not in a library" is anchor 0xFF, pos 0xFFFF, span 0.
- **Batches.** An instruction that puts several cards into a library in an order some seat did not
  see opens a batch (`ObsLibraryBatch`): Dig's unchosen rest, `ChangeZoneAll` into a library, and
  `DigUntil`'s revealed cards. Every seat other than the player who orders them is batched (both
  seats for a random order); the omniscient viewer never is. For a batched viewer every card the
  instruction puts into that library carries the batch and no destination, identifiable or not,
  and the batch's `library_place` gives the range they occupy afterwards.
- **Reveals.** Each opening call reports its new windows as one `reveal_start` (or `look`) per
  zone, to every viewer; a window that was already open is not reported again. `reveal_end` is
  reported, per zone and audience to each viewer the windows were shown to, for windows that end
  at a resolution's end or a shuffle; one that ends because its card moved or was turned face up
  has none. `reveal_start` / `look` carry the card's handle in `handle_after`, `reveal_end` in
  `handle_before`, `face_up` / `face_down` in both.
- **Source.** The object resolving when the event happens: a spell, or the source of a resolving
  ability while it is still that object (handle and card id, else its card id only); none outside
  a resolution.
- **Turn.** The game's turn counter (`Game::turn_state.turn`): 0 during the pregame and the first
  turn.
- `reorder` has no engine site (rearrangements are library → library `zone_move`s), and
  `face_down` only `SetState`'s `TurnFaceDown`, which no vocab card uses. The search server's
  determinizer moves hidden cards without events; events are not meant to be consumed under
  `--search-server`.

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
  must be distinguishable); only their identity is withheld. The same holds for a card exiled
  face down; turning a public object face up or down keeps its handle (it is the same object).
  Phased-out permanents are still on the battlefield and keep theirs (they are included, flagged).
- The handle table is per game, lives in `Game` (snapshot-covered), and resets at game start.

**Implementation** (`src/state_format/handles.{h,cpp}`, tables in `Game::handle_tables`: A, B and
an omniscient table of its own). `individually_identifiable` (`src/queries/visibility.h`) is the
predicate; `--trace-handles` prints each viewer's table at every decision (`HANDLES:`) and each
issue, link and retirement (`HANDLE:`).
- **Keyed by object.** An entry holds the object's exact identity stamp (`Zone::obj_gen` and the
  id's issue count), so a zone change, a token ceasing to exist or a reissued id ends it.
- **Retired where identifiability ends**, not only when a frame is built: at every zone change
  (`Orderer::add_to_zone`, the cast move and its rewind: the rewound card is its old object again,
  but its handles retired as it was cast, so it returns with a fresh one linked from its stack
  handle), as the reveal window that showed a card closes (resolution end, shuffle, the close hook
  in `visibility.cpp`), and as an ability or spell copy leaves the stack. Any lookup and every
  canonical issue retires stale entries too, as a backstop.
- **Links.** A zone change issues the new object's handle at once, in move order, for every viewer
  who held one before and can still identify it; `handles_begin_move` / `handles_end_move` return
  both handles per viewer (`HandleMove`) for the move event.
- **Issue order.** Handles are issued lazily: by links in move order, by `handle_of` in its
  caller's (event) order, and at frame start by `issue_handles_canonical` in
  `canonical_object_order`. That order uses only what the viewer sees: zone (battlefield, stack,
  graveyards, exile, hands, libraries, sideboards); seat, the viewer's own first (controller on
  the battlefield, owner elsewhere, none on the stack); position where the zone's order is visible
  (stack, graveyard and exile from the top; a library card in a window by its position); name when
  the identity is visible (a face-down object sorts first); battlefield timestamp; arrival order
  where the viewer saw the arrival (a public zone or their own hand).
- **With an event consumer.** With a frame writer, `--selfcheck-state` or `--trace-events` on,
  an object's handle is usually issued by the event that made it identifiable, so handle numbers
  follow the event stream the viewer receives; with none, every handle is issued canonically at
  the next frame. Both are functions of what the viewer saw: an event that shows several cards
  lists them in an order the viewer can derive (`reveal_cards` / `look_at_cards`: owner, zone,
  library position, then name, copies of one card tied by entity id as below), and draws and moves
  issue in the order the viewer sees them happen. The deck-order scenarios
  (`state_format_handles_deck_order_independent*.json`) build the same game from decklists in
  another order and require the viewer who cannot see the reordered cards to get identical
  handles, with and without an event consumer (and identical events under `--trace-events`).
- **Ties.** Objects equal on every key are copies of one card in the same unordered place (two
  Islands in a revealed hand, two copies in a sideboard), with identical rows in everything the
  viewer sees. They are tied by entity id: swapping them changes nothing visible, and among copies
  of one card the id order is fixed at deck build and unrelated to where the shuffle put each, so
  to their history. Arrival order is deliberately not used for an opponent's hand: it would say
  which copy was drawn when. A face-down object never reaches this tie (its zone order,
  timestamp or arrival is unique). One caveat sits in the engine, not here: permanents that
  enter together get their battlefield timestamps in entity order (the state-based pass that
  builds them walks entity ids). That is harmless while their identities are public; it would
  leak for face-down permanents entering together, which no vocab card makes.

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
  `--machine-format rmstate` (implies `--machine`; `src/state_format/stream.h`) is this
  protocol:
  - stdout carries only `RMSTATE` records. The engine moves stdout's descriptor aside for them
    at startup and points stdout at stderr, so everything else it prints (narrative under
    `--narrative`, `GAME_RESULT` / `MATCH_RESULT`, a `--player` seat's CLI prompt) goes to
    stderr. Passive `BSTATE` frames are not written; `--search-server` is refused.
  - The menu is unbounded. `--closed-decklist`'s name-a-card menus offer every vocab card
    (register 11a), more than BQUERY's `MAX_ACTIONS`, so a machine run with `--closed-decklist`
    is refused unless it uses this format (BQUERY's state vector shows the opponent's registered
    list anyway). A `--player` seat's CLI prompt is capped the same way: a debug build stops on
    such a menu there.
  - At each decision the deciding seat (the sideboarding player between games, else the
    priority holder) gets, the first time it decides in the match, its `SCHEMA`, `CARDS` and
    `DECKLISTS` frames, then its `STATE` frame with the menu. Its events are those queued for it
    since its previous frame, so a seat that does not decide for a while gets them all in its
    next frame.
  - Menu row *i* is the action reply *i* selects: the menu as offered, after the
    interchangeable-choice collapse (register 17), the same list BQUERY indexes. `-1` is the
    last row. End of input or an unparseable line reads as `-1`, as for BQUERY.
  - **Terminal frame.** Once the match is over (after its `match_end` event) each seat, A then B,
    gets one more `STATE` frame with `header.terminal = 1`, no menu and no reply: it carries the
    seat's last events (`game_end`, `match_end`). The engine then exits. `--rmstate-dump` ends
    with the same terminal frame for each of its three viewers.
  - The frames are the ones `--rmstate-dump` writes for that seat at that decision, apart from
    the events: each writer drains its own copy of the queues. At every decision, while either
    writer is on, every viewer's handles are issued canonically before any frame is built, so
    handle numbers (and so the stream's bytes) do not depend on whether a dump is written.
- **Versioning.** Every frame header carries the format version and schema hash; a decoder
  refuses a mismatch.
- **Cadence.** A frame at each of the viewer's decisions. Pushed passive frames (a spectator or
  GUI stream) are deferred; batching events into the next frame is enough for agents.
- **Debug view.** `--state-dump json`, rendered from the decoded binary (not a second encoder):
  every record `--rmstate-dump FILE` writes is decoded and written as one JSON line into
  `FILE.jsonl`, and `--rmstate-read FILE --state-dump json` prints a stream's lines on stdout.
  The renderer is `src/state_format/json.h` (rules there); `train/rmstate.py` renders the same
  bytes. One object per frame, no whitespace, keys in a fixed order:
  `{"frame":…,"viewer":…,"version":…,"schema_hash":"<16 hex>","tables":{"<table>":[{row},…],…}}`
  (plus `"skipped":[ids]` when a block's id is unknown). Tables in ascending id with the string
  pool left out; columns in schema order. Text renders as a string; `enum:<e>` as its name
  (`null` for all ones); `mask:<e>` as the list of set bits' names; `bool` as `true`/`false`;
  `card`, `index`, `row:` as a number or `null` for all ones; everything else (handles
  included) as a number.
- **Decoding.** `src/state_format/decode.h` (standard library and `schema.h` only) builds a
  schema from a `SCHEMA` frame (reading tables 1-4 with `schema.h`'s layout, which the frame must
  repeat, and checking its hash) and decodes any frame against it into tables of typed values;
  `encode_frame` re-encodes them with the frame writer, byte-identical for the engine's frames.
  It refuses a version or schema-hash mismatch, an unknown frame kind or viewer, a known table
  with another row width or in a frame of another kind, and malformed framing; it skips unknown
  table ids. `--rmstate-read FILE` (`src/state_format/recode.h`) decodes, re-encodes and compares
  every record of a stream (`ERROR:` lines, exit status 1 on any difference).
  `train/rmstate.py` is the Python decoder, built from the `SCHEMA` frame the same way (its
  bootstrap is the frame and block headers and the layout of tables 1-4).

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

**Implementation** (`--selfcheck-state`, `src/state_format/selfcheck.{h,cpp}`). At every decision
the self-check builds every viewer's STATE frame (A, B, omniscient) from a queue copy of its own
(`ObsConsumer::SELFCHECK`), so it runs with or without a frame writer; failures print `ERROR:`
lines (`selfcheck-state <check>: viewer …, decision … (game, turn, step): <first differing table,
row, card and column>`), and a `SELFCHECK-STATE: decisions=… frames=… events=… permutations=…
snapshots=… errors=…` summary goes to stderr at match end. Play, the RNG, BQUERY and the
`--rmstate-dump` bytes are the same with it on. It is off under `--search-server` and for the
in-process actor.
- **Hidden, decided independently.** Both leak checks decide what a seat cannot see from the
  table in section 3 directly (zone, owner, controller, face-down state, and the seat's windows in
  `Game::reveal_windows`), not through `queries/visibility.h`, so a visibility rule that wrongly
  shows a card is caught as well as an encoder read that bypasses the rules.
- **roundtrip:** every frame (and once per match each viewer's `SCHEMA`, `CARDS`, `DECKLISTS`)
  through `decode_frame` → `encode_frame`.
- **scramble** (the event-leak check, plus the snapshot at every decision): each seat's frame is
  encoded a second time inside `ScrambledIdentities`, which gives every card hidden from the seat
  another identity. The hidden cards' `CardData` are rotated among them (sorted by name, shifted
  by the largest copy count, so each holds another name's whenever no name has half the copies),
  by remapping their component slots (`Coordinator::PermuteComponents`: nothing is copied, and a
  reference held to a component still names the same data); a hidden object without `CardData`
  gets another id from `card_identity`'s test-mode scrambler. The bytes must match. Every event
  is built twice the same way as it is constructed (`emit` in `events.cpp`, and the move capture
  in `obs_begin_move`), and the two builds' rows (`encode_events_frame`) must match. A menu entry's
  card made public by the choice itself (`card_is_public`) keeps its identity.
- **permutation** (the snapshot-leak check, sampled every `--selfcheck-state-sample N`th decision,
  default 8): for each seat, the opponent's hidden hand, library and sideboard cards are dealt at
  random among those cards' places (zone and library position: entity ids move, as in the search
  server's determinizer), the seat's own hidden library cards among their positions, and then the
  identities of all cards hidden from the seat are shuffled per owner (a card in a window shown
  only to the other player keeps its place but not its identity; a face-down exiled card keeps
  its place, and the objects that refer to it, with another identity). Every card in a live
  window keeps its place. Owners, zone sizes, identity stamps and the handle tables stay as they
  were; the seat's frame must be the same bytes. Everything is put back exactly (saved `Zone`s,
  the inverse slot permutation, the seat's handle table), without a snapshot restore, so it can
  run at any decision, including a blocking prompt in the middle of a flow.
- **complete:** object handles unique and nonzero; every object the viewer can identify has a
  row, and every object of a public zone (and the seat's own hand and sideboard; everything for
  the omniscient viewer) is counted per zone and owner against the object rows; the player rows'
  zone counts against the zones; the reveal table against the viewer's windows; every `handle`
  column of a snapshot table names an object row (event tables exempt); every `row:` reference in
  range; the menu has one row per action as offered for the deciding seat and the omniscient
  viewer, none otherwise.
- **snapshot** (sampled): at a loop-safe decision outside sideboarding the game is saved
  (`snapshot_save`, with the continuous-effects registry copied); at the first loop-safe decision
  at least 3 decisions later the current state is saved, the earlier one restored, every viewer's
  frame encoded there (its menu, decider, priority-window flag, and the events it drained) and
  compared with the saved decision's bytes, and the current state restored. Both restores are
  exact (`snapshot_restore_exact`, registry put back from the copies): the search server's restore
  re-derives the continuous effects instead, which can change characteristics at a decision where
  the derived ones lag the state (`docs/rules_deviations.md` 2.3).
- **names** (every frame of a seat): the text columns of a STATE frame (the menu's `description`
  above all, and keyword texts, counter kinds and player-effect filters) may name only cards the
  frame itself shows. Every vocab card name a text cell contains as whole words (the longest name
  at each word start, case-sensitive, so `Island` is not found in `Islandwalk` and `Force` not in
  `Force of Will`) must be the name of a card some `card` column of the frame holds (objects,
  stack, LKI, events, the menu's `source_card`, chosen names, …; the seat's `own_deck` rows do
  not count: they say which cards the seat has, not where), a face of that card, or a card name
  that card's oracle text mentions (a fetch land's land types). The descriptions are built before
  encoding, so the scramble and the permutation, which change the world and not a string already
  built, cannot see what they name; this check can. It cannot tell copies apart: a description
  naming a hidden card whose name the frame shows elsewhere passes. Proved with a temporary
  bypass that appended the deciding seat's top library card's name to each first menu entry: the
  check reported it in every fuzz game, the scramble and permutation checks never.
- **Bypasses of `card_identity`.** Three identity reads do not go through `visible_card_id`. The
  card a `card_is_public` menu choice makes public (`public_card_id`) is safe where the choice
  itself is public: the name-a-card menus (Cabal Therapy, Disruptor Flute, Petrified Hamlet) take
  their candidates from the registered lists (public under open decklists) or, under
  `--closed-decklist`, from the whole vocab, with no object behind a candidate
  (`LegalAction::named_card`; register 11a). Captured public ids (last-known information of an
  object that left the battlefield, a delayed or floating trigger's creator) were public when
  captured. A zone-less sideboard template card is shown only to the sideboarding player. The
  menu's `description` strings are built by the engine before the encoder sees them; the encoder
  redacts an entry whose source or target is hidden (`(hidden card)`), and the names check covers
  the rest.

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
| 11a | Name-a-card candidate set follows decision 11. With open decklists (default) it stays as today: the distinct vocab cards of the relevant registered list, filtered by `ValidCards$`. The list is explicit information, so this no longer leaks, but it remains the documented CR 201.4 deviation. Under `--closed-decklist`: every vocab card passing `ValidCards$`, in name order, uncapped (`build_name_card_choices`); that menu is wider than BQUERY's `MAX_ACTIONS`, so a machine run with `--closed-decklist` needs `--machine-format rmstate` (refused otherwise) | decided |
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

The column list coded in `src/state_format/schema.cpp` (`table_defs`); the `SCHEMA` frame is
generated from it. Anything changed during implementation is updated here (step-1 exit). The
encoder is `src/state_format/encode.cpp`; every row it writes is checked against these
definitions in debug builds (storage type per value, column count per row, row width per table).

### A.1 Conventions

| Item | Rule |
|---|---|
| Storage | little-endian `u8 u16 u32 u64 i8 i16 i32 f32`; rows packed, no padding; `text` = u32 offset into the frame's string pool |
| String pool | table `strings` (id 1; one column `byte`, u8, tag `text`; one row per byte): UTF-8, each string NUL-terminated; offset 0 is the empty string (the pool starts with a NUL). Each distinct string is stored once, at its first use in table, row, column order, so re-encoding is byte-identical. The pool is always present |
| None values | handle 0; text 0 (empty); card, `index`, `row:` and nullable enum columns: all ones of their width (0xFF, 0xFFFF, …) |
| Enums | the C++ value (`Step`, `ActionCategory`, `Zone::ZoneValue`, …), string tables in `SCHEMA`; enums that have no C++ type yet are listed in A.7 |
| Seat pair | type `seat` below = two u8 columns: `<name>` (`enum:seat` = `Zone::Ownership`: 0 none, 1 A, 2 B; range 0-2) and `<name>_rel` (`player-rel`: 0 none, 1 self, 2 opponent; for `omniscient`, self = A; range 0-2). Register 31 |
| Library position | type `libpos` = `<name>_anchor` u8 `enum:lib_anchor` (top, bottom) + `<name>_pos` u16 `index` (0 = the end card) + `<name>_span` u16 `count` (1 exact, n = one of n from pos, 0 unknown); section 5 |
| Ability reference | type `abref` = `ab_card` u16 `card` + `ab_index` u16 `index` (row of that card in `card_ability`) + `ab_key` `text`. Registers 2, 25. `ab_key` is `AbilityDef::ref_key`: for a keyword-built ability the keyword table entry's name, plus `:<label>` when the ability's label differs (`Equip`, `Ninjutsu`, `Reconfigure`, `Reconfigure:unattach`); for an engine-built definition its `keyed_ability_def` key (`subtype_mana:1`, `keyword:Prowess`); empty for a plain script ability. An ability that is not in its card's list (granted, engine-built) has `ab_card` and `ab_index` none; its key is `ref_key`, else its keyword label, else the keyword it derives from, else `LandType` for a land-type mana ability. The lookup reads the card's loaded definition by card id, never an object the viewer cannot see |
| Parent links | a child row names its parent by handle when the parent has one (`object`, `stack`), else by the parent's row index in the same frame (`row:<table>`) |
| Masks | bit i = value i of the enum (`mask:color` over `Colors`, `mask:card_type` over the engine's type list, …) |
| Ranges | `bool` columns 0-1; seat pairs 0-2; no other column carries a range yet (enums are bounded by their string tables) |

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
| Order | blocks in ascending table id (so `strings`, id 1, comes first); a table with no rows is omitted; `n_tables` counts the blocks present |
| Stream | `--rmstate-dump FILE` writes `RMSTATE <nbytes>\n<payload>` records: per viewer (A, B, omniscient, in that order), at its first decision of the match `SCHEMA`, `CARDS`, `DECKLISTS`, then a `STATE` frame at every decision, and at match end a terminal `STATE` frame (`header.terminal`). `%p` in FILE is replaced by the process id. `--machine-format rmstate` writes the deciding seat's records on stdout (section 9) |
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
| | `script` | text | `text` | raw script; empty unless requested |
| `card_subtype` (11) | `card`, `face`, `subtype` | u16, u8, u16 | `card`, `enum:face`, `enum:subtype` | |
| `card_keyword` (12) | `card`, `face` | u16, u8 | `card`, `enum:face` | |
| | `kw` | u8 | `enum:keyword` | the longest `keyword` value the text starts with, followed by its end, `:` or a space (`Ward:2` → Ward, `Protection from blue` → Protection); 0xFF = none |
| | `text` | text | `text` | the keyword as held, e.g. `Ward:2` |
| `card_ability` (13) | `card`, `index` | u16, u16 | `card`, `index` | index: per card over both faces, front then back; per face its spell, activated and triggered abilities (`CardData::abilities` order: A: lines, T: lines, then keyword-built), gift, chapter and opening-hand abilities, then statics (S:), then replacements (R:) (`abref.ab_index`). A token's list: its abilities, statics, replacements |
| | `face` | u8 | `enum:face` | |
| | `ability_kind` | u8 | `enum:ability_kind` | spell, activated, triggered, static, replacement, chapter, opening hand |
| | `category` | u8 | `enum:effect_kind` | `effect_kinds.def`; 0xFF for statics and replacements |
| | `key` | text | `text` | `AbilityDef::ref_key` (register 25; see `abref`) |
| | `cost` | text | `text` | the script's cost: the line's `Cost$` (`T Sac<1/CARDNAME>`), or a keyword line's argument (`K:Equip:1 R` → `1 R`) |
| | `tap` | u8 | `bool` | `{T}` in the cost |
| | `timing` | u8 | `enum:timing` | spell: instant when an instant or with Flash, else sorcery; activated: sorcery for a loyalty, `SorcerySpeed$`, Equip/Reconfigure/Unattach ability, else instant; anything else any |
| | `limit` | u8 | `count` | activations per turn; 0 = none |
| | `line` | text | `text` | the script line with its key (`A:`, `T:`, `S:`, `R:`, `K:`) |
| `card_pip` (14) | `card`, `face` | u16, u8 | `card`, `enum:face` | hybrid and Phyrexian pips of the printed cost |
| | `kind` | u8 | `enum:pip_kind` | hybrid, twobrid, Phyrexian |
| | `colors` | u8 | `mask:color` | |
| | `generic_alt` | u8 | `count` | twobrid's 2 |
| `decklist` (15) | `player` | seat | | only the viewer's under `--closed-decklist`; the viewer's own first |
| | `name` | text | `text` | deck name (register 23) |
| | `main`, `side` | u16 | `count` | |
| `decklist_card` (16) | `player` | seat | | |
| | `card` | u16 | `card` | |
| | `main`, `side` | u8 | `count` | registered counts |

`CARDS` holds every card of both registered lists, in ascending card id, then every registered
token those cards can make, in ascending card id. Under `--closed-decklist` it holds every vocab
card instead (by front face; a back face's own vocab entry is that card's back row), for every
viewer: the opponent's cards need rows, and listing only the seat's own would show which are the
opponent's. A card no deck holds is read from the parse cache (`card_data_by_name`,
`src/card_db.h`), which creates no entity. A card's tokens are found from its parsed definitions: each `TokenScript$` of its abilities (sub-abilities, modes,
floating and granted abilities included), the engine-synthesized ones (Investigate → Clue,
Amass → the subtype's Army, Mobilize → Warrior, `src/effects/effects.h`), then the tokens those
tokens make. A token script with no token-band vocab entry or no script file is left out.

### A.5 Snapshot (`STATE` frame)

| Table (id) | Column | Type | Tag | Range / notes |
|---|---|---|---|---|
| `header` (20) | `game` | u8 | `count` | 0-based game of the match (between games, the upcoming one); version, schema hash and viewer are in the frame header |
| | `decision` | u32 | `count` | decision number in the game, from 1 (`Game::state_decisions`) |
| | `knowledge` | u8 | `enum:knowledge_level` | this viewer's level; OFF until the sidecar lands (T11) |
| | `terminal` | u8 | `bool` | the frame written once the match is over: no menu, no reply; it carries the last events (section 9) |
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
| | `mana_w`, `mana_u`, `mana_b`, `mana_r`, `mana_g`, `mana_c` | 6 × u16 | `count` | floating pool |
| | `hand`, `library`, `graveyard`, `exile`, `sideboard` | 5 × u16 | `count` | zone sizes (by owner) |
| | `mulligans`, `to_bottom` | u8 | `count` | London bottoming still owed |
| | `passed` | u8 | `bool` | pass flag |
| | `monarch`, `city_blessing` | u8 | `bool` | |
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
| | `copy_of` | u16 | `card` | the card a copy effect made a permanent a copy of (its printed card is `card`) |
| | `tapped`, `sick`, `phased_out` | u8 | `bool` | engine flags (register 15); `sick` is effective summoning sickness (`is_summoning_sick`: a creature, haste considered) |
| | `attacking`, `blocking`, `blocked` | u8 | `bool` | details in `combat` |
| | `damage` | u16 | `count` | |
| | `deathtouch_damage` | u8 | `bool` | |
| | `power`, `toughness` | i16 | `signed` | post-layer for a battlefield creature; printed (active face) elsewhere |
| | `types` | u32 | `mask:card_type` | post-layer on the battlefield |
| | `supertypes` | u8 | `mask:supertype` | |
| | `colors` | u8 | `mask:color` | `effective_colors` |
| | `mana_value` | u8 | `count` | |
| | `attached_to` | u32 | `handle` | |
| | `timestamp` | u32 | `index` | entered-battlefield order; none off the battlefield |
| | `entered_turn`, `cant_be_blocked`, `damage_shielded`, `delayed_subject` | u8 | `bool` | as BQUERY's per-turn statuses |
| | `activations_turn`, `resolutions_turn` | u8 | `count` | |
| | `chosen_name` | u16 | `card` | |
| | `chosen_type` | u16 | `enum:subtype` | 0xFFFF none |
| | `entered_x` | u16 | `count` | |
| | `monstrous`, `evoked`, `escaped`, `unearthed`, `cast_from_hand` | u8 | `bool` | |

A row's characteristic columns (`transformed`, `copy_of`, `power` … `mana_value`, `chosen_name`,
`chosen_type`) and its `object_subtype`, `object_keyword`, `object_ability`, `object_provenance`
and `object_link` rows are filled only when the viewer can see the object's identity; a
face-down object shows its zone, seats, position, public status and counters. Phased-out
permanents are included and flagged (the encoder, like the BQUERY serializer, reads
`Permanent::is_phased_out` directly). No `can_attack` or `chosen_color` column: the engine
computes neither outside its legal-action enumeration.
| `object_subtype` (25) | `obj`, `subtype` | u32, u16 | `handle`, `enum:subtype` | post-layer |
| `object_keyword` (26) | `obj` | u32 | `handle` | register 14; a battlefield permanent's `permanent_keywords`, else the active face's printed keywords |
| | `kw` | u8 | `enum:keyword` | as `card_keyword` |
| | `text` | text | `text` | |
| `object_counter` (27) | `obj` | u32 | `handle` | a permanent's counters, or a card's off the battlefield (`Zone::counters`); also for a face-down object |
| | `kind` | text | `text` | `P1P1`, `LOYALTY`, `TIME`, … |
| | `count` | u16 | `count` | |
| `object_ability` (28): a battlefield permanent's abilities as it has them now (`Permanent::abilities`: activated and mana abilities, keyword-derived triggers); printed abilities of other objects are in `CARDS` | `obj` | u32 | `handle` | |
| | ability | abref | | |
| | `granted` | u8 | `bool` | not in its card's own list, or granted by a static |
| | `activations_turn` | u8 | `count` | limit state |
| | `exhausted` | u8 | `bool` | its limit is reached this turn |
| `object_provenance` (29) | `obj`, `src` | u32 | `handle` | register 1, coarse: one row per kind of continuous effect the permanent records it is under (a copy; Animate types, keywords, abilities, set P/T; end-of-turn keyword grants and removals; ability removal; static, end-of-turn and counter P/T changes; P/T switch). Static abilities' contributions keep no source, so `src` is 0 |
| | `src_card` | u16 | `card` | none (no source recorded) |
| | `layer` | u8 | `enum:layer` | CR 613 layer / sublayer |
| | `duration` | u8 | `enum:duration` | |
| `object_link` (30) | `obj`, `linked` | u32 | `handle` | linked exiled cards |
| | `kind` | u8 | `enum:link_kind` | exiled with, returnable |
| `combat` (31): one row per blocker of each attacker, attackers then blockers in handle order | `attacker` | u32 | `handle` | |
| | `target_kind` | u8 | `enum:attack_target_kind` | player, planeswalker |
| | `target_player` | seat | | |
| | `target` | u32 | `handle` | planeswalker |
| | `blocker` | u32 | `handle` | 0 = unblocked row |
| | `assigned` | u16 | `count` | damage assigned so far (`Game::combat.damage_assignment`). No damage-assignment order column: the engine keeps no order (CR 510.1c-d assignment is chosen as damage is dealt) |
| `stack` (32): top first | `handle` | u32 | `handle` | |
| | `pos` | u16 | `index` | 0 = top |
| | `kind` | u8 | `enum:stack_kind` | spell, activated, triggered, copy |
| | `controller` | seat | | |
| | `source` | u32 | `handle` | an ability's source; 0 if gone, and for a spell |
| | `source_card` | u16 | `card` | as `object.card` (a spell's own card) |
| | ability | abref | | |
| | `x` | i16 | `signed` | −1 = no X |
| | `amount` | u16 | `count` | `Ability::amount` |
| | `kicked`, `replicate` | u8 | `count` | |
| | `variant` | u16 | `mask:cast_variant` | the `Spell` flags: flashback, evoke, escape, offspring, impending, warp, back face |
| | `copy`, `cant_be_countered`, `gift_promised` | u8 | `bool` | |
| | `mana_spent` | u8 | `count` | |
| | `colors_spent` | u8 | `mask:color` | |
| | `delayed` | u32 | `index` | `delayed_trigger.seq`; 0 none |
| `stack_mode` (33) | `stack`, `mode` | u32, u8 | `handle`, `index` | chosen modes |
| `stack_target` (34) | `stack` | u32 | `handle` | |
| | `order` | u8 | `index` | announcement order: the primary ability's targets, its sub-abilities', then each chosen mode's |
| | `kind` | u8 | `enum:target_kind` | object, player |
| | `target` | u32 | `handle` | |
| | `target_player` | seat | | |
| | `lki` | u16 | `row:lki` | the target left its zone (target 0); none when it has no last-known information |
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
| | `kind` | u8 | `enum:player_effect_kind` | as BQUERY's player-effect block (`player_effects`) plus floating triggers |
| | `colors` | u8 | `mask:color` | hexproof-from colours |
| | `duration` | u8 | `enum:duration` | from the resolved effect behind it (end of turn, until your next turn); STATIC when a battlefield static gives it |
| | `source_card` | u16 | `card` | a floating trigger's creator |
| | `filter` | text | `text` | a cast-as-though-it-had-flash permission's filter |
| `play_permission` (39): per graveyard / exile object and player | `obj` | u32 | `handle` | |
| | `player` | seat | | who may play it |
| | `kind` | u8 | `enum:permission_kind` | one row per source of `card_play_permission` (`CardPlayPermission::Source`) |
| | `resource` | u8 | `enum:permission_resource` | an exile grant's resource (`ImpulseCastPermission`: NORMAL → MANA); MANA otherwise |
| | `amount` | u16 | `count` | an exile grant's amount |
| | `expires_turn` | u8 | `bool` | lapses at this turn's cleanup |
| `shield` (40) | `kind` | u8 | `enum:shield_kind` | combat-damage prevention |
| | `obj` | u32 | `handle` | |
| | `as_source`, `as_target` | u8 | `bool` | |
| `lki` (41): rows for departed stack targets, as referenced | `card` | u16 | `card` | |
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
| `pending` (43): when a query is parked or a decision source is set | `tag` | u8 | `enum:pending_tag` | `PendingQuery::Tag`; NONE for a blocking prompt |
| | `chooser` | seat | | the parked query's chooser, else the deciding seat |
| | `source` | u32 | `handle` | `Game::pending.decision_source` (else the query's) |
| | `source_card` | u16 | `card` | |
| | `source_controller` | seat | | |
| `pending_cost` (44): `pending_cost` (`src/queries/pending_cost.h`): the cast flow at a cost step (not while choosing modes or targets), the activation flow at its X or cost steps, the interactive mana payment loop, an unless-cost offer | `for` | u32 | `handle` | spell or ability paid for (an unless offer: the spell it governs, else the asking source) |
| | `for_card` | u16 | `card` | |
| | `part` | u8 | `enum:cost_part` | mana, alternative, additional, unless, X, activation |
| | `rem_w`, `rem_u`, `rem_b`, `rem_r`, `rem_g`, `rem_c`, `rem_generic`, `rem_x` | 8 × u8 | `count` | remaining plain pips; `rem_x` = {X} pips still to announce |
| | `rem_life` | u16 | `count` | |
| | `rem_energy`, `rem_discard` | u8 | `count` | an energy cost; cards to discard (unless offers) |
| | `paid` | u8 | `count` | mana already paid (interactive payment) |
| `pending_cost_pip` (45) | `cost` | u16 | `row:pending_cost` | remaining non-plain pips |
| | `kind`, `colors`, `generic_alt` | u8 | `enum:pip_kind`, `mask:color`, `count` | as `card_pip` (plus snow) |
| `menu` (46): row = action index of the menu as offered (after the interchangeable-choice collapse); only in the deciding seat's and the omniscient frame | `category` | u8 | `enum:action_category` | |
| | `type` | u8 | `enum:action_type` | |
| | `source`, `target` | u32 | `handle` | |
| | `source_card` | u16 | `card` | `visible_card_id` of the source, or for a `card_public` choice the card it makes public (a name in a name-a-card menu) |
| | `source_player`, `target_player` | seat | | when the source / target is a player |
| | ability | abref | | an ACTIVATE_ABILITY action's ability |
| | `ordinal` | i16 | `signed` | `option_ordinal`; −1 = n/a |
| | `variant` | u16 | `mask:cast_variant` | the action's flags: flashback, escape, offspring, back face (cast or play), alt cost, impulse, companion, suspend, cancel |
| | `card_public` | u8 | `bool` | |
| | `description` | text | `text` | covered by the leak checks (register 12). For a seat, an entry whose source or target's identity the viewer cannot see (and that is not `card_public`) gets `(hidden card)` instead |
| `match` (47) | `game` | u8 | `count` | as `header.game` |
| | `plays_first` | seat | | this game, or the upcoming one while sideboarding |
| | `sideboard_phase` | u8 | `bool` | |
| | `sb_swaps` | u8 | `count` | |
| | `sb_drift` | i8 | `signed` | |
| | `closed_decklist` | u8 | `bool` | |
| `own_deck` (48): the viewer's current 75 (the omniscient frame: both players') | `player` | seat | | |
| | `card` | u16 | `card` | |
| | `main`, `side` | u8 | `count` | |

### A.6 Events and knowledge (`STATE` frame)

The encoder emits the event tables (`src/state_format/events.*`, §5); the knowledge tables are
defined but not emitted yet (T11).

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
| `cost_part` | MANA, ALTERNATIVE, ADDITIONAL, UNLESS, X, ACTIVATION |
| `layer` | COPY, CONTROL, TEXT, TYPE, COLOR, ABILITY, PT_CDA, PT_SET, PT_MODIFY, PT_COUNTERS, PT_SWITCH |
| `duration` | STATIC, END_OF_TURN, UNTIL_YOUR_NEXT_TURN, UNTIL_HOST_LEAVES, PERMANENT |
| `link_kind` | EXILED_WITH, RETURNABLE |
| `player_effect_kind` | PROTECTION_FROM_EVERYTHING, CANT_GAIN_LIFE, HEXPROOF_FROM, SPELLS_CANT_BE_COUNTERED, CAST_AS_FLASH, SORCERY_SPEED_ONLY, FLOATING_TRIGGER |
| `permission_kind` | FLASHBACK, ESCAPE, GRAVEYARD_CAST, GRAVEYARD_LAND, EXILE_GRANT (the `CardPlayPermission::Source` bits, in bit order) |
| `permission_resource` | MANA, ENERGY, LIFE, FREE |
| `shield_kind` | COMBAT_DAMAGE |
| `cast_variant` | FLASHBACK, EVOKE, ESCAPE, OFFSPRING, IMPENDING, WARP, BACK_FACE, ALT_COST, IMPULSE, COMPANION, SUSPEND, CANCEL |
| `knowledge_level` | OFF, EXPERT |
| `pool_kind` | HAND, LIBRARY_RANGE, SIDEBOARD, FACE_DOWN, SET_ASIDE |

These are X-macro lists in `schema.h`, so each C++ enum and its string table come from one list.

Existing C++ enums used (string tables built by `src/state_format/enum_names.cpp` from the
engine's own definitions, each a name function that switches over every enumerator with
`-Wswitch` as an error, so an enumerator without a name fails the build): `Step` (`step`),
`ActionCategory` (`action_category`), `ActionType` (`action_type`), `Zone::ZoneValue`
(`zone`), `Zone::Ownership` (`seat`: UNKNOWN, PLAYER_A, PLAYER_B), `Colors` (`color`, NO_COLOR
included), `MandatoryChoice` (`mandatory_choice`), `Game::DayNight` (`day_night`),
`PendingQuery::Tag` (`pending_tag`), `Game::RevealWindow::Kind` (`window_kind`: REVEAL, LOOK,
FACE_DOWN_LOOK, CONTINUOUS_REVEAL), `Game::RevealWindow::End` (`window_end`: RESOLUTION,
STACK_OBJECT, LEAVES_ZONE), `DelayedTriggerLink::FireKind` (`fire_kind`: FIRE_UPKEEP …
FIRE_LEAVES_BATTLEFIELD), `EffectKind` (`effect_kind`: None, then the categories of
`effect_kinds.def`, included as an X-macro). The type lists `all_types`, `all_subtypes`,
`all_supertypes` (`src/constants.cpp`, sorted) give `card_type`, `subtype`, `supertype`; the
parser's keyword table (`kKeywordTable`, `src/parse.cpp`, via `parser_keyword_names`) followed
by the observation's evergreen keywords it lacks (`OBS_KEYWORDS`, `src/machine_io.h`) gives
`keyword`. The SCHEMA frame lists the format's enums first, then these.
