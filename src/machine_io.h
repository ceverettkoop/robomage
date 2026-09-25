#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "classes/action.h"
#include <cmath>
#include <vector>

// BQUERY format (machine mode): a text header line
// "BQUERY: <num_choices> <STATE_SIZE> <MAX_ACTIONS>\n"
// (the two trailing sizes are a runtime layout handshake — the Python driver
// asserts them against its own imported constants so a C++ layout change without
// regenerated Python constants fails loudly instead of misframing the payload)
// followed immediately by a binary payload (see cli_emit_machine_query):
//   float32[STATE_SIZE] state, int32[MAX_ACTIONS] cats,
//   float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl,
//   float32[MAX_ACTIONS] pub, int32[MAX_ACTIONS] zone,
//   int32[MAX_ACTIONS] refs, int32[MAX_ACTIONS] ords.
// Per-action metadata is padded to MAX_ACTIONS; only the first num_choices
// entries are meaningful.
//
// The state vector (STATE_SIZE floats) is followed by:
//   - N ActionCategory integers (0 .. ACTION_CATEGORY_MAX, see the enum in
//     src/classes/action.h).
//   - N card vocab index floats: card_vocab_index / N_CARD_TYPES for card
//     entities, or -1.0 / N_CARD_TYPES (-0.0009765625) as a null sentinel for
//     non-card entities (players, confirm slots, fail-to-find, empty).
//   - N controller_is_self floats: 1.0 if entity is controlled by the priority
//     player, 0.0 if controlled by the opponent, null sentinel for
//     non-entity actions (pass priority, confirm slots, etc.).
//   - N card_is_public floats: 1.0 if the choice's card identity is public
//     knowledge to all players (a revealed tutor, e.g. Personal Tutor), else 0.0.
//     Lets observers show the card name for an otherwise-private choice.
//   - N zone_ref integers (ActionRefZone enum): which zone/side the choice's
//     entity lives in (self/opp battlefield, hand, stack, GY, exile, the
//     player objects themselves; REF_NONE = no referenced entity). Lets the
//     policy distinguish e.g. "target the opponent" from "target their creature".
//   - N slot_ref integers: the choice's source entity resolved into the unified
//     entity-reference slot space (see below; -1 = no serialized entity). This is
//     the action<->entity join — the policy can tell WHICH of two same-name
//     permanents a targeting action refers to.
//   - N option_ordinal integers: the choice's ordinal/value scalar (mode index, X
//     value, color index, activated-ability index, ...; -1 = not applicable). See
//     OPTION_ORDINAL_MAX below, the normalizer env.py divides these by.
//
// NOTE: ActionChoice.description is NOT emitted in the BQUERY payload.
// It is stored in Query for human-readable display (CLI) only.
//
// The Python env pads the per-action arrays to MAX_ACTIONS slots; cats, ids,
// ctrl, zone, refs, and ords go into the observation (STATE_SIZE +
// N_ACTION_OBS_BLOCKS*MAX_ACTIONS floats plus the matchup tail); pub stays a
// side-channel for observers.
//
// State is always serialized from the PRIORITY PLAYER'S perspective ("self").
// "Self" refers to the player who currently holds priority.
//
// ── Unified entity-reference slot space ──────────────────────────────────────
// Several fields point at OTHER serialized objects (attachments, combat pairing,
// stack targets, action sources). They all share one viewer-relative index space:
//   0-47   self permanent slots (pack order of the self-permanents block)
//   48-95  opp permanent slots
//   96-107 stack slots (96 = top of stack)
//   -1     none / not serialized (player targets, hand/GY entities, truncation)
// N_ENTITY_REF_SLOTS = 108. In the float state vector a ref is normalized via
// norm_ref(idx) = (idx + 1) / 108 — 0.0 = none, real refs in (0, 1] (avoids a
// sentinel/slot-0 collision); decode with round(v * 108) - 1. The BQUERY per-action
// refs array stays raw int32 with -1 sentinel; env.py normalizes.
//
// Fixed-size state vector layout (STATE_SIZE = 6490 floats):
// Card identity is a single normalized id float per slot (see norm_card_id):
// idx/N_CARD_TYPES, or -1/N_CARD_TYPES for empty/unknown. The id is NOT a one-hot.
//
//  [0-9]      Self player block (10 floats):
//               life/20, hand_ct/10, poison/10, mana[W,U,B,R,G,C]/10, energy/10
//  [10-19]    Opponent player block (10 floats, same layout)
//  [20-32]    Current step one-hot (13 steps: UNTAP..CLEANUP, includes FIRST_STRIKE_DAMAGE)
//  [33]       1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [34]       1.0 if self is Player A, 0.0 if self is Player B. Seat-routing signal for
//             the drivers only: both networks zero it out of their input (see
//             network_global_ctx in train/extractor.py), so no policy can condition on
//             which seat it plays.
//  [35]       Stack size / 10.0
//
//  [36-2099]     Self permanents: 48 slots x 43 floats = 2064
//  [2100-4163]   Opp permanents:  48 slots x 43 floats = 2064
//                The blocks are split by controller, so a slot carries no
//                controller flag. Per slot (offsets within the slot):
//                  [0]  power / 10
//                  [1]  toughness / 10
//                  [2]  is_tapped
//                  [3]  is_attacking
//                  [4]  is_blocking
//                  [5]  has_summoning_sickness
//                  [6]  damage / 10
//                  [7]  is_creature
//                  [8]  is_land
//                  [9]  loyalty / 10 (planeswalkers; 0 otherwise)
//                  [10] p1p1_net / 10 — net (+1/+1 minus -1/-1) counters, SIGNED
//                       (distinguishes persistent counters from EOT pumps; both are
//                       already folded into effective P/T above)
//                  [11] other_counters / 10 — total counters of every other kind
//                       (excl. P1P1/M1M1/LOYALTY; the card-id embedding disambiguates
//                       the kind: Saga -> lore, Chalice -> charge, ...)
//                  [12] attached_to_ref (norm_ref) — for equipment/auras: the slot of
//                       the permanent this is attached to
//                  [13] attached_by_ref (norm_ref) — for creatures: the slot of the
//                       equipment/aura attached to this
//                  [14] attack_target_ref (norm_ref) — attacked planeswalker's slot;
//                       0.0 while is_attacking means "attacking the player"
//                  [15] blocking_target_ref (norm_ref) — the attacker this blocker blocks
//                  [16] is_blocked — attacker was blocked at declare-blockers; stays
//                       blocked even if all blockers leave (CR 509.1h)
//                  [17] is_phased_out — phased-out permanents ARE serialized (with this
//                       flag set) even though the rules treat them as nonexistent
//                       (CR 702.26e), so the model can anticipate the phase-in
//                  [18] entered_this_turn — the permanent entered the battlefield this
//                       turn (the ThisTurnEntered filter predicate,
//                       entered_battlefield_this_turn in game_queries.h; Ocelot Pride,
//                       Phelia)
//                  [19] ability_resolutions_this_turn / PER_TURN_COUNT_NORMALIZER —
//                       triggered-ability resolutions from this permanent this turn
//                       (the Count$ResolvedThisTurn value; Scythecat Cub)
//                  [20] activations_this_turn / PER_TURN_COUNT_NORMALIZER — activations
//                       counted against this permanent's once-per-turn gates: the
//                       ActivationLimit$ counters of all its abilities summed, plus 1 if
//                       a loyalty ability was activated (CR 606.3). Both reset for every
//                       permanent at each untap step (permanent_activations_this_turn)
//                  [21] cant_be_blocked_this_turn — a "can't be blocked this turn"
//                       effect applies (Kappa Cannoneer, Manifold Key)
//                  [22] combat_damage_prevented — the permanent is the creature of an
//                       active combat-damage prevention shield, either direction (Maze
//                       of Ith; Game::combat_damage_shielded)
//                  [23] pending_delayed_subject — the permanent is watched by, or is a
//                       subject of, a delayed trigger still WAITING to fire (e.g. every
//                       Mobilize token, an earthbent land, a Static Prison host);
//                       is_waiting_delayed_trigger_subject in game_queries.h
//                  [24-39] effective keyword multi-hot x16, OBS_KEYWORDS order
//                       (post-layer, via permanent_has_keyword)
//                  [40] chosen_name_id — normalized vocab id of Permanent::chosen_name,
//                       the card name chosen on this permanent (Pithing Needle /
//                       Disruptor Flute named card, Petrified Hamlet named land);
//                       -1/N_CARD_TYPES sentinel when no name is chosen
//                  [41] returnable_exile_id — normalized vocab id of the most recently
//                       exiled card linked to this permanent that STILL has a return path
//                       (Static Prison holding a real card, Flickerwisp/Phelia EOT blink);
//                       -1/N_CARD_TYPES sentinel when none (no return, e.g. Skyclave
//                       Apparition, or nothing exiled). See returnable_exiled_card().
//                  [42] card_id (LAST)
//                Empty slots: 40 zeros + chosen_name sentinel + returnable_exile sentinel
//                + card_id sentinel (all three -1/N_CARD_TYPES).
//
//  [4164-4607]   Stack: 12 slots x 37 floats = 444 (slot 0 = top of stack)
//                Per slot (offsets within the slot):
//                  [0]  controller_is_self
//                  [1]  card_id
//                  [2]  is_spell — 1.0 for a cast spell; 0.0 for a triggered/activated ability
//                  [3]  x_or_amount / 10 — spell: X paid at cast (Spell::x_paid);
//                       ability: Ability::amount
//                  [4-10] cast qualifiers (all 0.0 for abilities): is_copy, kicked_any,
//                       cast_with_flashback, cast_with_evoke, cast_with_escape,
//                       cast_with_offspring, cast_with_impending
//                  [11-16] chosen-mode multi-hot: 1.0 at index i if modal mode i (of the
//                       spell's charm_choices) was announced at cast (CR 601.2b); all
//                       zeros when the object is not modal
//                  [17-36] 4 target sub-slots x 5 floats. Target sub-slots carry the
//                       object's ANNOUNCED targets (public info, CR 601.2c) in
//                       announcement order — primary ability targets, then targeting
//                       sub-abilities', then each chosen mode's — truncated at 4.
//                       Per target sub-slot:
//                         [+0] present (a target occupies this sub-slot)
//                         [+1] target_is_player
//                         [+2] target_controller_is_self (controller of the targeted
//                              permanent / the targeted player himself; owner for a
//                              non-permanent card)
//                         [+3] target_slot_ref (norm_ref; 0.0 for players / non-serialized)
//                         [+4] target_card_id (LAST; -1 sentinel for players)
//                       Empty sub-slots: 4 zeros + card_id sentinel.
//
//  [4608-4863]   Self graveyard: 64 slots x 4 floats = 256
//  [4864-5119]   Opp graveyard:  64 slots x 4 floats = 256
//                Split by OWNER. RECENCY order: slot 0 is the most recent arrival
//                (sorted by Zone::distance_from_top); filled slots are packed with no
//                holes. Per slot (offsets within the slot):
//                  [0]  card_id (FIRST, like the decklist slots; sentinel = empty)
//                  [1]  playable_by_self — the viewer has a permission to play (cast,
//                       or play as a land) this card from here, ignoring timing and
//                       cost: card_play_permission (game_queries.h), the same predicate
//                       the legal-action enumeration gates its graveyard/exile plays
//                       on. Covers flashback, escape, Emry's cast-this-turn grant, and
//                       a play-lands-from-graveyard static (Icetill Explorer, Mole Man)
//                  [2]  playable_by_opp — the same for the viewer's opponent
//                  [3]  play_expires_this_turn — 1.0 when the card is playable and every
//                       permission covering it lapses at this turn's cleanup; 0.0 for a
//                       static or keyword route, a grant that lasts longer (Light Up the
//                       Stage on its cast turn, warp), or an unplayable card
//                Empty slots: card_id sentinel + 3 zeros.
//
//  [5120-5439]   Self exile: 64 slots x 5 floats = 320
//  [5440-5759]   Opp exile:  64 slots x 5 floats = 320
//                Split by OWNER, RECENCY order and packed like the graveyards. Per
//                slot: the four graveyard fields (the exile routes are the
//                impulse-cast grants — Light Up the Stage, Ugin's -11, Amped Raptor,
//                a suspend free cast, warp), then
//                  [4]  counters / ZONE_COUNTER_NORMALIZER — suspend time counters
//                       (Game::suspend_time_counters, Rift Bolt) plus a void counter
//                       (Dauthi Voidwalker); exiled_card_counters in game_queries.h
//                Exile is public except an opponent's FACE-DOWN card (CR 708.2, The
//                Creation of Avacyn chapter I): that slot is filled but hidden — the
//                card_id sentinel with all four scalars 0.0. A suspended card is not
//                playable until its last time counter is removed, and a void-countered
//                card is not playable (Dauthi Voidwalker plays its chosen card while
//                its ability resolves).
//                Empty slots: card_id sentinel + 4 zeros.
//
//  [5760-5769]   Self hand: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty)
//
//  [5770-5773]   Match context (4 floats, all 0.0 in single-game mode):
//                game_number / 3.0, self_match_wins / 2.0,
//                opp_match_wins / 2.0, is_sideboard_phase (0.0 or 1.0)
//
//  [5774-5775]   Library context (2 floats):
//                self_library_ct / 60.0, opp_library_ct / 60.0
//
//  [5776]        Current game turn / 50.0
//
//  [5777-5781]   Known top-5 library cards for the viewer: 5 slots x 1 float = 5
//                Per slot: card_id (sentinel = unknown). Index 0 is the top of
//                the library. Entries are set when a card is placed on top (e.g.
//                Ponder, Brainstorm, Rearrange) and cleared when shuffled.
//
//  [5782-5791]   Known opponent-hand cards: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty/unknown). The specific
//                identities of opponent-hand cards the viewer has had revealed
//                (Duress/Thoughtseize/tutor) and that are still in hand. Unlike
//                the match-scoped revealed bits on the opponent decklist slots, this
//                tracks the exact card and a slot clears when that card leaves the
//                hand for another zone.
//
//  [5792-5793]   Pending decision context: 2 floats.
//                [5792] card_id of the spell/ability currently making a
//                mid-resolution choice (target select, dig/scry/surveil pick,
//                search, discard, modal, ...; sentinel = none). Set via
//                PendingDecisionScope — the source may not be on the stack yet,
//                since targets are announced before the spell moves there
//                (CR 601.2b/c), so this is the only place the observation shows
//                WHAT is asking for the current choice.
//                [5793] 1.0 if that source's controller is the viewer, else 0.0
//                (e.g. 0.0 while choosing a card for the opponent's Thoughtseize).
//
//  [5794-5820]   Global extras (27 floats):
//                  [5794] self lands_played_this_turn / 10
//                  [5795] opp  lands_played_this_turn / 10
//                  [5796] self is_monarch (CR 725)
//                  [5797] opp  is_monarch
//                  [5798] self city's blessing (CR 702.131c)
//                  [5799] opp  city's blessing
//                  [5800] self revolt (a permanent self controlled left the battlefield this turn)
//                  [5801] opp  revolt
//                  [5802] self pending extra turns / 3
//                  [5803] opp  pending extra turns / 3
//                  [5804] is_day  (CR 731.1; both 0.0 = neither)
//                  [5805] is_night
//                  Priority-window context (EXTRAS_PRIORITY_SIZE = 3). All three are
//                  0.0 unless the current decision is an ordinary priority window
//                  (priority_window_open(), game_driver.h): never set for a mandatory
//                  choice, a mid-resolution / mid-cast choice, a pregame decision, or
//                  a sideboard decision.
//                  [5806] self_has_passed — the viewer's Game::a/b_has_passed flag
//                  [5807] opp_has_passed — the other seat's flag. 1.0 means passing
//                       now resolves the top of the stack (or ends the step when the
//                       stack is empty). Forced and voluntary passes read the same.
//                  [5808] is_priority_window — 1.0 for an ordinary priority window
//                       (pass = pass priority), 0.0 for every other decision kind
//                  Mulligan state (EXTRAS_MULLIGAN_SIZE = 3), from Game::pregame:
//                  [5809] self mulligans taken / MULLIGAN_NORMALIZER
//                  [5810] opp  mulligans taken / MULLIGAN_NORMALIZER (public)
//                  [5811] self cards still to bottom / MULLIGAN_NORMALIZER — nonzero
//                       only while the viewer is bottoming (CR 103.5 London mulligan)
//                  All three are 0.0 during a bo3 sideboard phase.
//                  [5812-5817] MandatoryChoice one-hot x6 (NONE at index 0, then
//                       DECLARE_ATTACKERS_CHOICE, DECLARE_BLOCKERS_CHOICE,
//                       CLEANUP_DISCARD, CHOOSE_ENTITY, ASSIGN_COMBAT_DAMAGE_CHOICE)
//                  [5818] self_plays_first — the viewer is the starting player of the
//                       game this observation PERTAINS TO. In-game that is the current
//                       game (Game::pregame.a_goes_first); during a bo3 between-games
//                       sideboard phase it is the UPCOMING game, whose starting player
//                       play_bo3_match has already decided (the loser of the game that
//                       just ended) before either sideboard stage runs. Boarding plans
//                       differ substantially on the play vs the draw, and at 1-1 the
//                       upcoming starting player is otherwise unrecoverable from the
//                       observation.
//                  [5819] sideboard swaps completed this phase / SIDEBOARD_SWAP_CAP
//                       (0.0 outside the phase)
//                  [5820] sideboard maindeck drift, (d + 1) / 2 so -1/0/+1 map to
//                       0.0/0.5/1.0 and "balanced" is the 0.5 midpoint. Always 0.5
//                       outside the phase. The encoding is kept, but the 0.0 pole is
//                       unreachable: the menu is IN-FIRST, so drift is only ever 0
//                       or +1 and only 0.5/1.0 are ever emitted (see
//                       run_sideboard_phase in src/game_driver.cpp).
//
//  ── Deck-identity tail blocks ────────────────────────────────────────────────
//  Each self slot is (card_id, count) and each opponent slot is (card_id, count,
//  revealed): card_id via norm_card_id (empty slot = -1 sentinel), count
//  normalized /4.0 (basics may exceed 1.0). Slots are packed
//  ascending by vocab index with NO holes (a testable invariant that keeps the
//  encoding byte-stable across actors). Overflow (more distinct names than slots)
//  or a name absent from the vocab is a fatal_error, never a silent truncation.
//
//  [5821-5916]   Self LIVE library: 48 slots x (card_id, count) = 96.
//                The viewer's LIBRARY zone tallied live at serialization time, so
//                cards leaving/returning to the library are always reflected.
//                Viewer-only — the opponent's live library stays hidden.
//
//  [5917-6012]   Self LIVE maindeck:  48 slots x (card_id, count) = 96.
//  [6013-6044]   Self LIVE sideboard: 16 slots x (card_id, count) = 32.
//                (16, not 15: a legal sideboard is 15 cards, but this block is
//                also written mid-swap, when a cut card is momentarily the
//                sideboard's 16th — see DECKLIST_SIDE_SLOTS in gamestate.h.)
//                The viewer's OWN current 75 (deck_state's live store), which every
//                player legitimately knows. Distinct from the live-library block
//                above: that is the LIBRARY ZONE, which shrinks as you draw and is
//                stale between games, whereas this is the deck CONFIGURATION —
//                every card regardless of zone. It is the only place the observation
//                shows what the sideboarding player is choosing between: mid-phase it
//                tracks each completed swap, so the model can see its maindeck while
//                picking a card to bring in and its remaining sideboard while picking
//                a card to cut.
//
//  [6045-6188]   Opponent-of-viewer REGISTERED maindeck: 48 slots x (card_id, count, revealed) = 144.
//  [6189-6236]   Opponent-of-viewer REGISTERED sideboard: 16 slots x (card_id, count, revealed) = 48.
//                (Registered, so only 15 slots can ever fill; the width just
//                follows DECKLIST_SIDE_SLOTS.)
//                The opponent's decklist as REGISTERED at match start (open-decklist
//                ruleset), read from deck_state. FROZEN for the whole match: a bo3's
//                sideboard swaps never touch these blocks, so you know the opponent's
//                registered 75 but NOT which 60 of it they boarded into for game 2+.
//                That split is hidden information, and the MCTS determinizer models
//                it as such (see `opp_sideboard_hidden` in search_server.cpp).
//                revealed = 1.0 when the opponent-of-viewer has revealed that card
//                this match (match_state's reveal set): set whenever an opponent
//                card enters a public zone (battlefield/stack/graveyard/exile) or is
//                revealed by a tutor, accumulated across the games of a bo3 and
//                persisting over the per-game ECS reset. A double-faced card's slot
//                is also set when its back face was revealed (a transformed
//                permanent, an MDFC played back face up). 0.0 on empty slots.
//
//  ── Mana development ─────────────────────────────────────────────────────────
//  A summary of each player's MANA BASE, which nothing else in the observation
//  states: the player-block "mana" floats are the FLOATING pool (~always empty at a
//  decision point) and the land-play limit was never serialized at all, so
//  castability had to be inferred from 48 raw permanent slots through pooling.
//  Counts and mana are normalized by MANA_COUNT_NORMALIZER, land drops by
//  LAND_DROPS_NORMALIZER. Self first, then the opponent, like every other block.
//
//  [6237-6246]   Self mana development (10 floats):
//                  [0-5] potential_W/U/B/R/G/C — how many of this player's UNTAPPED
//                        battlefield mana sources could produce that color right now
//                        (mana_potential(), mana_system.h): a permanent with an
//                        activatable mana ability, phased-out excluded, a {T} ability on
//                        a summoning-sick creature excluded, activation limits and
//                        Activation$ gates honored. "Any"/"Combo"/reflected producers
//                        count toward EACH color they could make; a permanent counts at
//                        most ONCE per color no matter how many abilities offer it.
//                  [6]   potential_total — distinct untapped sources (each counted once,
//                        regardless of how much mana it makes) PLUS the floating pool's
//                        current size. Every per-color count is <= this.
//                  [7]   lands_in_play — battlefield lands this player controls
//                  [8]   lands_in_hand — land cards in hand (a modal DFC whose BACK face
//                        is a land counts: playing it is a land play). SELF ONLY — the
//                        opponent's hand is hidden information.
//                  [9]   land_drops_remaining / LAND_DROPS_NORMALIZER — lands this player
//                        may still play this turn, from the SAME expression the PLAY_LAND
//                        legal-action gate uses (rules_mod::land_drops_remaining, which
//                        folds in AdjustLandPlays statics), clamped at 0.
//  [6247-6255]   Opponent mana development (9 floats): the same fields MINUS
//                lands_in_hand, i.e. potential_W/U/B/R/G/C, potential_total,
//                lands_in_play, land_drops_remaining.
//
//  ── Log-scaled vitals ────────────────────────────────────────────────────────
//  The SAME life and library counts the player blocks / library-context block
//  already carry, re-warped through log1p. The linear floats (life/20, library/60)
//  give the net its LEAST resolution exactly where the stakes are highest: 2 life
//  vs 5 life is 0.10 vs 0.25 and 1 card vs 3 cards is 0.017 vs 0.050 — differences
//  a net has to spend capacity magnifying, right at the boundary where they decide
//  the game. log1p re-warps to PROPORTIONAL resolution (d log x = dx / x), so the
//  near-zero cliff — the region with the fewest training samples and the steepest
//  value gradient — becomes smooth, low-curvature structure learnable from little
//  data, while the mid/high range compresses (where a point of life genuinely does
//  matter less). Both encodings are kept, not one: absolute life-payment arithmetic
//  ("can I pay 4 life twice?", "is 6 damage lethal?") is genuinely LINEAR in the
//  mid-range, and that is what the linear floats state directly.
//    log_life    = log1p(max(life, 0)) / LOG_LIFE_DENOM       (= log1p(20))
//    log_library = log1p(library_count) / LOG_LIBRARY_DENOM   (= log1p(60))
//  The 20 and the 60 are LIFE_NORMALIZER / LIBRARY_NORMALIZER, the very divisors of
//  the linear floats, so the two encodings of a value hit 1.0 at the same point.
//  Life is clamped at 0 first (a player at -3 is dead; the SBA has just not run
//  yet, and log1p is undefined below -1). Both are 1.0 at the starting value and
//  EXCEED 1.0 above it (life > 20, library > 60) — exactly like the linear floats,
//  which are likewise unclamped above their normalizer.
//
//  [6256-6257]   Self log vitals (2 floats): log_life, log_library
//  [6258-6259]   Opponent log vitals (2 floats): same two fields. Library size is
//                public information, so unlike lands_in_hand above there is nothing
//                to withhold from the opponent half.
//
//  ── Per-turn counters ────────────────────────────────────────────────────────
//  Each player's Player per-turn tallies (reset at cleanup), all public. Counts are
//  divided by PER_TURN_COUNT_NORMALIZER, life by LIFE_NORMALIZER. Self half then the
//  opponent half, same fields both sides:
//                  [0]  spells_cast_this_turn / 10
//                  [1]  noncreature_spells_cast_this_turn / 10
//                  [2]  instant_sorcery_spells_cast_this_turn / 10
//                  [3]  cards_drawn_this_turn / 10
//                  [4]  life_gained_this_turn / 20
//                  [5]  life_lost_this_turn / 20
//                  [6-10] spell_colors_cast_this_turn multi-hot W, U, B, R, G
//
//  [6260-6270]   Self per-turn counters (11 floats)
//  [6271-6281]   Opponent per-turn counters (11 floats)
//
//  ── Pending delayed triggers (CR 603.7) ──────────────────────────────────────
//  A derived view, built at serialization time, of every delayed trigger from
//  registration (register_delayed_trigger, game_queries.h) until it resolves:
//  the records still WAITING in Game::delayed_triggers, plus the stack ability
//  objects whose DelayedTriggerLink::seq != 0 (fired, now ON THE STACK). An entry
//  disappears when its record expires unfired or its stack object leaves the
//  stack (resolved, countered, fizzled). Slots are packed with no holes in
//  ascending registration seq; more than 16 entries truncate (debug builds print a
//  stderr WARNING). All fields are public information.
//
//  [6282-6489]   Delayed triggers: 16 slots x 13 floats = 208
//                Per slot (offsets within the slot):
//                  [0]  present
//                  [1]  controller_is_self — the trigger's controller is the viewer
//                  [2]  state — 0.0 waiting, 1.0 on the stack
//                  [3]  stack_ref (norm_ref) — its stack object's slot while on the
//                       stack (0.0 while waiting, or past the 12 displayed slots)
//                  [4]  creator_card_id — the card whose ability set it up (Mishra's
//                       Bauble, Flickerwisp, the Mobilize creature, the Static Prison /
//                       Sheltered by Ghosts HOST, the earthbend spell), captured at
//                       registration (a token keeps its token-band id)
//                  [5]  creator_ref (norm_ref) — the creator's slot when it is on the
//                       battlefield or the stack, else 0.0
//                  [6]  subject_ref (norm_ref) — the first watched / affected
//                       permanent's battlefield slot: the watched object while
//                       waiting, else the first subject still on the battlefield
//                  [7]  subject_card_id — the first subject (the blinked or exiled
//                       card, the first token to sacrifice/exile, the watched land);
//                       captured at registration; sentinel when there is none
//                       (Mishra's Bauble's draw)
//                  [8-11] fire_on one-hot: upkeep, end step, end of combat, leaves the
//                       battlefield (all 0.0 for any other phase)
//                  [12] fires_this_turn — a WAITING phase trigger scheduled for a step
//                       still ahead this turn (delayed_trigger_fires_this_turn); 0.0
//                       for leaves-the-battlefield watches and on-stack entries
//                Empty slots: 4 zeros + creator_card_id sentinel + 2 zeros +
//                subject_card_id sentinel + 5 zeros.

static constexpr int STATE_SIZE             = 6490;
// Max sideboard swaps a player may complete in one between-games phase. Both the
// engine's phase cap and the normalizer for the serialized swaps-made scalar, so
// the two can never drift apart.
static constexpr int SIDEBOARD_SWAP_CAP     = 15;
static constexpr int N_CARD_TYPES      = 1024; // embedding vocab size (card identity is emitted as a normalized id, not a one-hot)
static constexpr int OPTION_ORDINAL_MAX = 63;  // normalizer for the per-action option_ordinal scalar
                                               // (see LegalAction::option_ordinal): mode index, X
                                               // value, color index, cast variant, top-of-library
                                               // depth, binary pole, activated-ability index within
                                               // the source's ability list (so same-permanent
                                               // activations — e.g. planeswalker loyalty abilities —
                                               // are distinguishable; synthesised equip/unattach use
                                               // 32/33); -1 = not applicable
// Number of per-action metadata arrays folded into the RL observation vector,
// in order: cats | ids | ctrl | zone_ref | slot_ref | option_ordinal. Each is
// MAX_ACTIONS wide. The `pub` array is ALSO emitted in the BQUERY payload but is a
// side-channel (observer-only), so it is NOT counted here. This is the ONE source
// of truth for the block count: src/actor/obs_builder.h derives ACTOR_OBS_SIZE from
// it, and train/env.py imports it (via _enums.py codegen) for OBS_SIZE — so the two
// obs reconstructions can never disagree on how many action blocks there are.
static constexpr int N_ACTION_OBS_BLOCKS = 6;
static constexpr int PERM_SLOT_SIZE    = 43;   // 10 stat/combat/type + 2 counters + 4 refs + 2 flags + 5 per-turn statuses + pending-delayed-subject + 16 keywords + chosen-name id + returnable-exile id + card-id float
static constexpr int STACK_MODE_SLOTS  = MAX_STACK_MODES; // chosen-mode multi-hot width per stack slot
static constexpr int STACK_TGT_SLOTS   = MAX_STACK_TGTS;  // serialized targets per stack slot (truncated)
static constexpr int STACK_TGT_FIELDS  = 5;    // present + is_player + controller_is_self + slot_ref + card-id

// ── State-vector block widths (THE source of truth) ──────────────────────────
// Every fixed-width block of the state vector, as a named constant. Four separate
// reconstructions of this layout exist — serialize_state (machine_io.cpp), the
// gym env (train/env.py), the policy extractor (train/extractor.py), and the C++
// AZ actor (src/actor/obs_builder.cpp) — and each used to re-spell these widths as
// bare literals. They are now single-sourced here and mirrored into Python by
// codegen (train/gen_enums.py's _MACHINE_INTS -> train/_enums.py), so a width
// change propagates to every consumer instead of being hand-copied four times.
//
// Why this matters beyond tidiness: the Python side's only structural guard was
// `_EXTRAS_END == STATE_SIZE`, which catches a change to the TOTAL but NOT a
// compensating one (a float moved from one block to another keeps the total and
// silently misaligns every field in between). The OFFSET_CHAIN below closes that
// hole on the C++ side by pinning each block's absolute start.
static constexpr int PLAYER_BLOCK_SIZE = 10;   // life, hand_ct, poison, mana[WUBRGC], energy
static constexpr int STEP_ONEHOT_SIZE  = 13;   // UNTAP..CLEANUP, incl. FIRST_STRIKE_DAMAGE
static constexpr int HEADER_FLAGS      = 3;    // is_active + self_is_a + stack_size
static constexpr int CARD_ID_SLOT_SIZE = 1;    // hand / known-top / known-opp-hand
// Graveyard and exile slots (see the layout comment above): card id FIRST, then the
// play-permission flags; an exile slot appends its counters.
static constexpr int GY_SLOT_SIZE      = 4;    // card id + playable_by_self + playable_by_opp + expires
static constexpr int EXILE_SLOT_SIZE   = 5;    // the graveyard fields + counters
static constexpr int ZONE_CARD_ID_OFF        = 0;  // card id within a graveyard / exile slot
static constexpr int ZONE_PLAYABLE_SELF_OFF  = 1;
static constexpr int ZONE_PLAYABLE_OPP_OFF   = 2;
static constexpr int ZONE_EXPIRES_OFF        = 3;
static constexpr int EXILE_COUNTERS_OFF      = 4;  // exile slots only
static constexpr int ZONE_COUNTER_NORMALIZER = 10; // divisor of the exile counters float
static constexpr int STACK_HEAD_FIELDS = 3;    // controller_is_self + card id + is_spell
static constexpr int STACK_XAMT_FIELDS = 1;    // x_or_amount / 10
static constexpr int STACK_QUAL_FIELDS = 7;    // is_copy, kicked, flashback, evoke, escape, offspring, impending
static constexpr int MATCH_CTX_SIZE    = 4;    // game#, self wins, opp wins, is_sideboard_phase
static constexpr int LIBRARY_CTX_SIZE  = 2;    // self lib/60, opp lib/60
static constexpr int CUR_TURN_SIZE     = 1;    // current turn / 50
static constexpr int PENDING_DECISION_SIZE = 2; // source card id + ctrl_is_self
static constexpr int EXTRAS_SCALARS    = 12;   // lands x2, monarch x2, blessing x2,
                                               // revolt x2, extra turns x2, is_day, is_night
static constexpr int EXTRAS_PRIORITY_SIZE = 3; // self_has_passed, opp_has_passed, is_priority_window
static constexpr int EXTRAS_MULLIGAN_SIZE = 3; // self/opp mulligans taken, self bottom remaining
static constexpr int MULLIGAN_NORMALIZER  = 7; // divisor of the three mulligan-state floats
static constexpr int EXTRAS_SB_CTX_SIZE = 3;   // self_plays_first + swaps made + maindeck drift
static constexpr int DECKLIST_SLOT_SIZE = 2;   // card id + count per self decklist / live-library slot
static constexpr int OPP_DECKLIST_SLOT_SIZE = 3; // card id + count + revealed per opp registered-decklist slot
static constexpr int OPP_DECKLIST_REVEALED_OFF = 2; // revealed bit within an opp decklist slot
// Mana-development block (see the layout comment above). The self half carries one
// extra field — lands_in_hand — that the opponent half cannot (hidden information).
static constexpr int MANA_DEV_COLORS    = 6;   // potential W, U, B, R, G, C
static constexpr int MANA_DEV_SELF_SIZE = 10;  // 6 colors + total + lands_in_play +
                                               // lands_in_hand + land_drops
static constexpr int MANA_DEV_OPP_SIZE  = 9;   // same minus lands_in_hand
// Normalizers for that block, kept as ints so train/gen_enums.py can mirror them
// (parse_int_constant only reads plain integer literals) and the Python invariants
// decode with exactly the divisor the engine used.
static constexpr int MANA_COUNT_NORMALIZER = 10;  // source/land counts and mana totals
static constexpr int LAND_DROPS_NORMALIZER = 3;   // land drops remaining this turn
// Log-scaled vitals block (see the layout comment above): log_life + log_library
// per player, self half then opponent half — same two fields both sides, since
// library size is public.
static constexpr int LOG_VITALS_PLAYER_SIZE = 2;  // log_life, log_library
// The life and library scales. These are the divisors of the LINEAR floats — the
// player block's life (push_player_block) and the library-context counts — AND the
// log1p scales below, deliberately the same numbers: both encodings of a value then
// reach 1.0 at the same point (a full life total / an unmilled library), so the net
// sees one consistent "100%" across the pair. Plain ints so gen_enums.py's
// parse_int_constant mirrors them into train/_enums.py; nothing re-spells a bare 20
// or 60 on either side of the boundary.
static constexpr int LIFE_NORMALIZER    = 20;  // starting life total
static constexpr int LIBRARY_NORMALIZER = 60;  // starting library size
// The log denominators. `const`, not `constexpr`: std::log1p is not a constant
// expression in C++17 (clang rejects even the __builtin_ form there), so these are
// initialized once at static-init from the scales above rather than spelled as
// pre-computed float literals that could drift from them.
// Per-turn counters block (see the layout comment above): per player, 6 scaled
// counts then the 5-color spell-color multi-hot, self half then opponent half.
static constexpr int PER_TURN_COUNT_FIELDS     = 6;   // spells, noncreature, instant/sorcery,
                                                      // cards drawn, life gained, life lost
static constexpr int PER_TURN_COLOR_FIELDS     = 5;   // spell colors cast W, U, B, R, G
static constexpr int PER_TURN_PLAYER_SIZE      = 11;  // count fields + color fields
static_assert(PER_TURN_PLAYER_SIZE == PER_TURN_COUNT_FIELDS + PER_TURN_COLOR_FIELDS,
              "per-turn block per player = counts + colors");
// Divisor of the per-turn counts (the per-player block's spell/draw counts and the
// permanent slot's resolution/activation counts); life gained/lost use LIFE_NORMALIZER.
static constexpr int PER_TURN_COUNT_NORMALIZER = 10;
// Pending delayed-trigger block (see the layout comment above).
static constexpr int DELAYED_SLOTS       = MAX_DELAYED_TRIGGER_SLOTS;  // 16
static constexpr int DELAYED_FIRE_KINDS  = N_DELAYED_FIRE_KINDS;       // upkeep, end step, end of combat, leaves bf
static constexpr int DELAYED_SLOT_SIZE   = 13;  // present, ctrl_is_self, state, stack_ref, creator id,
                                                // creator_ref, subject_ref, subject id, fire_on x4,
                                                // fires_this_turn
static_assert(DELAYED_SLOT_SIZE == 8 + DELAYED_FIRE_KINDS + 1, "delayed slot = 8 fields + fire one-hot + fires_this_turn");
static constexpr int DELAYED_CREATOR_ID_OFF = 4;  // creator_card_id within a delayed slot
static constexpr int DELAYED_SUBJECT_ID_OFF = 7;  // subject_card_id within a delayed slot
static const double LOG_LIFE_DENOM    = std::log1p(static_cast<double>(LIFE_NORMALIZER));
static const double LOG_LIBRARY_DENOM = std::log1p(static_cast<double>(LIBRARY_NORMALIZER));

static constexpr int STATE_HEADER_SIZE = 2 * PLAYER_BLOCK_SIZE + STEP_ONEHOT_SIZE + HEADER_FLAGS;
static constexpr int STACK_SLOT_SIZE   = STACK_HEAD_FIELDS + STACK_XAMT_FIELDS +
                                         STACK_QUAL_FIELDS + STACK_MODE_SLOTS +
                                         STACK_TGT_SLOTS * STACK_TGT_FIELDS;
static constexpr float TURN_NORMALIZER = 50.0f; // divisor for turn fields

// Unified entity-reference slot space width (see the layout comment above):
// 48 self perms + 48 opp perms + 12 stack slots.
static constexpr int N_ENTITY_REF_SLOTS = 2 * MAX_BATTLEFIELD_SLOTS + MAX_STACK_DISPLAY;
static_assert(N_ENTITY_REF_SLOTS == 108, "entity-ref space width documented as 108");
static_assert(STACK_SLOT_SIZE == 37, "stack slot layout documented as 37 floats");
static_assert(STATE_HEADER_SIZE == 36, "state header documented as 36 floats");

// ── Absolute block offsets (OFFSET_CHAIN) ────────────────────────────────────
// The state vector's blocks in serialization order, each derived from the widths
// above. This is the C++ counterpart of train/env.py's offset chain (and the one
// src/actor/obs_builder.cpp consumes, so the actor no longer keeps a third copy).
// The closing static_assert against STATE_SIZE means an edit that adds, removes,
// resizes, or REORDERS a block fails to compile unless STATE_SIZE moves with it.
static constexpr int SELF_PERM_START      = STATE_HEADER_SIZE;
static constexpr int OPP_PERM_START       = SELF_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int STACK_START          = OPP_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int GY_START             = STACK_START + MAX_STACK_DISPLAY * STACK_SLOT_SIZE;
static constexpr int EXILE_START          = GY_START + 2 * MAX_GY_SLOTS * GY_SLOT_SIZE;
static constexpr int HAND_START           = EXILE_START + 2 * MAX_GY_SLOTS * EXILE_SLOT_SIZE;
static constexpr int MATCH_CTX_START      = HAND_START + MAX_HAND_SLOTS * CARD_ID_SLOT_SIZE;
static constexpr int LIBRARY_CTX_START    = MATCH_CTX_START + MATCH_CTX_SIZE;
static constexpr int CUR_TURN_IDX         = LIBRARY_CTX_START + LIBRARY_CTX_SIZE;
static constexpr int KNOWN_TOP_LIB_START  = CUR_TURN_IDX + CUR_TURN_SIZE;
static constexpr int OPP_KNOWN_HAND_START = KNOWN_TOP_LIB_START + KNOWN_TOP_LIBRARY_SIZE * CARD_ID_SLOT_SIZE;
static constexpr int PENDING_DECISION_START = OPP_KNOWN_HAND_START + MAX_HAND_SLOTS * CARD_ID_SLOT_SIZE;
static constexpr int EXTRAS_START         = PENDING_DECISION_START + PENDING_DECISION_SIZE;
// Within the extras block: the priority-window context, the mulligan state, the
// MandatoryChoice one-hot, then the sideboard context.
static constexpr int EXTRAS_PRIORITY_START = EXTRAS_START + EXTRAS_SCALARS;
static constexpr int EXTRAS_MULLIGAN_START = EXTRAS_PRIORITY_START + EXTRAS_PRIORITY_SIZE;
static constexpr int EXTRAS_MC_ONEHOT_START = EXTRAS_MULLIGAN_START + EXTRAS_MULLIGAN_SIZE;
static constexpr int EXTRAS_SB_CTX_START  = EXTRAS_MC_ONEHOT_START + N_MANDATORY_CHOICES;
static constexpr int EXTRAS_END           = EXTRAS_SB_CTX_START + EXTRAS_SB_CTX_SIZE;
static constexpr int SELF_LIVE_LIB_START  = EXTRAS_END;
static constexpr int SELF_DECK_MAIN_START = SELF_LIVE_LIB_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int SELF_DECK_SIDE_START = SELF_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_MAIN_START  = SELF_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_SIDE_START  = OPP_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * OPP_DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_SIDE_END    = OPP_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * OPP_DECKLIST_SLOT_SIZE;
// Mana development (self half, then the opponent's).
static constexpr int MANA_DEV_START       = OPP_DECK_SIDE_END;
static constexpr int MANA_DEV_OPP_START   = MANA_DEV_START + MANA_DEV_SELF_SIZE;
static constexpr int MANA_DEV_END         = MANA_DEV_OPP_START + MANA_DEV_OPP_SIZE;
// Log-scaled vitals (self half, then the opponent's).
static constexpr int LOG_VITALS_START     = MANA_DEV_END;
static constexpr int LOG_VITALS_OPP_START = LOG_VITALS_START + LOG_VITALS_PLAYER_SIZE;
static constexpr int LOG_VITALS_END       = LOG_VITALS_OPP_START + LOG_VITALS_PLAYER_SIZE;
// Per-turn counters (self half, then the opponent's).
static constexpr int PER_TURN_START       = LOG_VITALS_END;
static constexpr int PER_TURN_OPP_START   = PER_TURN_START + PER_TURN_PLAYER_SIZE;
static constexpr int PER_TURN_END         = PER_TURN_OPP_START + PER_TURN_PLAYER_SIZE;
// Pending delayed triggers close the vector.
static constexpr int DELAYED_START        = PER_TURN_END;
static constexpr int DELAYED_END          = DELAYED_START + DELAYED_SLOTS * DELAYED_SLOT_SIZE;
static_assert(DELAYED_END == STATE_SIZE,
              "state-vector offset chain must end exactly at STATE_SIZE — a block "
              "was added/resized/reordered without updating STATE_SIZE (and the "
              "layout comment above, train/env.py, and train/extractor.py)");

// Effective-keyword multi-hot vocabulary for the permanent slots (offsets [24-39]),
// in serialized order. Exactly the engine-implemented keyword set; queried per slot
// via permanent_has_keyword (post-layer, so granted/removed keywords are honored).
static constexpr const char* OBS_KEYWORDS[] = {
    "Flying", "Reach", "First Strike", "Double Strike", "Deathtouch", "Lifelink",
    "Trample", "Vigilance", "Menace", "Haste", "Defender", "Indestructible",
    "Hexproof", "Shroud", "Ward", "Flash",
};
static_assert(sizeof(OBS_KEYWORDS) / sizeof(OBS_KEYWORDS[0]) == N_OBS_KEYWORDS,
              "OBS_KEYWORDS must have exactly N_OBS_KEYWORDS entries");

// Card identity is serialized as a single normalized id float per slot:
//   idx >= 0 -> idx / N_CARD_TYPES ;  empty/unknown -> -1.0 / N_CARD_TYPES.
// The policy network (extractor.py) maps these back to ids and looks them up in
// a learned nn.Embedding, so the observation cost is decoupled from vocab size.
inline float norm_card_id(int idx) {
    return (idx >= 0 ? static_cast<float>(idx) : -1.0f) / static_cast<float>(N_CARD_TYPES);
}

// Normalize an entity-reference slot index (see the layout comment): -1/none -> 0.0,
// slot idx -> (idx + 1) / N_ENTITY_REF_SLOTS, so real refs live in (0, 1] and can
// never collide with the "none" sentinel. Decode: round(v * 108) - 1.
inline float norm_ref(int idx) {
    return static_cast<float>(idx + 1) / static_cast<float>(N_ENTITY_REF_SLOTS);
}

// Log-scaled count for the LOG VITALS block: log1p(max(v, 0)) / denom, where denom
// is LOG_LIFE_DENOM / LOG_LIBRARY_DENOM. The clamp keeps a momentarily-negative life
// total (dead, SBA not yet run) inside log1p's domain and reads as 0.0, the same
// value a player at exactly 0 life gets. Computed in double and narrowed once, so
// every consumer of the layout (serializer and the AZ actor, which shares this
// serializer) produces bit-identical floats.
inline float norm_log_count(int v, double denom) {
    return static_cast<float>(std::log1p(static_cast<double>(v > 0 ? v : 0)) / denom);
}

// viewer: which player's perspective to fill from. Zone::UNKNOWN defaults to the priority player.
void populate_gamestate(GameState* gs, Zone::Ownership viewer = Zone::UNKNOWN);
void populate_query(Query* q, const std::vector<LegalAction>& actions);

// Card vocab index for an action's source entity (or a stack entity): the card's
// vocab index, TOKEN_SENTINEL for a token, or -1 for a non-card source. Walks the
// Permanent(token) -> CardData -> Ability.source(Permanent(token)/CardData) chain.
// Single source for every action/stack id the observation emits (populate_query,
// the stack block).
int action_card_vocab_idx(Entity e);
// Same, but honoring a modal-DFC back-face play: such an action's source entity
// carries the FRONT face's CardData, so the entity overload above would report
// the front face. When the action plays the back face, this resolves the back
// face's name instead. Use this for action emission/logging so the emitted id
// matches the face actually being played.
int action_card_vocab_idx(const LegalAction& la);
// Returns a reference to a reused internal scratch buffer (valid until the next
// serialize_state call) to avoid a ~135 KB heap allocation on every machine-mode
// decision. Consume it (e.g. fwrite) before calling serialize_state again.
const std::vector<float>& serialize_state(const GameState* gs);

#endif /* MACHINE_IO_H */
