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
// Fixed-size state vector layout (STATE_SIZE = 6354 floats):
// Card identity is a single normalized id float per slot (see norm_card_id):
// idx/N_CARD_TYPES, or -1/N_CARD_TYPES for empty/unknown. The id is NOT a one-hot.
//
//  [0-9]      Self player block (10 floats):
//               life/20, hand_ct/10, poison/10, mana[W,U,B,R,G,C]/10, energy/10
//  [10-19]    Opponent player block (10 floats, same layout)
//  [20-32]    Current step one-hot (13 steps: UNTAP..CLEANUP, includes FIRST_STRIKE_DAMAGE)
//  [33]       1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [34]       1.0 if self is Player A, 0.0 if self is Player B
//  [35]       Stack size / 10.0
//
//  [36-1859]     Self permanents: 48 slots x 38 floats = 1824
//  [1860-3683]   Opp permanents:  48 slots x 38 floats = 1824
//                Per slot (offsets within the slot):
//                  [0]  power / 10
//                  [1]  toughness / 10
//                  [2]  is_tapped
//                  [3]  is_attacking
//                  [4]  is_blocking
//                  [5]  has_summoning_sickness
//                  [6]  damage / 10
//                  [7]  controller_is_self
//                  [8]  is_creature
//                  [9]  is_land
//                  [10] loyalty / 10 (planeswalkers; 0 otherwise)
//                  [11] p1p1_net / 10 — net (+1/+1 minus -1/-1) counters, SIGNED
//                       (distinguishes persistent counters from EOT pumps; both are
//                       already folded into effective P/T above)
//                  [12] other_counters / 10 — total counters of every other kind
//                       (excl. P1P1/M1M1/LOYALTY; the card-id embedding disambiguates
//                       the kind: Saga -> lore, Chalice -> charge, ...)
//                  [13] attached_to_ref (norm_ref) — for equipment/auras: the slot of
//                       the permanent this is attached to
//                  [14] attached_by_ref (norm_ref) — for creatures: the slot of the
//                       equipment/aura attached to this
//                  [15] attack_target_ref (norm_ref) — attacked planeswalker's slot;
//                       0.0 while is_attacking means "attacking the player"
//                  [16] blocking_target_ref (norm_ref) — the attacker this blocker blocks
//                  [17] is_blocked — attacker was blocked at declare-blockers; stays
//                       blocked even if all blockers leave (CR 509.1h)
//                  [18] is_phased_out — phased-out permanents ARE serialized (with this
//                       flag set) even though the rules treat them as nonexistent
//                       (CR 702.26e), so the model can anticipate the phase-in
//                  [19-34] effective keyword multi-hot x16, OBS_KEYWORDS order
//                       (post-layer, via permanent_has_keyword)
//                  [35] chosen_name_id — normalized vocab id of Permanent::chosen_name,
//                       the card name chosen on this permanent (Pithing Needle /
//                       Disruptor Flute named card, Petrified Hamlet named land);
//                       -1/N_CARD_TYPES sentinel when no name is chosen
//                  [36] returnable_exile_id — normalized vocab id of the most recently
//                       exiled card linked to this permanent that STILL has a return path
//                       (Static Prison holding a real card, Flickerwisp/Phelia EOT blink);
//                       -1/N_CARD_TYPES sentinel when none (no return, e.g. Skyclave
//                       Apparition, or nothing exiled). See returnable_exiled_card().
//                  [37] card_id (LAST)
//                Empty slots: 35 zeros + chosen_name sentinel + returnable_exile sentinel
//                + card_id sentinel (all three -1/N_CARD_TYPES).
//
//  [3684-4127]   Stack: 12 slots x 37 floats = 444 (slot 0 = top of stack)
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
//  [4128-4191]   Self graveyard: 64 slots x 1 float = 64
//  [4192-4255]   Opp graveyard:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty). RECENCY order: slot 0 is the
//                most recent arrival (sorted by Zone::distance_from_top).
//
//  [4256-4319]   Self exile: 64 slots x 1 float = 64
//  [4320-4383]   Opp exile:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty). RECENCY order like the
//                graveyards (slot 0 = most recent arrival). All exile is public in
//                this engine, so both zones are fully serialized.
//
//  [4384-4393]   Self hand: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty)
//
//  [4394-4905]   Action history: 128 entries x 4 floats = 512 (newest first)
//                Per entry: category / ACTION_CATEGORY_MAX,
//                           card_vocab_idx / N_CARD_TYPES (or -1/N_CARD_TYPES sentinel),
//                           is_self (1.0 = viewer's action, 0.0 = opponent's),
//                           turn / 50.0 (the game turn when the action was taken)
//                Empty entries (beyond action_history_len) are all zeros.
//
//  [4906-4909]   Match context (4 floats, all 0.0 in single-game mode):
//                game_number / 3.0, self_match_wins / 2.0,
//                opp_match_wins / 2.0, is_sideboard_phase (0.0 or 1.0)
//
//  [4910-4912]   Library & post-board context (3 floats):
//                self_library_ct / 60.0, opp_library_ct / 60.0,
//                is_post_board (1.0 if game 2+ of bo3, else 0.0)
//
//  [4913]        Current game turn / 50.0
//
//  [4914-4918]   Known top-5 library cards for the viewer: 5 slots x 1 float = 5
//                Per slot: card_id (sentinel = unknown). Index 0 is the top of
//                the library. Entries are set when a card is placed on top (e.g.
//                Ponder, Brainstorm, Rearrange) and cleared when shuffled.
//
//  [4919-5942]   Opponent revealed-cards multi-hot (N_CARD_TYPES floats, zeros =
//                none seen yet). Binary "has the opponent-of-viewer ever revealed
//                card X this match"; accumulated across the games of a bo3 and
//                persists over the per-game ECS reset. Set whenever an opponent
//                card enters a public zone (battlefield/stack/graveyard/exile) or
//                is revealed by a tutor. This is the only vocab-width block.
//
//  [5943-5952]   Known opponent-hand cards: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty/unknown). The specific
//                identities of opponent-hand cards the viewer has had revealed
//                (Duress/Thoughtseize/tutor) and that are still in hand. Unlike
//                the multi-hot above this tracks the exact card and a slot clears
//                when that card leaves the hand for another zone.
//
//  [5953-5954]   Pending decision context: 2 floats.
//                [5953] card_id of the spell/ability currently making a
//                mid-resolution choice (target select, dig/scry/surveil pick,
//                search, discard, modal, ...; sentinel = none). Set via
//                PendingDecisionScope — the source may not be on the stack yet,
//                since targets are announced before the spell moves there
//                (CR 601.2b/c), so this is the only place the observation shows
//                WHAT is asking for the current choice.
//                [5954] 1.0 if that source's controller is the viewer, else 0.0
//                (e.g. 0.0 while choosing a card for the opponent's Thoughtseize).
//
//  [5955-5976]   Global extras (22 floats):
//                  [5955] self lands_played_this_turn / 10
//                  [5956] opp  lands_played_this_turn / 10
//                  [5957] viewer_has_priority (1.0 = the viewer holds priority)
//                  [5958] self is_monarch (CR 725)
//                  [5959] opp  is_monarch
//                  [5960] self city's blessing (CR 702.131c)
//                  [5961] opp  city's blessing
//                  [5962] self revolt (a permanent self controlled left the battlefield this turn)
//                  [5963] opp  revolt
//                  [5964] self pending extra turns / 3
//                  [5965] opp  pending extra turns / 3
//                  [5966] is_day  (CR 731.1; both 0.0 = neither)
//                  [5967] is_night
//                  [5968-5973] MandatoryChoice one-hot x6 (NONE at index 0, then
//                       DECLARE_ATTACKERS_CHOICE, DECLARE_BLOCKERS_CHOICE,
//                       CLEANUP_DISCARD, CHOOSE_ENTITY, ASSIGN_COMBAT_DAMAGE_CHOICE)
//                  [5974] self_plays_first — the viewer is the starting player of the
//                       game this observation PERTAINS TO. In-game that is the current
//                       game (Game::pregame.a_goes_first); during a bo3 between-games
//                       sideboard phase it is the UPCOMING game, whose starting player
//                       play_bo3_match has already decided (the loser of the game that
//                       just ended) before either sideboard stage runs. Boarding plans
//                       differ substantially on the play vs the draw, and at 1-1 the
//                       upcoming starting player is otherwise unrecoverable from the
//                       observation.
//                  [5975] sideboard swaps completed this phase / SIDEBOARD_SWAP_CAP
//                       (0.0 outside the phase)
//                  [5976] sideboard maindeck drift, (d + 1) / 2 so -1/0/+1 map to
//                       0.0/0.5/1.0 and "balanced" is the 0.5 midpoint. Always 0.5
//                       outside the phase. The encoding is kept, but the 0.0 pole is
//                       unreachable: the menu is IN-FIRST, so drift is only ever 0
//                       or +1 and only 0.5/1.0 are ever emitted (see
//                       run_sideboard_phase in src/game_driver.cpp).
//
//  ── Deck-identity tail blocks ────────────────────────────────────────────────
//  Each slot is (card_id, count): card_id via norm_card_id (empty slot = -1
//  sentinel), count normalized /4.0 (basics may exceed 1.0). Slots are packed
//  ascending by vocab index with NO holes (a testable invariant that keeps the
//  encoding byte-stable across actors). Overflow (more distinct names than slots)
//  or a name absent from the vocab is a fatal_error, never a silent truncation.
//
//  [5977-6072]   Self LIVE library: 48 slots x (card_id, count) = 96.
//                The viewer's LIBRARY zone tallied live at serialization time, so
//                cards leaving/returning to the library are always reflected.
//                Viewer-only — the opponent's live library stays hidden.
//
//  [6073-6168]   Self LIVE maindeck:  48 slots x (card_id, count) = 96.
//  [6169-6200]   Self LIVE sideboard: 16 slots x (card_id, count) = 32.
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
//  [6201-6296]   Opponent-of-viewer REGISTERED maindeck: 48 slots x (card_id, count) = 96.
//  [6297-6328]   Opponent-of-viewer REGISTERED sideboard: 16 slots x (card_id, count) = 32.
//                (Registered, so only 15 slots can ever fill; the width just
//                follows DECKLIST_SIDE_SLOTS.)
//                The opponent's decklist as REGISTERED at match start (open-decklist
//                ruleset), read from deck_state. FROZEN for the whole match: a bo3's
//                sideboard swaps never touch these blocks, so you know the opponent's
//                registered 75 but NOT which 60 of it they boarded into for game 2+.
//                That split is hidden information, and the MCTS determinizer models
//                it as such (see `opp_sideboard_hidden` in search_server.cpp).
//
//  ── Mana development ─────────────────────────────────────────────────────────
//  A summary of each player's MANA BASE, which nothing else in the observation
//  states: the player-block "mana" floats are the FLOATING pool (~always empty at a
//  decision point) and the land-play limit was never serialized at all, so
//  castability had to be inferred from 48 raw permanent slots through pooling.
//  Counts and mana are normalized by MANA_COUNT_NORMALIZER, land drops by
//  LAND_DROPS_NORMALIZER. Self first, then the opponent, like every other block.
//
//  [6329-6339]   Self mana development (11 floats):
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
//                  [10]  max_affordable_cmc_proxy — a PROXY, currently equal to
//                        potential_total: the engine runs no payment solver here, and a
//                        source's color feasibility/multi-mana output is not resolved
//                        against a concrete cost. The slot is kept deliberately so a real
//                        payer-based "highest mana value castable now" can replace it
//                        WITHOUT another layout break (every checkpoint dies on a layout
//                        change, so the reserved float is the cheap half of that trade).
//  [6340-6349]   Opponent mana development (10 floats): the same fields MINUS
//                lands_in_hand, i.e. potential_W/U/B/R/G/C, potential_total,
//                lands_in_play, land_drops_remaining, max_affordable_cmc_proxy.
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
//  [6350-6351]   Self log vitals (2 floats): log_life, log_library
//  [6352-6353]   Opponent log vitals (2 floats): same two fields. Library size is
//                public information, so unlike lands_in_hand above there is nothing
//                to withhold from the opponent half.

static constexpr int STATE_SIZE             = 6354;
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
static constexpr int PERM_SLOT_SIZE    = 38;   // 11 stat/combat/type + 2 counters + 4 refs + 2 flags + 16 keywords + chosen-name id + returnable-exile id + card-id float
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
static constexpr int CARD_ID_SLOT_SIZE = 1;    // GY / exile / hand / known-top / known-opp-hand
static constexpr int GY_SLOT_SIZE      = CARD_ID_SLOT_SIZE;
static constexpr int STACK_HEAD_FIELDS = 3;    // controller_is_self + card id + is_spell
static constexpr int STACK_XAMT_FIELDS = 1;    // x_or_amount / 10
static constexpr int STACK_QUAL_FIELDS = 7;    // is_copy, kicked, flashback, evoke, escape, offspring, impending
static constexpr int HIST_ENTRY_SIZE   = 4;    // category, card id, is_self, turn/50
static constexpr int MATCH_CTX_SIZE    = 4;    // game#, self wins, opp wins, is_sideboard_phase
static constexpr int LIBRARY_CTX_SIZE  = 3;    // self lib/60, opp lib/60, is_post_board
static constexpr int CUR_TURN_SIZE     = 1;    // current turn / 50
static constexpr int PENDING_DECISION_SIZE = 2; // source card id + ctrl_is_self
static constexpr int EXTRAS_SCALARS    = 13;   // lands x2, priority, monarch x2, blessing x2,
                                               // revolt x2, extra turns x2, is_day, is_night
static constexpr int EXTRAS_SB_CTX_SIZE = 3;   // self_plays_first + swaps made + maindeck drift
static constexpr int DECKLIST_SLOT_SIZE = 2;   // card id + count per decklist slot
// Mana-development block (see the layout comment above). The self half carries one
// extra field — lands_in_hand — that the opponent half cannot (hidden information).
static constexpr int MANA_DEV_COLORS    = 6;   // potential W, U, B, R, G, C
static constexpr int MANA_DEV_SELF_SIZE = 11;  // 6 colors + total + lands_in_play +
                                               // lands_in_hand + land_drops + max-cmc proxy
static constexpr int MANA_DEV_OPP_SIZE  = 10;  // same minus lands_in_hand
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
static constexpr int HAND_START           = EXILE_START + 2 * MAX_GY_SLOTS * GY_SLOT_SIZE;
static constexpr int HIST_START           = HAND_START + MAX_HAND_SLOTS * CARD_ID_SLOT_SIZE;
static constexpr int MATCH_CTX_START      = HIST_START + ACTION_HISTORY_SIZE * HIST_ENTRY_SIZE;
static constexpr int LIBRARY_CTX_START    = MATCH_CTX_START + MATCH_CTX_SIZE;
static constexpr int CUR_TURN_IDX         = LIBRARY_CTX_START + LIBRARY_CTX_SIZE;
static constexpr int KNOWN_TOP_LIB_START  = CUR_TURN_IDX + CUR_TURN_SIZE;
static constexpr int REVEALED_START       = KNOWN_TOP_LIB_START + KNOWN_TOP_LIBRARY_SIZE * CARD_ID_SLOT_SIZE;
static constexpr int OPP_KNOWN_HAND_START = REVEALED_START + N_CARD_TYPES;
static constexpr int PENDING_DECISION_START = OPP_KNOWN_HAND_START + MAX_HAND_SLOTS * CARD_ID_SLOT_SIZE;
static constexpr int EXTRAS_START         = PENDING_DECISION_START + PENDING_DECISION_SIZE;
// Within the extras block: the MandatoryChoice one-hot, then the sideboard context.
static constexpr int EXTRAS_MC_ONEHOT_START = EXTRAS_START + EXTRAS_SCALARS;
static constexpr int EXTRAS_SB_CTX_START  = EXTRAS_MC_ONEHOT_START + N_MANDATORY_CHOICES;
static constexpr int EXTRAS_END           = EXTRAS_SB_CTX_START + EXTRAS_SB_CTX_SIZE;
static constexpr int SELF_LIVE_LIB_START  = EXTRAS_END;
static constexpr int SELF_DECK_MAIN_START = SELF_LIVE_LIB_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int SELF_DECK_SIDE_START = SELF_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_MAIN_START  = SELF_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_SIDE_START  = OPP_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_SIDE_END    = OPP_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * DECKLIST_SLOT_SIZE;
// Mana development (self half, then the opponent's) closes the vector.
static constexpr int MANA_DEV_START       = OPP_DECK_SIDE_END;
static constexpr int MANA_DEV_OPP_START   = MANA_DEV_START + MANA_DEV_SELF_SIZE;
static constexpr int MANA_DEV_END         = MANA_DEV_OPP_START + MANA_DEV_OPP_SIZE;
// Log-scaled vitals (self half, then the opponent's) closes the vector.
static constexpr int LOG_VITALS_START     = MANA_DEV_END;
static constexpr int LOG_VITALS_OPP_START = LOG_VITALS_START + LOG_VITALS_PLAYER_SIZE;
static constexpr int LOG_VITALS_END       = LOG_VITALS_OPP_START + LOG_VITALS_PLAYER_SIZE;
static_assert(LOG_VITALS_END == STATE_SIZE,
              "state-vector offset chain must end exactly at STATE_SIZE — a block "
              "was added/resized/reordered without updating STATE_SIZE (and the "
              "layout comment above, train/env.py, and train/extractor.py)");

// Effective-keyword multi-hot vocabulary for the permanent slots (offsets [19-34]),
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
// Single source so the index logged for a chosen action (input_logger) matches the
// index emitted for that same action in the BQUERY (populate_query).
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
