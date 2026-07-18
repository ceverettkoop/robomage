#include "game_driver.h"

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "action_processor.h"
#include "card_db.h"
#include "companion.h"
#include "classes/deck.h"
#include "classes/deck_state.h"
#include "classes/game.h"
#include "classes/match_context.h"
#include "classes/match_state.h"
#include "cli_output.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/color_identity.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/effect.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/token.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "input_logger.h"
#include "machine_io.h"
#include "search_server.h"
#include "snapshot.h"
#include "systems/orderer.h"
#include "systems/stack_manager.h"
#include "systems/state_manager.h"

// Per-game seed for a bo3 game. Salt 0 (the real line) reproduces today's exact
// seed; a nonzero g_sim_seed_salt (set by a sideboard-root DETERMINIZE) folds in
// to sample a distinct next-game deal per searched world. One engine-side
// definition so the C++ actor and the Python stdio driver stay in parity.
static unsigned int match_game_seed(const MatchContext &ctx);

std::string RESOURCE_DIR;
Coordinator global_coordinator = Coordinator();
Deck DEFAULT_DECK_ONE;
Deck DEFAULT_DECK_TWO;
Game cur_game;
bool has_human_player = false;
bool human_player_is_a = false;
// Narrative spectator: pins game_log_private output to one player's view even
// when both seats are driven over the machine protocol (the TUI play mode wants
// one-sided narrative without routing input through the human-keyboard path).
bool log_viewer_set = false;
Zone::Ownership log_viewer_owner = Zone::UNKNOWN;

GameState gs;
const GameState *gs_ptr = &gs;
std::string replay_file_path;
bool replay_mode = false;
bool machine_mode = false;
std::string deck_a_name = "delver";
std::string deck_b_name = "delver";
bool seed_override = false;
unsigned int seed_value = 0;
bool no_shuffle = false;
bool narrative_mode = false;
bool bo3_mode = false;
// Machine mode writes no decision log unless --log-decisions is passed (training
// runs millions of episodes). CLI/interactive games always log — it's their save-game.
bool log_decisions_flag = false;
// --broadcast-steps: at every forced auto-pass (a priority window whose only
// legal action is pass), emit a passive BSTATE frame — the BQUERY payload with
// no stdin read — so an observing front end can render each step instead of
// fast-forwarding to the next real decision. Consumes no input and records
// nothing, so replays, action history, and search mirrors are unaffected.
bool broadcast_steps_mode = false;
std::vector<std::string> battlefield_a_cards;
std::vector<std::string> battlefield_b_cards;
std::vector<std::string> graveyard_a_cards;
std::vector<std::string> graveyard_b_cards;
std::vector<std::string> exile_a_cards;
std::vector<std::string> exile_b_cards;
std::vector<std::string> sideboard_a_cards;
std::vector<std::string> sideboard_b_cards;
// Test-harness starting-life overrides (-1 = use the normal 20). Let scenarios exercise
// life-payment costs (fetch lands, Toxic Deluge, phyrexian mana) at a chosen life total.
int life_a_override = -1;
int life_b_override = -1;

// match state (accessible for state serialization)
int match_game_number = -1;  // -1 = single game, 0-2 = bo3 game index
int match_wins_a = 0;
int match_wins_b = 0;
bool sideboard_phase = false;
Zone::Ownership sideboard_phase_player = Zone::UNKNOWN;

// The whole bo3 match's between-game state in one snapshottable value struct.
// play_bo3_match dispatches over its `stage`; the legacy globals above remain
// authoritative-in-sync (machine_io/input_logger read them). Exposed (Stage 2)
// so the match-scoped game snapshot can capture/restore it across init_ecs().
MatchContext g_match_ctx;

// Incremented each init_ecs(); the snapshot captures its value so a restore can
// tell a card_db.clear()+reload happened across the rollback (see game_driver.h).
uint64_t g_card_db_generation = 0;

// True while play_single_game's decision loop is running. Later batches gate
// blocking fallbacks on it: a prompt fired OUTSIDE the loop (pregame SBE on
// preplaced permanents) has no loop top to suspend to, so it must block inline.
static bool g_in_main_loop = false;
bool in_main_loop() { return g_in_main_loop; }

EcsSystems init_ecs() {
    card_db.clear();
    ++g_card_db_generation;
    global_coordinator.Init();
    global_coordinator.RegisterComponent<Ability>();
    global_coordinator.RegisterComponent<CardData>();
    global_coordinator.RegisterComponent<ColorIdentity>();
    global_coordinator.RegisterComponent<Creature>();
    global_coordinator.RegisterComponent<Damage>();
    global_coordinator.RegisterComponent<Permanent>();
    global_coordinator.RegisterComponent<Player>();
    global_coordinator.RegisterComponent<Spell>();
    global_coordinator.RegisterComponent<Zone>();
    global_coordinator.RegisterComponent<Token>();

    auto orderer = global_coordinator.RegisterSystem<Orderer>();
    auto state_manager = global_coordinator.RegisterSystem<StateManager>();
    auto stack_manager = global_coordinator.RegisterSystem<StackManager>();
    Orderer::init();
    StateManager::init();
    StackManager::init();

    return {orderer, state_manager, stack_manager};
}

// Returns winner: Zone::PLAYER_A (1) or Zone::PLAYER_B (2)
int play_single_game(EcsSystems &sys, const Deck &deck_a, const Deck &deck_b,
                     bool player_a_goes_first, unsigned int seed) {
    // A snapshot from a previous game of the match must never be restorable
    // into this game's fresh ECS — UNLESS a match-scoped snapshot is live: a
    // sideboard-rooted MCTS sim rolls forward into THIS game precisely to search,
    // and dropping its snapshot here would strand the rollback. The guard keeps
    // the original invariant for ordinary game starts (no match snapshot live).
    if (!snapshot_any_match_scope_live()) snapshot_release_all();
    // Refresh the static decklist store for the state serializer. Called each game
    // start so a bo3's post-sideboard (mutated) Deck structs are reflected in the
    // opponent-decklist observation blocks (see deck_state.h / machine_io.h tail).
    deck_state_set(Zone::PLAYER_A, deck_a);
    deck_state_set(Zone::PLAYER_B, deck_b);
    cur_game = Game(seed);
    cur_game.generate_players(deck_a, deck_b);
    // Apply starting-life overrides (test harness --life-a/--life-b) before any play.
    if (life_a_override >= 0)
        global_coordinator.GetComponent<Player>(cur_game.player_a_entity).life_total = life_a_override;
    if (life_b_override >= 0)
        global_coordinator.GetComponent<Player>(cur_game.player_b_entity).life_total = life_b_override;
    sys.orderer->generate_libraries(deck_a, deck_b);

    // Pre-game section header: mulligans, test-harness fiat setup, and the CR 103.6b
    // opening-hand actions below all print under PREGAME, before the first turn header.
    cli_print_pregame_header();
    sys.orderer->draw_hands();
    sys.orderer->do_london_mulligan(player_a_goes_first);

    // place pre-set battlefield permanents
    std::vector<Entity> preplaced;
    if (!battlefield_a_cards.empty()) {
        auto placed = sys.orderer->place_on_battlefield(battlefield_a_cards, Zone::PLAYER_A);
        preplaced.insert(preplaced.end(), placed.begin(), placed.end());
    }
    if (!battlefield_b_cards.empty()) {
        auto placed = sys.orderer->place_on_battlefield(battlefield_b_cards, Zone::PLAYER_B);
        preplaced.insert(preplaced.end(), placed.begin(), placed.end());
    }
    // pre-set graveyard cards (test harness): no Permanent/Creature components are
    // attached, so they are skipped by the summoning-sickness clear below.
    if (!graveyard_a_cards.empty())
        sys.orderer->place_in_graveyard(graveyard_a_cards, Zone::PLAYER_A);
    if (!graveyard_b_cards.empty())
        sys.orderer->place_in_graveyard(graveyard_b_cards, Zone::PLAYER_B);
    // pre-set exile / sideboard ("outside the game") cards so zone-change effects
    // that pull from those zones (e.g. Karn's -2) can be exercised in isolation.
    if (!exile_a_cards.empty())
        sys.orderer->place_in_zone(exile_a_cards, Zone::PLAYER_A, Zone::EXILE);
    if (!exile_b_cards.empty())
        sys.orderer->place_in_zone(exile_b_cards, Zone::PLAYER_B, Zone::EXILE);
    if (!sideboard_a_cards.empty())
        sys.orderer->place_in_zone(sideboard_a_cards, Zone::PLAYER_A, Zone::SIDEBOARD);
    if (!sideboard_b_cards.empty())
        sys.orderer->place_in_zone(sideboard_b_cards, Zone::PLAYER_B, Zone::SIDEBOARD);
    // Companion (CR 702.139): instantiate each player's chosen companion as a sideboard entity (if
    // not already one) and record it, gated on the starting deck meeting the companion's restriction.
    setup_companions(deck_a, deck_b, sys.orderer);
    // run SBE once to attach Permanent/Creature components, then clear summoning sickness
    if (!preplaced.empty()) {
        sys.state_manager->state_based_effects(cur_game, sys.orderer);
        for (auto e : preplaced) {
            if (global_coordinator.entity_has_component<Permanent>(e)) {
                global_coordinator.GetComponent<Permanent>(e).has_summoning_sickness = false;
            }
        }
    }

    // CR 103.6: after mulligans resolve (and any test-harness fiat setup above), each player
    // in APNAP order — starting player first — may take opening-hand actions (CR 103.6b,
    // Leyline of the Void: begin the game with it on the battlefield). If anything moved,
    // run state-based effects so a card put onto the battlefield gets its Permanent
    // component before the first turn begins.
    if (sys.orderer->do_opening_hand_actions(player_a_goes_first))
        sys.state_manager->state_based_effects(cur_game, sys.orderer);

    cur_game.player_a_turn = player_a_goes_first;
    cur_game.player_a_has_priority = player_a_goes_first;

    size_t prev_turn = (size_t)-1;
    g_in_main_loop = true;
    while (!cur_game.ended || search_intercept_game_end()) {
        // A RESTORE that arrived mid-decision unwound to here; apply it before
        // anything reads game state, so this iteration re-derives (and re-emits)
        // the restored decision. (Applies GAME-scoped restores only.)
        search_apply_pending_restore();
        // A MATCH-scoped restore stays latched: it targets a sideboard root one
        // level up, so bail out of this game's loop and let play_bo3_match's
        // dispatcher apply it (its loop top) and re-enter the restored stage. The
        // returned winner is ignored — the dispatcher checks the pending restore
        // before reading it.
        if (search_match_restore_pending()) {
            g_in_main_loop = false;
            return cur_game.winner;
        }
        // Set by the RESOLUTION dispatch below when a resumed resolution ran to
        // completion inside its direct advance_step call: skip straight to the
        // post-resolution half of the iteration (the post-advance SBE + legal
        // actions) — exactly where today's single-call flow stands after
        // advance_step returns false. Turn-based actions / mandatory choices /
        // the pre-advance SBE already ran when the resolution first began, and
        // SBAs never apply mid-resolution.
        bool resolution_just_completed = false;
        // A suspended decision (pending_query.h) is emitted here — the FIRST
        // thing each iteration, before the turn header and before turn-based
        // actions / mandatory choices / SBE run (a suspended trigger placement
        // or combat sub-prompt must not let combat damage run underneath it).
        // The stored menu is re-emitted verbatim; loop-safe, so SNAPSHOT/
        // DETERMINIZE are legal here.
        if (cur_game.pending_query.active) {
            PendingQuery &pq = cur_game.pending_query;
            if (!pq.answered) {
                if (cur_game.player_a_has_priority != pq.chooser_is_a)
                    fatal_error("pending query: priority is not at the chooser");
                // Reset the pending-decision baseline to the arm-time value (0:
                // every family arms from a loop-top-driven flow with no ambient
                // pending decision). A SNAPSHOT at this decision is captured
                // while the scope below holds pending_decision_source =
                // pq.decision_source, so a RESTORE re-enters with that value
                // still set — without the reset the recreated scope would
                // capture it as its prev and leak it past the answer, desyncing
                // the restored line's next observation from the natural one
                // (the same hazard run_sideboard_phase documents). No-op on the
                // natural line.
                cur_game.pending_decision_source = 0;
                PendingDecisionScope pending(pq.decision_source);
                search_set_loop_safe(true);
                int choice = InputLogger::instance().get_input(pq.menu);
                search_set_loop_safe(false);
                // A RESTORE latched during the query unwound it: loop back
                // WITHOUT storing the answer so the restored state re-emits.
                if (search_restore_pending() || search_match_restore_pending()) continue;
                pq.answer = choice;
                pq.answered = true;
            }
            // Dispatch the latched answer back to the suspended flow. Each tag's
            // resume path lands with its conversion batch.
            switch (pq.tag) {
                case PendingQuery::ATTACK_TARGET:
                case PendingQuery::BLOCK_TARGET:
                    resume_combat_target_choice(cur_game);
                    break;
                case PendingQuery::DAMAGE_ASSIGN:
                    resume_damage_assignment(cur_game, sys.orderer);
                    // The resume may arm the NEXT pick's query (same or next
                    // attacker): loop back so the pending branch emits it before
                    // anything (turn-based actions, SBE) runs underneath it. When
                    // the assignment completes instead, fall through — turn-based
                    // actions find every attacker decided and deal combat damage,
                    // exactly as the single-call flow did.
                    if (cur_game.pending_query.active) continue;
                    break;
                case PendingQuery::TRIGGER_PLACE:
                    resume_trigger_placement(cur_game, sys.orderer);
                    // The resume may park the NEXT placement decision (the next
                    // ordering pick, or a trigger target): loop back so the
                    // pending branch emits it before anything (turn-based
                    // actions, SBE) runs underneath it. When the placement
                    // completes instead, fall through to the normal flow — the
                    // suspended state_based_effects call returned early
                    // mid-contract, and re-running it from the top is a provable
                    // no-op: process_turn_based_actions is idempotent per step
                    // (flag-guarded), the SBA fixpoint re-applies nothing (state
                    // unchanged but for the stack), and the trigger re-scan
                    // drains only the placement's own push_ability_onto_stack
                    // zone-change events, which no trigger matches — the same
                    // (empty) yield today's next SBE call gets from them.
                    if (cur_game.pending_query.active) {
                        if (cur_game.pending_query.answered)
                            fatal_error("trigger placement resume did not consume its latched answer");
                        continue;
                    }
                    break;
                case PendingQuery::RESOLUTION: {
                    // Re-enter the suspended resolution DIRECTLY via advance_step,
                    // skipping turn-based actions / mandatory choices / SBE — SBAs
                    // don't apply mid-resolution (CR 704.3 timing), and today the
                    // whole resolution completed inside one advance_step call, so
                    // nothing may run between its halves. The pass flags are still
                    // true (left set at suspension), so advance_step re-enters
                    // resolve_top, the phased resolve skips to the suspended point,
                    // and the pending ask consumes the latched answer.
                    bool advanced = cur_game.advance_step(sys.stack_manager, sys.orderer);
                    // Purity tripwire: a resumed resolution must consume the
                    // latched answer at the very ask that armed it — a still-
                    // answered query means the handler diverged on re-entry.
                    if (cur_game.pending_query.active && cur_game.pending_query.answered)
                        fatal_error("resolution resume did not consume its latched answer");
                    // Suspended again (a follow-up ask armed a new query): loop
                    // back so the pending branch emits it before anything runs.
                    if (advanced) continue;
                    // Completed (advance_step reset the pass flags and returned
                    // false): fall through to the post-resolution half below.
                    resolution_just_completed = true;
                    break;
                }
                default:
                    fatal_error("pending query dispatch not implemented for tag " +
                                std::to_string(static_cast<int>(pq.tag)));
            }
        }
        Zone::Ownership viewer = (has_human_player)
            ? (human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B)
            : Zone::UNKNOWN;

        if (!resolution_just_completed) {
            if ((!InputLogger::instance().is_machine_mode() || narrative_mode) && cur_game.turn != prev_turn) {
                cli_print_turn_header(cur_game.turn, cur_game.player_a_turn);
                prev_turn = cur_game.turn;
            }

            sys.state_manager->process_turn_based_actions(cur_game, sys.orderer);
            if (cur_game.is_mandatory_choice_pending()) {
                populate_gamestate(&gs, viewer);
                proc_mandatory_choice(cur_game, sys.orderer);
                continue;
            }
            sys.state_manager->state_based_effects(cur_game, sys.orderer);
            if (cur_game.ended) {
                if (!search_intercept_game_end()) break;
                continue;
            }
            // A trigger placement suspended inside state_based_effects: loop
            // back so the pending branch emits the parked decision before
            // advance_step (or anything else) runs underneath it.
            if (decision_suspended()) continue;
            if (cur_game.advance_step(sys.stack_manager, sys.orderer)) {
                continue;
            }
        }
        // Post-advance SBE: something resolved (or nothing advanced) this
        // iteration — run state-based effects before offering priority. A
        // resolution completed via the RESOLUTION dispatch lands here directly,
        // which is exactly where the single-call flow stood after its
        // advance_step returned false.
        sys.state_manager->state_based_effects(cur_game, sys.orderer);
        if (cur_game.ended) {
            if (!search_intercept_game_end()) break;
            continue;
        }
        // A trigger placement suspended inside the post-advance SBE: loop back
        // so the pending branch emits the parked decision before priority is
        // offered on a half-placed batch.
        if (decision_suspended()) continue;

        auto legal_actions = sys.state_manager->determine_legal_actions(cur_game, sys.orderer, sys.stack_manager);
        if (legal_actions.size() == 1) {
            // --broadcast-steps: surface this forced pass to the machine driver as
            // a passive BSTATE frame (no stdin read, nothing recorded). Suppressed
            // while a search is simulating (snapshot live) or unwinding a restore —
            // those lines are hypothetical and must not reach the observer.
            if (machine_mode && broadcast_steps_mode && !search_restore_pending()
                && !search_any_snapshot_live()) {
                Query q;
                populate_gamestate(&gs, viewer);
                populate_query(&q, legal_actions);
                cli_emit_machine_bstate(&q, &gs);
            }
            cur_game.pass_priority();
            if (!machine_mode) populate_gamestate(&gs, viewer);
            continue;
        }

        // In machine mode this populate is redundant: print_game_state early-returns,
        // and InputLogger::get_input re-runs populate_gamestate into its own buffer.
        // Skip the wasted full-entity scan + memset on every machine-mode decision.
        if (!machine_mode) {
            populate_gamestate(&gs, viewer);
            print_game_state(&gs);
        }
        // The priority decision is loop-safe: the whole state lives in cur_game +
        // ECS here, and re-entering the loop top on that data re-derives this
        // same decision (the snapshot round-trip CI test proves it byte-for-byte).
        search_set_loop_safe(true);
        int choice = InputLogger::instance().get_input(legal_actions);
        search_set_loop_safe(false);
        process_action(legal_actions[static_cast<size_t>(choice)], cur_game, sys.orderer);
    }
    // A real game end must never strand a suspended resolution (plan risk R7):
    // the winner is decided by state-based effects, which never run while a
    // resolution is parked, so an active frame/query here is a protocol bug.
    if (cur_game.resolution.active || cur_game.pending_query.active)
        fatal_error("game ended with a suspended resolution/pending query still parked");
    g_in_main_loop = false;
    return cur_game.winner;
}

// present sideboard choices to a player via the standard query mechanism.
// All persistent phase state (swap count, one-shot sided-in/out bookkeeping,
// and the OUT-menu resumption point) lives in `st` so the phase is resumable:
// on re-entry with st.pending_in_sb_idx >= 0 we re-derive the chosen IN card
// and resume directly at the OUT menu.
void run_sideboard_phase(Deck &deck, SideboardPhaseState &st) {
    Zone::Ownership player = st.player;
    sideboard_phase = true;
    sideboard_phase_player = player;
    // Repoint priority to the sideboarding player (the established engine pattern:
    // every prompt is issued with player_a_has_priority pointing at the chooser).
    // record_chosen_action's actor stamp and populate_query's per-action
    // controller_is_self flags both read this flag, so without the repoint both
    // would carry whatever the just-ended game left behind. cur_game is discarded
    // (replaced by Game(seed)) when the next game starts, so nothing to restore.
    cur_game.player_a_has_priority = (player == Zone::PLAYER_A);
    // Reset the pending-decision baseline to 0 for this phase. The OUT menu wraps
    // its query in a PendingDecisionScope(in_card_eid) whose RAII prev-restore
    // depends on the baseline: on the natural first pass it is 0. A resume after a
    // MATCH-scoped restore re-enters with the snapshot's in_card_eid still sitting
    // in cur_game.pending_decision_source, so without this reset the recreated
    // scope would capture in_card_eid as its prev and leak it into the post-swap IN
    // menu — desyncing the restored line's observation from the natural one. This
    // is a no-op in normal play (no decision is pending between games).
    cur_game.pending_decision_source = 0;
    const char *player_name = (player == Zone::PLAYER_A) ? "Player A" : "Player B";
    // Each card is a one-shot decision per phase: once moved, it cannot be
    // moved back. Prevents oscillation and makes the 15-swap cap easier to
    // avoid. Both sets reset between phases (i.e. for game 3 sideboarding).
    std::unordered_set<std::string> &sided_out_names = st.sided_out_names;
    std::unordered_set<std::string> &sided_in_names = st.sided_in_names;
    // Index lookups from filtered action list slots back to deck indices.
    std::vector<size_t> in_action_to_sb_idx;
    std::vector<size_t> out_action_to_md_idx;

    // Resume directly at the OUT menu if we re-entered mid-swap (a chosen IN card
    // is pending). Otherwise fall through to the normal IN-menu loop.
    bool resume_at_out = (st.pending_in_sb_idx >= 0);

    while (true) {
        // A MATCH-scoped restore latched during a simulation unwinds through here
        // (its target is this very sideboard root, one dispatcher level up): bail
        // without building the menu or mutating the deck; the dispatcher applies
        // the restore and re-enters this phase from the restored state.
        if (search_match_restore_pending()) return;
        // The chosen IN card carries across the IN→OUT menu step. On a resume
        // (st.pending_in_sb_idx >= 0) it is re-derived below and we skip straight
        // to the OUT menu; otherwise it is picked from the IN menu.
        size_t sb_idx;
        std::string card_in;
        Entity in_card_eid;

        if (resume_at_out) {
            // Re-entered mid-swap: re-derive the pending IN card and resume at the
            // OUT menu (recreating the same pending-decision context below).
            sb_idx = static_cast<size_t>(st.pending_in_sb_idx);
            card_in = deck.sideboard[sb_idx].second;
            in_card_eid = load_card(card_in);
            resume_at_out = false;
        } else {
            // build action list: index 0 = done, 1..N = sideboard cards to bring in
            std::vector<LegalAction> actions;
            actions.emplace_back(ActionType::SPECIAL_ACTION, "Done sideboarding");
            actions.back().category = ActionCategory::SIDEBOARD_DONE;

            in_action_to_sb_idx.clear();
            for (size_t i = 0; i < deck.sideboard.size(); i++) {
                if (sided_out_names.count(deck.sideboard[i].second)) continue;
                std::string desc = "Sideboard in: " + std::to_string(deck.sideboard[i].first) + "x " + deck.sideboard[i].second;
                actions.emplace_back(ActionType::SPECIAL_ACTION, desc);
                actions.back().category = ActionCategory::SIDEBOARD_IN;
                // load the card so we can get its vocab index for ML
                Entity card_eid = load_card(deck.sideboard[i].second);
                actions.back().source_entity = card_eid;
                in_action_to_sb_idx.push_back(i);
            }

            if (actions.size() <= 1) break;  // no sideboard cards available

            // in machine mode, cap sideboarding at 15 swaps to prevent infinite loops
            // (schedule-affecting, so a replayed machine log must apply the same cap)
            if (InputLogger::instance().is_machine_schedule() && st.sb_swaps >= 15) {
                game_log("%s hit sideboard swap limit (15), auto-finishing.\n", player_name);
                break;
            }

            game_log("\n%s sideboarding (%zu cards in sideboard):\n", player_name, deck.sideboard.size());

            populate_gamestate(&gs, player);
            print_game_state(&gs);
            // Loop-safe: re-entering run_sideboard_phase from the restored match
            // state re-derives this exact IN menu (the snapshot round-trip proves
            // it byte-for-byte), so SNAPSHOT/DETERMINIZE are legal here.
            search_set_loop_safe(true);
            int choice = InputLogger::instance().get_input(actions);
            search_set_loop_safe(false);
            // A RESTORE latched during the query (stdio command) unwinds via
            // choice 0; bail before acting so no swap is performed on rollback.
            if (search_match_restore_pending()) return;

            if (choice == 0) break;  // done

            // player chose to bring in a sideboard card
            sb_idx = in_action_to_sb_idx[static_cast<size_t>(choice - 1)];
            card_in = deck.sideboard[sb_idx].second;
            in_card_eid = actions[static_cast<size_t>(choice)].source_entity;
            // Record the OUT-menu resumption point (cleared once the swap completes).
            st.pending_in_sb_idx = static_cast<long>(sb_idx);
        }

        // now ask which main deck card to swap out
        std::vector<LegalAction> out_actions;
        out_action_to_md_idx.clear();
        for (size_t i = 0; i < deck.main_deck.size(); i++) {
            if (sided_in_names.count(deck.main_deck[i].second)) continue;
            std::string desc = "Sideboard out: " + std::to_string(deck.main_deck[i].first) + "x " + deck.main_deck[i].second;
            out_actions.emplace_back(ActionType::SPECIAL_ACTION, desc);
            out_actions.back().category = ActionCategory::SIDEBOARD_OUT;
            Entity card_eid = load_card(deck.main_deck[i].second);
            out_actions.back().source_entity = card_eid;
            out_action_to_md_idx.push_back(i);
        }

        // No eligible main-deck cards to swap out: undo the selected IN and end phase.
        if (out_actions.empty()) {
            game_log("No eligible cards to swap out — ending sideboard phase.\n");
            break;
        }

        game_log("Choose card to remove from main deck (replacing with %s):\n", card_in.c_str());
        // Expose the chosen IN card as the pending-decision source so the OUT
        // query's observation shows which card this cut is FOR (the
        // pending-decision context slots — see the machine_io.h layout).
        PendingDecisionScope pending(in_card_eid);
        populate_gamestate(&gs, player);
        // A restore latched before we reach the query (e.g. during the IN-menu
        // wait) must bail here — the OUT menu's choice 0 is a real swap, not a
        // no-op, so unwinding through it would corrupt the deck.
        if (search_match_restore_pending()) return;
        // Loop-safe: on rollback we re-enter with resume_at_out and re-derive this
        // exact OUT menu (with its pending-decision context), so it round-trips.
        search_set_loop_safe(true);
        int out_choice = InputLogger::instance().get_input(out_actions);
        search_set_loop_safe(false);
        // A RESTORE latched during THIS query unwinds via choice 0; bail before
        // the swap so the deck is not mutated on the rollback path.
        if (search_match_restore_pending()) return;

        size_t md_idx = out_action_to_md_idx[static_cast<size_t>(out_choice)];
        std::string card_out = deck.main_deck[md_idx].second;

        // perform the swap (one copy at a time)
        if (deck.sideboard[sb_idx].first > 1) {
            deck.sideboard[sb_idx].first--;
        } else {
            deck.sideboard.erase(deck.sideboard.begin() + static_cast<long>(sb_idx));
        }
        // add the removed main deck card to sideboard
        bool found_in_sb = false;
        for (auto &entry : deck.sideboard) {
            if (entry.second == card_out) {
                entry.first++;
                found_in_sb = true;
                break;
            }
        }
        if (!found_in_sb) {
            deck.sideboard.push_back({1, card_out});
        }
        // add the sideboard card to main deck
        bool found_in_md = false;
        for (auto &entry : deck.main_deck) {
            if (entry.second == card_in) {
                entry.first++;
                found_in_md = true;
                break;
            }
        }
        if (!found_in_md) {
            deck.main_deck.push_back({1, card_in});
        }
        // remove one copy from main deck
        if (deck.main_deck[md_idx].first > 1) {
            deck.main_deck[md_idx].first--;
        } else {
            deck.main_deck.erase(deck.main_deck.begin() + static_cast<long>(md_idx));
        }

        sided_out_names.insert(card_out);
        sided_in_names.insert(card_in);

        game_log("Swapped out %s for %s\n", card_out.c_str(), card_in.c_str());
        st.sb_swaps++;
        // OUT swap completed: the phase is no longer resumable mid-swap.
        st.pending_in_sb_idx = -1;
    }

    // Phase ended (any break path): clear the mid-swap resumption marker.
    st.pending_in_sb_idx = -1;
    sideboard_phase = false;
    sideboard_phase_player = Zone::UNKNOWN;
}

int play_bo3_match(Deck deck_a, Deck deck_b, unsigned int seed,
                   const std::function<void(int, bool)> &before_game,
                   const std::function<void(int, int)> &after_game) {
    // Initialize the match context from the params. All between-game state lives
    // here so the loop is a resumable dispatcher over ctx.stage. The legacy
    // globals (match_wins_a/b, match_game_number, sideboard_phase*) are kept in
    // sync as we go — they remain the values machine_io/input_logger read.
    MatchContext &ctx = g_match_ctx;
    ctx = MatchContext();
    ctx.active = true;
    ctx.deck_a = deck_a;
    ctx.deck_b = deck_b;
    ctx.base_seed = seed;
    ctx.a_goes_first = true;
    ctx.game_num = 0;
    ctx.wins_a = 0;
    ctx.wins_b = 0;
    ctx.stage = MatchContext::PLAY_GAME;

    match_wins_a = 0;
    match_wins_b = 0;
    match_reset_revealed();  // clear revealed-cards accumulator for the whole match

    bool match_done = false;
    while (!match_done && ctx.game_num < 3) {
        // A MATCH-scoped restore latched by a sideboard-rooted simulation (its
        // unwind bailed out of the inner game/sideboard loop) is applied here,
        // rolling the whole match — cur_game, ECS, ctx (stage/decks/sb), deck_state
        // and the match globals — back to the sideboard root before the switch
        // re-enters the restored stage.
        if (search_match_restore_pending()) search_apply_pending_match_restore();

        switch (ctx.stage) {
        case MatchContext::PLAY_GAME: {
            match_game_number = ctx.game_num;
            // A sideboard-rooted sim rolls forward into this game only to search;
            // its real-match side effects (the actor's begin_game/backfill hooks,
            // the banner) must not fire. Suppress them while a match snapshot lives.
            bool simulating = snapshot_any_match_scope_live();
            if (!simulating) {
                game_log("\n----- MATCH GAME %d of 3 -----\n", ctx.game_num + 1);
                if (before_game) before_game(ctx.game_num, ctx.a_goes_first);
            }

            EcsSystems sys = init_ecs();
            int winner = play_single_game(sys, ctx.deck_a, ctx.deck_b, ctx.a_goes_first,
                                          match_game_seed(ctx));

            // A sim's unwind exited play_single_game with a MATCH restore still
            // latched: take none of the real-game bookkeeping below (winner
            // validation, tally, GAME_RESULT, after_game, stage advance) — loop
            // back so the dispatcher top applies the restore.
            if (search_match_restore_pending()) continue;

            // Every end-of-game path must have set a winner; the else-branch below
            // would otherwise silently credit a winnerless game to B.
            if (winner != Zone::PLAYER_A && winner != Zone::PLAYER_B)
                fatal_error("bo3 game " + std::to_string(ctx.game_num + 1) +
                            " ended with no winner (Game::winner unset)");

            if (winner == Zone::PLAYER_A) {
                ctx.wins_a++;
                match_wins_a = ctx.wins_a;
                std::printf("GAME_RESULT: %d Player A wins\n", ctx.game_num + 1);
            } else {
                ctx.wins_b++;
                match_wins_b = ctx.wins_b;
                std::printf("GAME_RESULT: %d Player B wins\n", ctx.game_num + 1);
            }
            std::fflush(stdout);

            if (after_game) after_game(ctx.game_num, winner);

            if (ctx.wins_a == 2) {
                std::printf("MATCH_RESULT: Player A wins %d-%d\n", ctx.wins_a, ctx.wins_b);
                std::fflush(stdout);
                match_done = true;
                break;
            }
            if (ctx.wins_b == 2) {
                std::printf("MATCH_RESULT: Player B wins %d-%d\n", ctx.wins_a, ctx.wins_b);
                std::fflush(stdout);
                match_done = true;
                break;
            }

            // loser goes first next game
            ctx.a_goes_first = (winner != Zone::PLAYER_A);

            // sideboarding phase - ECS from the just-ended game is still valid
            // (player entities exist for populate_gamestate, card_db works for load_card)
            ctx.sb = SideboardPhaseState();
            ctx.sb.player = Zone::PLAYER_A;
            ctx.stage = MatchContext::SIDEBOARD_A;
            break;
        }
        case MatchContext::SIDEBOARD_A: {
            run_sideboard_phase(ctx.deck_a, ctx.sb);
            // A sim's unwind (or a restore latched during this phase) exited
            // run_sideboard_phase with a MATCH restore latched: don't advance the
            // stage — loop back and apply the restore at the dispatcher top.
            if (search_match_restore_pending()) continue;
            ctx.sb = SideboardPhaseState();
            ctx.sb.player = Zone::PLAYER_B;
            ctx.stage = MatchContext::SIDEBOARD_B;
            break;
        }
        case MatchContext::SIDEBOARD_B: {
            run_sideboard_phase(ctx.deck_b, ctx.sb);
            if (search_match_restore_pending()) continue;
            ctx.game_num++;
            ctx.stage = MatchContext::PLAY_GAME;
            break;
        }
        }
    }

    match_game_number = -1;
    ctx.active = false;
    int result = ctx.wins_a > ctx.wins_b ? Zone::PLAYER_A : Zone::PLAYER_B;
    return result;
}

static unsigned int match_game_seed(const MatchContext &ctx) {
    // Real line (salt 0): exactly the historical seed, ctx.base_seed + game_num.
    // A searched world (salt != 0, set by a sideboard-root DETERMINIZE) mixes the
    // salt through a golden-ratio multiply so distinct salts give distinct deals.
    return ctx.base_seed + static_cast<unsigned int>(ctx.game_num) +
           (g_sim_seed_salt ? 0x9e3779b9u * g_sim_seed_salt : 0u);
}
