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
// authoritative-in-sync (machine_io/input_logger read them). Kept internal for
// Stage 1 (a later stage exposes it for snapshot/restore across init_ecs()).
static MatchContext g_match_ctx;

EcsSystems init_ecs() {
    card_db.clear();
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
    // into this game's fresh ECS.
    snapshot_release_all();
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
    while (!cur_game.ended || search_intercept_game_end()) {
        // A RESTORE that arrived mid-decision unwound to here; apply it before
        // anything reads game state, so this iteration re-derives (and re-emits)
        // the restored decision.
        search_apply_pending_restore();
        if ((!InputLogger::instance().is_machine_mode() || narrative_mode) && cur_game.turn != prev_turn) {
            cli_print_turn_header(cur_game.turn, cur_game.player_a_turn);
            prev_turn = cur_game.turn;
        }
        Zone::Ownership viewer = (has_human_player)
            ? (human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B)
            : Zone::UNKNOWN;

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
        if (cur_game.advance_step(sys.stack_manager, sys.orderer)) {
            continue;
        } else {
            sys.state_manager->state_based_effects(cur_game, sys.orderer);
            if (cur_game.ended) {
                if (!search_intercept_game_end()) break;
                continue;
            }
        }

        auto legal_actions = sys.state_manager->determine_legal_actions(cur_game, sys.orderer, sys.stack_manager);
        if (legal_actions.size() == 1) {
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
            int choice = InputLogger::instance().get_input(actions);

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
        int out_choice = InputLogger::instance().get_input(out_actions);

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
        switch (ctx.stage) {
        case MatchContext::PLAY_GAME: {
            match_game_number = ctx.game_num;
            game_log("\n----- MATCH GAME %d of 3 -----\n", ctx.game_num + 1);

            if (before_game) before_game(ctx.game_num, ctx.a_goes_first);

            EcsSystems sys = init_ecs();
            int winner = play_single_game(sys, ctx.deck_a, ctx.deck_b, ctx.a_goes_first,
                                          ctx.base_seed + static_cast<unsigned int>(ctx.game_num));

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
            ctx.sb = SideboardPhaseState();
            ctx.sb.player = Zone::PLAYER_B;
            ctx.stage = MatchContext::SIDEBOARD_B;
            break;
        }
        case MatchContext::SIDEBOARD_B: {
            run_sideboard_phase(ctx.deck_b, ctx.sb);
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
