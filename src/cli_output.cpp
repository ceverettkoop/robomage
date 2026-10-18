#include "cli_output.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "input_logger.h"

// ── Utility ───────────────────────────────────────────────────────────────────

const char* step_to_string(Step in_step) {
    switch (in_step) {
        case UNTAP:             return "Untap";
        case UPKEEP:            return "Upkeep";
        case DRAW:              return "Draw";
        case FIRST_MAIN:        return "First Main";
        case BEGIN_COMBAT:      return "Begin Combat";
        case DECLARE_ATTACKERS: return "Declare Attackers";
        case DECLARE_BLOCKERS:  return "Declare Blockers";
        case FIRST_STRIKE_DAMAGE: return "First Strike Damage";
        case COMBAT_DAMAGE:     return "Combat Damage";
        case END_OF_COMBAT:     return "End of Combat";
        case SECOND_MAIN:       return "Second Main";
        case END_STEP:          return "End Step";
        case CLEANUP:           return "Cleanup";
        default:                return "ERROR UNREACHABLE";
    }
}

std::string player_name(Zone::Ownership owner) {
    return (owner == Zone::PLAYER_A) ? "Player A" : "Player B";
}

std::string owner_possessive(Zone::Ownership actor, Zone::Ownership owner) {
    return actor == owner ? std::string("their") : player_name(owner) + "'s";
}

// ── game_log ──────────────────────────────────────────────────────────────────

static void game_log_va(const char* fmt, va_list ap) {
    vprintf(fmt, ap);
}

extern bool narrative_mode;

void game_log(const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// Resolve the perspective that one-sided narrative output should be limited to.
// Returns true and sets *owner when output is restricted: a human is at the
// keyboard (--player) or a spectator view is pinned (--log-viewer, used by the
// TUI which drives both seats over the machine protocol but still wants
// one-sided narrative). Returns false for full perfect-information narrative.
bool resolve_narrative_viewer(Zone::Ownership* owner) {
    extern bool has_human_player;
    extern bool human_player_is_a;
    extern bool log_viewer_set;
    extern Zone::Ownership log_viewer_owner;
    if (has_human_player) {
        *owner = human_player_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;
        return true;
    }
    if (log_viewer_set) {
        *owner = log_viewer_owner;
        return true;
    }
    return false;
}

void game_log_private(Zone::Ownership private_to, const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    Zone::Ownership viewer;
    if (resolve_narrative_viewer(&viewer) && private_to != viewer) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// Complement of game_log_private: emits only to a pinned viewer who is NOT
// `owner` — the redacted/generic line an opponent sees in place of hidden info
// (e.g. "draws a card" instead of "draws Ponder"). Silent when narrative is
// unrestricted or the viewer IS the owner, so pairing it with game_log_private
// on the same event yields exactly one line in every perspective.
void game_log_redacted(Zone::Ownership owner, const char* fmt, ...) {
    if (InputLogger::instance().is_machine_mode() && !narrative_mode) return;
    Zone::Ownership viewer;
    if (!resolve_narrative_viewer(&viewer) || viewer == owner) return;
    va_list ap;
    va_start(ap, fmt);
    game_log_va(fmt, ap);
    va_end(ap);
}

// ── Startup / meta ────────────────────────────────────────────────────────────

void cli_print_version(const char* version) {
    game_log("robomage %s\n", version);
}

void cli_print_help(const char* program, const char* version) {
    printf("robomage %s\n", version);
    printf("Usage: %s [options]\n", program);
    printf("\nDecks and match:\n");
    printf("  --deck <name>       Load named deck for both players (default: delver)\n");
    printf("  --deck-a <name>     Load named deck for player A (default: delver)\n");
    printf("  --deck-b <name>     Load named deck for player B (default: delver)\n");
    printf("  --bo3               Best-of-three match with sideboarding (default: one game)\n");
    printf("  --seed <n>          RNG seed (default: current time)\n");
    printf("  --game-seed <n>     The game the Python drivers and scenario files call seed\n");
    printf("                      <n>: the engine seed they derive from it, and the in-process\n");
    printf("                      agents' generator seeded with <n> (replaces --seed)\n");
    printf("  --max-decisions <n> Stop the run at decision <n>+1, before it is decided:\n");
    printf("                      MAX_DECISIONS: line, exit status 0, no result for that game.\n");
    printf("                      Recorded in the decision log, so its replay stops there too\n");
    printf("  --no-shuffle        Skip the initial library shuffle; cards are drawn in\n");
    printf("                      deck-file order\n");
    printf("  --offer-cancel      Offer \"Cancel\" at every cast and activation prompt; it\n");
    printf("                      reverses the spell or ability being proposed (CR 733.1).\n");
    printf("                      A testing aid (default: off)\n");
    printf("  --no-auto-mana      Turn auto-mana off for the match: every mana ability is\n");
    printf("                      offered at priority and costs are paid by choosing mana\n");
    printf("                      sources, for every seat and agent. By default ordinary\n");
    printf("                      mana abilities are hidden and costs are paid automatically\n");
    printf("                      (instant-speed sources and sources targeted by removal\n");
    printf("                      on the stack stay offered). Recorded in the decision log\n");
    printf("  --closed-decklist   Each seat's state-format frames carry only its own\n");
    printf("                      registered decklist (default: both lists are open);\n");
    printf("                      name-a-card then offers every vocab card\n");
    printf("\nInput and output:\n");
    printf("  (no option)         Hotseat: every seat without an in-process agent is a human\n");
    printf("                      seat at the terminal (the C++ CLI), each decision shown from\n");
    printf("                      the deciding seat's RMSTATE frame; type help at its prompt\n");
    printf("                      for the commands. A human game always writes its decision\n");
    printf("                      log and prints no narrative unless --narrative is given\n");
    printf("  --player <A|B>      Make that seat a human seat (any value other than A/a means\n");
    printf("                      B), e.g. --player A --agent-b explore: a human against an\n");
    printf("                      in-process agent (with --play: the human takes over once the\n");
    printf("                      script runs out). With --machine every other seat needs an\n");
    printf("                      agent or a script (the human seat reads stdin). Under\n");
    printf("                      --replay there is no human seat: the narrative's one-sided\n");
    printf("                      lines are limited to that seat's view, as with --log-viewer\n");
    printf("  --machine           Machine mode (speak RMSTATE; the decisions are the same\n");
    printf("                      for every consumer). Outside\n");
    printf("                      --replay, each seat that has no in-process agent and is\n");
    printf("                      not a human seat decides over the RMSTATE stream:\n");
    printf("                      its frames (\"RMSTATE <nbytes>\" + payload) on stdout, one\n");
    printf("                      integer reply per decision on stdin, a terminal frame per\n");
    printf("                      seat at match end; all other output then goes to stderr.\n");
    printf("                      Narrative is silenced unless --narrative is given\n");
    printf("  --narrative         In machine mode (and with a human seat), also print the\n");
    printf("                      engine's game narrative (perfect information, but for the\n");
    printf("                      lines limited to a --player / --log-viewer seat's view)\n");
    printf("  --log-viewer <A|B>  Restrict one-sided narrative to that seat's view\n");
    printf("                      (any value other than A/a means B)\n");
    printf("  --trace-visibility  Narrate each reveal/look window as it opens and closes\n");
    printf("                      (VISIBILITY: lines in the game log; a debugging aid)\n");
    printf("  --trace-handles     Print each viewer's handle table at every decision and\n");
    printf("                      narrate handles as they are issued, linked and retired\n");
    printf("                      (HANDLES: / HANDLE: lines in the game log; a debugging aid)\n");
    printf("  --trace-events      Narrate each observation event as it is queued, per\n");
    printf("                      viewer (EVENT: lines in the game log; a debugging aid)\n");
    printf("  --rmstate-dump <file>  Write every viewer's (A, B, omniscient) RMSTATE frames\n");
    printf("                      to <file> at each decision, and a terminal frame each at\n");
    printf("                      match end (%%p in <file> = the process id); the normal\n");
    printf("                      protocol is unchanged\n");
    printf("  --state-dump json   Also render every --rmstate-dump frame, decoded, as one\n");
    printf("                      JSON line into <file>.jsonl (with --rmstate-read: to stdout)\n");
    printf("  --machine-format rmstate\n");
    printf("                      The same as --machine (RMSTATE is the only machine\n");
    printf("                      protocol)\n");
    printf("  --rmstate-card-scripts  Every CARDS frame carries each card's raw script text\n");
    printf("                      (its front-face row's script column; a token's in its row)\n");
    printf("  --rmstate-read <file>  Play no game: decode every record of an RMSTATE stream\n");
    printf("                      (- = stdin), re-encode it and compare the bytes; ERROR:\n");
    printf("                      lines and a summary on stderr, exit status 1 on any\n");
    printf("                      mismatch. With --state-dump json, the JSON lines on stdout\n");
    printf("  --knowledge-a <off|expert|expert+belief>  --knowledge-b <...>\n");
    printf("                      That seat's knowledge sidecar in its RMSTATE frames (the\n");
    printf("                      k_* tables: recall and sound deduction from the seat's\n");
    printf("                      own frames; off = the k_meta row alone; expert+belief\n");
    printf("                      adds the b_* tables: a posterior over the reference\n");
    printf("                      lists). Default expert; play is unchanged\n");
    printf("  --selfcheck-state   Check the RMSTATE frames at every decision: decode and\n");
    printf("                      re-encode byte-identical, no hidden identity read (frames\n");
    printf("                      and events rebuilt with hidden cards substituted), complete\n");
    printf("                      tables; sampled: hidden zones permuted, snapshot restore.\n");
    printf("                      Failures print ERROR: lines; play is unchanged\n");
    printf("  --selfcheck-state-sample <n>\n");
    printf("                      Run the sampled self-checks every <n>th decision (default 8)\n");
    printf("  --selfcheck-knowledge\n");
    printf("                      Implies --selfcheck-state. Check every k_* fact of both\n");
    printf("                      seats' frames against the true world at every decision,\n");
    printf("                      and the knowledge state across the snapshot check's restore.\n");
    printf("                      Failures print ERROR: lines; play is unchanged\n");
    printf("  --trace-decisions   Print every decision's board, pending decision and menu, as\n");
    printf("                      the CLI renders the deciding seat's RMSTATE frame, and the\n");
    printf("                      choice made (lines starting \"| \" in the game log)\n");
    printf("  --check-decks       Play no game: check that every card of the top-level and\n");
    printf("                      league/ decks loads and is in the card vocabulary; one line\n");
    printf("                      per missing card, exit status 1 when any is missing\n");
    printf("  --check-decks-dir <dir>  With --check-decks: check this directory (relative to\n");
    printf("                      resources/decks/) instead; repeatable\n");
    printf("  --log-decisions     Write the decision log in machine mode (off by default\n");
    printf("                      there; CLI/interactive games always log) to\n");
    printf("                      resources/logs/game_<seed>.log\n");
    printf("  --decision-log <file>  Write the decision log to <file> (implies --log-decisions;\n");
    printf("                      no \"Logging inputs to\" line is printed)\n");
    printf("  --replay <logfile>  Replay a previously logged game. The log's RMLOG v2\n");
    printf("                      header alone determines the setup (decks, seed, --bo3,\n");
    printf("                      --no-shuffle, test setup); those flags given alongside\n");
    printf("                      --replay are ignored with a warning. It prints the\n");
    printf("                      narrative (under --machine, only with --narrative).\n");
    printf("                      Output flags (--machine, --narrative, --player,\n");
    printf("                      --log-viewer, --trace-visibility, --trace-handles,\n");
    printf("                      --trace-events, --rmstate-dump, --state-dump,\n");
    printf("                      --knowledge-a/-b, --selfcheck-state, --selfcheck-knowledge)\n");
    printf("                      still apply; --log-decisions has no effect\n");
    printf("\nIn-process agents (each option implies --machine; incompatible with --replay\n");
    printf("and an agent on a --player seat):\n");
    printf("  --agent-a <spec>    --agent-b <spec>\n");
    printf("                      That seat's decisions go to an in-process agent reading the\n");
    printf("                      seat's RMSTATE frames: first (row 0), random, explore,\n");
    printf("                      explore:patient, actions:<i,i,...> (then row 0) or\n");
    printf("                      play:<specs> (then row 0). A seat without one keeps its\n");
    printf("                      input path (the RMSTATE stream, or a --player human seat)\n");
    printf("  --play <specs>      One --play script for both seats: comma-separated semantic\n");
    printf("                      specs (cast:<card>, target:<card>@opp, A:attack:<card>, ...),\n");
    printf("                      each resolved against its decision's menu; once it runs out\n");
    printf("                      each seat's --agent-a/-b decides (default first). Prints\n");
    printf("                      \"resolved --actions: ...\" at the end, and \"SCRIPT\n");
    printf("                      INCOMPLETE: ...\" when specs were left unapplied\n");
    printf("  --actions <i,i,...> One list of replies for both seats, then their agents\n");
    printf("\nTest setup (card lists are comma-separated; applied at the start of each game):\n");
    printf("  --hand-a <cards>    --hand-b <cards>\n");
    printf("                      A stacked deck: these cards first (the opening hand), then\n");
    printf("                      --library-a/-b, padded to 15 cards with its last card (else a\n");
    printf("                      basic land of the hand, else Mountain); replaces --deck-a/-b\n");
    printf("                      and implies --no-shuffle for both seats\n");
    printf("  --library-a <cards> --library-b <cards>\n");
    printf("                      The stacked deck's library after the hand (needs --hand-a/-b)\n");
    printf("  --battlefield-a <cards>  --battlefield-b <cards>\n");
    printf("                      Start with these permanents on the battlefield\n");
    printf("  --graveyard-a <cards>    --graveyard-b <cards>\n");
    printf("                      Start with these cards in the graveyard\n");
    printf("  --exile-a <cards>        --exile-b <cards>\n");
    printf("                      Start with these cards in exile\n");
    printf("  --sideboard-a <cards>    --sideboard-b <cards>\n");
    printf("                      Add these cards to the sideboard\n");
    printf("  --life-a <n>        --life-b <n>\n");
    printf("                      Starting life total (default: 20)\n");
    printf("\n  --help, -h          Show this help message\n");
    printf("\nDeck names are filenames without .dk, relative to resources/decks/ under the\n");
    printf("working directory (run from bin/), e.g. league/ur_delver.\n");
    printf("An unrecognized option, or a value-taking option given last with no value, is\n");
    printf("skipped with a warning on stderr.\n");
}

void cli_print_seed(unsigned int seed) {
    game_log("Using seed: %u\n", seed);
}

// Header for the pre-game decisions (mulligans, CR 103.6b opening-hand actions) so they
// don't read as part of a numbered turn.
void cli_print_pregame_header() {
    game_log("\n-------- PREGAME --------\n");
}

void cli_print_turn_header(size_t turn, bool player_a_turn) {
    const char* name = player_a_turn ? "Player A" : "Player B";
    // Game::turn_state.turn counts from 0; display 1-based so the first header reads "TURN 1"
    // (never "TURN 0"). Display-only — the 0-based counter itself is untouched.
    game_log("\n-------- TURN %zu (%s) --------\n", turn + 1, name);
}

// ── Errors ────────────────────────────────────────────────────────────────────

void cli_warning(const std::string& msg) {
    fprintf(stderr, "WARNING: %s\n", msg.c_str());
}

void cli_error(const std::string& msg) {
    fprintf(stderr, "ERROR: %s\n", msg.c_str());
}

[[noreturn]] void cli_fatal_error(const std::string& msg) {
    fprintf(stderr, "FATAL: %s\n", msg.c_str());
    exit(1);
}
