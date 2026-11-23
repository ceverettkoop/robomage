#ifndef RMWORK_START_DATA_H
#define RMWORK_START_DATA_H

#include <cstdint>
#include <string>
#include <vector>

#include "card_images.h"

// ── The start page's data: recent games, replays and runs, and the status lines ─────────────────
//
// Kept cheap, so the page can list them at once and again every few seconds: a decision log's
// header and replies are read (logs are small), never replayed; a result comes from the result lines
// the engine already wrote: a Play game's stderr (<stem>.stderr beside its log: GAME_RESULT and
// MATCH_RESULT), else the replay cache's narrative of the log under the engine replays run with
// (replay_cache.h), else it is unknown ("…") until the log is opened. A Play game's opponent and
// seat come from the sidecar Play writes beside the log (<stem>.game.json), else the seat from the
// seat stream's name (<stem>.seat-a.rmstate). A run's phase, games and times come from its
// status.json (rmtrain's). scan_start does it all; the page runs it on a thread (start_page.h).
//
// The status blocks (engine, card images, inference server, files) look only at paths (stat), so
// they are built on the UI thread.

namespace rmwork {

// What a log's result lines say (GAME_RESULT: N Player A wins / N draw; MATCH_RESULT: Player B
// wins 1-2, A's wins first).
struct LogResult {
        bool match_line = false;  // a MATCH_RESULT line
        int match_winner = -1;    // 0 A, 1 B
        int wins[2] = {0, 0};     // the MATCH_RESULT's score
        std::vector<int> games;   // per GAME_RESULT line: 0 A, 1 B, -1 a draw
        // The match has a result: its MATCH_RESULT, or in best of one its game's.
        bool known(bool bo3) const { return match_line || (!bo3 && !games.empty()); }
};
LogResult parse_result_lines(const std::string &text);

enum class GameState {
    UNKNOWN,    // no result known (not opened since, or no stderr to read)
    PLAYING,    // Play's running game
    FINISHED,   // the result lines are there
    STOPPED,    // Play's game that ended without a result (quit, the engine killed)
    MALFORMED,  // the log does not read as RMLOG v3
};

struct RecentGame {
        std::string path;  // the decision log
        std::string when;  // "11-17 16:45" (the stem's timestamp, else the log's modification time)
        int64_t time = 0;  // the same as seconds since the epoch, for ordering
        GameState state = GameState::UNKNOWN;
        std::string error;  // MALFORMED: why
        std::string deck_a, deck_b;
        bool bo3 = false;
        size_t decisions = 0;
        int seat = -1;           // the human's seat in a Play game (0 A, 1 B), -1 unknown
        std::string opponent;    // the opponent's agent ("explore", "nn: final.ts.pt"), "" unknown
        uint64_t game_seed = 0;  // a Play game's --game-seed (the sidecar's), 0 unknown
        LogResult result;
        bool cached = false;  // the result came from the replay cache
};

// The result as the page shows it ("won 2-1", "lost", "A wins 2-0", "no result", "playing", "…"),
// with its tone: +1 good, -1 bad, 0 plain, 2 a warning.
std::string game_result_text(const RecentGame &g, int *tone);
// "Bo3"/"bo1".
const char *game_format_text(const RecentGame &g);
// The human's deck and the opponent's (seat unknown: A's and B's).
std::string game_own_deck(const RecentGame &g);
std::string game_opponent_deck(const RecentGame &g);

struct RecentRun {
        std::string path;
        std::string name;     // the directory's name
        bool recent = false;  // opened before (settings' recent), else found under the runs directory
        bool status = false;  // status.json read
        std::string phase;    // "done", "B3", "" unknown
        bool running = false;
        double updated = 0;  // status.json's updated_at, else the directory's modification time (s)
        double hours = -1;   // uptime, -1 unknown
        int64_t games = -1;  // totals.games, -1 unknown
        std::string error;   // status.json does not read
};

// "11-17 16:45" (local time).
std::string short_time(int64_t seconds);

struct StartScanRequest {
        std::string games_dir;            // Play's games (default_games_dir())
        std::vector<std::string> recent;  // settings' recent files and directories
        std::string run_dir;              // the runs directory (run_dir_path())
        std::string replay_engine;        // the engine a replay runs (its cached replays' results)
        std::string cache_root;           // replay_cache_root()
        std::string live_log;             // Play's running game's log, "" none
        size_t max_games = 40;
        size_t max_replays = 12;
        size_t max_runs = 30;
};

struct StartData {
        bool games_dir_exists = false;
        size_t games_total = 0;           // decision logs in the games directory
        std::vector<RecentGame> games;    // Play's games, newest first
        std::vector<RecentGame> replays;  // recent decision logs outside the games directory, latest first
        std::vector<RecentRun> runs;      // recent run directories, then the runs directory's, newest first
        uint64_t cache_bytes = 0;         // the replay cache's size
        size_t cache_entries = 0;
};

// Reads it all (what the page's thread does).
StartData scan_start(const StartScanRequest &r);
// One decision log: `play` a Play game (its stderr and sidecar beside it).
RecentGame read_recent_game(const std::string &log, bool play, const StartScanRequest &r);
// One run directory's line.
RecentRun read_recent_run(const std::string &dir, bool recent);
// A directory holds a run: status.json, or an rmlog/ directory.
bool is_run_directory(const std::string &dir);

// Play's sidecar for a game: what the log does not record (the human's seat, the opponent).
struct GameSidecar {
        int seat = 0;
        std::string opponent;  // the opponent kind's key ("explore", "model", …)
        std::string model;     // a model's export or a server's socket
        std::string deck, opponent_deck;
        bool bo3 = false;
        uint64_t seed = 0;
};
// "<stem>.game.json" for decision log "<stem>.rmlog".
std::string sidecar_path(const std::string &log);
std::string sidecar_json(const GameSidecar &s);
bool parse_sidecar(const std::string &text, GameSidecar *out);
// The opponent as the page names it: "explore", "nn: final.ts.pt", "nn:remote: s.sock", "scripted (hard)".
std::string sidecar_opponent_text(const GameSidecar &s);

// ── Status ──

enum class Health {
    NONE,  // nothing to say (dim)
    GOOD,
    WARN,
    BAD
};

struct StatusLine {
        std::string label, value;
        Health tone = Health::NONE;
};

struct StatusBlock {
        std::string title;
        Health health = Health::NONE;
        std::vector<StatusLine> lines;
        std::string fix;  // what fixes it ("make actor"), "" nothing
};

// The engine binary Settings → Engine names (or the --engine override), its working directory, the
// robomage_nn beside it (an nn: opponent's) and whether it is older than the plain engine.
StatusBlock engine_status(const std::string &binary, const std::string &workdir);
// The card-image cache.
StatusBlock image_status(const std::string &cache_dir, const ImageCacheStatus &s);
// The inference server: the socket configured and there, the policies listed, rmserve beside the
// engine, and Model's Read's last state ("" never asked; `read_failed` when it was refused or lost).
StatusBlock inference_status(const std::string &socket, size_t policies, const std::string &engine,
    const std::string &read_state, bool read_failed);
// Play's games directory and the replay cache.
StatusBlock files_status(const StartData &d, const std::string &games_dir, const std::string &cache_root);

// `path` as the page shows it: relative to the repository when under it, else "~/…" under $HOME.
std::string display_path(const std::string &path);
// "1.2 GB", "56 MB", "3 KB".
std::string byte_size_text(uint64_t bytes);

}  // namespace rmwork

#endif /* RMWORK_START_DATA_H */
