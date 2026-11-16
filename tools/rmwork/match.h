#ifndef RMWORK_MATCH_H
#define RMWORK_MATCH_H

#include <array>
#include <cstdint>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../../src/agents/seat_frames.h"
#include "../../src/featurize/episode.h"
#include "rmlog.h"

// ── A recorded match: its decisions, every viewer's frames, the logs ────────────────────────────
//
// Built from a replay's --rmstate-dump (every viewer's frames: per viewer its SCHEMA, CARDS and
// DECKLISTS frames once, then at each decision a STATE frame for A, B and the omniscient viewer in
// that order, then the three terminal frames) and the engine's narrative. The dump is indexed once
// (index_match): each frame's offset and size, and per decision what the status bar and the tools'
// headers show, read from its frames; frames are decoded again on demand (Match::frame) through a
// small LRU cache, so a whole match is never held decoded. The decoded frames' tables point into
// the Match's schemas: a frame must not outlive its Match.
//
// The logs, per viewer:
//   - seat A / seat B: that seat's own log, built only from its own STATE frames' event tables
//     (80–82: what the seat saw happen since its previous frame), rendered by the CLI's renderer
//     (cli::event_lines), turn and game headers between them. Decision i's lines are those of the
//     seat's frame at decision i; the terminal frame's events join the last decision.
//   - omniscient: the engine's --narrative (stdout of the replay), cut at the replay's per-decision
//     lines ("(REPLAY) [T… | step | player] Input: n", one per decision read from the log, written
//     as the decision is taken): decision i's lines run from the line after decision i-1's marker
//     to its own marker, which is rewritten as "» Player X: <the chosen menu row>". Lines after the
//     last marker join the last decision. The banner lines before the first event are dropped.

namespace rmwork {

inline constexpr int VIEW_A = 0;
inline constexpr int VIEW_B = 1;
inline constexpr int VIEW_OMNI = 2;
inline constexpr int VIEW_COUNT = 3;

// "seat A", "seat B", "omniscient".
const char *view_name(int view);

// A frame's payload in the dump.
struct FrameRef {
        uint64_t offset = 0;
        uint32_t size = 0;
        bool valid() const { return size != 0; }
};

// One decision of a trajectory episode, joined to the match's decision of the same key.
struct EpisodeStepInfo {
        std::string file;  // the episode it came from
        int seat = 0;      // 0 A, 1 B
        int64_t chosen = 0;
        bool has_mu = false;
        std::vector<float> mu;
        int64_t policy_version = -1;
        int64_t teacher = -1;
        std::vector<featurize::PolicyEvalRecord> policies;
};

struct DecisionInfo {
        // The state format's decision key (header.game 0-based, header.sideboarding, header.decision).
        int game = 0;
        bool sideboarding = false;
        int64_t decision = 0;
        std::array<FrameRef, VIEW_COUNT> frames;
        int decider = -1;    // 0 A, 1 B: the seat whose frame carries the menu
        int turn = 0;        // the frame's turn counter (0-based)
        std::string phase;   // "First main", "Opening hand", "Sideboarding"
        std::string header;  // cli::board_header of the omniscient frame
        int menu_rows = 0;
        int reply = 0;           // the decision log's reply
        std::string reply_text;  // the chosen menu row's description (a concession's wording)
        std::array<int64_t, 2> life{{0, 0}};
        int episode_step = -1;  // index into Match::episode_steps, -1 none
};

// A game or a sideboarding phase of the match: its decisions are [first, first + count).
struct GameSpan {
        int game = 0;
        bool sideboarding = false;
        size_t first = 0, count = 0;
        std::string label;  // "Game 1", "Sideboarding 2"
};

struct LogLine {
        std::string text;
        int decision = 0;  // the decision whose lines it is among
        enum class Kind {
            EVENT,
            HEADER,  // a turn or game header
            CHOICE   // the omniscient log's chosen-row line
        } kind = Kind::EVENT;
        int seat = -1;  // a CHOICE line's deciding seat (0 A, 1 B)
};

struct MatchResult {
        bool known = false;
        int winner = -1;  // 0 A, 1 B, -1 a draw or unknown
        std::array<int, 2> wins{{0, 0}};
        std::vector<int> game_winners;  // per game: 0 A, 1 B, -1 a draw
        std::string text;               // "Player B wins 0-2"
};

class Match {
    public:
        Match() = default;
        Match(const Match &) = delete;
        Match &operator=(const Match &) = delete;
        ~Match();

        std::string log_path;  // the decision log, absolute
        RmlogInfo log;
        std::string dump_path, narrative_path, stderr_path;
        std::string engine;       // the engine binary that replayed it
        bool from_cache = false;  // the replay's files came from the cache
        std::vector<DecisionInfo> decisions;
        std::vector<GameSpan> games;
        std::array<FrameRef, VIEW_COUNT> terminal;
        std::array<std::vector<LogLine>, VIEW_COUNT> logs;
        bool narrative_mapped = false;  // the narrative's decision markers matched the decisions
        MatchResult result;
        std::vector<EpisodeStepInfo> episode_steps;
        std::vector<std::string> episode_files;

        bool bo3() const { return log.has_flag("bo3"); }
        const agents::SeatStream &stream(int view) const { return *streams_[static_cast<size_t>(view)]; }
        // The decoded STATE frame of `view` at decision `index` (index == decisions.size(): the
        // terminal frame); nullptr with `err` when it cannot be read or decoded.
        std::shared_ptr<const rmstate::DecodedFrame> frame(size_t index, int view, std::string *err) const;
        // A frame's payload bytes from the dump.
        bool read_payload(const FrameRef &ref, std::string *out, std::string *err) const;
        // The games entry holding decision `index` (-1 when none).
        int span_of(size_t index) const;
        // The decision of key (game, sideboarding, decision), or -1.
        int find_decision(int game, bool sideboarding, int64_t decision) const;
        // "deck_a vs deck_b", the decks' stems.
        std::string title() const;

    private:
        friend bool index_match(Match *m, const std::function<bool(double)> &progress, std::string *err);
        std::array<std::unique_ptr<agents::SeatStream>, VIEW_COUNT> streams_;
        std::map<std::tuple<int, bool, int64_t>, int> by_key_;
        int fd_ = -1;
        static constexpr size_t LRU_CAPACITY = 48;
        mutable std::list<std::pair<uint64_t, std::shared_ptr<const rmstate::DecodedFrame>>> lru_;
};

// Splits a seat log's event line "T<n>: <text>" (cli::event_lines' turn prefix); false for a line
// without one (a game event).
bool split_turn_prefix(const std::string &line, int *turn, std::string *text);

// Indexes `m->dump_path` (the other fields set by the caller: log_path, log, narrative_path) and
// builds the logs: every frame decoded once with its viewer's schema. `progress` gets the fraction
// done and returns false to cancel. False with `err` on an unreadable dump, a frame that does not
// decode, STATE frames out of the A, B, omniscient order, a decision whose three frames disagree
// on its key, or a decision count other than the log's.
bool index_match(Match *m, const std::function<bool(double)> &progress, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_MATCH_H */
