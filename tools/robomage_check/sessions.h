#ifndef ROBOMAGE_CHECK_SESSIONS_H
#define ROBOMAGE_CHECK_SESSIONS_H

#include <string>
#include <vector>

#include "frames.h"
#include "gate.h"
#include "proc.h"

// ── robomage_check: the engine runs the state and knowledge tiers share ───────────────────────

namespace rmcheck {

// An explore-vs-explore game under the self-checks (train/test_selfcheck_state.py's and
// train/test_knowledge.py's FUZZ_GAMES): the C++ explore tier in-process, --game-seed `seed` (the
// Python tier's runner.run_match game for that seed), at most 2500 decisions.
struct FuzzGame {
        std::string deck_a, deck_b;
        bool bo3 = false;
        long seed = 0;
        std::vector<std::string> flags;
        bool dump = false;  // also write an --rmstate-dump
};

struct FuzzOutcome {
        std::string label, err_path, dump;  // dump: "" when none was asked for
        ProcResult result;
        bool finished = false;  // a result reached (not stopped at the decision cap)
};

std::string fuzz_label(const FuzzGame &g);
// Plays `g` (its stderr kept as <dir>/fuzz_<seed>.err, its dump as <dir>/fuzz_<seed>.rms).
FuzzOutcome play_fuzz_game(const Options &opt, const FuzzGame &g, const std::string &dir);

// A --machine-format rmstate session (train/test_rmstate.py's run_stream): --seed `seed` (the
// engine seed itself), random replies from Python's random.Random(seed), a game conceded past
// decision `cap`.
struct StreamGame {
        std::string deck_a, deck_b;
        bool bo3 = false;
        long seed = 0;
        std::vector<std::string> flags;
};

std::string stream_label(const char *what, const StreamGame &g);
// Plays `g`, with an --rmstate-dump to `dump` unless it is "". run.error also names an engine
// exit other than 0 and a terminal frame count other than 2.
StreamRun play_stream_session(const Options &opt, const StreamGame &g, const std::string &dump, uint64_t cap = 250);

// rmknow --isolation for both seats at `level` on a dump or a saved stream; failure messages (and
// the summary lines in `summary`).
std::vector<std::string> rmknow_isolation(
    const Options &opt, const std::string &path, const char *level, std::string *summary);
// rmknow --truth for `seat` at `level` on a dump; failure messages (and the summary line).
std::vector<std::string> rmknow_truth(
    const Options &opt, const std::string &path, const char *seat, const char *level, std::string *summary);
// The engine's --rmstate-read on a dump (decode, re-encode byte identity): failure messages (and
// its summary line); a live --state-dump json file beside the dump (<dump>.jsonl) must equal the
// JSON it renders.
std::vector<std::string> recode_dump(const Options &opt, const std::string &path, std::string *summary);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_SESSIONS_H */
