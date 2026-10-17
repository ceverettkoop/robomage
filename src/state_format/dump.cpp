#include "dump.h"

#include <unistd.h>

#include <cstdio>
#include <string>

#include "../classes/game.h"
#include "../error.h"
#include "../game_driver.h"
#include "../queries/players.h"
#include "encode.h"
#include "schema.h"

static FILE *dump_file();
static std::string dump_path();
static void write_record(FILE *f, const std::string &payload);
static Zone::Ownership deciding_seat();

// --rmstate-dump's FILE with each "%p" replaced by the engine's process id, so engines started in
// parallel with one flag (a test pool) write separate files.
static std::string dump_path() {
    std::string path = rmstate_dump_path;
    for (size_t at = path.find("%p"); at != std::string::npos; at = path.find("%p", at))
        path.replace(at, 2, std::to_string(static_cast<long>(getpid())));
    return path;
}

// The dump file, opened at the first decision (the dump is process output, like stdout).
static FILE *dump_file() {
    static FILE *f = nullptr;
    if (!f) {
        const std::string path = dump_path();
        f = std::fopen(path.c_str(), "wb");
        if (!f) fatal_error("--rmstate-dump: cannot open " + path);
    }
    return f;
}

static void write_record(FILE *f, const std::string &payload) {
    const std::string record = rmstate::frame_record(payload);
    std::fwrite(record.data(), 1, record.size(), f);
}

// The seat the decision belongs to: the sideboarding player between games, else the priority
// holder (every mid-flow prompt seats priority at its chooser).
static Zone::Ownership deciding_seat() {
    if (sideboard_phase && sideboard_phase_player != Zone::UNKNOWN) return sideboard_phase_player;
    return priority_seat();
}

void rmstate_dump_decision(const std::vector<LegalAction> &actions) {
    if (rmstate_dump_path.empty()) return;
    static bool match_frames_written = false;
    FILE *f = dump_file();
    ++cur_game.state_decisions;
    const Zone::Ownership decider = deciding_seat();
    for (Zone::Ownership viewer : {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN}) {
        if (!match_frames_written) {
            write_record(f, encode_schema_frame(viewer));
            write_record(f, encode_cards_frame(viewer));
            write_record(f, encode_decklists_frame(viewer));
        }
        write_record(f, encode_state_frame(viewer, actions, decider));
    }
    match_frames_written = true;
    std::fflush(f);
}
