#include "stream.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../error.h"
#include "../game_driver.h"
#include "../queries/players.h"
#include "agent_feed.h"
#include "dump.h"
#include "encode.h"
#include "events.h"
#include "handles.h"
#include "knowledge_feed.h"
#include "schema.h"
#include "selfcheck.h"

static void stream_write(const std::string &payload);
static void stream_match_frames_once(Zone::Ownership seat);
static void stream_state_frame(
    Zone::Ownership seat, const std::vector<LegalAction> &actions, Zone::Ownership decider, bool terminal);
static int read_reply();
static bool seat_reads_stdin(int seat);

// The descriptor stdout had at startup, which carries the frames (stdout itself is pointed at
// stderr by rmstate_outputs_init).
static FILE *stream_out = nullptr;
// Per seat (indexed by Zone::Ownership): its SCHEMA, CARDS, DECKLISTS and METAGAME frames were sent.
static bool stream_match_frames_sent[3] = {false, false, false};
// The stream is on (rmstate_outputs_init).
static bool stream_on = false;

// Whether machine seat `seat` (0 = A, 1 = B) decides over the stream: it has no in-process agent
// or script (agents/driver.h) and is not the --player seat.
static bool seat_reads_stdin(int seat) {
    if (play_script_given || actions_script_given) return false;
    if (!(seat == 0 ? agent_a_spec : agent_b_spec).empty()) return false;
    return !(has_human_player && human_player_is_a == (seat == 0));
}

void rmstate_outputs_init() {
    if (state_dump_json && rmstate_dump_path.empty() && rmstate_read_path.empty())
        cli_warning("--state-dump json renders the --rmstate-dump or --rmstate-read stream; neither is given");
    stream_on = machine_mode && !replay_mode && (seat_reads_stdin(0) || seat_reads_stdin(1));
    if (!stream_on) return;
    std::fflush(stdout);
    const int frames_fd = dup(STDOUT_FILENO);
    if (frames_fd < 0) fatal_error("the RMSTATE stream: cannot duplicate stdout");
    stream_out = fdopen(frames_fd, "wb");
    if (!stream_out) fatal_error("the RMSTATE stream: cannot open the frame stream");
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) fatal_error("the RMSTATE stream: cannot redirect stdout");
}

bool rmstate_stream_enabled() {
    return stream_on;
}

Zone::Ownership rmstate_deciding_seat() {
    if (sideboard_phase && sideboard_phase_player != Zone::UNKNOWN) return sideboard_phase_player;
    return priority_seat();
}

void rmstate_begin_decision() {
    if (rmstate_dump_path.empty() && !stream_on && !selfcheck_state_enabled() && !agent_feed_enabled()) return;
    ++(sideboard_phase ? cur_game.state_sideboard_decisions : cur_game.state_decisions);
    for (Zone::Ownership viewer : {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN}) issue_handles_canonical(viewer);
}

uint32_t rmstate_decision_number() {
    return sideboard_phase ? cur_game.state_sideboard_decisions : cur_game.state_decisions;
}

static void stream_write(const std::string &payload) {
    const std::string record = rmstate::frame_record(payload);
    std::fwrite(record.data(), 1, record.size(), stream_out);
}

static void stream_match_frames_once(Zone::Ownership seat) {
    if (stream_match_frames_sent[seat]) return;
    for (const std::string &payload : encode_match_frames(seat)) {
        knowledge_feed_match_frame(ObsConsumer::STREAM, seat, payload);
        stream_write(payload);
    }
    stream_match_frames_sent[seat] = true;
}

// The seat's STATE frame from its stream queue, with the stream's knowledge tables.
static void stream_state_frame(
    Zone::Ownership seat, const std::vector<LegalAction> &actions, Zone::Ownership decider, bool terminal) {
    std::string payload =
        encode_state_frame(seat, actions, decider, take_obs_events(seat, ObsConsumer::STREAM), terminal);
    knowledge_append_tables(ObsConsumer::STREAM, seat, &payload);
    stream_write(payload);
}

// One reply line from stdin: the integer it starts with, else -1
// (end of input or an unparseable line).
static int read_reply() {
    char line[128];
    if (!std::fgets(line, sizeof(line), stdin)) return -1;
    if (!std::strchr(line, '\n')) {
        int c;
        while ((c = std::getchar()) != '\n' && c != EOF) {
        }
    }
    char *end = nullptr;
    const long parsed = std::strtol(line, &end, 10);
    return end != line ? static_cast<int>(parsed) : -1;
}

int rmstate_stream_exchange(const std::vector<LegalAction> &actions) {
    // Everything printed so far (on stdout, which points at stderr) reaches stderr before the
    // frame, so a reader of both pipes sees each decision's narrative ahead of its frame.
    std::fflush(stdout);
    const Zone::Ownership seat = rmstate_deciding_seat();
    stream_match_frames_once(seat);
    stream_state_frame(seat, actions, seat, false);
    std::fflush(stream_out);
    return read_reply();
}

void rmstate_match_over() {
    rmstate_dump_match_over();
    selfcheck_state_match_over();
    if (!stream_on) return;
    std::fflush(stdout);
    for (Zone::Ownership seat : {Zone::PLAYER_A, Zone::PLAYER_B}) {
        stream_match_frames_once(seat);
        stream_state_frame(seat, {}, Zone::UNKNOWN, true);
    }
    std::fflush(stream_out);
}
