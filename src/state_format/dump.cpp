#include "dump.h"

#include <unistd.h>

#include <cstdio>
#include <string>

#include "../classes/game.h"
#include "../error.h"
#include "../game_driver.h"
#include "decode.h"
#include "encode.h"
#include "events.h"
#include "json.h"
#include "knowledge_feed.h"
#include "schema.h"
#include "stream.h"

static FILE *dump_file();
static FILE *json_file();
static std::string dump_path();
static const rmstate::Schema &json_schema();
static void write_record(const std::string &payload);
static void write_match_frames(Zone::Ownership viewer);
static void write_state_frame(
    Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider, bool terminal);

static const Zone::Ownership DUMP_VIEWERS[] = {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN};
static bool match_frames_written = false;

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

// --state-dump json's FILE.jsonl, opened with the dump.
static FILE *json_file() {
    static FILE *f = nullptr;
    if (!f) {
        const std::string path = dump_path() + ".jsonl";
        f = std::fopen(path.c_str(), "wb");
        if (!f) fatal_error("--state-dump json: cannot open " + path);
    }
    return f;
}

// The schema the JSON lines are rendered with, decoded from this engine's own SCHEMA frame.
static const rmstate::Schema &json_schema() {
    static rmstate::Schema schema;
    static bool built = false;
    if (!built) {
        const rmstate::DecodeResult res = rmstate::decode_schema_frame(encode_schema_frame(Zone::UNKNOWN), &schema);
        if (!res.ok()) fatal_error("--state-dump json: the SCHEMA frame does not decode: " + res.message);
        built = true;
    }
    return schema;
}

static void write_record(const std::string &payload) {
    const std::string record = rmstate::frame_record(payload);
    std::fwrite(record.data(), 1, record.size(), dump_file());
    if (!state_dump_json) return;
    rmstate::DecodedFrame frame;
    const rmstate::DecodeResult res = rmstate::decode_frame(payload, json_schema(), &frame);
    if (!res.ok()) fatal_error("--state-dump json: a frame does not decode: " + res.message);
    const std::string line = rmstate::frame_json(frame, json_schema()) + "\n";
    std::fwrite(line.data(), 1, line.size(), json_file());
}

static void write_match_frames(Zone::Ownership viewer) {
    for (const std::string &payload : encode_match_frames(viewer)) {
        knowledge_feed_match_frame(ObsConsumer::DUMP, viewer, payload);
        write_record(payload);
    }
}

// The viewer's STATE frame from its dump queue, a seat's with the dump's knowledge tables.
static void write_state_frame(
    Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider, bool terminal) {
    std::string payload =
        encode_state_frame(viewer, actions, decider, take_obs_events(viewer, ObsConsumer::DUMP), terminal);
    knowledge_append_tables(ObsConsumer::DUMP, viewer, &payload);
    write_record(payload);
}

void rmstate_dump_decision(const std::vector<LegalAction> &actions) {
    if (rmstate_dump_path.empty()) return;
    const Zone::Ownership decider = rmstate_deciding_seat();
    for (Zone::Ownership viewer : DUMP_VIEWERS) {
        if (!match_frames_written) write_match_frames(viewer);
        write_state_frame(viewer, actions, decider, false);
    }
    match_frames_written = true;
    std::fflush(dump_file());
    if (state_dump_json) std::fflush(json_file());
}

void rmstate_dump_match_over() {
    if (rmstate_dump_path.empty()) return;
    for (Zone::Ownership viewer : DUMP_VIEWERS) {
        if (!match_frames_written) write_match_frames(viewer);
        write_state_frame(viewer, {}, Zone::UNKNOWN, true);
    }
    match_frames_written = true;
    std::fflush(dump_file());
    if (state_dump_json) std::fflush(json_file());
}
