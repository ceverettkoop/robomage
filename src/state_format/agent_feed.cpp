#include "agent_feed.h"

#include "encode.h"
#include "events.h"
#include "knowledge_feed.h"

// Per seat (indexed by Zone::Ownership): served, and its once-per-match frames were built.
static bool served[3] = {false, false, false};
static bool match_frames_built[3] = {false, false, false};

static bool is_seat(Zone::Ownership p);
static void match_frames_once(Zone::Ownership seat, std::vector<std::string> *match_frames);

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

void agent_feed_serve(Zone::Ownership seat) {
    if (is_seat(seat)) served[seat] = true;
}

void agent_feed_reset() {
    for (int p = 0; p < 3; ++p) served[p] = match_frames_built[p] = false;
}

bool agent_feed_enabled() {
    return served[Zone::PLAYER_A] || served[Zone::PLAYER_B];
}

bool agent_feed_serves(Zone::Ownership viewer) {
    return is_seat(viewer) && served[viewer];
}

// The seat's once-per-match frames, the first time it gets a frame in the match.
static void match_frames_once(Zone::Ownership seat, std::vector<std::string> *match_frames) {
    if (match_frames_built[seat]) return;
    for (std::string &payload : encode_match_frames(seat)) {
        knowledge_feed_match_frame(ObsConsumer::AGENT, seat, payload);
        match_frames->push_back(std::move(payload));
    }
    match_frames_built[seat] = true;
}

void agent_feed_decision(Zone::Ownership seat, const std::vector<LegalAction> &actions,
    std::vector<std::string> *match_frames, std::string *state) {
    match_frames_once(seat, match_frames);
    *state = encode_state_frame(seat, actions, seat, take_obs_events(seat, ObsConsumer::AGENT), false);
    knowledge_append_tables(ObsConsumer::AGENT, seat, state);
}

void agent_feed_terminal(Zone::Ownership seat, std::vector<std::string> *match_frames, std::string *state) {
    match_frames_once(seat, match_frames);
    *state = encode_state_frame(seat, {}, Zone::UNKNOWN, take_obs_events(seat, ObsConsumer::AGENT), true);
    knowledge_append_tables(ObsConsumer::AGENT, seat, state);
}
