#include "agent_feed.h"

#include "encode.h"
#include "events.h"
#include "knowledge_feed.h"

// Per seat (indexed by Zone::Ownership): served, and its once-per-match frames were built.
static bool served[3] = {false, false, false};
static bool match_frames_built[3] = {false, false, false};

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

void agent_feed_serve(Zone::Ownership seat) {
    if (is_seat(seat)) served[seat] = true;
}

bool agent_feed_enabled() {
    return served[Zone::PLAYER_A] || served[Zone::PLAYER_B];
}

bool agent_feed_serves(Zone::Ownership viewer) {
    return is_seat(viewer) && served[viewer];
}

void agent_feed_decision(Zone::Ownership seat, const std::vector<LegalAction> &actions,
    std::vector<std::string> *match_frames, std::string *state) {
    if (!match_frames_built[seat]) {
        for (std::string &payload : encode_match_frames(seat)) {
            knowledge_feed_match_frame(ObsConsumer::AGENT, seat, payload);
            match_frames->push_back(std::move(payload));
        }
        match_frames_built[seat] = true;
    }
    *state = encode_state_frame(seat, actions, seat, take_obs_events(seat, ObsConsumer::AGENT), false);
    knowledge_append_tables(ObsConsumer::AGENT, seat, state);
}
