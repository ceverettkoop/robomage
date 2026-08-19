#ifndef ACTOR_ORACLE_CLIENT_H
#define ACTOR_ORACLE_CLIENT_H

// Client for the scripted-agent oracle (train/scripted_oracle.py): a Unix-socket
// server that answers scripted:hard actions from the machine-mode observation.
// Used by the actor's vs-scripted mode (--scripted-seat A|B --scripted-oracle
// <socket>): the scripted seat's REAL decisions are shipped to the oracle so the
// rule-based agent stays defined once, in Python — the actor never reimplements
// it. Wire protocol (little-endian `int32 kind, int32 len, payload` frames;
// kinds HELLO=1 / NEW_GAME=2 / QUERY=3) is documented in scripted_oracle.py.
//
// Every failure (connect, short read/write, layout mismatch, out-of-range
// reply) is FATAL AND LOUD via fatal_error — a silently wrong opponent action
// would corrupt the training stream, so there is no retry or fallback.
//
// Plain POSIX sockets, no torch — listed in the Makefile's ACTOR_SYNTAX_SRCS so
// the default `make check` (actorobs tier) keeps it compiling without libtorch.

#include <string>

class ScriptedOracle {
public:
    ScriptedOracle() = default;
    ~ScriptedOracle();
    ScriptedOracle(const ScriptedOracle&) = delete;
    ScriptedOracle& operator=(const ScriptedOracle&) = delete;

    // Connect and HELLO: configures the connection's per-agent state (deck
    // names + tier spec) and verifies the oracle's OBS_SIZE equals the actor's
    // ACTOR_OBS_SIZE (a stale oracle would misread every obs offset).
    void connect(const std::string& socket_path, const std::string& deck_a,
                 const std::string& deck_b, const std::string& spec);

    // agent.new_game() — send at match start and after every completed game,
    // the exact call sites az_selfplay's Python backend uses. No reply.
    void new_game();

    // One scripted decision: obs points at ACTOR_OBS_SIZE floats. Returns the
    // agent's action index in [0, num_choices).
    int act(const float* obs, int num_choices);

private:
    void send_all(const void* buf, long n);
    void recv_all(void* buf, long n);
    void send_frame(int kind, const void* payload, int len);
    // Read one frame, assert its kind, and require exactly `len` payload bytes.
    void recv_frame(int expect_kind, void* payload, int len);

    int fd_ = -1;
};

#endif /* ACTOR_ORACLE_CLIENT_H */
