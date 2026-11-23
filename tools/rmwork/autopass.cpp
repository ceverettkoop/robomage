#include "autopass.h"

#include <algorithm>

static bool gained(const std::vector<uint64_t> &before, const std::vector<uint64_t> &now);

// Whether `now` holds a handle `before` does not.
static bool gained(const std::vector<uint64_t> &before, const std::vector<uint64_t> &now) {
    for (uint64_t h : now)
        if (std::find(before.begin(), before.end(), h) == before.end()) return true;
    return false;
}

namespace rmwork {

PassFacts read_pass_facts(const agents::DecisionView &v) {
    PassFacts f;
    f.rows = static_cast<int>(v.n());
    if (!v.table("pending").rows() && !v.table("pending_cost").rows())
        for (size_t i = 0; i < v.n() && f.pass_row < 0; ++i)
            if (v.category(i) == "PASS_PRIORITY") f.pass_row = static_cast<int>(i);
    const agents::Table header = v.table("header");
    if (header.rows()) {
        f.game = static_cast<int>(header.u(0, header.col("game")));
        f.sideboarding = header.u(0, header.col("sideboarding")) != 0;
    }
    const agents::Table turn = v.table("turn");
    if (turn.rows() && !f.sideboarding) {
        f.turn = turn.u(0, turn.col("turn"));
        const uint64_t me =
            static_cast<uint64_t>(v.stream().enum_value("seat", v.seat() == 0 ? "PLAYER_A" : "PLAYER_B"));
        f.own_turn = turn.u(0, turn.col("active")) == me;
        f.step = stop_step_of(v.stream().enum_name("step", turn.u(0, turn.col("step"))));
    }
    const agents::Table &stack = v.stack();
    for (size_t r = 0; r < stack.rows(); ++r) f.stack.push_back(stack.u(r, stack.col("handle")));
    return f;
}

void start_auto_pass(AutoPass *ap, PassMode mode, const PassFacts &at) {
    ap->mode = mode;
    ap->game = at.game;
    ap->sideboarding = at.sideboarding;
    ap->turn = at.turn;
    ap->stack = at.stack;
}

AutoReply auto_reply(AutoPass *ap, const PassFacts &f, const StopSettings &stops) {
    AutoReply r;
    if (ap->mode != PassMode::NONE) {
        if (f.game != ap->game || f.sideboarding != ap->sideboarding || f.turn != ap->turn) {
            r.ended = "the turn ended";
            ap->mode = PassMode::NONE;
        } else if (ap->mode == PassMode::SKIP_TURN) {
            r.row = 0;
            return r;
        } else if (f.pass_row < 0) {
            r.ended = "this decision is not a priority pass";
            ap->mode = PassMode::NONE;
            return r;
        } else if (gained(ap->stack, f.stack)) {
            r.ended = "something went on the stack";
            ap->mode = PassMode::NONE;
            ap->stack = f.stack;
            return r;
        } else {
            ap->stack = f.stack;
            r.row = f.pass_row;
            return r;
        }
    }
    if (f.pass_row < 0 || !f.stack.empty() || f.step < 0) return r;
    const bool stop = f.own_turn ? stops.own[static_cast<size_t>(f.step)] : stops.opponent[static_cast<size_t>(f.step)];
    if (stop) return r;
    r.row = f.pass_row;
    r.by_stop = true;
    return r;
}

}  // namespace rmwork
