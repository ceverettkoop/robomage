#include "replay_session.h"

static unsigned next_generation();

static unsigned next_generation() {
    static unsigned counter = 0;
    return ++counter;
}

namespace rmwork {

void ReplaySession::set_match(std::unique_ptr<Match> m, size_t cursor) {
    match_ = std::move(m);
    cursor_ = 0;
    playing_ = false;
    generation_ = next_generation();
    go_to(cursor);
}

void ReplaySession::clear() {
    match_.reset();
    cursor_ = 0;
    playing_ = false;
    generation_ = next_generation();
}

const DecisionInfo *ReplaySession::current() const {
    if (!match_ || cursor_ >= match_->decisions.size()) return nullptr;
    return &match_->decisions[cursor_];
}

bool ReplaySession::go_to(size_t index) {
    if (count() == 0) return false;
    const size_t to = index >= count() ? count() - 1 : index;
    if (to != cursor_) generation_ = next_generation();
    cursor_ = to;
    return true;
}

void ReplaySession::touch() {
    generation_ = next_generation();
}

bool ReplaySession::step(int delta) {
    if (count() == 0) return false;
    if (delta < 0 && static_cast<size_t>(-delta) > cursor_) return go_to(0);
    return go_to(static_cast<size_t>(static_cast<long>(cursor_) + delta));
}

void ReplaySession::set_view(int view) {
    if (view < 0 || view >= VIEW_COUNT || view == view_) return;
    view_ = view;
    generation_ = next_generation();
}

void ReplaySession::set_playing(bool on, double now) {
    playing_ = on && count() > 0;
    if (playing_ && cursor_ + 1 >= count()) go_to(0);
    next_step_at_ = now + interval;
}

void ReplaySession::tick(double now) {
    if (!playing_ || now < next_step_at_) return;
    next_step_at_ = now + interval;
    if (cursor_ + 1 >= count()) {
        playing_ = false;
        return;
    }
    step(1);
    if (cursor_ + 1 >= count()) playing_ = false;
}

std::string ReplaySession::decision_text() const {
    if (!loaded()) return "Decision —";
    return "Decision " + std::to_string(cursor_ + 1) + " / " + std::to_string(count());
}

std::string ReplaySession::turn_text() const {
    const DecisionInfo *d = current();
    if (!d) return "Turn —";
    std::string out = "Game " + std::to_string(d->game + 1);
    if (!d->sideboarding) out += " · Turn " + std::to_string(d->turn + 1);
    if (!d->phase.empty()) out += " · " + d->phase;
    return out;
}

std::string ReplaySession::decider_text() const {
    const DecisionInfo *d = current();
    if (!d) return "Decider —";
    return d->decider == 0 ? "Player A decides" : d->decider == 1 ? "Player B decides" : "No seat decides";
}

std::string ReplaySession::game_label() const {
    if (!loaded()) return "No game loaded";
    const int g = match_->span_of(cursor_);
    return g < 0 ? "No game" : match_->games[static_cast<size_t>(g)].label;
}

}  // namespace rmwork
