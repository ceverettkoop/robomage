#include "agent.h"

#include <cctype>
#include <cstdlib>

#include "../str_util.h"
#include "explore.h"
#include "play_spec.h"

namespace agents {

namespace {

// Always row 0.
class FirstAgent : public Agent {
    public:
        int decide(const Decision &) override { return 0; }
        std::string label() const override { return "first"; }
};

// A uniform row: random.randrange(n); a menu of one row (or none) is 0 with no draw.
class RandomAgent : public Agent {
    public:
        explicit RandomAgent(PyRandom *rng) : rng_(rng) {}
        int decide(const Decision &d) override {
            const size_t n = d.view.n();
            return n <= 1 ? 0 : static_cast<int>(rng_->randrange(n));
        }
        std::string label() const override { return "random"; }

    private:
        PyRandom *rng_;
};

// A play: spec on one seat: the --play resolver with row 0 once its specs run out.
class SeatPlayAgent : public Agent {
    public:
        explicit SeatPlayAgent(std::vector<std::string> specs) : script_(std::move(specs), &first_, &first_) {}
        int decide(const Decision &d) override {
            const int r = script_.decide(d);
            if (r == AGENT_FAILED) failure_ = script_.failure();
            return r;
        }
        std::string label() const override { return "play"; }

    private:
        FirstAgent first_;
        PlayScript script_;
};

}  // namespace

static bool starts_with_ci(const std::string &s, const char *prefix, std::string *rest);

static bool starts_with_ci(const std::string &s, const char *prefix, std::string *rest) {
    const std::string p(prefix);
    if (s.size() < p.size() || ascii_lower(s.substr(0, p.size())) != p) return false;
    *rest = s.substr(p.size());
    return true;
}

bool parse_int_list(const std::string &csv, std::vector<int> *out) {
    for (const std::string &piece : split(csv, ',')) {
        size_t b = 0, e = piece.size();
        while (b < e && std::isspace(static_cast<unsigned char>(piece[b]))) ++b;
        while (e > b && std::isspace(static_cast<unsigned char>(piece[e - 1]))) --e;
        if (b == e) continue;
        const std::string t = piece.substr(b, e - b);
        char *end = nullptr;
        const long v = std::strtol(t.c_str(), &end, 10);
        if (end == t.c_str() || *end != '\0') return false;
        out->push_back(static_cast<int>(v));
    }
    return true;
}

ActionsAgent::ActionsAgent(std::vector<int> actions, Agent *player_a, Agent *player_b)
    : actions_(std::move(actions)), players_{player_a, player_b} {}

int ActionsAgent::decide(const Decision &d) {
    if (next_ < actions_.size()) return actions_[next_++];
    Agent *player = players_[d.view.seat()];
    if (!player) return 0;
    const int r = player->decide(d);
    if (r == AGENT_FAILED) failure_ = player->failure();
    return r;
}

std::unique_ptr<Agent> make_agent(const std::string &spec, PyRandom *rng, std::string *err) {
    std::string s = spec;
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    const std::string low = ascii_lower(s);
    std::string rest;
    if (low == "first" || low == "auto") return std::unique_ptr<Agent>(new FirstAgent());
    if (low == "random") return std::unique_ptr<Agent>(new RandomAgent(rng));
    if (low == "explore") return std::unique_ptr<Agent>(new ExploreAgent(rng, false));
    if (low == "explore:patient") return std::unique_ptr<Agent>(new ExploreAgent(rng, true));
    if (starts_with_ci(s, "actions:", &rest)) {
        std::vector<int> actions;
        if (!parse_int_list(rest, &actions)) {
            *err = "actions: expects comma-separated integers, got '" + rest + "'";
            return nullptr;
        }
        return std::unique_ptr<Agent>(new ActionsAgent(std::move(actions)));
    }
    if (starts_with_ci(s, "play:", &rest)) return std::unique_ptr<Agent>(new SeatPlayAgent(parse_spec_list(rest)));
    *err =
        "unknown agent spec '" + spec + "': first, random, actions:<i,...>, play:<specs>, explore or explore:patient";
    return nullptr;
}

}  // namespace agents
