#include "explore.h"

#include <algorithm>
#include <map>
#include <vector>

namespace agents {

static double category_weight(const std::string &cat, bool patient);

static constexpr double DEFAULT_WEIGHT = 2.0;
static constexpr double SEARCH_FAIL_WEIGHT = 0.3;
static constexpr double NOVELTY_FACTOR = 3.0;
static constexpr double PATIENT_CAST_MV_FACTOR = 1.0;
static constexpr double PATIENT_HOLD_MV_FACTOR = 1.0;

// The profiles' category weights (explore_agent._EXPLORE_WEIGHTS / _EXPLORE_PATIENT_WEIGHTS).
static double category_weight(const std::string &cat, bool patient) {
    static const std::map<std::string, double> plain = {{"CAST_SPELL", 8.0}, {"ACTIVATE_ABILITY", 6.0},
        {"PLAY_LAND", 4.0}, {"SELECT_ATTACKER", 3.0}, {"SELECT_BLOCKER", 3.0}, {"PASS_PRIORITY", 1.0},
        {"CONFIRM_ATTACKERS", 0.6}, {"CONFIRM_BLOCKERS", 0.6}, {"SIDEBOARD_DONE", 0.6}, {"COMPANION", 8.0}};
    static const std::map<std::string, double> big_mana = {{"CAST_SPELL", 8.0}, {"ACTIVATE_ABILITY", 1.0},
        {"PLAY_LAND", 100.0}, {"SELECT_ATTACKER", 3.0}, {"SELECT_BLOCKER", 3.0}, {"PASS_PRIORITY", 6.0},
        {"CONFIRM_ATTACKERS", 0.6}, {"CONFIRM_BLOCKERS", 0.6}, {"SIDEBOARD_DONE", 0.6}, {"COMPANION", 8.0}};
    const std::map<std::string, double> &table = patient ? big_mana : plain;
    auto it = table.find(cat);
    return it != table.end() ? it->second : DEFAULT_WEIGHT;
}

int ExploreAgent::decide(const Decision &d) {
    const DecisionView &v = d.view;
    const size_t n = v.n();
    if (n <= 1) return 0;
    std::vector<const std::string *> cats(n);
    for (size_t i = 0; i < n; ++i) cats[i] = &v.category(i);
    if (std::any_of(cats.begin(), cats.end(), [](const std::string *c) { return *c == "MULLIGAN"; })) seen_.clear();
    std::vector<std::string> hand;
    bool hand_read = false;
    std::vector<double> weights(n);
    for (size_t i = 0; i < n; ++i) {
        const std::string &c = *cats[i];
        double w = category_weight(c, patient_);
        if (c == "SEARCH_LIBRARY" && !v.card(i)) w *= SEARCH_FAIL_WEIGHT;
        if (c == "CAST_SPELL" || c == "ACTIVATE_ABILITY") {
            const std::optional<std::string> card = v.card(i);
            if (card && !seen_.count(*card) && (c == "CAST_SPELL" || !patient_)) w *= NOVELTY_FACTOR;
            if (patient_ && card) {
                if (c == "CAST_SPELL") {
                    w *= 1.0 + v.cards().mana_value(*card) * PATIENT_CAST_MV_FACTOR;
                } else {
                    if (!hand_read) {
                        hand = v.hand();
                        hand_read = true;
                    }
                    if (std::find(hand.begin(), hand.end(), *card) != hand.end())
                        w /= 1.0 + v.cards().mana_value(*card) * PATIENT_HOLD_MV_FACTOR;
                }
            }
        }
        weights[i] = w;
    }
    if (n == 2 && *cats[0] == "MULLIGAN" && *cats[1] == "MULLIGAN") weights = {3.0, 1.0};
    double total = 0.0;
    for (double w : weights) total += w;
    if (total <= 0) std::fill(weights.begin(), weights.end(), 1.0);
    const size_t choice = rng_->weighted_index(weights.data(), n);
    if (*cats[choice] == "CAST_SPELL" || *cats[choice] == "ACTIVATE_ABILITY") {
        const std::optional<std::string> card = v.card(choice);
        if (card) seen_.insert(*card);
    }
    return static_cast<int>(choice);
}

}  // namespace agents
