#include "card_plan.h"

#include <algorithm>

namespace rmwork {

void CardPlan::follow(const CardPlanKey &key) {
    if (key == key_) return;
    key_ = key;
    clear();
}

void CardPlan::clear() {
    cards_.clear();
    sending_ = false;
    answered_ = -1;
    ++generation_;
}

int CardPlan::order_of(uint64_t h) const {
    auto it = std::find(cards_.begin(), cards_.end(), h);
    return it == cards_.end() ? 0 : static_cast<int>(it - cards_.begin()) + 1;
}

bool CardPlan::toggle(uint64_t h, int owed) {
    auto it = std::find(cards_.begin(), cards_.end(), h);
    if (it != cards_.end()) {
        cards_.erase(it);
        ++generation_;
        return true;
    }
    if (static_cast<int>(cards_.size()) >= owed) return false;
    cards_.push_back(h);
    ++generation_;
    return true;
}

bool CardPlan::undo() {
    if (cards_.empty()) return false;
    cards_.pop_back();
    ++generation_;
    return true;
}

int next_planned_row(const CardPlan &plan, const agents::DecisionView &v, const char *category, uint64_t *card) {
    if (v.n() == 0) return -1;
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) != category) return -1;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source");
    for (uint64_t h : plan.cards())
        for (size_t i = 0; i < v.n(); ++i)
            if (menu.u(i, c_src) == h) {
                if (card) *card = h;
                return static_cast<int>(i);
            }
    return -1;
}

}  // namespace rmwork
