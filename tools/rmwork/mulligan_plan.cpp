#include "mulligan_plan.h"

namespace rmwork {

bool london_bottom_menu(const agents::DecisionView &v) {
    if (v.n() == 0 || v.table("pending").rows() != 0) return false;
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) != "BOTTOM_DECK_CARD") return false;
    return true;
}

int next_bottom_row(const CardPlan &plan, const agents::DecisionView &v, uint64_t *card) {
    if (!london_bottom_menu(v)) return -1;
    return next_planned_row(plan, v, "BOTTOM_DECK_CARD", card);
}

}  // namespace rmwork
