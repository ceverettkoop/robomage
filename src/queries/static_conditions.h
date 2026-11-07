#ifndef QUERIES_STATIC_CONDITIONS_H
#define QUERIES_STATIC_CONDITIONS_H

#include <string>

#include "../components/zone.h"
#include "../ecs/entity_set.h"

// ── A static's Condition$ (CR 604.3) ─────────────────────────────────────────────────────────────
//
// Each name a static's Condition$ can give is one row of kStaticConditionRows: the name and when it
// holds for the static's controller. The parser's report and the static pass both read the table.

// A row of the table.
struct StaticConditionRow {
        const char *name;
        bool (*met)(Zone::Ownership controller, const EntitySet &entities);
};

// The row of `name`, or nullptr for a name no static reads.
const StaticConditionRow *static_condition_row(const std::string &name);

// Each row's answer for one pass over the statics, asked at most once per seat.
class StaticConditionAnswers {
    public:
        bool met(const StaticConditionRow &row, Zone::Ownership controller, const EntitySet &entities);

    private:
        static constexpr size_t kMaxRows = 8;
        signed char answers_[kMaxRows][2] = {};  // 0 not asked, 1 holds, -1 doesn't
};

#endif /* QUERIES_STATIC_CONDITIONS_H */
