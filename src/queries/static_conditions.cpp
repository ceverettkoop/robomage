#include "static_conditions.h"

#include "../classes/game.h"
#include "zones.h"

extern Game cur_game;

static bool condition_delirium(Zone::Ownership controller, const EntitySet &entities);
static bool condition_player_turn(Zone::Ownership controller, const EntitySet &entities);

static const StaticConditionRow kStaticConditionRows[] = {
    {"Delirium", condition_delirium},
    {"PlayerTurn", condition_player_turn},
};

static bool condition_delirium(Zone::Ownership controller, const EntitySet &entities) {
    return check_delirium(controller, entities);
}

// Active during the source controller's own turn (Voice of Victory).
static bool condition_player_turn(Zone::Ownership controller, const EntitySet &) {
    const bool a_turn = cur_game.turn_state.player_a_turn;
    return controller == Zone::PLAYER_A ? a_turn : !a_turn;
}

const StaticConditionRow *static_condition_row(const std::string &name) {
    for (const StaticConditionRow &row : kStaticConditionRows)
        if (name == row.name) return &row;
    return nullptr;
}

bool StaticConditionAnswers::met(const StaticConditionRow &row, Zone::Ownership controller, const EntitySet &entities) {
    static_assert(sizeof(kStaticConditionRows) / sizeof(kStaticConditionRows[0]) <= kMaxRows, "");
    signed char &answer = answers_[&row - kStaticConditionRows][controller == Zone::PLAYER_A ? 0 : 1];
    if (answer == 0) answer = row.met(controller, entities) ? 1 : -1;
    return answer > 0;
}
