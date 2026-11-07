#include "speed.h"

#include <algorithm>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../ecs/coordinator.h"
#include "players.h"

static void set_speed(Zone::Ownership who, int speed);

static void set_speed(Zone::Ownership who, int speed) {
    global_coordinator.GetComponent<Player>(get_player_entity(who)).speed = speed;
    game_log("%s's speed is %d\n", player_name(who).c_str(), speed);
}

void start_speed(Zone::Ownership who) { set_speed(who, 1); }

void increase_speed(Zone::Ownership who, int by) {
    const Player &pl = global_coordinator.GetComponent<Player>(get_player_entity(who));
    set_speed(who, std::min(kMaxSpeed, speed_of(pl) + by));
}
