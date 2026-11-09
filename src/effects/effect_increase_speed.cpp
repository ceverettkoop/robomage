#include "effects.h"

#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../queries/players.h"
#include "../queries/speed.h"

namespace effects {

// CR 702.179d: "... if your speed is less than 4, your speed increases by 1." Its intervening if is
// read again here (CR 603.4); collect_speed_triggers read it as it triggered.
HandlerResult increase_speed(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    const Player &pl = global_coordinator.GetComponent<Player>(get_player_entity(ab.controller));
    if (speed_of(pl) < kMaxSpeed) ::increase_speed(ab.controller, 1);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
