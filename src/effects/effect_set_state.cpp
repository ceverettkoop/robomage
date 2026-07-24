#include "effects.h"

#include <string>

#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;

namespace effects {

// Resolve the entity a SetState acts on. Today the only Defined$ form used is ExiledWith (the
// card the source Saga chapter I exiled face down); Defined$ Self falls back to the source. A
// future SetState on a target would read ab.target here.
static Entity set_state_subject(const Ability &ab) {
    if (ab.defined_exiled_with) return exiled_with_card(ab.source);
    if (ab.defined_self) return ab.source;
    return ab.target;
}

// DB$ SetState | Mode$ <mode> — change an object's face-up/face-down state (CR 708 / 711.8).
// The Creation of Avacyn chapter II turns the face-down exiled card face up (Mode$ TurnFaceUp),
// revealing its real characteristics so the following chapters (and this chapter's own life-loss
// rider) can read them. Structured so TurnFaceDown (and other state modes) slot in later.
HandlerResult set_state(Ability &ab, std::shared_ptr<Orderer> /*orderer*/, FrameCtx & /*ctx*/) {
    Entity subject = set_state_subject(ab);
    if (subject == 0 || !global_coordinator.entity_has_component<Zone>(subject))
        return HandlerResult::DONE_RUN_SUBS;
    auto &z = global_coordinator.GetComponent<Zone>(subject);

    if (ab.set_state_mode == "TurnFaceUp") {
        if (z.is_face_down) {
            z.is_face_down = false;
            // Turning it face up makes its identity public knowledge (CR 708.2) — record it in the
            // owner's revealed multi-hot so the belief-state observation now reflects the reveal (the
            // face-down exile deliberately withheld it).
            mark_card_revealed(subject, z.owner);
            game_log("%s is turned face up.\n", entity_name(subject).c_str());
        }
    } else if (ab.set_state_mode == "TurnFaceDown") {
        z.is_face_down = true;
        game_log("%s is turned face down.\n", entity_name(subject).c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// DB$ SetState | Mode$ <mode>. Mode is a generic script key, so claim it only on a SetState
// ability (category set from the DB$/AB$ head before params are parsed).
bool parse_set_state(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "Mode" && ab.category == "SetState") {
        ab.set_state_mode = value;
        return true;
    }
    return false;
}

}  // namespace effects
