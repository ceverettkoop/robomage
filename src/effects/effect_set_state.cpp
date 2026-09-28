#include "effects.h"

#include <string>

#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/characteristics.h"
#include "../queries/zones.h"
#include "../queries/affected.h"

extern Coordinator global_coordinator;

namespace effects {

static void set_object_state(const Ability &ab, Entity subject);

// DB$ SetState | Mode$ <mode> — change an object's face-up/face-down state (CR 708 / 711.8).
// The Creation of Avacyn chapter II turns the face-down exiled card face up (Mode$ TurnFaceUp),
// revealing its real characteristics so the following chapters (and this chapter's own life-loss
// rider) can read them. Structured so TurnFaceDown (and other state modes) slot in later.
// It acts on the affected objects: Defined$ ExiledWith (the card the source Saga chapter I exiled
// face down), Defined$ Self (the source) or a target.
HandlerResult set_state(Ability &ab, std::shared_ptr<Orderer> /*orderer*/, FrameCtx & /*ctx*/) {
    for (Entity subject : affected_objects(ab))
        if (global_coordinator.entity_has_component<Zone>(subject)) set_object_state(ab, subject);
    return HandlerResult::DONE_RUN_SUBS;
}

// Apply `ab`'s Mode$ to `subject`.
static void set_object_state(const Ability &ab, Entity subject) {
    auto &z = global_coordinator.GetComponent<Zone>(subject);

    if (ab.def->set_state_mode == "TurnFaceUp") {
        if (z.is_face_down) {
            z.is_face_down = false;
            // Turning it face up makes its identity public knowledge (CR 708.2) — record it in the
            // owner's revealed multi-hot so the belief-state observation now reflects the reveal (the
            // face-down exile deliberately withheld it).
            mark_card_revealed(subject, z.owner);
            game_log("%s is turned face up.\n", entity_name(subject).c_str());
        }
    } else if (ab.def->set_state_mode == "TurnFaceDown") {
        z.is_face_down = true;
        game_log("%s is turned face down.\n", entity_name(subject).c_str());
    }
}

// DB$ SetState | Mode$ <mode>. Mode is a generic script key, so claim it only on a SetState
// ability (category set from the DB$/AB$ head before params are parsed).
bool parse_set_state(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (key == "Mode" && ab.category == "SetState") {
        ab.set_state_mode = value;
        return true;
    }
    return false;
}

}  // namespace effects
