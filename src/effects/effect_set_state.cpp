#include "effects.h"

#include <string>

#include "../cli_output.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/characteristics.h"
#include "../queries/face_down.h"
#include "../queries/zones.h"
#include "../queries/visibility.h"
#include "../queries/affected.h"
#include "../state_format/events.h"
#include "../transform.h"

extern Coordinator global_coordinator;

namespace effects {

static void set_object_state(const Ability &ab, Entity subject);

// DB$ SetState | Mode$ <mode> — change an object's face-up/face-down state (CR 708 / 711.8).
// The Creation of Avacyn chapter II turns the face-down exiled card face up (Mode$ TurnFaceUp),
// revealing its real characteristics so the following chapters (and this chapter's own life-loss
// rider) can read them; TurnFaceDown turns a permanent face down where it is (CR 708.2a).
// It acts on the affected objects: Defined$ ExiledWith (the card the source Saga chapter I exiled
// face down), Defined$ Self (the source) or a target.
HandlerResult set_state(Ability &ab, std::shared_ptr<Orderer> /*orderer*/, FrameCtx & /*ctx*/) {
    for (Entity subject : affected_objects(ab))
        if (global_coordinator.entity_has_component<Zone>(subject)) set_object_state(ab, subject);
    return HandlerResult::DONE_RUN_SUBS;
}

// Apply `ab`'s Mode$ to `subject`.
static void set_object_state(const Ability &ab, Entity subject) {
    if (ab.def->set_state_mode == "TurnFaceUp") {
        if (is_face_down(subject)) {
            turn_face_up(subject);
            close_windows_on_face_up(subject);
            obs_face_turned(subject, true);
            game_log("%s is turned face up.\n", entity_name(subject).c_str());
        }
    } else if (ab.def->set_state_mode == "TurnFaceDown") {
        const std::string name = entity_name(subject);
        if (!turn_face_down(subject, SpellOverlay::FACE_DOWN)) return;
        obs_face_turned(subject, false);
        game_log("%s is turned face down.\n", name.c_str());
    } else if (ab.def->set_state_mode == "Transform") {
        // CR 701.27f: not one that transformed since this ability went on the stack (Delver of
        // Secrets, which its PeekAndReveal has just transformed).
        if (ability_may_transform_source(ab)) transform_permanent(subject);
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
