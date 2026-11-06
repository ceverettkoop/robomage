#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../queries/affected.h"
#include "../queries/characteristics.h"

extern Game cur_game;

namespace effects {

static std::vector<LegalAction> tap_or_untap_menu(Entity object);

// The two rows of the choice, tap first: the choice is a mode of the effect, and a creature already
// in the chosen state stays as it is, as either row is legal on it.
static std::vector<LegalAction> tap_or_untap_menu(Entity object) {
    const std::string name = entity_name(object);
    std::vector<LegalAction> menu = {LegalAction(PASS_PRIORITY, object, "Tap " + name),
                                     LegalAction(PASS_PRIORITY, object, "Untap " + name)};
    for (size_t i = 0; i < menu.size(); i++) {
        menu[i].category = ActionCategory::CHOOSE_MODE;
        menu[i].option_ordinal = static_cast<int>(i);
        menu[i].card_is_public = true;
    }
    return menu;
}

// DB$ TapOrUntap (Sewer-veillance Cam): the controller chooses to tap or to untap what the ability
// acts on, a choice made as it resolves (CR 608.2d); the effect is then an ordinary Tap or Untap.
HandlerResult tap_or_untap(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const std::vector<Entity> objects = affected_objects(ab);
    if (objects.empty()) return HandlerResult::DONE_RUN_SUBS;
    int choice = ctx.ask(tap_or_untap_menu(objects.front()), ab.controller, ab.source.lki_entity());
    if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
    return choice == 0 ? tap(ab, orderer, ctx) : untap(ab, orderer, ctx);
}

}  // namespace effects
