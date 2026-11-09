#include "face_down.h"

#include <memory>
#include <utility>

#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/entry_info.h"
#include "../components/permanent.h"
#include "characteristics.h"
#include "copies.h"
#include "entry.h"
#include "overlays.h"

static void set_card_aside(Entity card, SpellOverlay how);

// `card` lies on the face `how` gives it, its card under it.
static void set_card_aside(Entity card, SpellOverlay how) {
    CardData &values = global_coordinator.GetComponent<CardData>(card);
    global_coordinator.AddComponent(card, FaceDownCard{std::make_shared<const CardData>(std::move(values))});
    values = face_down_face(how);
    global_coordinator.GetComponent<Zone>(card).is_face_down = true;
}

void take_card_back(Entity e) {
    global_coordinator.GetComponent<CardData>(e) = *global_coordinator.GetComponent<FaceDownCard>(e).card;
    global_coordinator.RemoveComponent<FaceDownCard>(e);
    global_coordinator.GetComponent<Zone>(e).is_face_down = false;
}

void enter_face_down(Entity card, SpellOverlay how) {
    set_card_aside(card, how);
    entry_info(card).overlay = how;
}

bool turn_face_down(Entity e, SpellOverlay how) {
    if (is_face_down(e)) return false;
    if (!global_coordinator.entity_has_component<Permanent>(e)) {
        global_coordinator.GetComponent<Zone>(e).is_face_down = true;
        return true;
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    set_card_aside(e, how);
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    perm.overlay = how;
    perm.transformed = false;
    permanent_takes_values(e, global_coordinator.GetComponent<CardData>(e));
    return true;
}

void turn_face_up(Entity e) {
    if (!face_down_card(e)) {
        global_coordinator.GetComponent<Zone>(e).is_face_down = false;
        return;
    }
    take_card_back(e);
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    global_coordinator.GetComponent<Permanent>(e).overlay = SpellOverlay::NONE;
    permanent_takes_values(e, global_coordinator.GetComponent<CardData>(e));
}

void reveal_as_it_leaves(Entity e) {
    if (!face_down_card(e)) return;
    const std::string was = entity_name(e);
    take_card_back(e);
    const CardData &card = global_coordinator.GetComponent<CardData>(e);
    game_log("%s is revealed as %s.\n", was.c_str(), card.name.c_str());
    if (!global_coordinator.entity_has_component<Permanent>(e)) return;
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    perm.overlay = SpellOverlay::NONE;
    perm.name = card.name;
    perm.types = card.types;
}
