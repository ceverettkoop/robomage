#include "visibility.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

#include "../cli_output.h"
#include "../components/zone.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../game_driver.h"
#include "../state_format/handles.h"
#include "characteristics.h"
#include "players.h"
#include "zones.h"

using Window = Game::RevealWindow;

static bool is_seat(Zone::Ownership p);
static Entity window_card(const Window &w);
static bool window_shown_to(const Window &w, Zone::Ownership viewer);
static bool in_hidden_zone(Entity card);
static bool sees_without_window(Entity card, Zone::Ownership viewer);
static bool outlives(Window::End a, Window::End b);
static void open_window(
    Entity card, Zone::Ownership audience, Window::Kind kind, Window::End until, Entity stack_object);
static std::vector<Entity> canonical_order(std::vector<Entity> cards);
static bool canonical_before(Entity a, Entity b);
template <class Pred>
static void close_windows_if(Pred pred, const char *reason);
static void trace_window(const char *verb, const Window &w, Entity card, const char *detail);
static const char *kind_word(Window::Kind kind);
static const char *end_word(Window::End until);
struct CanonicalKey;
static CanonicalKey canonical_key(Entity e, Zone::Ownership viewer);
static int zone_rank(Zone::ZoneValue zone);
static int seat_rank(Zone::Ownership seat, Zone::Ownership viewer);
static bool arrival_visible(const Zone &z, Zone::Ownership viewer);

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

// The card `w` shows, while it is still the object the window opened on (an exact identity stamp:
// the card moving makes it a new object, CR 400.7, and a window never follows it); else 0.
static Entity window_card(const Window &w) {
    const Entity e = w.card.lki_entity();
    if (e == 0 || !global_coordinator.entity_has_component<Zone>(e)) return 0;
    return global_coordinator.GetComponent<Zone>(e).obj_gen == w.card.gen ? e : 0;
}

static bool window_shown_to(const Window &w, Zone::Ownership viewer) {
    return !is_seat(viewer) || w.audience == Zone::UNKNOWN || w.audience == viewer;
}

// A card whose identity is not public where it is: in a library, a hand or the sideboard, or face
// down (CR 400.2, 406.3).
static bool in_hidden_zone(Entity card) {
    if (!global_coordinator.entity_has_component<Zone>(card)) return false;
    const Zone &z = global_coordinator.GetComponent<Zone>(card);
    return !is_public_zone(z.location) || z.is_face_down;
}

// What `viewer` sees of `card` by the zone rules alone, without a window.
static bool sees_without_window(Entity card, Zone::Ownership viewer) {
    if (!is_seat(viewer) || !global_coordinator.entity_has_component<Zone>(card)) return true;
    const Zone &z = global_coordinator.GetComponent<Zone>(card);
    if (z.is_face_down) {
        // CR 708.5: the controller may look at a face-down permanent or spell; a face-down card in
        // any other zone is seen only through a window (CR 406.3).
        return (z.location == Zone::BATTLEFIELD || z.location == Zone::STACK) && z.controller == viewer;
    }
    if (is_public_zone(z.location)) return true;
    return (z.location == Zone::HAND || z.location == Zone::SIDEBOARD) && z.owner == viewer;
}

// Whether a window lasting until `a` lasts at least as long as one lasting until `b`.
static bool outlives(Window::End a, Window::End b) {
    return static_cast<int>(a) > static_cast<int>(b);
}

static void open_window(
    Entity card, Zone::Ownership audience, Window::Kind kind, Window::End until, Entity stack_object) {
    if (card == 0 || !in_hidden_zone(card)) return;
    if (is_seat(audience) && sees_without_window(card, audience)) return;
    for (Window &w : cur_game.reveal_windows) {
        if (window_card(w) != card || w.audience != audience) continue;
        if (outlives(until, w.until)) {
            w.kind = kind;
            w.until = until;
            w.stack_object = ObjectRef::of(stack_object);
            trace_window("extend", w, card, end_word(until));
        }
        return;
    }
    Window w;
    w.card = ObjectRef::of(card);
    w.audience = audience;
    w.kind = kind;
    w.until = until;
    w.stack_object = ObjectRef::of(stack_object);
    cur_game.reveal_windows.push_back(w);
    trace_window("open", w, card, end_word(until));
}

// `cards` in an order every player could derive: a library from the top, other zones by card name
// (a hand's arrival order is hidden from the opponent).
static std::vector<Entity> canonical_order(std::vector<Entity> cards) {
    std::stable_sort(cards.begin(), cards.end(), canonical_before);
    return cards;
}

static bool canonical_before(Entity a, Entity b) {
    const bool za = global_coordinator.entity_has_component<Zone>(a);
    const bool zb = global_coordinator.entity_has_component<Zone>(b);
    if (!za || !zb) return za > zb;
    const Zone &x = global_coordinator.GetComponent<Zone>(a);
    const Zone &y = global_coordinator.GetComponent<Zone>(b);
    if (x.owner != y.owner) return x.owner < y.owner;
    if (x.location != y.location) return x.location < y.location;
    if (x.location == Zone::LIBRARY) return x.distance_from_top < y.distance_from_top;
    return entity_name(a) < entity_name(b);
}

// Ends every window for which pred(window, card) holds, narrating each with `reason`; a window
// whose card is no longer that object is dropped on the way.
// A viewer whose handle for a closed window's card depended on that window loses the handle here,
// as the window closes (state_format/handles.h).
template <class Pred>
static void close_windows_if(Pred pred, const char *reason) {
    std::vector<Window> &windows = cur_game.reveal_windows;
    std::vector<Entity> closed;
    size_t kept = 0;
    for (size_t i = 0; i < windows.size(); ++i) {
        const Window &w = windows[i];
        const Entity card = window_card(w);
        if (card == 0) {
            trace_window("close", w, 0, "gone");
            continue;
        }
        if (pred(w, card)) {
            trace_window("close", w, card, reason);
            closed.push_back(card);
            continue;
        }
        if (kept != i) windows[kept] = windows[i];
        ++kept;
    }
    windows.resize(kept);
    if (!closed.empty()) retire_unidentifiable_handles(closed);
}

// The --trace-visibility narration of one window event (a debugging aid; off by default).
static void trace_window(const char *verb, const Window &w, Entity card, const char *detail) {
    if (!trace_visibility) return;
    std::string where = "(gone)";
    if (card != 0) {
        const Zone &z = global_coordinator.GetComponent<Zone>(card);
        where = std::string("in ") + trace_zone_word(z.location) + (z.is_face_down ? " face down" : "") + " of " +
                trace_seat_word(z.owner);
        if (z.location == Zone::LIBRARY) where += " at " + std::to_string(z.distance_from_top);
    }
    const std::string name = card != 0 ? entity_name(card) : std::string("a card");
    game_log("VISIBILITY: %s %s %s %s to %s (%s)\n", verb, kind_word(w.kind), name.c_str(), where.c_str(),
        w.audience == Zone::UNKNOWN ? "all" : trace_seat_word(w.audience), detail);
}

static const char *kind_word(Window::Kind kind) {
    switch (kind) {
        case Window::REVEAL:
            return "reveal";
        case Window::LOOK:
            return "look";
        case Window::FACE_DOWN_LOOK:
            return "face-down-look";
        case Window::CONTINUOUS_REVEAL:
            return "continuous-reveal";
    }
    return "window";
}

static const char *end_word(Window::End until) {
    switch (until) {
        case Window::RESOLUTION:
            return "until resolution end";
        case Window::STACK_OBJECT:
            return "until its ability leaves the stack";
        case Window::LEAVES_ZONE:
            return "until it leaves its zone";
    }
    return "";
}

const char *trace_zone_word(Zone::ZoneValue zone) {
    switch (zone) {
        case Zone::LIBRARY:
            return "library";
        case Zone::BATTLEFIELD:
            return "battlefield";
        case Zone::HAND:
            return "hand";
        case Zone::STACK:
            return "stack";
        case Zone::GRAVEYARD:
            return "graveyard";
        case Zone::EXILE:
            return "exile";
        case Zone::SIDEBOARD:
            return "sideboard";
    }
    return "zone";
}

const char *trace_seat_word(Zone::Ownership seat) {
    return seat == Zone::PLAYER_A ? "A" : seat == Zone::PLAYER_B ? "B" : "?";
}

// What canonical_object_order sorts by (see visibility.h), in comparison order.
struct CanonicalKey {
        int zone = 0;
        int seat = 0;
        size_t pos = 0;
        std::string name;  // empty when the viewer cannot see the object's identity
        size_t timestamp = 0;
        uint64_t arrival = 0;
        Entity entity = 0;
        bool operator<(const CanonicalKey &o) const {
            return std::tie(zone, seat, pos, name, timestamp, arrival, entity) <
                   std::tie(o.zone, o.seat, o.pos, o.name, o.timestamp, o.arrival, o.entity);
        }
};

static CanonicalKey canonical_key(Entity e, Zone::Ownership viewer) {
    CanonicalKey k;
    k.entity = e;
    if (!global_coordinator.entity_has_component<Zone>(e)) return k;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    const bool visible = identity_visible(e, viewer);
    k.zone = zone_rank(z.location);
    if (z.location == Zone::BATTLEFIELD)
        k.seat = seat_rank(source_controller(e), viewer);
    else if (z.location != Zone::STACK)
        k.seat = seat_rank(z.owner, viewer);
    if (order_visible(z.location, z.owner, viewer) || (z.location == Zone::LIBRARY && visible))
        k.pos = z.distance_from_top;
    if (visible) k.name = entity_name(e);
    if (z.location == Zone::BATTLEFIELD)
        k.timestamp = global_coordinator.entity_has_component<Permanent>(e)
                          ? global_coordinator.GetComponent<Permanent>(e).timestamp_entered_battlefield
                          : SIZE_MAX;
    if (arrival_visible(z, viewer)) k.arrival = z.obj_gen;
    return k;
}

static int zone_rank(Zone::ZoneValue zone) {
    switch (zone) {
        case Zone::BATTLEFIELD:
            return 0;
        case Zone::STACK:
            return 1;
        case Zone::GRAVEYARD:
            return 2;
        case Zone::EXILE:
            return 3;
        case Zone::HAND:
            return 4;
        case Zone::LIBRARY:
            return 5;
        case Zone::SIDEBOARD:
            return 6;
    }
    return 7;
}

// The viewer's own seat first (Player A's for the omniscient viewer), then the other, then none.
static int seat_rank(Zone::Ownership seat, Zone::Ownership viewer) {
    if (!is_seat(seat)) return 2;
    const Zone::Ownership self = is_seat(viewer) ? viewer : Zone::PLAYER_A;
    return seat == self ? 0 : 1;
}

// Whether `viewer` saw the object enter where it is now: every entry into a public zone is seen
// by both players, and an entry into a hand by its owner.
static bool arrival_visible(const Zone &z, Zone::Ownership viewer) {
    return !is_seat(viewer) || is_public_zone(z.location) || (z.location == Zone::HAND && z.owner == viewer);
}

bool individually_identifiable(Entity entity, Zone::Ownership viewer) {
    if (!global_coordinator.entity_has_component<Zone>(entity)) return false;
    if (!is_seat(viewer)) return true;
    if (is_public_zone(global_coordinator.GetComponent<Zone>(entity).location)) return true;
    return identity_visible(entity, viewer);
}

std::vector<Entity> canonical_object_order(std::vector<Entity> objects, Zone::Ownership viewer) {
    std::vector<CanonicalKey> keys;
    keys.reserve(objects.size());
    for (Entity e : objects) keys.push_back(canonical_key(e, viewer));
    std::sort(keys.begin(), keys.end());
    for (size_t i = 0; i < keys.size(); ++i) objects[i] = keys[i].entity;
    return objects;
}

bool identity_visible(Entity entity, Zone::Ownership viewer) {
    return sees_without_window(entity, viewer) || in_reveal_window(entity, viewer);
}

bool order_visible(Zone::ZoneValue zone, Zone::Ownership /*owner*/, Zone::Ownership /*viewer*/) {
    return zone == Zone::STACK || zone == Zone::GRAVEYARD || zone == Zone::EXILE;
}

std::vector<VisibleWindow> reveal_windows_for(Zone::Ownership viewer) {
    std::vector<VisibleWindow> out;
    for (const Window &w : cur_game.reveal_windows) {
        const Entity card = window_card(w);
        if (card == 0 || !window_shown_to(w, viewer)) continue;
        VisibleWindow v;
        v.card = card;
        v.kind = w.kind;
        v.until = w.until;
        v.audience = w.audience;
        out.push_back(v);
    }
    return out;
}

bool in_reveal_window(Entity card, Zone::Ownership viewer) {
    if (card == 0) return false;
    for (const Window &w : cur_game.reveal_windows)
        if (window_card(w) == card && window_shown_to(w, viewer)) return true;
    return false;
}

void reveal_card(Entity card) {
    open_window(card, Zone::UNKNOWN, Window::REVEAL, Window::RESOLUTION, 0);
}

void reveal_cards(const std::vector<Entity> &cards) {
    for (Entity card : canonical_order(cards)) reveal_card(card);
}

void look_at_card(Entity card, Zone::Ownership looker) {
    if (is_seat(looker)) open_window(card, looker, Window::LOOK, Window::RESOLUTION, 0);
}

void look_at_cards(const std::vector<Entity> &cards, Zone::Ownership looker) {
    for (Entity card : canonical_order(cards)) look_at_card(card, looker);
}

void reveal_until_leaves_stack(Entity card, Entity stack_object) {
    open_window(card, Zone::UNKNOWN, Window::REVEAL, Window::STACK_OBJECT, stack_object);
}

void reveal_continuously(Entity card) {
    open_window(card, Zone::UNKNOWN, Window::CONTINUOUS_REVEAL, Window::LEAVES_ZONE, 0);
}

HiddenViewers hidden_identity_viewers(Entity card) {
    HiddenViewers saw;
    if (!in_hidden_zone(card)) return saw;
    saw.a = identity_visible(card, Zone::PLAYER_A);
    saw.b = identity_visible(card, Zone::PLAYER_B);
    return saw;
}

void close_windows_on_zone_change(Entity card) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if([card](const Window &, Entity e) { return e == card; }, "zone change");
}

void open_face_down_windows(Entity card, HiddenViewers saw) {
    if (saw.a) open_window(card, Zone::PLAYER_A, Window::FACE_DOWN_LOOK, Window::LEAVES_ZONE, 0);
    if (saw.b) open_window(card, Zone::PLAYER_B, Window::FACE_DOWN_LOOK, Window::LEAVES_ZONE, 0);
}

void close_windows_on_face_up(Entity card) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if([card](const Window &, Entity e) { return e == card; }, "turned face up");
}

void close_windows_on_shuffle(Zone::Ownership owner) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if(
        [owner](const Window &, Entity e) {
            const Zone &z = global_coordinator.GetComponent<Zone>(e);
            return z.location == Zone::LIBRARY && z.owner == owner;
        },
        "shuffle");
}

void close_resolution_windows(Entity resolving) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if(
        [resolving](const Window &w, Entity) {
            if (w.until == Window::RESOLUTION) return true;
            if (w.until != Window::STACK_OBJECT) return false;
            const Entity s = w.stack_object.get();
            return s == 0 || s == resolving || global_coordinator.GetComponent<Zone>(s).location != Zone::STACK;
        },
        "resolution end");
}
