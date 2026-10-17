#include "visibility.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <tuple>

#include "../card_vocab.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_driver.h"
#include "../state_format/events.h"
#include "../state_format/handles.h"
#include "characteristics.h"
#include "lki.h"
#include "players.h"
#include "zones.h"

using Window = Game::RevealWindow;

static bool is_seat(Zone::Ownership p);
static Entity window_card(const Window &w);
static bool window_shown_to(const Window &w, Zone::Ownership viewer);
static bool in_hidden_zone(Entity card);
static bool sees_without_window(Entity card, Zone::Ownership viewer);
static bool outlives(Window::End a, Window::End b);
static bool open_window(Entity card, Zone::Ownership audience, Window::Kind kind, Window::End until,
    Entity stack_object, bool position_public);
static void open_windows(const std::vector<Entity> &cards, Zone::Ownership audience, Window::Kind kind,
    Window::End until, Entity stack_object, bool position_public, bool draw_link);
static std::vector<Entity> canonical_order(std::vector<Entity> cards);
static bool canonical_before(Entity a, Entity b);
template <class Pred>
static void close_windows_if(Pred pred, const char *reason, bool report);
static void trace_window(const char *verb, const Window &w, Entity card, const char *detail);
static const char *kind_word(Window::Kind kind);
static const char *end_word(Window::End until);
struct CanonicalKey;
static CanonicalKey canonical_key(Entity e, Zone::Ownership viewer);
static int zone_rank(Zone::ZoneValue zone);
static int seat_rank(Zone::Ownership seat, Zone::Ownership viewer);
static bool arrival_visible(const Zone &z, Zone::Ownership viewer);
static int physical_card_id(Entity e);
static int token_card_id(Entity e);
static int last_known_card_id(Entity e);
static bool is_stack_ability(Entity e);

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

// Opens one window; true when it is new (not already open, possibly extended, nor needed).
static bool open_window(Entity card, Zone::Ownership audience, Window::Kind kind, Window::End until,
    Entity stack_object, bool position_public) {
    if (card == 0 || !in_hidden_zone(card)) return false;
    if (is_seat(audience) && sees_without_window(card, audience)) return false;
    for (Window &w : cur_game.reveal_windows) {
        if (window_card(w) != card || w.audience != audience) continue;
        if (outlives(until, w.until)) {
            w.kind = kind;
            w.until = until;
            w.stack_object = ObjectRef::of(stack_object);
            trace_window("extend", w, card, end_word(until));
        }
        return false;
    }
    Window w;
    w.card = ObjectRef::of(card);
    w.audience = audience;
    w.kind = kind;
    w.until = until;
    w.stack_object = ObjectRef::of(stack_object);
    w.position_public = position_public;
    cur_game.reveal_windows.push_back(w);
    trace_window("open", w, card, end_word(until));
    return true;
}

// Opens a window on each of `cards` in turn and reports the new ones as one event per zone
// (`draw_link`: the reveal shows its card as the one just drawn).
static void open_windows(const std::vector<Entity> &cards, Zone::Ownership audience, Window::Kind kind,
    Window::End until, Entity stack_object, bool position_public, bool draw_link) {
    std::vector<Entity> opened;
    for (Entity card : cards)
        if (open_window(card, audience, kind, until, stack_object, position_public)) opened.push_back(card);
    obs_windows_opened(opened, audience, kind == Window::LOOK, position_public, draw_link);
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
// whose card is no longer that object is dropped on the way. With `report`, the windows that end
// are reported (a reveal_end per zone and audience, state_format/events.h) while still open.
// A viewer whose handle for a closed window's card depended on that window loses the handle here,
// as the window closes (state_format/handles.h).
template <class Pred>
static void close_windows_if(Pred pred, const char *reason, bool report) {
    std::vector<Window> &windows = cur_game.reveal_windows;
    std::vector<bool> closing(windows.size(), false);
    std::vector<ObsClosedWindow> reported;
    for (size_t i = 0; i < windows.size(); ++i) {
        const Entity card = window_card(windows[i]);
        if (card == 0 || !pred(windows[i], card)) continue;
        closing[i] = true;
        if (report) reported.push_back(ObsClosedWindow{card, windows[i].audience});
    }
    obs_windows_closed(reported);
    std::vector<Entity> closed;
    size_t kept = 0;
    for (size_t i = 0; i < windows.size(); ++i) {
        const Window &w = windows[i];
        const Entity card = window_card(w);
        if (card == 0) {
            trace_window("close", w, 0, "gone");
            continue;
        }
        if (closing[i]) {
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

LibrarySpan library_position_seen_by(Entity card, Zone::Ownership viewer) {
    const Zone &z = global_coordinator.GetComponent<Zone>(card);
    LibrarySpan exact;
    exact.pos = z.distance_from_top;
    if (!is_seat(viewer) || identity_visible(card, viewer)) return exact;
    bool hidden_window = false;
    for (const Window &w : cur_game.reveal_windows) {
        if (window_card(w) != card || window_shown_to(w, viewer)) continue;
        if (!w.position_public) return LibrarySpan{0, 0};
        hidden_window = true;
    }
    if (!hidden_window) return exact;
    // The cards in windows hidden from the viewer in this library, that the viewer cannot tell
    // apart from this one.
    size_t lo = z.distance_from_top, hi = z.distance_from_top;
    for (const Window &w : cur_game.reveal_windows) {
        const Entity other = window_card(w);
        if (other == 0 || window_shown_to(w, viewer) || !w.position_public) continue;
        const Zone &oz = global_coordinator.GetComponent<Zone>(other);
        if (oz.location != Zone::LIBRARY || oz.owner != z.owner || identity_visible(other, viewer)) continue;
        lo = std::min(lo, oz.distance_from_top);
        hi = std::max(hi, oz.distance_from_top);
    }
    LibrarySpan s;
    s.pos = lo;
    s.span = hi - lo + 1;
    return s;
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
    open_windows({card}, Zone::UNKNOWN, Window::REVEAL, Window::RESOLUTION, 0, true, false);
}

void reveal_cards(const std::vector<Entity> &cards) {
    open_windows(canonical_order(cards), Zone::UNKNOWN, Window::REVEAL, Window::RESOLUTION, 0, true, false);
}

void look_at_card(Entity card, Zone::Ownership looker) {
    if (is_seat(looker)) open_windows({card}, looker, Window::LOOK, Window::RESOLUTION, 0, true, false);
}

void look_at_cards(const std::vector<Entity> &cards, Zone::Ownership looker) {
    if (is_seat(looker)) open_windows(canonical_order(cards), looker, Window::LOOK, Window::RESOLUTION, 0, true, false);
}

void look_at_found_card(Entity card, Zone::Ownership looker) {
    if (is_seat(looker)) open_windows({card}, looker, Window::LOOK, Window::RESOLUTION, 0, false, false);
}

void reveal_until_leaves_stack(Entity card, Entity stack_object) {
    open_windows({card}, Zone::UNKNOWN, Window::REVEAL, Window::STACK_OBJECT, stack_object, true, true);
}

void reveal_continuously(Entity card) {
    open_windows({card}, Zone::UNKNOWN, Window::CONTINUOUS_REVEAL, Window::LEAVES_ZONE, 0, true, false);
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
    close_windows_if([card](const Window &, Entity e) { return e == card; }, "zone change", false);
}

// Reported by the face-down exile's own zone_move (state_format/events.h), not as a look.
void open_face_down_windows(Entity card, HiddenViewers saw) {
    if (saw.a) open_window(card, Zone::PLAYER_A, Window::FACE_DOWN_LOOK, Window::LEAVES_ZONE, 0, true);
    if (saw.b) open_window(card, Zone::PLAYER_B, Window::FACE_DOWN_LOOK, Window::LEAVES_ZONE, 0, true);
}

void close_windows_on_face_up(Entity card) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if([card](const Window &, Entity e) { return e == card; }, "turned face up", false);
}

void close_windows_on_shuffle(Zone::Ownership owner) {
    if (cur_game.reveal_windows.empty()) return;
    close_windows_if(
        [owner](const Window &, Entity e) {
            const Zone &z = global_coordinator.GetComponent<Zone>(e);
            return z.location == Zone::LIBRARY && z.owner == owner;
        },
        "shuffle", true);
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
        "resolution end", true);
}

// A token's token-band index, keyed by its script stem (display names collide, stems don't).
static int token_card_id(Entity e) {
    if (!global_coordinator.entity_has_component<Token>(e)) return TOKEN_SENTINEL;
    const int idx = token_script_to_index(global_coordinator.GetComponent<Token>(e).script_name);
    return idx >= 0 ? idx : TOKEN_SENTINEL;
}

// The identity an object that left the battlefield took with it (CR 608.2h): its name's vocab
// index, or a token's token-band index; -1 with no last-known information.
static int last_known_card_id(Entity e) {
    const LastKnownInfo *lki = departed_lki_for(e);
    if (!lki || lki->name.empty()) return -1;
    if (!lki->is_token) return card_name_to_index(lki->name);
    const int idx = token_script_to_index(lki->token_script);
    return idx >= 0 ? idx : TOKEN_SENTINEL;
}

// An activated or triggered ability's own stack object (it has no card of its own).
static bool is_stack_ability(Entity e) {
    return global_coordinator.entity_has_component<Ability>(e) &&
           !global_coordinator.entity_has_component<CardData>(e) &&
           !global_coordinator.entity_has_component<Permanent>(e) && !global_coordinator.entity_has_component<Token>(e);
}

static int physical_card_id(Entity e) {
    if (e == 0) return -1;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token) return token_card_id(e);
        if (perm.printed_card) return card_name_to_index(perm.printed_card->name);
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    if (global_coordinator.entity_has_component<Token>(e)) return token_card_id(e);
    return last_known_card_id(e);
}

int card_identity(Entity entity, Zone::Ownership /*viewer*/) {
    return physical_card_id(entity);
}

int visible_card_id(Entity entity, Zone::Ownership viewer) {
    if (entity == 0 || global_coordinator.entity_has_component<Player>(entity)) return -1;
    if (is_stack_ability(entity)) {
        const Ability &ab = global_coordinator.GetComponent<Ability>(entity);
        const Entity live = ab.source.get();
        if (live != 0 && live != entity) return visible_card_id(live, viewer);
        const int lki = last_known_card_id(ab.source.lki_entity());
        if (lki >= 0) return lki;
        // A delayed or floating trigger with no source object: the card that created it, as
        // captured when it did (public: the creating ability was on the stack or battlefield).
        if (ab.delayed_link.seq != 0 && ab.delayed_link.creator_vocab_idx >= 0)
            return ab.delayed_link.creator_vocab_idx;
        return ab.floating_creator_vocab_idx;
    }
    if (!global_coordinator.entity_has_component<Zone>(entity)) {
        // A card with no zone is a sideboarding menu's template card: the sideboarding player's own
        // (their post-board configuration is hidden from the opponent). A departed object has
        // only its last-known identity.
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            const bool own = !is_seat(viewer) || (sideboard_phase && viewer == sideboard_phase_player);
            return own ? card_identity(entity, viewer) : -1;
        }
        return last_known_card_id(entity);
    }
    if (!identity_visible(entity, viewer)) return -1;
    return card_identity(entity, viewer);
}

int public_card_id(Entity entity) {
    return physical_card_id(entity);
}
