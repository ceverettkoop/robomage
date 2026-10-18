#include "events.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <tuple>

#include "../card_vocab.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/spell.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_driver.h"
#include "../queries/characteristics.h"
#include "../queries/players.h"
#include "../queries/visibility.h"
#include "../queries/zones.h"
#include "agent_feed.h"
#include "enum_names.h"
#include "handles.h"
#include "selfcheck.h"

using rmstate::EventKind;
using rmstate::LibAnchor;

// The viewers in the order every per-viewer loop (and the trace) visits them.
static const Zone::Ownership VIEWERS[] = {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN};

static bool is_seat(Zone::Ownership p);
static int slot(Zone::Ownership viewer);
static bool queues_enabled();
static bool reporting();
static bool is_card_object(Entity e);
static bool is_token_object(Entity e);
static ObsLibPos at_top(size_t pos, size_t span);
static ObsLibPos at_bottom(size_t pos, size_t span);
static ObsLibPos anywhere();
static ObsLibPos position_seen(Entity card, Zone::Ownership viewer);
static uint16_t clamp16(size_t v);
static ObsEvent new_event(EventKind kind, Zone::Ownership viewer);
static void fill_source(ObsEvent &ev, Zone::Ownership viewer);
static uint32_t push(Zone::Ownership viewer, ObsEvent ev);
template <class Build>
static uint32_t emit(Zone::Ownership viewer, Build build);
static ObsMoveCapture::Seen see_before_move(Entity e, Zone::Ownership viewer);
static void note_stack_departure(Entity e);
static bool same_seen(const ObsMoveCapture::Seen &a, const ObsMoveCapture::Seen &b);
static ObsEventCard object_card(Entity e, Zone::Ownership viewer, bool issue);
static void zone_counts(uint16_t library[3], uint16_t sideboard[3]);
static int game_index();
static void push_game_boundary(EventKind kind, int winner, int wins_a, int wins_b);
template <class Key, class Item>
static std::vector<std::vector<Item>> group_by(const std::vector<Item> &items, Key key);
static void trace_event(Zone::Ownership viewer, const ObsEvent &ev);
static std::string kind_word(EventKind kind);
static std::string zone_text(int zone, Zone::Ownership owner);
static std::string libpos_text(const ObsLibPos &p);
static std::string card_text(int card);
static const char *viewer_word(Zone::Ownership viewer);

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

static int slot(Zone::Ownership viewer) {
    return HandleMove::slot(viewer);
}

// Whether a frame writer (or the self-check) drains the queues; under --trace-events alone events
// are only narrated.
static bool queues_enabled() {
    return !rmstate_dump_path.empty() || machine_format_rmstate || selfcheck_state_enabled() || agent_feed_enabled();
}

bool rmstate_events_enabled() {
    return queues_enabled() || trace_events;
}

// Whether in-game events are reported now: someone consumes them and the game has started (the
// library build and its shuffle come before game_start and are not reported).
static bool reporting() {
    return rmstate_events_enabled() && cur_game.obs_events.game_started;
}

// A card, a token or a spell copy: the objects events report. An ability's stack object (no card
// of its own) is not one.
static bool is_card_object(Entity e) {
    return e != 0 && global_coordinator.entity_has_component<Zone>(e) &&
           (global_coordinator.entity_has_component<CardData>(e) || global_coordinator.entity_has_component<Token>(e));
}

static bool is_token_object(Entity e) {
    if (global_coordinator.entity_has_component<Token>(e)) return true;
    return global_coordinator.entity_has_component<Spell>(e) && global_coordinator.GetComponent<Spell>(e).is_copy;
}

static ObsLibPos at_top(size_t pos, size_t span) {
    ObsLibPos p;
    p.anchor = static_cast<uint8_t>(LibAnchor::TOP);
    p.pos = clamp16(pos);
    p.span = clamp16(span);
    return p;
}

static ObsLibPos at_bottom(size_t pos, size_t span) {
    ObsLibPos p = at_top(pos, span);
    p.anchor = static_cast<uint8_t>(LibAnchor::BOTTOM);
    return p;
}

// In a library, position unknown to the viewer.
static ObsLibPos anywhere() {
    return at_top(0, 0);
}

// Where `viewer` knows the library card `card` sits (library_position_seen_by).
static ObsLibPos position_seen(Entity card, Zone::Ownership viewer) {
    const LibrarySpan s = library_position_seen_by(card, viewer);
    return s.span == 0 ? anywhere() : at_top(s.pos, s.span);
}

static uint16_t clamp16(size_t v) {
    return static_cast<uint16_t>(std::min<size_t>(v, 0xFFFE));
}

static ObsEvent new_event(EventKind kind, Zone::Ownership viewer) {
    ObsEvent ev;
    ev.kind = kind;
    fill_source(ev, viewer);
    return ev;
}

// The source of the effect behind the event: the object resolving right now (a spell, or the
// source of a resolving ability), as `viewer` sees it; none outside a resolution. A source that has
// since become a new object (it left its zone, even within this resolution) has no handle, only
// its card.
static void fill_source(ObsEvent &ev, Zone::Ownership viewer) {
    if (!cur_game.resolution.active) return;
    const Entity s = cur_game.resolution.stack_entity;
    if (s == 0 || !global_coordinator.entity_has_component<Zone>(s)) return;
    ev.source_card = visible_card_id(s, viewer);
    Entity obj = s;
    if (!global_coordinator.entity_has_component<CardData>(s) && global_coordinator.entity_has_component<Ability>(s)) {
        const ObjectRef &src = global_coordinator.GetComponent<Ability>(s).source;
        obj = src.lki_entity();
        if (obj != 0 && (!global_coordinator.entity_has_component<Zone>(obj) ||
                            global_coordinator.GetComponent<Zone>(obj).obj_gen != src.gen))
            obj = 0;
    }
    if (obj != 0 && individually_identifiable(obj, viewer)) ev.source = handle_of(viewer, obj);
}

// Gives `ev` the viewer's next seq and the turn, narrates it under --trace-events and queues it
// when a frame writer drains the queues. Returns its seq.
static uint32_t push(Zone::Ownership viewer, ObsEvent ev) {
    ObsEventQueue &q = cur_game.obs_events.queues[slot(viewer)];
    ev.seq = q.next_seq++;
    ev.turn = clamp16(cur_game.turn_state.turn);
    if (trace_events) trace_event(viewer, ev);
    const uint32_t seq = ev.seq;
    // The stdout stream serves the seats only.
    if (machine_format_rmstate && is_seat(viewer)) q.stream_events.push_back(ev);
    if (selfcheck_state_enabled()) q.selfcheck_events.push_back(ev);
    if (agent_feed_serves(viewer)) q.agent_events.push_back(ev);
    if (!rmstate_dump_path.empty()) q.events.push_back(std::move(ev));
    return seq;
}

// Builds `viewer`'s event with `build` (which reads the game and may issue the viewer's handles,
// nothing else) and queues it; returns its seq. Under --selfcheck-state a seat's event is built a
// second time with every identity hidden from the seat substituted (ScrambledIdentities), and the
// two builds must project to the same rows. The first build has issued every handle the event
// needs, so the second issues none.
template <class Build>
static uint32_t emit(Zone::Ownership viewer, Build build) {
    ObsEvent ev = build();
    if (is_seat(viewer) && selfcheck_state_enabled()) {
        ObsEvent scrambled;
        {
            ScrambledIdentities scramble(viewer);
            scrambled = build();
        }
        selfcheck_state_event(viewer, ev, scrambled);
    }
    return push(viewer, std::move(ev));
}

// `e` where it is now, as `viewer` sees it: its handle (in both columns: the event does not move
// it; issued now if the viewer holds none and `issue`), its card id when visible, and its library
// position.
static ObsEventCard object_card(Entity e, Zone::Ownership viewer, bool issue) {
    ObsEventCard c;
    if (individually_identifiable(e, viewer))
        c.handle_before = c.handle_after = issue ? handle_of(viewer, e) : handle_peek(viewer, e);
    if (identity_visible(e, viewer)) c.card = visible_card_id(e, viewer);
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    if (z.location == Zone::LIBRARY) c.from = c.to = position_seen(e, viewer);
    c.face_down = z.is_face_down;
    c.token = is_token_object(e);
    return c;
}

// ── Zone moves ────────────────────────────────────────────────────────────────────────────────

ObsMoveCapture obs_begin_move(Entity e) {
    ObsMoveCapture cap;
    if (!reporting() || !is_card_object(e)) return cap;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    cap.active = true;
    cap.token = is_token_object(e);
    // A token is built in its controller's hand zone and moved onto the battlefield as it is
    // created: it never entered a zone before (obj_gen 0), so it comes from zone none.
    cap.created = z.location == Zone::HAND && z.obj_gen == 0;
    if (cap.created) return cap;
    cap.origin = z.location;
    cap.origin_owner = z.owner;
    cap.origin_hidden = !is_public_zone(z.location) || z.is_face_down;
    if (z.location == Zone::STACK) note_stack_departure(e);
    if (z.location == Zone::HAND) cap.hand_subset = cur_game.obs_events.hand_subset;
    for (Zone::Ownership viewer : VIEWERS) {
        ObsMoveCapture::Seen &s = cap.before[slot(viewer)];
        s = see_before_move(e, viewer);
        if (!is_seat(viewer) || !selfcheck_state_enabled()) continue;
        ObsMoveCapture::Seen scrambled;
        {
            ScrambledIdentities scramble(viewer);
            scrambled = see_before_move(e, viewer);
        }
        if (!same_seen(s, scrambled))
            selfcheck_state_mismatch("scramble", viewer, "a zone move's capture of the moving card differs");
    }
    return cap;
}

// What `viewer` sees of `e` as it starts to move. Only a handle the viewer holds: one issued now
// would name nothing they saw before.
static ObsMoveCapture::Seen see_before_move(Entity e, Zone::Ownership viewer) {
    ObsMoveCapture::Seen s;
    s.handle = handle_peek(viewer, e);
    s.visible = identity_visible(e, viewer);
    if (s.visible) s.card = visible_card_id(e, viewer);
    if (global_coordinator.GetComponent<Zone>(e).location == Zone::LIBRARY) s.from = position_seen(e, viewer);
    return s;
}

// Records `e`, leaving the stack, while another stack object can refer to it; when none is left
// the departures recorded so far are dropped too.
static void note_stack_departure(Entity e) {
    std::vector<ObsStackDeparture> &departed = cur_game.obs_events.stack_departed;
    bool others = false;
    for (Entity o : zoned_entities())
        if (o != e && global_coordinator.GetComponent<Zone>(o).location == Zone::STACK) {
            others = true;
            break;
        }
    if (!others) {
        departed.clear();
        return;
    }
    ObsStackDeparture d;
    d.ref = ObjectRef::of(e);
    for (Zone::Ownership viewer : VIEWERS) {
        ObsStackDeparture::Seen &s = d.seen[slot(viewer)];
        s.visible = identity_visible(e, viewer);
        if (s.visible) s.card = visible_card_id(e, viewer);
    }
    d.token = is_token_object(e);
    d.controller = source_controller(e);
    const Token *tok =
        global_coordinator.entity_has_component<Token>(e) ? &global_coordinator.GetComponent<Token>(e) : nullptr;
    const CardData *cd =
        global_coordinator.entity_has_component<CardData>(e) ? &global_coordinator.GetComponent<CardData>(e) : nullptr;
    const CardData *face = cd ? &active_face(e, *cd) : nullptr;
    d.power = face ? static_cast<int>(face->power) : tok ? static_cast<int>(tok->power) : 0;
    d.toughness = face ? static_cast<int>(face->toughness) : tok ? static_cast<int>(tok->toughness) : 0;
    const std::set<Type> empty_types;
    d.types = card_type_mask(face ? face->types : tok ? tok->types : empty_types);
    uint32_t colors = 0;
    for (Colors col : effective_colors(e))
        if (col >= WHITE && col <= NO_COLOR) colors |= 1u << static_cast<int>(col);
    d.colors = static_cast<uint8_t>(colors);
    departed.push_back(d);
}

const ObsStackDeparture *obs_stack_departure(const ObjectRef &ref, Zone::Ownership viewer, int &card) {
    for (const ObsStackDeparture &d : cur_game.obs_events.stack_departed) {
        if (d.ref != ref) continue;
        const ObsStackDeparture::Seen &s = d.seen[slot(viewer)];
        if (!s.visible) return nullptr;
        card = s.card;
        return &d;
    }
    return nullptr;
}

static bool same_seen(const ObsMoveCapture::Seen &a, const ObsMoveCapture::Seen &b) {
    return a.visible == b.visible && a.handle == b.handle && a.card == b.card && a.from.anchor == b.from.anchor &&
           a.from.pos == b.from.pos && a.from.span == b.from.span;
}

void obs_end_move(const ObsMoveCapture &cap, Entity e, const uint32_t after_handles[3], const ObsMoveInfo &info) {
    if (!cap.active) return;
    ObsEventState &st = cur_game.obs_events;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    const bool into_library = z.location == Zone::LIBRARY && !info.draw;
    const bool in_batch = into_library && st.batch.active && z.owner == st.batch.library_owner;
    if (in_batch) {
        if (st.batch.cards.empty()) st.batch.on_bottom = info.on_bottom;
        st.batch.cards.push_back(ObjectRef::of(e));
    }
    EventKind kind = EventKind::ZONE_MOVE;
    if (info.draw)
        kind = EventKind::DRAW;
    else if (cap.origin_hidden && z.location == Zone::STACK)
        kind = EventKind::CAST_FROM_HIDDEN;
    else if (cap.origin_hidden && z.location == Zone::BATTLEFIELD)
        kind = EventKind::PLAY_FROM_HIDDEN;
    for (Zone::Ownership viewer : VIEWERS) {
        const int s = slot(viewer);
        const ObsMoveCapture::Seen &seen = cap.before[s];
        const uint32_t seq = emit(viewer, [&] {
            ObsEvent ev = new_event(kind, viewer);
            ev.zone = cap.origin;
            ev.zone_owner = cap.origin_owner;
            ev.to_zone = z.location;
            ev.to_owner = z.owner;
            ev.count = 1;
            ObsEventCard c;
            c.handle_before = seen.handle;
            c.handle_after = after_handles[s];
            if (c.handle_after == 0 && individually_identifiable(e, viewer)) c.handle_after = handle_of(viewer, e);
            c.card = identity_visible(e, viewer) ? visible_card_id(e, viewer) : seen.card;
            c.from = seen.from;
            c.face_down = z.location == Zone::EXILE && z.is_face_down;
            c.token = cap.token;
            c.hand_subset = cap.hand_subset;
            if (info.draw) {
                ev.player = z.owner;
                ev.opening = info.opening;
                c.from = at_top(0, 1);
            } else if (into_library) {
                if (in_batch && st.batch.batched[s])
                    ev.batch = st.batch.id;
                else
                    c.to = info.on_bottom ? at_bottom(0, 1) : at_top(z.distance_from_top, 1);
            }
            ev.cards.push_back(c);
            return ev;
        });
        if (info.draw) st.last_draw_seq[s] = seq;
    }
}

void obs_object_created(Entity e) {
    if (!reporting() || !is_card_object(e)) return;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::ZONE_MOVE, viewer);
            ev.to_zone = z.location;
            ev.to_owner = z.owner;
            ev.count = 1;
            ObsEventCard c = object_card(e, viewer, true);
            c.handle_before = 0;
            c.from = ObsLibPos{};
            ev.cards.push_back(c);
            return ev;
        });
    }
}

void obs_object_ceased(Entity e, bool spell_copy) {
    if (!reporting() || !is_card_object(e)) return;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::ZONE_MOVE, viewer);
            ev.zone = z.location;
            ev.zone_owner = z.owner;
            ev.count = 1;
            ObsEventCard c = object_card(e, viewer, false);
            c.handle_after = 0;
            c.to = ObsLibPos{};
            c.token = c.token || spell_copy;
            ev.cards.push_back(c);
            return ev;
        });
    }
    retire_object_handles(e);
}

void obs_shuffle(Zone::Ownership owner) {
    if (!reporting()) return;
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::SHUFFLE, viewer);
            ev.player = owner;
            ev.zone = Zone::LIBRARY;
            ev.zone_owner = owner;
            return ev;
        });
    }
}

void obs_face_turned(Entity e, bool face_up) {
    if (!reporting() || !is_card_object(e)) return;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(face_up ? EventKind::FACE_UP : EventKind::FACE_DOWN, viewer);
            ev.zone = z.location;
            ev.zone_owner = z.owner;
            ev.count = 1;
            ev.cards.push_back(object_card(e, viewer, true));
            return ev;
        });
    }
}

void obs_search(Zone::Ownership searcher, Zone::ZoneValue zone, Zone::Ownership owner, int found) {
    if (!reporting()) return;
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::SEARCH, viewer);
            ev.player = searcher;
            ev.zone = zone;
            ev.zone_owner = owner;
            ev.count = clamp16(static_cast<size_t>(std::max(found, 0)));
            return ev;
        });
    }
}

void obs_note_miracle_draw(Entity card) {
    if (!reporting()) return;
    ObsEventState &st = cur_game.obs_events;
    st.miracle_card = ObjectRef::of(card);
    for (int s = 0; s < 3; ++s) st.miracle_draw_seq[s] = st.last_draw_seq[s];
}

// ── Reveal windows ────────────────────────────────────────────────────────────────────────────

// `items` split into groups of equal key, each group in first-appearance order and the items of a
// group in their order in `items`.
template <class Key, class Item>
static std::vector<std::vector<Item>> group_by(const std::vector<Item> &items, Key key) {
    std::vector<std::vector<Item>> groups;
    std::vector<decltype(key(items.front()))> keys;
    for (const Item &it : items) {
        const auto k = key(it);
        auto at = std::find(keys.begin(), keys.end(), k);
        if (at == keys.end()) {
            keys.push_back(k);
            groups.emplace_back();
            at = keys.end() - 1;
        }
        groups[static_cast<size_t>(at - keys.begin())].push_back(it);
    }
    return groups;
}

void obs_windows_opened(
    const std::vector<Entity> &cards, Zone::Ownership audience, bool look, bool position_public, bool draw_link) {
    if (!reporting() || cards.empty()) return;
    const ObsEventState &st = cur_game.obs_events;
    const auto zone_key = [](Entity c) {
        const Zone &z = global_coordinator.GetComponent<Zone>(c);
        return std::make_pair(static_cast<int>(z.location), static_cast<int>(z.owner));
    };
    for (const std::vector<Entity> &group : group_by(cards, zone_key)) {
        const Zone &gz = global_coordinator.GetComponent<Zone>(group.front());
        for (Zone::Ownership viewer : VIEWERS) {
            emit(viewer, [&] {
                ObsEvent ev = new_event(look ? EventKind::LOOK : EventKind::REVEAL_START, viewer);
                if (look) ev.player = audience;
                ev.to = audience;
                ev.zone = gz.location;
                ev.zone_owner = gz.owner;
                ev.count = clamp16(group.size());
                bool exact = true;
                size_t lo = SIZE_MAX, hi = 0;
                for (Entity card : group) {
                    ObsEventCard c = object_card(card, viewer, true);
                    c.handle_before = 0;
                    c.face_down = false;
                    if (gz.location == Zone::LIBRARY) {
                        const size_t pos = global_coordinator.GetComponent<Zone>(card).distance_from_top;
                        const bool knows = !is_seat(viewer) || position_public || identity_visible(card, viewer);
                        c.from = knows ? at_top(pos, 1) : anywhere();
                        c.to = ObsLibPos{};
                        exact = exact && knows;
                        lo = std::min(lo, pos);
                        hi = std::max(hi, pos);
                    }
                    if (draw_link && st.miracle_card == ObjectRef::of(card)) {
                        c.draw_seq = st.miracle_draw_seq[slot(viewer)];
                        c.draw_idx = 0;
                    }
                    ev.cards.push_back(c);
                }
                if (gz.location == Zone::LIBRARY)
                    ev.range = exact && hi - lo + 1 == group.size() ? at_top(lo, group.size()) : anywhere();
                return ev;
            });
        }
    }
}

void obs_windows_closed(const std::vector<ObsClosedWindow> &windows) {
    if (!reporting() || windows.empty()) return;
    const auto key = [](const ObsClosedWindow &w) {
        const Zone &z = global_coordinator.GetComponent<Zone>(w.card);
        return std::make_tuple(static_cast<int>(z.location), static_cast<int>(z.owner), static_cast<int>(w.audience));
    };
    for (const std::vector<ObsClosedWindow> &group : group_by(windows, key)) {
        const Zone &gz = global_coordinator.GetComponent<Zone>(group.front().card);
        const Zone::Ownership audience = group.front().audience;
        for (Zone::Ownership viewer : VIEWERS) {
            if (is_seat(viewer) && is_seat(audience) && audience != viewer) continue;
            emit(viewer, [&] {
                ObsEvent ev = new_event(EventKind::REVEAL_END, viewer);
                if (is_seat(audience)) ev.player = audience;
                ev.to = audience;
                ev.zone = gz.location;
                ev.zone_owner = gz.owner;
                ev.count = clamp16(group.size());
                for (const ObsClosedWindow &w : group) {
                    ObsEventCard c = object_card(w.card, viewer, false);
                    c.handle_after = 0;
                    c.to = ObsLibPos{};
                    c.face_down = false;
                    ev.cards.push_back(c);
                }
                return ev;
            });
        }
    }
}

// ── Instruction context ───────────────────────────────────────────────────────────────────────

ObsLibraryBatch::ObsLibraryBatch(Zone::Ownership library_owner, Zone::Ownership chooser) {
    ObsBatch &b = cur_game.obs_events.batch;
    if (!reporting() || b.active) return;
    opened_ = true;
    b = ObsBatch{};
    b.active = true;
    b.id = cur_game.obs_events.next_batch++;
    b.library_owner = library_owner;
    b.batched[slot(Zone::PLAYER_A)] = chooser != Zone::PLAYER_A;
    b.batched[slot(Zone::PLAYER_B)] = chooser != Zone::PLAYER_B;
}

ObsLibraryBatch::~ObsLibraryBatch() {
    close();
}

void ObsLibraryBatch::close() {
    if (!opened_) return;
    opened_ = false;
    ObsBatch b = cur_game.obs_events.batch;
    cur_game.obs_events.batch = ObsBatch{};
    // Where the batch's cards are now (a replacement may have sent one elsewhere): the
    // consecutive positions at the end they were put on, counted as the cards between each and
    // that end.
    std::vector<size_t> depths;
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && z.owner == b.library_owner) depths.push_back(z.distance_from_top);
    }
    size_t placed = 0, lo = SIZE_MAX;
    for (const ObjectRef &r : b.cards) {
        const Entity e = r.get();
        if (e == 0) continue;
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::LIBRARY || z.owner != b.library_owner) continue;
        ++placed;
        const size_t d = z.distance_from_top;
        const size_t from_end = static_cast<size_t>(
            std::count_if(depths.begin(), depths.end(), [d, &b](size_t o) { return b.on_bottom ? o > d : o < d; }));
        lo = std::min(lo, from_end);
    }
    if (placed == 0) return;
    for (Zone::Ownership viewer : VIEWERS) {
        if (!b.batched[slot(viewer)]) continue;
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::LIBRARY_PLACE, viewer);
            ev.player = b.library_owner;
            ev.zone = Zone::LIBRARY;
            ev.zone_owner = b.library_owner;
            ev.count = clamp16(placed);
            ev.batch = b.id;
            ev.range = b.on_bottom ? at_bottom(lo, placed) : at_top(lo, placed);
            ev.order_known = false;
            return ev;
        });
    }
}

ObsHandSubsetScope::ObsHandSubsetScope(rmstate::HandSubset subset) : prev_(cur_game.obs_events.hand_subset) {
    if (rmstate_events_enabled()) cur_game.obs_events.hand_subset = static_cast<uint8_t>(subset);
}

ObsHandSubsetScope::~ObsHandSubsetScope() {
    if (rmstate_events_enabled()) cur_game.obs_events.hand_subset = prev_;
}

// ── Game boundaries ───────────────────────────────────────────────────────────────────────────

ObsEventState obs_events_for_next_game() {
    ObsEventState next;
    if (match_game_number <= 0) return next;
    for (int s = 0; s < 3; ++s) next.queues[s] = std::move(cur_game.obs_events.queues[s]);
    next.next_batch = cur_game.obs_events.next_batch;
    return next;
}

// Library and sideboard sizes per seat (indexed by Zone::Ownership).
static void zone_counts(uint16_t library[3], uint16_t sideboard[3]) {
    size_t lib[3] = {0, 0, 0}, sb[3] = {0, 0, 0};
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (!is_seat(z.owner)) continue;
        if (z.location == Zone::LIBRARY) ++lib[z.owner];
        if (z.location == Zone::SIDEBOARD) ++sb[z.owner];
    }
    for (int i = 0; i < 3; ++i) {
        library[i] = clamp16(lib[i]);
        sideboard[i] = clamp16(sb[i]);
    }
}

// The 0-based game of the match (a single game is game 0).
static int game_index() {
    return match_game_number < 0 ? 0 : match_game_number;
}

void obs_game_start(bool player_a_goes_first) {
    if (!rmstate_events_enabled()) return;
    cur_game.obs_events.game_started = true;
    uint16_t library[3], sideboard[3];
    zone_counts(library, sideboard);
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(EventKind::GAME_START, viewer);
            ev.player = player_a_goes_first ? Zone::PLAYER_A : Zone::PLAYER_B;
            ev.game = static_cast<uint8_t>(game_index());
            for (Zone::Ownership p : {Zone::PLAYER_A, Zone::PLAYER_B}) {
                ObsEventPlayer row;
                row.player = p;
                row.library = library[p];
                row.sideboard = sideboard[p];
                ev.players.push_back(row);
            }
            return ev;
        });
    }
}

// A game_end or match_end: the game, the winner (none for a draw) and the match wins.
static void push_game_boundary(EventKind kind, int winner, int wins_a, int wins_b) {
    for (Zone::Ownership viewer : VIEWERS) {
        emit(viewer, [&] {
            ObsEvent ev = new_event(kind, viewer);
            ev.player = winner == Zone::PLAYER_A   ? Zone::PLAYER_A
                        : winner == Zone::PLAYER_B ? Zone::PLAYER_B
                                                   : Zone::UNKNOWN;
            ev.game = static_cast<uint8_t>(game_index());
            for (Zone::Ownership p : {Zone::PLAYER_A, Zone::PLAYER_B}) {
                ObsEventPlayer row;
                row.player = p;
                row.wins = static_cast<uint8_t>(std::max(0, p == Zone::PLAYER_A ? wins_a : wins_b));
                ev.players.push_back(row);
            }
            return ev;
        });
    }
}

void obs_game_end(int winner) {
    if (!rmstate_events_enabled() || !cur_game.obs_events.game_started) return;
    cur_game.obs_events.game_started = false;
    const bool bo3 = match_game_number >= 0;
    const int wins_a = (bo3 ? match_wins_a : 0) + (winner == Zone::PLAYER_A ? 1 : 0);
    const int wins_b = (bo3 ? match_wins_b : 0) + (winner == Zone::PLAYER_B ? 1 : 0);
    push_game_boundary(EventKind::GAME_END, winner, wins_a, wins_b);
}

void obs_match_end(int winner, int wins_a, int wins_b) {
    if (!rmstate_events_enabled()) return;
    push_game_boundary(EventKind::MATCH_END, winner, wins_a, wins_b);
}

std::vector<ObsEvent> take_obs_events(Zone::Ownership viewer, ObsConsumer consumer) {
    ObsEventQueue &q = cur_game.obs_events.queues[slot(viewer)];
    std::vector<ObsEvent> out;
    switch (consumer) {
        case ObsConsumer::DUMP:
            out.swap(q.events);
            break;
        case ObsConsumer::STREAM:
            out.swap(q.stream_events);
            break;
        case ObsConsumer::SELFCHECK:
            out.swap(q.selfcheck_events);
            break;
        case ObsConsumer::AGENT:
            out.swap(q.agent_events);
            break;
    }
    return out;
}

// ── --trace-events ────────────────────────────────────────────────────────────────────────────

// One EVENT: line per event and viewer (game log): "EVENT: <viewer> seq=<n> t<turn> <kind>", then
// the scalar fields that are set, then each card as [handle before>handle after, name or ?, from
// and to positions, flags] and each player as {seat library/sideboard/wins}.
static void trace_event(Zone::Ownership viewer, const ObsEvent &ev) {
    std::string line = std::string("EVENT: ") + viewer_word(viewer) + " seq=" + std::to_string(ev.seq) + " t" +
                       std::to_string(ev.turn) + " " + kind_word(ev.kind);
    if (is_seat(ev.player)) line += std::string(" player=") + trace_seat_word(ev.player);
    if (ev.zone >= 0 || ev.to_zone >= 0 || ev.kind == EventKind::ZONE_MOVE) {
        line += " " + zone_text(ev.zone, ev.zone_owner);
        if (ev.to_zone >= 0 || ev.kind == EventKind::ZONE_MOVE) line += " -> " + zone_text(ev.to_zone, ev.to_owner);
    }
    if (ev.count) line += " count=" + std::to_string(ev.count);
    if (ev.range.anchor != rmstate::NONE8) line += " range=" + libpos_text(ev.range);
    if (ev.batch) line += " batch=" + std::to_string(ev.batch);
    if (ev.kind == EventKind::REVEAL_START || ev.kind == EventKind::REVEAL_END || ev.kind == EventKind::LOOK)
        line += std::string(" to=") + (is_seat(ev.to) ? trace_seat_word(ev.to) : "all");
    if (ev.source) line += " src=h" + std::to_string(ev.source);
    if (ev.source_card >= 0) line += " src=" + card_text(ev.source_card);
    if (ev.opening) line += " opening";
    if (ev.kind == EventKind::GAME_START || ev.kind == EventKind::GAME_END || ev.kind == EventKind::MATCH_END)
        line += " game=" + std::to_string(ev.game);
    for (const ObsEventCard &c : ev.cards) {
        line += " [h" + std::to_string(c.handle_before) + ">h" + std::to_string(c.handle_after) + " " +
                (c.card >= 0 ? card_text(c.card) : std::string("?"));
        if (c.from.anchor != rmstate::NONE8) line += " from=" + libpos_text(c.from);
        if (c.to.anchor != rmstate::NONE8) line += " to=" + libpos_text(c.to);
        if (c.face_down) line += " face-down";
        if (c.token) line += " token";
        if (c.hand_subset == static_cast<uint8_t>(rmstate::HandSubset::DRAWN_THIS_TURN)) line += " drawn-this-turn";
        if (c.draw_seq) line += " draw=" + std::to_string(c.draw_seq) + "/" + std::to_string(c.draw_idx);
        line += "]";
    }
    for (const ObsEventPlayer &p : ev.players)
        line += std::string(" {") + trace_seat_word(p.player) + " library=" + std::to_string(p.library) +
                " sideboard=" + std::to_string(p.sideboard) + " wins=" + std::to_string(p.wins) + "}";
    game_log("%s\n", line.c_str());
}

static std::string kind_word(EventKind kind) {
#define RMSTATE_EVENT_KIND_NAME(name) #name,
    static const char *const names[] = {RMSTATE_EVENT_KIND(RMSTATE_EVENT_KIND_NAME)};
#undef RMSTATE_EVENT_KIND_NAME
    std::string w = names[static_cast<size_t>(kind)];
    for (char &ch : w) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return w;
}

static std::string zone_text(int zone, Zone::Ownership owner) {
    if (zone < 0) return "none";
    return std::string(trace_zone_word(static_cast<Zone::ZoneValue>(zone))) + ":" + trace_seat_word(owner);
}

// top3, bottom0, top0+2 (one of positions 0-1), top? (anywhere).
static std::string libpos_text(const ObsLibPos &p) {
    std::string out = p.anchor == static_cast<uint8_t>(LibAnchor::BOTTOM) ? "bottom" : "top";
    if (p.span == 0) return out + "?";
    out += std::to_string(p.pos);
    if (p.span > 1) out += "+" + std::to_string(p.span);
    return out;
}

static std::string card_text(int card) {
    const char *name = card_index_to_name(card);
    return name ? std::string(name) : "#" + std::to_string(card);
}

static const char *viewer_word(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A ? "A" : viewer == Zone::PLAYER_B ? "B" : "omni";
}
