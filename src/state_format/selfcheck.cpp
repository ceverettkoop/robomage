#include "selfcheck.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>

#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../game_driver.h"
#include "../queries/players.h"
#include "../queries/visibility.h"
#include "../queries/zones.h"
#include "../search_server.h"
#include "../snapshot.h"
#include "../stable_rng.h"
#include "../systems/state_manager.h"
#include "decode.h"
#include "encode.h"
#include "handles.h"
#include "schema.h"
#include "stream.h"

using rmstate::DecodedFrame;
using rmstate::DecodedTable;
using rmstate::TableId;

namespace {

// What the self-check has done, for the summary line.
struct Stats {
        size_t decisions = 0;
        size_t frames = 0;
        size_t events = 0;
        size_t permutations = 0;
        size_t snapshots = 0;
        size_t errors = 0;
};

// A snapshot check in progress: the decision it saved and the frames that decision encoded.
struct SnapshotCheck {
        bool active = false;
        size_t saved_at = 0;  // Stats::decisions at the save
        int game = -1;        // match_game_number at the save
        std::vector<LegalAction> actions;
        Zone::Ownership decider = Zone::UNKNOWN;
        bool priority_window = false;
        // The continuous-effects registry at the save (derived state the snapshot leaves out).
        std::vector<ActiveStatic> statics;
        std::string frames[3];  // indexed like VIEWERS
};

// The changes a permutation made, so that everything can be put back exactly.
struct Permutation {
        std::vector<std::pair<Entity, Zone>> zones;  // the original Zone of every card it moved
        std::vector<Entity> holders;                 // the cards whose identities it permuted
        std::vector<size_t> from;                    // holders[i] holds the card holders[from[i]] held
};

}  // namespace

static const Zone::Ownership VIEWERS[] = {Zone::PLAYER_A, Zone::PLAYER_B, Zone::UNKNOWN};
// The snapshot slots the snapshot check uses (the search server, which uses them too, is off).
static const int SAVED_SLOT = 2;
static const int RESUME_SLOT = 3;
// Decisions that must pass between a snapshot check's save and its restore.
static const size_t SNAPSHOT_K = 3;
// ERROR: lines printed before the rest are only counted.
static const size_t MAX_ERROR_LINES = 40;

static Stats stats;
static SnapshotCheck snap;
static bool match_frames_checked = false;
// The seat a live ScrambledIdentities scrambles for (UNKNOWN = none).
static Zone::Ownership scramble_viewer = Zone::UNKNOWN;

static bool is_seat(Zone::Ownership p);
static int viewer_index(Zone::Ownership viewer);
static const char *viewer_word(Zone::Ownership viewer);
static std::string context();
static void report(const char *check, Zone::Ownership viewer, const std::string &detail);
static bool in_window(Entity e, const Zone &z, Zone::Ownership viewer, bool any_audience);
static bool hidden_from(Entity e, Zone::Ownership viewer);
static std::vector<size_t> inverse(const std::vector<size_t> &from);
static int other_card_id(int id);
static int scramble_card_id(Entity e, int id);
static const rmstate::Schema &schema();
static std::string value_text(const rmstate::Value &v);
static std::string row_card(const DecodedTable &t, size_t row);
static std::string describe_difference(const std::string &a, const std::string &b);
static void compare_frames(
    const char *check, Zone::Ownership viewer, const std::string &expected, const std::string &actual);
static bool roundtrip(Zone::Ownership viewer, const std::string &payload, DecodedFrame *out);
static int column(const DecodedTable &t, const char *name);
static uint64_t cell(const DecodedTable &t, size_t row, int col);
static size_t rows_of(const DecodedFrame &f, TableId id);
static void check_complete(Zone::Ownership viewer, const DecodedFrame &f, size_t menu_rows);
static void check_references(Zone::Ownership viewer, const DecodedFrame &f, const std::set<uint64_t> &handles);
static void check_frame(Zone::Ownership viewer, const std::string &payload, size_t menu_rows);
static void check_match_frames();
static std::vector<Entity> public_choice_cards(const std::vector<LegalAction> &actions);
static void check_scramble(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, const std::vector<Entity> &keep, const std::string &frame, bool terminal);
static void permute_hidden(Zone::Ownership viewer, const std::vector<Entity> &keep, std::mt19937 &gen, Permutation &p);
static void undo_permutation(Permutation &p);
static void check_permutation(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, const std::vector<Entity> &keep, const std::string &frame);
static void start_snapshot_check(
    const std::vector<LegalAction> &actions, Zone::Ownership decider, const std::string frames[3]);
static void finish_snapshot_check();

// ── Reporting ─────────────────────────────────────────────────────────────────────────────────

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

static int viewer_index(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A ? 0 : viewer == Zone::PLAYER_B ? 1 : 2;
}

static const char *viewer_word(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A ? "A" : viewer == Zone::PLAYER_B ? "B" : "omniscient";
}

// Where the game is, for an ERROR: line: the decision as the frame header numbers it.
static std::string context() {
    return "decision " + std::to_string(cur_game.state_decisions) + " (game " +
           std::to_string(match_game_number < 0 ? 0 : match_game_number) + ", turn " +
           std::to_string(cur_game.turn_state.turn) + ", step " +
           std::to_string(static_cast<int>(cur_game.turn_state.step)) + (sideboard_phase ? ", sideboarding" : "") + ")";
}

static void report(const char *check, Zone::Ownership viewer, const std::string &detail) {
    ++stats.errors;
    if (stats.errors > MAX_ERROR_LINES) return;
    non_fatal_error(
        std::string("selfcheck-state ") + check + ": viewer " + viewer_word(viewer) + ", " + context() + ": " + detail);
    if (stats.errors == MAX_ERROR_LINES) non_fatal_error("selfcheck-state: further failures are only counted");
}

void selfcheck_state_mismatch(const char *check, Zone::Ownership viewer, const std::string &detail) {
    report(check, viewer, detail);
}

bool selfcheck_state_enabled() {
    return selfcheck_state && !search_server_mode;
}

// ── Hidden identities (docs/state_format.md §3, decided here without queries/visibility.h) ────

// Whether a reveal window on `e` (the object it is now) is shown to `viewer`, or with
// `any_audience` to anyone.
static bool in_window(Entity e, const Zone &z, Zone::Ownership viewer, bool any_audience) {
    for (const Game::RevealWindow &w : cur_game.reveal_windows) {
        if (w.card.lki_entity() != e || w.card.gen != z.obj_gen) continue;
        if (any_audience || w.audience == Zone::UNKNOWN || w.audience == viewer) return true;
    }
    return false;
}

// Whether `viewer` (a seat) cannot see `e`'s identity: a card in a library, another player's hand
// or sideboard, or face down (except a face-down permanent or spell the viewer controls), and not
// in one of the viewer's reveal windows.
static bool hidden_from(Entity e, Zone::Ownership viewer) {
    if (!is_seat(viewer) || !global_coordinator.entity_has_component<Zone>(e)) return false;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    bool seen = false;
    if (z.is_face_down) {
        seen = (z.location == Zone::BATTLEFIELD || z.location == Zone::STACK) && z.controller == viewer;
    } else {
        switch (z.location) {
            case Zone::BATTLEFIELD:
            case Zone::STACK:
            case Zone::GRAVEYARD:
            case Zone::EXILE:
                seen = true;
                break;
            case Zone::HAND:
            case Zone::SIDEBOARD:
                seen = z.owner == viewer;
                break;
            case Zone::LIBRARY:
                seen = false;
                break;
        }
    }
    return !seen && !in_window(e, z, viewer, false);
}

// The permutation that undoes `from`.
static std::vector<size_t> inverse(const std::vector<size_t> &from) {
    std::vector<size_t> inv(from.size());
    for (size_t i = 0; i < from.size(); ++i) inv[from[i]] = i;
    return inv;
}

// Another valid card id in the same band (cards, tokens) as `id`: the next registered one.
static int other_card_id(int id) {
    if (id >= TOKEN_VOCAB_BASE) {
        for (int i = 0; i < TOKEN_VOCAB_SIZE; ++i)
            if (token_vocab_entries[i].index == id) return token_vocab_entries[(i + 1) % TOKEN_VOCAB_SIZE].index;
        return token_vocab_entries[0].index;
    }
    for (int i = 0; i < CARD_VOCAB_SIZE; ++i)
        if (card_vocab_entries[i].index == id) return card_vocab_entries[(i + 1) % CARD_VOCAB_SIZE].index;
    return card_vocab_entries[0].index;
}

// card_identity's test-mode scrambler (set_card_identity_scrambler): a card with CardData already
// holds a substitute identity, so only an object without one (a token) needs its id replaced.
static int scramble_card_id(Entity e, int id) {
    if (id < 0 || global_coordinator.entity_has_component<CardData>(e) || !hidden_from(e, scramble_viewer)) return id;
    return other_card_id(id);
}

// The hidden cards, sorted by name, each take the identity of the card `shift` places on, where
// `shift` is the largest number of copies of one name among them: so every card gets another name
// whenever no name has more than half the copies.
ScrambledIdentities::ScrambledIdentities(Zone::Ownership viewer, const std::vector<Entity> &keep) {
    if (!is_seat(viewer) || is_seat(scramble_viewer)) return;
    active_ = true;
    scramble_viewer = viewer;
    std::vector<std::pair<std::string, Entity>> hidden;
    for (Entity e : zoned_entities()) {
        if (!global_coordinator.entity_has_component<CardData>(e) || !hidden_from(e, viewer)) continue;
        if (std::find(keep.begin(), keep.end(), e) != keep.end()) continue;
        hidden.push_back({global_coordinator.GetComponent<CardData>(e).name, e});
    }
    std::sort(hidden.begin(), hidden.end());
    size_t shift = 0;
    for (size_t i = 0, run = 0; i < hidden.size(); ++i) {
        run = i > 0 && hidden[i].first == hidden[i - 1].first ? run + 1 : 1;
        shift = std::max(shift, run);
    }
    for (size_t i = 0; i < hidden.size(); ++i) {
        holders_.push_back(hidden[i].second);
        from_.push_back((i + shift) % hidden.size());
    }
    global_coordinator.PermuteComponents<CardData>(holders_, from_);
    set_card_identity_scrambler(scramble_card_id);
}

ScrambledIdentities::~ScrambledIdentities() {
    if (!active_) return;
    global_coordinator.PermuteComponents<CardData>(holders_, inverse(from_));
    set_card_identity_scrambler(nullptr);
    scramble_viewer = Zone::UNKNOWN;
}

// ── Frames ────────────────────────────────────────────────────────────────────────────────────

// The schema every frame is decoded with, from this engine's own SCHEMA frame.
static const rmstate::Schema &schema() {
    static rmstate::Schema s;
    static bool built = false;
    if (!built) {
        const rmstate::DecodeResult res = rmstate::decode_schema_frame(encode_schema_frame(Zone::UNKNOWN), &s);
        if (!res.ok()) fatal_error("--selfcheck-state: the SCHEMA frame does not decode: " + res.message);
        built = true;
    }
    return s;
}

static std::string value_text(const rmstate::Value &v) {
    if (!v.text.empty()) return "\"" + v.text + "\"";
    return std::to_string(v.bits);
}

// " (<card name>)" for a row whose table has a card column naming a card, else empty.
static std::string row_card(const DecodedTable &t, size_t row) {
    const int c = column(t, "card");
    if (c < 0) return "";
    const uint64_t id = cell(t, row, c);
    const char *name = id == rmstate::NONE16 ? nullptr : card_index_to_name(static_cast<int>(id));
    return name ? std::string(" (") + name + ")" : "";
}

// The first difference between two frames, as table, row and column when both decode.
static std::string describe_difference(const std::string &a, const std::string &b) {
    DecodedFrame fa, fb;
    if (!rmstate::decode_frame(a, schema(), &fa).ok() || !rmstate::decode_frame(b, schema(), &fb).ok()) {
        size_t at = 0;
        while (at < a.size() && at < b.size() && a[at] == b[at]) ++at;
        return "bytes differ from offset " + std::to_string(at) + " (sizes " + std::to_string(a.size()) + ", " +
               std::to_string(b.size()) + ")";
    }
    std::set<uint16_t> ids;
    for (const DecodedTable &t : fa.tables) ids.insert(t.id);
    for (const DecodedTable &t : fb.tables) ids.insert(t.id);
    for (uint16_t id : ids) {
        const DecodedTable *ta = fa.table(id);
        const DecodedTable *tb = fb.table(id);
        const std::string name = (ta ? ta : tb)->def->name;
        const size_t na = ta ? ta->rows.size() : 0;
        const size_t nb = tb ? tb->rows.size() : 0;
        for (size_t r = 0; r < std::min(na, nb); ++r)
            for (size_t c = 0; c < ta->rows[r].size(); ++c)
                if (ta->rows[r][c].bits != tb->rows[r][c].bits || ta->rows[r][c].text != tb->rows[r][c].text)
                    return "table " + name + " row " + std::to_string(r) + row_card(*ta, r) + " column " +
                           ta->def->columns[c].name + ": " + value_text(ta->rows[r][c]) + " vs " +
                           value_text(tb->rows[r][c]);
        if (na != nb) return "table " + name + ": " + std::to_string(na) + " rows vs " + std::to_string(nb);
    }
    return "frame headers differ";
}

// `actual` must be `expected`'s bytes; a difference is reported for `check`.
static void compare_frames(
    const char *check, Zone::Ownership viewer, const std::string &expected, const std::string &actual) {
    if (expected != actual) report(check, viewer, describe_difference(expected, actual));
}

// Decodes `payload` into `out` and re-encodes it: the bytes must come back the same.
static bool roundtrip(Zone::Ownership viewer, const std::string &payload, DecodedFrame *out) {
    const rmstate::DecodeResult res = rmstate::decode_frame(payload, schema(), out);
    if (!res.ok()) {
        report("roundtrip", viewer, "the frame does not decode: " + res.message);
        return false;
    }
    std::string again;
    const rmstate::DecodeResult enc = rmstate::encode_frame(*out, &again);
    if (!enc.ok()) {
        report("roundtrip", viewer, "the decoded frame does not re-encode: " + enc.message);
        return false;
    }
    compare_frames("roundtrip", viewer, payload, again);
    return true;
}

static int column(const DecodedTable &t, const char *name) {
    for (size_t c = 0; c < t.def->columns.size(); ++c)
        if (t.def->columns[c].name == name) return static_cast<int>(c);
    return -1;
}

static uint64_t cell(const DecodedTable &t, size_t row, int col) {
    return col < 0 ? 0 : t.rows[row][static_cast<size_t>(col)].bits;
}

static size_t rows_of(const DecodedFrame &f, TableId id) {
    const DecodedTable *t = f.table(static_cast<uint16_t>(id));
    return t ? t->rows.size() : 0;
}

// The complete check (selfcheck.h) of one decoded STATE frame against the game.
static void check_complete(Zone::Ownership viewer, const DecodedFrame &f, size_t menu_rows) {
    const Zone::Ownership v = is_seat(viewer) ? viewer : Zone::UNKNOWN;
    // The object table: handles, and rows per (zone, owner).
    std::set<uint64_t> handles;
    std::map<std::pair<uint64_t, uint64_t>, size_t> rows_at;
    if (const DecodedTable *obj = f.table(static_cast<uint16_t>(TableId::OBJECT))) {
        const int hc = column(*obj, "handle"), zc = column(*obj, "zone"), oc = column(*obj, "owner");
        for (size_t r = 0; r < obj->rows.size(); ++r) {
            const uint64_t h = cell(*obj, r, hc);
            if (h == 0 || !handles.insert(h).second)
                report("complete", viewer,
                    "object row " + std::to_string(r) + " has handle " + std::to_string(h) +
                        (h == 0 ? "" : ", which another row has too"));
            ++rows_at[{cell(*obj, r, zc), cell(*obj, r, oc)}];
        }
    }
    // The game, counted independently: each zone's size per owner, and the objects the viewer
    // can identify (every public one, their own hand and sideboard, their windows).
    std::map<std::pair<uint64_t, uint64_t>, size_t> engine_at;
    size_t counts[3][7] = {};
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (is_seat(z.owner) && z.location >= 0 && z.location < 7) ++counts[z.owner][z.location];
        const bool public_zone = z.location == Zone::BATTLEFIELD || z.location == Zone::STACK ||
                                 z.location == Zone::GRAVEYARD || z.location == Zone::EXILE;
        const bool own = is_seat(v) && z.owner == v && (z.location == Zone::HAND || z.location == Zone::SIDEBOARD);
        const bool identifiable = individually_identifiable(e, v);
        const uint32_t h = handle_peek(v, e);
        if (identifiable && (h == 0 || !handles.count(h)))
            report("complete", viewer,
                std::string("an object it can identify (") + trace_zone_word(z.location) + " of " +
                    trace_seat_word(z.owner) + ") has no object row");
        if (!identifiable && h != 0)
            report("complete", viewer,
                std::string("an object it cannot identify (") + trace_zone_word(z.location) + " of " +
                    trace_seat_word(z.owner) + ") holds handle " + std::to_string(h));
        if (public_zone || own || !is_seat(v))
            ++engine_at[{static_cast<uint64_t>(z.location), static_cast<uint64_t>(z.owner)}];
    }
    for (const auto &kv : engine_at) {
        const auto at = rows_at.find(kv.first);
        const size_t got = at == rows_at.end() ? 0 : at->second;
        if (got != kv.second)
            report("complete", viewer,
                std::to_string(got) + " object rows in " +
                    trace_zone_word(static_cast<Zone::ZoneValue>(kv.first.first)) + " of " +
                    trace_seat_word(static_cast<Zone::Ownership>(kv.first.second)) + ", which holds " +
                    std::to_string(kv.second) + (is_seat(v) ? " visible objects" : " objects"));
    }
    // The player rows' zone counts.
    if (const DecodedTable *pl = f.table(static_cast<uint16_t>(TableId::PLAYER))) {
        const int pc = column(*pl, "player");
        const std::pair<const char *, Zone::ZoneValue> zones[] = {{"hand", Zone::HAND}, {"library", Zone::LIBRARY},
            {"graveyard", Zone::GRAVEYARD}, {"exile", Zone::EXILE}, {"sideboard", Zone::SIDEBOARD}};
        if (pl->rows.size() != 2) report("complete", viewer, std::to_string(pl->rows.size()) + " player rows");
        for (size_t r = 0; r < pl->rows.size(); ++r) {
            const uint64_t seat = cell(*pl, r, pc);
            if (seat != Zone::PLAYER_A && seat != Zone::PLAYER_B) continue;
            for (const auto &zn : zones) {
                const uint64_t got = cell(*pl, r, column(*pl, zn.first));
                if (got != counts[seat][zn.second])
                    report("complete", viewer,
                        std::string("player ") + trace_seat_word(static_cast<Zone::Ownership>(seat)) + " " + zn.first +
                            " count " + std::to_string(got) + ", zone holds " +
                            std::to_string(counts[seat][zn.second]));
            }
        }
    } else {
        report("complete", viewer, "no player table");
    }
    // The reveal table lists the viewer's windows; the menu, the actions as offered.
    const size_t windows = reveal_windows_for(v).size();
    if (rows_of(f, TableId::REVEAL) != windows)
        report("complete", viewer,
            std::to_string(rows_of(f, TableId::REVEAL)) + " reveal rows for " + std::to_string(windows) + " windows");
    if (rows_of(f, TableId::MENU) != menu_rows)
        report("complete", viewer,
            std::to_string(rows_of(f, TableId::MENU)) + " menu rows for " + std::to_string(menu_rows) + " actions");
    check_references(viewer, f, handles);
}

// Every handle a snapshot table names is an object row's, and every row: reference is in range.
// The event tables may name objects that have since gone, so only their row: references count.
static void check_references(Zone::Ownership viewer, const DecodedFrame &f, const std::set<uint64_t> &handles) {
    const uint16_t first_event = static_cast<uint16_t>(TableId::EVENT);
    for (const DecodedTable &t : f.tables) {
        for (size_t c = 0; c < t.def->columns.size(); ++c) {
            const rmstate::ColumnDef &col = t.def->columns[c];
            const bool handle = col.tag == "handle" && t.id < first_event &&
                                !(t.id == static_cast<uint16_t>(TableId::OBJECT) && col.name == "handle");
            const bool row_ref = col.tag.rfind("row:", 0) == 0;
            if (!handle && !row_ref) continue;
            size_t target_rows = 0;
            if (row_ref) {
                const std::string target = col.tag.substr(4);
                for (const DecodedTable &u : f.tables)
                    if (u.def->name == target) target_rows = u.rows.size();
            }
            for (size_t r = 0; r < t.rows.size(); ++r) {
                const rmstate::Value &val = t.rows[r][c];
                if (handle && val.bits != 0 && !handles.count(val.bits))
                    report("complete", viewer,
                        "table " + t.def->name + " row " + std::to_string(r) + " column " + col.name +
                            " names handle " + std::to_string(val.bits) + ", which has no object row");
                if (row_ref && !rmstate::value_is_all_ones(val, col.storage) && val.bits >= target_rows)
                    report("complete", viewer,
                        "table " + t.def->name + " row " + std::to_string(r) + " column " + col.name + " names row " +
                            std::to_string(val.bits) + " of " + std::to_string(target_rows));
            }
        }
    }
}

// Roundtrip and complete for one STATE frame.
static void check_frame(Zone::Ownership viewer, const std::string &payload, size_t menu_rows) {
    ++stats.frames;
    DecodedFrame f;
    if (roundtrip(viewer, payload, &f)) check_complete(viewer, f, menu_rows);
}

// Once per match: each viewer's SCHEMA, CARDS and DECKLISTS frames decode and re-encode the same.
static void check_match_frames() {
    if (match_frames_checked) return;
    match_frames_checked = true;
    for (Zone::Ownership viewer : VIEWERS) {
        for (const std::string &payload :
            {encode_schema_frame(viewer), encode_cards_frame(viewer), encode_decklists_frame(viewer)}) {
            DecodedFrame f;
            roundtrip(viewer, payload, &f);
        }
    }
}

// ── Leak checks ───────────────────────────────────────────────────────────────────────────────

// The cards menu choices make public themselves (LegalAction::card_is_public: a name to name, a
// revealed tutor target): their identities belong in the menu whatever zone they are in.
static std::vector<Entity> public_choice_cards(const std::vector<LegalAction> &actions) {
    std::vector<Entity> out;
    for (const LegalAction &la : actions) {
        if (!la.card_is_public) continue;
        for (Entity e : {la.source_entity, la.target_entity})
            if (e != 0) out.push_back(e);
    }
    return out;
}

// The scramble check of one seat's frame.
static void check_scramble(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, const std::vector<Entity> &keep, const std::string &frame, bool terminal) {
    std::string scrambled;
    {
        ScrambledIdentities scramble(viewer, keep);
        scrambled = encode_state_frame(viewer, actions, decider, events, terminal);
    }
    compare_frames("scramble", viewer, frame, scrambled);
}

void selfcheck_state_event(Zone::Ownership viewer, const ObsEvent &built, const ObsEvent &scrambled) {
    ++stats.events;
    const std::string a = encode_events_frame(viewer, {built});
    const std::string b = encode_events_frame(viewer, {scrambled});
    if (a != b) report("scramble", viewer, "an event as it is built: " + describe_difference(a, b));
}

// Permutes everything hidden from `viewer` (a seat), keeping every live reveal window fixed:
// the opponent's hidden hand, library and sideboard cards are dealt at random among those cards'
// places (zone and library position), the viewer's own hidden library cards among their
// positions, and then the identities (CardData) of every card hidden from the viewer, per owner,
// are shuffled among them (a card in a window shown only to the other player keeps its place
// but not its identity; a face-down exiled card stays where it is, as the objects that refer to it
// expect, with another identity). Owners, zone sizes, identity stamps and handles stay as they
// were. `keep` cards are left alone.
static void permute_hidden(Zone::Ownership viewer, const std::vector<Entity> &keep, std::mt19937 &gen, Permutation &p) {
    const Zone::Ownership opp = opponent_of(viewer);
    std::vector<Entity> dealt, own;
    std::vector<Entity> identities[3];
    for (Entity e : zoned_entities()) {
        if (!global_coordinator.entity_has_component<CardData>(e) || !hidden_from(e, viewer)) continue;
        if (std::find(keep.begin(), keep.end(), e) != keep.end()) continue;
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if (!is_seat(z.owner)) continue;
        const bool hidden_zone =
            z.location == Zone::HAND || z.location == Zone::LIBRARY || z.location == Zone::SIDEBOARD;
        if (hidden_zone || (z.location == Zone::EXILE && z.is_face_down)) identities[z.owner].push_back(e);
        if (!hidden_zone || z.is_face_down || in_window(e, z, viewer, true)) continue;
        if (z.owner == opp)
            dealt.push_back(e);
        else if (z.location == Zone::LIBRARY)
            own.push_back(e);
    }
    std::vector<Zone> places;
    for (Entity e : dealt) {
        p.zones.push_back({e, global_coordinator.GetComponent<Zone>(e)});
        places.push_back(p.zones.back().second);
    }
    stable_shuffle(places, gen);
    for (size_t i = 0; i < dealt.size(); ++i) {
        Zone &z = global_coordinator.GetComponent<Zone>(dealt[i]);
        z.location = places[i].location;
        z.distance_from_top = places[i].distance_from_top;
    }
    std::vector<size_t> positions;
    for (Entity e : own) {
        p.zones.push_back({e, global_coordinator.GetComponent<Zone>(e)});
        positions.push_back(p.zones.back().second.distance_from_top);
    }
    stable_shuffle(positions, gen);
    for (size_t i = 0; i < own.size(); ++i)
        global_coordinator.GetComponent<Zone>(own[i]).distance_from_top = positions[i];
    for (const std::vector<Entity> &pool : identities) {
        std::vector<size_t> order(pool.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        stable_shuffle(order, gen);
        const size_t base = p.holders.size();
        for (size_t i = 0; i < pool.size(); ++i) {
            p.holders.push_back(pool[i]);
            p.from.push_back(base + order[i]);
        }
    }
    global_coordinator.PermuteComponents<CardData>(p.holders, p.from);
}

static void undo_permutation(Permutation &p) {
    global_coordinator.PermuteComponents<CardData>(p.holders, inverse(p.from));
    for (const auto &kv : p.zones) global_coordinator.GetComponent<Zone>(kv.first) = kv.second;
    p = Permutation{};
}

// The permutation check of one seat's frame. The seat's handle table is put back as well: a
// permutation that changed what the seat can identify would retire handles (and show as a
// difference).
static void check_permutation(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, const std::vector<Entity> &keep, const std::string &frame) {
    ++stats.permutations;
    std::mt19937 gen(static_cast<uint32_t>(0x5e1fc4ecu ^ (stats.decisions * 2654435761u) ^ viewer));
    const Game::HandleTable table = cur_game.handle_tables[HandleMove::slot(viewer)];
    Permutation p;
    permute_hidden(viewer, keep, gen, p);
    const std::string permuted = encode_state_frame(viewer, actions, decider, events);
    undo_permutation(p);
    cur_game.handle_tables[HandleMove::slot(viewer)] = table;
    compare_frames("permutation", viewer, frame, permuted);
}

// ── Snapshot check ────────────────────────────────────────────────────────────────────────────

// Records the decision whose state SAVED_SLOT now holds.
static void start_snapshot_check(
    const std::vector<LegalAction> &actions, Zone::Ownership decider, const std::string frames[3]) {
    snap.active = true;
    snap.saved_at = stats.decisions;
    snap.game = match_game_number;
    snap.actions = actions;
    snap.decider = decider;
    snap.priority_window = priority_window_open();
    snap.statics = g_active_statics;
    for (int i = 0; i < 3; ++i) snap.frames[i] = frames[i];
}

// At a loop-safe decision at least SNAPSHOT_K decisions after the save: saves the current state,
// restores the saved decision, encodes every viewer's frame there (with that decision's menu,
// decider and priority window, and the events its frames drained) and compares, then restores
// the current state. The restore runs where the search server's protocol allows a snapshot (the
// decision can be re-derived from the restored data), and the state put back is the one just
// saved, so play goes on unchanged. Both restores are exact: the continuous-effects registry is
// put back from copies instead of re-derived (snapshot_restore_exact), because at a decision in
// the middle of a flow the characteristics the engine last derived can lag the state (they are
// re-derived at each state-based pass) and a re-derivation would change them.
static void finish_snapshot_check() {
    ++stats.snapshots;
    const bool window = priority_window_open();
    const std::vector<ActiveStatic> statics = g_active_statics;
    snapshot_save(RESUME_SLOT);
    snapshot_restore_exact(SAVED_SLOT);
    g_active_statics = snap.statics;
    set_priority_window_open(snap.priority_window);
    std::string frames[3];
    for (Zone::Ownership viewer : VIEWERS)
        frames[viewer_index(viewer)] =
            encode_state_frame(viewer, snap.actions, snap.decider, take_obs_events(viewer, ObsConsumer::SELFCHECK));
    set_priority_window_open(window);
    snapshot_restore_exact(RESUME_SLOT);
    g_active_statics = statics;
    for (Zone::Ownership viewer : VIEWERS)
        compare_frames("snapshot", viewer, snap.frames[viewer_index(viewer)], frames[viewer_index(viewer)]);
    snap = SnapshotCheck{};
}

// ── Entry points ──────────────────────────────────────────────────────────────────────────────

void selfcheck_state_decision(const std::vector<LegalAction> &actions) {
    if (!selfcheck_state_enabled()) return;
    ++stats.decisions;
    snapshot_set_quiet(true);
    const bool sampled = stats.decisions % static_cast<size_t>(std::max(1, selfcheck_state_sample)) == 0;
    // A snapshot is saved and restored only where the search server may take one (the state is
    // all in cur_game and the ECS), never between games.
    const bool restorable = search_loop_safe() && !sideboard_phase;
    if (snap.active && (snap.game != match_game_number || !snapshot_slot_live(SAVED_SLOT))) snap = SnapshotCheck{};
    const bool start = sampled && restorable && !snap.active;
    if (start) snapshot_save(SAVED_SLOT);

    check_match_frames();
    const Zone::Ownership decider = rmstate_deciding_seat();
    const std::vector<Entity> keep = public_choice_cards(actions);
    std::string frames[3];
    for (Zone::Ownership viewer : VIEWERS) {
        const int i = viewer_index(viewer);
        const std::vector<ObsEvent> events = take_obs_events(viewer, ObsConsumer::SELFCHECK);
        frames[i] = encode_state_frame(viewer, actions, decider, events);
        const bool gets_menu = !is_seat(viewer) || viewer == decider;
        check_frame(viewer, frames[i], gets_menu ? actions.size() : 0);
        if (!is_seat(viewer)) continue;
        check_scramble(viewer, actions, decider, events, keep, frames[i], false);
        if (sampled) check_permutation(viewer, actions, decider, events, keep, frames[i]);
    }
    if (start)
        start_snapshot_check(actions, decider, frames);
    else if (snap.active && restorable && stats.decisions - snap.saved_at >= SNAPSHOT_K)
        finish_snapshot_check();
}

void selfcheck_state_match_over() {
    if (!selfcheck_state_enabled()) return;
    snap = SnapshotCheck{};
    check_match_frames();
    for (Zone::Ownership viewer : VIEWERS) {
        const std::vector<ObsEvent> events = take_obs_events(viewer, ObsConsumer::SELFCHECK);
        const std::string frame = encode_state_frame(viewer, {}, Zone::UNKNOWN, events, true);
        check_frame(viewer, frame, 0);
        if (is_seat(viewer)) check_scramble(viewer, {}, Zone::UNKNOWN, events, {}, frame, true);
    }
    std::fprintf(stderr,
        "SELFCHECK-STATE: decisions=%zu frames=%zu events=%zu permutations=%zu snapshots=%zu errors=%zu\n",
        stats.decisions, stats.frames, stats.events, stats.permutations, stats.snapshots, stats.errors);
    std::fflush(stderr);
}
