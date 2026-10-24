#include "board.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <set>
#include <utility>

#include "../str_util.h"

namespace cli {

// Long lists are wrapped before this column.
static constexpr size_t WRAP = 100;

namespace {

// One decision's frames as the renderer reads them: the tables (found by name in the SCHEMA
// frame), the object and card lookups, the viewer's seat values and the zone enum values. Built
// once per rendering call.
struct Frame {
        explicit Frame(const agents::DecisionView &view);
        const agents::DecisionView &v;
        const agents::SeatStream &st;
        // STATE frame.
        agents::Table header, turn, player, pcounter, obj, counter, keyword, combat, stack, smode, starget, lki, reveal,
            pending, cost, pip, menu, match, own_deck, emblem, effect, delayed, permission;
        agents::Table event, event_card, event_player;
        agents::Table k_meta, k_seen, k_pool, k_pool_card, k_position, k_known_hand;
        // CARDS frame.
        agents::Table card, card_subtype;
        std::map<uint64_t, size_t> by_handle;  // object handle -> object row
        std::map<int, size_t> front, back;     // card id -> its CARDS `card` row, per face
        uint64_t me = 0, opp = 0;              // the viewer's and the opponent's seat values
        int z_library, z_battlefield, z_hand, z_stack, z_graveyard, z_exile, z_sideboard;
        int layout_split;  // the SPLIT value of the CARDS `layout` column
        uint64_t creature_bit = 0;
};

Frame::Frame(const agents::DecisionView &view) : v(view), st(view.stream()) {
    header = v.table("header");
    turn = v.table("turn");
    player = v.table("player");
    pcounter = v.table("player_counter");
    obj = v.objects();
    counter = v.table("object_counter");
    keyword = v.table("object_keyword");
    combat = v.table("combat");
    stack = v.stack();
    smode = v.table("stack_mode");
    starget = v.table("stack_target");
    lki = v.table("lki");
    reveal = v.table("reveal");
    pending = v.table("pending");
    cost = v.table("pending_cost");
    pip = v.table("pending_cost_pip");
    menu = v.table("menu");
    match = v.table("match");
    own_deck = v.table("own_deck");
    emblem = v.table("emblem");
    effect = v.table("player_effect");
    delayed = v.table("delayed_trigger");
    permission = v.table("play_permission");
    event = v.table("event");
    event_card = v.table("event_card");
    event_player = v.table("event_player");
    k_meta = v.table("k_meta");
    k_seen = v.table("k_seen");
    k_pool = v.table("k_pool");
    k_pool_card = v.table("k_pool_card");
    k_position = v.table("k_position");
    k_known_hand = v.table("k_known_hand");
    const rmstate::Schema &schema = st.schema();
    card = agents::Table(st.cards_frame(), schema, "card");
    card_subtype = agents::Table(st.cards_frame(), schema, "card_subtype");
    const int c_handle = obj.col("handle");
    for (size_t r = 0; r < obj.rows(); ++r) by_handle.emplace(obj.u(r, c_handle), r);
    const int c_card = card.col("card"), c_face = card.col("face");
    for (size_t r = 0; r < card.rows(); ++r) (card.u(r, c_face) ? back : front).emplace(card.card(r, c_card), r);
    const int a = st.enum_value("seat", "PLAYER_A"), b = st.enum_value("seat", "PLAYER_B");
    me = static_cast<uint64_t>(v.seat() == 0 ? a : b);
    opp = static_cast<uint64_t>(v.seat() == 0 ? b : a);
    z_library = st.enum_value("zone", "LIBRARY");
    z_battlefield = st.enum_value("zone", "BATTLEFIELD");
    z_hand = st.enum_value("zone", "HAND");
    z_stack = st.enum_value("zone", "STACK");
    z_graveyard = st.enum_value("zone", "GRAVEYARD");
    z_exile = st.enum_value("zone", "EXILE");
    z_sideboard = st.enum_value("zone", "SIDEBOARD");
    layout_split = st.enum_value("layout", "SPLIT");
    const int creature = st.enum_value("card_type", "Creature");
    if (creature >= 0) creature_bit = 1ull << creature;
}

// One rendered event line before merging: its text and whether it may merge with a neighbour.
struct EventLine {
        std::string text;
        bool game_event = false;
};

}  // namespace

static std::string humanize(const std::string &name);
static std::string lower_words(const std::string &name);
static std::string pad(const std::string &s, size_t width);
static std::string join(const std::vector<std::string> &items, const char *sep);
static std::vector<std::string> wrap(
    const std::string &head, const std::vector<std::string> &items, const char *indent);
static std::string ordinal(uint64_t n);
static std::string plural(uint64_t n, const char *one, const char *many);
static std::string name_key(const std::string &name);
static std::string you(const Frame &f, uint64_t seat);
static std::string possessive(const Frame &f, uint64_t seat);
static std::string ref(uint64_t handle);
static bool is_split_card(const Frame &f, int cid);
static std::string card_name(const Frame &f, int cid);
static std::string object_name(const Frame &f, size_t row);
static std::string object_ref(const Frame &f, uint64_t handle);
static std::string counter_kind(const std::string &kind);
static std::string object_counters(const Frame &f, uint64_t handle);
static bool is_creature(const Frame &f, size_t row);
static std::string power_toughness(const Frame &f, size_t row);
static std::vector<std::string> object_status(const Frame &f, size_t row);
static std::string mask_names(const Frame &f, const char *enum_name, uint64_t mask);
static std::string mana_pool(const Frame &f, size_t prow);
static std::vector<std::string> player_lines(const Frame &f, bool sideboarding);
static std::vector<std::string> battlefield_lines(const Frame &f, uint64_t controller);
static std::string stack_item(const Frame &f, size_t row);
static std::vector<std::string> stack_lines(const Frame &f);
static std::vector<std::string> hand_lines(const Frame &f);
static std::vector<size_t> zone_rows(const Frame &f, int zone, uint64_t owner);
static std::vector<std::string> zone_list(const Frame &f, const std::string &head, int zone, uint64_t owner);
static std::string zone_word(const Frame &f, int zone);
static std::string zone_of(const Frame &f, int zone, uint64_t owner);
static std::vector<std::string> reveal_lines(const Frame &f);
static std::vector<std::string> effect_lines(const Frame &f);
static std::vector<std::string> deck_lines(const Frame &f);
static bool opening_hand_decision(const Frame &f);
static std::string category_task(const std::string &category);
static std::string pending_context(const Frame &f);
static std::string cost_text(const Frame &f, size_t row);
static std::string card_cost(const Frame &f, int cid);
static std::string type_line(const Frame &f, size_t card_row);
static std::vector<std::string> card_face_lines(const Frame &f, size_t card_row);
static int object_card_id(const Frame &f, size_t row);
static std::string libpos(const Frame &f, const agents::Table &t, size_t row, const std::string &prefix);
static std::string pool_label(const Frame &f, size_t pool, bool owner_name);
static std::vector<std::string> pool_cards(const Frame &f, size_t pool, bool candidates);
static std::vector<std::string> known_hand(const Frame &f);
static std::vector<std::string> known_positions(const Frame &f, uint64_t owner, size_t limit);
static std::vector<std::string> library_zone(const Frame &f, uint64_t owner);
static std::string event_cards(const Frame &f, const std::vector<size_t> &rows, uint64_t *hidden);
static std::string items_text(const std::vector<std::string> &names, uint64_t hidden, bool article);
static EventLine event_line(
    const Frame &f, size_t row, const std::vector<size_t> &cards, const std::vector<size_t> &players);

// ── text helpers ──────────────────────────────────────────────────────────────────────────────

// FIRST_MAIN -> "First main".
static std::string humanize(const std::string &name) {
    std::string out = lower_words(name);
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

// FIRST_MAIN -> "first main".
static std::string lower_words(const std::string &name) {
    std::string out;
    for (char c : name) out += c == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

static std::string pad(const std::string &s, size_t width) {
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

static std::string join(const std::vector<std::string> &items, const char *sep) {
    std::string out;
    for (const std::string &s : items) out += (out.empty() ? "" : sep) + s;
    return out;
}

// `head`, then `items` separated by ", ", wrapped before column WRAP; continuation lines start
// with `indent`.
static std::vector<std::string> wrap(
    const std::string &head, const std::vector<std::string> &items, const char *indent) {
    std::vector<std::string> out;
    std::string line = head;
    bool line_start = false;
    for (size_t i = 0; i < items.size(); ++i) {
        const std::string piece = items[i] + (i + 1 < items.size() ? "," : "");
        if (!line_start && line.size() + 1 + piece.size() > WRAP && line.size() > std::string(indent).size()) {
            out.push_back(line);
            line = indent;
            line_start = true;
        }
        line += (line_start ? "" : " ") + piece;
        line_start = false;
    }
    out.push_back(line);
    return out;
}

// 1 -> "1st", 2 -> "2nd", 11 -> "11th".
static std::string ordinal(uint64_t n) {
    const uint64_t t = n % 100, d = n % 10;
    const char *suffix = (t >= 11 && t <= 13) ? "th" : d == 1 ? "st" : d == 2 ? "nd" : d == 3 ? "rd" : "th";
    return std::to_string(n) + suffix;
}

static std::string plural(uint64_t n, const char *one, const char *many) {
    return std::to_string(n) + " " + (n == 1 ? one : many);
}

// A card name as `card <name>` matches it: lowercased, accents folded.
static std::string name_key(const std::string &name) {
    return ascii_lower(ascii_fold_card_name(name));
}

// ── names ─────────────────────────────────────────────────────────────────────────────────────

std::string seat_name(const agents::DecisionView &v, uint64_t seat) {
    const std::string &name = v.stream().enum_name("seat", seat);
    if (name == "PLAYER_A") return "Player A";
    if (name == "PLAYER_B") return "Player B";
    return "no player";
}

// "Player A (you)" for the viewer's seat, else its name.
static std::string you(const Frame &f, uint64_t seat) {
    return seat_name(f.v, seat) + (seat == f.me ? " (you)" : "");
}

static std::string possessive(const Frame &f, uint64_t seat) {
    return seat_name(f.v, seat) + "'s";
}

static std::string ref(uint64_t handle) {
    return "#" + std::to_string(handle);
}

// Whether card id `cid` is a split card (CR 709): a CARDS front row of the SPLIT layout, with its
// second half as the back row.
static bool is_split_card(const Frame &f, int cid) {
    auto it = f.front.find(cid);
    return it != f.front.end() && f.layout_split >= 0 && f.back.count(cid) &&
           static_cast<int>(f.card.u(it->second, f.card.col("layout"))) == f.layout_split;
}

// A card id's name as a whole card: a split card's two names ("Dead // Gone"; CR 709.4: outside
// the stack it has both halves' characteristics), any other card's front-face name (CR 712.8a);
// "" for none.
static std::string card_name(const Frame &f, int cid) {
    if (cid < 0) return std::string();
    if (is_split_card(f, cid)) return f.v.cards().name(cid) + " // " + f.v.cards().name(cid, true);
    return f.v.cards().name(cid);
}

// An object row's name as it is now (a split card on the stack is the half cast, CR 709.3b, and
// the whole card elsewhere); "face-down card" when the viewer cannot see its card.
static std::string object_name(const Frame &f, size_t row) {
    const std::optional<std::string> name = f.v.object_name(row);
    if (!name) return "face-down card";
    const int cid = f.obj.card(row, f.obj.col("card"));
    if (static_cast<int>(f.obj.u(row, f.obj.col("zone"))) != f.z_stack && !f.obj.u(row, f.obj.col("token")) &&
        f.obj.card(row, f.obj.col("copy_of")) < 0 && is_split_card(f, cid))
        return card_name(f, cid);
    return *name;
}

// "Goblin Guide #30" for an object of the frame; "#30" for a handle it does not list.
static std::string object_ref(const Frame &f, uint64_t handle) {
    auto it = f.by_handle.find(handle);
    if (it == f.by_handle.end()) return ref(handle);
    return object_name(f, it->second) + " " + ref(handle);
}

// ── objects ───────────────────────────────────────────────────────────────────────────────────

// P1P1 -> "+1/+1", M1M1 -> "-1/-1", LOYALTY -> "loyalty".
static std::string counter_kind(const std::string &kind) {
    if (kind == "P1P1") return "+1/+1";
    if (kind == "M1M1") return "-1/-1";
    return lower_words(kind);
}

// "+1/+1 x2, loyalty 3"-style counters of an object; "" when it has none.
static std::string object_counters(const Frame &f, uint64_t handle) {
    std::vector<std::string> parts;
    const int c_obj = f.counter.col("obj"), c_kind = f.counter.col("kind"), c_count = f.counter.col("count");
    for (size_t r = 0; r < f.counter.rows(); ++r) {
        if (f.counter.u(r, c_obj) != handle) continue;
        const std::string kind = counter_kind(f.counter.text(r, c_kind));
        const uint64_t n = f.counter.u(r, c_count);
        parts.push_back(
            kind[0] == '+' || kind[0] == '-' ? kind + " x" + std::to_string(n) : kind + " " + std::to_string(n));
    }
    return join(parts, ", ");
}

static bool is_creature(const Frame &f, size_t row) {
    return (f.obj.u(row, f.obj.col("types")) & f.creature_bit) != 0;
}

// "2/2" for a creature whose card the viewer sees, else "".
static std::string power_toughness(const Frame &f, size_t row) {
    if (f.obj.card(row, f.obj.col("card")) < 0 || !is_creature(f, row)) return "";
    return std::to_string(f.obj.s(row, f.obj.col("power"))) + "/" +
           std::to_string(f.obj.s(row, f.obj.col("toughness")));
}

// An object's statuses as the frame has them: tapped, summoning sick, combat, damage, counters,
// attachment, token / copy marks.
static std::vector<std::string> object_status(const Frame &f, size_t row) {
    const agents::Table &o = f.obj;
    const uint64_t h = o.u(row, o.col("handle"));
    std::vector<std::string> out;
    if (o.u(row, o.col("tapped"))) out.push_back("tapped");
    if (o.u(row, o.col("sick"))) out.push_back("summoning sick");
    if (o.u(row, o.col("phased_out"))) out.push_back("phased out");
    if (o.u(row, o.col("face_down"))) out.push_back("face down");
    if (o.u(row, o.col("attacking"))) {
        std::string a = "attacking";
        const int c_att = f.combat.col("attacker"), c_kind = f.combat.col("target_kind"),
                  c_target = f.combat.col("target");
        for (size_t r = 0; r < f.combat.rows(); ++r)
            if (f.combat.u(r, c_att) == h &&
                f.st.enum_name("attack_target_kind", f.combat.u(r, c_kind)) == "PLANESWALKER") {
                a += " " + object_ref(f, f.combat.u(r, c_target));
                break;
            }
        out.push_back(a);
    }
    if (o.u(row, o.col("blocked"))) out.push_back("blocked");
    if (o.u(row, o.col("blocking"))) {
        std::vector<std::string> blocked;
        const int c_att = f.combat.col("attacker"), c_blk = f.combat.col("blocker");
        for (size_t r = 0; r < f.combat.rows(); ++r)
            if (f.combat.u(r, c_blk) == h) blocked.push_back(object_ref(f, f.combat.u(r, c_att)));
        out.push_back(blocked.empty() ? "blocking" : "blocking " + join(blocked, ", "));
    }
    if (const uint64_t dmg = o.u(row, o.col("damage")))
        out.push_back("damage " + std::to_string(dmg) + (o.u(row, o.col("deathtouch_damage")) ? " (deathtouch)" : ""));
    const std::string counters = object_counters(f, h);
    if (!counters.empty()) out.push_back(counters);
    if (const uint64_t att = o.u(row, o.col("attached_to"))) out.push_back("attached to " + object_ref(f, att));
    if (o.u(row, o.col("token"))) out.push_back("token");
    const int copy = o.card(row, o.col("copy_of"));
    if (copy >= 0) out.push_back("copy of " + card_name(f, copy));
    const int named = o.card(row, o.col("chosen_name"));
    if (named >= 0) out.push_back("named " + card_name(f, named));
    if (!o.none(row, o.col("chosen_type")))
        out.push_back("chose " + f.st.enum_name("subtype", o.u(row, o.col("chosen_type"))));
    return out;
}

// The names of the set bits of a `mask:<enum_name>` value, space-separated.
static std::string mask_names(const Frame &f, const char *enum_name, uint64_t mask) {
    std::vector<std::string> names;
    const std::vector<std::string> &values = f.st.enum_values(enum_name);
    for (size_t i = 0; i < values.size() && i < 64; ++i)
        if (mask & (1ull << i)) names.push_back(values[i]);
    return join(names, " ");
}

// A player's floating mana as pips ("{R}{R}{C}"); "" when the pool is empty.
static std::string mana_pool(const Frame &f, size_t prow) {
    static const char *colors[] = {"w", "u", "b", "r", "g", "c"};
    std::string out;
    for (const char *c : colors) {
        const uint64_t n = f.player.u(prow, f.player.col((std::string("mana_") + c).c_str()));
        for (uint64_t i = 0; i < n; ++i) out += std::string("{") + static_cast<char>(std::toupper(c[0])) + "}";
    }
    return out;
}

// One line per player (the viewer first): life, zone sizes, pool, counters and designations;
// while `sideboarding`, the match wins alone (the zones are the last game's).
static std::vector<std::string> player_lines(const Frame &f, bool sideboarding) {
    std::vector<std::string> out;
    const agents::Table &p = f.player;
    const int c_player = p.col("player");
    size_t width = 0;
    for (size_t r = 0; r < p.rows(); ++r) width = std::max(width, you(f, p.u(r, c_player)).size());
    bool show_wins = f.header.u(0, f.header.col("game")) > 0;
    for (size_t r = 0; r < p.rows(); ++r) show_wins = show_wins || p.u(r, p.col("match_wins")) > 0;
    for (size_t r = 0; r < p.rows(); ++r) {
        const uint64_t seat = p.u(r, c_player);
        if (sideboarding) {
            out.push_back("  " + pad(you(f, seat), width) + "  wins " + std::to_string(p.u(r, p.col("match_wins"))));
            continue;
        }
        std::string line = "  " + pad(you(f, seat), width) + "  life " + pad(std::to_string(p.s(r, p.col("life"))), 3);
        for (const char *z : {"hand", "library", "graveyard", "exile"})
            line += std::string("  ") + z + " " + pad(std::to_string(p.u(r, p.col(z))), 2);
        std::vector<std::string> extra;
        const std::string pool = mana_pool(f, r);
        if (!pool.empty()) extra.push_back("pool " + pool);
        const int c_cp = f.pcounter.col("player");
        for (size_t c = 0; c < f.pcounter.rows(); ++c)
            if (f.pcounter.u(c, c_cp) == seat && f.pcounter.u(c, f.pcounter.col("count")))
                extra.push_back(lower_words(f.pcounter.text(c, f.pcounter.col("kind"))) + " " +
                                std::to_string(f.pcounter.u(c, f.pcounter.col("count"))));
        if (p.u(r, p.col("monarch"))) extra.push_back("monarch");
        if (p.u(r, p.col("city_blessing"))) extra.push_back("city's blessing");
        if (const uint64_t n = p.u(r, p.col("to_bottom"))) extra.push_back(std::to_string(n) + " to bottom");
        if (const uint64_t n = p.u(r, p.col("extra_turns"))) extra.push_back(plural(n, "extra turn", "extra turns"));
        if (show_wins) extra.push_back("wins " + std::to_string(p.u(r, p.col("match_wins"))));
        if (!extra.empty()) line += "  " + join(extra, ", ");
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out.push_back(line);
    }
    return out;
}

// A battlefield: one line per permanent of `controller`, identical ones merged ("Mountain x3"),
// columns name, P/T, handles, statuses.
static std::vector<std::string> battlefield_lines(const Frame &f, uint64_t controller) {
    struct Entry {
            int rank;  // 0 creature, 1 other nonland, 2 land
            std::string name, pt, status;
            std::vector<uint64_t> handles;
    };
    const int land = f.st.enum_value("card_type", "Land");
    const uint64_t land_bit = land >= 0 ? 1ull << land : 0;
    std::vector<Entry> entries;
    const agents::Table &o = f.obj;
    const int c_zone = o.col("zone"), c_ctrl = o.col("controller"), c_handle = o.col("handle");
    for (size_t r = 0; r < o.rows(); ++r) {
        if (static_cast<int>(o.u(r, c_zone)) != f.z_battlefield || o.u(r, c_ctrl) != controller) continue;
        const uint64_t types = o.u(r, o.col("types"));
        const int rank = (types & f.creature_bit) ? 0 : (types & land_bit) ? 2 : 1;
        Entry e{rank, object_name(f, r), power_toughness(f, r), join(object_status(f, r), ", "), {o.u(r, c_handle)}};
        bool merged = false;
        for (Entry &x : entries)
            if (x.name == e.name && x.pt == e.pt && x.status == e.status) {
                x.handles.push_back(e.handles[0]);
                merged = true;
                break;
            }
        if (!merged) entries.push_back(e);
    }
    if (entries.empty()) return {"  (no permanents)"};
    // Creatures, then other nonlands, then lands; otherwise in the frame's (handle) order.
    std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.rank < b.rank; });
    std::vector<std::string> names, handles;
    size_t w_name = 0, w_pt = 0, w_handles = 0;
    for (const Entry &e : entries) {
        std::string name = e.name + (e.handles.size() > 1 ? " x" + std::to_string(e.handles.size()) : "");
        std::vector<std::string> hs;
        for (uint64_t h : e.handles) hs.push_back(ref(h));
        names.push_back(name);
        handles.push_back(join(hs, " "));
        w_name = std::max(w_name, name.size());
        w_pt = std::max(w_pt, e.pt.size());
        w_handles = std::max(w_handles, handles.back().size());
    }
    std::vector<std::string> out;
    for (size_t i = 0; i < entries.size(); ++i) {
        std::string line = "  " + pad(names[i], w_name) + (w_pt ? "  " + pad(entries[i].pt, w_pt) : "") + "  " +
                           pad(handles[i], w_handles);
        if (!entries[i].status.empty()) line += "  " + entries[i].status;
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out.push_back(line);
    }
    return out;
}

// One stack object: name, kind and controller, X / modes / kicker, targets.
static std::string stack_item(const Frame &f, size_t row) {
    const agents::Table &s = f.stack;
    const uint64_t h = s.u(row, s.col("handle"));
    const std::optional<std::string> name = f.v.stack_card(row);
    const std::string kind = f.st.enum_name("stack_kind", s.u(row, s.col("kind")));
    std::string line = name ? *name : std::string("ability");
    if (kind == "ACTIVATED")
        line += " (activated ability)";
    else if (kind == "TRIGGERED")
        line += " (triggered ability)";
    else if (kind == "COPY" || s.u(row, s.col("copy")))
        line += " (copy)";
    line += " " + ref(h) + " (" + seat_name(f.v, s.u(row, s.col("controller"))) + ")";
    std::vector<std::string> extra;
    const int64_t x = s.s(row, s.col("x"));
    if (x >= 0 && kind == "SPELL") extra.push_back("X=" + std::to_string(x));
    std::vector<std::string> modes;
    for (size_t r = 0; r < f.smode.rows(); ++r)
        if (f.smode.u(r, f.smode.col("stack")) == h)
            modes.push_back(std::to_string(f.smode.u(r, f.smode.col("mode")) + 1));
    if (!modes.empty()) extra.push_back("mode " + join(modes, ", "));
    if (const uint64_t k = s.u(row, s.col("kicked")))
        extra.push_back(k == 1 ? "kicked" : "kicked x" + std::to_string(k));
    uint64_t variants = s.u(row, s.col("variant"));
    // A split card's second half is named by its own name, not as a back face.
    auto self = f.by_handle.find(h);
    const int back = f.st.enum_value("cast_variant", "BACK_FACE");
    if (back >= 0 && self != f.by_handle.end() && is_split_card(f, f.obj.card(self->second, f.obj.col("card"))))
        variants &= ~(1ull << back);
    const std::string variant = mask_names(f, "cast_variant", variants);
    if (!variant.empty()) extra.push_back(lower_words(variant));
    std::vector<std::string> targets;
    const agents::Table &t = f.starget;
    for (size_t r = 0; r < t.rows(); ++r) {
        if (t.u(r, t.col("stack")) != h) continue;
        if (f.st.enum_name("target_kind", t.u(r, t.col("kind"))) == "PLAYER") {
            targets.push_back(seat_name(f.v, t.u(r, t.col("target_player"))));
        } else if (const uint64_t target = t.u(r, t.col("target"))) {
            targets.push_back(object_ref(f, target));
        } else {
            const size_t l = static_cast<size_t>(t.u(r, t.col("lki")));
            const bool has_lki = !t.none(r, t.col("lki")) && l < f.lki.rows();
            const int cid = has_lki ? f.lki.card(l, f.lki.col("card")) : -1;
            targets.push_back((cid >= 0 ? card_name(f, cid) : std::string("a card")) + " (gone)");
        }
    }
    if (!extra.empty()) line += ", " + join(extra, ", ");
    if (!targets.empty()) line += "  -> " + join(targets, ", ");
    return line;
}

static std::vector<std::string> stack_lines(const Frame &f) {
    std::vector<std::string> out;
    size_t n = 0;
    for (size_t r : f.v.stack_order()) out.push_back("  " + std::to_string(++n) + ". " + stack_item(f, r));
    return out;
}

// The viewer's hand: one line per card with its handle, mana cost and types.
static std::vector<std::string> hand_lines(const Frame &f) {
    const std::vector<size_t> rows = zone_rows(f, f.z_hand, f.me);
    std::vector<std::string> out;
    size_t w_name = 0, w_handle = 0, w_cost = 0;
    for (size_t r : rows) {
        w_name = std::max(w_name, object_name(f, r).size());
        w_handle = std::max(w_handle, ref(f.obj.u(r, f.obj.col("handle"))).size());
        w_cost = std::max(w_cost, card_cost(f, f.obj.card(r, f.obj.col("card"))).size());
    }
    for (size_t r : rows) {
        const int cid = f.obj.card(r, f.obj.col("card"));
        auto it = f.front.find(cid);
        std::string types = it != f.front.end() ? type_line(f, it->second) : "";
        if (is_split_card(f, cid)) types += " // " + type_line(f, f.back.at(cid));
        std::string line = "  " + pad(object_name(f, r), w_name) + "  " +
                           pad(ref(f.obj.u(r, f.obj.col("handle"))), w_handle) + "  " +
                           (w_cost ? pad(card_cost(f, cid), w_cost) + "  " : "") + types;
        while (!line.empty() && line.back() == ' ') line.pop_back();
        out.push_back(line);
    }
    return out;
}

// The object rows of `zone` owned by `owner`, by position (newest first in a graveyard or exile,
// from the top in a library) and then in the frame's order.
static std::vector<size_t> zone_rows(const Frame &f, int zone, uint64_t owner) {
    std::vector<size_t> rows;
    const int c_zone = f.obj.col("zone"), c_owner = f.obj.col("owner"), c_pos = f.obj.col("pos");
    for (size_t r = 0; r < f.obj.rows(); ++r)
        if (static_cast<int>(f.obj.u(r, c_zone)) == zone && f.obj.u(r, c_owner) == owner) rows.push_back(r);
    std::stable_sort(rows.begin(), rows.end(), [&](size_t a, size_t b) {
        const uint64_t pa = f.obj.none(a, c_pos) ? ~0ull : f.obj.u(a, c_pos);
        const uint64_t pb = f.obj.none(b, c_pos) ? ~0ull : f.obj.u(b, c_pos);
        return pa < pb;
    });
    return rows;
}

// "<head> Lightning Bolt, Chain Lightning" (newest first; "(empty)" when there is none), wrapped.
static std::vector<std::string> zone_list(const Frame &f, const std::string &head, int zone, uint64_t owner) {
    std::vector<std::string> names;
    for (size_t r : zone_rows(f, zone, owner)) names.push_back(object_name(f, r));
    if (names.empty()) names.push_back("(empty)");
    return wrap(head, names, "    ");
}

static std::string zone_word(const Frame &f, int zone) {
    if (zone < 0) return "nowhere";
    return lower_words(f.st.enum_name("zone", static_cast<uint64_t>(zone)));
}

// "Player B's hand"; "the battlefield", "the stack".
static std::string zone_of(const Frame &f, int zone, uint64_t owner) {
    if (zone == f.z_battlefield || zone == f.z_stack) return "the " + zone_word(f, zone);
    return possessive(f, owner) + " " + zone_word(f, zone);
}

// The viewer's open reveal and look windows.
static std::vector<std::string> reveal_lines(const Frame &f) {
    std::vector<std::string> out;
    const agents::Table &t = f.reveal;
    for (size_t r = 0; r < t.rows(); ++r) {
        const uint64_t h = t.u(r, t.col("handle"));
        const int zone = static_cast<int>(t.u(r, t.col("zone")));
        std::string where = zone_of(f, zone, t.u(r, t.col("owner")));
        if (!t.none(r, t.col("pos"))) where += ", " + ordinal(t.u(r, t.col("pos")) + 1) + " from the top";
        const uint64_t to = t.u(r, t.col("to"));
        const int cid = t.card(r, t.col("card"));
        const std::string name = cid >= 0 ? card_name(f, cid) : std::string("a card");
        out.push_back("  " + name + " " + ref(h) + " (" + where +
                      "): " + lower_words(f.st.enum_name("window_kind", t.u(r, t.col("window")))) +
                      (to ? ", to " + seat_name(f.v, to) : ", to both players"));
    }
    return out;
}

// Emblems, player effects, delayed triggers and graveyard / exile play permissions.
static std::vector<std::string> effect_lines(const Frame &f) {
    std::vector<std::string> out;
    for (size_t r = 0; r < f.emblem.rows(); ++r)
        out.push_back("  " + seat_name(f.v, f.emblem.u(r, f.emblem.col("controller"))) + ": emblem (" +
                      card_name(f, f.emblem.card(r, f.emblem.col("source_card"))) + ")");
    const agents::Table &e = f.effect;
    for (size_t r = 0; r < e.rows(); ++r) {
        std::string line = "  " + seat_name(f.v, e.u(r, e.col("player"))) + ": " +
                           lower_words(f.st.enum_name("player_effect_kind", e.u(r, e.col("kind"))));
        const std::string colors = mask_names(f, "color", e.u(r, e.col("colors")));
        if (!colors.empty()) line += " " + lower_words(colors);
        const std::string duration = f.st.enum_name("duration", e.u(r, e.col("duration")));
        if (duration != "STATIC") line += " (" + lower_words(duration) + ")";
        const int src = e.card(r, e.col("source_card"));
        if (src >= 0) line += " [" + card_name(f, src) + "]";
        if (!e.text(r, e.col("filter")).empty()) line += " " + e.text(r, e.col("filter"));
        out.push_back(line);
    }
    const agents::Table &d = f.delayed;
    for (size_t r = 0; r < d.rows(); ++r) {
        std::string line = "  " + seat_name(f.v, d.u(r, d.col("controller"))) + ": delayed trigger of " +
                           card_name(f, d.card(r, d.col("creator_card")));
        if (!d.none(r, d.col("fire"))) {
            std::string fire = f.st.enum_name("fire_kind", d.u(r, d.col("fire")));
            if (fire.rfind("FIRE_", 0) == 0) fire = fire.substr(5);
            line += ", at " + lower_words(fire);
        }
        if (f.st.enum_name("delayed_state", d.u(r, d.col("state"))) == "ON_STACK") line += " (on the stack)";
        out.push_back(line);
    }
    const agents::Table &p = f.permission;
    for (size_t r = 0; r < p.rows(); ++r)
        out.push_back("  " + seat_name(f.v, p.u(r, p.col("player"))) + " may play " +
                      object_ref(f, p.u(r, p.col("obj"))) + " (" +
                      lower_words(f.st.enum_name("permission_kind", p.u(r, p.col("kind")))) +
                      (p.u(r, p.col("expires_turn")) ? ", this turn" : "") + ")");
    return out;
}

// The viewer's current 75 (the sideboarding phase): main deck and sideboard counts per card.
static std::vector<std::string> deck_lines(const Frame &f) {
    const agents::Table &d = f.own_deck;
    std::vector<std::string> main, side;
    uint64_t n_main = 0, n_side = 0;
    for (size_t r = 0; r < d.rows(); ++r) {
        if (d.u(r, d.col("player")) != f.me) continue;
        const std::string name = card_name(f, d.card(r, d.col("card")));
        if (const uint64_t m = d.u(r, d.col("main"))) {
            main.push_back(std::to_string(m) + " " + name);
            n_main += m;
        }
        if (const uint64_t s = d.u(r, d.col("side"))) {
            side.push_back(std::to_string(s) + " " + name);
            n_side += s;
        }
    }
    if (main.empty() && side.empty()) return {};
    std::vector<std::string> out = wrap("Your deck (" + std::to_string(n_main) + "):", main, "    ");
    for (const std::string &l : wrap("Your sideboard (" + std::to_string(n_side) + "):", side, "    "))
        out.push_back(l);
    return out;
}

// A decision about the opening hand (keep / mulligan / bottoming).
static bool opening_hand_decision(const Frame &f) {
    for (size_t i = 0; i < f.v.n(); ++i) {
        const std::string &c = f.v.category(i);
        if (c == "KEEP_HAND" || c == "MULLIGAN" || c == "BOTTOM_DECK_CARD") return true;
    }
    return false;
}

std::string board_header(const agents::DecisionView &v) {
    const Frame f(v);
    const uint64_t game = f.header.rows() ? f.header.u(0, f.header.col("game")) + 1 : 1;
    const std::string g = "Game " + std::to_string(game);
    const uint64_t first = f.match.rows() ? f.match.u(0, f.match.col("plays_first")) : 0;
    if ((f.header.rows() && f.header.u(0, f.header.col("sideboarding"))) || f.turn.rows() == 0)
        return g + ", sideboarding" + (first ? " (" + seat_name(v, first) + " plays first)" : "");
    if (opening_hand_decision(f))
        return g + ", opening hand" + (first ? " (" + seat_name(v, first) + " plays first)" : "");
    const agents::Table &t = f.turn;
    std::string out = g + ", turn " + std::to_string(t.u(0, t.col("turn")) + 1) + ", " +
                      humanize(f.st.enum_name("step", t.u(0, t.col("step")))) + ": " +
                      possessive(f, t.u(0, t.col("active"))) + " turn";
    if (t.u(0, t.col("priority_window"))) out += "; " + seat_name(v, t.u(0, t.col("priority"))) + " has priority";
    return out;
}

std::vector<std::string> board_lines(const agents::DecisionView &v) {
    const Frame f(v);
    const bool sideboarding = (f.header.rows() && f.header.u(0, f.header.col("sideboarding"))) || f.turn.rows() == 0;
    std::vector<std::string> out = player_lines(f, sideboarding);
    if (sideboarding) {
        for (const std::string &l : deck_lines(f)) out.push_back(l);
        return out;
    }
    for (uint64_t seat : {f.opp, f.me}) {
        out.push_back(possessive(f, seat) + " battlefield:");
        for (const std::string &l : battlefield_lines(f, seat)) out.push_back(l);
    }
    const std::vector<std::string> stack = stack_lines(f);
    if (!stack.empty()) {
        out.push_back("Stack (top first):");
        for (const std::string &l : stack) out.push_back(l);
    }
    const std::vector<std::string> hand = hand_lines(f);
    out.push_back("Your hand (" + std::to_string(hand.size()) + "):" + (hand.empty() ? " (empty)" : ""));
    for (const std::string &l : hand) out.push_back(l);
    for (uint64_t seat : {f.me, f.opp})
        for (const std::string &l : zone_list(f, possessive(f, seat) + " graveyard:", f.z_graveyard, seat))
            out.push_back(l);
    for (uint64_t seat : {f.me, f.opp})
        if (!zone_rows(f, f.z_exile, seat).empty())
            for (const std::string &l : zone_list(f, possessive(f, seat) + " exile:", f.z_exile, seat))
                out.push_back(l);
    const std::vector<std::string> reveals = reveal_lines(f);
    if (!reveals.empty()) {
        out.push_back("Revealed:");
        for (const std::string &l : reveals) out.push_back(l);
    }
    const std::vector<std::string> effects = effect_lines(f);
    if (!effects.empty()) {
        out.push_back("Effects:");
        for (const std::string &l : effects) out.push_back(l);
    }
    for (const std::string &l : knowledge_lines(v, false)) out.push_back(l);
    return out;
}

// ── the pending decision ──────────────────────────────────────────────────────────────────────

// What a menu of `category` asks for.
static std::string category_task(const std::string &category) {
    static const std::map<std::string, std::string> tasks = {
        {"PASS_PRIORITY", "act or pass priority"},
        {"SELECT_ATTACKER", "declare attackers (pick each attacker, then confirm)"},
        {"CONFIRM_ATTACKERS", "declare attackers (pick each attacker, then confirm)"},
        {"SELECT_BLOCKER", "declare blockers (pick each blocker, then confirm)"},
        {"CONFIRM_BLOCKERS", "declare blockers (pick each blocker, then confirm)"},
        {"ACTIVATE_ABILITY", "act or pass priority"},
        {"CAST_SPELL", "act or pass priority"},
        {"PLAY_LAND", "act or pass priority"},
        {"SELECT_TARGET", "choose a target"},
        {"MULLIGAN", "keep this hand or take a mulligan"},
        {"KEEP_HAND", "keep this hand or take a mulligan"},
        {"BOTTOM_DECK_CARD", "put a card from your hand on the bottom of your library"},
        {"MANA_W", "pay the cost"},
        {"MANA_U", "pay the cost"},
        {"MANA_B", "pay the cost"},
        {"MANA_R", "pay the cost"},
        {"MANA_G", "pay the cost"},
        {"MANA_C", "pay the cost"},
        {"PAYING_COSTS", "pay the cost"},
        {"SEARCH_LIBRARY", "choose a card from the search"},
        {"TOP_LIBRARY", "choose a card to put on top of the library"},
        {"SHUFFLE", "shuffle or not"},
        {"DONT_SHUFFLE", "shuffle or not"},
        {"DIG_CHOICE", "choose a card"},
        {"SIDEBOARD_IN", "sideboard: swap cards in and out, then done"},
        {"SIDEBOARD_OUT", "sideboard: swap cards in and out, then done"},
        {"SIDEBOARD_DONE", "sideboard: swap cards in and out, then done"},
        {"SACRIFICE_PERMANENT", "choose a permanent to sacrifice"},
        {"RETURN_PERMANENT", "choose a permanent to return to its owner's hand"},
        {"CHOOSE_X", "choose X"},
        {"DISCARD", "choose a card to discard"},
        {"CHOOSE_MODE", "choose a mode"},
        {"CHOOSE_MANA_COLOR", "choose a mana color"},
        {"PAY_UNLESS", "pay or decline"},
        {"NAME_CARD", "name a card"},
        {"CHOOSE_TYPE", "choose a creature type"},
        {"KEEP_LEGEND", "choose which legendary permanent to keep"},
        {"ORDER_TRIGGERS", "choose the trigger to put on the stack next"},
        {"CHOOSE_REPLACEMENT", "choose a replacement effect"},
        {"ATTACK_TARGET", "choose what the creature attacks"},
        {"BLOCK_TARGET", "choose which attacker the creature blocks"},
        {"OPTIONAL_YESNO", "yes or no"},
        {"SYLVAN_CHOICE", "choose for Sylvan Library"},
        {"CHOOSE_CARD", "choose a card"},
        {"ASSIGN_DAMAGE", "assign combat damage"},
        {"COMPANION", "put your companion into your hand, or not"},
        {"EXILE_FROM_YARD", "choose a card to exile from your graveyard"},
    };
    auto it = tasks.find(category);
    return it != tasks.end() ? it->second : lower_words(category);
}

// What the paused flow is doing ("casting Force of Will"), from the pending table; "" when none.
static std::string pending_context(const Frame &f) {
    if (f.pending.rows() == 0) return "";
    const agents::Table &p = f.pending;
    const std::string tag = f.st.enum_name("pending_tag", p.u(0, p.col("tag")));
    // The source object as it is now (the half of a split card being cast) when the viewer sees its
    // card, else the source's card.
    auto obj = f.by_handle.find(p.u(0, p.col("source")));
    const std::string src = obj != f.by_handle.end() && f.v.object_name(obj->second)
                                ? object_name(f, obj->second)
                                : card_name(f, p.card(0, p.col("source_card")));
    const std::string of = src.empty() ? "" : " " + src;
    if (tag == "CAST") return "casting" + of;
    if (tag == "ACTIVATION") return "activating" + of;
    if (tag == "RESOLUTION") return src.empty() ? "resolving" : src + " resolving";
    if (tag == "TRIGGER_PLACE") return "putting triggers on the stack" + (src.empty() ? "" : " (" + src + ")");
    if (tag == "TURN_DRAW") return "drawing for the turn";
    return src;
}

std::vector<std::string> pending_lines(const agents::DecisionView &v) {
    const Frame f(v);
    std::vector<std::string> out;
    std::string task;
    bool priority = false;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &c = v.category(i);
        if (c == "PASS_PRIORITY" || c == "CAST_SPELL" || c == "PLAY_LAND" || c == "ACTIVATE_ABILITY" ||
            c.rfind("MANA_", 0) == 0)
            priority = priority || c == "PASS_PRIORITY";
        else if (task.empty())
            task = category_task(c);
    }
    if (task.empty() && v.n() > 0) task = category_task(v.category(0));
    if (priority && f.turn.rows() && f.turn.u(0, f.turn.col("priority_window"))) task = "act or pass priority";
    if (f.turn.rows()) {
        const std::string mandatory = f.st.enum_name("mandatory_choice", f.turn.u(0, f.turn.col("mandatory")));
        if (mandatory == "CLEANUP_DISCARD") task = "discard down to your maximum hand size";
    }
    const std::string context = pending_context(f);
    out.push_back("Decision: " + (context.empty() ? task : context + ": " + task));
    for (size_t r = 0; r < f.cost.rows(); ++r) out.push_back("Cost: " + cost_text(f, r));
    return out;
}

// One pending_cost row: what it pays for, which cost, what is still owed, what is paid.
static std::string cost_text(const Frame &f, size_t row) {
    const agents::Table &c = f.cost;
    std::string pips;
    for (uint64_t i = 0; i < c.u(row, c.col("rem_x")); ++i) pips += "{X}";
    if (const uint64_t g = c.u(row, c.col("rem_generic"))) pips += "{" + std::to_string(g) + "}";
    static const char *colors[] = {"w", "u", "b", "r", "g", "c"};
    for (const char *col : colors) {
        const uint64_t n = c.u(row, c.col((std::string("rem_") + col).c_str()));
        for (uint64_t i = 0; i < n; ++i) pips += std::string("{") + static_cast<char>(std::toupper(col[0])) + "}";
    }
    const agents::Table &p = f.pip;
    static const std::map<std::string, char> letter = {
        {"WHITE", 'W'}, {"BLUE", 'U'}, {"BLACK", 'B'}, {"RED", 'R'}, {"GREEN", 'G'}, {"COLORLESS", 'C'}};
    for (size_t r = 0; r < p.rows(); ++r) {
        if (p.u(r, p.col("cost")) != row) continue;
        const std::string kind = f.st.enum_name("pip_kind", p.u(r, p.col("kind")));
        std::vector<std::string> cs;
        const std::vector<std::string> &values = f.st.enum_values("color");
        for (size_t i = 0; i < values.size(); ++i)
            if (p.u(r, p.col("colors")) & (1ull << i)) {
                auto it = letter.find(values[i]);
                if (it != letter.end()) cs.push_back(std::string(1, it->second));
            }
        if (kind == "SNOW")
            pips += "{S}";
        else if (kind == "PHYREXIAN")
            pips += "{" + join(cs, "/") + "/P}";
        else if (kind == "TWOBRID")
            pips += "{" + std::to_string(p.u(r, p.col("generic_alt"))) + "/" + join(cs, "/") + "}";
        else
            pips += "{" + join(cs, "/") + "}";
    }
    std::vector<std::string> owed;
    if (!pips.empty()) owed.push_back(pips);
    if (const uint64_t n = c.u(row, c.col("rem_life"))) owed.push_back(std::to_string(n) + " life");
    if (const uint64_t n = c.u(row, c.col("rem_energy"))) owed.push_back(std::to_string(n) + " energy");
    if (const uint64_t n = c.u(row, c.col("rem_discard"))) owed.push_back("discard " + plural(n, "card", "cards"));
    const uint64_t paid_for = c.u(row, c.col("for"));
    std::string what = paid_for ? object_ref(f, paid_for) : card_name(f, c.card(row, c.col("for_card")));
    std::string line = what + " (" + lower_words(f.st.enum_name("cost_part", c.u(row, c.col("part")))) +
                       " cost): " + (owed.empty() ? "nothing left to pay" : join(owed, ", ") + " left to pay");
    if (const uint64_t paid = c.u(row, c.col("paid"))) line += "; " + std::to_string(paid) + " mana paid";
    return line;
}

std::vector<std::string> menu_lines(const agents::DecisionView &v, bool annotate) {
    std::vector<std::string> out;
    const size_t width = annotate ? std::to_string(v.n() ? v.n() - 1 : 0).size() : 0;
    for (size_t i = 0; i < v.n(); ++i) {
        std::string idx = std::to_string(i);
        if (idx.size() < width) idx = std::string(width - idx.size(), ' ') + idx;
        std::string line = idx + ": " + v.description(i);
        const std::string &c = v.category(i);
        if (annotate && i + 1 == v.n() && (c == "CONFIRM_ATTACKERS" || c == "CONFIRM_BLOCKERS")) line += "   (or -1)";
        out.push_back(line);
    }
    return out;
}

// ── cards ─────────────────────────────────────────────────────────────────────────────────────

// A card's printed mana cost ("{1}{U}"; a split card's halves' costs, "{R} // {2}{R}"); "" for a
// land, a card the frame does not list or none.
static std::string card_cost(const Frame &f, int cid) {
    auto it = f.front.find(cid);
    if (it == f.front.end()) return "";
    const std::string cost = f.card.text(it->second, f.card.col("mana_cost"));
    if (!is_split_card(f, cid)) return cost;
    return cost + " // " + f.card.text(f.back.at(cid), f.card.col("mana_cost"));
}

// "Legendary Creature - Human Wizard" for a CARDS `card` row.
static std::string type_line(const Frame &f, size_t card_row) {
    const agents::Table &c = f.card;
    std::string line = mask_names(f, "supertype", c.u(card_row, c.col("supertypes")));
    const std::string types = mask_names(f, "card_type", c.u(card_row, c.col("types")));
    line += (line.empty() || types.empty() ? "" : " ") + types;
    if (c.u(card_row, c.col("token"))) line += (line.empty() ? "" : " ") + std::string("Token");
    const int cid = c.card(card_row, c.col("card"));
    const uint64_t face = c.u(card_row, c.col("face"));
    std::vector<std::string> subs;
    const agents::Table &s = f.card_subtype;
    for (size_t r = 0; r < s.rows(); ++r)
        if (s.card(r, s.col("card")) == cid && s.u(r, s.col("face")) == face)
            subs.push_back(f.st.enum_name("subtype", s.u(r, s.col("subtype"))));
    if (!subs.empty()) line += " - " + join(subs, " ");
    return line;
}

// One face of a card: name and cost, type line, P/T or loyalty, oracle text.
static std::vector<std::string> card_face_lines(const Frame &f, size_t card_row) {
    const agents::Table &c = f.card;
    std::vector<std::string> out;
    const std::string cost = c.text(card_row, c.col("mana_cost"));
    out.push_back(c.text(card_row, c.col("name")) + (cost.empty() ? "" : "  " + cost));
    std::string types = type_line(f, card_row);
    if (c.u(card_row, c.col("types")) & f.creature_bit)
        types += "  " + std::to_string(c.s(card_row, c.col("power"))) + "/" +
                 std::to_string(c.s(card_row, c.col("toughness")));
    if (const uint64_t loyalty = c.u(card_row, c.col("loyalty"))) types += "  loyalty " + std::to_string(loyalty);
    out.push_back("  " + types);
    const std::string &oracle = c.text(card_row, c.col("oracle"));
    size_t start = 0;
    while (start < oracle.size()) {
        size_t nl = oracle.find('\n', start);
        if (nl == std::string::npos) nl = oracle.size();
        if (nl > start) out.push_back("  " + oracle.substr(start, nl - start));
        start = nl + 1;
    }
    return out;
}

// The card id an object row shows (a copy's copied card), -1 when hidden.
static int object_card_id(const Frame &f, size_t row) {
    const int copy = f.obj.card(row, f.obj.col("copy_of"));
    return copy >= 0 ? copy : f.obj.card(row, f.obj.col("card"));
}

std::vector<std::string> card_lines(const agents::DecisionView &v, const std::string &query) {
    const Frame f(v);
    const std::string q = trim(query);
    if (q.empty()) return {"card: give a handle from the board (card 12) or a card name (card bolt)"};
    std::vector<std::string> out;
    int cid = -1;
    const std::string digits = q[0] == '#' ? q.substr(1) : q;
    if (!digits.empty() && digits.find_first_not_of("0123456789") == std::string::npos) {
        const uint64_t h = std::strtoull(digits.c_str(), nullptr, 10);
        auto it = f.by_handle.find(h);
        if (it == f.by_handle.end()) return {"card: no object " + ref(h) + " in your view"};
        const size_t row = it->second;
        const int zone = static_cast<int>(f.obj.u(row, f.obj.col("zone")));
        std::string head =
            object_name(f, row) + " " + ref(h) + ", " + zone_of(f, zone, f.obj.u(row, f.obj.col("owner")));
        if (zone == f.z_battlefield || zone == f.z_stack)
            head += ", controlled by " + seat_name(v, f.obj.u(row, f.obj.col("controller")));
        out.push_back(head);
        if (zone == f.z_battlefield) {
            std::string now = power_toughness(f, row);
            const std::vector<std::string> status = object_status(f, row);
            if (!status.empty()) now += (now.empty() ? "" : ", ") + join(status, ", ");
            std::vector<std::string> kws;
            for (size_t r = 0; r < f.keyword.rows(); ++r)
                if (f.keyword.u(r, f.keyword.col("obj")) == h) kws.push_back(f.keyword.text(r, f.keyword.col("text")));
            if (!kws.empty()) now += (now.empty() ? "" : "; ") + join(kws, ", ");
            if (!now.empty()) out.push_back("  now: " + now);
        }
        cid = object_card_id(f, row);
        if (cid < 0) {
            out.push_back("  (its card is hidden from you)");
            return out;
        }
    } else {
        // Either face's name matches; the card is its physical (front-face) id.
        std::vector<int> exact, partial;
        const std::string key = name_key(q);
        for (size_t r = 0; r < f.card.rows(); ++r) {
            const std::string name = name_key(f.card.text(r, f.card.col("name")));
            const int card = f.card.card(r, f.card.col("card"));
            if (name == key)
                exact.push_back(card);
            else if (name.find(key) != std::string::npos)
                partial.push_back(card);
        }
        std::vector<int> &hits = exact.empty() ? partial : exact;
        std::sort(hits.begin(), hits.end());
        hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
        if (hits.empty()) return {"card: no card named like '" + q + "' in this match"};
        if (hits.size() > 1) {
            std::vector<std::string> names;
            for (int h : hits) names.push_back(card_name(f, h));
            return {"card: '" + q + "' matches " + join(names, ", ")};
        }
        cid = hits[0];
    }
    auto front = f.front.find(cid);
    if (front == f.front.end()) {
        out.push_back("  (card " + std::to_string(cid) + " is not in the CARDS frame)");
        return out;
    }
    for (const std::string &l : card_face_lines(f, front->second)) out.push_back(l);
    auto back = f.back.find(cid);
    if (back != f.back.end()) {
        out.push_back(is_split_card(f, cid) ? "// second half:" : "// back face:");
        for (const std::string &l : card_face_lines(f, back->second)) out.push_back(l);
    }
    return out;
}

// ── knowledge ─────────────────────────────────────────────────────────────────────────────────

// A library position of table `t` (columns <prefix>_anchor, _pos, _span) in words: "top",
// "3rd from the top", "within the top 1-3", "bottom"; "" for an unknown position.
static std::string libpos(const Frame &f, const agents::Table &t, size_t row, const std::string &prefix) {
    const int c_anchor = t.col((prefix + "_anchor").c_str());
    if (t.none(row, c_anchor)) return "";
    const std::string end = lower_words(f.st.enum_name("lib_anchor", t.u(row, c_anchor)));
    const uint64_t pos = t.u(row, t.col((prefix + "_pos").c_str()));
    const uint64_t span = t.u(row, t.col((prefix + "_span").c_str()));
    if (span == 0) return "";
    if (span == 1) return pos == 0 ? end : ordinal(pos + 1) + " from the " + end;
    return "within the " + end + " " + std::to_string(pos + 1) + "-" + std::to_string(pos + span);
}

// "Player B's hand (5 cards)", "Player A's library, within the top 1-3 (3 cards)"; without
// `owner_name` the zone alone ("library, within the top 1-3 (3 cards)").
static std::string pool_label(const Frame &f, size_t pool, bool owner_name) {
    const agents::Table &p = f.k_pool;
    const uint64_t owner = p.u(pool, p.col("owner"));
    const std::string kind = f.st.enum_name("pool_kind", p.u(pool, p.col("kind")));
    const uint64_t size = p.u(pool, p.col("size"));
    std::string where = owner_name ? possessive(f, owner) + " " : "";
    if (kind == "HAND")
        where += "hand";
    else if (kind == "SIDEBOARD")
        where += "sideboard";
    else if (kind == "FACE_DOWN")
        where += "face-down card " + ref(p.u(pool, p.col("obj")));
    else if (kind == "SET_ASIDE")
        where += "set-aside cards";
    else {
        where += "library";
        const std::string range = libpos(f, p, pool, "range");
        const uint64_t lib = [&] {
            for (size_t r = 0; r < f.player.rows(); ++r)
                if (f.player.u(r, f.player.col("player")) == owner) return f.player.u(r, f.player.col("library"));
            return uint64_t(0);
        }();
        if (!range.empty() && size != lib) where += ", " + range;
    }
    return where + " (" + plural(size, "card", "cards") + ")";
}

// A pool's cards with their copy bounds ("Force of Will 1", "Daze 0-2"), by name; with
// `candidates` only the names.
static std::vector<std::string> pool_cards(const Frame &f, size_t pool, bool candidates) {
    std::vector<std::pair<std::string, std::string>> items;
    const agents::Table &c = f.k_pool_card;
    for (size_t r = 0; r < c.rows(); ++r) {
        if (c.u(r, c.col("pool")) != pool) continue;
        const uint64_t lo = c.u(r, c.col("min")), hi = c.u(r, c.col("max"));
        const std::string name = card_name(f, c.card(r, c.col("card")));
        items.emplace_back(name,
            candidates ? name
                       : name + " " + (lo == hi ? std::to_string(lo) : std::to_string(lo) + "-" + std::to_string(hi)));
    }
    std::sort(items.begin(), items.end());
    std::vector<std::string> out;
    for (auto &i : items) out.push_back(i.second);
    return out;
}

// The cards known to be in the opponent's hand ("Force of Will", "Daze x2").
static std::vector<std::string> known_hand(const Frame &f) {
    std::vector<std::string> out;
    const agents::Table &k = f.k_known_hand;
    for (size_t r = 0; r < k.rows(); ++r) {
        const uint64_t n = k.u(r, k.col("copies"));
        out.push_back(card_name(f, k.card(r, k.col("card"))) + (n > 1 ? " x" + std::to_string(n) : ""));
    }
    return out;
}

// The known library positions of `owner` ("top: Brainstorm", "2nd from the top: one of Daze,
// Ponder", "bottom: Force of Will": a position in the library's lower half is counted from the
// bottom), from the top, at most `limit` (0 = all).
static std::vector<std::string> known_positions(const Frame &f, uint64_t owner, size_t limit) {
    std::vector<std::pair<uint64_t, std::string>> rows;
    const agents::Table &k = f.k_position;
    uint64_t size = 0;
    for (size_t r = 0; r < f.player.rows(); ++r)
        if (f.player.u(r, f.player.col("player")) == owner) size = f.player.u(r, f.player.col("library"));
    for (size_t r = 0; r < k.rows(); ++r) {
        if (k.u(r, k.col("owner")) != owner) continue;
        const int cid = k.card(r, k.col("card"));
        std::string what;
        if (cid >= 0) {
            what = card_name(f, cid);
        } else if (!k.none(r, k.col("pool"))) {
            const std::vector<std::string> names = pool_cards(f, static_cast<size_t>(k.u(r, k.col("pool"))), true);
            what =
                names.size() <= 6 ? "one of " + join(names, ", ") : "one of " + std::to_string(names.size()) + " cards";
        }
        const uint64_t pos = k.u(r, k.col("at_pos"));
        const bool top = f.st.enum_name("lib_anchor", k.u(r, k.col("at_anchor"))) == "TOP";
        std::string at = libpos(f, k, r, "at");
        if (top && size > 0 && pos < size && pos >= (size + 1) / 2)
            at = pos + 1 == size ? "bottom" : ordinal(size - pos) + " from the bottom";
        rows.emplace_back(pos, at + ": " + what);
    }
    std::sort(rows.begin(), rows.end());
    std::vector<std::string> out;
    for (auto &r : rows) {
        if (limit && out.size() == limit) {
            out.push_back("...");
            break;
        }
        out.push_back(r.second);
    }
    return out;
}

std::vector<std::string> knowledge_lines(const agents::DecisionView &v, bool full) {
    const Frame f(v);
    const bool off =
        f.k_meta.rows() == 0 || f.st.enum_name("knowledge_level", f.k_meta.u(0, f.k_meta.col("level"))) == "OFF";
    if (off)
        return full ? std::vector<std::string>{"Knowledge: off (no k_* tables in your frames)"}
                    : std::vector<std::string>{};
    std::vector<std::string> out;
    if (!full) {
        std::vector<std::string> facts;
        const std::vector<std::string> hand = known_hand(f);
        if (!hand.empty()) facts.push_back(possessive(f, f.opp) + " hand holds " + join(hand, ", "));
        for (uint64_t seat : {f.opp, f.me}) {
            const std::vector<std::string> pos = known_positions(f, seat, 3);
            if (!pos.empty()) facts.push_back(possessive(f, seat) + " library: " + join(pos, "; "));
        }
        if (facts.empty()) return out;
        out.push_back("You know:");
        for (const std::string &s : facts) out.push_back("  " + s);
        return out;
    }
    const agents::Table &meta = f.k_meta;
    out.push_back("Knowledge (" + lower_words(f.st.enum_name("knowledge_level", meta.u(0, meta.col("level")))) + ", " +
                  (meta.u(0, meta.col("exact")) ? "exact" : "bounds only: the solver ran out of steps") + "):");
    for (uint64_t seat : {f.opp, f.me}) {
        out.push_back(you(f, seat) + ":");
        if (seat == f.opp) {
            const std::vector<std::string> hand = known_hand(f);
            out.push_back("  known in hand: " + (hand.empty() ? std::string("nothing") : join(hand, ", ")));
        }
        for (size_t p = 0; p < f.k_pool.rows(); ++p) {
            if (f.k_pool.u(p, f.k_pool.col("owner")) != seat) continue;
            std::vector<std::string> cards = pool_cards(f, p, false);
            if (cards.empty()) cards.push_back("nothing possible");
            for (const std::string &l : wrap("  " + pool_label(f, p, false) + ":", cards, "      ")) out.push_back(l);
        }
        const std::vector<std::string> pos = known_positions(f, seat, 0);
        if (!pos.empty())
            for (const std::string &l : wrap("  known library positions:", pos, "      ")) out.push_back(l);
        std::vector<std::string> seen;
        const agents::Table &s = f.k_seen;
        for (size_t r = 0; r < s.rows(); ++r)
            if (s.u(r, s.col("player")) == seat)
                seen.push_back(
                    card_name(f, s.card(r, s.col("card"))) + " " + std::to_string(s.u(r, s.col("game"))) +
                    (s.u(r, s.col("match")) != s.u(r, s.col("game")) ? "/" + std::to_string(s.u(r, s.col("match")))
                                                                     : ""));
        if (!seen.empty())
            for (const std::string &l : wrap("  copies seen this game (/match):", seen, "      ")) out.push_back(l);
    }
    return out;
}

// ── zones ─────────────────────────────────────────────────────────────────────────────────────

const std::vector<std::string> &zone_names() {
    static const std::vector<std::string> names = {
        "hand", "library", "graveyard", "exile", "battlefield", "stack", "sideboard"};
    return names;
}

// A library as the viewer sees it: its size, cards the frame shows in it (open windows), known
// positions, and the knowledge pools' contents.
static std::vector<std::string> library_zone(const Frame &f, uint64_t owner) {
    std::vector<std::string> out;
    uint64_t size = 0;
    for (size_t r = 0; r < f.player.rows(); ++r)
        if (f.player.u(r, f.player.col("player")) == owner) size = f.player.u(r, f.player.col("library"));
    out.push_back(possessive(f, owner) + " library: " + plural(size, "card", "cards") + ", order hidden");
    for (size_t r : zone_rows(f, f.z_library, owner)) {
        const agents::Table &o = f.obj;
        out.push_back(
            "  " +
            (o.none(r, o.col("pos")) ? std::string("somewhere") : ordinal(o.u(r, o.col("pos")) + 1) + " from the top") +
            ": " + object_name(f, r) + " " + ref(o.u(r, o.col("handle"))) + " (you see it now)");
    }
    const std::vector<std::string> pos = known_positions(f, owner, 0);
    for (const std::string &p : pos) out.push_back("  known " + p);
    for (size_t p = 0; p < f.k_pool.rows(); ++p) {
        if (f.k_pool.u(p, f.k_pool.col("owner")) != owner ||
            f.st.enum_name("pool_kind", f.k_pool.u(p, f.k_pool.col("kind"))) != "LIBRARY_RANGE")
            continue;
        for (const std::string &l :
            wrap("  contents, " + pool_label(f, p, false) + ":", pool_cards(f, p, false), "      "))
            out.push_back(l);
    }
    return out;
}

std::vector<std::string> zone_lines(const agents::DecisionView &v, const std::string &zone, bool opponent) {
    const Frame f(v);
    const uint64_t owner = opponent ? f.opp : f.me;
    const std::string z = ascii_lower(zone);
    std::vector<std::string> out;
    if (z == "battlefield") {
        out.push_back(possessive(f, owner) + " battlefield:");
        for (const std::string &l : battlefield_lines(f, owner)) out.push_back(l);
        return out;
    }
    if (z == "stack") {
        const std::vector<std::string> stack = stack_lines(f);
        out.push_back(stack.empty() ? "The stack is empty." : "Stack (top first):");
        for (const std::string &l : stack) out.push_back(l);
        return out;
    }
    if (z == "library") return library_zone(f, owner);
    int zv = -1;
    if (z == "hand")
        zv = f.z_hand;
    else if (z == "graveyard")
        zv = f.z_graveyard;
    else if (z == "exile")
        zv = f.z_exile;
    else if (z == "sideboard")
        zv = f.z_sideboard;
    if (zv < 0) return {"zone: unknown zone '" + zone + "' (" + join(zone_names(), ", ") + ")"};
    const std::vector<size_t> rows = zone_rows(f, zv, owner);
    uint64_t size = rows.size();
    const char *count_col = zv == f.z_hand        ? "hand"
                            : zv == f.z_graveyard ? "graveyard"
                            : zv == f.z_exile     ? "exile"
                                                  : "sideboard";
    for (size_t r = 0; r < f.player.rows(); ++r)
        if (f.player.u(r, f.player.col("player")) == owner) size = f.player.u(r, f.player.col(count_col));
    std::string head = possessive(f, owner) + " " + z + ": " + plural(size, "card", "cards");
    if (rows.size() < size) head += rows.empty() ? ", hidden" : ", " + std::to_string(rows.size()) + " visible";
    if (zv == f.z_graveyard || zv == f.z_exile) head += rows.size() > 1 ? " (newest first)" : "";
    out.push_back(head);
    if (zv == f.z_hand && owner == f.me) {
        for (const std::string &l : hand_lines(f)) out.push_back(l);
    } else if (zv == f.z_sideboard && owner == f.me) {
        std::map<std::string, size_t> counts;
        for (size_t r : rows) ++counts[object_name(f, r)];
        for (const auto &c : counts)
            out.push_back("  " + c.first + (c.second > 1 ? " x" + std::to_string(c.second) : ""));
    } else {
        for (size_t r : rows) {
            std::string line = "  " + object_name(f, r) + " " + ref(f.obj.u(r, f.obj.col("handle")));
            const std::string counters = object_counters(f, f.obj.u(r, f.obj.col("handle")));
            if (!counters.empty()) line += "  " + counters;
            out.push_back(line);
        }
    }
    if (zv == f.z_hand && owner == f.opp) {
        const std::vector<std::string> hand = known_hand(f);
        if (!hand.empty()) out.push_back("  you know it holds: " + join(hand, ", "));
    }
    if (zv == f.z_hand || zv == f.z_sideboard)
        for (size_t p = 0; p < f.k_pool.rows(); ++p) {
            if (f.k_pool.u(p, f.k_pool.col("owner")) != owner) continue;
            const std::string kind = f.st.enum_name("pool_kind", f.k_pool.u(p, f.k_pool.col("kind")));
            if ((zv == f.z_hand && kind == "HAND") || (zv == f.z_sideboard && kind == "SIDEBOARD"))
                for (const std::string &l : wrap("  copies it can hold:", pool_cards(f, p, false), "      "))
                    out.push_back(l);
        }
    return out;
}

// ── events ────────────────────────────────────────────────────────────────────────────────────

// The names of an event's identified cards (", "-joined); `hidden` receives how many it does not
// identify.
static std::string event_cards(const Frame &f, const std::vector<size_t> &rows, uint64_t *hidden) {
    std::vector<std::string> names;
    *hidden = 0;
    const agents::Table &c = f.event_card;
    for (size_t r : rows) {
        const int cid = c.card(r, c.col("card"));
        if (cid < 0) {
            ++*hidden;
            continue;
        }
        // A card the event put on the stack is named as the spell it is there (a split card's half).
        auto after = f.by_handle.find(c.u(r, c.col("handle_after")));
        const bool on_stack =
            after != f.by_handle.end() && static_cast<int>(f.obj.u(after->second, f.obj.col("zone"))) == f.z_stack;
        names.push_back((on_stack ? object_name(f, after->second) : card_name(f, cid)) +
                        (c.u(r, c.col("token")) ? " token" : "") + (c.u(r, c.col("face_down")) ? " (face down)" : ""));
    }
    return join(names, ", ");
}

// "Mountain, Lightning Bolt and 2 cards"; with `article`, one unidentified card alone is "a card".
static std::string items_text(const std::vector<std::string> &names, uint64_t hidden, bool article) {
    std::string out = join(names, ", ");
    if (hidden == 0) return out;
    const std::string rest =
        hidden == 1 ? (article || !out.empty() ? "a card" : "card") : plural(hidden, "card", "cards");
    return out.empty() ? rest : out + " and " + rest;
}

// One event as a line ("" for one that has none).
static EventLine event_line(
    const Frame &f, size_t row, const std::vector<size_t> &cards, const std::vector<size_t> &players) {
    const agents::Table &e = f.event;
    const std::string kind = f.st.enum_name("event_kind", e.u(row, e.col("kind")));
    const uint64_t who = e.u(row, e.col("player"));
    const int zone = e.none(row, e.col("zone")) ? -1 : static_cast<int>(e.u(row, e.col("zone")));
    const int to_zone = e.none(row, e.col("to_zone")) ? -1 : static_cast<int>(e.u(row, e.col("to_zone")));
    const uint64_t owner = e.u(row, e.col("zone_owner")), to_owner = e.u(row, e.col("to_owner"));
    const uint64_t game = e.u(row, e.col("game")) + 1;
    uint64_t hidden = 0;
    const std::string known = event_cards(f, cards, &hidden);
    std::vector<std::string> names;
    if (!known.empty()) names.push_back(known);
    const std::string from_pos = cards.empty() ? "" : libpos(f, f.event_card, cards[0], "from");
    const std::string to_pos = cards.empty() ? "" : libpos(f, f.event_card, cards[0], "to");
    if (kind == "GAME_START")
        return {"Game " + std::to_string(game) + " begins; " + seat_name(f.v, who) + " plays first", true};
    if (kind == "GAME_END")
        return {
            "Game " + std::to_string(game) + " ends: " + (who ? seat_name(f.v, who) + " wins" : std::string("a draw")),
            true};
    if (kind == "MATCH_END") {
        std::vector<std::string> wins;
        for (size_t r : players) wins.push_back(std::to_string(f.event_player.u(r, f.event_player.col("wins"))));
        return {"The match ends: " + (who ? seat_name(f.v, who) + " wins" : std::string("a draw")) +
                    (wins.empty() ? "" : " " + join(wins, "-")),
            true};
    }
    if (kind == "REVEAL_END") return {"", false};
    if (kind == "SHUFFLE") return {possessive(f, owner) + " library is shuffled", false};
    if (kind == "CAST_FROM_HIDDEN")
        return {
            seat_name(f.v, owner) + " casts " + items_text(names, hidden, true) + " from " + zone_word(f, zone), false};
    if (kind == "LOOK" || kind == "REVEAL_START") {
        std::string where = zone_of(f, zone, owner);
        const std::string range = libpos(f, e, row, "range");
        if (!range.empty()) where += " (" + range + ")";
        const uint64_t to = e.u(row, e.col("to"));
        if (kind == "LOOK")
            return {seat_name(f.v, who) + " looks at " + plural(e.u(row, e.col("count")), "card", "cards") + " of " +
                        where + (known.empty() ? "" : ": " + known),
                false};
        return {
            "Revealed from " + where + (to ? " to " + seat_name(f.v, to) : "") + ": " + items_text(names, hidden, true),
            false};
    }
    if (kind == "SEARCH") {
        const uint64_t found = e.u(row, e.col("count"));
        std::string what = found == 0 ? std::string("nothing") : items_text(names, hidden, true);
        if (found > 0 && names.empty() && hidden == 0) what = plural(found, "card", "cards");
        return {seat_name(f.v, who) + " searches " + zone_of(f, zone, owner) + ": finds " + what, false};
    }
    if (kind == "LIBRARY_PLACE") {
        const std::string range = libpos(f, e, row, "range");
        return {plural(e.u(row, e.col("count")), "card goes", "cards go") + " into " +
                    zone_of(f, to_zone >= 0 ? to_zone : zone, to_owner ? to_owner : owner) +
                    (range.empty() ? "" : ", " + range) + (e.u(row, e.col("order_known")) ? "" : ", order unknown"),
            false};
    }
    if (kind == "REORDER") return {possessive(f, owner) + " library is reordered", false};
    if (kind == "FACE_DOWN" || kind == "FACE_UP")
        return {items_text(names, hidden, true) + " is turned face " + (kind == "FACE_UP" ? "up" : "down"), false};
    if (kind == "PLAY_FROM_HIDDEN" && hidden == 0 && cards.size() == 1) {
        const int cid = f.event_card.card(cards[0], f.event_card.col("card"));
        auto it = f.front.find(cid);
        const int land = f.st.enum_value("card_type", "Land");
        const bool is_land =
            it != f.front.end() && land >= 0 && (f.card.u(it->second, f.card.col("types")) & (1ull << land));
        if (is_land && zone == f.z_hand) return {seat_name(f.v, owner) + " plays " + known, false};
        return {seat_name(f.v, owner) + " puts " + known + " onto the battlefield from " + zone_word(f, zone), false};
    }
    // ZONE_MOVE, PLAY_FROM_HIDDEN, DRAW (handled by the caller) and anything new: a move.
    const uint64_t subject = zone >= 0 ? owner : to_owner;
    const std::string what = possessive(f, subject) + " " + items_text(names, hidden, false);
    if (zone < 0) return {what + ": created on " + zone_of(f, to_zone, to_owner), false};
    if (to_zone < 0) return {what + ": ceases to exist", false};
    std::string from = zone_word(f, zone);
    if (!from_pos.empty()) from += " (" + from_pos + ")";
    std::string to = to_owner != owner && to_zone != f.z_battlefield && to_zone != f.z_stack
                         ? zone_of(f, to_zone, to_owner)
                         : zone_word(f, to_zone);
    if (!to_pos.empty()) to += " (" + to_pos + ")";
    return {what + ": " + from + " -> " + to, false};
}

std::vector<std::string> event_lines(const agents::DecisionView &v) {
    const Frame f(v);
    const agents::Table &e = f.event;
    const size_t n = e.rows();
    std::vector<std::vector<size_t>> cards(n), players(n);
    for (size_t r = 0; r < f.event_card.rows(); ++r) {
        const uint64_t ev = f.event_card.u(r, f.event_card.col("event"));
        if (ev < n) cards[static_cast<size_t>(ev)].push_back(r);
    }
    for (size_t r = 0; r < f.event_player.rows(); ++r) {
        const uint64_t ev = f.event_player.u(r, f.event_player.col("event"));
        if (ev < n) players[static_cast<size_t>(ev)].push_back(r);
    }
    const int c_kind = e.col("kind"), c_player = e.col("player"), c_turn = e.col("turn"), c_open = e.col("opening"),
              c_src = e.col("source_card");
    std::vector<std::string> out;
    std::string last;
    size_t repeats = 0;
    auto flush = [&]() {
        if (!last.empty()) out.push_back(repeats > 1 ? last + " (x" + std::to_string(repeats) + ")" : last);
        last.clear();
        repeats = 0;
    };
    size_t i = 0;
    while (i < n) {
        const std::string kind = f.st.enum_name("event_kind", e.u(i, c_kind));
        EventLine line;
        const size_t start = i;
        if (kind == "DRAW") {
            std::vector<size_t> drawn;
            uint64_t count = 0;
            size_t j = i;
            while (j < n && f.st.enum_name("event_kind", e.u(j, c_kind)) == "DRAW" &&
                   e.u(j, c_player) == e.u(i, c_player) && e.u(j, c_turn) == e.u(i, c_turn) &&
                   e.u(j, c_open) == e.u(i, c_open) && e.card(j, c_src) == e.card(i, c_src)) {
                count += e.u(j, e.col("count"));
                for (size_t r : cards[j]) drawn.push_back(r);
                ++j;
            }
            uint64_t hidden = 0;
            const std::string known = event_cards(f, drawn, &hidden);
            line.text = seat_name(v, e.u(i, c_player)) + " draws " +
                        (count == 1 ? std::string("a card") : plural(count, "card", "cards")) +
                        (e.u(i, c_open) ? " (opening hand)" : "") + (known.empty() ? "" : ": " + known);
            i = j;
        } else {
            line = event_line(f, i, cards[i], players[i]);
            ++i;
        }
        if (line.text.empty()) continue;
        std::string text = line.text;
        const int src = e.card(start, c_src);
        if (src >= 0 && !line.game_event) text += " (by " + card_name(f, src) + ")";
        if (!line.game_event) text = "T" + std::to_string(e.u(start, c_turn) + 1) + ": " + text;
        if (text == last) {
            ++repeats;
            continue;
        }
        flush();
        last = text;
        repeats = 1;
    }
    flush();
    return out;
}

}  // namespace cli
