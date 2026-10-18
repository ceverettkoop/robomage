#include "trace.h"

#include <map>

namespace agents {

static std::string seat_label(const SeatStream &st, uint64_t seat_value);
static std::string join(const std::vector<std::string> &items);
static std::string object_line(
    const DecisionView &v, size_t row, const std::map<uint64_t, std::string> &counters, uint64_t creature_bit);
static std::string zone_names(const DecisionView &v, int zone, uint64_t owner_value);

static std::string seat_label(const SeatStream &st, uint64_t seat_value) {
    const std::string &name = st.enum_name("seat", seat_value);
    if (name == "PLAYER_A") return "Player A";
    if (name == "PLAYER_B") return "Player B";
    return name;
}

static std::string join(const std::vector<std::string> &items) {
    if (items.empty()) return "(none)";
    std::string out;
    for (const std::string &s : items) out += (out.empty() ? "" : ", ") + s;
    return out;
}

// One battlefield object: its name, P/T for a creature, and its statuses and counters.
static std::string object_line(
    const DecisionView &v, size_t row, const std::map<uint64_t, std::string> &counters, uint64_t creature_bit) {
    const Table &o = v.objects();
    const std::optional<std::string> name = v.object_name(row);
    std::string out = name ? *name : "(face-down)";
    if (o.u(row, o.col("types")) & creature_bit)
        out += " " + std::to_string(o.s(row, o.col("power"))) + "/" + std::to_string(o.s(row, o.col("toughness")));
    std::vector<std::string> marks;
    static const char *flags[] = {"tapped", "sick", "attacking", "blocking", "phased_out", "face_down"};
    for (const char *f : flags)
        if (o.u(row, o.col(f))) marks.push_back(f);
    if (const uint64_t dmg = o.u(row, o.col("damage"))) marks.push_back("damage " + std::to_string(dmg));
    if (const uint64_t att = o.u(row, o.col("attached_to"))) marks.push_back("attached to #" + std::to_string(att));
    auto c = counters.find(o.u(row, o.col("handle")));
    if (c != counters.end()) marks.push_back(c->second);
    if (!marks.empty()) {
        out += " (";
        for (size_t i = 0; i < marks.size(); ++i) out += (i ? ", " : "") + marks[i];
        out += ")";
    }
    return out;
}

// The names of the objects in `zone` owned by seat value `owner_value`, in object-table order.
static std::string zone_names(const DecisionView &v, int zone, uint64_t owner_value) {
    const Table &o = v.objects();
    const int c_zone = o.col("zone"), c_owner = o.col("owner");
    std::vector<std::string> names;
    for (size_t r = 0; r < o.rows(); ++r) {
        if (static_cast<int>(o.u(r, c_zone)) != zone || o.u(r, c_owner) != owner_value) continue;
        const std::optional<std::string> name = v.object_name(r);
        names.push_back(name ? *name : "(hidden)");
    }
    return join(names);
}

std::vector<std::string> trace_decision_lines(const DecisionView &v, size_t number) {
    const SeatStream &st = v.stream();
    std::vector<std::string> out;
    const Table header = v.table("header");
    const Table turn = v.table("turn");
    const uint64_t me = static_cast<uint64_t>(st.enum_value("seat", v.seat() == 0 ? "PLAYER_A" : "PLAYER_B"));
    std::string head = "| decision " + std::to_string(number) + ": " + seat_label(st, me) + " (game " +
                       std::to_string(header.u(0, header.col("game")));
    if (header.u(0, header.col("sideboarding")) || turn.rows() == 0)
        head += ", sideboarding)";
    else
        head += ", turn " + std::to_string(turn.u(0, turn.col("turn"))) + ", " +
                st.enum_name("step", turn.u(0, turn.col("step"))) + ", " +
                seat_label(st, turn.u(0, turn.col("active"))) + " active)";
    out.push_back(head);

    const Table players = v.table("player");
    static const char *pool[] = {"w", "u", "b", "r", "g", "c"};
    for (size_t r = 0; r < players.rows(); ++r) {
        std::string line = "| " + seat_label(st, players.u(r, players.col("player"))) + ": life " +
                           std::to_string(players.s(r, players.col("life")));
        for (const char *z : {"hand", "library", "graveyard", "exile"})
            line += std::string(", ") + z + " " + std::to_string(players.u(r, players.col(z)));
        std::string mana;
        for (const char *c : pool) {
            const uint64_t n = players.u(r, players.col((std::string("mana_") + c).c_str()));
            if (n) mana += std::to_string(n) + static_cast<char>(c[0] - 'a' + 'A');
        }
        if (!mana.empty()) line += ", pool " + mana;
        out.push_back(line);
    }

    const Table counters_t = v.table("object_counter");
    std::map<uint64_t, std::string> counters;
    for (size_t r = 0; r < counters_t.rows(); ++r) {
        std::string &s = counters[counters_t.u(r, counters_t.col("obj"))];
        s += (s.empty() ? "" : ", ") + counters_t.text(r, counters_t.col("kind")) + " x" +
             std::to_string(counters_t.u(r, counters_t.col("count")));
    }
    uint64_t creature_bit = 0;
    const std::vector<std::string> &types = st.enum_values("card_type");
    for (size_t i = 0; i < types.size(); ++i)
        if (types[i] == "Creature") creature_bit = 1ull << i;
    const Table &o = v.objects();
    const int z_battlefield = st.enum_value("zone", "BATTLEFIELD");
    const int c_zone = o.col("zone"), c_controller = o.col("controller");
    const uint64_t seats[2] = {static_cast<uint64_t>(st.enum_value("seat", "PLAYER_A")),
        static_cast<uint64_t>(st.enum_value("seat", "PLAYER_B"))};
    for (uint64_t seat : seats) {
        std::vector<std::string> perms;
        for (size_t r = 0; r < o.rows(); ++r)
            if (static_cast<int>(o.u(r, c_zone)) == z_battlefield && o.u(r, c_controller) == seat)
                perms.push_back(object_line(v, r, counters, creature_bit));
        out.push_back("| " + seat_label(st, seat) + " battlefield: " + join(perms));
    }

    const Table &stack = v.stack();
    if (!v.stack_order().empty()) {
        std::vector<std::string> items;
        for (size_t r : v.stack_order()) {
            const std::optional<std::string> name = v.stack_card(r);
            items.push_back((name ? *name : std::string("(ability)")) + " (" +
                            seat_label(st, stack.u(r, stack.col("controller"))) + ")");
        }
        out.push_back("| stack: " + join(items));
    }
    out.push_back("| " + seat_label(st, me) + " hand: " + zone_names(v, st.enum_value("zone", "HAND"), me));
    for (uint64_t seat : seats)
        out.push_back(
            "| " + seat_label(st, seat) + " graveyard: " + zone_names(v, st.enum_value("zone", "GRAVEYARD"), seat));

    out.push_back("| menu:");
    for (size_t i = 0; i < v.n(); ++i) out.push_back("|   " + std::to_string(i) + ": " + v.description(i));
    return out;
}

std::string trace_choice_line(const DecisionView &v, int choice) {
    const SeatStream &st = v.stream();
    std::string line =
        "| >> " +
        seat_label(st, static_cast<uint64_t>(st.enum_value("seat", v.seat() == 0 ? "PLAYER_A" : "PLAYER_B"))) + ": " +
        std::to_string(choice);
    if (choice == -2) return line + " (concede game)";
    if (choice == -3) return line + " (concede match)";
    if (choice >= 0 && static_cast<size_t>(choice) < v.n()) return line + " (" + v.description(choice) + ")";
    return line;
}

}  // namespace agents
