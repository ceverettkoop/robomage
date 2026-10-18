#include "trace.h"

#include "../cli/board.h"

namespace agents {

static std::string seat_label(const SeatStream &st, uint64_t seat_value);

static std::string seat_label(const SeatStream &st, uint64_t seat_value) {
    const std::string &name = st.enum_name("seat", seat_value);
    if (name == "PLAYER_A") return "Player A";
    if (name == "PLAYER_B") return "Player B";
    return name;
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

    for (const std::string &line : cli::board_lines(v)) out.push_back("| " + line);
    for (const std::string &line : cli::pending_lines(v)) out.push_back("| " + line);
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
