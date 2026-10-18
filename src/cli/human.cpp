#include "human.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "../str_util.h"
#include "board.h"

namespace cli {

static bool parse_int(const std::string &s, long *out);
static const char *zone_alias(const std::string &word);
static int side_word(const std::string &word);
static uint64_t frame_game(const agents::DecisionView &v);
static uint64_t frame_turn(const agents::DecisionView &v, const char *column);

// The news lines shown at a decision; older ones are left to `log`.
static constexpr size_t NEWS_SHOWN = 20;
static constexpr size_t LOG_DEFAULT = 30;

// A whole-string integer ("3", "-1").
static bool parse_int(const std::string &s, long *out) {
    if (s.empty()) return false;
    char *end = nullptr;
    *out = std::strtol(s.c_str(), &end, 10);
    return end != s.c_str() && *end == '\0';
}

// The zone a `zone` argument names (aliases gy, yard, bf, lib, sb), or nullptr.
static const char *zone_alias(const std::string &word) {
    const std::string w = ascii_lower(word);
    if (w == "hand") return "hand";
    if (w == "library" || w == "lib" || w == "deck") return "library";
    if (w == "graveyard" || w == "gy" || w == "yard") return "graveyard";
    if (w == "exile") return "exile";
    if (w == "battlefield" || w == "bf" || w == "board") return "battlefield";
    if (w == "stack") return "stack";
    if (w == "sideboard" || w == "sb") return "sideboard";
    return nullptr;
}

// 1 for a word naming the opponent's side, 0 for one naming the viewer's, -1 for neither.
static int side_word(const std::string &word) {
    const std::string w = ascii_lower(word);
    if (w == "opp" || w == "opponent" || w == "their" || w == "theirs" || w == "them") return 1;
    if (w == "own" || w == "my" || w == "mine" || w == "me" || w == "you" || w == "your") return 0;
    return -1;
}

// The frame's game number (0-based).
static uint64_t frame_game(const agents::DecisionView &v) {
    const agents::Table header = v.table("header");
    return header.rows() ? header.u(0, header.col("game")) : 0;
}

// A column of the frame's turn row (0 when the frame has none: the sideboarding phase).
static uint64_t frame_turn(const agents::DecisionView &v, const char *column) {
    const agents::Table turn = v.table("turn");
    return turn.rows() ? turn.u(0, turn.col(column)) : 0;
}

HumanAgent::HumanAgent(bool banner) : banner_(banner) {}

uint64_t HumanAgent::seat_value(const agents::DecisionView &v, bool opponent) {
    const bool a = (v.seat() == 0) != opponent;
    return static_cast<uint64_t>(v.stream().enum_value("seat", a ? "PLAYER_A" : "PLAYER_B"));
}

std::set<uint64_t> HumanAgent::stack_handles(const agents::DecisionView &v, uint64_t seat) {
    std::set<uint64_t> out;
    const agents::Table &stack = v.stack();
    const int c_handle = stack.col("handle"), c_controller = stack.col("controller");
    for (size_t r = 0; r < stack.rows(); ++r)
        if (seat == 0 || stack.u(r, c_controller) == seat) out.insert(stack.u(r, c_handle));
    return out;
}

int HumanAgent::pass_row(const agents::DecisionView &v) {
    if (v.table("pending").rows() || v.table("pending_cost").rows()) return -1;
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) == "PASS_PRIORITY") return static_cast<int>(i);
    return -1;
}

bool HumanAgent::start_auto_pass(const agents::DecisionView &v, bool mine, int *reply) {
    const int pass = pass_row(v);
    if (pass < 0) {
        std::printf("Auto-pass starts at a priority decision with a Pass priority row.\n");
        return false;
    }
    const uint64_t opp = seat_value(v, true);
    auto_.active = true;
    auto_.mine = mine;
    auto_.game = frame_game(v);
    auto_.turn = frame_turn(v, "turn");
    auto_.opp_stack = stack_handles(v, opp);
    std::printf(
        "Auto-passing until %s turn begins (it stops at any other decision, and when %s puts "
        "something on the stack).\n",
        mine ? "your next" : "the next", seat_name(v, opp).c_str());
    *reply = pass;
    return true;
}

int HumanAgent::continue_auto_pass(const agents::DecisionView &v, std::string *why) {
    const uint64_t me = seat_value(v, false), opp = seat_value(v, true);
    const uint64_t game = frame_game(v), turn = frame_turn(v, "turn");
    const std::set<uint64_t> opp_stack = stack_handles(v, opp);
    int pass = -1;
    if (game != auto_.game)
        *why = "a new game began";
    else if (turn > auto_.turn && (!auto_.mine || frame_turn(v, "active") == me))
        *why = auto_.mine ? "your turn began" : "a new turn began";
    else if ((pass = pass_row(v)) < 0)
        *why = "this decision is not a priority pass";
    else
        for (uint64_t h : opp_stack)
            if (!auto_.opp_stack.count(h)) {
                *why = seat_name(v, opp) + " put something on the stack";
                pass = -1;
                break;
            }
    auto_.opp_stack = opp_stack;
    if (pass < 0) auto_.active = false;
    return pass;
}

void HumanAgent::print(const std::vector<std::string> &lines, const char *indent) {
    for (const std::string &l : lines) std::printf("%s%s\n", indent, l.c_str());
}

void HumanAgent::print_help() {
    std::printf(
        "Commands:\n"
        "  <n>                  choose menu row n (-1 = the last row: confirm attackers/blockers)\n"
        "  concede              concede this game;  concede match: the whole match\n"
        "  board                show the board and the menu again\n"
        "  zone <name> [opp]    one zone, yours or the opponent's: hand, library, graveyard,\n"
        "                       exile, battlefield, stack, sideboard\n"
        "  card <#n|name>       card details: an object of the board by its #handle, or a card\n"
        "                       of this match by name (e.g. card 12, card bolt)\n"
        "  know                 what you know and can deduce (the knowledge panel, in full)\n"
        "  log [n|all]          what you saw happen this match (the last n lines, default %zu)\n"
        "  z [mine]             auto-pass: pass priority until the next turn begins (mine: until\n"
        "                       your next turn); it stops at any other decision and when the\n"
        "                       opponent puts something on the stack  (also: pass turn [mine])\n"
        "  help                 this list\n"
        "  quit                 end the game\n",
        LOG_DEFAULT);
}

int HumanAgent::quit(const char *why) {
    std::printf("%s\n", why);
    std::fflush(stdout);
    return agents::AGENT_QUIT;
}

std::vector<std::string> HumanAgent::take_news(const agents::DecisionView &v) {
    std::vector<std::string> news = event_lines(v);
    const agents::Table players = v.table("player");
    for (size_t r = 0; r < players.rows(); ++r) {
        const uint64_t seat = players.u(r, players.col("player"));
        const int64_t life = players.s(r, players.col("life"));
        auto it = life_.find(seat);
        if (it != life_.end() && it->second != life)
            news.push_back(
                "life: " + seat_name(v, seat) + " " + std::to_string(it->second) + " -> " + std::to_string(life));
        life_[seat] = life;
    }
    log_.insert(log_.end(), news.begin(), news.end());
    return news;
}

void HumanAgent::show(const agents::DecisionView &v, bool force) {
    const std::vector<std::string> board = board_lines(v);
    std::printf("%s\n", board_header(v).c_str());
    if (force || board != shown_) {
        print(board, "");
        shown_ = board;
    }
    print(pending_lines(v), "");
    print(menu_lines(v, true), "  ");
}

void HumanAgent::show_log(const std::string &arg) const {
    size_t n = LOG_DEFAULT;
    long k = 0;
    if (ascii_lower(arg) == "all")
        n = log_.size();
    else if (parse_int(arg, &k) && k > 0)
        n = static_cast<size_t>(k);
    if (log_.empty()) {
        std::printf("(nothing has happened yet)\n");
        return;
    }
    const size_t from = log_.size() > n ? log_.size() - n : 0;
    if (from > 0) std::printf("(%zu earlier lines: log all)\n", from);
    for (size_t i = from; i < log_.size(); ++i) std::printf("  %s\n", log_[i].c_str());
}

bool HumanAgent::run_command(const std::string &line, const agents::DecisionView &v, int *reply) {
    const std::vector<std::string> w = split(trim(line), ' ', true);
    if (w.empty()) return false;
    const std::string cmd = ascii_lower(w[0]);
    const std::string rest = line.substr(line.find(w[0]) + w[0].size());
    long n = 0;
    if (parse_int(w[0], &n) && w.size() == 1) {
        const long rows = static_cast<long>(v.n());
        if ((n >= 0 && n < rows) || (n == -1 && rows > 0) || n == -2 || n == -3) {
            *reply = static_cast<int>(n);
            return true;
        }
        std::printf("No menu row %ld: choose 0-%ld (or help).\n", n, rows - 1);
        return false;
    }
    if (cmd == "concede") {
        if (w.size() == 1 || (w.size() == 2 && ascii_lower(w[1]) == "game")) {
            *reply = -2;
            return true;
        }
        if (w.size() == 2 && ascii_lower(w[1]) == "match") {
            *reply = -3;
            return true;
        }
        std::printf("concede (this game) or concede match.\n");
        return false;
    }
    if (cmd == "board" || cmd == "b") {
        show(v, true);
        return false;
    }
    if (cmd == "menu" || cmd == "m") {
        print(menu_lines(v, true), "  ");
        return false;
    }
    if (cmd == "z" || (cmd == "pass" && w.size() >= 2 && ascii_lower(w[1]) == "turn")) {
        const size_t arg = cmd == "z" ? 1 : 2;
        const bool mine = w.size() == arg + 1 && ascii_lower(w[arg]) == "mine";
        if (w.size() > arg + (mine ? 1 : 0)) {
            std::printf("z [mine] (or pass turn [mine]).\n");
            return false;
        }
        return start_auto_pass(v, mine, reply);
    }
    if (cmd == "zone") {
        const char *zone = nullptr;
        bool opponent = false;
        bool bad = w.size() < 2;
        for (size_t i = 1; i < w.size(); ++i) {
            const int side = side_word(w[i]);
            const char *z = zone_alias(w[i]);
            if (side >= 0)
                opponent = side == 1;
            else if (z)
                zone = z;
            else
                bad = true;
        }
        if (bad || !zone) {
            std::printf("zone <hand|library|graveyard|exile|battlefield|stack|sideboard> [opp]\n");
            return false;
        }
        print(zone_lines(v, zone, opponent), "");
        return false;
    }
    if (cmd == "card" || cmd == "c") {
        print(card_lines(v, rest), "");
        return false;
    }
    if (cmd == "know" || cmd == "k") {
        print(knowledge_lines(v, true), "");
        return false;
    }
    if (cmd == "log" || cmd == "l") {
        show_log(w.size() > 1 ? w[1] : "");
        return false;
    }
    if (cmd == "help" || cmd == "h" || cmd == "?") {
        print_help();
        return false;
    }
    if (cmd == "quit" || cmd == "q" || cmd == "exit") {
        *reply = quit("Quitting.");
        return true;
    }
    std::printf("Unknown command '%s' (type help).\n", w[0].c_str());
    return false;
}

int HumanAgent::decide(const agents::Decision &d) {
    const agents::DecisionView &v = d.view;
    const std::string me = seat_name(v, seat_value(v, false));
    const bool first = log_.empty() && shown_.empty();
    const std::vector<std::string> news = take_news(v);
    std::string stopped;
    if (auto_.active) {
        const int pass = continue_auto_pass(v, &stopped);
        if (pass >= 0) {
            print(news, "  ");
            std::printf("%s passes (auto): %s\n", me.c_str(), board_header(v).c_str());
            return pass;
        }
    }
    std::printf("\n");
    if (banner_) std::printf("======== %s to decide ========\n", me.c_str());
    if (!stopped.empty()) std::printf("Auto-pass stopped: %s.\n", stopped.c_str());
    if (!news.empty()) {
        std::printf("%s\n", first ? "So far:" : "Since your last decision:");
        if (news.size() > NEWS_SHOWN) std::printf("  (%zu earlier lines: log)\n", news.size() - NEWS_SHOWN);
        for (size_t i = news.size() > NEWS_SHOWN ? news.size() - NEWS_SHOWN : 0; i < news.size(); ++i)
            std::printf("  %s\n", news[i].c_str());
    }
    show(v, false);
    while (true) {
        std::printf("%s [0-%zu, help]> ", me.c_str(), v.n() ? v.n() - 1 : 0);
        std::fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) {
            std::printf("\n");
            return quit("Input closed; quitting.");
        }
        int reply = 0;
        if (run_command(line, v, &reply)) return reply;
    }
}

}  // namespace cli
