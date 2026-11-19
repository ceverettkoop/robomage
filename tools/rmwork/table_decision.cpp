#include "table_decision.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "../../src/cli/board.h"
#include "card_picker.h"
#include "mulligan_plan.h"

namespace {

// How far back chosen_targets looks.
const size_t MAX_TARGET_HISTORY = 64;

}  // namespace

static int seat_of(uint64_t seat);
static rmwork::DecisionKind decision_kind(const agents::DecisionView &v);
static bool is_mana_category(const std::string &category);
static bool shown_object(const rmwork::TableModel &m, uint64_t handle);
static std::string prompt_of(const agents::DecisionView &v);
static void read_cost(const agents::DecisionView &v, rmwork::TableDecision *d);
static std::string pip_symbol(const agents::DecisionView &v, const agents::Table &p, size_t row);
static uint64_t pending_source(const agents::DecisionView &v);
static int lethal_of(const std::string &description);
static void read_block_facts(const agents::DecisionView &v, rmwork::BlockFacts *b);
static void read_damage(const rmwork::TableModel &m, rmwork::TableDecision *d);
static bool holds(const std::vector<uint64_t> &v, uint64_t h);
static rmwork::MulliganFacts read_mulligan(const agents::DecisionView &v, int seat);
static std::string mulligan_prompt(const rmwork::MulliganFacts &f, bool bottom);
static std::string mulligan_note(const rmwork::MulliganFacts &f);
static bool has(const agents::DecisionView &v, const char *category);
static bool all(const agents::DecisionView &v, const char *category);
static bool priority_category(const std::string &category);
static std::string priority_prompt(const agents::DecisionView &v, const rmwork::TableModel &m);
static bool button_category(const std::string &category);
static bool panel_row(rmwork::DecisionKind k, const std::string &category);
static std::string cost_part_text(const agents::DecisionView &v);
static void read_numbers(const agents::DecisionView &v, rmwork::TableDecision *d);
static int number_of(const agents::DecisionView &v, size_t row);
static void read_discard(const agents::DecisionView &v, int seat, rmwork::TableDecision *d);
static std::string divided_so_far(const rmwork::TableDecision &d);
static std::string card_cost_hint(
    const agents::DecisionView &v, const rmwork::TableModel &m, const rmwork::TableDecision &d);

// The `seat` enum (1 A, 2 B) as 0 A, 1 B; -1 for none.
static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static bool is_mana_category(const std::string &category) {
    return category.rfind("MANA_", 0) == 0;
}

static bool has(const agents::DecisionView &v, const char *category) {
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) == category) return true;
    return false;
}

static bool all(const agents::DecisionView &v, const char *category) {
    for (size_t i = 0; i < v.n(); ++i)
        if (v.category(i) != category) return false;
    return v.n() > 0;
}

// A row of a priority decision: an action taken with priority (a mana ability's MANA_* rows aside).
static bool priority_category(const std::string &c) {
    return c == "PASS_PRIORITY" || c == "CAST_SPELL" || c == "ACTIVATE_ABILITY" || c == "PLAY_LAND" ||
           c == "TURN_FACE_UP" || c == "COMPANION";
}

// What a priority decision offers, as the prompt: "Play a land or cast a spell, or pass"; with the stack
// not empty, what is responded to first ("Respond to Player B's Lightning Bolt: cast a spell, or pass").
static std::string priority_prompt(const agents::DecisionView &v, const rmwork::TableModel &m) {
    static const std::pair<const char *, const char *> VERBS[] = {{"PLAY_LAND", "play a land"},
        {"CAST_SPELL", "cast a spell"}, {"ACTIVATE_ABILITY", "activate an ability"},
        {"TURN_FACE_UP", "turn a permanent face up"}, {"COMPANION", "put your companion into your hand"},
        {"MANA_", "tap for mana"}};
    std::vector<std::string> verbs;
    for (const auto &verb : VERBS)
        for (size_t i = 0; i < v.n(); ++i)
            if (v.category(i).rfind(verb.first, 0) == 0) {
                verbs.push_back(verb.second);
                break;
            }
    std::string what;
    for (size_t k = 0; k < verbs.size(); ++k) what += (k == 0 ? "" : k + 1 == verbs.size() ? " or " : ", ") + verbs[k];
    std::string p = what.empty() ? std::string("pass") : what + ", or pass";
    if (!m.stack.empty()) {
        const rmwork::TableStackItem &top = m.stack[0];
        const std::string whose = top.controller == m.decider
                                      ? std::string("your")
                                      : m.players[static_cast<size_t>(top.controller)].name + "'s";
        p = "Respond to " + whose + " " + top.name + ": " + p;
    }
    p[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(p[0])));
    return p;
}

// A row the prompt line offers as a button even when it names an object (a yes/no's accept names the
// permanent it is about; a replacement names the card it applies to).
static bool button_category(const std::string &c) {
    return c == "OPTIONAL_YESNO" || c == "PAY_UNLESS" || c == "SHUFFLE" || c == "DONT_SHUFFLE" ||
           c == "CHOOSE_REPLACEMENT" || c == "OTHER_CHOICE" || c == "CHOOSE_COLOR" || c == "CHOOSE_TYPE" ||
           c == "CHOOSE_MANA_COLOR" || c == "CONFIRM_MODES";
}

// A row the choice panel offers for a decision of kind `k`, or (NUMBER) the prompt line's stepper.
static bool panel_row(rmwork::DecisionKind k, const std::string &c) {
    using rmwork::DecisionKind;
    return (k == DecisionKind::MODES && c == "CHOOSE_MODE") || k == DecisionKind::ORDER || k == DecisionKind::NAME ||
           k == DecisionKind::NUMBER;
}

static rmwork::DecisionKind decision_kind(const agents::DecisionView &v) {
    using rmwork::DecisionKind;
    if (v.n() == 0) return DecisionKind::NONE;
    bool targets = true, mana = false, priority = false;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &c = v.category(i);
        if (c.rfind("SIDEBOARD_", 0) == 0) return DecisionKind::NONE;
        if (c == "PAYING_COSTS") return DecisionKind::PAY;
        if (c == "SELECT_ATTACKER" || c == "CONFIRM_ATTACKERS") return DecisionKind::ATTACKERS;
        if (c == "ATTACK_TARGET") return DecisionKind::ATTACK_TARGET;
        if (c == "SELECT_BLOCKER" || c == "CONFIRM_BLOCKERS") return DecisionKind::BLOCKERS;
        if (c == "BLOCK_TARGET") return DecisionKind::BLOCK_TARGET;
        if (c == "ASSIGN_DAMAGE") return DecisionKind::DAMAGE;
        if (c == "KEEP_HAND" || c == "MULLIGAN") return DecisionKind::MULLIGAN;
        if (c == "PAY_UNLESS") return DecisionKind::UNLESS;
        if (c == "CHOOSE_MODE" || c == "CONFIRM_MODES") return DecisionKind::MODES;
        if (c != "SELECT_TARGET" && c != "DIVIDE_DAMAGE") targets = false;
        if (is_mana_category(c)) mana = true;
        if (priority_category(c)) priority = true;
    }
    if (targets) return DecisionKind::TARGET;
    if (all(v, "CHOOSE_X")) return DecisionKind::NUMBER;
    if (all(v, "ORDER_TRIGGERS")) return DecisionKind::ORDER;
    if (all(v, "NAME_CARD")) return DecisionKind::NAME;
    if (mana || priority) return DecisionKind::PRIORITY;
    if (rmwork::london_bottom_menu(v)) return DecisionKind::BOTTOM;
    if (rmwork::card_pick_menu(v)) return DecisionKind::CARDS;
    return DecisionKind::CHOICE;
}

// The pending cost's part when it is not a plain mana cost: "alternative cost", "additional cost",
// "unless cost", …; "" for none, mana or X.
static std::string cost_part_text(const agents::DecisionView &v) {
    const agents::Table c = v.table("pending_cost");
    if (c.rows() == 0) return "";
    const std::string part = v.stream().enum_name("cost_part", c.u(0, c.col("part")));
    if (part == "ALTERNATIVE") return "alternative cost";
    if (part == "ADDITIONAL") return "additional cost";
    if (part == "UNLESS") return "unless cost";
    if (part == "ACTIVATION") return "activation cost";
    if (part == "TRIGGER") return "the trigger's cost";
    return "";
}

// A CHOOSE_X row's value: its ordinal, else the number in its description ("X = 2", "Exile 4 cards …").
static int number_of(const agents::DecisionView &v, size_t row) {
    const agents::Table menu = v.table("menu");
    const int64_t ord = menu.s(row, menu.col("ordinal"));
    if (ord >= 0) return static_cast<int>(ord);
    const std::string &t = v.description(row);
    const size_t at = t.find_first_of("0123456789");
    return at == std::string::npos ? 0 : std::atoi(t.c_str() + at);
}

// NUMBER: each row's value, whether the rows count delve's exiles (they name the spell), and the range
// offered for the prompt.
static void read_numbers(const agents::DecisionView &v, rmwork::TableDecision *d) {
    const agents::Table menu = v.table("menu");
    for (size_t i = 0; i < v.n(); ++i) {
        d->numbers.push_back(number_of(v, i));
        d->delve =
            d->delve || (menu.u(i, menu.col("source")) != 0 && v.description(i).find("Delve") != std::string::npos);
    }
    if (d->numbers.empty()) return;
    const auto mm = std::minmax_element(d->numbers.begin(), d->numbers.end());
    const std::string range = std::to_string(*mm.first) + " to " + std::to_string(*mm.second);
    d->owed = d->delve ? range + " cards" : "X from " + range;
}

// A discard of seat `seat`'s hand cards: how many the plan holds (an unless offer's or an escalate
// cost's `rem_discard`; at the cleanup step, the hand beyond seven, CR 402.2 and 514.1: the frame names
// no maximum hand size) and what the prompt adds.
static void read_discard(const agents::DecisionView &v, int seat, rmwork::TableDecision *d) {
    if (!all(v, "DISCARD")) return;
    const agents::Table c = v.table("pending_cost");
    if (c.rows() && c.u(0, c.col("rem_discard")) > 0) {
        d->plan_owed = static_cast<int>(c.u(0, c.col("rem_discard")));
        d->owed = "discard " + std::to_string(d->plan_owed) + (d->plan_owed == 1 ? " card" : " cards");
        return;
    }
    const agents::Table t = v.table("turn");
    if (v.table("pending").rows() || !t.rows() || v.stream().enum_name("step", t.u(0, t.col("step"))) != "CLEANUP")
        return;
    const agents::Table p = v.table("player");
    for (size_t r = 0; r < p.rows(); ++r) {
        if (seat_of(p.u(r, p.col("player"))) != seat) continue;
        const int hand = static_cast<int>(p.u(r, p.col("hand")));
        d->plan_owed = std::max(1, hand - 7);
        d->owed = std::to_string(hand) + " in hand, " + std::to_string(d->plan_owed) + " to discard (hand size 7)";
    }
}

// PAY by a card rather than mana (an alternative cost's exile, convoke's tap): what a click does, from the
// rows' shared verb and where their cards are: "exile a card from your hand", "tap a creature (convoke)";
// "" for a mana payment.
static std::string card_cost_hint(
    const agents::DecisionView &v, const rmwork::TableModel &m, const rmwork::TableDecision &d) {
    if (d.picks.empty()) return "";
    std::string verb;
    bool convoke = true;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &t = v.description(i);
        if (v.category(i) != "PAYING_COSTS" || t.find(" for (") != std::string::npos) return "";
        const std::string w = t.substr(0, t.find(' '));
        if (!verb.empty() && w != verb) return "";
        verb = w;
        convoke = convoke && t.find("(Convoke)") != std::string::npos;
    }
    if (convoke) return "tap a creature to pay (convoke)";
    bool hand = true, field = true;
    for (const rmwork::DecisionPick &p : d.picks) {
        const int o = m.find(p.handle);
        const bool in_hand = o >= 0 && m.objects[static_cast<size_t>(o)].in_hand;
        hand = hand && in_hand;
        field = field && o >= 0 && !in_hand;
    }
    std::string w = verb;
    for (char &ch : w) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return hand ? w + " a card from your hand" : field ? w + " a permanent" : w + " a card";
}

// TARGET at a divided-damage decision: the points given so far (the frame gives no total).
static std::string divided_so_far(const rmwork::TableDecision &d) {
    int given = 0;
    for (const rmwork::DecisionPick &p : d.picks) given += std::max(0, p.so_far);
    return given > 0 ? std::to_string(given) + " of the damage given so far" : "";
}

// Seat `seat`'s mulligans and cards owed to the bottom, its hand, play or draw, the other seat's mulligans.
static rmwork::MulliganFacts read_mulligan(const agents::DecisionView &v, int seat) {
    rmwork::MulliganFacts f;
    const agents::Table p = v.table("player");
    for (size_t r = 0; r < p.rows(); ++r) {
        const int s = seat_of(p.u(r, p.col("player")));
        if (s == seat) {
            f.mulligans = static_cast<int>(p.u(r, p.col("mulligans")));
            f.to_bottom = static_cast<int>(p.u(r, p.col("to_bottom")));
            f.hand = static_cast<int>(p.u(r, p.col("hand")));
        } else if (s >= 0) {
            f.opponent_mulligans = static_cast<int>(p.u(r, p.col("mulligans")));
        }
    }
    const agents::Table m = v.table("match");
    if (m.rows() && seat >= 0) {
        const int first = seat_of(m.u(0, m.col("plays_first")));
        if (first >= 0) f.on_play = first == seat ? 1 : 0;
    }
    return f;
}

// "Mulligan 2: keep these 7 and put 2 on the bottom, or mulligan" / "Keeping 5: put 1 more card on the
// bottom of your library (1 of 2 done)".
static std::string mulligan_prompt(const rmwork::MulliganFacts &f, bool bottom) {
    std::string p;
    if (bottom) {
        const int done = std::max(0, f.mulligans - f.to_bottom);
        p = "Keeping " + std::to_string(f.hand - f.to_bottom) + ": put " + std::to_string(f.to_bottom) +
            (done ? " more" : "") + (f.to_bottom == 1 ? " card" : " cards") + " on the bottom of your library";
        if (done) p += " (" + std::to_string(done) + " of " + std::to_string(f.mulligans) + " done)";
    } else if (f.mulligans == 0) {
        p = "Opening hand: keep these " + std::to_string(f.hand) + ", or mulligan (draw 7, then put 1 on the bottom)";
    } else {
        p = "Mulligan " + std::to_string(f.mulligans) + ": keep these " + std::to_string(f.hand) + " and put " +
            std::to_string(f.mulligans) + " on the bottom, or mulligan to " +
            std::to_string(std::max(0, f.hand - f.mulligans - 1));
    }
    return p;
}

// "on the play · the opponent took 2 mulligans".
static std::string mulligan_note(const rmwork::MulliganFacts &f) {
    std::string n;
    if (f.on_play >= 0) n = f.on_play ? "on the play" : "on the draw";
    if (f.opponent_mulligans > 0)
        n += (n.empty() ? "" : " \xC2\xB7 ") + std::string("the opponent took ") +
             std::to_string(f.opponent_mulligans) + (f.opponent_mulligans == 1 ? " mulligan" : " mulligans");
    return n;
}

static bool holds(const std::vector<uint64_t> &v, uint64_t h) {
    return std::find(v.begin(), v.end(), h) != v.end();
}

// An ASSIGN_DAMAGE row's "(lethal 2)": 2; -1 when it has none (Done assigning).
static int lethal_of(const std::string &description) {
    const std::string key = "(lethal ";
    const size_t at = description.rfind(key);
    if (at == std::string::npos) return -1;
    const size_t from = at + key.size();
    size_t end = from;
    while (end < description.size() && std::isdigit(static_cast<unsigned char>(description[end]))) ++end;
    return end > from ? std::atoi(description.substr(from, end - from).c_str()) : -1;
}

// The attackers (the combat table), and the objects with flying, with reach (object_keyword) and
// that can't be blocked (object.cant_be_blocked).
static void read_block_facts(const agents::DecisionView &v, rmwork::BlockFacts *b) {
    const agents::Table c = v.table("combat");
    for (size_t r = 0; r < c.rows(); ++r) {
        const uint64_t a = c.u(r, c.col("attacker"));
        if (!holds(b->attackers, a)) b->attackers.push_back(a);
    }
    const agents::Table k = v.table("object_keyword");
    const int c_obj = k.col("obj"), c_kw = k.col("kw");
    for (size_t r = 0; r < k.rows(); ++r) {
        const std::string kw = v.stream().enum_name("keyword", k.u(r, c_kw));
        if (kw == "Flying") b->flying.push_back(k.u(r, c_obj));
        if (kw == "Reach") b->reach.push_back(k.u(r, c_obj));
    }
    const agents::Table o = v.table("object");
    const int c_h = o.col("handle"), c_cbb = o.col("cant_be_blocked");
    for (size_t r = 0; c_cbb >= 0 && r < o.rows(); ++r)
        if (o.u(r, c_cbb)) b->unblockable.push_back(o.u(r, c_h));
}

// DAMAGE: what the attacker (the decision's source) has left to assign, after the prompt.
static void read_damage(const rmwork::TableModel &m, rmwork::TableDecision *d) {
    const int obj = m.find(d->source);
    if (obj < 0) return;
    const int power = m.objects[static_cast<size_t>(obj)].power;
    int assigned = 0;
    for (const rmwork::CombatBlock &b : m.blocks)
        if (b.attacker == d->source) assigned += b.assigned;
    d->owed = std::to_string(std::max(0, power - assigned)) + " of " + std::to_string(power) + " left to assign";
}

// An object the Table draws: a battlefield or hand card of its model, or a stack object.
static bool shown_object(const rmwork::TableModel &m, uint64_t handle) {
    if (handle == 0) return false;
    if (m.find(handle) >= 0) return true;
    for (const rmwork::TableStackItem &it : m.stack)
        if (it.handle == handle) return true;
    return false;
}

// The CLI's "Decision: casting Prismari Charm: choose a target", without "Decision: ", capitalised.
static std::string prompt_of(const agents::DecisionView &v) {
    const std::vector<std::string> lines = cli::pending_lines(v);
    std::string p = lines.empty() ? std::string() : lines[0];
    const std::string lead = "Decision: ";
    if (p.rfind(lead, 0) == 0) p = p.substr(lead.size());
    if (!p.empty()) p[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(p[0])));
    return p;
}

static uint64_t pending_source(const agents::DecisionView &v) {
    const agents::Table p = v.table("pending");
    return p.rows() ? p.u(0, p.col("source")) : 0;
}

// A non-plain pip of the `pending_cost_pip` table as a mana symbol.
static std::string pip_symbol(const agents::DecisionView &v, const agents::Table &p, size_t row) {
    static const char *const LETTERS[] = {"W", "U", "B", "R", "G", "C"};
    static const char *const COLORS[] = {"WHITE", "BLUE", "BLACK", "RED", "GREEN", "COLORLESS"};
    const std::string kind = v.stream().enum_name("pip_kind", p.u(row, p.col("kind")));
    const std::vector<std::string> &values = v.stream().enum_values("color");
    std::string colors;
    const uint64_t mask = p.u(row, p.col("colors"));
    for (int k = 0; k < 6; ++k)
        for (size_t i = 0; i < values.size(); ++i)
            if ((mask & (1ull << i)) && values[i] == COLORS[k])
                colors += (colors.empty() ? "" : "/") + std::string(LETTERS[k]);
    if (kind == "SNOW") return "S";
    if (kind == "PHYREXIAN") return colors + "/P";
    if (kind == "TWOBRID") return std::to_string(p.u(row, p.col("generic_alt"))) + "/" + colors;
    return colors.empty() ? "?" : colors;
}

// The first `pending_cost` row: its remaining pips, life, energy and discards, the mana paid.
static void read_cost(const agents::DecisionView &v, rmwork::TableDecision *d) {
    const agents::Table c = v.table("pending_cost");
    if (c.rows() == 0) return;
    for (uint64_t i = 0; i < c.u(0, c.col("rem_x")); ++i) d->pips.push_back("X");
    if (const uint64_t g = c.u(0, c.col("rem_generic"))) d->pips.push_back(std::to_string(g));
    static const char *const COLS[] = {"rem_w", "rem_u", "rem_b", "rem_r", "rem_g", "rem_c"};
    static const char *const SYMS[] = {"W", "U", "B", "R", "G", "C"};
    for (int k = 0; k < 6; ++k)
        for (uint64_t i = 0; i < c.u(0, c.col(COLS[k])); ++i) d->pips.push_back(SYMS[k]);
    const agents::Table p = v.table("pending_cost_pip");
    for (size_t r = 0; r < p.rows(); ++r)
        if (p.u(r, p.col("cost")) == 0) d->pips.push_back(pip_symbol(v, p, r));
    std::vector<std::string> parts;
    if (const uint64_t n = c.u(0, c.col("rem_life"))) parts.push_back(std::to_string(n) + " life");
    if (const uint64_t n = c.u(0, c.col("rem_energy"))) parts.push_back(std::to_string(n) + " energy");
    if (const uint64_t n = c.u(0, c.col("rem_discard")))
        parts.push_back("discard " + std::to_string(n) + (n == 1 ? " card" : " cards"));
    if (const uint64_t n = c.u(0, c.col("paid"))) parts.push_back(std::to_string(n) + " mana paid");
    for (const std::string &s : parts) d->owed += (d->owed.empty() ? "" : " \xC2\xB7 ") + s;
}

namespace rmwork {

const DecisionPick *TableDecision::pick_of_object(uint64_t handle) const {
    if (handle == 0) return nullptr;
    for (const DecisionPick &p : picks)
        if (p.handle == handle) return &p;
    return nullptr;
}

const DecisionPick *TableDecision::pick_of_seat(int seat) const {
    if (seat < 0) return nullptr;
    for (const DecisionPick &p : picks)
        if (p.handle == 0 && p.seat == seat) return &p;
    return nullptr;
}

int TableDecision::chosen_order_of_object(uint64_t handle) const {
    for (const ChosenTarget &c : chosen)
        if (handle != 0 && c.handle == handle) return c.order;
    return 0;
}

bool declares_attackers(DecisionKind k) {
    return k == DecisionKind::ATTACKERS || k == DecisionKind::ATTACK_TARGET;
}

bool declares_blockers(DecisionKind k) {
    return k == DecisionKind::BLOCKERS || k == DecisionKind::BLOCK_TARGET;
}

bool TableDecision::may_block(uint64_t blocker, uint64_t attacker, std::string *why) const {
    if (holds(block.unblockable, attacker)) {
        *why = "it can't be blocked";
        return false;
    }
    if (holds(block.flying, attacker) && !holds(block.flying, blocker) && !holds(block.reach, blocker)) {
        *why = "it has flying";
        return false;
    }
    return true;
}

bool TableDecision::holds_chosen(const DecisionPick &p) const {
    return chosen_row >= 0 && std::find(p.rows.begin(), p.rows.end(), chosen_row) != p.rows.end();
}

int TableDecision::chosen_order_of_seat(int seat) const {
    for (const ChosenTarget &c : chosen)
        if (seat >= 0 && c.handle == 0 && c.seat == seat) return c.order;
    return 0;
}

int TableDecision::number_row(int value) const {
    for (size_t i = 0; i < numbers.size(); ++i)
        if (numbers[i] == value) return static_cast<int>(i);
    return -1;
}

TableDecision read_table_decision(const agents::DecisionView &v, const TableModel &m, const CardBook *book) {
    TableDecision d;
    d.kind = decision_kind(v);
    if (d.kind == DecisionKind::NONE) return d;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source"), c_tgt = menu.col("target"), c_sp = menu.col("source_player"),
              c_tp = menu.col("target_player"), c_ord = menu.col("ordinal");
    for (size_t i = 0; i < v.n(); ++i) d.row_text.push_back(v.description(i));
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &category = v.category(i);
        if (category == "PASS_PRIORITY") d.pass = static_cast<int>(i);
        if (panel_row(d.kind, category)) {
            if (d.kind != DecisionKind::NUMBER) d.panel_rows.push_back(static_cast<int>(i));
            continue;
        }
        const uint64_t object = menu.u(i, c_src) ? menu.u(i, c_src) : menu.u(i, c_tgt);
        const int seat = seat_of(menu.u(i, c_sp)) >= 0 ? seat_of(menu.u(i, c_sp)) : seat_of(menu.u(i, c_tp));
        if (category == "CONFIRM_ATTACKERS" || category == "CONFIRM_BLOCKERS") d.confirm = static_cast<int>(i);
        // A row naming nothing, a row of a button category, and (but at a card picker's decision) a row
        // naming an object the Table does not show: a button.
        if ((object == 0 && seat < 0) || button_category(category) ||
            (object != 0 && !shown_object(m, object) && d.kind != DecisionKind::CARDS)) {
            if (d.prompted()) d.buttons.push_back(static_cast<int>(i));
            continue;
        }
        if (object != 0 && !shown_object(m, object)) continue;
        DecisionPick *pick = nullptr;
        for (DecisionPick &p : d.picks)
            if (object != 0 ? p.handle == object : (p.handle == 0 && p.seat == seat)) pick = &p;
        if (!pick) {
            d.picks.emplace_back();
            pick = &d.picks.back();
            pick->handle = object;
            pick->seat = object != 0 ? -1 : seat;
        }
        pick->rows.push_back(static_cast<int>(i));
        if (category == "DIVIDE_DAMAGE") pick->so_far = static_cast<int>(menu.s(i, c_ord));
        if (category == "ASSIGN_DAMAGE") pick->lethal = lethal_of(d.row_text[i]);
    }
    // Too many buttons for the prompt line: the choice panel lists them.
    if (d.buttons.size() > MAX_PROMPT_ROWS && d.kind != DecisionKind::ATTACKERS && d.kind != DecisionKind::BLOCKERS &&
        d.kind != DecisionKind::CARDS && d.kind != DecisionKind::PRIORITY) {
        d.panel_rows.insert(d.panel_rows.end(), d.buttons.begin(), d.buttons.end());
        std::sort(d.panel_rows.begin(), d.panel_rows.end());
        d.buttons.clear();
    }
    d.source = pending_source(v);
    if (d.prompted()) d.prompt = d.kind == DecisionKind::PRIORITY ? priority_prompt(v, m) : prompt_of(v);
    if (d.kind == DecisionKind::PRIORITY && book)
        for (size_t i = 0; i < v.n(); ++i) d.costs.push_back(way_cost(v, *book, i));
    if (d.kind == DecisionKind::MULLIGAN || d.kind == DecisionKind::BOTTOM) {
        d.mulligan = read_mulligan(v, m.decider >= 0 ? m.decider : v.seat());
        d.prompt = mulligan_prompt(d.mulligan, d.kind == DecisionKind::BOTTOM);
        d.owed = mulligan_note(d.mulligan);
    }
    d.cost_part = cost_part_text(v);
    if (d.kind == DecisionKind::PAY || d.kind == DecisionKind::UNLESS || d.kind == DecisionKind::NUMBER ||
        d.kind == DecisionKind::CARDS)
        read_cost(v, &d);
    if (d.kind == DecisionKind::PAY) {
        const std::string hint = card_cost_hint(v, m, d);
        if (!hint.empty()) d.owed = hint + (d.owed.empty() ? "" : " \xC2\xB7 then " + d.owed);
    }
    if (d.kind == DecisionKind::NUMBER) read_numbers(v, &d);
    if (d.kind == DecisionKind::CARDS) read_discard(v, m.decider >= 0 ? m.decider : v.seat(), &d);
    if (d.kind == DecisionKind::TARGET && has(v, "DIVIDE_DAMAGE")) d.owed = divided_so_far(d);
    if (d.kind == DecisionKind::BLOCKERS) read_block_facts(v, &d.block);
    if (d.kind == DecisionKind::DAMAGE) read_damage(m, &d);
    return d;
}

std::vector<std::string> number_pips(const TableDecision &d, int value) {
    std::vector<std::string> out;
    int generic = 0, xs = 0;
    for (const std::string &p : d.pips) {
        if (p == "X") {
            ++xs;
        } else if (!p.empty() && std::isdigit(static_cast<unsigned char>(p[0])) && p.find('/') == std::string::npos) {
            generic += std::atoi(p.c_str());
        }
    }
    generic = d.delve ? std::max(0, generic - value) : generic + xs * value;
    if (generic > 0) out.push_back(std::to_string(generic));
    for (const std::string &p : d.pips) {
        const bool plain_generic =
            !p.empty() && std::isdigit(static_cast<unsigned char>(p[0])) && p.find('/') == std::string::npos;
        if (p != "X" && !plain_generic) out.push_back(p);
    }
    return out;
}

int mana_count(const std::vector<std::string> &pips) {
    int n = 0;
    for (const std::string &p : pips)
        n += !p.empty() && std::isdigit(static_cast<unsigned char>(p[0])) && p.find('/') == std::string::npos
                 ? std::atoi(p.c_str())
                 : 1;
    return n;
}

std::vector<ChosenTarget> chosen_targets(const Match &m, size_t index, int view, uint64_t source) {
    std::vector<ChosenTarget> out;
    if (index >= m.decisions.size() || source == 0) return out;
    const DecisionInfo &now = m.decisions[index];
    for (size_t back = 1; back <= std::min(index, MAX_TARGET_HISTORY); ++back) {
        const size_t i = index - back;
        const DecisionInfo &d = m.decisions[i];
        if (d.decider != now.decider || d.game != now.game || d.sideboarding != now.sideboarding || d.reply < 0) break;
        std::string err;
        const auto frame = m.frame(i, view, &err);
        if (!frame) break;
        const agents::DecisionView v(m.stream(view), *frame, d.decider);
        if (pending_source(v) != source || static_cast<size_t>(d.reply) >= v.n()) break;
        const std::string &category = v.category(static_cast<size_t>(d.reply));
        if (category == "DIVIDE_DAMAGE") continue;
        if (category != "SELECT_TARGET") break;
        const agents::Table menu = v.table("menu");
        const size_t r = static_cast<size_t>(d.reply);
        ChosenTarget c;
        c.handle = menu.u(r, menu.col("source"));
        c.seat = c.handle ? -1 : seat_of(menu.u(r, menu.col("source_player")));
        c.label = v.description(r);
        if (c.handle != 0 || c.seat >= 0) out.push_back(c);
    }
    std::reverse(out.begin(), out.end());
    for (size_t k = 0; k < out.size(); ++k) out[k].order = static_cast<int>(k) + 1;
    return out;
}

}  // namespace rmwork
