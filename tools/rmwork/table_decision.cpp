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

// The `seat` enum (1 A, 2 B) as 0 A, 1 B; -1 for none.
static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static bool is_mana_category(const std::string &category) {
    return category.rfind("MANA_", 0) == 0;
}

static rmwork::DecisionKind decision_kind(const agents::DecisionView &v) {
    using rmwork::DecisionKind;
    if (v.n() == 0) return DecisionKind::NONE;
    bool targets = true, mana = false;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &c = v.category(i);
        if (c == "PAYING_COSTS") return DecisionKind::PAY;
        if (c == "SELECT_ATTACKER" || c == "CONFIRM_ATTACKERS") return DecisionKind::ATTACKERS;
        if (c == "ATTACK_TARGET") return DecisionKind::ATTACK_TARGET;
        if (c == "SELECT_BLOCKER" || c == "CONFIRM_BLOCKERS") return DecisionKind::BLOCKERS;
        if (c == "BLOCK_TARGET") return DecisionKind::BLOCK_TARGET;
        if (c == "ASSIGN_DAMAGE") return DecisionKind::DAMAGE;
        if (c == "KEEP_HAND" || c == "MULLIGAN") return DecisionKind::MULLIGAN;
        if (c != "SELECT_TARGET" && c != "DIVIDE_DAMAGE") targets = false;
        if (is_mana_category(c)) mana = true;
    }
    if (targets) return DecisionKind::TARGET;
    if (mana) return DecisionKind::MANA;
    if (rmwork::london_bottom_menu(v)) return DecisionKind::BOTTOM;
    return rmwork::card_pick_menu(v) ? DecisionKind::CARDS : DecisionKind::NONE;
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

TableDecision read_table_decision(const agents::DecisionView &v, const TableModel &m) {
    TableDecision d;
    d.kind = decision_kind(v);
    if (d.kind == DecisionKind::NONE) return d;
    const agents::Table menu = v.table("menu");
    const int c_src = menu.col("source"), c_tgt = menu.col("target"), c_sp = menu.col("source_player"),
              c_tp = menu.col("target_player"), c_ord = menu.col("ordinal");
    for (size_t i = 0; i < v.n(); ++i) d.row_text.push_back(v.description(i));
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &category = v.category(i);
        if (d.kind == DecisionKind::MANA && !is_mana_category(category)) continue;
        const uint64_t object = menu.u(i, c_src) ? menu.u(i, c_src) : menu.u(i, c_tgt);
        const int seat = seat_of(menu.u(i, c_sp)) >= 0 ? seat_of(menu.u(i, c_sp)) : seat_of(menu.u(i, c_tp));
        if (category == "CONFIRM_ATTACKERS" || category == "CONFIRM_BLOCKERS") d.confirm = static_cast<int>(i);
        if (object == 0 && seat < 0) {
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
    d.source = pending_source(v);
    if (d.prompted()) d.prompt = prompt_of(v);
    if (d.kind == DecisionKind::MULLIGAN || d.kind == DecisionKind::BOTTOM) {
        d.mulligan = read_mulligan(v, m.decider >= 0 ? m.decider : v.seat());
        d.prompt = mulligan_prompt(d.mulligan, d.kind == DecisionKind::BOTTOM);
        d.owed = mulligan_note(d.mulligan);
    }
    if (d.kind == DecisionKind::PAY) read_cost(v, &d);
    if (d.kind == DecisionKind::BLOCKERS) read_block_facts(v, &d.block);
    if (d.kind == DecisionKind::DAMAGE) read_damage(m, &d);
    return d;
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
