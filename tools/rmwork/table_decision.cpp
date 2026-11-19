#include "table_decision.h"

#include <algorithm>
#include <cctype>

#include "../../src/cli/board.h"

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
        if (c != "SELECT_TARGET" && c != "DIVIDE_DAMAGE") targets = false;
        if (is_mana_category(c)) mana = true;
    }
    if (targets) return DecisionKind::TARGET;
    return mana ? DecisionKind::MANA : DecisionKind::NONE;
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
    }
    d.source = pending_source(v);
    if (d.prompted()) d.prompt = prompt_of(v);
    if (d.kind == DecisionKind::PAY) read_cost(v, &d);
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
