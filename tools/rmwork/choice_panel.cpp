#include "choice_panel.h"

#include <algorithm>
#include <cctype>

#include "sideboard_plan.h"

namespace {

// How far back the modes chosen earlier are looked for.
const size_t MAX_MODE_HISTORY = 64;

// A mode chosen earlier in the same cast, with the targets chosen for it.
struct PastMode {
        int order = 0;
        std::string text;
        std::vector<std::string> targets;
};

}  // namespace

static std::string lower(std::string s);
static uint64_t pending_source(const agents::DecisionView &v);
static int pending_card(const agents::DecisionView &v);
static int row_card(const agents::DecisionView &v, size_t row);
static std::string card_name(const rmwork::CardBook &book, int card, const std::string &fallback);
static std::string ordinal_word(int n);
static std::vector<PastMode> past_modes(const rmwork::Match &m, size_t index, int view, uint64_t source);
static void read_modes(const rmwork::Match &m, size_t index, int view, const agents::DecisionView &v,
    const rmwork::TableDecision &d, rmwork::ChoicePanel *p);
static void read_order(const agents::DecisionView &v, const rmwork::TableDecision &d, rmwork::ChoicePanel *p);
static void read_names(const rmwork::Match &m, int view, const agents::DecisionView &v, const rmwork::TableDecision &d,
    const rmwork::CardBook &book, rmwork::ChoicePanel *p);
static void read_list(const agents::DecisionView &v, const rmwork::TableDecision &d, rmwork::ChoicePanel *p);

static std::string lower(std::string s) {
    for (char &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

static uint64_t pending_source(const agents::DecisionView &v) {
    const agents::Table p = v.table("pending");
    return p.rows() ? p.u(0, p.col("source")) : 0;
}

static int pending_card(const agents::DecisionView &v) {
    const agents::Table p = v.table("pending");
    return p.rows() && !p.none(0, p.col("source_card")) ? p.card(0, p.col("source_card")) : -1;
}

static int row_card(const agents::DecisionView &v, size_t row) {
    const agents::Table menu = v.table("menu");
    const int c = menu.col("source_card");
    return menu.none(row, c) ? -1 : menu.card(row, c);
}

static std::string card_name(const rmwork::CardBook &book, int card, const std::string &fallback) {
    const rmwork::CardFace *f = card >= 0 ? book.face(card, false) : nullptr;
    return f ? f->name : fallback;
}

// "1st", "2nd", "3rd", "4th", ….
static std::string ordinal_word(int n) {
    const int t = n % 100, d = n % 10;
    const char *suffix = (t >= 11 && t <= 13) ? "th" : d == 1 ? "st" : d == 2 ? "nd" : d == 3 ? "rd" : "th";
    return std::to_string(n) + suffix;
}

// The modes chosen at the deciding seat's decisions just before `index` for the same pending source,
// oldest first, each with the targets chosen after it.
static std::vector<PastMode> past_modes(const rmwork::Match &m, size_t index, int view, uint64_t source) {
    std::vector<PastMode> out;
    if (index >= m.decisions.size() || source == 0) return out;
    const rmwork::DecisionInfo &now = m.decisions[index];
    std::vector<std::string> targets;  // chosen after the mode found next (walking back)
    for (size_t back = 1; back <= std::min(index, MAX_MODE_HISTORY); ++back) {
        const size_t i = index - back;
        const rmwork::DecisionInfo &d = m.decisions[i];
        if (d.decider != now.decider || d.game != now.game || d.sideboarding != now.sideboarding || d.reply < 0) break;
        std::string err;
        const auto frame = m.frame(i, view, &err);
        if (!frame) break;
        const agents::DecisionView v(m.stream(view), *frame, d.decider);
        if (pending_source(v) != source || static_cast<size_t>(d.reply) >= v.n()) break;
        const size_t r = static_cast<size_t>(d.reply);
        const std::string &category = v.category(r);
        if (category == "SELECT_TARGET") {
            const agents::Table menu = v.table("menu");
            const bool names = menu.u(r, menu.col("source")) != 0 || menu.u(r, menu.col("source_player")) != 0;
            if (names) targets.insert(targets.begin(), v.description(r));
        } else if (category == "CHOOSE_MODE") {
            const agents::Table menu = v.table("menu");
            PastMode pm;
            pm.order = static_cast<int>(std::max<int64_t>(0, menu.s(r, menu.col("ordinal"))));
            pm.text = v.description(r);
            pm.targets = targets;
            out.push_back(pm);
            targets.clear();
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

static void read_modes(const rmwork::Match &m, size_t index, int view, const agents::DecisionView &v,
    const rmwork::TableDecision &d, rmwork::ChoicePanel *p) {
    const agents::Table menu = v.table("menu");
    const int c_ord = menu.col("ordinal");
    const std::vector<PastMode> past = past_modes(m, index, view, d.source);
    for (size_t k = 0; k < past.size(); ++k) {
        rmwork::ChoiceItem it;
        it.text = past[k].text;
        it.card = p->source_card;
        it.order = past[k].order;
        it.chosen_before = true;
        it.note = "chosen " + ordinal_word(static_cast<int>(k) + 1);
        for (size_t t = 0; t < past[k].targets.size(); ++t)
            it.note += (t ? ", " : " \xE2\x86\x92 ") + past[k].targets[t];
        p->items.push_back(it);
    }
    for (int row : d.panel_rows) {
        rmwork::ChoiceItem it;
        it.row = row;
        it.text = v.description(static_cast<size_t>(row));
        it.card = p->source_card;
        it.order = static_cast<int>(std::max<int64_t>(0, menu.s(static_cast<size_t>(row), c_ord)));
        it.note = "row " + std::to_string(row);
        p->items.push_back(it);
    }
    std::stable_sort(p->items.begin(), p->items.end(),
        [](const rmwork::ChoiceItem &a, const rmwork::ChoiceItem &b) { return a.order < b.order; });
    for (int row : d.buttons)
        if (v.category(static_cast<size_t>(row)) == "CONFIRM_MODES") p->buttons.push_back(row);
    const std::string chosen = past.empty() ? "" : " (" + std::to_string(past.size()) + " chosen)";
    p->title += " \xC2\xB7 choose a mode" + chosen;
    p->note = p->buttons.empty() ? "Click a mode to choose it; a mode's targets are asked as it is chosen"
                                 : "Click another mode, or Done; a mode's targets are asked as it is chosen";
}

static void read_order(const agents::DecisionView &v, const rmwork::TableDecision &d, rmwork::ChoicePanel *p) {
    for (int row : d.panel_rows) {
        rmwork::ChoiceItem it;
        it.row = row;
        it.text = v.description(static_cast<size_t>(row));
        it.card = row_card(v, static_cast<size_t>(row));
        it.note = "row " + std::to_string(row);
        p->items.push_back(it);
    }
    p->title = "Your triggers \xC2\xB7 " + std::to_string(d.panel_rows.size()) + " waiting to go on the stack";
    p->note =
        "The trigger chosen goes on the stack now, above those put there before: the last one put there "
        "resolves first";
}

// The candidates grouped by the DECKLISTS frame: the chooser's opponent's main deck and sideboard, the
// chooser's deck, the rest; alphabetical within a group.
static void read_names(const rmwork::Match &m, int view, const agents::DecisionView &v, const rmwork::TableDecision &d,
    const rmwork::CardBook &book, rmwork::ChoicePanel *p) {
    const rmwork::Decklists lists = rmwork::read_decklists(m.stream(view));
    const agents::Table pend = v.table("pending");
    int chooser = v.seat();
    if (pend.rows()) {
        const uint64_t c = pend.u(0, pend.col("chooser"));
        chooser = c == 1 ? 0 : c == 2 ? 1 : chooser;
    }
    const int opp = chooser == 1 ? 0 : 1;
    const std::string opp_name = opp == 0 ? "Player A" : "Player B";
    p->groups = {opp_name + "'s main deck", opp_name + "'s sideboard", "Your deck", "Other cards"};
    const std::string lead = "Name card: ";
    for (int row : d.panel_rows) {
        rmwork::ChoiceItem it;
        it.row = row;
        it.card = row_card(v, static_cast<size_t>(row));
        std::string text = v.description(static_cast<size_t>(row));
        if (text.rfind(lead, 0) == 0) text = text.substr(lead.size());
        it.text = card_name(book, it.card, text);
        const rmwork::CardFace *f = it.card >= 0 ? book.face(it.card, false) : nullptr;
        if (f) it.type_line = f->type_line;
        const size_t o = static_cast<size_t>(opp), c = static_cast<size_t>(chooser == 1 ? 1 : 0);
        it.group = lists.known[o] && lists.lists[o].main_of(it.card) > 0                                            ? 0
                   : lists.known[o] && lists.lists[o].side_of(it.card) > 0                                          ? 1
                   : lists.known[c] && (lists.lists[c].main_of(it.card) > 0 || lists.lists[c].side_of(it.card) > 0) ? 2
                                                                                                                    : 3;
        it.note = "row " + std::to_string(row);
        p->items.push_back(it);
    }
    std::stable_sort(p->items.begin(), p->items.end(), [](const rmwork::ChoiceItem &a, const rmwork::ChoiceItem &b) {
        return a.group != b.group ? a.group < b.group : lower(a.text) < lower(b.text);
    });
    p->searchable = true;
    p->title = "Name a card \xC2\xB7 " + std::to_string(d.panel_rows.size()) + " candidates";
    p->note = "The engine's candidate set: the distinct cards of the decklists the effect may name, not every card";
}

// A decision whose buttons overflow the prompt line: its rows, searchable.
static void read_list(const agents::DecisionView &v, const rmwork::TableDecision &d, rmwork::ChoicePanel *p) {
    for (int row : d.panel_rows) {
        rmwork::ChoiceItem it;
        it.row = row;
        it.text = v.description(static_cast<size_t>(row));
        it.card = row_card(v, static_cast<size_t>(row));
        it.note = "row " + std::to_string(row);
        p->items.push_back(it);
    }
    p->searchable = true;
    p->title = d.prompt.empty() ? std::string("Choose") : d.prompt;
    p->note = "Type to filter; click a choice";
}

namespace rmwork {

const ChoiceItem *ChoicePanel::item_of_row(int row) const {
    for (const ChoiceItem &it : items)
        if (row >= 0 && it.row == row) return &it;
    return nullptr;
}

bool choice_item_matches(const ChoiceItem &it, const std::string &filter) {
    if (filter.empty()) return true;
    const std::string f = lower(filter);
    return lower(it.text).find(f) != std::string::npos || lower(it.type_line).find(f) != std::string::npos;
}

ChoicePanel read_choice_panel(const Match &m, size_t index, int view, const agents::DecisionView &v,
    const TableDecision &d, const CardBook &book, int chosen_row) {
    ChoicePanel p;
    if (!d.paneled()) return p;
    p.active = true;
    p.kind = d.kind;
    for (size_t i = 0; i < v.n(); ++i) p.row_text.push_back(v.description(i));
    p.chosen_row = chosen_row >= 0 && static_cast<size_t>(chosen_row) < v.n() ? chosen_row : -1;
    p.source_card = pending_card(v);
    p.title = card_name(book, p.source_card, "");
    switch (d.kind) {
        case DecisionKind::MODES:
            if (p.title.empty()) p.title = "Modal spell";
            read_modes(m, index, view, v, d, &p);
            break;
        case DecisionKind::ORDER:
            read_order(v, d, &p);
            break;
        case DecisionKind::NAME:
            read_names(m, view, v, d, book, &p);
            break;
        default:
            read_list(v, d, &p);
            break;
    }
    return p;
}

}  // namespace rmwork
