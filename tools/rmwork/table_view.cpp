#include "table_view.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>

#include "../../src/agents/decision_view.h"
#include "imgui_internal.h"
#include "magnifier.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

namespace {

// Card content colours of the state badges (on the card, like its frame).
const ImU32 PT_BG = rmwork::hex_color(0xf2f2f2);
const ImU32 PT_INK = rmwork::hex_color(0x000000);
const ImU32 PT_UP = rmwork::hex_color(0x0a6e25);
const ImU32 PT_DOWN = rmwork::hex_color(0xb3261e);
const ImU32 PT_MIXED = rmwork::hex_color(0x1f55b0);
const ImU32 DAMAGE_BG = rmwork::hex_color(0xb3261e);
const ImU32 WHITE = rmwork::hex_color(0xffffff);
const ImU32 SHADOW = rmwork::hex_color(0x000000, 140);
// Over a card that is no choice of a targeting or payment decision.
const ImU32 DIM_OVER = rmwork::hex_color(0x000000, 125);
// The hand's band over the table background (shades of it, like the cards' shadows).
const ImU32 HAND_SHADE = rmwork::hex_color(0x000000, 46);
const ImU32 RULE_DARK = rmwork::hex_color(0x000000, 115);
const ImU32 RULE_LIGHT = rmwork::hex_color(0xffffff, 31);
// A priority decision's playable card: a thin outline of GLOW at this alpha, thicker under the mouse.
const unsigned WAY_ALPHA = 200;
const float WAY_THICKNESS = 1.25f;
const float WAY_HOT_THICKNESS = 2.5f;
// Below a step's label in Play: its stop dot.
const float STOP_DOT_H = 9;
// The prompt line: its height beyond a line of text, and a button's padding beyond its label.
const float PROMPT_PAD = 8;
const float BUTTON_PAD = 16;

// One piece of the middle strip's first line.
struct HeaderPart {
        std::string text;
        bool dim = false;
        int seat_chip = -1;  // a seat chip after the text
};

}  // namespace

static ImVec2 at(ImVec2 o, float x, float y);
static ImVec2 rmin(ImVec2 o, const rmwork::TRect &r);
static ImVec2 rmax(ImVec2 o, const rmwork::TRect &r);
static std::vector<HeaderPart> header_parts(const rmwork::TableModel &m);
static float parts_width(const std::vector<HeaderPart> &parts);
static float seat_chip_width();
static float step_cell_width();
static std::string stop_tooltip(int step, bool own_turn, bool on);
static float badge(ImDrawList *dl, ImVec2 pos, bool right, bool bottom, const std::string &text, float size, ImU32 bg,
    ImU32 fg, ImU32 border, float rounding);
static ImU32 pt_color(const rmwork::TableObject &o);
static void outline(ImDrawList *dl, ImVec2 a, ImVec2 b, float r, ImU32 color, float pad, float thickness);
static void arrow(ImDrawList *dl, ImVec2 from, ImVec2 to, ImU32 color, float thickness);
static void heart(ImDrawList *dl, ImVec2 c, float s, ImU32 color);
static void seat_chip(ImDrawList *dl, ImVec2 pos, int seat);
static std::string player_text(const rmwork::TablePlayer &p);
static std::vector<std::string> state_lines(const rmwork::TableModel &m, const rmwork::TableObject &o, int stacked);
static ImVec2 rect_center(ImVec2 o, const rmwork::TRect &r);
static void number_badge(ImDrawList *dl, ImVec2 c, float r, int n, ImU32 bg);
static std::string join_labels(const std::vector<rmwork::ChosenTarget> &chosen);
static std::string short_name(const std::string &name);
static bool combat_kind(rmwork::DecisionKind k);
static const char *source_tag(rmwork::DecisionKind k);
static std::string stop_text(rmwork::PlanStop stop, const std::string &creature);
static std::string lower_text(std::string s);
static std::vector<std::string> split_names(const std::string &spec);
static bool dims_board(const rmwork::TableDecision &d);
static std::string join_names(const std::vector<std::string> &names);
static ImU32 with_alpha(ImU32 color, unsigned alpha);

static ImVec2 at(ImVec2 o, float x, float y) {
    return ImVec2(o.x + x, o.y + y);
}

static ImVec2 rmin(ImVec2 o, const rmwork::TRect &r) {
    return ImVec2(o.x + r.x0, o.y + r.y0);
}

static ImVec2 rmax(ImVec2 o, const rmwork::TRect &r) {
    return ImVec2(o.x + r.x1, o.y + r.y1);
}

static ImVec2 rect_center(ImVec2 o, const rmwork::TRect &r) {
    return ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + (r.y0 + r.y1) / 2);
}

// A filled circle at `c` with `n` in it.
static void number_badge(ImDrawList *dl, ImVec2 c, float r, int n, ImU32 bg) {
    const std::string t = std::to_string(n);
    dl->AddCircleFilled(c, r, bg);
    dl->AddCircle(c, r, SHADOW, 0, 1.5f);
    ImFont *font = ImGui::GetFont();
    const float fs = r * 1.3f;
    const ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, t.c_str());
    dl->AddText(font, fs, ImVec2(c.x - ts.x / 2, c.y - ts.y / 2), rmwork::tok().TEXT, t.c_str());
}

// "Orcish Bowmasters, Player B (20 life)".
static std::string join_labels(const std::vector<rmwork::ChosenTarget> &chosen) {
    std::string out;
    for (const rmwork::ChosenTarget &c : chosen) out += (out.empty() ? "" : ", ") + c.label;
    return out;
}

// A planeswalker's name up to its comma: "Kaito, Bane of Nightmares" -> "Kaito".
static std::string short_name(const std::string &name) {
    const size_t comma = name.find(',');
    return comma == std::string::npos ? name : name.substr(0, comma);
}

static bool combat_kind(rmwork::DecisionKind k) {
    return rmwork::declares_attackers(k) || rmwork::declares_blockers(k) || k == rmwork::DecisionKind::DAMAGE;
}

// The tag on the decision's source: the creature being declared or assigning its damage, else "Source".
static const char *source_tag(rmwork::DecisionKind k) {
    switch (k) {
        case rmwork::DecisionKind::ATTACK_TARGET:
            return "Attacking";
        case rmwork::DecisionKind::BLOCK_TARGET:
            return "Blocking";
        case rmwork::DecisionKind::DAMAGE:
            return "Assigning";
        default:
            return "Source";
    }
}

static std::string lower_text(std::string s) {
    for (char &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    return s;
}

// Whether the decision dims what it does not act with: a decision answered by clicking the Table's
// objects (targets, payments, combat, the opening hand, cards); one answered in the choice panel or on
// the prompt line only when some of its rows name objects on the Table.
static bool dims_board(const rmwork::TableDecision &d) {
    using rmwork::DecisionKind;
    if (!d.prompted()) return false;
    switch (d.kind) {
        case DecisionKind::MODES:
        case DecisionKind::NUMBER:
        case DecisionKind::ORDER:
        case DecisionKind::NAME:
        case DecisionKind::UNLESS:
        case DecisionKind::CHOICE:
            return !d.picks.empty();
        case DecisionKind::PRIORITY:
            return false;
        default:
            return true;
    }
}

static ImU32 with_alpha(ImU32 color, unsigned alpha) {
    return (color & ~IM_COL32_A_MASK) | (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

static std::string join_names(const std::vector<std::string> &names) {
    std::string out;
    for (const std::string &n : names) out += (out.empty() ? "" : ", ") + n;
    return out;
}

// A plan spec's comma-separated items.
static std::vector<std::string> split_names(const std::string &spec) {
    std::vector<std::string> out;
    size_t from = 0;
    while (from <= spec.size()) {
        size_t end = spec.find(',', from);
        if (end == std::string::npos) end = spec.size();
        if (end > from) out.push_back(spec.substr(from, end - from));
        from = end + 1;
    }
    return out;
}

// Why sending a plan stopped, for the status bar.
static std::string stop_text(rmwork::PlanStop stop, const std::string &creature) {
    switch (stop) {
        case rmwork::PlanStop::NO_TARGET:
            return "Sending stopped: " + creature + " cannot attack what was planned; choose what it attacks";
        case rmwork::PlanStop::CANNOT_BLOCK:
            return "Sending stopped: " + creature +
                   " cannot block the attacker planned; choose which attacker it blocks (the rest of the plan is "
                   "kept)";
        case rmwork::PlanStop::UNPLANNED:
            return "Sending stopped: the engine asks about " + creature + ", which the plan does not hold";
        case rmwork::PlanStop::ENDED:
            return "The declaration ended before " + creature + " was declared";
        case rmwork::PlanStop::NONE:
            break;
    }
    return "";
}

// "Turn 9 · Player A's turn", the step, "Deciding:" with the seat chip, day or night; the opening
// hands, sideboarding and the match's end have their own words.
static std::vector<HeaderPart> header_parts(const rmwork::TableModel &m) {
    std::vector<HeaderPart> out;
    const auto seat = [](int s) { return std::string(s == 0 ? "Player A" : "Player B"); };
    if (m.terminal) {
        out.push_back({"Match over", false, -1});
        return out;
    }
    if (m.sideboarding) {
        out.push_back({"Sideboarding", false, -1});
    } else if (m.pregame || m.turn == 0) {
        out.push_back({"Pregame", false, -1});
        out.push_back({"Opening hands", true, -1});
    } else {
        out.push_back(
            {"Turn " + std::to_string(m.turn) + (m.active >= 0 ? " \xC2\xB7 " + seat(m.active) + "'s turn" : ""), false,
                -1});
        if (!m.step_name.empty()) out.push_back({m.step_name, false, -1});
    }
    if (m.decider >= 0) out.push_back({"Deciding:", true, m.decider});
    if (!m.day_night.empty()) out.push_back({m.day_night, true, -1});
    return out;
}

static float parts_width(const std::vector<HeaderPart> &parts) {
    float w = 0;
    for (const HeaderPart &p : parts) {
        w += ImGui::CalcTextSize(p.text.c_str()).x + 14;
        if (p.seat_chip >= 0) w += 4 + seat_chip_width();
    }
    return w;
}

static float seat_chip_width() {
    return ImGui::CalcTextSize("B").x + 12;
}

static float step_cell_width() {
    float w = 0;
    for (int i = 0; i < rmwork::STOP_STEP_COUNT; ++i)
        w = std::max(w, ImGui::CalcTextSize(rmwork::stop_step(i).abbrev).x);
    return w + 8;
}

// "Stop in Upkeep in your turn: on.", then "Click to turn it off (the same stop as Settings → Stops)."
static std::string stop_tooltip(int step, bool own_turn, bool on) {
    return std::string("Stop in ") + rmwork::stop_step(step).name +
           (own_turn ? " in your turn" : " in the opponent's turn") + ": " + (on ? "on" : "off") +
           ".\nClick to turn it " + (on ? "off" : "on") + " (the same stop as Settings \xE2\x86\x92 Stops).";
}

// A text badge with its corner at `pos` (`right` / `bottom`: `pos` is its right / bottom edge);
// returns its height.
static float badge(ImDrawList *dl, ImVec2 pos, bool right, bool bottom, const std::string &text, float size, ImU32 bg,
    ImU32 fg, ImU32 border, float rounding) {
    ImFont *font = ImGui::GetFont();
    const ImVec2 ts = font->CalcTextSizeA(size, FLT_MAX, 0, text.c_str());
    const float pw = std::max(2.0f, size * 0.25f), ph = 1;
    const float w = ts.x + 2 * pw, h = ts.y + 2 * ph;
    const ImVec2 a(right ? pos.x - w : pos.x, bottom ? pos.y - h : pos.y);
    dl->AddRectFilled(a, ImVec2(a.x + w, a.y + h), bg, rounding);
    if (border) dl->AddRect(a, ImVec2(a.x + w, a.y + h), border, rounding);
    dl->AddText(font, size, ImVec2(a.x + pw, a.y + ph), fg, text.c_str());
    return h;
}

// Green above the printed P/T, red below, blue for one up and one down.
static ImU32 pt_color(const rmwork::TableObject &o) {
    if (!o.pt_differs) return PT_INK;
    const int dp = o.power - o.printed_power, dt = o.toughness - o.printed_toughness;
    if (dp >= 0 && dt >= 0) return PT_UP;
    if (dp <= 0 && dt <= 0) return PT_DOWN;
    return PT_MIXED;
}

static void outline(ImDrawList *dl, ImVec2 a, ImVec2 b, float r, ImU32 color, float pad, float thickness) {
    dl->AddRect(ImVec2(a.x - pad, a.y - pad), ImVec2(b.x + pad, b.y + pad), color, r + pad, 0, thickness);
}

// A line with an arrowhead at `to`, over a darker line so it shows on any background.
static void arrow(ImDrawList *dl, ImVec2 from, ImVec2 to, ImU32 color, float thickness) {
    const float dx = to.x - from.x, dy = to.y - from.y;
    const float len = std::sqrt(dx * dx + dy * dy);
    if (len < 1) return;
    const float ux = dx / len, uy = dy / len;
    const float head = thickness * 4;
    const ImVec2 base(to.x - ux * head, to.y - uy * head);
    const ImVec2 l(base.x - uy * head * 0.6f, base.y + ux * head * 0.6f),
        r(base.x + uy * head * 0.6f, base.y - ux * head * 0.6f);
    dl->AddLine(from, base, SHADOW, thickness + 2);
    dl->AddLine(from, base, color, thickness);
    dl->AddTriangleFilled(to, l, r, color);
    dl->AddTriangle(to, l, r, SHADOW, 1);
}

static void heart(ImDrawList *dl, ImVec2 c, float s, ImU32 color) {
    const float r = s * 0.27f;
    dl->AddCircleFilled(ImVec2(c.x - r * 0.95f, c.y - r * 0.5f), r, color);
    dl->AddCircleFilled(ImVec2(c.x + r * 0.95f, c.y - r * 0.5f), r, color);
    dl->AddTriangleFilled(ImVec2(c.x - r * 1.9f, c.y - r * 0.25f), ImVec2(c.x + r * 1.9f, c.y - r * 0.25f),
        ImVec2(c.x, c.y + s * 0.5f), color);
}

// The seat's letter on its colour.
static void seat_chip(ImDrawList *dl, ImVec2 pos, int seat) {
    const char *letter = seat == 0 ? "A" : "B";
    const float h = ImGui::GetTextLineHeight();
    dl->AddRectFilled(
        pos, ImVec2(pos.x + seat_chip_width(), pos.y + h), seat == 0 ? rmwork::tok().SEAT_A : rmwork::tok().SEAT_B, 2);
    dl->AddText(ImVec2(pos.x + 6, pos.y), rmwork::tok().TEXT, letter);
}

// "Player B · burn" and the counts after the life.
static std::string player_text(const rmwork::TablePlayer &p) {
    std::string out = "hand " + std::to_string(p.hand) + "  library " + std::to_string(p.library);
    if (p.poison) out += "  poison " + std::to_string(p.poison);
    if (p.energy) out += "  energy " + std::to_string(p.energy);
    for (const rmwork::TableCounter &c : p.counters) out += "  " + c.kind + " " + std::to_string(c.count);
    if (p.monarch) out += "  monarch";
    if (p.city_blessing) out += "  city's blessing";
    if (p.speed >= 0) out += "  speed " + std::to_string(p.speed);
    return out;
}

// What the magnifier lists under a card: its state on the table; `stacked` is the size of the stack
// of identical permanents it stands in (0 none).
static std::vector<std::string> state_lines(const rmwork::TableModel &m, const rmwork::TableObject &o, int stacked) {
    std::vector<std::string> out;
    std::string first = o.in_hand ? std::string("In ") + (o.owner == 0 ? "Player A" : "Player B") + "'s hand"
                                  : std::string("Controlled by ") + (o.controller == 0 ? "Player A" : "Player B");
    if (o.token) first += " \xC2\xB7 token";
    if (o.face_down) first += " \xC2\xB7 face down";
    out.push_back(first);
    std::string status;
    const auto add = [&status](const char *s) { status += (status.empty() ? "" : " \xC2\xB7 ") + std::string(s); };
    if (o.tapped) add("tapped");
    if (o.sick) add("summoning sick");
    if (o.phased_out) add("phased out");
    if (o.attacking) add(o.blocked ? "attacking, blocked" : "attacking");
    if (o.blocking) add("blocking");
    if (!status.empty()) out.push_back(status);
    if (o.has_pt) {
        std::string pt = "P/T " + std::to_string(o.power) + "/" + std::to_string(o.toughness);
        if (o.pt_differs)
            pt += " (printed " + std::to_string(o.printed_power) + "/" + std::to_string(o.printed_toughness) + ")";
        out.push_back(pt);
    }
    if (o.damage) out.push_back("Damage " + std::to_string(o.damage) + (o.deathtouch_damage ? " (deathtouch)" : ""));
    if (o.loyalty >= 0) out.push_back("Loyalty " + std::to_string(o.loyalty));
    for (const rmwork::TableCounter &c : o.counters)
        out.push_back("Counters: " + c.kind + " \xC3\x97" + std::to_string(c.count));
    for (const std::string &t : o.tags) out.push_back(t);
    if (stacked > 1)
        out.push_back("One of " + std::to_string(stacked) + " identical, stacked to fit the Table's width");
    if (o.attached_to) {
        const int h = m.find(o.attached_to);
        out.push_back("Attached to " + (h >= 0 && !m.objects[static_cast<size_t>(h)].name.empty()
                                               ? m.objects[static_cast<size_t>(h)].name
                                               : std::string("a face-down card")));
    }
    return out;
}

namespace rmwork {

TableView::TableView() : renderer_(&text_renderer_) {}

void TableView::set_renderer(const CardRenderer *r) {
    renderer_ = r ? r : &text_renderer_;
    picker_view_.set_renderer(r);
    panel_view_.set_renderer(r);
}

const CardBook &TableView::book(const Match &m, int view) {
    if (book_match_ != &m || book_dump_ != m.dump_path) {
        book_match_ = &m;
        book_dump_ = m.dump_path;
        book_built_ = {{false, false, false}};
    }
    const size_t v = static_cast<size_t>(view);
    if (!book_built_[v]) {
        books_[v].build(m.stream(view));
        book_built_[v] = true;
    }
    return books_[v];
}

int TableView::take_answer() {
    const int a = answer_;
    answer_ = -1;
    return a;
}

std::string TableView::take_status() {
    std::string s = backdrop_.take_message();
    if (s.empty()) s.swap(status_);
    return s;
}

bool TableView::refresh(ReplaySession &s, bool answer) {
    const Match *m = s.match();
    // The session's generation moves with the match, the cursor and the viewer.
    if (have_model_ && m == model_match_ && s.generation() == model_generation_ && answer == model_answer_ &&
        plan_.generation() == model_plan_generation_)
        return true;
    std::string err;
    const auto frame = m->frame(s.cursor(), s.view(), &err);
    if (!frame) {
        error_ = err;
        have_model_ = false;
        model_match_ = nullptr;
        return false;
    }
    const DecisionInfo *d = s.current();
    model_ = build_table_model(m->stream(s.view()), book(*m, s.view()), *frame, s.view(), d ? d->decider : -1,
        d && d->phase == "Opening hand");
    model_match_ = m;
    model_generation_ = s.generation();
    model_view_ = s.view();
    have_model_ = true;
    const agents::DecisionView dv(m->stream(s.view()), *frame, model_.bottom);
    decision_ = read_table_decision(dv, model_, &book(*m, s.view()));
    if (decision_.kind == DecisionKind::TARGET)
        decision_.chosen = chosen_targets(*m, s.cursor(), s.view(), decision_.source);
    // The row chosen here, once it is (Replay, an earlier decision of Play).
    if (!answer && d && d->reply >= 0 && static_cast<size_t>(d->reply) < decision_.row_text.size())
        decision_.chosen_row = d->reply;
    picker_ = CardPicker();
    if (decision_.kind == DecisionKind::CARDS) {
        const CardBook &b = book(*m, s.view());
        picker_ = read_card_picker(*m, s.cursor(), s.view(), dv, b,
            build_zone_model(m->stream(s.view()), b, *frame, s.view()), decision_.chosen_row);
        picker_view_.request_scroll();
    }
    panel_ = read_choice_panel(*m, s.cursor(), s.view(), dv, decision_, book(*m, s.view()), decision_.chosen_row);
    // The stepper starts at the value chosen here (Replay), else at the first row, at each new decision.
    const std::string number_key = std::to_string(reinterpret_cast<uintptr_t>(m)) + "/" + std::to_string(s.cursor()) +
                                   "/" + std::to_string(s.view());
    if (number_key != number_key_ || (decision_.chosen_row >= 0 && decision_.kind == DecisionKind::NUMBER)) {
        number_key_ = number_key;
        number_index_ = decision_.chosen_row >= 0 && decision_.kind == DecisionKind::NUMBER ? decision_.chosen_row : 0;
    }
    if (answer && (decision_.kind == DecisionKind::BOTTOM || decision_.plan_owed > 1)) {
        CardPlanKey key;
        key.game = d ? d->game : -1;
        key.turn = decision_.kind == DecisionKind::BOTTOM ? -1 : model_.turn;
        key.purpose = decision_.kind == DecisionKind::BOTTOM ? "bottom" : "discard";
        card_plan_.follow(key);
    }
    model_answer_ = answer;
    if (answer && (declares_attackers(decision_.kind) || declares_blockers(decision_.kind))) {
        PlanKey key;
        key.game = d ? d->game : -1;
        key.turn = model_.turn;
        key.blockers = declares_blockers(decision_.kind);
        plan_.follow(key);
    }
    apply_plan_to_model();
    model_plan_generation_ = plan_.generation();
    error_.clear();
    if (mag_.handle() && model_.find(mag_.handle()) < 0) mag_.close();
    return true;
}

TableMetrics TableView::measure(bool stop_dots) const {
    TableMetrics mx;
    mx.line_h = ImGui::GetTextLineHeight();
    mx.header_w = parts_width(header_parts(model_));
    float w = 0;
    for (int i = 0; i < STOP_STEP_COUNT; ++i) w += step_cell_width() + 1 + (step_gap_before(i) ? STEP_GAP : 0);
    mx.steps_w = w;
    mx.steps_h = mx.line_h + 2 + (stop_dots ? STOP_DOT_H : 0);
    if (decision_.prompted()) {
        mx.prompt_h = mx.line_h + PROMPT_PAD;
        for (const PromptButton &b : buttons_)
            mx.button_w.push_back(ImGui::CalcTextSize(b.label.c_str()).x + BUTTON_PAD);
    }
    return mx;
}

const TablePile *TableView::pile_of(const PlacedCard &c) const {
    if (c.in_hand || c.row < 0) return nullptr;
    const std::vector<TablePile> &piles = layout_.piles[static_cast<size_t>(c.row)];
    return c.pile >= 0 && static_cast<size_t>(c.pile) < piles.size() ? &piles[static_cast<size_t>(c.pile)] : nullptr;
}

CardLook TableView::look_of(const TableObject &o) const {
    CardLook look;
    const CardBook *b =
        book_built_[static_cast<size_t>(model_view_)] ? &books_[static_cast<size_t>(model_view_)] : nullptr;
    if (!o.face_down && o.card >= 0 && b) {
        look.face = b->face(o.card, o.back);
        look.other = b->face(o.card, !o.back);
    }
    if (o.face_down && !o.name.empty()) look.note = o.name;
    look.colors = o.colors;
    look.land = o.land || (look.face && look.face->land && !o.creature);
    look.token = o.token;
    look.land_mana =
        look.land ? basic_land_mana(o.subtypes.empty() && look.face ? look.face->subtypes : o.subtypes) : 0;
    return look;
}

TableView::Hit TableView::hit_test(ImVec2 p, bool steps) const {
    Hit h;
    for (size_t i = 0; i < layout_.prompt_buttons.size(); ++i)
        if (layout_.prompt_buttons[i].contains(p.x, p.y)) {
            h.button = static_cast<int>(i);
            return h;
        }
    for (size_t i = 0; steps && i < layout_.step_cells.size(); ++i)
        if (layout_.step_cells[i].contains(p.x, p.y)) {
            h.step = static_cast<int>(i);
            return h;
        }
    for (int i = static_cast<int>(layout_.stack_items.size()) - 1; i >= 0; --i)
        if (layout_.stack_items[static_cast<size_t>(i)].contains(p.x, p.y)) {
            h.stack = i;
            return h;
        }
    for (int s = 0; s < 2; ++s)
        if (chip_rects_[static_cast<size_t>(s)].contains(p.x, p.y)) {
            h.seat = s;
            return h;
        }
    for (int i = static_cast<int>(layout_.cards.size()) - 1; i >= 0; --i)
        if (layout_.cards[static_cast<size_t>(i)].rect.contains(p.x, p.y)) {
            h.card = i;
            return h;
        }
    return h;
}

void TableView::draw(const TableDraw &d) {
    ReplaySession &s = *d.replay;
    answering_ = d.answer && !plan_.sending();
    if (!refresh(s, d.answer)) {
        centered_text_block("Frame not readable", error_.c_str(), 46.0f * ImGui::CalcTextSize("M").x);
        return;
    }
    ImGuiWindow *win = ImGui::GetCurrentWindow();
    const ImVec2 o = win->InnerRect.Min;
    const float W = win->InnerRect.GetWidth(), H = win->InnerRect.GetHeight();
    if (W < 8 || H < 8) return;
    // An item under the whole panel: a drag on the table does not move a floating window.
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::InvisibleButton("##table", ImVec2(std::max(1.0f, avail.x), std::max(1.0f, avail.y)),
        ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(win->InnerRect.Min, win->InnerRect.Max, false);
    backdrop_.draw(dl, win->InnerRect.Min, win->InnerRect.Max, *d.background);
    const bool stop_dots = d.play && d.stops;
    apply_force_plan();
    build_buttons();
    layout_ = layout_table(model_, W, H, measure(stop_dots));

    if (d.links) {
        d.links->set_menu(model_.menu);
        std::map<uint64_t, std::vector<StackTarget>> targets;
        for (const TableStackItem &it : model_.stack) targets[it.handle] = it.targets;
        d.links->set_stack_targets(targets);
    }
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    const bool over =
        ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem) && win->InnerRect.Contains(mouse);
    // Chip rectangles are laid out as the chips draw; last frame's serve the hit test.
    Hit hit = over ? hit_test(ImVec2(mouse.x - o.x, mouse.y - o.y), stop_dots) : Hit();
    if (force_stack_ >= 0 && static_cast<size_t>(force_stack_) < layout_.stack_items.size()) {
        hit = Hit();
        hit.stack = force_stack_;
    }
    if (stop_dots && force_step_ >= 0 && static_cast<size_t>(force_step_) < layout_.step_cells.size()) {
        hit = Hit();
        hit.step = force_step_;
    }
    bool forced_card = false;
    for (size_t i = 0; !force_card_.empty() && !forced_card && i < layout_.cards.size(); ++i) {
        const std::string &n = model_.objects[static_cast<size_t>(layout_.cards[i].object)].name;
        if (n.empty() || lower_text(n).find(lower_text(force_card_)) == std::string::npos) continue;
        hit = Hit();
        hit.card = static_cast<int>(i);
        forced_card = true;
    }
    hovered_handle_ =
        hit.card >= 0 ? model_.objects[static_cast<size_t>(layout_.cards[static_cast<size_t>(hit.card)].object)].handle
                      : 0;
    hovered_stack_ = hit.stack;
    hovered_step_ = hit.step;
    hovered_button_ = hit.button;
    const bool clicked = over && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    const bool answered = clicked && answering_ && hit.step < 0 && answer_click(hit);
    if (answering_ && planning() && ImGui::IsMouseReleased(ImGuiMouseButton_Left)) plan_release(hit);
    if (over || forced_card) hover_hint(hit, d);
    if (hit.step >= 0 && !stop_settable(hit.step)) {
        ImGui::SetTooltip("%s", stop_step(hit.step).no_priority);
    } else if (hit.step >= 0) {
        // A click on a step toggles its stop for the side whose turn the Table shows.
        const bool own = model_.active == model_.bottom;
        bool *stop = stop_flag(d.stops, own, hit.step);
        if (over && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) *stop = !*stop;
        ImGui::SetTooltip("%s", stop_tooltip(hit.step, own, *stop).c_str());
    } else if (hit.card >= 0) {
        const PlacedCard &c = layout_.cards[static_cast<size_t>(hit.card)];
        const TablePile *p = pile_of(c);
        if (p && p->stacked) {
            const std::string &name = model_.objects[static_cast<size_t>(c.object)].name;
            ImGui::SetTooltip("%s \xC3\x97%d: identical permanents stacked to fit the Table's width", name.c_str(),
                static_cast<int>(p->objects.size()));
        }
    }
    if (d.links && hit.step < 0) {
        if (hit.button >= 0 && buttons_[static_cast<size_t>(hit.button)].row >= 0)
            d.links->hover_menu_row(buttons_[static_cast<size_t>(hit.button)].row);
        if (hit.card >= 0) d.links->hover_object(hovered_handle_);
        if (hit.stack >= 0) d.links->hover_object(model_.stack[static_cast<size_t>(hit.stack)].handle);
        if (hit.seat >= 0) d.links->hover_player(hit.seat);
        if (clicked && !answered && hit.button < 0) {
            LinkItem item;
            if (hit.card >= 0 || hit.stack >= 0) {
                item.kind = LinkItem::Kind::OBJECT;
                item.handle = hit.card >= 0 ? hovered_handle_ : model_.stack[static_cast<size_t>(hit.stack)].handle;
                d.links->select(item);
            } else if (hit.seat >= 0) {
                item.kind = LinkItem::Kind::PLAYER;
                item.seat = hit.seat;
                d.links->select(item);
            } else {
                d.links->clear_selection();
            }
        }
    }

    // The hand's band: a shade under the bottom seat's hand and chip, ruled off from the battlefield.
    const float band_y = o.y + layout_.hand.y0 - 5;
    dl->AddRectFilled(ImVec2(o.x, band_y), win->InnerRect.Max, HAND_SHADE);
    dl->AddLine(ImVec2(o.x, band_y), ImVec2(o.x + W, band_y), RULE_DARK);
    dl->AddLine(ImVec2(o.x, band_y + 1), ImVec2(o.x + W, band_y + 1), RULE_LIGHT);
    draw_cards(dl, o, d);
    draw_mid(dl, o, d);
    // Over the middle strip: blockers and attackers face each other across it.
    draw_combat(dl, o, d);
    draw_chips(dl, o, d);
    draw_attribution(dl, o, d);
    if (clicked && attr_legend_.w() > 0 && d.attr_layer && mouse.x - o.x >= attr_legend_.x0 &&
        mouse.x - o.x <= attr_legend_.x1 && mouse.y - o.y >= attr_legend_.y0 && mouse.y - o.y <= attr_legend_.y1)
        *d.attr_layer = !*d.attr_layer;
    dl->PopClipRect();
    draw_picker(d);
    draw_panel(d);
    update_magnifier(hit, d, over);
    apply_force_pick(d);
    draw_pick_popup(d);
}

const DecisionPick *TableView::pick_at(const Hit &hit) const {
    if (decision_.kind == DecisionKind::NONE) return nullptr;
    if (hit.stack >= 0) return decision_.pick_of_object(model_.stack[static_cast<size_t>(hit.stack)].handle);
    if (hit.seat >= 0) return decision_.pick_of_seat(hit.seat);
    if (hit.card < 0) return nullptr;
    // A stack of identical permanents answers with the first of its members that is a pick.
    const PlacedCard &c = layout_.cards[static_cast<size_t>(hit.card)];
    if (const DecisionPick *p = decision_.pick_of_object(model_.objects[static_cast<size_t>(c.object)].handle))
        return p;
    const TablePile *pile = pile_of(c);
    if (!pile || !pile->stacked) return nullptr;
    for (int k : pile->objects)
        if (const DecisionPick *p = decision_.pick_of_object(model_.objects[static_cast<size_t>(k)].handle)) return p;
    return nullptr;
}

bool TableView::answer_click(const Hit &hit) {
    if (hit.button >= 0) return press_button(buttons_[static_cast<size_t>(hit.button)]);
    if (planning()) return plan_press(hit);
    if (card_planning()) return card_plan_press(hit);
    const DecisionPick *p = pick_at(hit);
    if (!p || p->rows.empty()) return false;
    if (p->rows.size() == 1) {
        answer_ = p->rows[0];
        return true;
    }
    open_pick_popup(*p);
    return true;
}

// The popup of `p`'s rows, titled with the object's name (or the player's).
void TableView::open_pick_popup(const DecisionPick &p) {
    const int obj = p.handle ? model_.find(p.handle) : -1;
    open_rows_popup(p.rows, obj >= 0 && !model_.objects[static_cast<size_t>(obj)].name.empty()
                                ? model_.objects[static_cast<size_t>(obj)].name
                            : p.handle == 0 ? model_.players[static_cast<size_t>(p.seat == 1 ? 1 : 0)].name
                                            : std::string("Choose"));
}

void TableView::open_rows_popup(const std::vector<int> &rows, const std::string &title) {
    popup_rows_ = rows;
    popup_title_ = title;
    popup_generation_ = model_generation_;
    open_popup_ = true;
}

std::string TableView::row_label(int row) const {
    const std::string &text = decision_.row_text[static_cast<size_t>(row)];
    if (decision_.kind != DecisionKind::PRIORITY || static_cast<size_t>(row) >= decision_.costs.size()) return text;
    return way_label(text, decision_.costs[static_cast<size_t>(row)]);
}

// Screenshots: the pick popup of the first card named like force_pick_ with several rows.
void TableView::apply_force_pick(const TableDraw &d) {
    if (force_pick_generation_ != model_generation_) force_picked_ = false;
    if (force_pick_.empty() || force_picked_ || !answering_) return;
    std::string want = force_pick_;
    for (char &ch : want) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    for (const DecisionPick &p : decision_.picks) {
        const int obj = model_.find(p.handle);
        if (obj < 0 || p.rows.size() < 2) continue;
        std::string n = model_.objects[static_cast<size_t>(obj)].name;
        for (char &ch : n) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (n.find(want) == std::string::npos) continue;
        open_pick_popup(p);
        force_picked_ = true;
        force_pick_generation_ = model_generation_;
        return;
    }
}

// While the human chooses: what a click on a pick or a button does.
void TableView::hover_hint(const Hit &hit, const TableDraw &d) {
    if (!answering_ || hit.step >= 0) return;
    if (hit.button >= 0) {
        const PromptButton &b = buttons_[static_cast<size_t>(hit.button)];
        const bool blockers = decision_.kind == DecisionKind::BLOCKERS;
        switch (b.act) {
            case PromptButton::Act::SEND:
                if (decision_.kind == DecisionKind::BOTTOM) {
                    ImGui::SetTooltip("Click: put the planned cards on the bottom, one decision at a time");
                    break;
                }
                if (card_planning()) {
                    ImGui::SetTooltip("Click: discard the planned cards, one decision at a time");
                    break;
                }
                ImGui::SetTooltip(blockers ? "Click: declare the planned blocks (each blocker, then the attacker it "
                                             "blocks), then confirm"
                                           : "Click: declare the planned attackers (each with what it attacks), then "
                                             "confirm");
                break;
            case PromptButton::Act::ALL:
                ImGui::SetTooltip("Click: plan every creature that can attack to attack %s",
                    model_.players[static_cast<size_t>(model_.top)].name.c_str());
                break;
            case PromptButton::Act::CLEAR:
                ImGui::SetTooltip(card_planning() ? "Click: drop the plan"
                                                  : "Click: drop the plan (creatures declared already stay declared)");
                break;
            case PromptButton::Act::UNDO:
                ImGui::SetTooltip("Click: take back the last card planned");
                break;
            case PromptButton::Act::DEC:
                ImGui::SetTooltip("Click: one less (the rows offered bound it)");
                break;
            case PromptButton::Act::INC:
                ImGui::SetTooltip("Click: one more (the rows offered bound it)");
                break;
            case PromptButton::Act::POPUP:
                ImGui::SetTooltip("Click to choose one of %d ways to cast or activate a card off the Table",
                    static_cast<int>(b.rows.size()));
                break;
            case PromptButton::Act::ROW:
                if (decision_.kind == DecisionKind::PRIORITY && b.row == decision_.pass) {
                    std::string keys;
                    if (d.pass) keys = chord_label(d.pass);
                    if (d.to_end_of_turn)
                        keys += (keys.empty() ? "" : "; ") + chord_label(d.to_end_of_turn) +
                                " passes to the end of the turn";
                    ImGui::SetTooltip("Click: pass priority%s", keys.empty() ? "" : (" (" + keys + ")").c_str());
                } else {
                    ImGui::SetTooltip("Click: %s", row_label(b.row).c_str());
                }
                break;
        }
        return;
    }
    if (planning()) {
        const uint64_t h = hit_object(hit);
        const uint64_t armed = plan_.armed();
        std::string tip;
        if (decision_.kind == DecisionKind::ATTACKERS) {
            if (armed && hit.seat == model_.top) {
                tip = name_of(armed) + " attacks " + model_.players[static_cast<size_t>(model_.top)].name;
            } else if (armed && armed_target(h)) {
                tip = name_of(armed) + " attacks " + name_of(h);
            } else if (h && decision_.pick_of_object(h)) {
                tip =
                    plan_.attack_of(h) ? name_of(h) + " does not attack" : name_of(h) + " attacks (sent with the plan)";
            }
        } else if (h && decision_.pick_of_object(h)) {
            tip = plan_.block_of(h) || armed == h ? name_of(h) + " does not block"
                                                  : name_of(h) + " blocks: then click (or drag onto) an attacker";
        } else if (armed && armed_target(h)) {
            tip = name_of(armed) + " blocks " + name_of(h);
        }
        if (!tip.empty()) ImGui::SetTooltip("Click: %s", tip.c_str());
        return;
    }
    const DecisionPick *p = pick_at(hit);
    if (!p || p->rows.empty()) return;
    if (card_planning()) {
        const uint64_t h = hit_object(hit);
        const bool bottom = decision_.kind == DecisionKind::BOTTOM;
        ImGui::SetTooltip(card_plan_.order_of(h) ? "Click: keep %s (take it back from the plan)"
                                                 : (bottom ? "Click: put %s on the bottom (sent with the others)"
                                                           : "Click: discard %s (sent with the others)"),
            name_of(h).c_str());
        return;
    }
    if (p->rows.size() == 1) {
        ImGui::SetTooltip("Click: %s", row_label(p->rows[0]).c_str());
    } else if (decision_.kind == DecisionKind::PRIORITY) {
        std::string ways;
        for (int row : p->rows) ways += "\n  " + row_label(row);
        ImGui::SetTooltip("Click to choose one of %d:%s", static_cast<int>(p->rows.size()), ways.c_str());
    } else {
        ImGui::SetTooltip("Click to choose one of %d: %s \xE2\x80\xA6", static_cast<int>(p->rows.size()),
            decision_.row_text[static_cast<size_t>(p->rows[0])].c_str());
    }
}

// The rows of the object clicked, one to choose (Esc or a click elsewhere closes it).
void TableView::draw_pick_popup(const TableDraw &d) {
    if (open_popup_) {
        ImGui::OpenPopup("##table_pick");
        open_popup_ = false;
    }
    popup_items_.clear();
    popup_shown_ = ImGui::BeginPopup("##table_pick");
    if (!popup_shown_) return;
    if (popup_generation_ != model_generation_ || !d.answer) {
        ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
        popup_shown_ = false;
        return;
    }
    ImGui::TextDisabled("%s", popup_title_.c_str());
    ImGui::Separator();
    // At a priority decision each way's cost follows it in a column: pips, then the rest in words.
    const bool costs = decision_.kind == DecisionKind::PRIORITY;
    const float lh = ImGui::GetTextLineHeight();
    const float pd = lh * 0.95f, gap = 18;
    float text_w = 0, cost_w = 0;
    for (int row : popup_rows_) {
        const size_t r = static_cast<size_t>(row);
        text_w = std::max(text_w, ImGui::CalcTextSize(decision_.row_text[r].c_str()).x);
        if (!costs || r >= decision_.costs.size()) continue;
        const WayCost &c = decision_.costs[r];
        float w = c.pips.empty() ? 0 : mana_symbols_width(c.pips.size(), pd);
        if (!c.text.empty()) w += (w > 0 ? 6 : 0) + ImGui::CalcTextSize(c.text.c_str()).x;
        cost_w = std::max(cost_w, w);
    }
    const float width = text_w + (cost_w > 0 ? gap + cost_w : 0);
    for (int row : popup_rows_) {
        const size_t r = static_cast<size_t>(row);
        ImGui::PushID(row);
        const bool chosen = ImGui::Selectable(decision_.row_text[r].c_str(), false, 0, ImVec2(width, 0));
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        popup_items_.push_back(TRect{a.x, a.y, b.x, b.y});
        if (costs && r < decision_.costs.size()) {
            const WayCost &c = decision_.costs[r];
            ImDrawList *dl = ImGui::GetWindowDrawList();
            float x = a.x + text_w + gap;
            const float cy = (a.y + b.y) / 2;
            if (!c.pips.empty()) x += draw_mana_symbols(dl, c.pips, ImVec2(x, cy), pd) + 6;
            if (!c.text.empty()) dl->AddText(ImVec2(x, cy - lh / 2), tok().DIM, c.text.c_str());
        }
        if (chosen) answer_ = row;
        ImGui::PopID();
    }
    ImGui::EndPopup();
}

// A stack's members behind its front card take no outline: the front card shows the whole stack's.
// At a targeting or payment decision only its picks glow, and what is neither a pick, a chosen target
// nor the decision's source is dimmed. At a priority decision, the most common, a card that can be cast,
// played or activated has a thin outline (thicker under the mouse while the human chooses) and nothing
// is dimmed.
void TableView::draw_cards(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const bool glow = model_.viewer < 2 && model_.viewer == model_.decider;
    const bool focus = dims_board(decision_);
    const bool priority = decision_.kind == DecisionKind::PRIORITY;
    const bool combat = combat_kind(decision_.kind);
    const bool plan_shown = d.answer && (declares_attackers(decision_.kind) || declares_blockers(decision_.kind));
    const PlannedAttack *armed_attack = answering_ ? plan_.attack_of(plan_.armed()) : nullptr;
    for (const PlacedCard &c : layout_.cards) {
        const TableObject &obj = model_.objects[static_cast<size_t>(c.object)];
        const ImVec2 a = rmin(o, c.rect), b = rmax(o, c.rect);
        const float cw = c.rotated ? c.rect.h() : c.rect.w();
        dl->AddRectFilled(ImVec2(a.x + 1, a.y + 2), ImVec2(b.x + 1, b.y + 3), SHADOW, cw * 0.05f);
        renderer_->draw(dl, look_of(obj), a, b, c.rotated, obj.phased_out ? 0.35f : 1.0f);
        draw_card_state(dl, obj, c, o);
        const TablePile *pile = pile_of(c);
        const bool stacked = pile && pile->stacked;
        if (stacked && !c.front) continue;
        std::vector<uint64_t> members{obj.handle};
        if (stacked)
            for (int k : pile->objects) members.push_back(model_.objects[static_cast<size_t>(k)].handle);
        bool source = false, linked = false, keep = false, planned = false, armed = false, aimed = false;
        bool way = false, hot = false;
        for (uint64_t h : members) {
            if (focus) {
                source = source || decision_.pick_of_object(h) != nullptr || (answering_ && armed_target(h));
            } else if (priority) {
                way = way || decision_.pick_of_object(h) != nullptr;
                hot = hot || (answering_ && h == hovered_handle_);
            } else if (glow) {
                for (const MenuLink &m : model_.menu) source = source || (m.source == h);
            }
            keep = keep || decision_.chosen_order_of_object(h) > 0 || h == decision_.source ||
                   (combat && combat_kept(h)) || (decision_.kind == DecisionKind::MULLIGAN && obj.in_hand);
            linked = linked || (d.links && d.links->object_linked(h));
            if (plan_shown) {
                const PlannedAttack *pa = plan_.attack_of(h);
                planned = planned || (pa && !obj.attacking) || plan_.block_of(h) != nullptr;
                armed = armed || (answering_ && h == plan_.armed());
            }
            aimed = aimed || (armed_attack && armed_attack->target == h);
            planned = planned || (card_planning() && card_plan_.order_of(h) > 0);
        }
        if (focus && !source && !keep) dl->AddRectFilled(a, b, DIM_OVER, cw * 0.05f);
        if (source && !planned && !aimed) outline(dl, a, b, cw * 0.05f, tok().GLOW, 2, 3);
        if (way)
            outline(dl, a, b, cw * 0.05f, hot ? tok().GLOW : with_alpha(tok().GLOW, WAY_ALPHA), hot ? 2 : 1,
                hot ? WAY_HOT_THICKNESS : WAY_THICKNESS);
        if (planned || aimed) outline(dl, a, b, cw * 0.05f, tok().ACCENT, 2, 3);
        if (armed) outline(dl, a, b, cw * 0.05f, tok().GLOW, 5, 2);
        draw_decision_marks(dl, c, a, b, cw);
        draw_combat_marks(dl, c, a, b, cw);
        if (linked) {
            outline(dl, a, b, cw * 0.05f, hex_color(0x000000), 2, 3);
            outline(dl, a, b, cw * 0.05f, tok().LINK, 5, 3);
        }
    }
}

// A chosen target's outline and number, the divided damage given to a target so far, the decision's
// source's tag.
void TableView::draw_decision_marks(ImDrawList *dl, const PlacedCard &c, ImVec2 a, ImVec2 b, float cw) {
    if (decision_.kind == DecisionKind::NONE) return;
    const uint64_t h = model_.objects[static_cast<size_t>(c.object)].handle;
    const float fs = std::max(8.0f, std::min(13.0f, cw * 0.15f));
    if (const int order = decision_.chosen_order_of_object(h)) {
        outline(dl, a, b, cw * 0.05f, tok().ACCENT, 2, 3);
        const float r = std::max(7.0f, fs * 0.8f);
        number_badge(dl, ImVec2(b.x - 1, a.y + 1), r, order, tok().ACCENT);
    }
    if (card_planning())
        if (const int k = card_plan_.order_of(h))
            badge(dl, ImVec2(a.x + 2, a.y + 2), false, false,
                (decision_.kind == DecisionKind::BOTTOM ? "Bottom " : "Discard ") + std::to_string(k), fs, tok().ACCENT,
                tok().TEXT, 0, 2);
    if (const DecisionPick *p = decision_.pick_of_object(h))
        if (decision_.holds_chosen(*p)) {
            outline(dl, a, b, cw * 0.05f, tok().ACCENT, 2, 3);
            badge(dl, ImVec2(a.x + 2, b.y - 2), false, true, "Chosen", fs, tok().ACCENT, tok().TEXT, 0, 2);
        }
    if (const DecisionPick *p = decision_.pick_of_object(h))
        if (p->so_far > 0)
            badge(dl, ImVec2(a.x + 2, a.y + 2), false, false, std::to_string(p->so_far) + " dmg", fs, DAMAGE_BG, WHITE,
                0, 2);
    if (h == decision_.source && decision_.prompted())
        badge(dl, ImVec2(a.x + 2, a.y + 2), false, false, source_tag(decision_.kind), fs, tok().TABLE_SHADE, tok().GLOW,
            tok().GLOW, 2);
}

void TableView::draw_card_state(ImDrawList *dl, const TableObject &o, const PlacedCard &c, ImVec2 origin) {
    const ImVec2 a = rmin(origin, c.rect), b = rmax(origin, c.rect);
    const float cw = c.rotated ? c.rect.h() : c.rect.w();
    if (cw < 14) return;
    const float fs = std::max(8.0f, std::min(13.0f, cw * 0.15f));
    const float r = 2;
    float bottom_right = b.x - 2;
    if (o.has_pt) {
        const std::string pt = std::to_string(o.power) + "/" + std::to_string(o.toughness);
        ImFont *font = ImGui::GetFont();
        const float w = font->CalcTextSizeA(fs, FLT_MAX, 0, pt.c_str()).x + 2 * std::max(2.0f, fs * 0.25f);
        badge(dl, ImVec2(bottom_right, b.y - 2), true, true, pt, fs, PT_BG, pt_color(o), PT_INK, 0);
        bottom_right -= w + 2;
    }
    if (o.damage > 0)
        badge(dl, ImVec2(bottom_right, b.y - 2), true, true, "\xE2\x88\x92" + std::to_string(o.damage), fs, DAMAGE_BG,
            WHITE, 0, 0);
    float left_bottom = b.y - 2;
    if (o.loyalty >= 0)
        badge(dl, ImVec2(a.x + 2, left_bottom), false, true, std::to_string(o.loyalty), fs, PT_INK, WHITE, WHITE, 0);
    // Counters, then the designations, stack up the left edge above the bottom line's badges.
    if (o.has_pt || o.damage > 0 || o.loyalty >= 0)
        left_bottom -= ImGui::GetFont()->CalcTextSizeA(fs, FLT_MAX, 0, "0").y + 3;
    for (const TableCounter &k : o.counters)
        left_bottom -= 1 + badge(dl, ImVec2(a.x + 2, left_bottom), false, true, k.kind + " " + std::to_string(k.count),
                               fs, tok().COUNTER, tok().TEXT, 0, fs * 0.6f);
    if (cw >= 50 && !o.tags.empty()) {
        const std::string t = o.tags.size() == 1 ? o.tags[0] : o.tags[0] + " +" + std::to_string(o.tags.size() - 1);
        badge(dl, ImVec2(a.x + 2, left_bottom), false, true, t, fs * 0.9f, tok().TABLE_SHADE, tok().TEXT, 0, r);
    }
    if (c.stack_count > 1)
        badge(dl, ImVec2(b.x - 2, a.y + 2), true, false, "\xC3\x97" + std::to_string(c.stack_count), fs,
            tok().TABLE_SHADE, tok().TEXT, tok().LINE, r);
    if (o.sick && !o.in_hand)
        badge(dl, ImVec2(b.x - 2, a.y + 2), true, false, "zz", fs, tok().TABLE_SHADE, tok().DIM, 0, r);
}

// An arrow from each blocker to the attacker it blocks; in Play, while the human chooses, the planned
// blocks in the accent colour, an armed blocker's line to the mouse (or to the attacker under it), and
// at BLOCK_TARGET the blocker's line to the attacker under the mouse; then the attack labels.
void TableView::draw_combat(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    for (const CombatBlock &b : model_.blocks) block_arrow(dl, o, b.blocker, b.attacker, ImVec2(), tok().BAD, 3);
    if (d.answer && declares_blockers(decision_.kind))
        for (const PlannedBlock &b : plan_.blocks()) {
            bool declared = false;
            for (const CombatBlock &c : model_.blocks) declared = declared || c.blocker == b.blocker;
            if (!declared) block_arrow(dl, o, b.blocker, b.attacker, ImVec2(), tok().ACCENT, 3);
        }
    const ImVec2 mouse = ImGui::GetIO().MousePos;
    if (answering_ && decision_.kind == DecisionKind::BLOCKERS && plan_.armed()) {
        const uint64_t under = armed_target(hovered_handle_) ? hovered_handle_ : 0;
        if (under || ImGui::IsWindowHovered()) block_arrow(dl, o, plan_.armed(), under, mouse, tok().ACCENT, 2);
    }
    if (answering_ && decision_.kind == DecisionKind::BLOCK_TARGET && decision_.pick_of_object(hovered_handle_))
        block_arrow(dl, o, decision_.source, hovered_handle_, ImVec2(), tok().ACCENT, 2);
    draw_attack_labels(dl, o);
}

void TableView::block_arrow(
    ImDrawList *dl, ImVec2 o, uint64_t from_h, uint64_t to_h, ImVec2 to, ImU32 color, float thickness) {
    const PlacedCard *from = placed(from_h);
    const PlacedCard *dest = to_h ? placed(to_h) : nullptr;
    if (!from || (to_h && !dest)) return;
    const TRect &fr = from->rect;
    const ImVec2 fc = rect_center(o, fr);
    const ImVec2 tc = dest ? rect_center(o, dest->rect) : to;
    // From the blocker's edge facing the attacker to the attacker's facing edge.
    const bool down = tc.y > fc.y;
    const ImVec2 start(fc.x, down ? o.y + fr.y1 - 4 : o.y + fr.y0 + 4);
    const ImVec2 end = dest ? ImVec2(tc.x, down ? o.y + dest->rect.y0 + 4 : o.y + dest->rect.y1 - 4) : to;
    arrow(dl, start, end, color, thickness);
}

// "→ Player B" / "→ Kaito" by each attacker (and, while the human plans, each planned attacker), in the
// room its step forward left: under a bottom-seat attacker, over a top-seat one. Shown at the combat
// decisions, and whenever an attacker attacks a planeswalker.
void TableView::draw_attack_labels(ImDrawList *dl, ImVec2 o) {
    const bool all = combat_kind(decision_.kind);
    std::vector<CombatAttack> shown = model_.attacks;
    if (model_answer_)
        for (const PlannedAttack &p : plan_.attacks()) {
            const int i = model_.find(p.attacker);
            if (i >= 0 && model_.objects[static_cast<size_t>(i)].planned_attack)
                shown.push_back(CombatAttack{p.attacker, p.target, p.target_seat});
        }
    ImFont *font = ImGui::GetFont();
    for (const CombatAttack &a : shown) {
        if (!all && a.target == 0) continue;
        const PlacedCard *c = placed(a.attacker);
        if (!c) continue;
        const float cw = c->rotated ? c->rect.h() : c->rect.w();
        if (cw < 40) continue;
        const std::string target = a.target             ? short_name(name_of(a.target))
                                   : a.target_seat >= 0 ? model_.players[static_cast<size_t>(a.target_seat)].name
                                                        : std::string("nothing");
        const std::string text = "\xE2\x86\x92 " + target;
        const float fs = std::max(8.0f, std::min(13.0f, cw * 0.14f));
        const ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, text.c_str());
        const float w = std::min(ts.x + 8, c->rect.w() + 16);
        const int i = model_.find(a.attacker);
        const bool bottom = i >= 0 && model_.objects[static_cast<size_t>(i)].controller == model_.bottom;
        const float cx = o.x + (c->rect.x0 + c->rect.x1) / 2;
        const float y = bottom ? o.y + c->rect.y1 + 2 : o.y + c->rect.y0 - ts.y - 4;
        const bool planned = i >= 0 && model_.objects[static_cast<size_t>(i)].planned_attack;
        const ImVec2 p0(cx - w / 2, y), p1(cx + w / 2, y + ts.y + 2);
        dl->AddRectFilled(p0, p1, tok().TABLE_SHADE, 2);
        dl->AddRect(p0, p1, planned ? tok().ACCENT : tok().LINE, 2);
        draw_fitted_text(dl, font, fs, ImVec2(p0.x + 4, y + 1), tok().TEXT, text, w - 7.5f);
    }
}

// DAMAGE: "lethal N" on the blockers a click gives lethal damage; at any decision the combat damage an
// attacker has assigned to a blocker so far ("2 assigned").
void TableView::draw_combat_marks(ImDrawList *dl, const PlacedCard &c, ImVec2 a, ImVec2 b, float cw) {
    const uint64_t h = model_.objects[static_cast<size_t>(c.object)].handle;
    const float fs = std::max(8.0f, std::min(13.0f, cw * 0.15f));
    float y = a.y + 2;
    if (decision_.kind == DecisionKind::DAMAGE)
        if (const DecisionPick *p = decision_.pick_of_object(h))
            if (p->lethal >= 0)
                y += 1 + badge(dl, ImVec2(a.x + 2, y), false, false, "lethal " + std::to_string(p->lethal), fs,
                             tok().TABLE_SHADE, tok().GLOW, tok().GLOW, 2);
    for (const CombatBlock &k : model_.blocks)
        if (k.blocker == h && k.assigned > 0)
            badge(dl, ImVec2(a.x + 2, y), false, false, std::to_string(k.assigned) + " assigned", fs, tok().ACCENT,
                tok().TEXT, 0, 2);
    (void)b;
}

void TableView::draw_mid(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const ImVec2 a = rmin(o, layout_.mid), b = rmax(o, layout_.mid);
    dl->AddRectFilled(a, b, tok().TABLE_SHADE);
    dl->AddRect(a, b, tok().LINE);
    float x = o.x + layout_.mid_header.x0;
    const float y = o.y + layout_.mid_header.y0;
    const std::vector<HeaderPart> parts = header_parts(model_);
    for (size_t i = 0; i < parts.size(); ++i) {
        const HeaderPart &p = parts[i];
        dl->AddText(ImVec2(x, y), p.dim ? tok().DIM : tok().TEXT, p.text.c_str());
        if (i == 0) dl->AddText(ImVec2(x + 0.6f, y), tok().TEXT, p.text.c_str());  // bold-ish turn line
        x += ImGui::CalcTextSize(p.text.c_str()).x;
        if (p.seat_chip >= 0) {
            seat_chip(dl, ImVec2(x + 4, y), p.seat_chip);
            x += 4 + seat_chip_width();
        }
        x += 14;
    }
    draw_steps(dl, o, d);
    if (!model_.stack.empty()) draw_stack(dl, o, d);
    draw_prompt(dl, o, d);
}

// The decision's prompt line: whose decision, what is decided, the cost still to pay as pips with the
// rest of it after them, the targets chosen so far; then the buttons.
void TableView::draw_prompt(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const TRect &r = layout_.prompt;
    if (!decision_.prompted() || r.w() <= 0) return;
    const float lh = ImGui::GetTextLineHeight();
    const float line_h = lh + PROMPT_PAD;
    const ImVec2 a = rmin(o, r);
    const float cy = a.y + line_h / 2;
    const float text_y = cy - lh / 2;
    float x = a.x;
    const bool yours = d.play && model_.decider == model_.bottom;
    const std::string chip = yours ? "YOUR DECISION" : std::string(model_.decider == 1 ? "B" : "A") + " DECIDES";
    const float chip_w = ImGui::CalcTextSize(chip.c_str()).x + 12;
    dl->AddRectFilled(ImVec2(x, text_y - 2), ImVec2(x + chip_w, text_y + lh + 2),
        yours ? tok().GOOD : (model_.decider == 1 ? tok().SEAT_B : tok().SEAT_A), 2);
    dl->AddText(ImVec2(x + 6, text_y), tok().TEXT, chip.c_str());
    x += chip_w + 8;
    // The text stops short of the first button on the same line.
    float right = a.x + r.w();
    for (const TRect &b : layout_.prompt_buttons)
        if (b.y0 < r.y0 + line_h) right = std::min(right, o.x + b.x0 - 8);
    ImFont *font = ImGui::GetFont();
    const float fs = ImGui::GetFontSize();
    x += draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().TEXT, prompt_text(), right - x);
    const std::vector<std::string> pips = prompt_pips();
    if (!pips.empty() && x + lh * 2 < right) {
        x += 10;
        const float pd = lh * 1.15f;
        const float pw = mana_symbols_width(pips.size(), pd);
        if (x + pw < right) x += draw_mana_symbols(dl, pips, ImVec2(x, cy), pd) + 6;
        const int mana = mana_count(pips);
        const std::string tail =
            decision_.kind != DecisionKind::NUMBER
                ? std::string("left to pay")
                : std::string(decision_.delve ? "left after delve" : "to pay") + " = " + std::to_string(mana) + " mana";
        x += draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().DIM, tail, right - x);
    }
    std::string rest = decision_.cost_part;
    if (!decision_.owed.empty()) rest += (rest.empty() ? "" : " \xC2\xB7 ") + decision_.owed;
    if (!decision_.chosen.empty())
        rest += (rest.empty() ? "" : " \xC2\xB7 ") + std::string("chosen: ") + join_labels(decision_.chosen);
    if (!rest.empty() && x + 20 < right)
        draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().DIM, " \xC2\xB7 " + rest, right - x);
    for (size_t i = 0; i < layout_.prompt_buttons.size() && i < buttons_.size(); ++i) {
        const TRect &b = layout_.prompt_buttons[i];
        const ImVec2 ba = rmin(o, b), bz = rmax(o, b);
        const bool hot = answering_ && static_cast<int>(i) == hovered_button_;
        const bool send = buttons_[i].act == PromptButton::Act::SEND;
        const std::vector<int> &rows = buttons_[i].rows;
        const bool chosen =
            (buttons_[i].row >= 0 && buttons_[i].row == decision_.chosen_row) ||
            (decision_.chosen_row >= 0 && std::find(rows.begin(), rows.end(), decision_.chosen_row) != rows.end());
        dl->AddRectFilled(ba, bz, hot ? tok().BTN_HOVER : chosen ? tok().SEL : send ? tok().TAB_ACTIVE : tok().BTN, 2);
        const bool row_linked = buttons_[i].row >= 0 && d.links && d.links->row_linked(buttons_[i].row);
        dl->AddRect(ba, bz, row_linked ? tok().LINK : tok().LINE, 2);
        draw_fitted_text(dl, font, fs, ImVec2(ba.x + BUTTON_PAD / 2, ba.y + (b.h() - lh) / 2),
            answering_ ? tok().TEXT : tok().DIM, buttons_[i].label, b.w() - BUTTON_PAD);
    }
}

// The twelve steps, the current one highlighted; in Play a stop dot under each step with priority
// (filled where the stops stop for the side whose turn it is), the step under the mouse outlined (a
// click toggles it).
void TableView::draw_steps(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const float h = ImGui::GetTextLineHeight();
    const bool own = model_.active == model_.bottom;
    for (size_t i = 0; i < layout_.step_cells.size(); ++i) {
        const TRect &r = layout_.step_cells[i];
        const ImVec2 a = rmin(o, r), z = rmax(o, r);
        const bool on = static_cast<int>(i) == model_.step;
        dl->AddRectFilled(a, ImVec2(z.x, a.y + h + 2), on ? tok().TAB_ACTIVE : tok().FIELD);
        const char *ab = stop_step(static_cast<int>(i)).abbrev;
        const float tw = ImGui::CalcTextSize(ab).x;
        dl->AddText(ImVec2(a.x + (r.w() - tw) / 2, a.y + 1), on ? tok().TEXT : tok().DIM, ab);
        if (!d.play || !d.stops || !stop_settable(static_cast<int>(i))) continue;
        const ImVec2 c((a.x + z.x) / 2, a.y + h + 2 + STOP_DOT_H / 2);
        if (*stop_flag(d.stops, own, static_cast<int>(i))) {
            dl->AddCircleFilled(c, 3, tok().BAD);
        } else {
            dl->AddCircle(c, 3, tok().DIM);
        }
        if (static_cast<int>(i) == hovered_step_) dl->AddRect(a, z, tok().LINK, 2, 0, 1.5f);
    }
}

void TableView::draw_stack(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const size_t n = model_.stack.size();
    std::string label = "Stack \xC2\xB7 " + std::to_string(n) + (n == 1 ? " object" : " objects") + ", top first";
    if (layout_.stack_items.size() < n) label += " (" + std::to_string(layout_.stack_items.size()) + " shown)";
    dl->AddText(rmin(o, layout_.stack_label), tok().DIM, label.c_str());
    const float lh = ImGui::GetTextLineHeight();
    ImFont *font = ImGui::GetFont();
    const float fs = ImGui::GetFontSize();
    const CardBook &b = books_[static_cast<size_t>(model_view_)];
    for (size_t i = 0; i < layout_.stack_items.size(); ++i) {
        const TableStackItem &it = model_.stack[i];
        const TRect &r = layout_.stack_items[i];
        const ImVec2 a = rmin(o, r), z = rmax(o, r);
        dl->AddRectFilled(a, z, tok().PANEL_2);
        dl->AddRect(a, z, i == 0 ? tok().GLOW : tok().LINE);
        const bool linked = (d.links && d.links->object_linked(it.handle)) || static_cast<int>(i) == hovered_stack_ ||
                            static_cast<int>(i) == force_stack_;
        if (it.handle == decision_.source && decision_.prompted()) outline(dl, a, z, 0, tok().GLOW, 1, 2);
        if (decision_.pick_of_object(it.handle)) outline(dl, a, z, 0, tok().GLOW, 3, 3);
        if (const int order = decision_.chosen_order_of_object(it.handle)) {
            outline(dl, a, z, 0, tok().ACCENT, 3, 3);
            number_badge(dl, ImVec2(z.x - 1, a.y + 1), 8, order, tok().ACCENT);
        }
        if (linked) outline(dl, a, z, 0, tok().LINK, 1, 2);
        const float th = r.h() - 6, tw = th / CARD_ASPECT;
        CardLook look;
        look.face = it.card >= 0 ? b.face(it.card, it.back) : nullptr;
        if (look.face) {
            look.colors = look.face->colors;
            look.land = look.face->land;
            look.token = look.face->token;
        }
        renderer_->draw(dl, look, ImVec2(a.x + 3, a.y + 3), ImVec2(a.x + 3 + tw, a.y + 3 + th), false, 1.0f);
        const float tx = a.x + 3 + tw + 6, tmax = z.x - 4 - tx;
        float ty = a.y + 3;
        float x = tx;
        if (i == 0) x += draw_fitted_text(dl, font, fs, ImVec2(x, ty), tok().TEXT, "Top: ", tmax) + 1;
        const float chip_w = seat_chip_width() + 6;
        x += draw_fitted_text(dl, font, fs, ImVec2(x, ty), tok().TEXT, it.name, tx + tmax - x - chip_w);
        seat_chip(dl, ImVec2(x + 6, ty), it.controller);
        ty += lh;
        if (ty + lh <= z.y) draw_fitted_text(dl, font, fs, ImVec2(tx, ty), tok().DIM, it.text, tmax);
        ty += lh;
        if (!it.targets.empty() && ty + lh <= z.y + 2) {
            std::string t = "\xE2\x86\x92 ";
            for (size_t k = 0; k < it.targets.size(); ++k) t += (k ? ", " : "") + it.targets[k].label;
            draw_fitted_text(dl, font, fs, ImVec2(tx, ty), tok().DIM, t, tmax);
        }
    }
}

// Attribution's layer: each card, stack object and player chip with a value tinted by it (a stack of
// identical permanents by its first member's value), the legend at the top right with its toggle (the
// legend alone, saying "off", while the layer is hidden).
void TableView::draw_attribution(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    tinted_.clear();
    tinted_players_ = {{0, 0}};
    attr_legend_ = TRect();
    if (!d.tints || !d.tints->on || !d.attr_layer) return;
    const AttrTints &t = *d.tints;
    const bool on = *d.attr_layer;
    const bool sign = attr_target_signed(t.target);
    if (on) {
        for (const PlacedCard &c : layout_.cards) {
            const TableObject &obj = model_.objects[static_cast<size_t>(c.object)];
            const TablePile *pile = pile_of(c);
            if (pile && pile->stacked && !c.front) continue;
            std::vector<uint64_t> members{obj.handle};
            if (pile && pile->stacked)
                for (int k : pile->objects) members.push_back(model_.objects[static_cast<size_t>(k)].handle);
            for (uint64_t h : members) {
                auto it = t.objects.find(h);
                if (it == t.objects.end()) continue;
                const ImU32 col = heat_color(it->second, t.scale, sign, 170);
                const float cw = c.rotated ? c.rect.h() : c.rect.w();
                dl->AddRectFilled(rmin(o, c.rect), rmax(o, c.rect), col, cw * 0.05f);
                outline(
                    dl, rmin(o, c.rect), rmax(o, c.rect), cw * 0.05f, heat_color(it->second, t.scale, sign, 255), 1, 2);
                tinted_[obj.handle] = col;
                break;
            }
        }
        for (size_t i = 0; i < layout_.stack_items.size() && i < model_.stack.size(); ++i) {
            auto it = t.objects.find(model_.stack[i].handle);
            if (it == t.objects.end()) continue;
            const ImU32 col = heat_color(it->second, t.scale, sign, 150);
            dl->AddRectFilled(rmin(o, layout_.stack_items[i]), rmax(o, layout_.stack_items[i]), col);
            tinted_[model_.stack[i].handle] = col;
        }
        for (int seat = 0; seat < 2; ++seat) {
            if (!t.has_player[static_cast<size_t>(seat)]) continue;
            const ImU32 col = heat_color(t.players[static_cast<size_t>(seat)], t.scale, sign, 150);
            const TRect &r = chip_rects_[static_cast<size_t>(seat)];
            dl->AddRectFilled(rmin(o, r), rmax(o, r), col, 2);
            tinted_players_[static_cast<size_t>(seat)] = col;
        }
    }
    // The legend: top right, below the top band.
    const float lh = ImGui::GetTextLineHeight();
    const float w = std::max(ImGui::CalcTextSize(t.legend.c_str()).x + 12, 14 * lh);
    const float x0 = layout_.panel.x1 - w - 10, y0 = layout_.top_band.y1 + 4;
    const std::string label = on ? t.legend : "Attribution layer off (click to show)";
    const float h = on ? lh * 2.6f + 8 : lh + 6;
    attr_legend_ = TRect{x0, y0, x0 + w, y0 + h};
    dl->AddRectFilled(rmin(o, attr_legend_), rmax(o, attr_legend_), tok().TABLE_SHADE, 3);
    dl->AddRect(rmin(o, attr_legend_), rmax(o, attr_legend_), tok().LINE, 3);
    if (on) {
        draw_heat_legend(dl, ImVec2(o.x + x0 + 6, o.y + y0 + 3), w - 12, t.scale, sign, label);
    } else {
        dl->AddText(ImVec2(o.x + x0 + 6, o.y + y0 + 3), tok().DIM, label.c_str());
    }
}

// The player chips (top seat's at the top left, the bottom seat's at the bottom left), each
// followed by its effect and emblem chips.
void TableView::draw_chips(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const float lh = ImGui::GetTextLineHeight();
    for (int which = 0; which < 2; ++which) {
        const int seat = which == 0 ? model_.top : model_.bottom;
        const TablePlayer &p = model_.players[static_cast<size_t>(seat)];
        const TRect &band = which == 0 ? layout_.top_band : layout_.bottom_band;
        const float cy = (band.y0 + band.y1) / 2 - (lh + 4) / 2;
        float x = 8;
        const std::string name = p.name + (p.deck.empty() ? "" : " \xC2\xB7 " + p.deck);
        const std::string life = std::to_string(p.life);
        const std::string rest = player_text(p);
        const std::string pool = pool_text(p);
        const std::string pool_label = pool.empty() ? "" : "  pool " + pool;
        const float w = seat_chip_width() + 6 + ImGui::CalcTextSize(name.c_str()).x + 12 +
                        ImGui::CalcTextSize(life.c_str()).x + 4 + lh + 10 + ImGui::CalcTextSize(rest.c_str()).x +
                        ImGui::CalcTextSize(pool_label.c_str()).x + 16;
        const TRect chip{x, cy, x + w, cy + lh + 4};
        chip_rects_[static_cast<size_t>(seat)] = chip;
        const ImVec2 a = rmin(o, chip), z = rmax(o, chip);
        dl->AddRectFilled(a, z, tok().TABLE_SHADE, 2);
        dl->AddRect(a, z, tok().LINE, 2);
        const PlannedAttack *armed = answering_ && decision_.kind == DecisionKind::ATTACKERS && seat == model_.top
                                         ? plan_.attack_of(plan_.armed())
                                         : nullptr;
        if (decision_.pick_of_seat(seat) || (armed && armed->target != 0)) outline(dl, a, z, 2, tok().GLOW, 2, 3);
        if (armed && armed->target == 0) outline(dl, a, z, 2, tok().ACCENT, 2, 3);
        if (const int order = decision_.chosen_order_of_seat(seat)) {
            outline(dl, a, z, 2, tok().ACCENT, 2, 3);
            number_badge(dl, ImVec2(z.x, a.y), 8, order, tok().ACCENT);
        }
        if (d.links && d.links->player_linked(seat)) outline(dl, a, z, 2, tok().LINK, 1, 2);
        float tx = a.x + 8;
        seat_chip(dl, ImVec2(tx, a.y + 2), seat);
        tx += seat_chip_width() + 6;
        dl->AddText(ImVec2(tx, a.y + 2), tok().TEXT, name.c_str());
        tx += ImGui::CalcTextSize(name.c_str()).x + 12;
        dl->AddText(ImVec2(tx, a.y + 2), tok().TEXT, life.c_str());
        dl->AddText(ImVec2(tx + 0.6f, a.y + 2), tok().TEXT, life.c_str());
        tx += ImGui::CalcTextSize(life.c_str()).x + 4;
        heart(dl, ImVec2(tx + lh * 0.45f, a.y + 2 + lh * 0.5f), lh * 0.7f, tok().BAD);
        tx += lh + 10;
        dl->AddText(ImVec2(tx, a.y + 2), tok().DIM, rest.c_str());
        if (!pool_label.empty())
            dl->AddText(ImVec2(tx + ImGui::CalcTextSize(rest.c_str()).x, a.y + 2), tok().GLOW, pool_label.c_str());
        x += w + 6;
        for (const std::string &e : p.effects) {
            const float ew = ImGui::CalcTextSize(e.c_str()).x + 12;
            if (o.x + x + ew > o.x + layout_.panel.x1 - 4) break;
            const ImVec2 ea = at(o, x, cy), ez = at(o, x + ew, cy + lh + 4);
            dl->AddRectFilled(ea, ez, tok().TABLE_SHADE, 2);
            dl->AddRect(ea, ez, tok().LINE, 2);
            dl->AddText(ImVec2(ea.x + 6, ea.y + 2), tok().DIM, e.c_str());
            x += ew + 4;
        }
    }
}

void TableView::update_magnifier(const Hit &hit, const TableDraw &d, bool over) {
    uint64_t under = 0;
    if (hit.card >= 0) {
        under = hovered_handle_;
    } else if (hit.stack >= 0 && force_stack_ < 0) {
        under = model_.stack[static_cast<size_t>(hit.stack)].handle;
    }
    mag_.update(under, over, d.magnify);
    if (!force_mag_.empty()) {
        std::string want = force_mag_;
        for (char &ch : want) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        for (const PlacedCard &c : layout_.cards) {
            std::string n = model_.objects[static_cast<size_t>(c.object)].name;
            for (char &ch : n) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (!n.empty() && n.find(want) != std::string::npos) {
                mag_.open(model_.objects[static_cast<size_t>(c.object)].handle);
                break;
            }
        }
    }
    if (!mag_.handle()) return;
    const ImVec2 origin = ImGui::GetCurrentWindow()->InnerRect.Min;
    const int obj = model_.find(mag_.handle());
    if (obj >= 0) {
        for (const PlacedCard &c : layout_.cards)
            if (c.object == obj) {
                const TablePile *p = pile_of(c);
                draw_magnifier(model_.objects[static_cast<size_t>(obj)], c.rect, origin,
                    p && p->stacked ? static_cast<int>(p->objects.size()) : 0);
                return;
            }
    }
    for (size_t k = 0; k < layout_.stack_items.size(); ++k) {
        if (model_.stack[k].handle != mag_.handle()) continue;
        TableObject face;
        face.handle = model_.stack[k].handle;
        face.card = model_.stack[k].card;
        face.back = model_.stack[k].back;
        face.name = model_.stack[k].name;
        face.controller = model_.stack[k].controller;
        const CardFace *f =
            face.card >= 0 ? books_[static_cast<size_t>(model_view_)].face(face.card, face.back) : nullptr;
        if (f) face.colors = f->colors;
        draw_magnifier(face, layout_.stack_items[k], origin, 0);
        return;
    }
    mag_.close();
}

// The card large beside `at`, its state under it.
void TableView::draw_magnifier(const TableObject &o, const TRect &at_rect, ImVec2 origin, int stacked) {
    const bool on_table = model_.find(o.handle) >= 0;
    const std::vector<std::string> lines =
        on_table ? state_lines(model_, o, stacked) : std::vector<std::string>{o.name};
    std::function<void(ImDrawList *, ImVec2, ImVec2)> overlay;
    if (on_table)
        overlay = [this, &o](ImDrawList *fg, ImVec2 a, ImVec2 b) {
            PlacedCard pc;
            pc.rect = TRect{a.x, a.y, b.x, b.y};
            draw_card_state(fg, o, pc, ImVec2(0, 0));
        };
    draw_magnified_card(*renderer_, look_of(o), lines, rmin(origin, at_rect), rmax(origin, at_rect), overlay);
}

const PlacedCard *TableView::placed(uint64_t h) const {
    const int obj = model_.find(h);
    if (obj < 0) return nullptr;
    for (const PlacedCard &c : layout_.cards) {
        if (c.in_hand || !c.front) continue;
        if (c.object == obj) return &c;
        const TablePile *p = pile_of(c);
        if (p && p->stacked && std::find(p->objects.begin(), p->objects.end(), obj) != p->objects.end()) return &c;
    }
    return nullptr;
}

uint64_t TableView::hit_object(const Hit &hit) const {
    if (hit.card < 0) return 0;
    const PlacedCard &c = layout_.cards[static_cast<size_t>(hit.card)];
    const uint64_t front = model_.objects[static_cast<size_t>(c.object)].handle;
    const TablePile *pile = pile_of(c);
    if (!pile || !pile->stacked || decision_.pick_of_object(front)) return front;
    for (int k : pile->objects)
        if (decision_.pick_of_object(model_.objects[static_cast<size_t>(k)].handle))
            return model_.objects[static_cast<size_t>(k)].handle;
    return front;
}

std::string TableView::name_of(uint64_t h) const {
    const int i = model_.find(h);
    return i >= 0 && !model_.objects[static_cast<size_t>(i)].name.empty() ? model_.objects[static_cast<size_t>(i)].name
                                                                          : std::string("a face-down creature");
}

bool TableView::planning() const {
    return answering_ && (decision_.kind == DecisionKind::ATTACKERS || decision_.kind == DecisionKind::BLOCKERS);
}

bool TableView::combat_kept(uint64_t h) const {
    const int i = model_.find(h);
    if (i < 0) return false;
    const TableObject &o = model_.objects[static_cast<size_t>(i)];
    if (o.attacking || o.blocking || o.planned_attack || plan_.block_of(h) != nullptr || armed_target(h)) return true;
    // A planeswalker attacked, or planned to be.
    for (const CombatAttack &a : model_.attacks)
        if (a.target == h) return true;
    for (const PlannedAttack &a : plan_.attacks())
        if (a.target == h && model_answer_) return true;
    return false;
}

bool TableView::armed_target(uint64_t h) const {
    const uint64_t armed = plan_.armed();
    if (!armed || !h || !planning()) return false;
    if (decision_.kind == DecisionKind::ATTACKERS) {
        const std::vector<uint64_t> pw = planeswalkers();
        return std::find(pw.begin(), pw.end(), h) != pw.end();
    }
    const std::vector<uint64_t> &att = decision_.block.attackers;
    std::string why;
    return std::find(att.begin(), att.end(), h) != att.end() && decision_.may_block(armed, h, &why);
}

std::vector<uint64_t> TableView::planeswalkers() const {
    std::vector<uint64_t> out;
    for (const TableObject &o : model_.objects)
        if (!o.in_hand && o.planeswalker && o.controller == model_.top) out.push_back(o.handle);
    return out;
}

// While the human plans attackers: each planned attacker not yet attacking is marked to stand forward,
// and its pile leaves any group of identical permanents (so no stack holds it).
void TableView::apply_plan_to_model() {
    if (!model_answer_ || decision_.kind != DecisionKind::ATTACKERS) return;
    for (const PlannedAttack &a : plan_.attacks()) {
        const int i = model_.find(a.attacker);
        if (i < 0 || model_.objects[static_cast<size_t>(i)].attacking) continue;
        model_.objects[static_cast<size_t>(i)].planned_attack = true;
        for (std::vector<TablePile> &row : model_.rows)
            for (TablePile &p : row)
                if (!p.objects.empty() && p.objects[0] == i) p.group = -1;
    }
}

// The prompt line's buttons: while the human plans, the plan's (send, all, clear); else the rows
// naming nothing.
void TableView::build_buttons() {
    buttons_.clear();
    if (!decision_.prompted()) return;
    if (decision_.kind == DecisionKind::BOTTOM && !answering_) return;
    if (card_planning()) {
        const int owed = card_plan_owed();
        const int n = static_cast<int>(card_plan_.cards().size());
        const bool bottom = decision_.kind == DecisionKind::BOTTOM;
        if (owed > 0 && n == owed)
            buttons_.push_back(
                PromptButton{bottom ? "Put " + std::to_string(n) + " on the bottom" : "Discard " + std::to_string(n),
                    -1, PromptButton::Act::SEND});
        if (n > 0) {
            buttons_.push_back(PromptButton{"Undo", -1, PromptButton::Act::UNDO});
            buttons_.push_back(PromptButton{"Clear", -1, PromptButton::Act::CLEAR});
        }
        return;
    }
    if (decision_.kind == DecisionKind::NUMBER) {
        // The stepper: down, the value (it sends its row), up.
        const size_t i =
            static_cast<size_t>(std::max(0, std::min(number_index_, static_cast<int>(decision_.numbers.size()) - 1)));
        if (decision_.numbers.empty()) return;
        const std::string value = std::to_string(decision_.numbers[i]);
        buttons_.push_back(PromptButton{"\xE2\x88\x92", -1, PromptButton::Act::DEC});
        buttons_.push_back(PromptButton{
            decision_.delve ? "Exile " + value : "X = " + value, static_cast<int>(i), PromptButton::Act::ROW});
        buttons_.push_back(PromptButton{"+", -1, PromptButton::Act::INC});
        return;
    }
    if (decision_.kind == DecisionKind::PRIORITY) {
        // The ways off the Table (a flashback, an escape, a companion), one button opening their popup
        // when they are many, then Pass at the right.
        std::vector<int> off;
        for (int row : decision_.buttons)
            if (row != decision_.pass) off.push_back(row);
        if (off.size() < MAX_PROMPT_ROWS) {
            for (int row : off)
                buttons_.push_back(
                    PromptButton{decision_.row_text[static_cast<size_t>(row)], row, PromptButton::Act::ROW, {}});
        } else {
            buttons_.push_back(PromptButton{
                std::to_string(off.size()) + " ways off the Table\xE2\x80\xA6", -1, PromptButton::Act::POPUP, off});
        }
        if (decision_.pass >= 0) buttons_.push_back(PromptButton{"Pass", decision_.pass, PromptButton::Act::ROW, {}});
        return;
    }
    if (!planning()) {
        // The card picker shows the rows naming no card under its cards.
        if (decision_.kind == DecisionKind::CARDS && picker_.active && picker_.off_table()) return;
        for (int row : decision_.buttons)
            buttons_.push_back(PromptButton{decision_.row_text[static_cast<size_t>(row)], row, PromptButton::Act::ROW});
        return;
    }
    const bool blockers = decision_.kind == DecisionKind::BLOCKERS;
    size_t n = blockers ? model_.blocks.size() : model_.attacks.size();
    bool unplanned = false;
    if (blockers) {
        for (const PlannedBlock &b : plan_.blocks()) {
            bool declared = false;
            for (const CombatBlock &c : model_.blocks) declared = declared || c.blocker == b.blocker;
            if (!declared) ++n;
        }
    } else {
        for (const PlannedAttack &a : plan_.attacks()) {
            const int i = model_.find(a.attacker);
            if (i >= 0 && model_.objects[static_cast<size_t>(i)].planned_attack) ++n;
        }
        for (const DecisionPick &p : decision_.picks) unplanned = unplanned || (p.handle && !plan_.attack_of(p.handle));
    }
    const std::string count = std::to_string(n);
    buttons_.push_back(PromptButton{
        n ? (blockers ? "Block with " : "Attack with ") + count : std::string(blockers ? "No blocks" : "No attack"),
        decision_.confirm, PromptButton::Act::SEND});
    if (unplanned) buttons_.push_back(PromptButton{"Attack with all", -1, PromptButton::Act::ALL});
    if (!plan_.empty()) buttons_.push_back(PromptButton{"Clear", -1, PromptButton::Act::CLEAR});
}

// While the human plans: the declaration so far (declared and planned) and what a click does next;
// else the decision's own prompt.
std::string TableView::prompt_text() const {
    if (card_planning()) {
        std::vector<std::string> planned;
        for (uint64_t h : card_plan_.cards()) planned.push_back(name_of(h));
        const bool bottom = decision_.kind == DecisionKind::BOTTOM;
        return decision_.prompt + (planned.empty() ? std::string(bottom ? " \xC2\xB7 click the cards to put there"
                                                                        : " \xC2\xB7 click the cards to discard")
                                                   : " \xC2\xB7 planned: " + join_names(planned));
    }
    if (!planning()) return decision_.prompt;
    const bool blockers = decision_.kind == DecisionKind::BLOCKERS;
    const uint64_t armed = plan_.armed();
    const auto target_name = [this](uint64_t target, int seat) {
        return target ? short_name(name_of(target)) : model_.players[static_cast<size_t>(seat == 1 ? 1 : 0)].name;
    };
    if (armed && blockers) return name_of(armed) + " blocks: click the attacker it blocks (or drag onto it)";
    if (armed) {
        // What else it may attack: the planeswalkers, and the player once a planeswalker is chosen.
        const PlannedAttack *a = plan_.attack_of(armed);
        std::string others;
        for (uint64_t pw : planeswalkers())
            if (!a || pw != a->target) others += (others.empty() ? "" : " or ") + short_name(name_of(pw));
        if (a && a->target) others += (others.empty() ? "" : " or ") + target_name(0, model_.top);
        return name_of(armed) + " attacks " + (a ? target_name(a->target, a->target_seat) : std::string("?")) +
               " \xC2\xB7 or click " + others;
    }
    std::string list;
    const auto add = [&list](const std::string &item) { list += (list.empty() ? "" : ", ") + item; };
    if (blockers) {
        for (const CombatBlock &b : model_.blocks) add(name_of(b.blocker) + " \xE2\x86\x92 " + name_of(b.attacker));
        for (const PlannedBlock &b : plan_.blocks()) {
            bool declared = false;
            for (const CombatBlock &c : model_.blocks) declared = declared || c.blocker == b.blocker;
            if (!declared) add(name_of(b.blocker) + " \xE2\x86\x92 " + name_of(b.attacker));
        }
    } else {
        for (const CombatAttack &a : model_.attacks)
            add(name_of(a.attacker) + " \xE2\x86\x92 " + target_name(a.target, a.target_seat));
        for (const PlannedAttack &a : plan_.attacks()) {
            const int i = model_.find(a.attacker);
            if (i >= 0 && model_.objects[static_cast<size_t>(i)].planned_attack)
                add(name_of(a.attacker) + " \xE2\x86\x92 " + target_name(a.target, a.target_seat));
        }
    }
    return std::string(blockers ? "Declare blockers" : "Declare attackers") + (list.empty() ? "" : ": " + list) +
           (blockers ? " \xC2\xB7 click a blocker, then an attacker (or drag)"
                     : " \xC2\xB7 click creatures to attack, then confirm");
}

bool TableView::press_button(const PromptButton &b) {
    switch (b.act) {
        case PromptButton::Act::ROW:
            answer_ = b.row;
            return true;
        case PromptButton::Act::ALL:
            plan_all();
            return true;
        case PromptButton::Act::CLEAR:
            if (card_planning()) {
                card_plan_.clear();
            } else {
                plan_.clear();
            }
            return true;
        case PromptButton::Act::UNDO:
            card_plan_.undo();
            return true;
        case PromptButton::Act::DEC:
            number_index_ = std::max(0, number_index_ - 1);
            return true;
        case PromptButton::Act::INC:
            number_index_ = std::min(static_cast<int>(decision_.numbers.size()) - 1, number_index_ + 1);
            return true;
        case PromptButton::Act::POPUP:
            open_rows_popup(b.rows, "Off the Table");
            return true;
        case PromptButton::Act::SEND:
            break;
    }
    if (card_planning()) {
        card_plan_.start_sending();
        status_ = (decision_.kind == DecisionKind::BOTTOM ? "Putting " : "Discarding ") +
                  std::to_string(card_plan_.cards().size()) +
                  (decision_.kind == DecisionKind::BOTTOM ? " cards on the bottom" : " cards") +
                  " as planned, one decision at a time";
        return true;
    }
    // Nothing planned left to send: the confirm row alone.
    bool unsent = false;
    for (const PlannedAttack &a : plan_.attacks()) {
        const int i = model_.find(a.attacker);
        unsent = unsent || (i >= 0 && model_.objects[static_cast<size_t>(i)].planned_attack);
    }
    for (const PlannedBlock &p : plan_.blocks()) {
        bool declared = false;
        for (const CombatBlock &c : model_.blocks) declared = declared || c.blocker == p.blocker;
        unsent = unsent || !declared;
    }
    if (!unsent) {
        answer_ = b.row;
        plan_.clear();
        return true;
    }
    plan_.start_sending();
    status_ = std::string("Sending the plan: ") +
              (decision_.kind == DecisionKind::BLOCKERS ? "each blocker, then the attacker it blocks"
                                                        : "each attacker, then what it attacks") +
              ", one decision at a time";
    return true;
}

// A press while the human plans: attackers planned, taken back or aimed at a planeswalker; blockers
// armed, their blocks planned or taken back. False when it plans nothing (the click selects).
bool TableView::plan_press(const Hit &hit) {
    const uint64_t h = hit_object(hit);
    const uint64_t armed = plan_.armed();
    press_handle_ = 0;
    if (decision_.kind == DecisionKind::ATTACKERS) {
        if (armed && (hit.seat == model_.top || armed_target(h))) {
            PlannedAttack a = *plan_.attack_of(armed);
            a.target = hit.seat >= 0 ? 0 : h;
            a.target_seat = hit.seat >= 0 ? model_.top : -1;
            plan_.plan_attack(a);
            plan_.arm(0);
            return true;
        }
        if (h && decision_.pick_of_object(h)) {
            if (plan_.attack_of(h)) {
                plan_.drop(h);
                return true;
            }
            PlannedAttack a;
            a.attacker = h;
            a.target_seat = model_.top;
            plan_.plan_attack(a);
            plan_.arm(planeswalkers().empty() ? 0 : h);
            return true;
        }
        plan_.arm(0);
        return false;
    }
    if (h && decision_.pick_of_object(h)) {
        if (plan_.block_of(h)) {
            plan_.drop(h);
            return true;
        }
        press_was_armed_ = armed == h;
        press_handle_ = h;
        plan_.arm(h);
        return true;
    }
    const std::vector<uint64_t> &att = decision_.block.attackers;
    if (armed && h && std::find(att.begin(), att.end(), h) != att.end()) {
        std::string why;
        if (decision_.may_block(armed, h, &why)) {
            plan_.plan_block(PlannedBlock{armed, h});
            plan_.arm(0);
        } else {
            status_ = name_of(armed) + " can't block " + name_of(h) + ": " + why +
                      " (the engine's own menu in Choices has the last word)";
        }
        return true;
    }
    plan_.arm(0);
    return false;
}

// A click on a hand card while the human plans cards (the London mulligan's bottom pick, a discard of
// several cards): planned, or taken back. False when it is no pick (the click selects).
bool TableView::card_plan_press(const Hit &hit) {
    const uint64_t h = hit_object(hit);
    if (!h || !decision_.pick_of_object(h)) return false;
    const int owed = card_plan_owed();
    if (!card_plan_.toggle(h, owed))
        status_ = std::to_string(owed) + (owed == 1 ? " card is" : " cards are") +
                  " planned already: click a planned card (or Undo) to take one back";
    return true;
}

bool TableView::card_planning() const {
    return answering_ && (decision_.kind == DecisionKind::BOTTOM || decision_.plan_owed > 1);
}

int TableView::card_plan_owed() const {
    return decision_.kind == DecisionKind::BOTTOM ? decision_.mulligan.to_bottom : decision_.plan_owed;
}

int TableView::number_value() const {
    if (decision_.kind != DecisionKind::NUMBER || decision_.numbers.empty()) return -1;
    return decision_.numbers[static_cast<size_t>(
        std::max(0, std::min(number_index_, static_cast<int>(decision_.numbers.size()) - 1)))];
}

std::vector<std::string> TableView::prompt_pips() const {
    if (decision_.kind != DecisionKind::NUMBER) return decision_.pips;
    const int value = number_value();
    return value < 0 ? decision_.pips : number_pips(decision_, value);
}

// The card picker over the opponent's half, when the decision has one: its click answers as the Table's.
void TableView::draw_picker(const TableDraw &d) {
    if (!picker_.active) {
        picker_view_.hide();
        return;
    }
    const ImVec2 o = ImGui::GetCurrentWindow()->InnerRect.Min;
    CardPickerView::Place pl;
    pl.x0 = o.x + layout_.panel.x0 + 10;
    pl.x1 = o.x + layout_.panel.x1 - 10;
    pl.y0 = o.y + layout_.top_band.y1 + 2;
    pl.y_long = o.y + (layout_.prompt.w() > 0 ? layout_.prompt.y0 - 4 : layout_.mid.y1 - 4);
    pl.y_short = std::min(pl.y_long, o.y + layout_.mid.y0 - 6);
    pl.y_tall = std::max(pl.y_long, o.y + layout_.bottom_band.y0 - 4);
    pl.prompt = decision_.prompt;
    const int row =
        picker_view_.draw(picker_, books_[static_cast<size_t>(model_view_)], pl, answering_, d.magnify, d.links);
    if (row >= 0) answer_ = row;
}

// The choice panel where the picker goes, when the decision has one: its click answers as the Table's.
void TableView::draw_panel(const TableDraw &d) {
    if (!panel_.active) {
        panel_view_.hide();
        return;
    }
    const ImVec2 o = ImGui::GetCurrentWindow()->InnerRect.Min;
    CardPickerView::Place pl;
    pl.x0 = o.x + layout_.panel.x0 + 10;
    pl.x1 = o.x + layout_.panel.x1 - 10;
    pl.y0 = o.y + layout_.top_band.y1 + 2;
    pl.y_long = o.y + (layout_.prompt.w() > 0 ? layout_.prompt.y0 - 4 : layout_.mid.y1 - 4);
    pl.y_short = std::min(pl.y_long, o.y + layout_.mid.y0 - 6);
    pl.y_tall = std::max(pl.y_long, o.y + layout_.bottom_band.y0 - 4);
    pl.prompt = decision_.prompt;
    const int row = panel_view_.draw(panel_, books_[static_cast<size_t>(model_view_)], pl, answering_, d.links);
    if (row >= 0) answer_ = row;
}

// A release after a press on a blocker: dragged onto an attacker it may block, the block is planned;
// a click without a drag on a blocker armed before disarms it.
void TableView::plan_release(const Hit &hit) {
    if (decision_.kind != DecisionKind::BLOCKERS || press_handle_ == 0) return;
    const uint64_t blocker = press_handle_;
    press_handle_ = 0;
    const ImGuiIO &io = ImGui::GetIO();
    const bool dragged = io.MouseDragMaxDistanceSqr[0] > io.MouseDragThreshold * io.MouseDragThreshold;
    if (!dragged) {
        if (press_was_armed_) plan_.arm(0);
        return;
    }
    const uint64_t h = hit_object(hit);
    const std::vector<uint64_t> &att = decision_.block.attackers;
    if (!h || h == blocker || std::find(att.begin(), att.end(), h) == att.end()) return;
    std::string why;
    if (decision_.may_block(blocker, h, &why)) {
        plan_.plan_block(PlannedBlock{blocker, h});
        plan_.arm(0);
    } else {
        status_ = name_of(blocker) + " can't block " + name_of(h) + ": " + why;
    }
}

void TableView::plan_all() {
    for (const DecisionPick &p : decision_.picks) {
        if (!p.handle || plan_.attack_of(p.handle)) continue;
        PlannedAttack a;
        a.attacker = p.handle;
        a.target_seat = model_.top;
        plan_.plan_attack(a);
    }
    plan_.arm(0);
}

// Screenshots: force_plan_'s items, once per declaration.
void TableView::apply_force_plan() {
    if (!force_plan_.empty() && card_planning() && card_plan_.empty() && !card_plan_.sending()) {
        // The bottom picks, a discard: each name plans the first hand card of that name not planned yet.
        for (const std::string &want : split_names(force_plan_))
            for (const DecisionPick &p : decision_.picks) {
                const int i = model_.find(p.handle);
                if (i < 0 || card_plan_.order_of(p.handle) ||
                    lower_text(model_.objects[static_cast<size_t>(i)].name).find(lower_text(want)) == std::string::npos)
                    continue;
                card_plan_.toggle(p.handle, card_plan_owed());
                break;
            }
        return;
    }
    if (force_plan_.empty() || !planning() || plan_.key() == force_planned_) return;
    force_planned_ = plan_.key();
    const auto lower = [](std::string t) {
        for (char &ch : t) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return t;
    };
    const auto find = [&](const std::string &want, bool pick) -> uint64_t {
        for (const TableObject &o : model_.objects) {
            if (o.in_hand || o.name.empty() || lower(o.name).find(lower(want)) == std::string::npos) continue;
            if (pick ? decision_.pick_of_object(o.handle) && !plan_.attack_of(o.handle) && !plan_.block_of(o.handle)
                     : true)
                return o.handle;
        }
        return 0;
    };
    size_t from = 0;
    while (from <= force_plan_.size()) {
        size_t end = force_plan_.find(',', from);
        if (end == std::string::npos) end = force_plan_.size();
        const std::string item = force_plan_.substr(from, end - from);
        from = end + 1;
        const size_t gt = item.find('>');
        const std::string name = item.substr(0, gt), target = gt == std::string::npos ? "" : item.substr(gt + 1);
        const uint64_t c = find(name, true);
        if (!c) continue;
        if (decision_.kind == DecisionKind::ATTACKERS) {
            PlannedAttack a;
            a.attacker = c;
            a.target_seat = model_.top;
            if (!target.empty() && target != "?" && lower(target) != "player") {
                a.target = find(target, false);
                a.target_seat = a.target ? -1 : model_.top;
            }
            plan_.plan_attack(a);
            if (target == "?") plan_.arm(c);
        } else if (target.empty() || target == "?") {
            plan_.arm(c);
        } else if (const uint64_t att = find(target, false)) {
            plan_.plan_block(PlannedBlock{c, att});
        }
    }
}

int TableView::plan_reply(const Match &m, long index, int seat) {
    if (card_plan_.sending()) {
        if (index < 0 || index == card_plan_.answered()) return -1;
        std::string err;
        const std::shared_ptr<const rmstate::DecodedFrame> f = m.frame(static_cast<size_t>(index), seat, &err);
        if (!f) {
            card_plan_.clear();
            status_ = "Sending stopped: decision " + std::to_string(index + 1) + "'s frame does not read: " + err;
            return -1;
        }
        const agents::DecisionView v(m.stream(seat), *f, seat);
        const bool bottom = card_plan_.key().purpose == "bottom";
        const int row =
            bottom ? next_bottom_row(card_plan_, v, nullptr) : next_planned_row(card_plan_, v, "DISCARD", nullptr);
        if (row < 0) {
            status_ = bottom ? "Put on the bottom as planned" : "Discarded as planned";
            card_plan_.clear();
            return -1;
        }
        card_plan_.record(index);
        return row;
    }
    if (!plan_.sending() || index < 0 || index == plan_.answered()) return -1;
    std::string err;
    const std::shared_ptr<const rmstate::DecodedFrame> f = m.frame(static_cast<size_t>(index), seat, &err);
    if (!f) {
        plan_.stop_sending();
        status_ = "Sending stopped: decision " + std::to_string(index + 1) + "'s frame does not read: " + err;
        return -1;
    }
    const agents::DecisionView v(m.stream(seat), *f, seat);
    const PlanStep step = next_plan_step(plan_, v);
    if (step.row >= 0) plan_.record(step, index);
    if (step.stop != PlanStop::NONE) {
        status_ = stop_text(step.stop, name_of(step.creature));
        if (step.stop == PlanStop::ENDED) {
            plan_.clear();
        } else {
            plan_.stop_sending();
            plan_.drop(step.creature);
        }
    } else if (step.finished) {
        status_ = plan_.key().blockers ? "Blockers declared as planned" : "Attackers declared as planned";
        plan_.clear();
    }
    return step.row;
}

}  // namespace rmwork
