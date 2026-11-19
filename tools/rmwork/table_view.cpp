#include "table_view.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>

#include "../../src/agents/decision_view.h"
#include "imgui_internal.h"
#include "magnifier.h"
#include "theme.h"
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

bool TableView::refresh(ReplaySession &s) {
    const Match *m = s.match();
    // The session's generation moves with the match, the cursor and the viewer.
    if (have_model_ && m == model_match_ && s.generation() == model_generation_) return true;
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
    decision_ = read_table_decision(agents::DecisionView(m->stream(s.view()), *frame, model_.bottom), model_);
    if (decision_.kind == DecisionKind::TARGET)
        decision_.chosen = chosen_targets(*m, s.cursor(), s.view(), decision_.source);
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
        for (int row : decision_.buttons)
            mx.button_w.push_back(
                ImGui::CalcTextSize(decision_.row_text[static_cast<size_t>(row)].c_str()).x + BUTTON_PAD);
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
    if (!refresh(s)) {
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
    hovered_handle_ =
        hit.card >= 0 ? model_.objects[static_cast<size_t>(layout_.cards[static_cast<size_t>(hit.card)].object)].handle
                      : 0;
    hovered_stack_ = hit.stack;
    hovered_step_ = hit.step;
    hovered_button_ = hit.button;
    const bool clicked = over && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
    const bool answered = clicked && d.answer && hit.step < 0 && answer_click(hit);
    if (over) hover_hint(hit, d);
    if (hit.step >= 0) {
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
        if (hit.button >= 0) d.links->hover_menu_row(decision_.buttons[static_cast<size_t>(hit.button)]);
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
    draw_combat(dl, o);
    draw_mid(dl, o, d);
    draw_chips(dl, o, d);
    dl->PopClipRect();
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
    if (hit.button >= 0) {
        answer_ = decision_.buttons[static_cast<size_t>(hit.button)];
        return true;
    }
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
    popup_rows_ = p.rows;
    const int obj = p.handle ? model_.find(p.handle) : -1;
    popup_title_ = obj >= 0 && !model_.objects[static_cast<size_t>(obj)].name.empty()
                       ? model_.objects[static_cast<size_t>(obj)].name
                   : p.handle == 0 ? model_.players[static_cast<size_t>(p.seat == 1 ? 1 : 0)].name
                                   : std::string("Choose");
    popup_generation_ = model_generation_;
    open_popup_ = true;
}

// Screenshots: the pick popup of the first card named like force_pick_ with several rows.
void TableView::apply_force_pick(const TableDraw &d) {
    if (force_pick_generation_ != model_generation_) force_picked_ = false;
    if (force_pick_.empty() || force_picked_ || !d.answer) return;
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
    if (!d.answer || hit.step >= 0) return;
    if (hit.button >= 0) {
        ImGui::SetTooltip("Click: %s",
            decision_.row_text[static_cast<size_t>(decision_.buttons[static_cast<size_t>(hit.button)])].c_str());
        return;
    }
    const DecisionPick *p = pick_at(hit);
    if (!p || p->rows.empty()) return;
    if (p->rows.size() == 1) {
        ImGui::SetTooltip("Click: %s", decision_.row_text[static_cast<size_t>(p->rows[0])].c_str());
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
    for (int row : popup_rows_) {
        ImGui::PushID(row);
        const bool chosen = ImGui::Selectable(decision_.row_text[static_cast<size_t>(row)].c_str());
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        popup_items_.push_back(TRect{a.x, a.y, b.x, b.y});
        if (chosen) answer_ = row;
        ImGui::PopID();
    }
    ImGui::EndPopup();
}

// A stack's members behind its front card take no outline: the front card shows the whole stack's.
// At a targeting or payment decision only its picks glow, and what is neither a pick, a chosen target
// nor the decision's source is dimmed.
void TableView::draw_cards(ImDrawList *dl, ImVec2 o, const TableDraw &d) {
    const bool glow = model_.viewer < 2 && model_.viewer == model_.decider;
    const bool focus = decision_.prompted();
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
        bool source = false, linked = false, keep = false;
        for (uint64_t h : members) {
            if (focus) {
                source = source || decision_.pick_of_object(h) != nullptr;
            } else if (glow) {
                for (const MenuLink &m : model_.menu) source = source || (m.source == h);
            }
            keep = keep || decision_.chosen_order_of_object(h) > 0 || h == decision_.source;
            linked = linked || (d.links && d.links->object_linked(h));
        }
        if (focus && !source && !keep) dl->AddRectFilled(a, b, DIM_OVER, cw * 0.05f);
        if (source) outline(dl, a, b, cw * 0.05f, tok().GLOW, 2, 3);
        draw_decision_marks(dl, c, a, b, cw);
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
    if (const DecisionPick *p = decision_.pick_of_object(h))
        if (p->so_far > 0)
            badge(dl, ImVec2(a.x + 2, a.y + 2), false, false, std::to_string(p->so_far) + " dmg", fs, DAMAGE_BG, WHITE,
                0, 2);
    if (h == decision_.source && decision_.prompted())
        badge(dl, ImVec2(a.x + 2, a.y + 2), false, false, "Source", fs, tok().TABLE_SHADE, tok().GLOW, tok().GLOW, 2);
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

// A line from each blocker to the attacker it blocks.
void TableView::draw_combat(ImDrawList *dl, ImVec2 o) {
    for (const CombatBlock &b : model_.blocks) {
        int from = -1, to = -1;
        for (size_t i = 0; i < layout_.cards.size(); ++i) {
            const uint64_t h = model_.objects[static_cast<size_t>(layout_.cards[i].object)].handle;
            if (h == b.blocker) from = static_cast<int>(i);
            if (h == b.attacker) to = static_cast<int>(i);
        }
        if (from < 0 || to < 0) continue;
        const TRect &fr = layout_.cards[static_cast<size_t>(from)].rect,
                    &tr = layout_.cards[static_cast<size_t>(to)].rect;
        const ImVec2 fc = rect_center(o, fr), tc = rect_center(o, tr);
        // From the blocker's edge facing the attacker to the attacker's facing edge.
        const bool down = tc.y > fc.y;
        const ImVec2 start(fc.x, down ? o.y + fr.y1 - 4 : o.y + fr.y0 + 4);
        const ImVec2 end(tc.x, down ? o.y + tr.y0 + 4 : o.y + tr.y1 - 4);
        arrow(dl, start, end, tok().BAD, 3);
    }
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
    x += draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().TEXT, decision_.prompt, right - x);
    if (!decision_.pips.empty() && x + lh * 2 < right) {
        x += 10;
        const float pd = lh * 1.15f;
        const float pw = mana_symbols_width(decision_.pips.size(), pd);
        if (x + pw < right) x += draw_mana_symbols(dl, decision_.pips, ImVec2(x, cy), pd) + 6;
        x += draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().DIM, "left to pay", right - x);
    }
    std::string rest = decision_.owed;
    if (!decision_.chosen.empty())
        rest += (rest.empty() ? "" : " \xC2\xB7 ") + std::string("chosen: ") + join_labels(decision_.chosen);
    if (!rest.empty() && x + 20 < right)
        draw_fitted_text(dl, font, fs, ImVec2(x, text_y), tok().DIM, " \xC2\xB7 " + rest, right - x);
    for (size_t i = 0; i < layout_.prompt_buttons.size() && i < decision_.buttons.size(); ++i) {
        const TRect &b = layout_.prompt_buttons[i];
        const ImVec2 ba = rmin(o, b), bz = rmax(o, b);
        const bool hot = d.answer && static_cast<int>(i) == hovered_button_;
        dl->AddRectFilled(ba, bz, hot ? tok().BTN_HOVER : tok().BTN, 2);
        dl->AddRect(ba, bz, d.links && d.links->row_linked(decision_.buttons[i]) ? tok().LINK : tok().LINE, 2);
        const std::string &t = decision_.row_text[static_cast<size_t>(decision_.buttons[i])];
        draw_fitted_text(dl, font, fs, ImVec2(ba.x + BUTTON_PAD / 2, ba.y + (b.h() - lh) / 2),
            d.answer ? tok().TEXT : tok().DIM, t, b.w() - BUTTON_PAD);
    }
}

// The twelve steps, the current one highlighted; in Play a stop dot under each (filled where the
// stops stop for the side whose turn it is), the step under the mouse outlined (a click toggles it).
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
        if (!d.play || !d.stops) continue;
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
        if (decision_.pick_of_seat(seat)) outline(dl, a, z, 2, tok().GLOW, 2, 3);
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

}  // namespace rmwork
