#include "choice_panel_view.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "imgui_internal.h"
#include "magnifier.h"
#include "table_layout.h"
#include "theme.h"

namespace {

const float PAD = 6;
const float ROW_PAD = 5;
// The spell's card at the left of a modal spell's modes, and a trigger's source beside its row, in font
// sizes wide.
const float MODE_CARD_FS = 7.5f;
const float ROW_CARD_FS = 2.6f;
// The panel's width in font sizes: the rows' and the lists'.
const float ROWS_WIDTH_FS = 54;
const float LIST_WIDTH_FS = 50;

}  // namespace

static void outline(ImDrawList *dl, ImVec2 a, ImVec2 b, ImU32 color, float pad, float thickness);
static float badge(ImDrawList *dl, ImVec2 pos, bool right, const std::string &text, ImU32 bg, ImU32 fg);
static bool rows_kind(const rmwork::ChoicePanel &p);
static float row_card_width();
static float mode_card_width(const rmwork::ChoicePanel &p);

static void outline(ImDrawList *dl, ImVec2 a, ImVec2 b, ImU32 color, float pad, float thickness) {
    dl->AddRect(ImVec2(a.x - pad, a.y - pad), ImVec2(b.x + pad, b.y + pad), color, 4, 0, thickness);
}

// A small label in a filled box at `pos` (its top-left, or top-right when `right`); returns its width.
static float badge(ImDrawList *dl, ImVec2 pos, bool right, const std::string &text, ImU32 bg, ImU32 fg) {
    const ImVec2 ts = ImGui::CalcTextSize(text.c_str());
    const float w = ts.x + 8, h = ts.y + 2;
    const ImVec2 a(right ? pos.x - w : pos.x, pos.y);
    dl->AddRectFilled(a, ImVec2(a.x + w, a.y + h), bg, 3);
    dl->AddText(ImVec2(a.x + 4, a.y + 1), fg, text.c_str());
    return w;
}

// Modes and triggers are a column of rows; names and an overflowing decision's choices a searchable list.
static bool rows_kind(const rmwork::ChoicePanel &p) {
    return !p.searchable;
}

// A trigger's source card beside its row.
static float row_card_width() {
    return std::round(ImGui::GetFontSize() * ROW_CARD_FS);
}

// A modal spell's card at the left of its modes; 0 for the other kinds.
static float mode_card_width(const rmwork::ChoicePanel &p) {
    return p.kind == rmwork::DecisionKind::MODES && p.source_card >= 0 ? std::round(ImGui::GetFontSize() * MODE_CARD_FS)
                                                                       : 0;
}

namespace rmwork {

void ChoicePanelView::set_filter(const std::string &f) {
    filter_ = f.substr(0, sizeof filter_buf_ - 1);
    std::memset(filter_buf_, 0, sizeof filter_buf_);
    std::memcpy(filter_buf_, filter_.data(), filter_.size());
}

// A new decision starts with the forced filter (none by default), the panel open.
void ChoicePanelView::reset_for(const ChoicePanel &p) {
    std::string key = p.title;
    for (const ChoiceItem &it : p.items) key += "|" + std::to_string(it.row) + it.text;
    if (key == key_) return;
    key_ = key;
    set_filter(force_filter_);
    collapsed_ = false;
}

CardLook ChoicePanelView::look_of(const CardBook &book, int card, bool back) const {
    CardLook look;
    look.face = card >= 0 ? book.face(card, back) : nullptr;
    if (look.face) {
        look.other = book.face(card, !back);
        look.colors = look.face->colors;
        look.land = look.face->land;
        look.token = look.face->token;
        look.land_mana = look.land ? basic_land_mana(look.face->subtypes) : 0;
    }
    return look;
}

// The rows' height at `width` (the text wrapped beside the card and before the note).
float ChoicePanelView::rows_height(const ChoicePanel &p, float width) const {
    const float lh = ImGui::GetTextLineHeight();
    const float cw = p.kind == DecisionKind::ORDER ? row_card_width() : 0;
    float h = 0;
    for (const ChoiceItem &it : p.items) {
        const float note_w = ImGui::CalcTextSize(it.note.c_str()).x + 12;
        const float text_w = std::max(40.0f, width - 2 * ROW_PAD - (cw > 0 ? cw + 8 : lh + 8) - note_w);
        const float th = ImGui::CalcTextSize(it.text.c_str(), nullptr, false, text_w).y;
        h += std::max(th, cw > 0 ? std::round(cw * CARD_ASPECT) : lh) + 2 * ROW_PAD + 4;
    }
    return h;
}

int ChoicePanelView::draw(
    const ChoicePanel &p, const CardBook &book, const CardPickerView::Place &place, bool answering, LinkState *links) {
    reset_for(p);
    items_.clear();
    items_shown_ = 0;
    shown_ = true;
    const float lh = ImGui::GetTextLineHeight(), fh = ImGui::GetFrameHeight(), fs = ImGui::GetFontSize();
    const float sp = ImGui::GetStyle().ItemSpacing.y;
    const float full = place.x1 - place.x0;
    const float title_h = lh + 8;
    const float footer_h = fh + sp + 4;
    const float tool_h = p.searchable ? fh + sp : 0;
    const float chrome = title_h + tool_h + footer_h + 2 * PAD;
    const float width = std::min(full, std::round(fs * (rows_kind(p) ? ROWS_WIDTH_FS : LIST_WIDTH_FS)));
    const float mode_w = mode_card_width(p);
    float body = 0;
    if (rows_kind(p)) {
        const float rows_w = width - 2 * PAD - (mode_w > 0 ? mode_w + PAD : 0) - ImGui::GetStyle().ScrollbarSize;
        body = std::max(rows_height(p, rows_w), mode_w > 0 ? std::round(mode_w * CARD_ASPECT) : 0.0f) + 4;
    } else {
        const float rows = 1 + static_cast<float>(p.items.size() + p.groups.size());
        body = rows * (fh + 2) + 8;
    }
    // The short band (to the middle strip) when it holds the rows, else the long one (to the prompt line),
    // scrolling past that.
    float height = chrome + body;
    if (height > place.y_short - place.y0) height = std::min(height, place.y_long - place.y0);
    if (collapsed_) height = title_h + 2;
    height = std::max(height, title_h + 2);
    const float x0 = std::round(place.x0 + (full - width) / 2), x1 = x0 + width;
    min_ = ImVec2(x0, place.y0);
    max_ = ImVec2(x1, place.y0 + height);

    int answer = -1;
    ImGui::SetCursorScreenPos(min_);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PANEL);
    ImGui::PushStyleColor(ImGuiCol_Border, tok().LINE);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("##choice_panel", ImVec2(width, height), ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    title_bar(p, width);
    if (!collapsed_) {
        const float body_h = std::max(lh, height - chrome);
        float y = min_.y + title_h + PAD;
        if (p.searchable) {
            ImGui::SetCursorScreenPos(ImVec2(x0 + PAD, y));
            ImGui::AlignTextToFramePadding();
            ImGui::TextUnformatted("Search");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(std::round(fs * 16));
            if (ImGui::InputTextWithHint("##choice_filter", "type a name\xE2\x80\xA6", filter_buf_, sizeof filter_buf_))
                filter_ = filter_buf_;
            filter_min_ = ImGui::GetItemRectMin();
            filter_max_ = ImGui::GetItemRectMax();
            if (!filter_.empty()) {
                ImGui::SameLine(0, 2);
                if (ImGui::SmallButton("x##choice_clear")) set_filter("");
            }
            size_t shown = 0;
            for (const ChoiceItem &it : p.items) shown += choice_item_matches(it, filter_) ? 1 : 0;
            ImGui::SameLine(0, 16);
            ImGui::TextDisabled("%zu of %zu", shown, p.items.size());
            y += tool_h;
        }
        if (mode_w > 0) {
            const ImVec2 a(x0 + PAD, y), b(a.x + mode_w, a.y + std::round(mode_w * CARD_ASPECT));
            renderer_->draw(ImGui::GetWindowDrawList(), look_of(book, p.source_card, false), a, b, false, 1.0f);
        }
        const float bx = x0 + PAD + (mode_w > 0 ? mode_w + PAD : 0);
        ImGui::SetCursorScreenPos(ImVec2(bx, y));
        const ImVec2 size(x1 - PAD - bx, body_h);
        const int a =
            rows_kind(p) ? draw_rows(p, book, size, answering, links) : draw_list(p, book, size, answering, links);
        if (a >= 0) answer = a;
        ImGui::SetCursorScreenPos(ImVec2(x0 + PAD, max_.y - footer_h - PAD + sp));
        const int f = draw_footer(p, answering);
        if (f >= 0) answer = f;
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    return answering ? answer : -1;
}

// The title bar: what is chosen, and the collapse button.
void ChoicePanelView::title_bar(const ChoicePanel &p, float width) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float lh = ImGui::GetTextLineHeight();
    const ImVec2 a = min_, b(min_.x + width, min_.y + lh + 8);
    dl->AddRectFilled(a, b, tok().TAB_ACTIVE);
    const float bw = lh + 6;
    draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(a.x + PAD + 2, a.y + 4), tok().TITLE_TEXT,
        p.title + (collapsed_ ? "  (collapsed)" : ""), width - bw - 3 * PAD);
    ImGui::SetCursorScreenPos(ImVec2(b.x - bw - 3, a.y + 2));
    ImGui::PushStyleColor(ImGuiCol_Button, 0);
    if (ImGui::Button(collapsed_ ? "+##choice_fold" : "_##choice_fold", ImVec2(bw, lh + 4))) collapsed_ = !collapsed_;
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(collapsed_ ? "Open the panel" : "Collapse the panel to see the board");
}

// Modes and triggers: a row each, the text wrapped, the note at the right.
int ChoicePanelView::draw_rows(
    const ChoicePanel &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links) {
    int answer = -1;
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("##choice_rows", size, ImGuiChildFlags_None);
    ImGui::PopStyleVar();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImFont *font = ImGui::GetFont();
    const float fs = ImGui::GetFontSize(), lh = ImGui::GetTextLineHeight();
    const float width = size.x - ImGui::GetStyle().ScrollbarSize - 2;
    const float cw = p.kind == DecisionKind::ORDER ? row_card_width() : 0;
    for (size_t i = 0; i < p.items.size(); ++i) {
        const ChoiceItem &it = p.items[i];
        const float note_w = ImGui::CalcTextSize(it.note.c_str()).x + 12;
        const float lead = cw > 0 ? cw + 8 : lh + 8;
        const float text_w = std::max(40.0f, width - 2 * ROW_PAD - lead - note_w);
        const float th = ImGui::CalcTextSize(it.text.c_str(), nullptr, false, text_w).y;
        const float h = std::max(th, cw > 0 ? std::round(cw * CARD_ASPECT) : lh) + 2 * ROW_PAD;
        const ImVec2 a = ImGui::GetCursorScreenPos(), b(a.x + width, a.y + h);
        ImGui::PushID(static_cast<int>(i));
        ImGui::InvisibleButton("##choice", ImVec2(width, h));
        const bool hovered = ImGui::IsItemHovered();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && answering && it.row >= 0) answer = it.row;
        ImGui::PopID();
        ImGui::Dummy(ImVec2(1, 2));
        const bool chosen = it.row >= 0 && it.row == p.chosen_row;
        const bool live = it.row >= 0;
        dl->AddRectFilled(a, b, hovered && answering && live ? tok().BTN_HOVER : tok().PANEL_2, 3);
        if (chosen) {
            outline(dl, a, b, tok().ACCENT, 0, 2);
        } else {
            outline(dl, a, b, live ? tok().GLOW : tok().LINE, 0, live ? 1.5f : 1.0f);
        }
        if (links && live && links->row_linked(it.row)) outline(dl, a, b, tok().LINK, 2, 2);
        float x = a.x + ROW_PAD;
        if (cw > 0) {
            const ImVec2 ca(x, a.y + ROW_PAD), cb(x + cw, a.y + ROW_PAD + std::round(cw * CARD_ASPECT));
            renderer_->draw(dl, look_of(book, it.card, it.back), ca, cb, false, 1.0f);
        } else {
            // A tick box: filled for a mode chosen (earlier, or here in Replay).
            const float s = lh * 0.8f;
            const ImVec2 ba(x + 1, a.y + ROW_PAD + (lh - s) / 2), bb(ba.x + s, ba.y + s);
            dl->AddRect(ba, bb, live ? tok().TEXT : tok().DIM, 2);
            if (it.chosen_before || chosen) {
                dl->AddRectFilled(ImVec2(ba.x + 2, ba.y + 2), ImVec2(bb.x - 2, bb.y - 2), tok().ACCENT, 1);
            }
        }
        x += lead;
        dl->AddText(
            font, fs, ImVec2(x, a.y + ROW_PAD), live ? tok().TEXT : tok().DIM, it.text.c_str(), nullptr, text_w);
        const ImVec2 ns = ImGui::CalcTextSize(it.note.c_str());
        dl->AddText(
            ImVec2(b.x - ROW_PAD - ns.x, a.y + ROW_PAD), it.chosen_before ? tok().ACCENT : tok().DIM, it.note.c_str());
        if (chosen) badge(dl, ImVec2(b.x - ROW_PAD, b.y - lh - 4), true, "Chosen", tok().ACCENT, tok().TEXT);
        if (hovered) {
            if (links && live) links->hover_menu_row(it.row);
            if (answering && live) ImGui::SetTooltip("Click: %s", p.row_text[static_cast<size_t>(it.row)].c_str());
        }
        Item rec;
        rec.row = it.row;
        rec.label = it.text;
        rec.min = a;
        rec.max = b;
        items_.push_back(rec);
        ++items_shown_;
    }
    ImGui::EndChild();
    return answer;
}

// Names and an overflowing decision's choices: a list under the filter, grouped; a name's card magnified
// beside the panel while hovered.
int ChoicePanelView::draw_list(
    const ChoicePanel &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links) {
    int answer = -1;
    ImGui::BeginChild("##choice_list_scroll", size, ImGuiChildFlags_None);
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp;
    const ChoiceItem *hovered_item = nullptr;
    ImVec2 hover_a, hover_b;
    const bool typed = !p.groups.empty();
    if (ImGui::BeginTable("##choice_list", typed ? 3 : 2, flags)) {
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000").x);
        ImGui::TableSetupColumn("Choice", ImGuiTableColumnFlags_WidthStretch, 3);
        if (typed) ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 2);
        const size_t groups = std::max<size_t>(1, p.groups.size());
        for (size_t g = 0; g < groups; ++g) {
            bool header = false;
            for (size_t i = 0; i < p.items.size(); ++i) {
                const ChoiceItem &it = p.items[i];
                if ((typed && it.group != static_cast<int>(g)) || !choice_item_matches(it, filter_)) continue;
                if (typed && !header) {
                    header = true;
                    size_t n = 0;
                    for (const ChoiceItem &x : p.items) n += x.group == static_cast<int>(g) ? 1 : 0;
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(1);
                    ImGui::TextDisabled("%s \xC2\xB7 %zu", p.groups[g].c_str(), n);
                }
                ++items_shown_;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::TextDisabled("%d", it.row);
                ImGui::TableSetColumnIndex(1);
                ImGui::PushID(static_cast<int>(i));
                const bool chosen = it.row >= 0 && it.row == p.chosen_row;
                const std::string label = (chosen ? "\xE2\x96\xB6 " : "") + it.text;
                if (ImGui::Selectable(label.c_str(), chosen, ImGuiSelectableFlags_SpanAllColumns) && answering &&
                    it.row >= 0)
                    answer = it.row;
                ImGui::PopID();
                Item rec;
                rec.row = it.row;
                rec.label = it.text;
                rec.min = ImGui::GetItemRectMin();
                rec.max = ImGui::GetItemRectMax();
                items_.push_back(rec);
                if (ImGui::IsItemHovered()) {
                    hovered_item = &it;
                    hover_a = rec.min;
                    hover_b = rec.max;
                    if (links) links->hover_menu_row(it.row);
                }
                if (typed) {
                    ImGui::TableSetColumnIndex(2);
                    ImGui::TextDisabled("%s", it.type_line.c_str());
                }
            }
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    if (hovered_item && hovered_item->card >= 0 && book.face(hovered_item->card, false)) {
        // The card beside the panel, not over the list.
        std::vector<std::string> lines{hovered_item->text};
        if (!hovered_item->type_line.empty()) lines.push_back(hovered_item->type_line);
        lines.push_back(answering ? "Click to choose it" : "Read-only");
        draw_magnified_card(*renderer_, look_of(book, hovered_item->card, false), lines, ImVec2(min_.x, hover_a.y),
            ImVec2(max_.x, hover_b.y));
    } else if (hovered_item && answering) {
        ImGui::SetTooltip("Click: %s", p.row_text[static_cast<size_t>(hovered_item->row)].c_str());
    }
    return answer;
}

// The note, and the rows offered at the foot (Done choosing modes) as buttons at the right.
int ChoicePanelView::draw_footer(const ChoicePanel &p, bool answering) {
    int answer = -1;
    const float right = max_.x - PAD;
    float bw = 0;
    for (int r : p.buttons)
        bw += ImGui::CalcTextSize(p.row_text[static_cast<size_t>(r)].c_str()).x + 2 * ImGui::GetStyle().FramePadding.x +
              6;
    const std::string hint = answering ? p.note : "Read-only: the row chosen is marked";
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::AlignTextToFramePadding();
    draw_fitted_text(ImGui::GetWindowDrawList(), ImGui::GetFont(), ImGui::GetFontSize(),
        ImVec2(at.x, at.y + ImGui::GetStyle().FramePadding.y), tok().DIM, hint, std::max(0.0f, right - bw - at.x - 8));
    float x = right - bw;
    for (int r : p.buttons) {
        const std::string &label = p.row_text[static_cast<size_t>(r)];
        ImGui::SetCursorScreenPos(ImVec2(x, at.y));
        const bool chosen = r == p.chosen_row;
        if (chosen) ImGui::PushStyleColor(ImGuiCol_Button, tok().SEL);
        ImGui::BeginDisabled(!answering && !chosen);
        ImGui::PushID(r);
        if (ImGui::Button(label.c_str()) && answering) answer = r;
        ImGui::PopID();
        ImGui::EndDisabled();
        if (chosen) ImGui::PopStyleColor();
        Item it;
        it.row = r;
        it.button = true;
        it.label = label;
        it.min = ImGui::GetItemRectMin();
        it.max = ImGui::GetItemRectMax();
        items_.push_back(it);
        x = it.max.x + 6;
    }
    if (p.buttons.empty()) {
        ImGui::SetCursorScreenPos(at);
        ImGui::Dummy(ImVec2(1, ImGui::GetFrameHeight()));
    }
    return answer;
}

}  // namespace rmwork
