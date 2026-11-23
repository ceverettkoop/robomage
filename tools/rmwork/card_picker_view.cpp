#include "card_picker_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "imgui_internal.h"
#include "table_layout.h"
#include "theme.h"

namespace {

const float PAD = 6;
const float MAX_CARD = 120;  // the grid's card width, largest and smallest
const float MIN_CARD = 52;
// The smallest card the short and long bands take before the picker grows into the next.
const float BAND_CARD = 92;
const float CARD_STEP = 4;
const float DIM_ALPHA = 0.35f;

}  // namespace

static void outline(ImDrawList *dl, ImVec2 a, ImVec2 b, ImU32 color, float pad, float thickness);
static float badge(ImDrawList *dl, ImVec2 pos, bool right, const std::string &text, ImU32 bg, ImU32 fg);
static std::string place_text(const rmwork::CardPickEntry &e);
static std::string join_names(const std::vector<rmwork::PileCard> &cards);

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

// Where an entry lies: a library card's place from the top, "" elsewhere.
static std::string place_text(const rmwork::CardPickEntry &e) {
    if (e.zone != rmwork::ZoneKind::LIBRARY || e.other_zone || e.pos < 0) return "";
    return rmwork::depth_label(e.pos);
}

static std::string join_names(const std::vector<rmwork::PileCard> &cards) {
    std::string out;
    for (const rmwork::PileCard &c : cards) out += (out.empty() ? "" : ", ") + c.name;
    return out;
}

namespace rmwork {

void CardPickerView::set_filter(const std::string &f) {
    filter_ = f.substr(0, sizeof filter_buf_ - 1);
    std::memset(filter_buf_, 0, sizeof filter_buf_);
    std::memcpy(filter_buf_, filter_.data(), filter_.size());
}

// A new resolution starts with no filter, the picker open.
void CardPickerView::reset_for(const CardPicker &p) {
    if (p.source == source_) return;
    source_ = p.source;
    set_filter(force_filter_);
    collapsed_ = false;
}

bool CardPickerView::visible(const CardPickEntry &e) const {
    return (unpickable_ || e.pickable()) && picker_entry_matches(e, filter_);
}

CardLook CardPickerView::look_of(const CardBook &book, const CardPickEntry &e) const {
    CardLook look;
    if (e.card >= 0) {
        look.face = book.face(e.card, e.back);
        look.other = book.face(e.card, !e.back);
    }
    look.colors = e.colors ? e.colors : (look.face ? look.face->colors : 0);
    look.land = e.land || (look.face && look.face->land);
    look.token = e.token;
    look.land_mana =
        look.land ? basic_land_mana(e.subtypes.empty() && look.face ? look.face->subtypes : e.subtypes) : 0;
    return look;
}

CardLook CardPickerView::look_of(const CardBook &book, const PileCard &c) const {
    CardLook look;
    look.face = c.card >= 0 ? book.face(c.card, c.back) : nullptr;
    if (look.face) {
        look.colors = look.face->colors;
        look.land = look.face->land;
        look.land_mana = look.land ? basic_land_mana(look.face->subtypes) : 0;
    }
    return look;
}

// The grid's cells at card width `cw` in `width`: each group on lines of its own under its heading.
CardPickerView::Flow CardPickerView::flow(const CardPicker &p, float cw, float width, int buttons) const {
    Flow f;
    const float lh = ImGui::GetTextLineHeight();
    const float ch = std::round(cw * CARD_ASPECT), gap = std::round(cw * 0.12f), group_gap = 3 * gap;
    const float cell_h = ch + lh + 4 + static_cast<float>(buttons) * (ImGui::GetFrameHeight() + 2);
    const float head = lh + 4;
    // A group follows the one before on its line when it fits there whole; one wider wraps onto lines
    // of its own.
    float x = 0, y = 0, line_h = 0;
    for (size_t g = 0; g < p.groups.size(); ++g) {
        size_t n = 0;
        for (const CardPickEntry &e : p.groups[g].entries) n += visible(e) ? 1 : 0;
        if (n == 0) {
            f.labels.push_back(ImVec2(0, y));
            continue;
        }
        const float title_w =
            ImGui::CalcTextSize(p.groups[g].title.c_str()).x + ImGui::CalcTextSize(" \xC2\xB7 00 cards, top first").x;
        const float gw = std::max(static_cast<float>(n) * cw + static_cast<float>(n - 1) * gap, title_w);
        if (x > 0 && x + gw > width) {
            x = 0;
            y += line_h + gap;
            line_h = 0;
        }
        f.labels.push_back(ImVec2(x, y));
        f.width = std::max(f.width, x + title_w);
        float cx = x, cy = y + head;
        bool wrapped = false;
        for (size_t i = 0; i < p.groups[g].entries.size(); ++i) {
            if (!visible(p.groups[g].entries[i])) continue;
            if (cx > x && cx + cw > width) {
                cx = x;
                cy += cell_h + gap;
                wrapped = true;
            }
            f.cells.push_back(Cell{g, i, ImVec2(cx, cy)});
            f.width = std::max(f.width, cx + cw);
            cx += cw + gap;
        }
        line_h = std::max(line_h, cy + cell_h - y);
        if (wrapped) {
            x = 0;
            y += line_h + gap;
            line_h = 0;
        } else {
            x = std::max(cx - gap, x + gw) + group_gap;
        }
    }
    f.height = y + line_h;
    return f;
}

float CardPickerView::list_height(const CardPicker &p) const {
    float rows = 1;  // the header
    for (const CardPickGroup &g : p.groups) {
        bool any = false;
        for (const CardPickEntry &e : g.entries)
            if (visible(e)) {
                ++rows;
                any = true;
            }
        if (any) ++rows;
    }
    return rows * (ImGui::GetFrameHeight() + 2) + 8;
}

// The lines of placement buttons under a card `cw` wide: 1 when every entry's fit side by side, else the
// most an entry has (stacked); 0 when no entry has several rows.
int CardPickerView::placement_rows(const CardPicker &p, float cw) const {
    size_t n = 0;
    bool side_by_side = true;
    for (const CardPickGroup &g : p.groups)
        for (const CardPickEntry &e : g.entries) {
            if (e.rows.size() < 2) continue;
            n = std::max(n, e.rows.size());
            side_by_side = side_by_side && buttons_fit(p, e, cw);
        }
    return n == 0 ? 0 : side_by_side ? 1 : static_cast<int>(n);
}

// Whether entry `e`'s placement buttons fit side by side under a card `cw` wide.
bool CardPickerView::buttons_fit(const CardPicker &p, const CardPickEntry &e, float cw) const {
    float w = 0;
    for (size_t k = 0; k < e.rows.size(); ++k) {
        const std::string &label = e.actions[k].empty() ? p.row_text[static_cast<size_t>(e.rows[k])] : e.actions[k];
        w += ImGui::CalcTextSize(label.c_str()).x + 2 * ImGui::GetStyle().FramePadding.x + (k ? 2 : 0);
    }
    return w <= cw;
}

std::string CardPickerView::click_text(const CardPicker &p, const CardPickEntry &e) const {
    if (e.rows.size() == 1) return "Click: " + p.row_text[static_cast<size_t>(e.rows[0])];
    if (e.rows.size() > 1) return "Choose where it goes with the buttons under it";
    return "Shown only: no row picks it";
}

int CardPickerView::draw(const CardPicker &p, const CardBook &book, const Place &place, bool answering,
    ImGuiKeyChord magnify, LinkState *links) {
    reset_for(p);
    items_.clear();
    entries_shown_ = 0;
    shown_ = true;
    const bool grid = p.off_table() || p.entry_count() > 0;
    const float lh = ImGui::GetTextLineHeight(), fh = ImGui::GetFrameHeight(), fs = ImGui::GetFontSize();
    const float sp = ImGui::GetStyle().ItemSpacing.y;
    const float pile_w = p.pile.shown ? std::round(fs * 15) : 0;
    const float full = place.x1 - place.x0;
    const float title_h = lh + 8;
    const float footer_h = fh + sp + 4;
    const float tool_h = grid ? fh + sp : 0;
    const float chrome = title_h + tool_h + footer_h + 2 * PAD;
    const float side = 2 * PAD + (pile_w > 0 ? pile_w + PAD : 0) + ImGui::GetStyle().ScrollbarSize + 4;
    const float grid_w = full - side;
    // The grid's content beyond its cells: its last Dummy.
    const float grid_extra = 4;
    float pile_need = 0;
    if (p.pile.shown) {
        const float th = std::round(lh * 2.2f);
        pile_need = 3 * lh + static_cast<float>(p.pile.top.size()) * (th + 4) + 4 * lh;
        if (!p.pile.bottom.empty()) pile_need += 2 * lh;
        if (!p.pile.graveyard.empty()) pile_need += 2 * lh;
        if (!p.pile.chosen.empty()) pile_need += 2 * lh;
    }
    // The largest card width whose grid fits the short band with cards at least BAND_CARD wide, else the
    // long one, else the tall one down to the smallest.
    float height = 0, cw = MIN_CARD;
    bool found = false;
    int chosen_band = 2;
    tall_ = false;
    for (int band = 0; band < 3 && !found; ++band) {
        const float room = (band == 0 ? place.y_short : band == 1 ? place.y_long : place.y_tall) - place.y0;
        for (float w = MAX_CARD; w >= (band < 2 ? BAND_CARD : MIN_CARD) && !found; w -= CARD_STEP) {
            const int buttons = placement_rows(p, w);
            const float body = grid ? (list_ ? list_height(p) : flow(p, w, grid_w, buttons).height + grid_extra) : 0;
            const float need = chrome + std::max(body, pile_need);
            if (need <= room) {
                height = need;
                cw = w;
                found = true;
                chosen_band = band;
                tall_ = band == 2;
            }
        }
    }
    if (!found) {
        height = place.y_tall - place.y0;
        tall_ = true;
    }
    if (collapsed_) height = title_h + 2;
    height = std::max(height, title_h + 2);
    card_w_ = cw;
    // Over the opponent's half as wide as its cards (and the toolbar) need, centred; over the middle strip
    // as wide as the Table, covering the strip's header; the pile alone at the right.
    float width = full;
    if (!grid) {
        width = std::min(full, pile_w + 2 * PAD + 4);
    } else if (!list_ && chosen_band == 0 && !collapsed_) {
        width =
            std::min(full, std::max(std::round(fs * 46), flow(p, cw, grid_w, placement_rows(p, cw)).width + side + 2));
    }
    const float x0 = grid ? std::round(place.x0 + (full - width) / 2) : place.x1 - width;
    const float x1 = x0 + width;
    min_ = ImVec2(x0, place.y0);
    max_ = ImVec2(x1, place.y0 + height);

    int answer = -1;
    ImGui::SetCursorScreenPos(min_);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PANEL);
    ImGui::PushStyleColor(ImGuiCol_Border, tok().LINE);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("##card_picker", ImVec2(width, height), ImGuiChildFlags_Borders,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    ImGui::PopStyleVar();
    title_bar(p, width);
    if (!collapsed_) {
        ImGui::SetCursorScreenPos(ImVec2(x0 + PAD, min_.y + title_h + PAD));
        if (grid) toolbar(p);
        const float body_h = std::max(lh, height - chrome);
        const ImVec2 body_at(x0 + PAD, min_.y + title_h + PAD + tool_h);
        if (grid) {
            ImGui::SetCursorScreenPos(body_at);
            const ImVec2 size(width - 2 * PAD - (pile_w > 0 ? pile_w + PAD : 0), body_h);
            const int a = list_ ? draw_list(p, book, size, answering, links)
                                : draw_grid(p, book, size, answering, magnify, links);
            if (a >= 0) answer = a;
        }
        if (pile_w > 0) {
            ImGui::SetCursorScreenPos(ImVec2(x1 - PAD - pile_w, body_at.y));
            draw_pile(p, book, ImVec2(pile_w, body_h));
        }
        ImGui::SetCursorScreenPos(ImVec2(x0 + PAD, max_.y - footer_h - PAD + sp));
        const int a = draw_footer(p, answering, tall_ ? place.prompt : std::string());
        if (a >= 0) answer = a;
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(2);
    return answering ? answer : -1;
}

// The title bar: what is picked from where, and the collapse button.
void CardPickerView::title_bar(const CardPicker &p, float width) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float lh = ImGui::GetTextLineHeight();
    const ImVec2 a = min_, b(min_.x + width, min_.y + lh + 8);
    dl->AddRectFilled(a, b, tok().TAB_ACTIVE);
    const float bw = lh + 6;
    draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(a.x + PAD + 2, a.y + 4), tok().TITLE_TEXT,
        p.title + (collapsed_ ? "  (collapsed)" : ""), width - bw - 3 * PAD);
    ImGui::SetCursorScreenPos(ImVec2(b.x - bw - 3, a.y + 2));
    ImGui::PushStyleColor(ImGuiCol_Button, 0);
    if (ImGui::Button(collapsed_ ? "+##picker_fold" : "_##picker_fold", ImVec2(bw, lh + 4))) collapsed_ = !collapsed_;
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(collapsed_ ? "Open the picker" : "Collapse the picker to see the board");
}

// The filter, the Cards / List switch, the dimmed cards' toggle and the count shown.
void CardPickerView::toolbar(const CardPicker &p) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Filter");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::round(ImGui::GetFontSize() * 13));
    if (ImGui::InputTextWithHint("##picker_filter", "name or type", filter_buf_, sizeof filter_buf_))
        filter_ = filter_buf_;
    filter_min_ = ImGui::GetItemRectMin();
    filter_max_ = ImGui::GetItemRectMax();
    if (!filter_.empty()) {
        ImGui::SameLine(0, 2);
        if (ImGui::SmallButton("x##picker_clear")) set_filter("");
    }
    ImGui::SameLine(0, 16);
    if (ImGui::RadioButton("Cards", !list_)) list_ = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("List", list_)) list_ = true;
    bool dimmed = false;
    size_t total = 0, shown = 0;
    for (const CardPickGroup &g : p.groups)
        for (const CardPickEntry &e : g.entries) {
            dimmed = dimmed || !e.pickable();
            ++total;
            if (visible(e)) ++shown;
        }
    if (dimmed) {
        ImGui::SameLine(0, 16);
        ImGui::Checkbox("Unpickable", &unpickable_);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Show the cards no row picks (dimmed): the rest of what is revealed");
    }
    ImGui::SameLine(0, 16);
    ImGui::TextDisabled("%zu of %zu shown", shown, total);
}

// The placement buttons of an entry with several rows, side by side in `width`; the row pressed, else -1.
int CardPickerView::placement_buttons(
    const CardPicker &p, const CardPickEntry &e, size_t g, size_t i, bool answering, float width, const ImVec2 *at) {
    int answer = -1;
    const float bw =
        at ? width
           : std::floor((width - 2 * static_cast<float>(e.rows.size() - 1)) / static_cast<float>(e.rows.size()));
    for (size_t k = 0; k < e.rows.size(); ++k) {
        if (at) {
            ImGui::SetCursorScreenPos(ImVec2(at->x, at->y + static_cast<float>(k) * (ImGui::GetFrameHeight() + 2)));
        } else if (k) {
            ImGui::SameLine(0, 2);
        }
        const int row = e.rows[k];
        const std::string label = e.actions[k].empty() ? p.row_text[static_cast<size_t>(row)] : e.actions[k];
        const bool chosen = row == p.chosen_row;
        if (chosen) ImGui::PushStyleColor(ImGuiCol_Button, tok().SEL);
        ImGui::BeginDisabled(!answering && !chosen);
        ImGui::PushID(row);
        if (ImGui::Button(label.c_str(), ImVec2(std::max(bw, 10.0f), 0)) && answering) answer = row;
        ImGui::PopID();
        ImGui::EndDisabled();
        if (chosen) ImGui::PopStyleColor();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", p.row_text[static_cast<size_t>(row)].c_str());
        Item it;
        it.row = row;
        it.group = g;
        it.entry = i;
        it.label = label;
        it.min = ImGui::GetItemRectMin();
        it.max = ImGui::GetItemRectMax();
        items_.push_back(it);
    }
    return answer;
}

int CardPickerView::draw_grid(
    const CardPicker &p, const CardBook &book, ImVec2 size, bool answering, ImGuiKeyChord magnify, LinkState *links) {
    int answer = -1;
    const int buttons = placement_rows(p, card_w_);
    // No padding: the grid's width is the one draw() fitted the cards to.
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::BeginChild("##picker_cards", size, ImGuiChildFlags_None);
    ImGui::PopStyleVar();
    const float cw = card_w_, ch = std::round(cw * CARD_ASPECT), lh = ImGui::GetTextLineHeight();
    const float width = size.x - ImGui::GetStyle().ScrollbarSize - 4;
    const Flow f = flow(p, cw, width, buttons);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    for (size_t g = 0; g < p.groups.size(); ++g) {
        const CardPickGroup &grp = p.groups[g];
        bool any = false;
        for (const CardPickEntry &e : grp.entries) any = any || visible(e);
        if (!any) continue;
        const ImVec2 at(origin.x + f.labels[g].x, origin.y + f.labels[g].y);
        dl->AddText(at, tok().TEXT, grp.title.c_str());
        std::string sub;
        if (!grp.other_zone) sub = std::to_string(grp.zone_size) + (grp.zone_size == 1 ? " card" : " cards");
        if (grp.zone == ZoneKind::LIBRARY && !grp.other_zone) sub += ", top first";
        if (!sub.empty())
            dl->AddText(ImVec2(at.x + ImGui::CalcTextSize(grp.title.c_str()).x + 8, at.y), tok().DIM,
                ("\xC2\xB7 " + sub).c_str());
    }
    uint64_t under = 0;
    const CardPickEntry *under_e = nullptr;
    ImVec2 under_a, under_b;
    for (const Cell &c : f.cells) {
        const CardPickEntry &e = p.groups[c.group].entries[c.entry];
        ++entries_shown_;
        const ImVec2 a(origin.x + c.at.x, origin.y + c.at.y), b(a.x + cw, a.y + ch);
        ImGui::SetCursorScreenPos(a);
        ImGui::PushID(static_cast<int>(c.group * 1000 + c.entry));
        ImGui::InvisibleButton(
            "##pick", ImVec2(cw, ch), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
        const bool hovered = ImGui::IsItemHovered();
        if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && answering && e.rows.size() == 1) answer = e.rows[0];
        ImGui::PopID();
        Item it;
        it.card = true;
        it.row = e.rows.size() == 1 ? e.rows[0] : -1;
        it.group = c.group;
        it.entry = c.entry;
        it.label = e.name;
        it.min = a;
        it.max = b;
        items_.push_back(it);
        dl->AddRectFilled(ImVec2(a.x + 1, a.y + 2), ImVec2(b.x + 1, b.y + 3), hex_color(0x000000, 140), cw * 0.05f);
        renderer_->draw(dl, look_of(book, e), a, b, false, e.pickable() ? 1.0f : DIM_ALPHA);
        bool chosen = false;
        for (int r : e.rows) chosen = chosen || r == p.chosen_row;
        if (e.pickable() && !chosen)
            outline(dl, a, b, hovered && answering ? tok().LINK : tok().GLOW, 2, hovered ? 3 : 2);
        if (chosen) {
            outline(dl, a, b, tok().ACCENT, 2, 3);
            badge(dl, ImVec2(a.x + 2, a.y + 2), false, "Chosen", tok().ACCENT, tok().TEXT);
        }
        if (links && links->object_linked(e.handles[0])) outline(dl, a, b, tok().LINK, 5, 2);
        if (e.count() > 1)
            badge(dl, ImVec2(b.x - 2, a.y + 2), true, "\xC3\x97" + std::to_string(e.count()), tok().TABLE_SHADE,
                tok().TEXT);
        const std::string where = place_text(e);
        draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(a.x, b.y + 3),
            e.pickable() ? tok().DIM : tok().LINE, where.empty() ? e.name : where, cw);
        if (e.rows.size() > 1) {
            const ImVec2 under(a.x, b.y + lh + 5);
            ImGui::SetCursorScreenPos(under);
            const int r = placement_buttons(p, e, c.group, c.entry, answering, cw, buttons > 1 ? &under : nullptr);
            if (r >= 0) answer = r;
        }
        if (hovered) {
            under = e.handles[0];
            under_e = &e;
            under_a = a;
            under_b = b;
            if (links) links->hover_object(e.handles[0]);
        }
    }
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + f.height));
    ImGui::Dummy(ImVec2(1, 1));
    // A new decision shows the card chosen there (Replay) in view.
    if (scroll_) {
        scroll_ = false;
        size_t cg = 0, ce = 0;
        if (p.chosen_row >= 0 && p.entry_of_row(p.chosen_row, &cg, &ce))
            for (const Cell &c : f.cells)
                if (c.group == cg && c.entry == ce)
                    ImGui::SetScrollY(std::max(0.0f, c.at.y - ImGui::GetTextLineHeight() - 4));
    }
    const bool over = ImGui::IsWindowHovered();
    ImGui::EndChild();
    mag_.update(under, over, magnify);
    if (under_e && mag_.handle() == under) {
        std::vector<std::string> lines{under_e->name};
        if (!under_e->type_line.empty()) lines.push_back(under_e->type_line);
        if (under_e->count() > 1) lines.push_back(std::to_string(under_e->count()) + " identical cards");
        const std::string where = place_text(*under_e);
        if (!where.empty()) lines.push_back(where == "Top" ? "On top of the library" : where + " from the top");
        draw_magnified_card(*renderer_, look_of(book, *under_e), lines, under_a, under_b);
    } else if (under_e) {
        std::string tip = under_e->name;
        if (under_e->count() > 1) tip += " \xC3\x97" + std::to_string(under_e->count());
        if (!under_e->type_line.empty()) tip += "\n" + under_e->type_line;
        tip += "\n" + click_text(p, *under_e);
        if (magnify) tip += std::string("\nHold ") + ImGui::GetKeyChordName(magnify) + " or right-click to magnify";
        ImGui::SetTooltip("%s", tip.c_str());
    }
    return answer;
}

int CardPickerView::draw_list(
    const CardPicker &p, const CardBook &book, ImVec2 size, bool answering, LinkState *links) {
    int answer = -1;
    (void)book;
    ImGui::BeginChild("##picker_list_scroll", size, ImGuiChildFlags_None);
    const ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp;
    if (ImGui::BeginTable("##picker_list", 5, flags)) {
        ImGui::TableSetupColumn("Card", ImGuiTableColumnFlags_WidthStretch, 3);
        ImGui::TableSetupColumn("\xC3\x97", ImGuiTableColumnFlags_WidthFixed,
            ImGui::CalcTextSize("\xC3\x97"
                                "00")
                .x);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 3);
        ImGui::TableSetupColumn("Place", ImGuiTableColumnFlags_WidthStretch, 2);
        ImGui::TableSetupColumn("Pick", ImGuiTableColumnFlags_WidthStretch, 2);
        ImGui::TableHeadersRow();
        for (size_t g = 0; g < p.groups.size(); ++g) {
            const CardPickGroup &grp = p.groups[g];
            bool any = false;
            for (const CardPickEntry &e : grp.entries) any = any || visible(e);
            if (!any) continue;
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TextDisabled("%s", grp.title.c_str());
            for (size_t i = 0; i < grp.entries.size(); ++i) {
                const CardPickEntry &e = grp.entries[i];
                if (!visible(e)) continue;
                ++entries_shown_;
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(static_cast<int>(g * 1000 + i));
                bool chosen = false;
                for (int r : e.rows) chosen = chosen || r == p.chosen_row;
                if (!e.pickable()) ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
                const std::string label = (chosen ? "\xE2\x96\xB6 " : "") + e.name;
                if (ImGui::Selectable(label.c_str(), chosen, ImGuiSelectableFlags_None) && answering &&
                    e.rows.size() == 1)
                    answer = e.rows[0];
                if (!e.pickable()) ImGui::PopStyleColor();
                Item it;
                it.card = true;
                it.row = e.rows.size() == 1 ? e.rows[0] : -1;
                it.group = g;
                it.entry = i;
                it.label = e.name;
                it.min = ImGui::GetItemRectMin();
                it.max = ImGui::GetItemRectMax();
                items_.push_back(it);
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("%s", click_text(p, e).c_str());
                    if (links) links->hover_object(e.handles[0]);
                }
                ImGui::TableSetColumnIndex(1);
                if (e.count() > 1) ImGui::Text("%d", e.count());
                ImGui::TableSetColumnIndex(2);
                ImGui::TextDisabled("%s", e.type_line.c_str());
                ImGui::TableSetColumnIndex(3);
                ImGui::TextDisabled("%s", place_text(e).c_str());
                ImGui::TableSetColumnIndex(4);
                if (e.rows.size() > 1) {
                    const int r = placement_buttons(p, e, g, i, answering, ImGui::GetContentRegionAvail().x, nullptr);
                    if (r >= 0) answer = r;
                } else if (e.pickable()) {
                    ImGui::TextDisabled("click");
                }
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    ImGui::EndChild();
    return answer;
}

// The pile being built: the library's top, top first, the next pick's place marked; the other piles.
void CardPickerView::draw_pile(const CardPicker &p, const CardBook &book, ImVec2 size) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PANEL_2);
    ImGui::BeginChild("##picker_pile", size, ImGuiChildFlags_Borders);
    const PickerPile &pile = p.pile;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float lh = ImGui::GetTextLineHeight();
    if (!pile.top.empty()) {
        ImGui::TextUnformatted("Library top, as it will be");
        const float th = std::round(lh * 2.2f), tw = std::round(th / CARD_ASPECT);
        const float label_w = ImGui::CalcTextSize("10th").x + 6;
        for (const PileSlot &s : pile.top) {
            const ImVec2 a = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            dl->AddText(ImVec2(a.x, a.y + (th - lh) / 2), s.next ? tok().GLOW : tok().DIM, s.label.c_str());
            const ImVec2 ca(a.x + label_w, a.y), cb(ca.x + tw, a.y + th);
            if (s.filled) {
                renderer_->draw(dl, look_of(book, s.card), ca, cb, false, 1.0f);
                const std::string from = s.card.from.empty() ? "" : " (" + s.card.from + ")";
                draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(cb.x + 6, a.y + (th - lh) / 2),
                    tok().TEXT, s.card.name + from, a.x + w - cb.x - 6);
            } else {
                dl->AddRect(ca, cb, s.next ? tok().GLOW : tok().LINE, 3, 0, s.next ? 2.0f : 1.0f);
                draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(cb.x + 6, a.y + (th - lh) / 2),
                    s.next ? tok().GLOW : tok().DIM, s.next ? "\xE2\x86\x90 next pick" : "to fill", a.x + w - cb.x - 6);
            }
            ImGui::Dummy(ImVec2(w, th + 4));
        }
    }
    const auto list = [](const char *title, const std::vector<PileCard> &cards) {
        if (cards.empty()) return;
        ImGui::Spacing();
        ImGui::TextUnformatted(title);
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextDisabled("%s", join_names(cards).c_str());
        ImGui::PopTextWrapPos();
    };
    list("On the bottom", pile.bottom);
    list("Into the graveyard", pile.graveyard);
    list("Chosen so far", pile.chosen);
    ImGui::EndChild();
    ImGui::PopStyleColor();
}

// The hint, and the rows naming no card as buttons at the right; the row pressed, else -1.
int CardPickerView::draw_footer(const CardPicker &p, bool answering, const std::string &prompt) {
    int answer = -1;
    const bool grid = p.off_table() || p.entry_count() > 0;
    const float right = max_.x - PAD;
    std::vector<int> rows = grid ? p.buttons : std::vector<int>();
    float bw = 0;
    for (int r : rows)
        bw += ImGui::CalcTextSize(p.row_text[static_cast<size_t>(r)].c_str()).x + 2 * ImGui::GetStyle().FramePadding.x +
              6;
    const char *hint = answering ? (grid ? "Click a lit card to choose it" : "Click a lit card in your hand")
                                 : "Read-only: the row chosen is marked";
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::AlignTextToFramePadding();
    draw_fitted_text(ImGui::GetWindowDrawList(), ImGui::GetFont(), ImGui::GetFontSize(),
        ImVec2(at.x, at.y + ImGui::GetStyle().FramePadding.y), prompt.empty() ? tok().DIM : tok().TEXT,
        prompt.empty() ? std::string(hint) : prompt + " \xC2\xB7 " + hint, std::max(0.0f, right - bw - at.x - 8));
    float x = right - bw;
    for (int r : rows) {
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
        it.label = label;
        it.min = ImGui::GetItemRectMin();
        it.max = ImGui::GetItemRectMax();
        items_.push_back(it);
        x = it.max.x + 6;
    }
    if (rows.empty()) {
        ImGui::SetCursorScreenPos(at);
        ImGui::Dummy(ImVec2(1, ImGui::GetFrameHeight()));
    }
    return answer;
}

}  // namespace rmwork
