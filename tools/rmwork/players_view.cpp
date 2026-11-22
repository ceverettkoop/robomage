#include "players_view.h"

#include <algorithm>
#include <cctype>

#include "imgui_internal.h"
#include "layout.h"
#include "table_layout.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

static_assert(rmwork::ZONE_KINDS == rmwork::ZONE_TOOLS_PER_SEAT, "a zone tool per seat and zone");

namespace {

// A card a zone window lists without a handle (one its knowledge holds) is magnified by this id
// plus its index.
const uint64_t KNOWN_CARD_ID = 1ull << 62;
// The hidden cards' back in a zone window.
const uint64_t HIDDEN_ID = (1ull << 62) - 1;

}  // namespace

static std::string lower(std::string s);
static std::string player_extras(const rmwork::TablePlayer &p);

static std::string lower(std::string s) {
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// "poison 2 · energy 3 · monarch · speed 2 · pool {U}", the counters and designations after the life.
static std::string player_extras(const rmwork::TablePlayer &p) {
    std::vector<std::string> parts;
    if (p.poison) parts.push_back("poison " + std::to_string(p.poison));
    if (p.energy) parts.push_back("energy " + std::to_string(p.energy));
    for (const rmwork::TableCounter &c : p.counters) parts.push_back(lower(c.kind) + " " + std::to_string(c.count));
    if (p.monarch) parts.push_back("monarch");
    if (p.city_blessing) parts.push_back("city's blessing");
    if (p.speed >= 0) parts.push_back("speed " + std::to_string(p.speed));
    const std::string pool = rmwork::pool_text(p);
    if (!pool.empty()) parts.push_back("pool " + pool);
    std::string out;
    for (const std::string &s : parts) out += (out.empty() ? "" : " \xC2\xB7 ") + s;
    return out;
}

namespace rmwork {

void PlayersView::draw(ToolData &data, ReplaySession &s, LinkState *links) {
    (void)s;
    if (!data.ok()) {
        centered_text_block("Frame not readable", data.error().c_str(), 46.0f * ImGui::CalcTextSize("M").x);
        return;
    }
    const TableModel &t = data.models().table;
    draw_box(data, t.top, links);
    ImGui::Dummy(ImVec2(1, 2));
    draw_box(data, t.bottom, links);
}

void PlayersView::draw_box(const ToolData &data, int seat, LinkState *links) {
    const TablePlayer &p = data.models().table.players[static_cast<size_t>(seat)];
    const ZoneModel &zones = data.models().zones;
    const DecisionFrame &f = data.frame();
    ImGui::PushID(seat);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PANEL_2);
    ImGui::PushStyleColor(ImGuiCol_Border, tok().LINE);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(6, 4));
    ImGui::BeginChild("##player", ImVec2(0, 0),
        ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY | ImGuiChildFlags_AlwaysUseWindowPadding,
        ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    seat_chip(seat);
    ImGui::SameLine();
    const std::string agent = agent_label(*f.match, seat);
    ImGui::TextUnformatted(agent.empty() ? seat_name(seat) : agent.c_str());
    if (!agent.empty()) ImGui::SetItemTooltip("%s", f.match->agents[static_cast<size_t>(seat)].c_str());
    if (!p.deck.empty() || !agent.empty()) {
        ImGui::SameLine(0, 4);
        const std::string rest = std::string(agent.empty() ? "" : "\xC2\xB7 ") + seat_name(seat) +
                                 (p.deck.empty() ? "" : " \xC2\xB7 " + p.deck);
        ImGui::TextDisabled("%s", agent.empty() ? ("\xC2\xB7 " + p.deck).c_str() : rest.c_str());
    }
    // The life, large.
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 2.1f);
    ImGui::TextUnformatted(std::to_string(p.life).c_str());
    ImGui::PopFont();
    ImGui::SameLine(0, 6);
    const float base = ImGui::GetCursorPosY();
    ImGui::SetCursorPosY(base + ImGui::GetStyle().FontSizeBase * 2.1f - ImGui::GetTextLineHeight() - 3);
    const bool deciding = f.info && f.info->decider == seat;
    ImGui::TextDisabled("life%s", deciding ? " \xC2\xB7 deciding" : "");
    const std::string extras = player_extras(p);
    if (!extras.empty()) ImGui::TextUnformatted(extras.c_str());
    for (const std::string &e : p.effects) {
        ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
        ImGui::TextWrapped("%s", e.c_str());
        ImGui::PopStyleColor();
    }
    for (int z = 0; z < ZONE_KINDS; ++z) {
        const ZoneKind kind = static_cast<ZoneKind>(z);
        const ZoneList &list = zones.get(seat, kind);
        std::string tip = std::string(zone_title(kind)) + ": " + std::to_string(list.size);
        if (!list.all_known && list.size > 0)
            tip += " (" + std::to_string(list.listed()) + (list.listed() == 1 ? " card" : " cards") + " known to " +
                   view_name(f.view) + ")";
        tip += zone_open(seat, kind) ? " — open" : " (click to view)";
        if (z) ImGui::SameLine(0, 8);
        const std::string id = std::string("##zone_") + zone_key(kind);
        if (zone_button(id.c_str(), kind, list.size, tip, zone_open(seat, kind))) {
            if (zone_open(seat, kind)) {
                close_zone(seat, kind);
            } else {
                open_zone(seat, kind);
            }
        }
    }
    const bool hovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows);
    ImGui::EndChild();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor(2);
    if (links) {
        if (hovered) links->hover_player(seat);
        if (links->player_linked(seat))
            ImGui::GetWindowDrawList()->AddRect(
                ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), tok().LINK, 0.0f, ImDrawFlags_None, 2.0f);
    }
    ImGui::PopID();
}

void PlayersView::open_zone(int seat, ZoneKind z) {
    if (zone_open(seat, z)) return;
    ZoneWindow w;
    w.pane.seat = seat;
    w.pane.kind = z;
    windows_.push_back(w);
}

void PlayersView::close_zone(int seat, ZoneKind z) {
    windows_.erase(std::remove_if(windows_.begin(), windows_.end(),
                       [&](const ZoneWindow &w) { return w.pane.seat == seat && w.pane.kind == z; }),
        windows_.end());
}

bool PlayersView::zone_open(int seat, ZoneKind z) const {
    for (const ZoneWindow &w : windows_)
        if (w.pane.seat == seat && w.pane.kind == z) return true;
    return false;
}

uint64_t PlayersView::magnified() const {
    for (const ZoneWindow &w : windows_)
        if (w.pane.mag.handle()) return w.pane.mag.handle();
    for (const ZonePane &p : tool_panes_)
        if (p.mag.handle()) return p.mag.handle();
    return 0;
}

bool PlayersView::card_rect(int seat, ZoneKind z, size_t index, ImVec2 *min, ImVec2 *max) const {
    const ZonePane *found = nullptr;
    for (const ZoneWindow &w : windows_)
        if (!found && w.pane.seat == seat && w.pane.kind == z && index < w.pane.rects.size()) found = &w.pane;
    const ZonePane &tool = tool_panes_[static_cast<size_t>(seat * ZONE_KINDS + static_cast<int>(z))];
    if (!found && index < tool.rects.size()) found = &tool;
    if (!found) return false;
    *min = found->rects[index].first;
    *max = found->rects[index].second;
    return true;
}

std::string PlayersView::zone_window_title(int seat, ZoneKind z) {
    return std::string(seat_name(seat)) + " \xC2\xB7 " + zone_title(z);
}

void PlayersView::draw_zone_windows(ToolData &data, LinkState *links, ImGuiKeyChord magnify) {
    // Esc closes a magnifier before any window.
    bool magnifying = false;
    for (const ZoneWindow &w : windows_) magnifying = magnifying || w.pane.mag.handle() != 0;
    for (size_t i = 0; i < windows_.size();) {
        if (draw_zone_window(data, windows_[i], links, magnify, i, !magnifying)) {
            ++i;
        } else {
            windows_.erase(windows_.begin() + static_cast<std::ptrdiff_t>(i));
        }
    }
}

CardLook PlayersView::look_of(const ToolData &data, const ZoneCard &c) const {
    CardLook look;
    if (c.card >= 0 && !c.face_down) {
        look.face = data.frame().book->face(c.card, c.back);
        look.other = data.frame().book->face(c.card, !c.back);
    }
    if (c.face_down && !c.name.empty()) look.note = c.name;
    look.colors = c.colors;
    look.land = c.land;
    look.token = c.token;
    look.land_mana = c.land ? basic_land_mana(c.subtypes) : 0;
    return look;
}

// "6 cards, newest first", "53 cards; Player A knows 2", "7 cards: 1 known, 6 hidden".
std::string PlayersView::zone_caption(const ToolData &data, const ZoneList &z) {
    const std::string n = std::to_string(z.size) + (z.size == 1 ? " card" : " cards");
    const int view = data.frame().view;
    switch (z.kind) {
        case ZoneKind::GRAVEYARD:
        case ZoneKind::EXILE:
            return n + (z.size > 1 ? ", newest first" : "");
        case ZoneKind::LIBRARY:
            if (z.all_known) return n + ", top first";
            return n + "; " + std::string(view_name(view)) + " knows " + std::to_string(z.listed()) +
                   (z.listed() ? " of them, top first" : " of them");
        case ZoneKind::HAND:
            if (z.all_known) return n;
            return n + ": " + std::to_string(z.listed()) + " known to " + view_name(view) + ", " +
                   std::to_string(z.hidden) + " hidden";
    }
    return n;
}

// What is printed under a card in a zone window.
std::string PlayersView::card_label(const ZoneList &z, const ZoneCard &c) {
    if (z.kind == ZoneKind::LIBRARY && c.pos >= 0) {
        const std::string at = c.pos == 0 ? "top" : "#" + std::to_string(c.pos + 1) + " from top";
        return z.all_known ? at : at + " \xC2\xB7 " + c.note;
    }
    if (c.copies > 1) return "\xC3\x97" + std::to_string(c.copies) + " " + c.note;
    if (!c.note.empty()) return c.note;
    if (!c.counters.empty()) return c.counters.front();
    return c.name;
}

// The magnifier's lines for a card.
std::vector<std::string> PlayersView::card_lines(const ZoneList &z, const ZoneCard &c) {
    std::vector<std::string> out;
    out.push_back(c.name.empty() ? (c.face_down ? "A face-down card" : "A hidden card") : c.name);
    out.push_back(std::string(seat_name(z.seat)) + "'s " + lower(zone_title(z.kind)) +
                  (c.face_down ? " \xC2\xB7 face down" : "") + (c.token ? " \xC2\xB7 token" : ""));
    if (z.kind == ZoneKind::LIBRARY && c.pos >= 0)
        out.push_back(c.pos == 0 ? "On top" : "Position " + std::to_string(c.pos + 1) + " from the top");
    if (c.copies > 1) out.push_back(std::to_string(c.copies) + " copies");
    if (!c.note.empty()) out.push_back(c.note);
    for (const std::string &k : c.counters) out.push_back("Counters: " + k);
    return out;
}

// False when the window was closed.
bool PlayersView::draw_zone_window(
    ToolData &data, ZoneWindow &w, LinkState *links, ImGuiKeyChord magnify, size_t index, bool esc_closes) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    if (!w.placed) {
        // Side by side across the middle of the workbench, in the order they were opened.
        const float ww = std::min(ImGui::GetFontSize() * 40, vp->Size.x / 2 - 24), wh = ImGui::GetFontSize() * 26;
        const size_t per_row = std::max<size_t>(1, static_cast<size_t>((vp->Size.x - 32) / (ww + 16)));
        const size_t n = std::min(windows_.size(), per_row);
        const float row_w = static_cast<float>(n) * (ww + 16) - 16;
        const float x = vp->Pos.x + (vp->Size.x - row_w) / 2 + static_cast<float>(index % per_row) * (ww + 16);
        const float y = vp->Pos.y + vp->Size.y * 0.22f + static_cast<float>(index / per_row % 3) * 32;
        ImGui::SetNextWindowPos(ImVec2(x, y), ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(ww, wh), ImGuiCond_Always);
        ImGui::SetNextWindowFocus();
        w.placed = true;
    }
    bool open = true;
    const std::string title = zone_window_title(w.pane.seat, w.pane.kind) + "###zone." + std::to_string(w.pane.seat) +
                              "." + zone_key(w.pane.kind);
    const ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse;
    ImGui::PushStyleColor(ImGuiCol_WindowBg, tok().PANEL);
    const bool visible = ImGui::Begin(title.c_str(), &open, flags);
    ImGui::PopStyleColor();
    if (!visible) {
        ImGui::End();
        return open;
    }
    const bool focused = ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
    draw_zone_pane(data, w.pane, links, magnify);
    if (focused && esc_closes && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) open = false;
    ImGui::End();
    return open;
}

void PlayersView::draw_zone_tool(ToolData &data, int seat, ZoneKind z, LinkState *links, ImGuiKeyChord magnify) {
    ZonePane &p = tool_panes_[static_cast<size_t>(seat * ZONE_KINDS + static_cast<int>(z))];
    p.seat = seat;
    p.kind = z;
    draw_zone_pane(data, p, links, magnify);
}

void PlayersView::draw_zone_pane(ToolData &data, ZonePane &p, LinkState *links, ImGuiKeyChord magnify) {
    if (!data.ok()) {
        ImGui::TextDisabled("%s", data.error().c_str());
        return;
    }
    const ZoneList &z = data.models().zones.get(p.seat, p.kind);
    ImGui::TextDisabled("%s", zone_caption(data, z).c_str());
    const float cw = std::round(ImGui::GetFontSize() * 6.0f), ch = std::round(cw * CARD_ASPECT);
    const float gap = 8, label_h = ImGui::GetTextLineHeight();
    uint64_t under = 0;
    ImVec2 under_min, under_max;
    const ZoneCard *under_card = nullptr;
    ZoneCard hidden_card;
    std::vector<std::pair<ImVec2, ImVec2>> rects;  // per card (and the hidden cards' back) its rectangle
    const bool glow = data.frame().sees_decision() && data.frame().view < 2;
    ImGui::BeginChild("##cards", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()), ImGuiChildFlags_None);
    const float avail = ImGui::GetContentRegionAvail().x;
    float x = 0;
    const size_t n = z.cards.size() + (z.hidden > 0 ? 1 : 0);
    if (n == 0) ImGui::TextDisabled("Empty.");
    for (size_t i = 0; i < n; ++i) {
        const bool hidden = i == z.cards.size();
        const ZoneCard &c = hidden ? hidden_card : z.cards[i];
        if (i > 0 && x + gap + cw <= avail) {
            ImGui::SameLine(0, gap);
            x += gap;
        } else {
            x = 0;
        }
        ImGui::PushID(static_cast<int>(i));
        ImGui::BeginGroup();
        ImGui::InvisibleButton("##card", ImVec2(cw, ch), ImGuiButtonFlags_MouseButtonLeft);
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        rects.emplace_back(a, b);
        ImDrawList *dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(ImVec2(a.x + 1, a.y + 2), ImVec2(b.x + 1, b.y + 3), hex_color(0x000000, 120), cw * 0.05f);
        renderer_->draw(dl, hidden ? CardLook() : look_of(data, c), a, b, false, 1.0f);
        const uint64_t id = hidden ? HIDDEN_ID : c.handle ? c.handle : KNOWN_CARD_ID + i;
        if (ImGui::IsItemHovered()) {
            under = id;
            under_min = a;
            under_max = b;
            under_card = hidden ? nullptr : &c;
            if (links && c.handle && !hidden) links->hover_object(c.handle);
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) ImGui::SetWindowFocus();
            if (links && ImGui::IsItemClicked(ImGuiMouseButton_Left) && c.handle && !hidden) {
                LinkItem item;
                item.kind = LinkItem::Kind::OBJECT;
                item.handle = c.handle;
                links->select(item);
            }
        }
        if (!hidden && c.handle && links) {
            if (glow && links->is_source(c.handle))
                dl->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), tok().GLOW, cw * 0.05f + 2,
                    ImDrawFlags_None, 3.0f);
            if (links->object_linked(c.handle)) {
                dl->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), hex_color(0x000000), cw * 0.05f + 2,
                    ImDrawFlags_None, 3.0f);
                dl->AddRect(ImVec2(a.x - 5, a.y - 5), ImVec2(b.x + 5, b.y + 5), tok().LINK, cw * 0.05f + 5,
                    ImDrawFlags_None, 3.0f);
            }
        }
        const std::string label = hidden ? std::to_string(z.hidden) + " hidden" : card_label(z, c);
        const ImVec2 lp = ImGui::GetCursorScreenPos();
        draw_fitted_text(dl, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(lp.x, lp.y + 2), tok().DIM, label, cw);
        ImGui::Dummy(ImVec2(cw, label_h + 4));
        ImGui::EndGroup();
        ImGui::PopID();
        x += cw;
    }
    const bool over = ImGui::IsWindowHovered();
    ImGui::EndChild();
    ImGui::TextDisabled("Hold %s over a card, or right-click it, to magnify it.",
        magnify ? ImGui::GetKeyChordName(magnify) : "the magnify key");
    p.mag.update(under, over, magnify);
    if (!force_mag_.empty()) {
        const std::string want = lower(force_mag_);
        for (size_t i = 0; i < z.cards.size(); ++i)
            if (!z.cards[i].name.empty() && lower(z.cards[i].name).find(want) != std::string::npos) {
                p.mag.open(z.cards[i].handle ? z.cards[i].handle : KNOWN_CARD_ID + i);
                break;
            }
    }
    if (p.mag.handle()) {
        // The magnified card: the one under the mouse, else the one it was opened on (found again).
        const ZoneCard *card = under == p.mag.handle() ? under_card : nullptr;
        ImVec2 mn = under_min, mx = under_max;
        if (!card && p.mag.handle() != HIDDEN_ID)
            for (size_t i = 0; i < z.cards.size(); ++i)
                if ((z.cards[i].handle ? z.cards[i].handle : KNOWN_CARD_ID + i) == p.mag.handle()) {
                    card = &z.cards[i];
                    mn = rects[i].first;
                    mx = rects[i].second;
                }
        if (p.mag.handle() == HIDDEN_ID && under != HIDDEN_ID && !rects.empty()) {
            mn = rects.back().first;
            mx = rects.back().second;
        }
        if (card) {
            draw_magnified_card(*renderer_, look_of(data, *card), card_lines(z, *card), mn, mx);
        } else if (p.mag.handle() == HIDDEN_ID) {
            draw_magnified_card(*renderer_, CardLook(),
                {std::to_string(z.hidden) + " hidden " + (z.hidden == 1 ? "card" : "cards"),
                    std::string(view_name(data.frame().view)) + " cannot name them"},
                mn, mx);
        } else {
            p.mag.close();
        }
    }
    p.rects = rects;
}

}  // namespace rmwork
