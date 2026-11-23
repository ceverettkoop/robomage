#include "start_page.h"

#include <algorithm>

#include "imgui.h"
#include "imgui_internal.h"
#include "modal.h"
#include "theme.h"
#include "tool_widgets.h"

namespace {

// The side columns' widths, in characters ("M").
const float LEFT_CHARS = 34.0f;
const float RIGHT_CHARS = 44.0f;
// The share of the middle column's height the recent games and the recent replays may take.
const float GAMES_SHARE = 0.44f;
const float REPLAYS_SHARE = 0.26f;

}  // namespace

static ImU32 tone_color(int tone);
static ImU32 health_color(rmwork::Health h);
static std::string short_log_path(const std::string &path);
static float grid_height(size_t rows, float cap);
static void cell_text(const std::string &text, ImU32 color);
static bool open_row(const char *id, bool selected);
static void key_chip(const std::string &chord);
static std::string grouped(int64_t n);

// The page's result tones: +1 good, -1 bad, 2 a warning, 0 plain.
static ImU32 tone_color(int tone) {
    using rmwork::tok;
    return tone == 1 ? tok().GOOD : tone == -1 ? tok().BAD : tone == 2 ? tok().FOLDER : tok().TEXT;
}

static ImU32 health_color(rmwork::Health h) {
    using rmwork::tok;
    switch (h) {
        case rmwork::Health::GOOD:
            return tok().GOOD;
        case rmwork::Health::WARN:
            return tok().FOLDER;
        case rmwork::Health::BAD:
            return tok().BAD;
        case rmwork::Health::NONE:
            break;
    }
    return tok().DIM;
}

// ".../<parent>/<name>.rmlog".
static std::string short_log_path(const std::string &path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos || slash == 0) return path;
    const size_t parent = path.rfind('/', slash - 1);
    return parent == std::string::npos ? path : "\xE2\x80\xA6" + path.substr(parent);
}

// A grid of `rows` rows and its header, at most `cap` tall.
static float grid_height(size_t rows, float cap) {
    const float row = ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().CellPadding.y;
    return std::min(static_cast<float>(rows + 1) * row + 4, std::max(cap, 3 * row));
}

static void cell_text(const std::string &text, ImU32 color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

// A row's first cell: a selectable spanning the row; true when it was double-clicked.
static bool open_row(const char *id, bool selected) {
    const ImGuiSelectableFlags flags =
        ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick | ImGuiSelectableFlags_AllowOverlap;
    const bool clicked = ImGui::Selectable(id, selected, flags);
    return clicked && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
}

// A key chord in a thin frame, as an item ("unbound" dimmed for none).
static void key_chip(const std::string &chord) {
    const std::string text = chord.empty() ? "unbound" : chord;
    const ImVec2 q = ImGui::GetCursorScreenPos();
    const ImVec2 sz = ImGui::CalcTextSize(text.c_str());
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRect(ImVec2(q.x, q.y - 1), ImVec2(q.x + sz.x + 8, q.y + sz.y + 1), rmwork::tok().LINE, 2);
    dl->AddText(ImVec2(q.x + 4, q.y), chord.empty() ? rmwork::tok().DIM : rmwork::tok().TEXT, text.c_str());
    ImGui::Dummy(ImVec2(sz.x + 8, sz.y));
}

// "430 595": thousands set apart by a space.
static std::string grouped(int64_t n) {
    std::string digits = std::to_string(n < 0 ? -n : n), out;
    for (size_t i = 0; i < digits.size(); ++i) {
        if (i && (digits.size() - i) % 3 == 0) out += " ";
        out += digits[i];
    }
    return (n < 0 ? "-" : "") + out;
}

namespace rmwork {

StartPage::~StartPage() {
    if (thread_.joinable()) thread_.join();
}

void StartPage::refresh(const StartScanRequest &r) {
    if (running_) return;
    if (thread_.joinable()) thread_.join();
    done_.store(false);
    running_ = true;
    stale_ = false;
    thread_ = std::thread([this, r] {
        result_ = scan_start(r);
        done_.store(true);
    });
}

bool StartPage::poll() {
    if (!running_ || !done_.load()) return false;
    thread_.join();
    running_ = false;
    scanned_ = true;
    data_ = std::move(result_);
    result_ = StartData();
    if (selected_game_ >= static_cast<int>(data_.games.size())) selected_game_ = -1;
    if (selected_replay_ >= static_cast<int>(data_.replays.size())) selected_replay_ = -1;
    if (selected_run_ >= static_cast<int>(data_.runs.size())) selected_run_ = -1;
    return true;
}

StartRequest StartPage::draw(const StartContext &c) {
    StartRequest req;
    // The tab strip: "Start", and the ✕ that shows the mode's layout.
    if (ImGui::BeginTabBar("##start_tabs", ImGuiTabBarFlags_None)) {
        if (ImGui::BeginTabItem("Start")) ImGui::EndTabItem();
        ImGui::EndTabBar();
    }
    {
        const ImGuiWindow *w = ImGui::GetCurrentWindow();
        const float size = ImGui::GetFontSize();
        const ImVec2 pos(w->Pos.x + w->Size.x - size - 10, w->Pos.y + ImGui::GetStyle().WindowPadding.y + 2);
        if (ImGui::CloseButton(ImGui::GetID("##start_close"), pos)) req.kind = StartRequest::Kind::HIDE;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Close the start page: the mode's layout (File \xE2\x86\x92 Start Page brings it back)");
    }
    ImGui::Spacing();
    const float cw = char_width();
    const float gap = 1.5f * cw;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float left = std::min(LEFT_CHARS * cw, avail.x * 0.26f);
    const float right = std::min(RIGHT_CHARS * cw, avail.x * 0.3f);
    const float middle = std::max(10 * cw, avail.x - left - right - 2 * gap);
    ImGui::BeginChild("##start_left", ImVec2(left, avail.y), ImGuiChildFlags_None);
    draw_left(c, &req);
    ImGui::EndChild();
    ImGui::SameLine(0, gap);
    ImGui::BeginChild("##start_middle", ImVec2(middle, avail.y), ImGuiChildFlags_None);
    draw_middle(&req);
    ImGui::EndChild();
    ImGui::SameLine(0, gap);
    ImGui::BeginChild("##start_right", ImVec2(0, avail.y), ImGuiChildFlags_None);
    draw_right(c, &req);
    ImGui::EndChild();
    return req;
}

// The title, the actions, the keys and the setting.
void StartPage::draw_left(const StartContext &c, StartRequest *req) {
    const float width = ImGui::GetContentRegionAvail().x;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float k = ImGui::GetFontSize() / 14.0f;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float logo = 32 * k;
    dl->AddRectFilled(p, ImVec2(p.x + logo, p.y + logo), tok().LOGO_BG, 3);
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.6f);
    const ImVec2 r = ImGui::CalcTextSize("R");
    dl->AddText(ImVec2(p.x + (logo - r.x) / 2, p.y + (logo - r.y) / 2), tok().LOGO_FG, "R");
    ImGui::PopFont();
    ImGui::SetCursorScreenPos(ImVec2(p.x + logo + 10, p.y - 2));
    ImGui::BeginGroup();
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.3f);
    ImGui::TextUnformatted("Robomage Workbench");
    ImGui::PopFont();
    ImGui::PushTextWrapPos(p.x + width - ImGui::GetWindowPos().x);
    ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
    ImGui::TextWrapped("Play a model \xC2\xB7 replay and analyse games");
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    ImGui::SetCursorScreenPos(ImVec2(p.x, std::max(ImGui::GetCursorScreenPos().y, p.y + logo) + 12));
    for (size_t i = 0; i < c.actions.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (action_card(c.actions[i], width)) req->kind = c.actions[i].kind;
        ImGui::PopID();
        ImGui::Dummy(ImVec2(1, 4));
    }
    ImGui::Dummy(ImVec2(1, 6));
    if (ImGui::BeginChild("##start_keys", ImVec2(width, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY)) {
        ImGui::TextUnformatted("Keys");
        const ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit;
        if (ImGui::BeginTable("##keys", 2, flags)) {
            ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed);
            ImGui::TableSetupColumn("what", ImGuiTableColumnFlags_WidthStretch);
            for (const auto &kv : c.keys) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                key_chip(kv.first);
                ImGui::TableNextColumn();
                ImGui::TextWrapped("%s", kv.second.c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();
    ImGui::Dummy(ImVec2(1, 6));
    if (c.show_at_start) {
        ImGui::Checkbox("##show_at_start", c.show_at_start);
        ImGui::SetItemTooltip("At start, and when a game or replay is closed (File \xE2\x86\x92 Close)");
        ImGui::SameLine(0, 6);
        ImGui::TextWrapped("Show this page when nothing is open");
    }
}

// The recent games, replays and runs.
void StartPage::draw_middle(StartRequest *req) {
    const float h = ImGui::GetContentRegionAvail().y;
    const float header = ImGui::GetTextLineHeightWithSpacing() + 10;
    section_header("Recent games", "Play's games", "double-click: Replay");
    draw_games(grid_height(data_.games.size(), h * GAMES_SHARE - header), req);
    ImGui::Dummy(ImVec2(1, 8));
    section_header("Recent replays", "decision logs opened", "double-click: Replay");
    draw_replays(grid_height(data_.replays.size(), h * REPLAYS_SHARE - header), req);
    ImGui::Dummy(ImVec2(1, 8));
    section_header("Recent runs", "opened, and the runs directory's", "double-click: the run's logs");
    draw_runs(std::max(ImGui::GetContentRegionAvail().y, 3 * ImGui::GetTextLineHeightWithSpacing()), req);
}

void StartPage::draw_games(float height, StartRequest *req) {
    if (data_.games.empty()) {
        empty_note(!scanned_ ? "Reading the games directory\xE2\x80\xA6"
                             : "No games yet: New Game plays one, and its decision log is listed here.");
        return;
    }
    if (!begin_grid("##start_games", 6, ImVec2(0, height), true)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("When");
    ImGui::TableSetupColumn("You");
    ImGui::TableSetupColumn("Opponent", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Format");
    ImGui::TableSetupColumn("Result");
    ImGui::TableSetupColumn("Dec.");
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < data_.games.size(); ++i) {
        const RecentGame &g = data_.games[i];
        ImGui::TableNextRow();
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextColumn();
        const bool bad = g.state == GameState::MALFORMED;
        if (open_row(g.when.c_str(), selected_game_ == static_cast<int>(i)) && !bad) {
            req->kind = StartRequest::Kind::OPEN_LOG;
            req->path = g.path;
        }
        if (ImGui::IsItemClicked()) selected_game_ = static_cast<int>(i);
        ImGui::SetItemTooltip("%s%s%s", g.path.c_str(), bad ? "\n" : "", bad ? g.error.c_str() : "");
        ImGui::TableNextColumn();
        if (bad) {
            cell_text(g.error, tok().DIM);
        } else {
            if (g.seat >= 0) {
                seat_chip(g.seat);
                ImGui::SameLine(0, 6);
            }
            ImGui::TextUnformatted(g.seat >= 0 ? game_own_deck(g).c_str() : g.deck_a.c_str());
            ImGui::TableNextColumn();
            const std::string opp = g.seat >= 0 ? game_opponent_deck(g) : g.deck_b;
            cell_text((g.opponent.empty() ? std::string("\xE2\x80\xA6") : g.opponent) + " \xC2\xB7 " + opp, tok().TEXT);
        }
        if (bad) ImGui::TableNextColumn();
        ImGui::TableNextColumn();
        if (!bad) ImGui::TextUnformatted(game_format_text(g));
        ImGui::TableNextColumn();
        int tone = 0;
        const std::string result = game_result_text(g, &tone);
        cell_text(result, tone_color(tone));
        if (g.state == GameState::UNKNOWN)
            ImGui::SetItemTooltip("No result lines to read: open it to see the result");
        else if (g.state == GameState::STOPPED)
            ImGui::SetItemTooltip(
                "The game ended without a result (the workbench quit, or the engine was stopped); "
                "Replay shows it to its last decision");
        ImGui::TableNextColumn();
        if (!bad) cell_text(std::to_string(g.decisions), tok().TEXT);
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void StartPage::draw_replays(float height, StartRequest *req) {
    if (data_.replays.empty()) {
        empty_note(!scanned_ ? "Reading\xE2\x80\xA6"
                             : "Decision logs opened in Replay (File \xE2\x86\x92 Open Replay) are listed here.");
        return;
    }
    if (!begin_grid("##start_replays", 5, ImVec2(0, height), true)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Decision log", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Decks (A vs B)", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Result");
    ImGui::TableSetupColumn("Dec.");
    ImGui::TableSetupColumn("Modified");
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < data_.replays.size(); ++i) {
        const RecentGame &g = data_.replays[i];
        const bool bad = g.state == GameState::MALFORMED;
        ImGui::TableNextRow();
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextColumn();
        if (open_row(short_log_path(g.path).c_str(), selected_replay_ == static_cast<int>(i)) && !bad) {
            req->kind = StartRequest::Kind::OPEN_LOG;
            req->path = g.path;
        }
        if (ImGui::IsItemClicked()) selected_replay_ = static_cast<int>(i);
        ImGui::SetItemTooltip("%s%s%s", g.path.c_str(), bad ? "\n" : "", bad ? g.error.c_str() : "");
        ImGui::TableNextColumn();
        cell_text(bad ? g.error : g.deck_a + " vs " + g.deck_b, bad ? tok().DIM : tok().TEXT);
        ImGui::TableNextColumn();
        int tone = 0;
        cell_text(game_result_text(g, &tone), tone_color(tone));
        if (g.state == GameState::UNKNOWN)
            ImGui::SetItemTooltip("Not replayed by this engine build yet: the result shows once it is opened");
        ImGui::TableNextColumn();
        if (!bad) cell_text(std::to_string(g.decisions), tok().TEXT);
        ImGui::TableNextColumn();
        cell_text(g.when, tok().DIM);
        ImGui::PopID();
    }
    ImGui::EndTable();
}

void StartPage::draw_runs(float height, StartRequest *req) {
    if (data_.runs.empty()) {
        empty_note(!scanned_ ? "Reading\xE2\x80\xA6"
                             : "No runs: the runs directory (Settings \xE2\x86\x92 Models) has none, and none was "
                               "opened (File \xE2\x86\x92 Open Run Directory).");
        return;
    }
    if (!begin_grid("##start_runs", 6, ImVec2(0, height), true)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Phase");
    ImGui::TableSetupColumn("Updated");
    ImGui::TableSetupColumn("Hours");
    ImGui::TableSetupColumn("Games");
    ImGui::TableSetupColumn("From");
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < data_.runs.size(); ++i) {
        const RecentRun &run = data_.runs[i];
        ImGui::TableNextRow();
        ImGui::PushID(static_cast<int>(i));
        ImGui::TableNextColumn();
        if (open_row(run.name.c_str(), selected_run_ == static_cast<int>(i))) {
            req->kind = StartRequest::Kind::OPEN_RUN;
            req->path = run.path;
        }
        if (ImGui::IsItemClicked()) selected_run_ = static_cast<int>(i);
        ImGui::SetItemTooltip(
            "%s%s%s", run.path.c_str(), run.error.empty() ? "" : "\nstatus.json: ", run.error.c_str());
        ImGui::TableNextColumn();
        const std::string phase =
            !run.status ? "\xE2\x80\xA6"
                        : (run.phase.empty() ? "?" : run.phase) + (run.running ? " \xC2\xB7 running" : "");
        cell_text(phase, run.running ? tok().GOOD : run.status ? tok().TEXT : tok().DIM);
        ImGui::TableNextColumn();
        cell_text(run.updated > 0 ? short_time(static_cast<int64_t>(run.updated)) : "\xE2\x80\xA6", tok().DIM);
        ImGui::TableNextColumn();
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.1f", run.hours);
        cell_text(run.hours >= 0 ? buf : "\xE2\x80\xA6", tok().TEXT);
        ImGui::TableNextColumn();
        cell_text(run.games >= 0 ? grouped(run.games) : "\xE2\x80\xA6", tok().TEXT);
        ImGui::TableNextColumn();
        cell_text(run.recent ? "recent" : "runs/", tok().DIM);
        ImGui::PopID();
    }
    ImGui::EndTable();
}

// The status blocks.
void StartPage::draw_right(const StartContext &c, StartRequest *req) {
    const float width = ImGui::GetContentRegionAvail().x;
    for (size_t i = 0; i < c.status.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        status_box(c.status[i], width, req);
        ImGui::PopID();
        ImGui::Dummy(ImVec2(1, 6));
    }
}

// A framed action: an accent bar, its icon, title and key, and what it does; true when clicked.
bool StartPage::action_card(const StartAction &a, float width) {
    const float pad = 8;
    const float icon_w = 22 * ImGui::GetFontSize() / 14.0f;
    const float text_w = width - pad * 2 - icon_w - 6;
    const float line = ImGui::GetTextLineHeight();
    const float body = ImGui::CalcTextSize(a.text.c_str(), nullptr, false, text_w).y;
    const float h = pad * 2 + line + 4 + body;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const bool pressed = ImGui::InvisibleButton("##card", ImVec2(width, h));
    const bool hovered = ImGui::IsItemHovered();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + width, p.y + h), hovered ? tok().BTN_HOVER : tok().PANEL_2, 2);
    dl->AddRect(p, ImVec2(p.x + width, p.y + h), tok().LINE, 2);
    dl->AddRectFilled(
        p, ImVec2(p.x + 3, p.y + h), a.kind == StartRequest::Kind::NEW_GAME ? tok().GOOD : tok().ACCENT, 1);
    const ImVec2 isz = icon_size(a.icon);
    draw_icon(dl, a.icon, ImVec2(p.x + pad + (icon_w - isz.x) / 2, p.y + pad + (line - isz.y) / 2));
    const float tx = p.x + pad + icon_w + 6;
    dl->AddText(ImVec2(tx, p.y + pad), tok().TEXT, a.title.c_str());
    if (!a.key.empty()) {
        const ImVec2 ks = ImGui::CalcTextSize(a.key.c_str());
        const ImVec2 k0(p.x + width - pad - ks.x - 8, p.y + pad - 1);
        dl->AddRect(k0, ImVec2(k0.x + ks.x + 8, k0.y + ks.y + 2), tok().LINE, 2);
        dl->AddText(ImVec2(k0.x + 4, k0.y + 1), tok().DIM, a.key.c_str());
    }
    dl->AddText(ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(tx, p.y + pad + line + 4), tok().DIM, a.text.c_str(),
        nullptr, text_w);
    return pressed;
}

// A section's title, a dim subtitle and a hint at the right.
void StartPage::section_header(const char *title, const char *subtitle, const char *hint) {
    ImGui::TextUnformatted(title);
    ImGui::SameLine(0, 10);
    ImGui::TextDisabled("%s", subtitle);
    const float hw = ImGui::CalcTextSize(hint).x;
    const float right = ImGui::GetContentRegionMax().x;
    ImGui::SameLine(std::max(ImGui::GetCursorPosX() + 10, right - hw));
    ImGui::TextDisabled("%s", hint);
    ImGui::Dummy(ImVec2(1, 2));
}

void StartPage::status_box(const StartStatus &s, float width, StartRequest *req) {
    const StatusBlock &b = s.block;
    if (!ImGui::BeginChild("##status", ImVec2(width, 0), ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY)) {
        ImGui::EndChild();
        return;
    }
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float line = ImGui::GetTextLineHeight();
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + 4, p.y + line / 2), 3.5f, health_color(b.health));
    ImGui::Dummy(ImVec2(12, line));
    ImGui::SameLine(0, 2);
    ImGui::TextUnformatted(b.title.c_str());
    if (ImGui::BeginTable("##lines", 2, ImGuiTableFlags_SizingFixedFit)) {
        ImGui::TableSetupColumn("label", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch);
        for (const StatusLine &l : b.lines) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            cell_text(l.label, tok().DIM);
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, l.tone == Health::NONE ? tok().TEXT : health_color(l.tone));
            ImGui::TextWrapped("%s", l.value.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::EndTable();
    }
    if (!b.fix.empty()) {
        ImGui::TextDisabled("fix:");
        ImGui::SameLine(0, 6);
        ImGui::TextUnformatted(b.fix.c_str());
    }
    if (s.settings_tab >= 0) {
        if (ImGui::SmallButton(s.settings_label.c_str())) {
            req->kind = StartRequest::Kind::SETTINGS;
            req->settings_tab = s.settings_tab;
        }
    }
    ImGui::EndChild();
}

void StartPage::empty_note(const char *text) {
    ImGui::Indent(6);
    note(text);
    ImGui::Unindent(6);
    ImGui::Dummy(ImVec2(1, 4));
}

}  // namespace rmwork
