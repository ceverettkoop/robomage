#include "model_manager_view.h"

#include <unistd.h>

#include <algorithm>
#include <cstdio>

#include "imgui.h"
#include "modal.h"
#include "paths.h"
#include "theme.h"

namespace {

const char *const DOT = " \xC2\xB7 ";
const char *const ALL_RUNS = "All runs";
// How often the rows' compatibility is asked for again while the tool is open (an engine rebuilt).
const double REQUEST_EVERY_S = 1.0;

}  // namespace

static ImU32 compat_color(const rmwork::CompatVerdict &v);
static void cell_text(const std::string &text, ImU32 color);

static ImU32 compat_color(const rmwork::CompatVerdict &v) {
    if (v.checking || v.compat == rmwork::PolicyCompat::UNKNOWN) return rmwork::tok().DIM;
    return v.compat == rmwork::PolicyCompat::COMPATIBLE ? rmwork::tok().GOOD : rmwork::tok().BAD;
}

static void cell_text(const std::string &text, ImU32 color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

namespace rmwork {

void ModelManagerView::set_open(bool open) {
    if (open && !open_) {
        scan_ = true;
        ++generation_;
        column_ = ModelColumn::DATE;
        ascending_ = false;
    }
    open_ = open;
}

void ModelManagerView::rescan(const Settings &s, bool recheck) {
    rows_ = scan_policies(run_dir_path(s), default_models_dir());
    runs_ = model_runs(rows_);
    if (!filter_.run.empty() && std::find(runs_.begin(), runs_.end(), filter_.run) == runs_.end()) filter_.run.clear();
    if (recheck) checker_.forget_files();
    requested_at_ = -1;
}

bool ModelManagerView::select_path(const std::string &path) {
    for (size_t i : shown_)
        if (rows_[i].play == path || (rows_[i].play.empty() && rows_[i].checkpoint == path)) {
            selected_ = row_key(rows_[i]);
            return true;
        }
    return false;
}

const PolicyRow *ModelManagerView::selected() const {
    if (selected_.empty()) return nullptr;
    for (const PolicyRow &r : rows_)
        if (row_key(r) == selected_) return &r;
    return nullptr;
}

std::string ModelManagerView::row_key(const PolicyRow &r) {
    if (r.latest) return "latest";
    return r.archive + "|" + r.run + "/" + r.name;
}

// What Copy path copies: the export to play, else the first file.
std::string ModelManagerView::row_path(const PolicyRow &r) {
    for (const std::string *p : {&r.play, &r.ts, &r.pt2, &r.onnx, &r.checkpoint})
        if (!p->empty()) return *p;
    return "";
}

std::string ModelManagerView::row_tooltip(const PolicyRow &r, const CompatVerdict &v) {
    std::string out = r.label;
    const std::pair<const char *, const std::string *> files[] = {{"plays", &r.play}, {"ts", &r.ts},
        {"pt2", &r.pt2}, {"onnx", &r.onnx}, {"checkpoint", &r.checkpoint}, {"tables", &r.tables_manifest}};
    for (const auto &f : files)
        if (!f.second->empty()) out += std::string("\n") + f.first + ": " + *f.second;
    if (r.play.empty()) out += "\nNo export to play: only a checkpoint (rmtrain exports it)";
    out += std::string("\nCompatibility: ") + compat_verdict_text(v) + (v.why.empty() ? "" : " (" + v.why + ")");
    if (v.compat == PolicyCompat::INCOMPATIBLE)
        out += "\nRebuild its exports on the engine's tables: " + retable_hint(r);
    return out;
}

ModelRequest ModelManagerView::draw(const ModelManagerContext &ctx) {
    ModelRequest req;
    if (!ctx.settings) return req;
    if (scan_) {
        scan_ = false;
        rescan(*ctx.settings, false);
    }
    tables_.engine = ctx.engine;
    tables_.workdir = engine_workdir_path(*ctx.settings);
    tables_.cards_path = workbench_cache_dir() + "/model_manager/cards-" + std::to_string(getpid()) + ".rmst";
    const double now = ImGui::GetTime();
    if (requested_at_ < 0 || now - requested_at_ >= REQUEST_EVERY_S) {
        requested_at_ = now;
        checker_.request(tables_, rows_);
    }
    checker_.poll();
    verdicts_.resize(rows_.size());
    for (size_t i = 0; i < rows_.size(); ++i) {
        verdicts_[i] = checker_.verdict(rows_[i]);
        rows_[i].compat = verdicts_[i].compat;
    }
    draw_server_strip(ctx, &req);
    ImGui::Separator();
    draw_filters(ctx);
    draw_actions(&req);
    draw_table(&req);
    return req;
}

// The server's state, the socket, Start / Restart / Stop and Settings → Models.
void ModelManagerView::draw_server_strip(const ModelManagerContext &ctx, ModelRequest *req) {
    const Settings &s = *ctx.settings;
    cell_text(ctx.server_state, ctx.server_bad ? tok().BAD : tok().TEXT);
    const std::string socket = socket_path(s);
    ImGui::SameLine(0, 0);
    cell_text(DOT + (socket.empty() ? std::string("no socket") : socket), tok().DIM);
    if (!ctx.server_error.empty()) ImGui::SetItemTooltip("%s", ctx.server_error.c_str());
    ImGui::SameLine();
    if (!s.models.start_server) {
        if (ImGui::SmallButton("Start server")) req->kind = ModelRequest::Kind::START_SERVER;
        ImGui::SetItemTooltip("Runs rmserve with Settings \xE2\x86\x92 Models' policies while the workbench is open");
    } else {
        if (ImGui::SmallButton("Restart server")) req->kind = ModelRequest::Kind::RESTART_SERVER;
        ImGui::SameLine();
        if (ImGui::SmallButton("Stop server")) req->kind = ModelRequest::Kind::STOP_SERVER;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("Settings \xE2\x86\x92 Models\xE2\x80\xA6")) req->kind = ModelRequest::Kind::SETTINGS;
    if (!ctx.server_error.empty()) colored_note(ctx.server_error.c_str(), tok().BAD);
}

// The filter, the run, Show all, Refresh and the engine tables' state.
void ModelManagerView::draw_filters(const ModelManagerContext &ctx) {
    const float cw = char_width();
    if (filter_.text != filter_text_) std::snprintf(filter_text_, sizeof filter_text_, "%s", filter_.text.c_str());
    ImGui::SetNextItemWidth(22 * cw);
    if (ImGui::InputTextWithHint("##filter", "Filter: run or name", filter_text_, sizeof filter_text_))
        filter_.text = filter_text_;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(20 * cw);
    if (ImGui::BeginCombo("##run", filter_.run.empty() ? ALL_RUNS : filter_.run.c_str(), ImGuiComboFlags_HeightLarge)) {
        if (ImGui::Selectable(ALL_RUNS, filter_.run.empty())) filter_.run.clear();
        for (const std::string &r : runs_)
            if (ImGui::Selectable(r.c_str(), r == filter_.run)) filter_.run = r;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Checkbox("Show all", &filter_.show_all);
    ImGui::SetItemTooltip("Also the periodic checkpoints, the seed and the live export");
    ImGui::SameLine();
    if (ImGui::SmallButton("Refresh")) rescan(*ctx.settings, true);
    ImGui::SameLine();
    if (checker_.engine_ready()) {
        cell_text("Engine tables read", tok().DIM);
    } else if (!checker_.engine_error().empty()) {
        cell_text("Engine tables: unknown", tok().BAD);
        ImGui::SetItemTooltip("%s", checker_.engine_error().c_str());
    } else {
        cell_text("Reading the engine's tables\xE2\x80\xA6", tok().DIM);
    }
}

// Play against, Serve and Copy path on the selected row.
void ModelManagerView::draw_actions(ModelRequest *req) {
    const PolicyRow *sel = selected();
    const bool plays = sel && !sel->play.empty();
    ImGui::BeginDisabled(!plays);
    if (ImGui::Button("Play against")) *req = {ModelRequest::Kind::PLAY_AGAINST, sel->play};
    ImGui::SetItemTooltip("Game \xE2\x86\x92 New Game with this policy as the opponent's model");
    ImGui::SameLine();
    if (ImGui::Button("Serve")) *req = {ModelRequest::Kind::SERVE, sel->play};
    ImGui::SetItemTooltip("Adds it to Settings \xE2\x86\x92 Models' policies and starts the server");
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!sel);
    if (ImGui::Button("Copy path")) *req = {ModelRequest::Kind::COPY_PATH, row_path(*sel)};
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (!sel)
        cell_text("Select a policy", tok().DIM);
    else
        cell_text(sel->label + (plays ? "" : DOT + std::string("no export to play")), tok().DIM);
}

void ModelManagerView::draw_table(ModelRequest *req) {
    const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV |
                                  ImGuiTableFlags_BordersOuter | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable |
                                  ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_NoSavedSettings;
    const float cw = char_width();
    ImGui::PushID(generation_);
    if (!ImGui::BeginTable("##policies", MODEL_COLUMN_COUNT, flags)) {
        ImGui::PopID();
        return;
    }
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Date",
        ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_PreferSortDescending, 10 * cw, 0);
    ImGui::TableSetupColumn("Run", ImGuiTableColumnFlags_WidthFixed, 16 * cw, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 14 * cw, 2);
    ImGui::TableSetupColumn("Steps", ImGuiTableColumnFlags_PreferSortDescending, 7 * cw, 3);
    ImGui::TableSetupColumn("Formats", ImGuiTableColumnFlags_WidthFixed, 18 * cw, 4);
    ImGui::TableSetupColumn("Compatible", ImGuiTableColumnFlags_WidthStretch, 1.0f, 5);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs *specs = ImGui::TableGetSortSpecs()) {
        if (specs->SpecsCount > 0) {
            column_ = static_cast<ModelColumn>(specs->Specs[0].ColumnUserID);
            ascending_ = specs->Specs[0].SortDirection == ImGuiSortDirection_Ascending;
        }
        specs->SpecsDirty = false;
    }
    shown_ = model_view(rows_, filter_, column_, ascending_);
    for (size_t i : shown_) draw_row(i, req);
    if (shown_.empty()) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", rows_.empty() ? "No policies: no runs under Settings \xE2\x86\x92 Models' run "
                                                  "directory and no zipped exports in models/"
                                                : "No policy matches");
    }
    ImGui::EndTable();
    ImGui::PopID();
}

void ModelManagerView::draw_row(size_t index, ModelRequest *req) {
    const PolicyRow &r = rows_[index];
    const CompatVerdict &v = verdicts_[index];
    const bool grey = v.compat == PolicyCompat::INCOMPATIBLE;
    const ImU32 text = grey ? tok().DIM : tok().TEXT;
    ImGui::PushID(static_cast<int>(index));
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    const std::string key = row_key(r);
    ImGui::PushStyleColor(ImGuiCol_Text, text);
    const std::string date = r.latest ? "latest" : policy_date_text(r.trained_at);
    if (ImGui::Selectable(date.empty() ? "\xE2\x80\x94" : date.c_str(), key == selected_,
            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
        selected_ = key;
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && !r.play.empty())
            *req = {ModelRequest::Kind::PLAY_AGAINST, r.play};
    }
    ImGui::PopStyleColor();
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) ImGui::SetTooltip("%s", row_tooltip(r, v).c_str());
    ImGui::TableNextColumn();
    // A zipped export: its run (when its path names one) and its archive.
    cell_text(r.archive.empty() ? r.run : r.run.empty() ? r.archive : r.run + DOT + r.archive, text);
    ImGui::TableNextColumn();
    cell_text(r.latest ? r.name + DOT + policy_date_text(r.trained_at) : r.name, text);
    ImGui::TableNextColumn();
    cell_text(r.learner_steps < 0 ? "" : std::to_string(r.learner_steps), text);
    ImGui::TableNextColumn();
    // A row with no export to play says so in place of its formats.
    if (r.play.empty())
        cell_text("checkpoint only", tok().DIM);
    else
        cell_text(policy_formats_text(r.formats), text);
    ImGui::TableNextColumn();
    cell_text(compat_verdict_text(v), compat_color(v));
    if (v.compat == PolicyCompat::INCOMPATIBLE) {
        ImGui::SameLine();
        cell_text(retable_hint(r), tok().DIM);
    }
    ImGui::PopID();
}

}  // namespace rmwork
