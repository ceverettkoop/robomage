#include "reevaluate_dialog.h"

#include <algorithm>

#include "modal.h"
#include "theme.h"

namespace {

const char *const POPUP = "Re-evaluate the Replay";

}  // namespace

static std::string file_name(const std::string &path);

static std::string file_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

namespace rmwork {

void ReevaluateDialog::open(const std::vector<ServedPolicy> &policies, bool on_open) {
    policies_ = policies;
    ticked_.assign(policies_.size(), true);
    for (size_t i = 0; i < policies_.size(); ++i)
        if (std::find(unticked_.begin(), unticked_.end(), policies_[i].id) != unticked_.end()) ticked_[i] = false;
    on_open_ = on_open;
    want_open_ = true;
}

std::vector<ServedPolicy> ReevaluateDialog::chosen() const {
    std::vector<ServedPolicy> out;
    for (size_t i = 0; i < policies_.size(); ++i)
        if (ticked_[i]) out.push_back(policies_[i]);
    return out;
}

ReevaluateDialog::Choice ReevaluateDialog::draw(ReevalRequest *out, bool *on_open) {
    if (want_open_) {
        ImGui::OpenPopup(POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(560, 0));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!open_) return Choice::NONE;
    Choice choice = Choice::NONE;
    note(
        "Each policy evaluates each seat's decisions in that seat's own frame through the inference server, in "
        "batches; the values join the recorded ones as series of their own (\"re p0\"). Results are cached, so "
        "reopening this replay costs nothing.");
    ImGui::Spacing();
    if (policies_.empty()) {
        colored_note("No served policy: add one with a numeric id in Settings \xE2\x86\x92 Models.", tok().BAD);
    } else {
        ImGui::TextDisabled("Policies served");
        for (size_t i = 0; i < policies_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            bool on = ticked_[i];
            const std::string label =
                "p" + std::to_string(policies_[i].id) + "  " + file_name(policies_[i].model) + "##tick";
            if (ImGui::Checkbox(label.c_str(), &on)) {
                ticked_[i] = on;
                const uint32_t id = policies_[i].id;
                unticked_.erase(std::remove(unticked_.begin(), unticked_.end(), id), unticked_.end());
                if (!on) unticked_.push_back(id);
            }
            ImGui::SetItemTooltip("%s", policies_[i].model.c_str());
            ImGui::PopID();
        }
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Seats");
    ImGui::SameLine(0, 12);
    ImGui::Checkbox("A", &seats_[0]);
    ImGui::SameLine(0, 12);
    ImGui::Checkbox("B", &seats_[1]);
    ImGui::Spacing();
    ImGui::Checkbox("Re-evaluate a replay when it opens", &on_open_);
    ImGui::SetItemTooltip(
        "With the served policies, both seats: cached results at once; the rest while the server runs");
    ImGui::Separator();
    const std::vector<ServedPolicy> pols = chosen();
    const bool can = !pols.empty() && (seats_[0] || seats_[1]);
    bool close = !keep || close_requested();
    ImGui::BeginDisabled(!can);
    if (ImGui::Button("Re-evaluate") || (can && ImGui::IsKeyPressed(ImGuiKey_Enter, false))) {
        out->policies = pols;
        out->seats = seats_;
        choice = Choice::REEVALUATE;
        close = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Settings \xE2\x86\x92 Models\xE2\x80\xA6")) {
        choice = Choice::SETTINGS;
        close = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) close = true;
    *on_open = on_open_;
    if (close) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return choice;
}

}  // namespace rmwork
