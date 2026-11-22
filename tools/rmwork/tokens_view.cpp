#include "tokens_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

static std::string short_number(double v);

static std::string short_number(double v) {
    char buf[32];
    if (v == std::floor(v) && std::fabs(v) < 1e6)
        std::snprintf(buf, sizeof buf, "%.0f", v);
    else
        std::snprintf(buf, sizeof buf, "%.2f", v);
    return buf;
}

namespace rmwork {

// The decision featurized (the attribution's tensors when it is of this decision and seat); false
// with `why` when it cannot be.
bool TokensView::refresh(const TokensContext &c, std::string *why) {
    const DecisionFrame &f = c.data->frame();
    if (c.att && c.att->tensors() && c.att->units() && c.att->spec()) {
        const std::string key = "att/" + std::to_string(f.index) + "/" + std::to_string(c.seat) + "/" +
                                std::to_string(reinterpret_cast<uintptr_t>(c.att->tensors())) + "/" +
                                std::to_string(c.att->tensors()->n());
        if (key != key_) {
            key_ = key;
            tensors_ = *c.att->tensors();
            list_ = *c.att->units();
            slots_ = c.att->slots();
            features_ = c.features;
        }
        return true;
    }
    if (!c.features) {
        *why =
            "The featurizer's inputs come with the first evaluation through the inference server (Attribution's "
            "Explain, or a re-evaluation): none has run yet.";
        return false;
    }
    const std::string key = "own/" + std::to_string(c.data->generation()) + "/" + std::to_string(c.seat) + "/" +
                            std::to_string(reinterpret_cast<uintptr_t>(c.features.get()));
    if (key == key_) return true;
    std::string err;
    const auto frame = f.match->frame(f.index, c.seat, &err);
    std::vector<featurize::TokenSource> sources;
    if (!frame ||
        !featurize::featurize_traced(
            c.features->spec, c.features->tables, *frame, nullptr, &tensors_, &sources, &err) ||
        !featurize::list_units(c.features->spec, c.features->tables, f.match->stream(c.seat).schema(), *frame, tensors_,
            sources, &list_, &err)) {
        *why = "The frame does not featurize: " + err;
        key_.clear();
        return false;
    }
    features_ = c.features;
    slots_ = featurize::type_slots(c.features->spec);
    key_ = key;
    return true;
}

std::string TokensView::cell_text(const featurize::SlotDef &s, size_t tok) const {
    const size_t F = tensors_.n() ? tensors_.tok_num.size() / tensors_.n() : 0,
                 C = tensors_.n() ? tensors_.tok_cat.size() / tensors_.n() : 0,
                 K = tensors_.n() ? tensors_.tok_card.size() / tensors_.n() : 0,
                 Q = tensors_.n() ? tensors_.tok_ability.size() / tensors_.n() : 0;
    std::string out;
    const auto add = [&out](const std::string &part) { out += (out.empty() ? "" : " ") + part; };
    for (size_t i = 0; i < s.card.size(); ++i) {
        const int64_t v = tensors_.tok_card[tok * K + s.card[i]];
        if (i > 0 && (v < 0 || v == tensors_.tok_card[tok * K + s.card[0]])) continue;
        const std::vector<std::string> *names = features_ ? &features_->tables.names : nullptr;
        if (v < 0)
            add("\xE2\x80\x94");
        else if (names && static_cast<size_t>(v) < names->size() && !(*names)[static_cast<size_t>(v)].empty())
            add((*names)[static_cast<size_t>(v)]);
        else
            add("#" + std::to_string(v));
    }
    for (const uint32_t c : s.ability) {
        const int64_t v = tensors_.tok_ability[tok * Q + c];
        add(v < 0 ? "\xE2\x80\x94" : "ab" + std::to_string(v));
    }
    for (size_t i = 0; i < s.cat.size(); ++i) {
        const int64_t v = tensors_.tok_cat[tok * C + s.cat[i]] - s.cat_none[i];
        add(v <= 0 ? "\xE2\x80\x94" : std::to_string(v - 1));
    }
    if (s.kind == "mask") {
        int set = 0;
        for (const uint32_t c : s.num) set += tensors_.tok_num[tok * F + c] != 0.0f ? 1 : 0;
        add(std::to_string(set) + " set");
    } else if (s.card.empty()) {
        for (size_t i = 0; i < s.num.size(); ++i)
            if (!s.num_flag[i] || tensors_.tok_num[tok * F + s.num[i]] != 0.0f)
                add((s.num_flag[i] ? "none" : short_number(tensors_.tok_num[tok * F + s.num[i]])));
    }
    return out;
}

void TokensView::draw(const TokensContext &c) {
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    heat_cells_ = 0;
    if (!c.data || !c.data->ok()) {
        centered_text_block("No decision", "Open a replay or start a game.", width);
        return;
    }
    if (c.seat < 0) {
        centered_text_block("No seat frame", "The featurizer reads a seat's frame.", width);
        return;
    }
    std::string why;
    if (!refresh(c, &why)) {
        centered_text_block("Tokens not read", why.c_str(), width);
        return;
    }
    const featurize::FeatureSpec *spec = c.att && c.att->tensors() ? c.att->spec()
                                         : features_               ? &features_->spec
                                                                   : nullptr;
    if (!spec) return;
    // The types present, with their token counts.
    std::vector<size_t> count(spec->types.size(), 0);
    for (const int64_t t : tensors_.tok_type) ++count[static_cast<size_t>(t)];
    for (size_t t = 0; !force_type_.empty() && t < count.size(); ++t)
        if (count[t] && spec->types[t].name == force_type_) type_ = static_cast<int>(t);
    if (type_ < 0 || static_cast<size_t>(type_) >= count.size() || !count[static_cast<size_t>(type_)]) {
        type_ = -1;
        for (size_t t = 0; t < count.size(); ++t)
            if (count[t] && (type_ < 0 || count[t] > count[static_cast<size_t>(type_)])) type_ = static_cast<int>(t);
    }
    if (type_ < 0) return;
    const float em = ImGui::CalcTextSize("M").x;
    ImGui::SetNextItemWidth(em * 14);
    const std::string current =
        spec->types[static_cast<size_t>(type_)].name + " (" + std::to_string(count[static_cast<size_t>(type_)]) + ")";
    if (ImGui::BeginCombo("##type", current.c_str())) {
        for (size_t t = 0; t < count.size(); ++t) {
            if (!count[t]) continue;
            const std::string name = spec->types[t].name + " (" + std::to_string(count[t]) + ")";
            if (ImGui::Selectable(name.c_str(), static_cast<int>(t) == type_)) type_ = static_cast<int>(t);
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    const bool heat = c.att && c.tints && c.tints->on;
    ImGui::TextDisabled("%zu tokens \xC2\xB7 %zu relations \xC2\xB7 %zu menu rows%s", tensors_.n(), tensors_.r(),
        tensors_.a(), heat ? (" \xC2\xB7 heat: " + c.tints->legend).c_str() : "");
    const std::vector<featurize::SlotDef> &slots = slots_[static_cast<size_t>(type_)];
    const int columns = 1 + static_cast<int>(slots.size());
    if (columns > 512) return;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit |
                                  ImGuiTableFlags_Resizable;
    if (!ImGui::BeginTable("##tokens", columns, flags)) return;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("token", ImGuiTableColumnFlags_WidthFixed, em * 16);
    for (const featurize::SlotDef &s : slots)
        ImGui::TableSetupColumn(s.column.c_str(), ImGuiTableColumnFlags_WidthFixed);
    ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
    for (int col = 0; col < columns; ++col) {
        ImGui::TableSetColumnIndex(col);
        ImGui::TableHeader(ImGui::TableGetColumnName(col));
        if (col > 0 && ImGui::IsItemHovered())
            ImGui::SetTooltip("%s", slots[static_cast<size_t>(col - 1)].label.c_str());
    }
    // The type's tokens, those with heat first (by their largest |value|), else in token order.
    std::vector<std::pair<double, size_t>> order;
    for (size_t i = 0; i < tensors_.n(); ++i) {
        if (tensors_.tok_type[i] != type_) continue;
        double most = -1;
        for (size_t s = 0; heat && s < slots.size(); ++s) {
            auto it = c.tints->token_cells.find({static_cast<int64_t>(i), static_cast<int>(s)});
            if (it != c.tints->token_cells.end()) most = std::max(most, std::fabs(it->second));
        }
        order.push_back({most, i});
    }
    std::stable_sort(order.begin(), order.end(),
        [](const std::pair<double, size_t> &a, const std::pair<double, size_t> &b) { return a.first > b.first; });
    for (const std::pair<double, size_t> &o : order) {
        const size_t i = o.second;
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        const std::string &label = i < list_.tokens.size() ? list_.tokens[i].label : std::string();
        ImGui::TextUnformatted(label.c_str());
        if (ImGui::IsItemHovered() && i < list_.tokens.size()) ImGui::SetTooltip("%s", list_.tokens[i].tensors.c_str());
        for (size_t s = 0; s < slots.size(); ++s) {
            ImGui::TableNextColumn();
            if (heat) {
                auto it = c.tints->token_cells.find({static_cast<int64_t>(i), static_cast<int>(s)});
                if (it != c.tints->token_cells.end()) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                        heat_color(it->second, c.tints->cell_scale, attr_target_signed(c.tints->target), 220));
                    ++heat_cells_;
                }
            }
            ImGui::TextUnformatted(cell_text(slots[s], i).c_str());
        }
    }
    ImGui::EndTable();
}

}  // namespace rmwork
