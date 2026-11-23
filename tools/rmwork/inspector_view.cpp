#include "inspector_view.h"

#include <cctype>
#include <cmath>
#include <cstdio>

#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

static std::string number(double v, int decimals);
static std::string humanize(const std::string &s);
static std::string object_name(const rmwork::DecisionFrame &f, uint64_t handle, int card);
static std::string pending_text(const rmwork::DecisionFrame &f);
static std::string pip_text(const agents::SeatStream &s, const agents::Table &pips, size_t cost_row);
static std::string pending_cost_text(const rmwork::DecisionFrame &f);
static std::string game_text(const rmwork::DecisionFrame &f);
static std::string turn_text(const rmwork::DecisionFrame &f);
static void episode_fields(const rmwork::ToolData &data, std::vector<rmwork::InspectorField> *out);

static std::string number(double v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
    return buf;
}

// RESOLUTION -> "Resolution".
static std::string humanize(const std::string &s) {
    std::string out;
    for (char c : s) out += c == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

// The name the frame gives an object (by its handle in the object table, else the row's card
// column `card`); "" when it gives none.
static std::string object_name(const rmwork::DecisionFrame &f, uint64_t handle, int card) {
    const agents::Table o(*f.frame, f.stream->schema(), "object");
    const int c_handle = o.col("handle"), c_card = o.col("card");
    for (size_t r = 0; handle && r < o.rows(); ++r)
        if (o.u(r, c_handle) == handle) {
            const int id = o.card(r, c_card);
            if (id >= 0) return f.stream->cards().name(id);
            break;
        }
    return card >= 0 ? f.stream->cards().name(card) : "";
}

// "Resolution · Brainstorm (Player A chooses)", "—".
static std::string pending_text(const rmwork::DecisionFrame &f) {
    const agents::Table p(*f.frame, f.stream->schema(), "pending");
    if (!p.rows()) return "\xE2\x80\x94";
    std::string out = humanize(f.stream->enum_name("pending_tag", p.u(0, p.col("tag"))));
    const std::string src = object_name(f, p.u(0, p.col("source")), p.card(0, p.col("source_card")));
    if (!src.empty()) out += " \xC2\xB7 " + src;
    const uint64_t chooser = p.u(0, p.col("chooser"));
    if (chooser == 1 || chooser == 2) out += std::string(" (") + rmwork::seat_name(chooser == 1 ? 0 : 1) + " chooses)";
    return out;
}

// The pending cost's non-plain pips of `cost_row`: "{W/U}", "{2/R}", "{B/P}", "{S}".
static std::string pip_text(const agents::SeatStream &s, const agents::Table &pips, size_t cost_row) {
    std::string out;
    for (size_t r = 0; r < pips.rows(); ++r) {
        if (pips.u(r, pips.col("cost")) != cost_row) continue;
        const std::string kind = s.enum_name("pip_kind", pips.u(r, pips.col("kind")));
        const uint64_t colors = pips.u(r, pips.col("colors"));
        const std::vector<std::string> &names = s.enum_values("color");
        std::string cs;
        for (size_t i = 0; i < names.size() && i < 8; ++i) {
            if (!(colors & (1ull << i))) continue;
            const std::string &n = names[i];
            const char *letter = n == "WHITE"   ? "W"
                                 : n == "BLUE"  ? "U"
                                 : n == "BLACK" ? "B"
                                 : n == "RED"   ? "R"
                                 : n == "GREEN" ? "G"
                                                : nullptr;
            cs += (cs.empty() ? "" : "/") + (letter ? std::string(letter) : n.substr(0, 1));
        }
        if (kind == "PHYREXIAN") {
            out += "{" + cs + "/P}";
        } else if (kind == "TWOBRID") {
            out += "{" + std::to_string(pips.u(r, pips.col("generic_alt"))) + "/" + cs + "}";
        } else if (kind == "SNOW") {
            out += "{S}";
        } else {
            out += "{" + cs + "}";
        }
    }
    return out;
}

// "{1}{U} · 2 life for Counterspell (mana, 1 paid)", "" when no cost is being paid.
static std::string pending_cost_text(const rmwork::DecisionFrame &f) {
    const agents::Table c(*f.frame, f.stream->schema(), "pending_cost");
    if (!c.rows()) return "";
    const agents::Table pips(*f.frame, f.stream->schema(), "pending_cost_pip");
    std::string out;
    for (size_t r = 0; r < c.rows(); ++r) {
        std::string mana;
        for (uint64_t i = 0; i < c.u(r, c.col("rem_x")); ++i) mana += "{X}";
        if (const uint64_t g = c.u(r, c.col("rem_generic"))) mana += "{" + std::to_string(g) + "}";
        static const char *const COLS[] = {"rem_w", "rem_u", "rem_b", "rem_r", "rem_g", "rem_c"};
        static const char *const SYMS[] = {"{W}", "{U}", "{B}", "{R}", "{G}", "{C}"};
        for (size_t k = 0; k < 6; ++k)
            for (uint64_t i = 0; i < c.u(r, c.col(COLS[k])); ++i) mana += SYMS[k];
        mana += pip_text(*f.stream, pips, r);
        std::vector<std::string> parts;
        if (!mana.empty()) parts.push_back(mana);
        if (const uint64_t l = c.u(r, c.col("rem_life"))) parts.push_back(std::to_string(l) + " life");
        if (const uint64_t e = c.u(r, c.col("rem_energy"))) parts.push_back(std::to_string(e) + " energy");
        if (const uint64_t d = c.u(r, c.col("rem_discard"))) parts.push_back("discard " + std::to_string(d));
        std::string line;
        for (const std::string &p : parts) line += (line.empty() ? "" : " \xC2\xB7 ") + p;
        if (line.empty()) line = "nothing left";
        const std::string what = object_name(f, c.u(r, c.col("for")), c.card(r, c.col("for_card")));
        if (!what.empty()) line += " for " + what;
        line += " (" + humanize(f.stream->enum_name("cost_part", c.u(r, c.col("part"))));
        if (const uint64_t paid = c.u(r, c.col("paid"))) line += ", " + std::to_string(paid) + " paid";
        line += ")";
        out += (out.empty() ? "" : "; ") + line;
    }
    return out;
}

// "Game 1", "Sideboarding 2".
static std::string game_text(const rmwork::DecisionFrame &f) {
    const int span = f.match->span_of(f.index);
    return span >= 0 ? f.match->games[static_cast<size_t>(span)].label : "Game " + std::to_string(f.info->game + 1);
}

// "3 · Player A's turn", "Pregame", "Sideboarding".
static std::string turn_text(const rmwork::DecisionFrame &f) {
    if (f.info->sideboarding) return "Sideboarding";
    if (f.info->phase == "Opening hand") return "Pregame";
    const agents::Table t(*f.frame, f.stream->schema(), "turn");
    std::string out = std::to_string(f.info->turn + 1);
    if (t.rows()) {
        const uint64_t a = t.u(0, t.col("active"));
        if (a == 1 || a == 2) out += std::string(" \xC2\xB7 ") + rmwork::seat_name(a == 1 ? 0 : 1) + "'s turn";
    }
    return out;
}

static void episode_fields(const rmwork::ToolData &data, std::vector<rmwork::InspectorField> *out) {
    using rmwork::InspectorField;
    const rmwork::DecisionFrame &f = data.frame();
    const rmwork::EpisodeStepInfo *e = f.visible_episode();
    if (!e) {
        out->push_back({"Evaluations",
            f.sees_decision() ? "no episode for this decision"
                              : std::string("not shown in the ") + rmwork::view_name(f.view) + " view",
            false, 0});
        return;
    }
    for (const featurize::PolicyEvalRecord &r : e->policies) {
        const double v = data.scale_a ? rmwork::on_a_scale(e->seat, r.value) : r.value;
        out->push_back({"V \xC2\xB7 " + rmwork::policy_label(r.id, r.version),
            rmwork::signed_text(v, 3) + (data.scale_a ? " (Player A's scale)" : " (own seat)"), true, v});
    }
    if (e->policies.empty()) out->push_back({"V", "not evaluated", false, 0});
    const size_t n = data.models().menu.rows.size();
    if (const featurize::PolicyEvalRecord *r = rmwork::find_policy(*e, data.shown_policy())) {
        if (r->log_pi.size() == n) {
            out->push_back({"H(\xCF\x80) \xC2\xB7 " + rmwork::policy_name(r->id),
                number(rmwork::entropy(rmwork::policy_pi(*r)), 3) + " nats", false, 0});
            if (e->chosen >= 0 && static_cast<size_t>(e->chosen) < n)
                out->push_back(
                    {"log \xCF\x80(chosen)", number(r->log_pi[static_cast<size_t>(e->chosen)], 3), false, 0});
        }
    }
    // A step made for re-evaluated records alone has nothing recorded.
    if (e->file == rmwork::REEVAL_STEP_FILE) return;
    if (e->has_mu && e->mu.size() == n) {
        size_t support = 0, at = 0;
        for (size_t i = 0; i < n; ++i)
            if (e->mu[i] > 0) {
                ++support;
                at = i;
            }
        out->push_back({"\xCE\xBC",
            support == 1 ? "one-hot (row " + std::to_string(at) + ")"
                         : "support " + std::to_string(support) + " of " + std::to_string(n) + " rows",
            false, 0});
    } else {
        out->push_back({"\xCE\xBC", "absent", false, 0});
    }
    out->push_back(
        {"Policy version", e->policy_version >= 0 ? std::to_string(e->policy_version) : "\xE2\x80\x94", false, 0});
    std::string teacher = "\xE2\x80\x94";
    if (e->teacher >= 0)
        teacher = "row " + std::to_string(e->teacher) +
                  (e->teacher == e->chosen ? " = chosen" : " \xE2\x89\xA0 chosen row " + std::to_string(e->chosen));
    out->push_back({"Teacher", teacher, false, 0});
}

namespace rmwork {

std::vector<InspectorField> inspector_fields(const ToolData &data) {
    std::vector<InspectorField> out;
    if (!data.ok()) return out;
    const DecisionFrame &f = data.frame();
    const MenuModel &menu = data.models().menu;
    out.push_back(
        {"Decision", std::to_string(f.index + 1) + " of " + std::to_string(f.match->decisions.size()), false, 0});
    out.push_back({"Game", game_text(f), false, 0});
    out.push_back({"Turn", turn_text(f), false, 0});
    out.push_back({"Step", f.info->phase, false, 0});
    const std::string agent = agent_label(*f.match, f.info->decider);
    out.push_back({"Deciding seat",
        std::string(seat_name(f.info->decider)) + (agent.empty() ? "" : " \xC2\xB7 " + agent), false, 0});
    out.push_back({"Pending", pending_text(f), false, 0});
    const std::string cost = pending_cost_text(f);
    if (!cost.empty()) out.push_back({"Pending cost", cost, false, 0});
    const size_t stack = data.models().table.stack.size();
    out.push_back({"Stack", stack ? std::to_string(stack) + (stack == 1 ? " object" : " objects") : "empty", false, 0});
    out.push_back({"Menu rows",
        menu.shown ? std::to_string(menu.rows.size())
                   : std::string("\xE2\x80\x94 (") + seat_name(menu.decider) + " decides)",
        false, 0});
    episode_fields(data, &out);
    return out;
}

void InspectorView::draw(ToolData &data, LinkState *links) {
    if (!data.ok()) {
        centered_text_block("Frame not readable", data.error().c_str(), 46.0f * ImGui::CalcTextSize("M").x);
        return;
    }
    const std::vector<InspectorField> fields = inspector_fields(data);
    if (begin_grid("##inspector", 2)) {
        ImGui::TableSetupColumn(
            "Name", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("log \xCF\x80(chosen)  ").x);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (const InspectorField &fl : fields) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(fl.name.c_str());
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, fl.has_sign ? signed_color(fl.sign) : tok().TEXT);
            ImGui::TextWrapped("%s", fl.value.c_str());
            ImGui::PopStyleColor();
        }
        ImGui::EndTable();
    }
    const DecisionFrame &f = data.frame();
    const EpisodeStepInfo *e = f.visible_episode();
    ImGui::Spacing();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("Value scale");
    ImGui::SameLine(0, 10);
    if (ImGui::RadioButton("Deciding seat", !data.scale_a)) data.scale_a = false;
    ImGui::SameLine(0, 12);
    if (ImGui::RadioButton("Player A", data.scale_a)) data.scale_a = true;
    if (data.values().policies.size() > 1) {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("\xCF\x80 shown");
        for (int64_t p : data.values().policies) {
            ImGui::SameLine(0, 12);
            ImGui::PushID(std::to_string(p).c_str());
            const std::string label = policy_label(p, data.values().versions.at(p));
            if (ImGui::RadioButton(label.c_str(), data.shown_policy() == p)) data.policy = p;
            ImGui::PopID();
        }
    }
    const MenuModel &menu = data.models().menu;
    if (!menu.shown) return;
    std::vector<const featurize::PolicyEvalRecord *> recs;
    if (e)
        for (const featurize::PolicyEvalRecord &r : e->policies)
            if (r.log_pi.size() == menu.rows.size()) recs.push_back(&r);
    const bool mu = e && e->has_mu && e->mu.size() == menu.rows.size();
    ImGui::Spacing();
    const int columns = 2 + static_cast<int>(recs.size()) + (mu ? 1 : 0);
    if (begin_grid("##inspector_menu", columns, ImVec2(0, std::max(ImGui::GetContentRegionAvail().y, 80.0f)), true)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("\xE2\x96\xBA 00").x);
        ImGui::TableSetupColumn("Menu row", ImGuiTableColumnFlags_WidthStretch);
        std::vector<std::vector<double>> pis;
        for (const featurize::PolicyEvalRecord *r : recs) {
            const std::string h = "\xCF\x80 " + policy_label(r->id, r->version);
            ImGui::TableSetupColumn(h.c_str(), ImGuiTableColumnFlags_WidthFixed, 60 + ImGui::CalcTextSize(" 0.00").x);
            pis.push_back(policy_pi(*r));
        }
        if (mu)
            ImGui::TableSetupColumn("\xCE\xBC", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x + 4);
        ImGui::TableHeadersRow();
        for (size_t r = 0; r < menu.rows.size(); ++r) {
            const int row = static_cast<int>(r);
            ImGui::TableNextRow();
            if (row == menu.chosen) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, tok().SEL);
            } else if (links && links->row_linked(row)) {
                ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, linked_row_color());
            }
            ImGui::TableNextColumn();
            ImGui::PushID(row);
            ImGui::Selectable("##r", false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap);
            if (ImGui::IsItemHovered() && links) links->hover_menu_row(row);
            ImGui::SameLine(0, 0);
            ImGui::Text("%s%d", row == menu.chosen ? "\xE2\x96\xBA " : "  ", row);
            ImGui::TableNextColumn();
            ImGui::TextWrapped("%s", menu.rows[r].text.c_str());
            for (const std::vector<double> &pi : pis) {
                ImGui::TableNextColumn();
                value_bar(static_cast<float>(pi[r]), 56, tok().ACCENT);
                ImGui::SameLine(0, 4);
                ImGui::TextUnformatted(number(pi[r], 2).c_str());
            }
            if (mu) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(number(e->mu[r], 3).c_str());
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

}  // namespace rmwork
