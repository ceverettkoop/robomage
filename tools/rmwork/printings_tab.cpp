#include "printings_tab.h"

#include <algorithm>
#include <cctype>

#include "files.h"
#include "modal.h"
#include "paths.h"
#include "theme.h"

namespace {

// The thumbnail's width in the printings list.
const float THUMB_W = 34.0f;
const float CARD_RATIO = 1.3934f;

}  // namespace

static std::string lower(const std::string &s);
static std::string year_of(const std::string &date);
static std::string last_line(const std::string &text);

static std::string lower(const std::string &s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

static std::string year_of(const std::string &date) {
    return date.size() >= 4 ? date.substr(0, 4) : date;
}

// The last non-empty line of `text`.
static std::string last_line(const std::string &text) {
    size_t end = text.size();
    while (end > 0 && (text[end - 1] == '\n' || text[end - 1] == ' ')) --end;
    const size_t start = text.rfind('\n', end == 0 ? 0 : end - 1);
    return text.substr(start == std::string::npos ? 0 : start + 1, end - (start == std::string::npos ? 0 : start + 1));
}

namespace rmwork {

bool PrintingsTab::matches(const ImageCard &card, const std::string &query) {
    if (query.empty()) return true;
    if (lower(card.display()).find(query) != std::string::npos) return true;
    for (const std::string &n : card.names)
        if (lower(n).find(query) != std::string::npos) return true;
    return false;
}

void PrintingsTab::draw(Settings *s, CardImages *images) {
    if (!images) {
        note("Card images are not available here.");
        return;
    }
    poll_fetch(*images);
    draw_status(s, *images);
    if (!images->has_manifest()) return;
    ImGui::Separator();
    ImGui::Spacing();
    if (!want_select_.empty()) {
        const std::string want = lower(want_select_);
        for (const ImageCard &c : images->manifest().cards())
            if (matches(c, want)) {
                selected_ = c.key;
                scroll_shown_ = true;
                break;
            }
        want_select_.clear();
    }
    const float h = std::max(120.0f, ImGui::GetContentRegionAvail().y);
    const float list_w = 24 * char_width();
    ImGui::BeginChild("##printing_cards", ImVec2(list_w, h), ImGuiChildFlags_Borders);
    draw_list(*images);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild("##printing_card", ImVec2(0, h));
    const ImageCard *card = images->manifest().by_key(selected_);
    if (card) {
        draw_card(*images, *card);
    } else {
        ImGui::TextDisabled("Choose a card on the left to see its printings.");
    }
    ImGui::EndChild();
}

// Faces, the cache directory and what is in it.
void PrintingsTab::draw_status(Settings *s, CardImages &images) {
    const float field_x = 11 * char_width();
    field_label("Card faces", field_x);
    if (ImGui::RadioButton("Images", s->card_images)) s->card_images = true;
    ImGui::SameLine();
    if (ImGui::RadioButton("Text", !s->card_images)) s->card_images = false;
    ImGui::SameLine();
    ImGui::TextDisabled("(a card with no cached image is drawn as text)");
    const ImageCacheStatus st = images.status();
    field_label("Cache", field_x);
    if (!st.manifest) {
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(tok().BAD), "%s",
            st.error.empty() ? "no manifest.json: every card is drawn as text" : st.error.c_str());
    } else {
        ImGui::Text("%d of %d vocab cards and %d of %d tokens have images", st.with_images, st.vocab, st.token_images,
            st.tokens);
        if (st.missing > 0) {
            ImGui::SameLine();
            ImGui::TextDisabled("· %d not on Scryfall", st.missing);
            if (ImGui::IsItemHovered()) {
                std::string tip;
                for (const auto &m : images.manifest().missing()) tip += m.first + ": " + m.second + "\n";
                ImGui::SetTooltip("%s", tip.c_str());
            }
        }
    }
    if (!st.manifest || st.with_images < st.vocab || st.token_images < st.tokens) {
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + field_x);
        const std::string cmd = fetch_all_command(images.cache_dir());
        ImGui::TextDisabled("Fetch the rest (network, a few minutes):");
        ImGui::SameLine();
        ImGui::TextUnformatted(cmd.c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Copy")) ImGui::SetClipboardText(cmd.c_str());
    }
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + field_x);
    if (ImGui::SmallButton("Reload the cache")) images.reload();
    if (!images.overrides_error().empty()) {
        ImGui::SameLine();
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(tok().BAD), "%s", images.overrides_error().c_str());
    }
}

// The cards and tokens of the manifest, filtered by the search.
void PrintingsTab::draw_list(CardImages &images) {
    ImGui::SetNextItemWidth(-1);
    ImGui::InputTextWithHint("##printing_search", "search a card", query_, sizeof query_);
    const std::string q = lower(query_);
    const PrintingOverrides &o = images.overrides();
    for (const ImageCard &c : images.manifest().cards()) {
        if (!matches(c, q)) continue;
        const bool overridden = o.count(c.key) > 0;
        const std::string label = c.display() + (overridden ? "  *" : "") + "##" + c.key;
        if (ImGui::Selectable(label.c_str(), selected_ == c.key) && selected_ != c.key) {
            selected_ = c.key;
            scroll_shown_ = true;
        }
        if (selected_ == c.key && ImGui::IsWindowAppearing()) ImGui::SetScrollHereY();
    }
}

// The card's printings: a row each, the shown one selected.
void PrintingsTab::draw_card(CardImages &images, const ImageCard &card) {
    bool overridden = false;
    const ImagePrinting *shown = shown_printing(card, images.overrides(), &overridden);
    ImGui::TextUnformatted(card.display().c_str());
    if (!card.scryfall_name.empty() && card.scryfall_name != card.display()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(%s)", card.scryfall_name.c_str());
    }
    const ImagePrinting &def = card.printings[static_cast<size_t>(card.chosen)];
    ImGui::TextDisabled("Default: %s (%s) #%s, %s", def.set_name.c_str(), def.set.c_str(), def.number.c_str(),
        year_of(def.released).c_str());
    ImGui::BeginDisabled(!overridden);
    if (ImGui::SmallButton("Default (oldest)")) {
        std::string err;
        message_error_ = !images.set_override(card.key, PrintingChoice(), &err);
        message_ = message_error_ ? err : card.display() + ": the default printing";
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(images.overrides().empty());
    if (ImGui::SmallButton("Reset All")) {
        std::string err;
        message_error_ = !images.clear_overrides(&err);
        message_ = message_error_ ? err : "Every card: the default printing";
    }
    ImGui::EndDisabled();
    if (!message_.empty()) {
        ImGui::SameLine();
        ImGui::TextColored(
            ImGui::ColorConvertU32ToFloat4(message_error_ ? tok().BAD : tok().DIM), "%s", message_.c_str());
    }
    if (shown && !images.printing_cached(*shown)) {
        const std::string cmd = fetch_printing_command(images.cache_dir(), card, *shown);
        ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(tok().BAD), "%s #%s: not downloaded", shown->set.c_str(),
            shown->number.c_str());
        ImGui::SameLine();
        const bool busy = fetching();
        ImGui::BeginDisabled(busy);
        if (ImGui::SmallButton(busy ? "Fetching…" : "Fetch")) start_fetch(images, card, *shown);
        ImGui::EndDisabled();
        ImGui::TextDisabled("%s", cmd.c_str());
    }
    if (!fetch_message_.empty() && fetch_key_ == card.key) ImGui::TextDisabled("%s", fetch_message_.c_str());
    ImGui::Spacing();
    const ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("##printings", 5, flags, ImVec2(0, -1))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, THUMB_W);
    ImGui::TableSetupColumn("Set", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Code", ImGuiTableColumnFlags_WidthFixed, 5 * char_width());
    ImGui::TableSetupColumn("No.", ImGuiTableColumnFlags_WidthFixed, 5 * char_width());
    ImGui::TableSetupColumn("Year", ImGuiTableColumnFlags_WidthFixed, 16 * char_width());
    ImGui::TableHeadersRow();
    for (size_t i = 0; i < card.printings.size(); ++i) draw_printing_row(images, card, i, shown == &card.printings[i]);
    scroll_shown_ = false;
    ImGui::EndTable();
}

void PrintingsTab::draw_printing_row(CardImages &images, const ImageCard &card, size_t i, bool shown) {
    const ImagePrinting &p = card.printings[i];
    const float th = THUMB_W * CARD_RATIO;
    ImGui::TableNextRow(0, th + 4);
    ImGui::TableNextColumn();
    ImGui::PushID(static_cast<int>(i));
    const ImVec2 at = ImGui::GetCursorScreenPos();
    // The whole row is one selectable; the thumbnail is drawn over it.
    if (ImGui::Selectable("##row", shown, ImGuiSelectableFlags_SpanAllColumns, ImVec2(0, th))) {
        std::string err;
        const bool is_default = static_cast<int>(i) == card.chosen;
        message_error_ =
            !images.set_override(card.key, is_default ? PrintingChoice() : PrintingChoice{p.set, p.number}, &err);
        message_ = message_error_ ? err : card.display() + ": " + p.set_name + " #" + p.number;
    }
    if (shown && scroll_shown_) ImGui::SetScrollHereY(0.3f);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    ImTextureData *tex = nullptr;
    if (!p.faces.empty() && images.cached(p.faces[0])) tex = images.textures().get(images.path_of(p.faces[0]), THUMB_W);
    if (tex) {
        dl->AddImageRounded(
            tex->GetTexRef(), at, ImVec2(at.x + THUMB_W, at.y + th), ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, 2.0f);
    } else {
        dl->AddRect(at, ImVec2(at.x + THUMB_W, at.y + th), tok().LINE, 2.0f);
    }
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(p.set_name.c_str());
    std::string tags;
    if (static_cast<int>(i) == card.chosen) tags += "default";
    if (p.promo) tags += std::string(tags.empty() ? "" : " · ") + "promo";
    if (!images.printing_cached(p)) tags += std::string(tags.empty() ? "" : " · ") + "not downloaded";
    if (!tags.empty()) ImGui::TextDisabled("%s", tags.c_str());
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(p.set.c_str());
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(p.number.c_str());
    ImGui::TableNextColumn();
    ImGui::TextUnformatted(year_of(p.released).c_str());
    if (!p.set_type.empty()) ImGui::TextDisabled("%s", p.set_type.c_str());
    ImGui::PopID();
}

void PrintingsTab::start_fetch(CardImages &images, const ImageCard &card, const ImagePrinting &p) {
    const std::string dir = workbench_cache_dir();
    make_dirs(dir);
    fetch_err_path_ = dir + "/fetch_images.stderr";
    fetch_ = std::make_unique<ChildProcess>();
    fetch_key_ = card.key;
    std::string err;
    if (!fetch_->start(fetch_printing_argv(repo_root(), images.cache_dir(), card, p), repo_root(),
            dir + "/fetch_images.stdout", fetch_err_path_, &err)) {
        fetch_message_ = "Fetch did not start: " + err;
        fetching_ = false;
        return;
    }
    fetching_ = true;
    fetch_message_ = "Fetching " + p.set + " #" + p.number + " from Scryfall…";
}

void PrintingsTab::poll_fetch(CardImages &images) {
    if (!fetch_ || !fetching_ || fetch_->poll()) return;
    fetching_ = false;
    if (fetch_->ok()) {
        fetch_message_ = "Fetched.";
    } else {
        std::string text;
        read_file(fetch_err_path_, &text);
        const std::string tail = last_line(text);
        fetch_message_ = "Fetch failed (" + fetch_->status_text() + ")" + (tail.empty() ? "" : ": " + tail);
    }
    images.reload();
}

}  // namespace rmwork
