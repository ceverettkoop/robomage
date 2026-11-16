#include "keys.h"

#include <cstring>

namespace {

const rmwork::ActionInfo ACTIONS[rmwork::ACTION_COUNT] = {
    {"replay.first", "First decision", ImGuiKey_Home, rmwork::Scope::REPLAY, false},
    {"replay.prev", "Previous decision", ImGuiKey_LeftArrow, rmwork::Scope::REPLAY, true},
    {"replay.next", "Next decision", ImGuiKey_RightArrow, rmwork::Scope::REPLAY, true},
    {"replay.last", "Last decision", ImGuiKey_End, rmwork::Scope::REPLAY, false},
    {"play.to_end_of_turn", "To end of turn", ImGuiKey_F2, rmwork::Scope::PLAY, false},
    {"play.skip_turn", "Skip turn", ImGuiKey_F6, rmwork::Scope::PLAY, false},
    {"magnify", "Magnify the card under the pointer (hold)", ImGuiKey_Q, rmwork::Scope::ANY, false},
    {"edit.settings", "Settings", ImGuiMod_Ctrl | ImGuiKey_Comma, rmwork::Scope::ANY, false},
    {"file.open_replay", "Open Replay", ImGuiMod_Ctrl | ImGuiKey_O, rmwork::Scope::ANY, false},
    {"file.save_decision_log", "Save Decision Log", ImGuiMod_Ctrl | ImGuiKey_S, rmwork::Scope::ANY, false},
    {"game.new", "New Game vs Model", ImGuiMod_Ctrl | ImGuiKey_N, rmwork::Scope::ANY, false},
    {"edit.find_in_log", "Find in Log", ImGuiMod_Ctrl | ImGuiKey_F, rmwork::Scope::ANY, false},
};

struct ModName {
        ImGuiKeyChord mod;
        const char *name;
};

const ModName MODS[] = {
    {ImGuiMod_Ctrl, "Ctrl"}, {ImGuiMod_Shift, "Shift"}, {ImGuiMod_Alt, "Alt"}, {ImGuiMod_Super, "Super"}};

}  // namespace

namespace rmwork {

const ActionInfo &action_info(Action a) {
    return ACTIONS[action_index(a)];
}

int find_action(const std::string &key) {
    for (int i = 0; i < ACTION_COUNT; ++i)
        if (key == ACTIONS[i].key) return i;
    return -1;
}

KeyBindings default_bindings() {
    KeyBindings b{};
    for (int i = 0; i < ACTION_COUNT; ++i) b[i] = ACTIONS[i].chord;
    return b;
}

std::string chord_name(ImGuiKeyChord chord) {
    if (chord == 0) return "";
    std::string out;
    for (const ModName &m : MODS)
        if (chord & m.mod) out += std::string(m.name) + "+";
    const ImGuiKey key = static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
    out += ImGui::GetKeyName(key);
    return out;
}

std::string chord_label(ImGuiKeyChord chord) {
    std::string name = chord_name(chord);
    const char *const arrows[][2] = {{"LeftArrow", "←"}, {"RightArrow", "→"}, {"UpArrow", "↑"}, {"DownArrow", "↓"}};
    for (const auto &a : arrows) {
        const size_t at = name.find(a[0]);
        if (at != std::string::npos) name.replace(at, std::strlen(a[0]), a[1]);
    }
    return name;
}

bool parse_chord(const std::string &text, ImGuiKeyChord *out) {
    if (text.empty()) {
        *out = 0;
        return true;
    }
    ImGuiKeyChord chord = 0;
    size_t start = 0;
    for (size_t plus = text.find('+'); plus != std::string::npos; plus = text.find('+', start)) {
        const std::string mod = text.substr(start, plus - start);
        bool known = false;
        for (const ModName &m : MODS) {
            if (mod == m.name) {
                chord |= m.mod;
                known = true;
            }
        }
        if (!known) return false;
        start = plus + 1;
    }
    const std::string key = text.substr(start);
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_NamedKey_END; ++k) {
        if (key == ImGui::GetKeyName(static_cast<ImGuiKey>(k))) {
            *out = chord | k;
            return true;
        }
    }
    return false;
}

bool chord_pressed(ImGuiKeyChord chord, bool repeat) {
    if (chord == 0) return false;
    const ImGuiKeyChord mods = chord & ImGuiMod_Mask_;
    if ((ImGui::GetIO().KeyMods & ImGuiMod_Mask_) != mods) return false;
    return ImGui::IsKeyPressed(static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_), repeat);
}

bool scopes_overlap(Scope a, Scope b) {
    return a == Scope::ANY || b == Scope::ANY || a == b;
}

int binding_conflict(const KeyBindings &b, Action a, ImGuiKeyChord chord) {
    if (chord == 0) return -1;
    for (int i = 0; i < ACTION_COUNT; ++i)
        if (i != action_index(a) && b[i] == chord && scopes_overlap(ACTIONS[i].scope, action_info(a).scope)) return i;
    return -1;
}

bool rebind(KeyBindings *b, Action a, ImGuiKeyChord chord, std::string *why) {
    const int other = binding_conflict(*b, a, chord);
    if (other >= 0) {
        const ActionInfo &o = ACTIONS[other];
        *why = chord_label(chord) + " is already the key of " + o.label + " (" + scope_name(o.scope) +
               "). Rebind that first, or choose another key.";
        return false;
    }
    (*b)[action_index(a)] = chord;
    return true;
}

ImGuiKeyChord pressed_chord() {
    for (int k = ImGuiKey_NamedKey_BEGIN; k < ImGuiKey_GamepadStart; ++k) {
        const ImGuiKey key = static_cast<ImGuiKey>(k);
        if (key == ImGuiKey_Escape || (key >= ImGuiKey_LeftCtrl && key <= ImGuiKey_RightSuper)) continue;
        if (ImGui::IsKeyPressed(key, false)) return (ImGui::GetIO().KeyMods & ImGuiMod_Mask_) | key;
    }
    return 0;
}

const char *scope_name(Scope s) {
    switch (s) {
        case Scope::PLAY:
            return "Play";
        case Scope::REPLAY:
            return "Replay";
        case Scope::ANY:
            break;
    }
    return "Anywhere";
}

}  // namespace rmwork
