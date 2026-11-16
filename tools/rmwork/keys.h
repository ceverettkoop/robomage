#ifndef RMWORK_KEYS_H
#define RMWORK_KEYS_H

#include <array>
#include <string>

#include "imgui.h"

// ── Key bindings: one table of actions, each with its default key chord ──────────────────────────
//
// The settings file stores a chord per action by the action's key, so a binding is rebound there
// (and, later, from Settings → Keys) without touching the code that reacts to it.

namespace rmwork {

enum class Action {
    REPLAY_FIRST,
    REPLAY_PREV,
    REPLAY_NEXT,
    REPLAY_LAST,
    PLAY_TO_END_OF_TURN,
    PLAY_SKIP_TURN,
    MAGNIFY,
    OPEN_SETTINGS,
    OPEN_REPLAY,
    SAVE_DECISION_LOG,
    NEW_GAME,
    FIND_IN_LOG
};
inline constexpr int ACTION_COUNT = 12;

// Where an action's key works.
enum class Scope {
    ANY,
    PLAY,
    REPLAY
};

struct ActionInfo {
        const char *key;      // settings key ("replay.next")
        const char *label;    // what it does ("Next decision")
        ImGuiKeyChord chord;  // default binding
        Scope scope;
        bool repeat;  // fires again while the key is held
};

// Per action: its bound chord (0 = unbound).
using KeyBindings = std::array<ImGuiKeyChord, ACTION_COUNT>;

const ActionInfo &action_info(Action a);
inline int action_index(Action a) {
    return static_cast<int>(a);
}
// The action whose settings key is `key`, or -1.
int find_action(const std::string &key);
KeyBindings default_bindings();
// "Ctrl+O", "RightArrow", "" for 0.
std::string chord_name(ImGuiKeyChord chord);
// chord_name for display: the arrow keys as arrows ("Ctrl+→"), "" for 0.
std::string chord_label(ImGuiKeyChord chord);
// The inverse of chord_name; false for an unknown key or modifier.
bool parse_chord(const std::string &text, ImGuiKeyChord *out);
// The chord was pressed this frame with exactly its modifiers held.
bool chord_pressed(ImGuiKeyChord chord, bool repeat);
// "Play", "Replay" or "Anywhere".
const char *scope_name(Scope s);

}  // namespace rmwork

#endif /* RMWORK_KEYS_H */
