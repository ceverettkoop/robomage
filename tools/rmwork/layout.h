#ifndef RMWORK_LAYOUT_H
#define RMWORK_LAYOUT_H

#include <array>
#include <string>
#include <vector>

#include "imgui.h"

// ── Modes, tools and the preset layouts ─────────────────────────────────────────────────────────
//
// Each mode has its own dockspace and its own copy of every tool window: a window's name carries
// the mode in its ID ("Table###replay.table"), so ImGui's one ini file keeps both modes' layouts
// and switching mode submits the other dockspace and its windows. A preset is the mock-up's tree
// of splits and tab groups, built on the current mode's dockspace with the DockBuilder API.

namespace rmwork {

enum class Mode {
    PLAY,
    REPLAY
};
inline constexpr int MODE_COUNT = 2;

enum class Tool {
    TABLE,
    PLAYERS,
    CHOICES,
    INSPECTOR,
    TREE,
    LOG,
    SWINGS,
    GRAPH,
    MODEL
};
inline constexpr int TOOL_COUNT = 9;

// Per tool: open (true) or closed.
using ToolFlags = std::array<bool, TOOL_COUNT>;

struct ToolInfo {
        const char *key;    // settings key and window ID suffix ("table")
        const char *title;  // tab title ("Table")
        const char *blurb;  // what the tool will show, for its placeholder
};

const ToolInfo &tool_info(Tool t);
inline int tool_index(Tool t) {
    return static_cast<int>(t);
}
inline int mode_index(Mode m) {
    return static_cast<int>(m);
}
// "play" / "replay".
const char *mode_key(Mode m);
// "Play" / "Replay".
const char *mode_title(Mode m);
// The tool's window name in `m`: its title, then "###<mode>.<tool>".
std::string tool_window_name(Mode m, Tool t);
// The mode's dockspace ID and the name of the window that hosts it.
ImGuiID dockspace_id(Mode m);
const char *dockspace_host_name(Mode m);

// A node of a preset's tree: a tab group (its tools, the first one selected) or a row / column of
// children with their relative sizes.
struct LayoutNode {
        enum class Kind {
            GROUP,
            ROW,
            COL
        };
        Kind kind = Kind::GROUP;
        std::vector<Tool> tools;
        std::vector<float> sizes;
        std::vector<LayoutNode> children;
};

struct Preset {
        const char *name;  // "Replay (ImHex)"
        const char *slug;  // "replay", for --preset
        LayoutNode root;
};

// The four presets, in menu order: Replay (ImHex), Play, Analysis, Table only.
const std::vector<Preset> &presets();
// The preset a mode starts with and View → Reset Layout restores.
int default_preset(Mode m);
// The preset by name or slug (case-insensitive), or -1.
int find_preset(const std::string &name);
// The preset's tab groups in tree order.
std::vector<std::vector<Tool>> preset_groups(const Preset &p);
// Each tab group's rectangle (min, max) when the preset fills `size` from `pos`, in
// preset_groups order (splitter widths not deducted).
std::vector<std::pair<ImVec2, ImVec2>> preset_group_rects(const Preset &p, ImVec2 pos, ImVec2 size);
// The tools the preset opens.
ToolFlags preset_tools(const Preset &p);
// Every tool: the preset's groups in order, then the tools it closes. Windows begun in this order
// on the frames after apply_preset get their tabs in the preset's order.
std::vector<Tool> preset_order(const Preset &p);

// Rebuilds `m`'s dockspace as `p` over `size`: every tool of the preset docked into its group (in
// the group's order, the first one selected), every other tool docked behind the first group and
// closed; `open` becomes the preset's tools.
void apply_preset(Mode m, const Preset &p, ImVec2 size, ToolFlags *open);

// Puts each of the preset's tab groups' tabs in the preset's order (tabs that join a node on
// different frames are otherwise appended as they come); run on the frames after apply_preset.
void order_preset_tabs(Mode m, const Preset &p);

}  // namespace rmwork

#endif /* RMWORK_LAYOUT_H */
