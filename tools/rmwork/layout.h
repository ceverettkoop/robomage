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
// and switching mode submits the other dockspace and its windows. A layout is a tree of splits and
// tab groups (LayoutNode), built on the current mode's dockspace with the DockBuilder API: the four
// presets are the mock-up's, and a user's named layouts (View → Save Layout As…) are trees read
// back from the dockspace (capture_layout) and kept per mode in settings.json.

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
    MODEL,
    SIDEBOARD
};
inline constexpr int TOOL_COUNT = 10;

// Per tool: open (true) or closed.
using ToolFlags = std::array<bool, TOOL_COUNT>;

struct ToolInfo {
        const char *key;    // settings key and window ID suffix ("table")
        const char *title;  // tab title ("Table")
        const char *blurb;  // what the tool will show, for its placeholder
};

const ToolInfo &tool_info(Tool t);
// The tool whose key or title is `name` (case-insensitive), or -1.
int find_tool(const std::string &name);
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

// A node of a layout's tree: a tab group (its tools in tab order, the `selected` one shown) or a
// row (left to right) / column (top to bottom) of children with their relative sizes.
struct LayoutNode {
        enum class Kind {
            GROUP,
            ROW,
            COL
        };
        Kind kind = Kind::GROUP;
        std::vector<Tool> tools;
        int selected = 0;
        std::vector<float> sizes;
        std::vector<LayoutNode> children;
};

// A layout the user saved under a name, in one mode.
struct NamedLayout {
        std::string name;
        LayoutNode root;
};
using NamedLayouts = std::vector<NamedLayout>;
inline constexpr size_t LAYOUT_NAME_MAX = 48;

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
// The layout's tab groups in tree order (each with its selected index in LayoutNode::selected).
std::vector<const LayoutNode *> layout_groups(const LayoutNode &root);
// Each tab group's rectangle (min, max) when the layout fills `size` from `pos`, in
// layout_groups order (splitter widths not deducted).
std::vector<std::pair<ImVec2, ImVec2>> layout_group_rects(const LayoutNode &root, ImVec2 pos, ImVec2 size);
// The tools the layout opens.
ToolFlags layout_tools(const LayoutNode &root);
// Every tool: the layout's groups in order, then the tools it closes. Windows begun in this order
// on the frames after apply_layout get their tabs in the layout's order.
std::vector<Tool> layout_order(const LayoutNode &root);

// Rebuilds `m`'s dockspace as `root` over `size`: every tool of the layout docked into its group (in
// the group's order, its selected tool shown), every other tool docked behind the first group and
// closed; `open` becomes the layout's tools.
void apply_layout(Mode m, const LayoutNode &root, ImVec2 size, ToolFlags *open);

// Puts each of the layout's tab groups' tabs in the layout's order (tabs that join a node on
// different frames are otherwise appended as they come); run on the frames after apply_layout.
void order_layout_tabs(Mode m, const LayoutNode &root);

// `m`'s dockspace as a layout: its splits (same-direction nesting flattened, sizes as fractions of
// the split) and tab groups (the `open` tools in tab order, the shown one selected; empty groups
// dropped). An open tool outside the dockspace (floating) joins the first group, behind its tabs.
LayoutNode capture_layout(Mode m, const ToolFlags &open);
// The same tree: kinds, tools, selected tabs, and sizes within `tolerance` (fractions).
bool layouts_match(const LayoutNode &a, const LayoutNode &b, float tolerance);
// The layout as one line ("row[0.70 col[...] ...]"), for messages.
std::string describe_layout(const LayoutNode &root);
// False with `why` for a malformed tree: a split with fewer than two children or sizes that do not
// match them or are not positive, a group's selected tab out of range, a tool twice.
bool layout_valid(const LayoutNode &root, std::string *why);

// A name for a saved layout: 1 to LAYOUT_NAME_MAX characters, not blank, no control characters,
// not a preset's name or slug; false with `why`.
bool layout_name_valid(const std::string &name, std::string *why);
// The saved layout named `name` (case-insensitive), or -1.
int find_named_layout(const NamedLayouts &all, const std::string &name);
enum class SaveResult {
    SAVED,
    EXISTS,  // a layout has the name and `overwrite` was false
    INVALID  // the name is not valid
};
// Saves `root` as `name`: a new entry at the end, or (with `overwrite`) over the one already so
// named, which keeps its place and takes this spelling; `why` says what is wrong when not SAVED.
SaveResult save_named_layout(
    NamedLayouts *all, const std::string &name, const LayoutNode &root, bool overwrite, std::string *why);
// Renames `from` to `to` (a valid name no other layout has); false with `why`.
bool rename_named_layout(NamedLayouts *all, const std::string &from, const std::string &to, std::string *why);
// Removes the layout named `name`; false when there is none.
bool delete_named_layout(NamedLayouts *all, const std::string &name);

}  // namespace rmwork

#endif /* RMWORK_LAYOUT_H */
