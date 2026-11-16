#include "layout.h"

#include <algorithm>
#include <cctype>
#include <utility>

#include "imgui_internal.h"

namespace {

using rmwork::LayoutNode;
using rmwork::Tool;

// A built tab group: its dock node and its tools.
struct BuiltGroup {
        ImGuiID node;
        const std::vector<Tool> *tools;
};

const rmwork::ToolInfo TOOLS[rmwork::TOOL_COUNT] = {
    {"table", "Table", "The board as the chosen viewer saw it: permanents, the stack and the hands."},
    {"players", "Players", "Each player's life, zone counts and designations."},
    {"choices", "Choices", "The decision's menu: every legal choice, the one taken, and the policy's π."},
    {"inspector", "Decision Inspector", "The pending decision and its cost, with μ and each policy's evaluation."},
    {"tree", "State Tree", "Every table of the current RMSTATE frame, row by row."},
    {"log", "Game Log", "The game's narrative, or the viewing seat's own log in a seat view."},
    {"swings", "Swings", "The decisions where the value moved the most."},
    {"graph", "Value Graph", "V per seat and per policy across the game, with life totals."},
    {"model", "Model's Read", "What the model expects here: V, π over the menu and its top choices."},
};

const char *const MODE_KEYS[rmwork::MODE_COUNT] = {"play", "replay"};
const char *const MODE_TITLES[rmwork::MODE_COUNT] = {"Play", "Replay"};
const char *const DOCK_HOSTS[rmwork::MODE_COUNT] = {"##rmwork.dockhost.play", "##rmwork.dockhost.replay"};

}  // namespace

static LayoutNode group(std::vector<Tool> tools);
static LayoutNode split(LayoutNode::Kind kind, std::vector<float> sizes, std::vector<LayoutNode> children);
static std::vector<rmwork::Preset> make_presets();
static std::string lower(const std::string &s);
static void collect_groups(const LayoutNode &n, std::vector<std::vector<Tool>> *out);
static void collect_rects(const LayoutNode &n, ImVec2 pos, ImVec2 size, std::vector<std::pair<ImVec2, ImVec2>> *out);
static void build_node(const LayoutNode &n, ImGuiID id, std::vector<BuiltGroup> *out);
static void set_dock_order(const std::string &window_name, int order);
static ImGuiID tab_id(const std::string &window_name);
static int position_in(const std::vector<Tool> &tools, rmwork::Mode m, const ImGuiWindow *w);

static LayoutNode group(std::vector<Tool> tools) {
    LayoutNode n;
    n.kind = LayoutNode::Kind::GROUP;
    n.tools = std::move(tools);
    return n;
}

static LayoutNode split(LayoutNode::Kind kind, std::vector<float> sizes, std::vector<LayoutNode> children) {
    LayoutNode n;
    n.kind = kind;
    n.sizes = std::move(sizes);
    n.children = std::move(children);
    return n;
}

// The mock-up's presets with its split ratios: a ROW splits left to right, a COL top to bottom.
static std::vector<rmwork::Preset> make_presets() {
    using K = LayoutNode::Kind;
    const LayoutNode replay = split(K::ROW, {0.7f, 0.3f},
        {split(K::COL, {0.7f, 0.3f},
             {split(K::ROW, {0.66f, 0.34f}, {group({Tool::TABLE}), group({Tool::INSPECTOR, Tool::PLAYERS})}),
                 group({Tool::TREE, Tool::SWINGS})}),
            split(K::COL, {0.46f, 0.54f}, {group({Tool::LOG, Tool::CHOICES, Tool::MODEL}), group({Tool::GRAPH})})});
    const LayoutNode play = split(K::ROW, {0.17f, 0.58f, 0.25f},
        {split(K::COL, {0.6f, 0.4f}, {group({Tool::PLAYERS}), group({Tool::MODEL})}), group({Tool::TABLE}),
            split(K::COL, {0.62f, 0.38f}, {group({Tool::LOG, Tool::TREE}), group({Tool::CHOICES})})});
    const LayoutNode analysis = split(K::COL, {0.52f, 0.48f},
        {split(K::ROW, {0.36f, 0.32f, 0.32f},
             {group({Tool::INSPECTOR}), group({Tool::CHOICES, Tool::LOG}), group({Tool::SWINGS, Tool::TREE})}),
            split(K::ROW, {0.75f, 0.25f}, {group({Tool::GRAPH}), group({Tool::MODEL, Tool::PLAYERS})})});
    const LayoutNode table_only =
        split(K::ROW, {0.78f, 0.22f}, {group({Tool::TABLE}), group({Tool::CHOICES, Tool::LOG})});
    return {{"Replay (ImHex)", "replay", replay}, {"Play", "play", play}, {"Analysis", "analysis", analysis},
        {"Table only", "table", table_only}};
}

static std::string lower(const std::string &s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

static void collect_groups(const LayoutNode &n, std::vector<std::vector<Tool>> *out) {
    if (n.kind == LayoutNode::Kind::GROUP) {
        out->push_back(n.tools);
        return;
    }
    for (const LayoutNode &c : n.children) collect_groups(c, out);
}

static void collect_rects(const LayoutNode &n, ImVec2 pos, ImVec2 size, std::vector<std::pair<ImVec2, ImVec2>> *out) {
    if (n.kind == LayoutNode::Kind::GROUP) {
        out->push_back({pos, ImVec2(pos.x + size.x, pos.y + size.y)});
        return;
    }
    float total = 0;
    for (float s : n.sizes) total += s;
    const bool row = n.kind == LayoutNode::Kind::ROW;
    float at = 0;
    for (size_t i = 0; i < n.children.size(); ++i) {
        const float f = n.sizes[i] / total;
        const ImVec2 p = row ? ImVec2(pos.x + at * size.x, pos.y) : ImVec2(pos.x, pos.y + at * size.y);
        const ImVec2 sz = row ? ImVec2(f * size.x, size.y) : ImVec2(size.x, f * size.y);
        collect_rects(n.children[i], p, sz, out);
        at += f;
    }
}

// Splits node `id` as `n` says, recording each group's node: a split of k children cuts off the
// first child's share of what remains, k - 1 times.
static void build_node(const LayoutNode &n, ImGuiID id, std::vector<BuiltGroup> *out) {
    if (n.kind == LayoutNode::Kind::GROUP) {
        out->push_back({id, &n.tools});
        return;
    }
    const ImGuiDir dir = n.kind == LayoutNode::Kind::ROW ? ImGuiDir_Left : ImGuiDir_Up;
    float remaining = 0;
    for (float s : n.sizes) remaining += s;
    ImGuiID rest = id;
    for (size_t i = 0; i < n.children.size(); ++i) {
        if (i + 1 == n.children.size()) {
            build_node(n.children[i], rest, out);
            break;
        }
        ImGuiID first = 0, other = 0;
        ImGui::DockBuilderSplitNode(rest, dir, n.sizes[i] / remaining, &first, &other);
        build_node(n.children[i], first, out);
        remaining -= n.sizes[i];
        rest = other;
    }
}

// The window's position among its node's tabs (the window, or its saved settings when it has not
// been created yet).
static void set_dock_order(const std::string &window_name, int order) {
    const ImGuiID id = ImHashStr(window_name.c_str());
    if (ImGuiWindow *w = ImGui::FindWindowByID(id))
        w->DockOrder = static_cast<short>(order);
    else if (ImGuiWindowSettings *s = ImGui::FindWindowSettingsByID(id))
        s->DockOrder = static_cast<short>(order);
}

// The tab ID ImGui gives a docked window (ImGuiWindow::TabId).
static ImGuiID tab_id(const std::string &window_name) {
    return ImHashStr("#TAB", 4, ImHashStr(window_name.c_str()));
}

// The index in `tools` of the tool whose window in `m` is `w` (tools.size() for another window).
static int position_in(const std::vector<Tool> &tools, rmwork::Mode m, const ImGuiWindow *w) {
    for (size_t i = 0; i < tools.size(); ++i)
        if (w && w->ID == ImHashStr(rmwork::tool_window_name(m, tools[i]).c_str())) return static_cast<int>(i);
    return static_cast<int>(tools.size());
}

namespace rmwork {

const ToolInfo &tool_info(Tool t) {
    return TOOLS[tool_index(t)];
}

const char *mode_key(Mode m) {
    return MODE_KEYS[mode_index(m)];
}

const char *mode_title(Mode m) {
    return MODE_TITLES[mode_index(m)];
}

std::string tool_window_name(Mode m, Tool t) {
    const ToolInfo &info = tool_info(t);
    return std::string(info.title) + "###" + mode_key(m) + "." + info.key;
}

ImGuiID dockspace_id(Mode m) {
    return ImHashStr(m == Mode::PLAY ? "rmwork.dockspace.play" : "rmwork.dockspace.replay");
}

const char *dockspace_host_name(Mode m) {
    return DOCK_HOSTS[mode_index(m)];
}

const std::vector<Preset> &presets() {
    static const std::vector<Preset> all = make_presets();
    return all;
}

int default_preset(Mode m) {
    return m == Mode::PLAY ? 1 : 0;
}

int find_preset(const std::string &name) {
    const std::string want = lower(name);
    const std::vector<Preset> &all = presets();
    for (size_t i = 0; i < all.size(); ++i)
        if (want == lower(all[i].name) || want == all[i].slug) return static_cast<int>(i);
    return -1;
}

std::vector<std::vector<Tool>> preset_groups(const Preset &p) {
    std::vector<std::vector<Tool>> out;
    collect_groups(p.root, &out);
    return out;
}

std::vector<std::pair<ImVec2, ImVec2>> preset_group_rects(const Preset &p, ImVec2 pos, ImVec2 size) {
    std::vector<std::pair<ImVec2, ImVec2>> out;
    collect_rects(p.root, pos, size, &out);
    return out;
}

ToolFlags preset_tools(const Preset &p) {
    ToolFlags open{};
    for (const std::vector<Tool> &g : preset_groups(p))
        for (Tool t : g) open[tool_index(t)] = true;
    return open;
}

std::vector<Tool> preset_order(const Preset &p) {
    std::vector<Tool> order;
    for (const std::vector<Tool> &g : preset_groups(p)) order.insert(order.end(), g.begin(), g.end());
    const ToolFlags open = preset_tools(p);
    for (int i = 0; i < TOOL_COUNT; ++i)
        if (!open[i]) order.push_back(static_cast<Tool>(i));
    return order;
}

void apply_preset(Mode m, const Preset &p, ImVec2 size, ToolFlags *open) {
    const ImGuiID ds = dockspace_id(m);
    ImGui::DockBuilderRemoveNode(ds);
    ImGui::DockBuilderAddNode(ds, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(ds, size);
    std::vector<BuiltGroup> groups;
    build_node(p.root, ds, &groups);
    *open = preset_tools(p);
    for (const BuiltGroup &g : groups) {
        int order = 0;
        for (Tool t : *g.tools) {
            const std::string name = tool_window_name(m, t);
            ImGui::DockBuilderDockWindow(name.c_str(), g.node);
            set_dock_order(name, order++);
        }
        ImGui::DockBuilderGetNode(g.node)->SelectedTabId = tab_id(tool_window_name(m, g.tools->front()));
    }
    // The closed tools wait behind the first group, so reopening one from View adds it there.
    int order = static_cast<int>(groups.front().tools->size());
    for (int i = 0; i < TOOL_COUNT; ++i) {
        if ((*open)[i]) continue;
        const std::string name = tool_window_name(m, static_cast<Tool>(i));
        ImGui::DockBuilderDockWindow(name.c_str(), groups.front().node);
        set_dock_order(name, order++);
    }
    ImGui::DockBuilderFinish(ds);
    // No tool keeps the focus, so each group shows its first tab.
    ImGui::SetWindowFocus(nullptr);
}

void order_preset_tabs(Mode m, const Preset &p) {
    for (const std::vector<Tool> &g : preset_groups(p)) {
        const ImGuiWindow *first = ImGui::FindWindowByName(tool_window_name(m, g.front()).c_str());
        ImGuiDockNode *node = first ? first->DockNode : nullptr;
        if (!node || !node->TabBar) continue;
        ImVector<ImGuiTabItem> &tabs = node->TabBar->Tabs;
        std::stable_sort(tabs.begin(), tabs.end(), [&](const ImGuiTabItem &a, const ImGuiTabItem &b) {
            return position_in(g, m, a.Window) < position_in(g, m, b.Window);
        });
    }
}

}  // namespace rmwork
