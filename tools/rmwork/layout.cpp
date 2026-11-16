#include "layout.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <utility>

#include "imgui_internal.h"

namespace {

using rmwork::LayoutNode;
using rmwork::Tool;

// A built tab group: its dock node and its layout node.
struct BuiltGroup {
        ImGuiID node;
        const LayoutNode *group;
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
static void collect_groups(const LayoutNode &n, std::vector<const LayoutNode *> *out);
static void collect_rects(const LayoutNode &n, ImVec2 pos, ImVec2 size, std::vector<std::pair<ImVec2, ImVec2>> *out);
static void build_node(const LayoutNode &n, ImGuiID id, std::vector<BuiltGroup> *out);
static void set_dock_order(const std::string &window_name, int order);
static ImGuiID tab_id(const std::string &window_name);
static int position_in(const std::vector<Tool> &tools, rmwork::Mode m, const ImGuiWindow *w);
static int tool_of_window(rmwork::Mode m, const ImGuiWindow *w);
static LayoutNode capture_node(rmwork::Mode m, const ImGuiDockNode *n, const rmwork::ToolFlags &open);
static void add_child(LayoutNode *split, LayoutNode child, float size);
static bool nodes_match(const LayoutNode &a, const LayoutNode &b, float tolerance);
static void describe(const LayoutNode &n, std::string *out);
static bool valid_node(const LayoutNode &n, int depth, rmwork::ToolFlags *seen, std::string *why);

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

static void collect_groups(const LayoutNode &n, std::vector<const LayoutNode *> *out) {
    if (n.kind == LayoutNode::Kind::GROUP) {
        out->push_back(&n);
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
        out->push_back({id, &n});
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

// The tool whose window in `m` is `w`, or -1.
static int tool_of_window(rmwork::Mode m, const ImGuiWindow *w) {
    for (int t = 0; t < rmwork::TOOL_COUNT; ++t)
        if (w && w->ID == ImHashStr(rmwork::tool_window_name(m, static_cast<Tool>(t)).c_str())) return t;
    return -1;
}

// The subtree under dock node `n`, sizes in pixels along the split (capture_layout normalises
// them); an empty group where no open tool is docked.
static LayoutNode capture_node(rmwork::Mode m, const ImGuiDockNode *n, const rmwork::ToolFlags &open) {
    LayoutNode out;
    if (n->IsSplitNode()) {
        const bool row = n->SplitAxis == ImGuiAxis_X;
        out.kind = row ? LayoutNode::Kind::ROW : LayoutNode::Kind::COL;
        for (const ImGuiDockNode *c : n->ChildNodes) {
            if (!c) continue;
            add_child(&out, capture_node(m, c, open), row ? c->Size.x : c->Size.y);
        }
        if (out.children.size() == 1) return out.children.front();
        if (out.children.empty()) return LayoutNode();
        return out;
    }
    ImGuiID shown = n->TabBar ? n->TabBar->SelectedTabId : 0;
    if (n->TabBar) {
        for (const ImGuiTabItem &tab : n->TabBar->Tabs) {
            const int t = tool_of_window(m, tab.Window);
            if (t < 0 || !open[t]) continue;
            if (tab.ID == shown) out.selected = static_cast<int>(out.tools.size());
            out.tools.push_back(static_cast<Tool>(t));
        }
    } else {
        for (const ImGuiWindow *w : n->Windows) {
            const int t = tool_of_window(m, w);
            if (t < 0 || !open[t]) continue;
            if (w == n->VisibleWindow) out.selected = static_cast<int>(out.tools.size());
            out.tools.push_back(static_cast<Tool>(t));
        }
    }
    return out;
}

// Appends `child` (of pixel extent `size`) to `split`: an empty group is dropped, a split in the
// same direction is spliced in child by child.
static void add_child(LayoutNode *split, LayoutNode child, float size) {
    if (child.kind == LayoutNode::Kind::GROUP && child.tools.empty()) return;
    if (child.kind == split->kind) {
        for (size_t i = 0; i < child.children.size(); ++i) {
            split->children.push_back(std::move(child.children[i]));
            split->sizes.push_back(child.sizes[i]);
        }
        return;
    }
    split->children.push_back(std::move(child));
    split->sizes.push_back(size);
}

static bool nodes_match(const LayoutNode &a, const LayoutNode &b, float tolerance) {
    if (a.kind != b.kind) return false;
    if (a.kind == LayoutNode::Kind::GROUP) return a.tools == b.tools && (a.tools.empty() || a.selected == b.selected);
    if (a.children.size() != b.children.size() || a.sizes.size() != b.sizes.size()) return false;
    float ta = 0, tb = 0;
    for (float s : a.sizes) ta += s;
    for (float s : b.sizes) tb += s;
    for (size_t i = 0; i < a.children.size(); ++i) {
        if (std::fabs(a.sizes[i] / ta - b.sizes[i] / tb) > tolerance) return false;
        if (!nodes_match(a.children[i], b.children[i], tolerance)) return false;
    }
    return true;
}

static void describe(const LayoutNode &n, std::string *out) {
    if (n.kind == LayoutNode::Kind::GROUP) {
        *out += "[";
        for (size_t i = 0; i < n.tools.size(); ++i) {
            if (i) *out += " ";
            if (static_cast<int>(i) == n.selected) *out += "*";
            *out += rmwork::tool_info(n.tools[i]).key;
        }
        *out += "]";
        return;
    }
    *out += n.kind == LayoutNode::Kind::ROW ? "row(" : "col(";
    float total = 0;
    for (float s : n.sizes) total += s;
    for (size_t i = 0; i < n.children.size(); ++i) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%s%.3f:", i ? " " : "", i < n.sizes.size() ? n.sizes[i] / total : 0.0f);
        *out += buf;
        describe(n.children[i], out);
    }
    *out += ")";
}

static bool valid_node(const LayoutNode &n, int depth, rmwork::ToolFlags *seen, std::string *why) {
    if (depth > 16) {
        *why = "the layout is nested more than 16 deep";
        return false;
    }
    if (n.kind == LayoutNode::Kind::GROUP) {
        for (Tool t : n.tools) {
            if ((*seen)[rmwork::tool_index(t)]) {
                *why = std::string("tool \"") + rmwork::tool_info(t).key + "\" appears twice";
                return false;
            }
            (*seen)[rmwork::tool_index(t)] = true;
        }
        if (!n.tools.empty() && (n.selected < 0 || n.selected >= static_cast<int>(n.tools.size()))) {
            *why = "a tab group's selected tab is out of range";
            return false;
        }
        return true;
    }
    if (n.children.size() < 2 || n.sizes.size() != n.children.size()) {
        *why = "a split has fewer than two children, or not one size per child";
        return false;
    }
    for (float s : n.sizes) {
        if (!(s > 0.0f) || !std::isfinite(s)) {
            *why = "a split's size is not a positive number";
            return false;
        }
    }
    for (const LayoutNode &c : n.children)
        if (!valid_node(c, depth + 1, seen, why)) return false;
    return true;
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

std::vector<const LayoutNode *> layout_groups(const LayoutNode &root) {
    std::vector<const LayoutNode *> out;
    collect_groups(root, &out);
    return out;
}

std::vector<std::pair<ImVec2, ImVec2>> layout_group_rects(const LayoutNode &root, ImVec2 pos, ImVec2 size) {
    std::vector<std::pair<ImVec2, ImVec2>> out;
    collect_rects(root, pos, size, &out);
    return out;
}

ToolFlags layout_tools(const LayoutNode &root) {
    ToolFlags open{};
    for (const LayoutNode *g : layout_groups(root))
        for (Tool t : g->tools) open[tool_index(t)] = true;
    return open;
}

std::vector<Tool> layout_order(const LayoutNode &root) {
    std::vector<Tool> order;
    for (const LayoutNode *g : layout_groups(root)) order.insert(order.end(), g->tools.begin(), g->tools.end());
    const ToolFlags open = layout_tools(root);
    for (int i = 0; i < TOOL_COUNT; ++i)
        if (!open[i]) order.push_back(static_cast<Tool>(i));
    return order;
}

void apply_layout(Mode m, const LayoutNode &root, ImVec2 size, ToolFlags *open) {
    const ImGuiID ds = dockspace_id(m);
    ImGui::DockBuilderRemoveNode(ds);
    ImGui::DockBuilderAddNode(ds, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(ds, size);
    std::vector<BuiltGroup> groups;
    build_node(root, ds, &groups);
    *open = layout_tools(root);
    for (const BuiltGroup &g : groups) {
        int order = 0;
        for (Tool t : g.group->tools) {
            const std::string name = tool_window_name(m, t);
            ImGui::DockBuilderDockWindow(name.c_str(), g.node);
            set_dock_order(name, order++);
        }
        if (g.group->tools.empty()) continue;
        const size_t shown = std::min(static_cast<size_t>(std::max(0, g.group->selected)), g.group->tools.size() - 1);
        ImGui::DockBuilderGetNode(g.node)->SelectedTabId = tab_id(tool_window_name(m, g.group->tools[shown]));
    }
    // The closed tools wait behind the first group, so reopening one from View adds it there.
    int order = static_cast<int>(groups.front().group->tools.size());
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

void order_layout_tabs(Mode m, const LayoutNode &root) {
    for (const LayoutNode *group : layout_groups(root)) {
        const std::vector<Tool> &g = group->tools;
        if (g.empty()) continue;
        const ImGuiWindow *first = ImGui::FindWindowByName(tool_window_name(m, g.front()).c_str());
        ImGuiDockNode *node = first ? first->DockNode : nullptr;
        if (!node || !node->TabBar) continue;
        ImVector<ImGuiTabItem> &tabs = node->TabBar->Tabs;
        std::stable_sort(tabs.begin(), tabs.end(), [&](const ImGuiTabItem &a, const ImGuiTabItem &b) {
            return position_in(g, m, a.Window) < position_in(g, m, b.Window);
        });
    }
}

LayoutNode capture_layout(Mode m, const ToolFlags &open) {
    LayoutNode root;
    if (const ImGuiDockNode *ds = ImGui::DockBuilderGetNode(dockspace_id(m))) root = capture_node(m, ds, open);
    if (root.kind != LayoutNode::Kind::GROUP) {
        std::vector<LayoutNode *> stack = {&root};
        while (!stack.empty()) {
            LayoutNode *n = stack.back();
            stack.pop_back();
            if (n->kind == LayoutNode::Kind::GROUP) continue;
            float sum = 0;
            for (float s : n->sizes) sum += s;
            for (float &s : n->sizes) s = sum > 0 ? s / sum : 1.0f / static_cast<float>(n->sizes.size());
            for (LayoutNode &c : n->children) stack.push_back(&c);
        }
    }
    const ToolFlags placed = layout_tools(root);
    LayoutNode *first = &root;
    while (first->kind != LayoutNode::Kind::GROUP) first = &first->children.front();
    for (int t = 0; t < TOOL_COUNT; ++t)
        if (open[t] && !placed[t]) first->tools.push_back(static_cast<Tool>(t));
    return root;
}

bool layouts_match(const LayoutNode &a, const LayoutNode &b, float tolerance) {
    return nodes_match(a, b, tolerance);
}

std::string describe_layout(const LayoutNode &root) {
    std::string out;
    describe(root, &out);
    return out;
}

bool layout_valid(const LayoutNode &root, std::string *why) {
    ToolFlags seen{};
    return valid_node(root, 0, &seen, why);
}

bool layout_name_valid(const std::string &name, std::string *why) {
    bool blank = true;
    for (char c : name) {
        if (static_cast<unsigned char>(c) < 0x20 || c == 0x7f) {
            *why = "a name has no control characters";
            return false;
        }
        if (c != ' ') blank = false;
    }
    if (blank) {
        *why = "the name is empty";
        return false;
    }
    if (name.size() > LAYOUT_NAME_MAX) {
        *why = "the name is longer than " + std::to_string(LAYOUT_NAME_MAX) + " characters";
        return false;
    }
    if (find_preset(name) >= 0) {
        *why = "\"" + name + "\" is a preset's name";
        return false;
    }
    return true;
}

int find_named_layout(const NamedLayouts &all, const std::string &name) {
    const std::string want = lower(name);
    for (size_t i = 0; i < all.size(); ++i)
        if (lower(all[i].name) == want) return static_cast<int>(i);
    return -1;
}

SaveResult save_named_layout(
    NamedLayouts *all, const std::string &name, const LayoutNode &root, bool overwrite, std::string *why) {
    if (!layout_name_valid(name, why)) return SaveResult::INVALID;
    const int i = find_named_layout(*all, name);
    if (i >= 0 && !overwrite) {
        *why = "a layout is already named \"" + (*all)[static_cast<size_t>(i)].name + "\"";
        return SaveResult::EXISTS;
    }
    if (i >= 0)
        (*all)[static_cast<size_t>(i)] = {name, root};
    else
        all->push_back({name, root});
    return SaveResult::SAVED;
}

bool rename_named_layout(NamedLayouts *all, const std::string &from, const std::string &to, std::string *why) {
    const int i = find_named_layout(*all, from);
    if (i < 0) {
        *why = "no layout is named \"" + from + "\"";
        return false;
    }
    if (!layout_name_valid(to, why)) return false;
    const int other = find_named_layout(*all, to);
    if (other >= 0 && other != i) {
        *why = "another layout is named \"" + (*all)[static_cast<size_t>(other)].name + "\"";
        return false;
    }
    (*all)[static_cast<size_t>(i)].name = to;
    return true;
}

bool delete_named_layout(NamedLayouts *all, const std::string &name) {
    const int i = find_named_layout(*all, name);
    if (i < 0) return false;
    all->erase(all->begin() + i);
    return true;
}

}  // namespace rmwork
