#include "settings.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "../../src/featurize/json.h"
#include "files.h"
#include "paths.h"

namespace {

const char *const BACKGROUND_KINDS[] = {"color", "texture", "image"};

const rmwork::StepInfo STOP_STEPS[rmwork::STOP_STEP_COUNT] = {{"untap", "Untap", "Unt"}, {"upkeep", "Upkeep", "Upk"},
    {"draw", "Draw", "Drw"}, {"main1", "Main 1", "M1"}, {"beginning_of_combat", "Beginning of combat", "BC"},
    {"declare_attackers", "Declare attackers", "Atk"}, {"declare_blockers", "Declare blockers", "Blk"},
    {"combat_damage", "Combat damage", "Dmg"}, {"end_of_combat", "End of combat", "EC"}, {"main2", "Main 2", "M2"},
    {"end_step", "End step", "End"}, {"cleanup", "Cleanup", "Cln"}};
const int DEFAULT_STOPS[] = {1, 3, 5, 6, 10};
const char *const STOP_TURNS[] = {"own", "opponent"};

}  // namespace

static const char *background_kind_key(rmwork::TableBackground::Kind k);
static bool read_mode(const featurize::Json &v, rmwork::Mode *out, std::string *err);
static bool read_tools(
    const featurize::Json &v, std::array<rmwork::ToolFlags, rmwork::MODE_COUNT> *out, std::string *err);
static bool read_background(const featurize::Json &v, rmwork::TableBackground *out, std::string *err);
static bool read_font(const featurize::Json &v, rmwork::FontChoice *out, std::string *err);
static bool read_keys(const featurize::Json &v, rmwork::KeyBindings *out, std::string *err);
static bool read_string(
    const featurize::Json &obj, const char *key, const char *where, std::string *out, std::string *err);
static bool read_engine(const featurize::Json &v, rmwork::EngineSettings *out, std::string *err);
static bool read_models(const featurize::Json &v, rmwork::ModelSettings *out, std::string *err);
static bool read_stops(const featurize::Json &v, rmwork::StopSettings *out, std::string *err);
static bool read_images(const featurize::Json &v, std::string *out, std::string *err);
static bool read_recent(const featurize::Json &v, std::vector<std::string> *out, std::string *err);
static bool read_layout_node(const featurize::Json &v, int depth, rmwork::LayoutNode *out, std::string *err);
static bool read_layouts(
    const featurize::Json &v, std::array<rmwork::NamedLayouts, rmwork::MODE_COUNT> *out, std::string *err);
static std::string layout_json(const rmwork::LayoutNode &n);
static std::string stops_json(const std::array<bool, rmwork::STOP_STEP_COUNT> &stops);

static const char *background_kind_key(rmwork::TableBackground::Kind k) {
    return BACKGROUND_KINDS[static_cast<int>(k)];
}

static bool read_mode(const featurize::Json &v, rmwork::Mode *out, std::string *err) {
    for (int m = 0; m < rmwork::MODE_COUNT; ++m) {
        if (v.is_string() && v.as_string() == rmwork::mode_key(static_cast<rmwork::Mode>(m))) {
            *out = static_cast<rmwork::Mode>(m);
            return true;
        }
    }
    *err = "\"mode\" is not \"play\" or \"replay\"";
    return false;
}

static bool read_tools(
    const featurize::Json &v, std::array<rmwork::ToolFlags, rmwork::MODE_COUNT> *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"tools\" is not an object";
        return false;
    }
    for (int m = 0; m < rmwork::MODE_COUNT; ++m) {
        const featurize::Json *mode = v.get(rmwork::mode_key(static_cast<rmwork::Mode>(m)));
        if (!mode) continue;
        if (!mode->is_object()) {
            *err = std::string("\"tools\".\"") + rmwork::mode_key(static_cast<rmwork::Mode>(m)) + "\" is not an object";
            return false;
        }
        for (int t = 0; t < rmwork::TOOL_COUNT; ++t) {
            const char *key = rmwork::tool_info(static_cast<rmwork::Tool>(t)).key;
            const featurize::Json *open = mode->get(key);
            if (!open) continue;
            if (!open->is_bool()) {
                *err = std::string("tool \"") + key + "\" is not true or false";
                return false;
            }
            (*out)[m][t] = open->as_bool();
        }
    }
    return true;
}

static bool read_background(const featurize::Json &v, rmwork::TableBackground *out, std::string *err) {
    const featurize::Json *kind = v.get("kind");
    const featurize::Json *value = v.get("value");
    if (!v.is_object() || !kind || !kind->is_string() || !value || !value->is_string()) {
        *err = "\"table_background\" is not {\"kind\": …, \"value\": …}";
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if (kind->as_string() == BACKGROUND_KINDS[k]) {
            out->kind = static_cast<rmwork::TableBackground::Kind>(k);
            out->value = value->as_string();
            return true;
        }
    }
    *err = "\"table_background\".\"kind\" is not color, texture or image";
    return false;
}

// {"family": …, "weight": …, "size": …}, each optional; a value of a known kind but unknown (a
// family not embedded, a weight the family lacks, a size out of range or fractional) is the
// default's.
static bool read_font(const featurize::Json &v, rmwork::FontChoice *out, std::string *err) {
    using namespace rmwork;
    const featurize::Json *family = v.get("family");
    const featurize::Json *weight = v.get("weight");
    const featurize::Json *size = v.get("size");
    if (!v.is_object() || (family && !family->is_string()) || (weight && !weight->is_string()) ||
        (size && !size->is_number())) {
        *err = "\"font\" is not {\"family\": \"…\", \"weight\": \"…\", \"size\": N}";
        return false;
    }
    const FontChoice def;
    *out = def;
    if (family && find_font_family(family->as_string()) >= 0) out->family = family->as_string();
    if (weight && !parse_font_weight(find_font_family(out->family), weight->as_string(), &out->weight))
        out->weight = def.weight;
    if (size && size->is_integer() && size->as_int() >= FONT_SIZE_MIN && size->as_int() <= FONT_SIZE_MAX)
        out->size = static_cast<int>(size->as_int());
    return true;
}

// `obj`.`key`, when present, as a string; `where` names `obj` in the error.
static bool read_string(
    const featurize::Json &obj, const char *key, const char *where, std::string *out, std::string *err) {
    const featurize::Json *v = obj.get(key);
    if (!v) return true;
    if (!v->is_string()) {
        *err = std::string("\"") + where + "\".\"" + key + "\" is not a string";
        return false;
    }
    *out = v->as_string();
    return true;
}

static bool read_engine(const featurize::Json &v, rmwork::EngineSettings *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"engine\" is not an object";
        return false;
    }
    return read_string(v, "binary", "engine", &out->binary, err) &&
           read_string(v, "workdir", "engine", &out->workdir, err);
}

static bool read_models(const featurize::Json &v, rmwork::ModelSettings *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"models\" is not an object";
        return false;
    }
    if (!read_string(v, "socket", "models", &out->socket, err) ||
        !read_string(v, "run_dir", "models", &out->run_dir, err))
        return false;
    const featurize::Json *policies = v.get("policies");
    if (!policies) return true;
    if (!policies->is_array()) {
        *err = "\"models\".\"policies\" is not a list";
        return false;
    }
    out->policies.clear();
    for (const featurize::Json &row : policies->items()) {
        const featurize::Json *id = row.get("id");
        const featurize::Json *model = row.get("model");
        if (!row.is_object() || !id || !id->is_string() || !model || !model->is_string()) {
            *err = "a policy is not {\"id\": \"…\", \"model\": \"…\"}";
            return false;
        }
        std::string why;
        if (!rmwork::add_policy(out, id->as_string(), model->as_string(), &why)) {
            *err = "policy \"" + id->as_string() + "\": " + why;
            return false;
        }
    }
    return true;
}

// {"own": [step, …], "opponent": [step, …]}: the steps stopped in, by key.
static bool read_stops(const featurize::Json &v, rmwork::StopSettings *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"stops\" is not an object";
        return false;
    }
    for (int turn = 0; turn < 2; ++turn) {
        const featurize::Json *list = v.get(STOP_TURNS[turn]);
        if (!list) continue;
        if (!list->is_array()) {
            *err = std::string("\"stops\".\"") + STOP_TURNS[turn] + "\" is not a list of steps";
            return false;
        }
        std::array<bool, rmwork::STOP_STEP_COUNT> &row = turn == 0 ? out->own : out->opponent;
        row.fill(false);
        for (const featurize::Json &step : list->items()) {
            const int i = step.is_string() ? rmwork::find_stop_step(step.as_string()) : -1;
            if (i < 0) {
                *err = std::string("\"stops\".\"") + STOP_TURNS[turn] + "\" holds " +
                       (step.is_string() ? "\"" + step.as_string() + "\"" : std::string("a non-string")) +
                       ", not a step such as \"upkeep\" or \"declare_attackers\"";
                return false;
            }
            row[static_cast<size_t>(i)] = true;
        }
    }
    return true;
}

static bool read_images(const featurize::Json &v, std::string *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"images\" is not an object";
        return false;
    }
    return read_string(v, "cache", "images", out, err);
}

// {"tabs": [tool, …], "selected": i} or {"split": "row" | "col", "sizes": [f, …], "children": […]}.
static bool read_recent(const featurize::Json &v, std::vector<std::string> *out, std::string *err) {
    if (!v.is_array()) {
        *err = "\"recent\" is not a list of paths";
        return false;
    }
    out->clear();
    for (const featurize::Json &p : v.items()) {
        if (!p.is_string()) {
            *err = "\"recent\" holds something other than a path";
            return false;
        }
        if (out->size() < rmwork::RECENT_MAX && !p.as_string().empty()) out->push_back(p.as_string());
    }
    return true;
}

static bool read_layout_node(const featurize::Json &v, int depth, rmwork::LayoutNode *out, std::string *err) {
    using rmwork::LayoutNode;
    if (!v.is_object() || depth > 16) {
        *err = "a layout node is not an object";
        return false;
    }
    if (const featurize::Json *tabs = v.get("tabs")) {
        out->kind = LayoutNode::Kind::GROUP;
        if (!tabs->is_array()) {
            *err = "a layout's \"tabs\" is not a list of tools";
            return false;
        }
        for (const featurize::Json &tab : tabs->items()) {
            int found = -1;
            for (int t = 0; t < rmwork::TOOL_COUNT && tab.is_string(); ++t)
                if (tab.as_string() == rmwork::tool_info(static_cast<rmwork::Tool>(t)).key) found = t;
            if (found < 0) {
                *err = "a layout's tab is not a tool such as \"table\" or \"log\"";
                return false;
            }
            out->tools.push_back(static_cast<rmwork::Tool>(found));
        }
        const featurize::Json *selected = v.get("selected");
        if (selected && !selected->is_integer()) {
            *err = "a layout's \"selected\" is not a tab index";
            return false;
        }
        out->selected = selected ? static_cast<int>(selected->as_int()) : 0;
        return true;
    }
    const featurize::Json *split = v.get("split");
    const featurize::Json *sizes = v.get("sizes");
    const featurize::Json *children = v.get("children");
    if (!split || !split->is_string() || (split->as_string() != "row" && split->as_string() != "col") || !sizes ||
        !sizes->is_array() || !children || !children->is_array()) {
        *err =
            "a layout node is neither {\"tabs\": …} nor {\"split\": \"row\" | \"col\", \"sizes\": …, \"children\": …}";
        return false;
    }
    out->kind = split->as_string() == "row" ? LayoutNode::Kind::ROW : LayoutNode::Kind::COL;
    for (const featurize::Json &s : sizes->items()) {
        if (!s.is_number()) {
            *err = "a layout's \"sizes\" holds a non-number";
            return false;
        }
        out->sizes.push_back(static_cast<float>(s.as_number()));
    }
    for (const featurize::Json &c : children->items()) {
        out->children.emplace_back();
        if (!read_layout_node(c, depth + 1, &out->children.back(), err)) return false;
    }
    return true;
}

// {"play": {name: node, …}, "replay": {…}}.
static bool read_layouts(
    const featurize::Json &v, std::array<rmwork::NamedLayouts, rmwork::MODE_COUNT> *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"layouts\" is not an object";
        return false;
    }
    for (int m = 0; m < rmwork::MODE_COUNT; ++m) {
        const char *mode = rmwork::mode_key(static_cast<rmwork::Mode>(m));
        const featurize::Json *named = v.get(mode);
        if (!named) continue;
        if (!named->is_object()) {
            *err = std::string("\"layouts\".\"") + mode + "\" is not an object of named layouts";
            return false;
        }
        for (const auto &member : named->members()) {
            rmwork::NamedLayout layout;
            layout.name = member.first;
            std::string why;
            if (!rmwork::layout_name_valid(layout.name, &why) ||
                rmwork::find_named_layout((*out)[m], layout.name) >= 0) {
                *err = std::string("layout \"") + layout.name + "\" (" + mode +
                       "): " + (why.empty() ? std::string("named twice") : why);
                return false;
            }
            if (!read_layout_node(member.second, 0, &layout.root, err) || !rmwork::layout_valid(layout.root, &why)) {
                *err = std::string("layout \"") + layout.name + "\" (" + mode + "): " + (why.empty() ? *err : why);
                return false;
            }
            (*out)[m].push_back(std::move(layout));
        }
    }
    return true;
}

static std::string layout_json(const rmwork::LayoutNode &n) {
    using featurize::json_quote;
    if (n.kind == rmwork::LayoutNode::Kind::GROUP) {
        std::string out = "{\"tabs\": [";
        for (size_t i = 0; i < n.tools.size(); ++i)
            out += std::string(i ? ", " : "") + json_quote(rmwork::tool_info(n.tools[i]).key);
        return out + "], \"selected\": " + std::to_string(n.selected) + "}";
    }
    std::string out =
        std::string("{\"split\": \"") + (n.kind == rmwork::LayoutNode::Kind::ROW ? "row" : "col") + "\", \"sizes\": [";
    for (size_t i = 0; i < n.sizes.size(); ++i) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%s%.4f", i ? ", " : "", static_cast<double>(n.sizes[i]));
        out += buf;
    }
    out += "], \"children\": [";
    for (size_t i = 0; i < n.children.size(); ++i) out += std::string(i ? ", " : "") + layout_json(n.children[i]);
    return out + "]}";
}

static std::string stops_json(const std::array<bool, rmwork::STOP_STEP_COUNT> &stops) {
    std::string out = "[";
    bool first = true;
    for (int i = 0; i < rmwork::STOP_STEP_COUNT; ++i) {
        if (!stops[static_cast<size_t>(i)]) continue;
        out += std::string(first ? "" : ", ") + featurize::json_quote(rmwork::stop_step(i).key);
        first = false;
    }
    return out + "]";
}

static bool read_keys(const featurize::Json &v, rmwork::KeyBindings *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"keys\" is not an object";
        return false;
    }
    for (const auto &member : v.members()) {
        const int a = rmwork::find_action(member.first);
        if (a < 0) continue;
        ImGuiKeyChord chord = 0;
        if (!member.second.is_string() || !rmwork::parse_chord(member.second.as_string(), &chord)) {
            *err = "key binding \"" + member.first + "\" is not a key such as \"Ctrl+O\" or \"RightArrow\"";
            return false;
        }
        (*out)[a] = chord;
    }
    for (int a = 0; a < rmwork::ACTION_COUNT; ++a) {
        const int other = rmwork::binding_conflict(*out, static_cast<rmwork::Action>(a), (*out)[a]);
        if (other >= 0) {
            *err = std::string("key bindings \"") + rmwork::action_info(static_cast<rmwork::Action>(a)).key +
                   "\" and \"" + rmwork::action_info(static_cast<rmwork::Action>(other)).key + "\" share " +
                   rmwork::chord_name((*out)[a]) + " where both work";
            return false;
        }
    }
    return true;
}

namespace rmwork {

const TextureInfo TABLE_TEXTURES[TEXTURE_COUNT] = {
    {"wood", "Wood"}, {"felt", "Felt"}, {"linen", "Linen"}, {"tiles", "Tiles"}, {"slate", "Slate"}};

const StepInfo &stop_step(int index) {
    return STOP_STEPS[index];
}

int find_stop_step(const std::string &key) {
    for (int i = 0; i < STOP_STEP_COUNT; ++i)
        if (key == STOP_STEPS[i].key) return i;
    return -1;
}

StopSettings default_stops() {
    StopSettings s;
    for (int i : DEFAULT_STOPS) s.own[static_cast<size_t>(i)] = s.opponent[static_cast<size_t>(i)] = true;
    return s;
}

std::string engine_binary_path(const Settings &s) {
    return resolve_path(s.engine.binary, default_engine_binary());
}

std::string engine_workdir_path(const Settings &s) {
    return resolve_path(s.engine.workdir, default_engine_workdir());
}

std::string run_dir_path(const Settings &s) {
    return resolve_path(s.models.run_dir, default_run_dir());
}

std::string image_cache_path(const Settings &s) {
    return resolve_path(s.image_cache, default_image_cache());
}

std::string socket_path(const Settings &s) {
    return resolve_path(s.models.socket, "");
}

std::string policy_model_path(const PolicyEntry &p) {
    return resolve_path(p.model, "");
}

bool policy_id_valid(const std::string &id, std::string *why) {
    if (id.empty()) {
        *why = "the id is empty";
        return false;
    }
    for (char c : id) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-' || c == '.';
        if (!ok) {
            *why = "an id is letters, digits, '_', '-' and '.'";
            return false;
        }
    }
    return true;
}

bool add_policy(ModelSettings *m, const std::string &id, const std::string &model, std::string *why) {
    if (!policy_id_valid(id, why)) return false;
    for (const PolicyEntry &p : m->policies) {
        if (p.id == id) {
            *why = "another policy has the id \"" + id + "\"";
            return false;
        }
    }
    if (model.empty()) {
        *why = "the model path is empty";
        return false;
    }
    m->policies.push_back({id, model});
    return true;
}

void add_recent(std::vector<std::string> *recent, const std::string &path) {
    for (size_t i = 0; i < recent->size(); ++i) {
        if ((*recent)[i] != path) continue;
        recent->erase(recent->begin() + static_cast<long>(i));
        break;
    }
    recent->insert(recent->begin(), path);
    if (recent->size() > RECENT_MAX) recent->resize(RECENT_MAX);
}

Settings default_settings() {
    Settings s;
    for (int m = 0; m < MODE_COUNT; ++m)
        s.tools[m] = layout_tools(presets()[default_preset(static_cast<Mode>(m))].root);
    s.keys = default_bindings();
    return s;
}

ConfigPaths config_paths(const std::string &dir) {
    ConfigPaths p;
    if (!dir.empty()) {
        p.dir = dir;
    } else {
        const char *xdg = std::getenv("XDG_CONFIG_HOME");
        const char *home = std::getenv("HOME");
        if (xdg && *xdg)
            p.dir = std::string(xdg) + "/robomage";
        else
            p.dir = std::string(home && *home ? home : ".") + "/.config/robomage";
    }
    p.settings = p.dir + "/settings.json";
    p.imgui_ini = p.dir + "/workbench/imgui.ini";
    return p;
}

std::string settings_to_json(const Settings &s) {
    using featurize::json_quote;
    std::string out = "{\n  \"version\": 1,\n";
    out += "  \"mode\": " + json_quote(mode_key(s.mode)) + ",\n";
    out += "  \"tools\": {\n";
    for (int m = 0; m < MODE_COUNT; ++m) {
        out += "    " + json_quote(mode_key(static_cast<Mode>(m))) + ": {";
        for (int t = 0; t < TOOL_COUNT; ++t) {
            out += std::string(t ? ", " : "") + json_quote(tool_info(static_cast<Tool>(t)).key) + ": " +
                   (s.tools[m][t] ? "true" : "false");
        }
        out += m + 1 < MODE_COUNT ? "},\n" : "}\n";
    }
    out += "  },\n";
    out += "  \"table_background\": {\"kind\": " + json_quote(background_kind_key(s.table.kind)) +
           ", \"value\": " + json_quote(s.table.value) + "},\n";
    out += "  \"theme\": " + json_quote(find_theme(s.theme) >= 0 ? s.theme : DEFAULT_THEME) + ",\n";
    const FontChoice font = sanitize_font_choice(s.font);
    out += "  \"font\": {\"family\": " + json_quote(font.family) +
           ", \"weight\": " + json_quote(font_weight_key(find_font_family(font.family), font.weight)) +
           ", \"size\": " + std::to_string(font.size) + "},\n";
    out += "  \"keys\": {\n";
    for (int a = 0; a < ACTION_COUNT; ++a) {
        out += "    " + json_quote(action_info(static_cast<Action>(a)).key) + ": " + json_quote(chord_name(s.keys[a])) +
               (a + 1 < ACTION_COUNT ? ",\n" : "\n");
    }
    out += "  },\n";
    out += "  \"engine\": {\"binary\": " + json_quote(s.engine.binary) +
           ", \"workdir\": " + json_quote(s.engine.workdir) + "},\n";
    out += "  \"models\": {\"socket\": " + json_quote(s.models.socket) +
           ", \"run_dir\": " + json_quote(s.models.run_dir) + ", \"policies\": [";
    for (size_t i = 0; i < s.models.policies.size(); ++i) {
        const PolicyEntry &p = s.models.policies[i];
        out +=
            std::string(i ? ", " : "") + "{\"id\": " + json_quote(p.id) + ", \"model\": " + json_quote(p.model) + "}";
    }
    out += "]},\n";
    out += "  \"stops\": {\"own\": " + stops_json(s.stops.own) + ", \"opponent\": " + stops_json(s.stops.opponent) +
           "},\n";
    out += "  \"images\": {\"cache\": " + json_quote(s.image_cache) + "},\n";
    out += "  \"recent\": [";
    for (size_t i = 0; i < s.recent.size(); ++i) out += std::string(i ? ", " : "") + json_quote(s.recent[i]);
    out += "],\n";
    out += "  \"layouts\": {";
    for (int m = 0; m < MODE_COUNT; ++m) {
        out += std::string(m ? ",\n" : "\n") + "    " + json_quote(mode_key(static_cast<Mode>(m))) + ": {";
        const NamedLayouts &named = s.layouts[m];
        for (size_t i = 0; i < named.size(); ++i)
            out +=
                std::string(i ? "," : "") + "\n      " + json_quote(named[i].name) + ": " + layout_json(named[i].root);
        out += named.empty() ? "}" : "\n    }";
    }
    out += "\n  }\n}\n";
    return out;
}

bool settings_from_json(const std::string &text, Settings *out, std::string *err) {
    *out = default_settings();
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) return false;
    if (!doc.is_object()) {
        *err = "not a JSON object";
        return false;
    }
    if (const featurize::Json *v = doc.get("mode"))
        if (!read_mode(*v, &out->mode, err)) return false;
    if (const featurize::Json *v = doc.get("tools"))
        if (!read_tools(*v, &out->tools, err)) return false;
    if (const featurize::Json *v = doc.get("table_background"))
        if (!read_background(*v, &out->table, err)) return false;
    if (const featurize::Json *v = doc.get("theme")) {
        if (!v->is_string()) {
            *err = "\"theme\" is not a theme name such as \"imhex-dark\"";
            return false;
        }
        if (find_theme(v->as_string()) >= 0) out->theme = v->as_string();
    }
    if (const featurize::Json *v = doc.get("font"))
        if (!read_font(*v, &out->font, err)) return false;
    if (const featurize::Json *v = doc.get("keys"))
        if (!read_keys(*v, &out->keys, err)) return false;
    if (const featurize::Json *v = doc.get("engine"))
        if (!read_engine(*v, &out->engine, err)) return false;
    if (const featurize::Json *v = doc.get("models"))
        if (!read_models(*v, &out->models, err)) return false;
    if (const featurize::Json *v = doc.get("stops"))
        if (!read_stops(*v, &out->stops, err)) return false;
    if (const featurize::Json *v = doc.get("images"))
        if (!read_images(*v, &out->image_cache, err)) return false;
    if (const featurize::Json *v = doc.get("layouts"))
        if (!read_layouts(*v, &out->layouts, err)) return false;
    if (const featurize::Json *v = doc.get("recent"))
        if (!read_recent(*v, &out->recent, err)) return false;
    return true;
}

bool load_settings(const std::string &path, Settings *out, std::string *err) {
    *out = default_settings();
    if (!file_exists(path)) return true;
    std::string text;
    if (!read_file(path, &text)) {
        *err = "cannot read " + path;
        return false;
    }
    std::string why;
    if (!settings_from_json(text, out, &why)) {
        *out = default_settings();
        *err = path + ": " + why;
        return false;
    }
    return true;
}

bool save_settings(const std::string &path, const Settings &s, std::string *err) {
    const std::string dir = parent_dir(path);
    if (!dir.empty() && !make_dirs(dir)) {
        *err = "cannot create " + dir;
        return false;
    }
    return write_file_atomic(path, settings_to_json(s), err);
}

}  // namespace rmwork
