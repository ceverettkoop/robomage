#include "decision_frame.h"

static std::string base_name(const std::string &path);

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

namespace rmwork {

bool load_decision_frame(
    const Match &m, size_t index, int view, const CardBook &book, DecisionFrame *out, std::string *err) {
    *out = DecisionFrame();
    if (index >= m.decisions.size()) {
        *err = "no decision " + std::to_string(index + 1);
        return false;
    }
    out->frame = m.frame(index, view, err);
    if (!out->frame) return false;
    out->match = &m;
    out->index = index;
    out->view = view;
    out->info = &m.decisions[index];
    out->stream = &m.stream(view);
    out->book = &book;
    if (out->info->episode_step >= 0 && static_cast<size_t>(out->info->episode_step) < m.episode_steps.size())
        out->episode = &m.episode_steps[static_cast<size_t>(out->info->episode_step)];
    return true;
}

MenuModel build_menu_model(const DecisionFrame &f) {
    MenuModel m;
    m.decider = f.info ? f.info->decider : -1;
    if (!f.frame || !f.sees_decision()) return m;
    m.shown = true;
    const agents::Table t(*f.frame, f.stream->schema(), "menu");
    const int c_desc = t.col("description"), c_cat = t.col("category"), c_src = t.col("source"),
              c_tgt = t.col("target"), c_sp = t.col("source_player"), c_tp = t.col("target_player");
    const auto seat = [](uint64_t s) { return s == 1 ? 0 : s == 2 ? 1 : -1; };
    for (size_t r = 0; r < t.rows(); ++r) {
        MenuRow row;
        row.text = t.text(r, c_desc);
        row.category = f.stream->enum_name("action_category", t.u(r, c_cat));
        row.link = {t.u(r, c_src), t.u(r, c_tgt), seat(t.u(r, c_sp)), seat(t.u(r, c_tp))};
        m.rows.push_back(row);
    }
    if (f.info && f.info->reply >= 0 && static_cast<size_t>(f.info->reply) < m.rows.size()) m.chosen = f.info->reply;
    return m;
}

FrameModels build_frame_models(const DecisionFrame &f) {
    FrameModels out;
    if (!f.frame) return out;
    const DecisionInfo *d = f.info;
    out.table =
        build_table_model(*f.stream, *f.book, *f.frame, f.view, d ? d->decider : -1, d && d->phase == "Opening hand");
    out.zones = build_zone_model(*f.stream, *f.book, *f.frame, f.view);
    out.menu = build_menu_model(f);
    return out;
}

const char *seat_name(int seat) {
    return seat == 0 ? "Player A" : seat == 1 ? "Player B" : "no player";
}

std::string agent_label(const Match &m, int seat) {
    if (seat < 0 || seat > 1) return "";
    const std::string &a = m.agents[static_cast<size_t>(seat)];
    const size_t colon = a.find(':');
    if (colon == std::string::npos || a.find('/', colon) == std::string::npos) return a;
    // "nn:/path/to/current.ts.pt:temp=…" -> "nn:current.ts.pt:temp=…"
    const std::string kind = a.substr(0, colon + 1);
    std::string rest = a.substr(colon + 1);
    const size_t opts = rest.find(':');
    const std::string path = opts == std::string::npos ? rest : rest.substr(0, opts);
    return kind + base_name(path) + (opts == std::string::npos ? "" : rest.substr(opts));
}

}  // namespace rmwork
