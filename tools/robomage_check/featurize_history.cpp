#include "featurize_history.h"

#include <algorithm>
#include <cmath>

#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/frame_stream.h"
#include "corpus.h"
#include "proc.h"
#include "traj.h"
#include "util.h"

using featurize::Featurization;
using featurize::Tensors;

namespace {

// The history cap the part builds the spec with.
const uint32_t HISTORY_EVENTS = 32;
const size_t MAX_FAILS = 20;

// One pass over a dump: per seat decision (a seat frame with a menu), in dump order, its seat and
// its tensors with the history on and off.
struct Pass {
        std::vector<int> seat;
        std::vector<Tensors> on, off;
};

}  // namespace

static bool featurize_dump(
    const Featurization &on, const Featurization &off, const std::string &dump, Pass *out, std::string *err);
static std::string check_spec(const rmcheck::Options &opt, const Featurization &on, const Featurization &off,
    const std::string &cards, const std::string &text);
static std::string check_prefix(const Featurization &on, const Featurization &off, const Tensors &a, const Tensors &b);
static std::string check_history_tokens(const featurize::FeatureSpec &spec, const Tensors &t);
static void add(std::vector<std::string> *fails, const std::string &m);
static void featurize_history_game(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &on,
    const Featurization &off, const std::string &cards, const std::string &text);

static void add(std::vector<std::string> *fails, const std::string &m) {
    if (fails->size() < MAX_FAILS) fails->push_back(m);
}

static bool featurize_dump(
    const Featurization &on, const Featurization &off, const std::string &dump, Pass *out, std::string *err) {
    featurize::EventHistory history[2];
    const uint16_t menu_table = on.spec.types[static_cast<size_t>(on.spec.menu_type)].table;
    return featurize::read_stream(
        dump,
        [&](const featurize::ViewerFrames &, const rmstate::DecodedFrame &state, std::string *e) {
            const uint8_t viewer = state.header.viewer;
            if (viewer != static_cast<uint8_t>(rmstate::Viewer::A) &&
                viewer != static_cast<uint8_t>(rmstate::Viewer::B))
                return true;
            history[viewer].observe(on.spec, state);
            const rmstate::DecodedTable *menu = state.table(menu_table);
            if (!menu || menu->rows.empty()) return true;
            out->seat.push_back(viewer);
            out->on.emplace_back();
            out->off.emplace_back();
            return featurize::featurize_seat(on.spec, on.tables, state, &history[viewer], &out->on.back(), e) &&
                   featurize::featurize_seat(off.spec, off.tables, state, &out->off.back(), e);
        },
        err);
}

// "" when the specs and rmfeat's agree with the header's rules.
static std::string check_spec(const rmcheck::Options &opt, const Featurization &on, const Featurization &off,
    const std::string &cards, const std::string &text) {
    if (off.spec.history.on() || off.spec.json.find("\"history\"") != std::string::npos)
        return "the spec built without options has history";
    for (const featurize::TokenType &t : off.spec.types)
        if (t.kind == featurize::TypeKind::HISTORY) return "the spec built without options has a history type";
    if (!on.spec.history.on() || on.spec.hash == off.spec.hash) return "the history spec has the off spec's hash";
    if (on.spec.types.size() != off.spec.types.size() + 2) return "the history spec does not add two token types";
    for (size_t i = 0; i < off.spec.types.size(); ++i)
        if (on.spec.types[i].name != off.spec.types[i].name) return "the history spec reorders the off spec's types";
    rmcheck::ProcSpec spec;
    spec.argv = {
        opt.rmfeat, "--spec", "--cards", cards, "--text-table", text, "--history", std::to_string(HISTORY_EVENTS)};
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    if (!r.ok() || r.out != on.spec.json + "\n") return "rmfeat --spec --history differs (" + r.status_text() + ")";
    return "";
}

// "" when `a` (history on) less its history tokens is `b` (off): the same tokens first, every slot of
// them (the categorical width may grow, its extra slots 0), the same relations first, the same menu.
static std::string check_prefix(const Featurization &on, const Featurization &off, const Tensors &a, const Tensors &b) {
    const featurize::FeatureSpec &so = on.spec, &sf = off.spec;
    const size_t n = b.n();
    if (a.n() < n || a.r() < b.r() || a.act_tok != b.act_tok || a.act_mask != b.act_mask) return "the token counts";
    for (size_t i = 0; i < a.n(); ++i) {
        if (i < n && a.tok_type[i] != b.tok_type[i]) return "tok_type";
        if (i >= n && on.spec.types[static_cast<size_t>(a.tok_type[i])].kind != featurize::TypeKind::HISTORY)
            return "a non-history token after the off tokens";
    }
    if (so.F != sf.F || so.K != sf.K || so.Q != sf.Q || so.C < sf.C) return "the widths";
    for (size_t i = 0; i < n; ++i) {
        for (size_t c = 0; c < so.C; ++c)
            if (a.tok_cat[i * so.C + c] != (c < sf.C ? b.tok_cat[i * sf.C + c] : 0)) return "tok_cat";
    }
    if (!std::equal(b.tok_num.begin(), b.tok_num.end(), a.tok_num.begin())) return "tok_num";
    if (!std::equal(b.tok_card.begin(), b.tok_card.end(), a.tok_card.begin())) return "tok_card";
    if (!std::equal(b.tok_ability.begin(), b.tok_ability.end(), a.tok_ability.begin())) return "tok_ability";
    if (!std::equal(b.rel.begin(), b.rel.end(), a.rel.begin())) return "rel";
    return "";
}

// "" when the history tokens keep the spec's caps and slots and every card token has one relation
// to an event token.
static std::string check_history_tokens(const featurize::FeatureSpec &spec, const Tensors &t) {
    const featurize::HistoryPlan &h = spec.history;
    const featurize::TokenType &ct = spec.types[static_cast<size_t>(h.card_type)];
    int32_t event_rel = -1;
    for (const int32_t r : ct.relations)
        if (spec.relations[static_cast<size_t>(r)].target == "hist_event") event_rel = r;
    std::vector<size_t> rank(t.n(), 0), cards_of(t.n(), 0), links(t.n(), 0);
    size_t events = 0;
    for (size_t i = 0; i < t.n(); ++i)
        if (t.tok_type[i] == h.event_type) {
            const float *num = &t.tok_num[i * spec.F];
            if (num[h.offset_slot] != 0.0f && num[h.offset_slot] != 1.0f) return "a turn offset other than 0 or 1";
            if (std::fabs(num[h.age_slot] - static_cast<float>(std::log1p(static_cast<double>(events)))) > 1e-6f)
                return "an event's age is not its rank";
            rank[i] = ++events;
        }
    if (events > h.events) return "more event tokens than the cap";
    for (size_t r = 0; r < t.r(); ++r) {
        const int64_t src = t.rel[3 * r], dst = t.rel[3 * r + 1], type = t.rel[3 * r + 2];
        if (type != event_rel) continue;
        if (t.tok_type[static_cast<size_t>(src)] != h.card_type || rank[static_cast<size_t>(dst)] == 0)
            return "an event relation not from a card token to an event token";
        ++links[static_cast<size_t>(src)];
        if (++cards_of[static_cast<size_t>(dst)] > h.cards_per_event) return "more card tokens than the cap";
    }
    for (size_t i = 0; i < t.n(); ++i)
        if (t.tok_type[i] == h.card_type && links[i] != 1) return "a card token without exactly one event";
    return "";
}

// One corpus game replayed with a dump and the history's episodes; the checks of the header.
static void featurize_history_game(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &on,
    const Featurization &off, const std::string &cards, const std::string &text) {
    const std::vector<std::string> names = rmcheck::corpus_names(opt);
    if (names.empty()) {
        rep.error("featurize", "history: no corpus game");
        return;
    }
    const std::string dir = rmcheck::path_join(opt.out("featurize"), "history");
    const std::string dump = rmcheck::path_join(dir, "game.rms"), eps_dir = rmcheck::path_join(dir, "episodes");
    rmcheck::remove_tree(dir);
    rmcheck::make_dirs(eps_dir);
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--replay", rmcheck::corpus_log(opt, names[0]), "--machine", "--rmstate-dump", dump,
        "--trajectory-out", eps_dir, "--feature-history", std::to_string(HISTORY_EVENTS)};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 900;
    const rmcheck::ProcResult pr = rmcheck::run_process(spec);
    if (!pr.ok()) {
        rep.error("featurize", "history: replay " + names[0] + " " + pr.status_text());
        return;
    }
    std::vector<std::string> fails;
    Pass a, b;
    std::string err;
    if (!featurize_dump(on, off, dump, &a, &err) || !featurize_dump(on, off, dump, &b, &err)) {
        rep.error("featurize", "history: " + err);
        return;
    }
    size_t tokens = 0;
    for (size_t d = 0; d < a.on.size(); ++d) {
        const std::string at = "history: " + names[0] + " decision " + std::to_string(d + 1) + ": ";
        if (!featurize::same_tensors(a.on[d], b.on[d])) add(&fails, at + "featurizing twice gives other tensors");
        const std::string p = check_prefix(on, off, a.on[d], a.off[d]);
        if (!p.empty()) add(&fails, at + p + " of the off tensors differ");
        const std::string h = check_history_tokens(on.spec, a.on[d]);
        if (!h.empty()) add(&fails, at + h);
        tokens += a.on[d].n() - a.off[d].n();
    }
    std::vector<rmcheck::EpisodeFile> eps;
    if (!rmcheck::read_episodes(eps_dir, &eps, &err)) add(&fails, "history: " + err);
    size_t d = 0;
    for (const rmcheck::EpisodeFile &e : eps)
        for (const featurize::EpisodeStep &s : e.ep.steps) {
            if (d >= a.on.size() || a.seat[d] != s.seat) {
                add(&fails, "history: " + e.name + ": the steps are not the dump's seat decisions");
                break;
            }
            const std::string diff = rmcheck::compare_tensors(s.tensors, a.on[d++]);
            if (!diff.empty())
                add(&fails, "history: " + e.name + ": the episode's " + diff + " differs from the dump's");
        }
    if (d != a.on.size()) add(&fails, "history: the episodes hold " + std::to_string(d) + " of the dump's steps");
    std::vector<Tensors> seat_a;
    for (size_t i = 0; i < a.on.size(); ++i)
        if (a.seat[i] == 0) seat_a.push_back(a.on[i]);
    const std::string out = rmcheck::path_join(dir, "tensors_A.rmtb");
    spec = rmcheck::ProcSpec();
    spec.argv = {opt.rmfeat, "--featurize", dump, "--viewer", "A", "--out", out, "--cards", cards, "--text-table", text,
        "--history", std::to_string(HISTORY_EVENTS)};
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    std::string bytes;
    rmcheck::read_file(out, &bytes);
    if (!r.ok() || bytes != featurize::rmtb_encode(featurize::tensors_file(on.spec, seat_a)))
        add(&fails,
            "history: rmfeat --featurize --history differs from the in-process tensors (" + r.status_text() + ")");
    for (const std::string &m : fails) rep.error("featurize", m);
    if (fails.empty()) {
        const double n = static_cast<double>(std::max<size_t>(a.on.size(), 1));
        rep.info("OK   history: spec " + featurize::hex16(on.spec.hash) + "; " + names[0] + ": " +
                 std::to_string(a.on.size()) + " seat decisions deterministic, the off tensors plus " +
                 rmcheck::fmt1(static_cast<double>(tokens) / n) +
                 " history tokens each; the episodes and rmfeat's the same");
        rmcheck::remove_tree(dir);
    }
}

namespace rmcheck {

void featurize_history_part(const Options &opt, Report &rep, const Featurization &off, const std::string &cards) {
    const std::string text = path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb");
    Featurization on;
    on.options.history_events = HISTORY_EVENTS;
    std::string err;
    if (!on.load(cards, text, "", &err)) {
        rep.error("featurize", "history: the featurizer's inputs: " + err);
        return;
    }
    const std::string s = check_spec(opt, on, off, cards, text);
    if (!s.empty()) {
        rep.error("featurize", "history: " + s);
        return;
    }
    featurize_history_game(opt, rep, on, off, cards, text);
}

}  // namespace rmcheck
