#include "knowledge_meta.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>

#include "frames.h"
#include "json.h"
#include "proc.h"
#include "scan.h"
#include "sessions.h"
#include "util.h"

using rmcheck::Rows;
using rmcheck::StreamGame;

// archetypes.json: its labels (`_archetypes`) and deck stem → label.
struct Tags {
        std::vector<std::string> labels;
        std::map<std::string, std::string> decks;
        std::string error;
};

// A stream's first METAGAME frame.
struct Metagame {
        bool found = false;
        std::vector<std::string> names, list_labels, labels;  // lists, each list's label, the labels
        std::vector<std::map<uint64_t, uint64_t>> counts;     // per list: card → main + side copies
        std::vector<std::pair<uint64_t, uint64_t>> sizes;     // per list: (main, side)
};

// One seat frame of a calibration session.
struct CalibrationRecord {
        std::string session;
        int seat;
        uint64_t game;
        std::vector<double> probs;  // the opponent's list probabilities
        size_t truth;               // the opponent's true list
        std::vector<std::string> labels;
        bool exact;
};

static std::string normalize_deck(const std::string &deck);
static Tags load_tags(const rmcheck::Options &opt);
static Metagame metagame_of(const std::vector<std::string> &payloads);
static std::vector<std::string> check_metagame(const Metagame &m, const Tags &tags, const std::string &label);
static std::vector<std::string> archetype_session(
    const rmcheck::Options &opt, const StreamGame &g, const Tags &tags, const std::string &dir, std::string *summary);
static std::string list_text(const std::vector<size_t> &v);
static std::vector<std::string> calibration_session(
    const rmcheck::Options &opt, const StreamGame &g, const std::string &dir, std::vector<CalibrationRecord> *records);
static void calibration_report(rmcheck::Report &rep, const std::vector<CalibrationRecord> &records);

// archetypes.py's normalize_deck: a deck spec as its lower-case decks/-relative stem.
static std::string normalize_deck(const std::string &deck) {
    std::string s = rmcheck::trim(deck);
    std::replace(s.begin(), s.end(), '\\', '/');
    std::string lower = s;
    for (char &c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const size_t at = lower.rfind("resources/decks/");
    if (at != std::string::npos) s = s.substr(at + 16);
    while (rmcheck::starts_with(s, "./")) s = s.substr(2);
    while (!s.empty() && s[0] == '/') s = s.substr(1);
    if (s.size() >= 3) {
        std::string tail = s.substr(s.size() - 3);
        for (char &c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tail == ".dk") s = s.substr(0, s.size() - 3);
    }
    while (!s.empty() && s.back() == '/') s.pop_back();
    while (!s.empty() && s[0] == '/') s = s.substr(1);
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static Tags load_tags(const rmcheck::Options &opt) {
    Tags t;
    const std::string path = opt.decks_dir() + "/archetypes.json";
    std::string text, err;
    rmcheck::Json doc;
    if (!rmcheck::read_file(path, &text) || !rmcheck::Json::parse(text, &doc, &err) || !doc.is_object()) {
        t.error = path + ": cannot read (" + err + ")";
        return t;
    }
    const rmcheck::Json *labels = doc.get("_archetypes");
    if (labels && labels->is_array())
        for (const rmcheck::Json &l : labels->items()) t.labels.push_back(l.as_string());
    if (t.labels.empty()) t.error = path + ": '_archetypes' must be a non-empty list";
    for (const auto &m : doc.members()) {
        if (rmcheck::starts_with(m.first, "_")) continue;
        if (std::find(t.labels.begin(), t.labels.end(), m.second.as_string()) == t.labels.end())
            t.error = path + ": deck " + m.first + " is tagged with an unknown archetype";
        t.decks[normalize_deck(m.first)] = m.second.as_string();
    }
    return t;
}

static Metagame metagame_of(const std::vector<std::string> &payloads) {
    Metagame m;
    rmcheck::FrameDecoder dec;
    std::string err;
    for (const std::string &p : payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, &err)) break;
        if (rmcheck::frame_kind(p) != rmcheck::KIND_METAGAME) continue;
        const rmstate::Schema &s = *dec.schema(rmcheck::frame_viewer(p));
        const Rows arch(f, s, "meta_archetype"), lists(f, s, "meta_list"), cards(f, s, "meta_list_card");
        for (size_t r = 0; r < arch.size(); ++r) m.labels.push_back(arch.text(r, arch.col("name")));
        for (size_t r = 0; r < lists.size(); ++r) {
            m.names.push_back(lists.text(r, lists.col("name")));
            const uint64_t a = lists.u(r, lists.col("archetype"));
            m.list_labels.push_back(a < m.labels.size() ? m.labels[static_cast<size_t>(a)] : "?");
            m.sizes.emplace_back(lists.u(r, lists.col("main")), lists.u(r, lists.col("side")));
        }
        m.counts.assign(m.names.size(), {});
        for (size_t r = 0; r < cards.size(); ++r) {
            const uint64_t l = cards.u(r, cards.col("list"));
            if (l < m.counts.size())
                m.counts[static_cast<size_t>(l)][cards.u(r, cards.col("card"))] +=
                    cards.u(r, cards.col("main")) + cards.u(r, cards.col("side"));
        }
        m.found = true;
        return m;
    }
    return m;
}

// The METAGAME frame against archetypes.json: its labels (plus unknown) and its reference lists.
static std::vector<std::string> check_metagame(const Metagame &m, const Tags &tags, const std::string &label) {
    if (!m.found) return {label + ": no METAGAME frame"};
    std::vector<std::string> fails;
    std::vector<std::string> want = tags.labels;
    want.push_back("unknown");
    if (m.labels != want)
        fails.push_back(label + ": METAGAME labels " + rmcheck::join(m.labels, ",") + ", archetypes.json has " +
                        rmcheck::join(want, ","));
    std::vector<std::string> missing;
    for (const auto &d : tags.decks)
        if (std::find(m.names.begin(), m.names.end(), d.first) == m.names.end()) missing.push_back(d.first);
    if (!missing.empty())
        fails.push_back(label + ": tagged deck(s) " + rmcheck::join(missing, ", ") +
                        " are not reference lists (missing .dk or unimplemented cards)");
    for (size_t i = 0; i < m.names.size(); ++i) {
        auto it = tags.decks.find(m.names[i]);
        if (it == tags.decks.end() || it->second != m.list_labels[i])
            fails.push_back(label + ": reference list " + m.names[i] + " has label " + m.list_labels[i] +
                            ", archetypes.json " + (it == tags.decks.end() ? "None" : it->second));
    }
    return fails;
}

static std::string list_text(const std::vector<size_t> &v) {
    std::vector<std::string> parts;
    for (size_t x : v) parts.push_back(std::to_string(x));
    return "[" + rmcheck::join(parts, ", ") + "]";
}

// One archetype session (knowledge_meta.h's archetype_part).
static std::vector<std::string> archetype_session(
    const rmcheck::Options &opt, const StreamGame &g, const Tags &tags, const std::string &dir, std::string *summary) {
    bool closed = false;
    for (const std::string &f : g.flags) closed = closed || f == "--closed-decklist";
    const std::string label = rmcheck::stream_label("archetype", g);
    const std::string err_path = rmcheck::path_join(dir, "archetype_" + std::to_string(g.seed) + ".err");
    const rmcheck::StreamRun run = rmcheck::play_stream_session(opt, g, "");
    rmcheck::write_file(err_path, run.proc.err);
    if (!run.error.empty()) return {label + ": " + run.error + " (see " + err_path + ")"};
    std::vector<std::string> fails = rmcheck::selfcheck_failures(run.proc.err, err_path, label, -1, {});
    const Metagame m = metagame_of(run.payloads);
    for (const std::string &f : check_metagame(m, tags, label)) fails.push_back(f);
    if (!m.found) return fails;
    const std::string deck[2] = {g.deck_a, g.deck_b};
    std::map<int, std::pair<uint64_t, uint64_t>> started;  // seat → the other's (library, sideboard) at game start
    std::map<int, size_t> frames, first_possible;
    std::map<int, std::pair<size_t, size_t>> exact_at;  // whose archetype → (frame of the other seat, cards seen)
    size_t unproven = 0;
    rmcheck::FrameDecoder dec;
    std::string err;
    for (const std::string &p : run.payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, &err)) {
            fails.push_back(label + ": " + err);
            break;
        }
        const int seat = rmcheck::frame_viewer(p);
        if (rmcheck::frame_kind(p) != rmcheck::KIND_STATE || (seat != rmcheck::VIEWER_A && seat != rmcheck::VIEWER_B))
            continue;
        const rmstate::Schema &s = *dec.schema(seat);
        const int other = 1 - seat;
        ++frames[seat];
        const Rows ep(f, s, "event_player"), ev(f, s, "event");
        for (size_t r = 0; r < ep.size(); ++r) {
            const uint64_t e = ep.u(r, ep.col("event"));
            if (ep.u(r, ep.col("player_rel")) == 2 && e < ev.size() &&
                ev.name(static_cast<size_t>(e), ev.col("kind")) == "GAME_START")
                started[seat] = {ep.u(r, ep.col("library")), ep.u(r, ep.col("sideboard"))};
        }
        const std::string where = label + ": " + rmcheck::viewer_name(seat) + " frame " + std::to_string(frames[seat]);
        const Rows tp(f, s, "k_template"), ka(f, s, "k_archetype"), seen_t(f, s, "k_seen");
        std::map<std::pair<uint64_t, uint64_t>, size_t> rows, arch;
        for (size_t r = 0; r < tp.size(); ++r) rows[{tp.u(r, tp.col("player_rel")), tp.u(r, tp.col("list"))}] = r;
        for (size_t r = 0; r < ka.size(); ++r) arch[{ka.u(r, ka.col("player_rel")), ka.u(r, ka.col("archetype"))}] = r;
        if (rows.size() != 2 * m.names.size() || arch.size() != 2 * m.labels.size()) {
            fails.push_back(where + ": " + std::to_string(rows.size()) + " k_template / " +
                            std::to_string(arch.size()) + " k_archetype rows");
            continue;
        }
        for (const auto &r : rows) unproven += tp.u(r.second, tp.col("exact")) ? 0 : 1;
        for (uint64_t rel : {1u, 2u}) {
            const int who = rel == 1 ? seat : other;
            const char *who_name = rmcheck::viewer_name(who);
            auto tl = std::find(m.names.begin(), m.names.end(), deck[who]);
            const bool have_true = tl != m.names.end();
            const size_t true_list = static_cast<size_t>(tl - m.names.begin());
            size_t true_label = 0;
            if (have_true)
                true_label = static_cast<size_t>(
                    std::find(m.labels.begin(), m.labels.end(), m.list_labels[true_list]) - m.labels.begin());
            std::vector<size_t> consistent, registered, exact;
            for (size_t l = 0; l < m.names.size(); ++l) {
                const size_t r = rows[{rel, l}];
                if (tp.u(r, tp.col("consistent"))) consistent.push_back(l);
                if (tp.u(r, tp.col("registered"))) registered.push_back(l);
            }
            for (size_t a = 0; a < m.labels.size(); ++a)
                if (ka.u(arch[{rel, a}], ka.col("exact"))) exact.push_back(a);
            if (have_true && std::find(consistent.begin(), consistent.end(), true_list) == consistent.end())
                fails.push_back(where + ": " + who_name + "'s true list " + deck[who] + " is reported inconsistent");
            if (!exact.empty() && !(have_true && exact == std::vector<size_t>{true_label}))
                fails.push_back(where + ": " + who_name + "'s archetype reported exactly " + m.labels[exact[0]]);
            const bool visible = rel == 1 || !closed;
            if (visible) {
                const std::vector<size_t> only = {true_list}, only_label = {true_label};
                if (!have_true || consistent != only || registered != only || exact != only_label)
                    fails.push_back(where + ": " + who_name + "'s open list: consistent " + list_text(consistent) +
                                    ", registered " + list_text(registered) + ", exact archetype " + list_text(exact));
                continue;
            }
            if (!registered.empty())
                fails.push_back(where + ": " + who_name + "'s hidden list is classified as " + list_text(registered));
            std::map<uint64_t, uint64_t> seen;
            for (size_t r = 0; r < seen_t.size(); ++r)
                if (seen_t.u(r, seen_t.col("player_rel")) == 2 && seen_t.u(r, seen_t.col("match")) > 0)
                    seen[seen_t.u(r, seen_t.col("card"))] = seen_t.u(r, seen_t.col("match"));
            std::vector<size_t> fits;
            for (size_t l = 0; l < m.names.size(); ++l) {
                bool fit = true;
                for (const auto &c : seen) {
                    auto it = m.counts[l].find(c.first);
                    fit = fit && it != m.counts[l].end() && it->second >= c.second;
                }
                if (fit) fits.push_back(l);
            }
            std::vector<size_t> extra;
            for (size_t l : consistent)
                if (std::find(fits.begin(), fits.end(), l) == fits.end()) extra.push_back(l);
            if (!extra.empty())
                fails.push_back(where + ": lists " + list_text(extra) + " are consistent but lack a seen card");
            std::set<std::string> fit_labels;
            for (size_t l : fits) fit_labels.insert(m.list_labels[l]);
            const bool true_fits = have_true && std::find(fits.begin(), fits.end(), true_list) != fits.end();
            if (fit_labels.size() == 1 && true_fits && exact != std::vector<size_t>{true_label})
                fails.push_back(where + ": the cards seen leave only " + *fit_labels.begin() + ", not reported exact");
            if (frames[seat] == 1 && seen.empty()) {
                const std::pair<uint64_t, uint64_t> start =
                    started.count(seat) ? started[seat] : std::make_pair<uint64_t, uint64_t>(0, 0);
                std::set<std::string> want, got;
                for (size_t l = 0; l < m.names.size(); ++l)
                    if (m.sizes[l].first >= start.first && m.sizes[l].second >= start.second)
                        want.insert(m.list_labels[l]);
                for (size_t a = 0; a < m.labels.size(); ++a)
                    if (ka.u(arch[{rel, a}], ka.col("possible"))) got.insert(m.labels[a]);
                if (got != want)
                    fails.push_back(where + ": before any card of " + who_name + " is seen, possible " +
                                    rmcheck::join(std::vector<std::string>(got.begin(), got.end()), ",") +
                                    ", the list sizes allow " +
                                    rmcheck::join(std::vector<std::string>(want.begin(), want.end()), ","));
                first_possible[who] = got.size();
            }
            if (!exact.empty() && !exact_at.count(who)) exact_at[who] = {frames[seat], seen.size()};
        }
    }
    if (closed)
        for (int who : {0, 1})
            if (!exact_at.count(who))
                fails.push_back(
                    label + ": " + rmcheck::viewer_name(who) + "'s archetype never became exact for the other seat");
    if (fails.empty()) {
        std::string when;
        for (const auto &e : exact_at)
            when += std::string(when.empty() ? "" : "; ") + rmcheck::viewer_name(e.first) + ": " +
                    (first_possible.count(e.first) ? std::to_string(first_possible[e.first]) : "?") +
                    " labels possible at first, exact at the other seat's frame " + std::to_string(e.second.first);
        *summary = label + ": " + std::to_string(frames[0] + frames[1]) + " seat frames, " + std::to_string(unproven) +
                   " unproven row(s)" + (when.empty() ? "" : "; " + when);
    }
    return fails;
}

// One calibration session (knowledge_meta.h's calibration_part): its records.
static std::vector<std::string> calibration_session(
    const rmcheck::Options &opt, const StreamGame &g, const std::string &dir, std::vector<CalibrationRecord> *records) {
    const std::string label = "calibration " + g.deck_a + " vs " + g.deck_b + " seed " + std::to_string(g.seed);
    const std::string err_path = rmcheck::path_join(dir, "calibration_" + std::to_string(g.seed) + ".err");
    const rmcheck::StreamRun run = rmcheck::play_stream_session(opt, g, "");
    rmcheck::write_file(err_path, run.proc.err);
    if (!run.error.empty()) return {label + ": " + run.error + " (see " + err_path + ")"};
    std::vector<std::string> fails = rmcheck::selfcheck_failures(run.proc.err, err_path, label, -1, {});
    const Metagame m = metagame_of(run.payloads);
    const std::string deck[2] = {g.deck_a, g.deck_b};
    rmcheck::FrameDecoder dec;
    std::string err;
    for (const std::string &p : run.payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, &err)) break;
        const int seat = rmcheck::frame_viewer(p);
        if (!m.found || rmcheck::frame_kind(p) != rmcheck::KIND_STATE ||
            (seat != rmcheck::VIEWER_A && seat != rmcheck::VIEWER_B))
            continue;
        const rmstate::Schema &s = *dec.schema(seat);
        if (rmcheck::header_value(f, s, "terminal") || rmcheck::header_value(f, s, "sideboarding")) continue;
        const Rows b(f, s, "b_list");
        std::vector<std::tuple<uint64_t, double, bool>> rows;  // (list, prob, exact)
        for (size_t r = 0; r < b.size(); ++r)
            if (b.u(r, b.col("player_rel")) == 2)
                rows.emplace_back(b.u(r, b.col("list")), static_cast<double>(b.u(r, b.col("prob"))) / 1e9,
                    b.u(r, b.col("exact")) != 0);
        std::sort(
            rows.begin(), rows.end(), [](const auto &x, const auto &y) { return std::get<0>(x) < std::get<0>(y); });
        if (rows.size() != m.names.size()) continue;
        auto t = std::find(m.names.begin(), m.names.end(), deck[1 - seat]);
        if (t == m.names.end()) continue;
        CalibrationRecord rec;
        rec.session = label;
        rec.seat = seat;
        rec.game = rmcheck::header_value(f, s, "game");
        rec.truth = static_cast<size_t>(t - m.names.begin());
        rec.labels = m.list_labels;
        rec.exact = true;
        for (const auto &row : rows) {
            rec.probs.push_back(std::get<1>(row));
            if (std::get<1>(row) > 0) rec.exact = rec.exact && std::get<2>(row);
        }
        records->push_back(rec);
    }
    bool any = false;
    for (const CalibrationRecord &r : *records) any = any || r.session == label;
    if (!any) fails.push_back(label + ": no seat frame with b_list rows");
    return fails;
}

// The calibration report over every session's records, and its failures (knowledge_meta.h).
static void calibration_report(rmcheck::Report &rep, const std::vector<CalibrationRecord> &records) {
    const double eps = 1e-9;
    const size_t n = records.size();
    if (!n) {
        rep.error("knowledge", "calibration: no frame");
        return;
    }
    const size_t n_lists = records[0].probs.size();
    const int bins = 10;
    double ll = 0, ll_cons = 0, ll_arch = 0, ll_arch_uniform = 0, brier = 0, brier_arch = 0;
    double amb = 0, amb_ll = 0, amb_cons = 0;
    std::vector<std::tuple<size_t, double, size_t>> rel(bins, std::make_tuple(0, 0.0, 0));
    size_t exact = 0;
    std::vector<std::tuple<std::string, int, uint64_t>> order;
    std::map<std::tuple<std::string, int, uint64_t>, std::vector<std::pair<double, double>>> games;
    for (const CalibrationRecord &r : records) {
        exact += r.exact ? 1 : 0;
        const double p_true = r.probs[r.truth];
        size_t support = 0;
        for (double p : r.probs) support += p > 0 ? 1 : 0;
        ll -= std::log(std::max(p_true, eps));
        ll_cons -= std::log(1.0 / static_cast<double>(support));
        if (support > 1) {
            amb += 1;
            amb_ll -= std::log(std::max(p_true, eps));
            amb_cons -= std::log(1.0 / static_cast<double>(support));
        }
        std::map<std::string, double> arch;
        for (size_t i = 0; i < r.probs.size(); ++i) arch[r.labels[i]] += r.probs[i];
        const std::string &tl = r.labels[r.truth];
        ll_arch -= std::log(std::max(arch[tl], eps));
        ll_arch_uniform -= std::log(static_cast<double>(std::count(r.labels.begin(), r.labels.end(), tl)) /
                                    static_cast<double>(r.labels.size()));
        for (size_t i = 0; i < r.probs.size(); ++i) brier += std::pow(r.probs[i] - (i == r.truth ? 1.0 : 0.0), 2);
        for (const std::string &a : std::set<std::string>(r.labels.begin(), r.labels.end()))
            brier_arch += std::pow(arch[a] - (a == tl ? 1.0 : 0.0), 2);
        for (size_t i = 0; i < r.probs.size(); ++i) {
            const int b = std::min(bins - 1, static_cast<int>(r.probs[i] * bins));
            std::get<0>(rel[static_cast<size_t>(b)]) += 1;
            std::get<1>(rel[static_cast<size_t>(b)]) += r.probs[i];
            std::get<2>(rel[static_cast<size_t>(b)]) += i == r.truth ? 1 : 0;
        }
        const auto key = std::make_tuple(r.session, r.seat, r.game);
        if (!games.count(key)) order.push_back(key);
        games[key].emplace_back(p_true, arch[tl]);
    }
    const double dn = static_cast<double>(n);
    double first_list = 0, last_list = 0, first_arch = 0, last_arch = 0;
    for (const auto &k : order) {
        first_list += games[k].front().first;
        last_list += games[k].back().first;
        first_arch += games[k].front().second;
        last_arch += games[k].back().second;
    }
    const double ng = static_cast<double>(order.size());
    double ece = 0;
    for (const auto &b : rel) ece += std::fabs(std::get<1>(b) - static_cast<double>(std::get<2>(b)));
    ece /= dn * static_cast<double>(n_lists);
    const double uniform = std::log(static_cast<double>(n_lists));
    char buf[1024];
    std::vector<std::string> lines;
    std::snprintf(buf, sizeof buf, "%zu seat frames (%zu exact), %zu games", n, exact, order.size());
    lines.push_back(buf);
    std::snprintf(buf, sizeof buf,
        "list log-loss %.3f (uniform %.3f, uniform over the consistent lists %.3f); archetype log-loss %.3f (uniform "
        "prior %.3f); Brier list %.3f, archetype %.3f; ECE %.4f",
        ll / dn, uniform, ll_cons / dn, ll_arch / dn, ll_arch_uniform / dn, brier / dn, brier_arch / dn, ece);
    lines.push_back(buf);
    std::snprintf(buf, sizeof buf,
        "%d frames with several consistent lists: list log-loss %.3f, uniform over the consistent lists %.3f",
        static_cast<int>(amb), amb ? amb_ll / amb : 0.0, amb ? amb_cons / amb : 0.0);
    lines.push_back(buf);
    std::snprintf(buf, sizeof buf, "true list p first/last frame of a game %.3f/%.3f; true archetype %.3f/%.3f",
        first_list / ng, last_list / ng, first_arch / ng, last_arch / ng);
    lines.push_back(buf);
    std::string bins_text;
    for (int i = 0; i < bins; ++i) {
        const auto &b = rel[static_cast<size_t>(i)];
        const size_t c = std::get<0>(b);
        if (!c) continue;
        std::snprintf(buf, sizeof buf, "[%.1f,%.1f) %zu %.3f %.3f", i / 10.0, (i + 1) / 10.0, c,
            std::get<1>(b) / static_cast<double>(c), static_cast<double>(std::get<2>(b)) / static_cast<double>(c));
        bins_text += (bins_text.empty() ? "" : "; ") + std::string(buf);
    }
    lines.push_back("reliability (bin: pairs, mean predicted, frequency true): " + bins_text);
    std::vector<std::string> fails;
    if (!(ll / dn < uniform)) fails.push_back("calibration: list log-loss is not below the uniform prior's");
    if (!(ll_arch / dn < ll_arch_uniform / dn))
        fails.push_back("calibration: archetype log-loss is not below the uniform prior's");
    if (!(last_arch > first_arch))
        fails.push_back("calibration: the true archetype's probability does not rise over a game");
    for (const std::string &l : lines) rep.info(std::string(fails.empty() ? "OK   " : "FAIL ") + "calibration: " + l);
    for (const std::string &f : fails) rep.error("knowledge", f);
}

namespace rmcheck {

void archetype_part(const Options &opt, Report &rep, const std::string &dir) {
    const Tags tags = load_tags(opt);
    if (!tags.error.empty()) {
        rep.error("knowledge", "archetype: " + tags.error);
        return;
    }
    struct Row {
            const char *a, *b;
            bool bo3;
            long seed;
            bool closed;
    };
    static const Row rows[] = {{"league/bug", "delver", false, 41, true},
        {"league/ur_delver", "league/lands", true, 42, true}, {"league/burn", "league/bw_dnt", false, 43, true},
        {"league/uw_control", "league/reanimator", false, 44, false}};
    std::vector<StreamGame> games;
    for (const Row &r : rows) {
        StreamGame g;
        g.deck_a = r.a;
        g.deck_b = r.b;
        g.bo3 = r.bo3;
        g.seed = r.seed;
        if (r.closed) g.flags.push_back("--closed-decklist");
        games.push_back(g);
    }
    std::vector<std::vector<std::string>> fails(games.size());
    std::vector<std::string> summaries(games.size());
    parallel_for(games.size(), opt.jobs,
        [&](size_t i) { fails[i] = archetype_session(opt, games[i], tags, dir, &summaries[i]); });
    for (size_t i = 0; i < games.size(); ++i) {
        for (const std::string &f : fails[i]) rep.error("knowledge", f);
        if (fails[i].empty()) rep.info("OK   " + summaries[i]);
    }
}

void calibration_part(const Options &opt, Report &rep, const std::string &dir) {
    const Tags tags = load_tags(opt);
    if (!tags.error.empty()) {
        rep.error("knowledge", "calibration: " + tags.error);
        return;
    }
    std::vector<std::string> decks;
    for (const auto &d : tags.decks) decks.push_back(d.first);
    std::vector<StreamGame> games;
    for (size_t i = 0; i < decks.size(); ++i) {
        StreamGame g;
        g.deck_a = decks[i];
        g.deck_b = decks[(i + 5) % decks.size()];
        g.seed = 61 + static_cast<long>(i);
        g.flags = {"--knowledge-a", "expert+belief", "--knowledge-b", "expert+belief", "--closed-decklist"};
        games.push_back(g);
    }
    std::vector<std::vector<CalibrationRecord>> records(games.size());
    std::vector<std::vector<std::string>> fails(games.size());
    parallel_for(
        games.size(), opt.jobs, [&](size_t i) { fails[i] = calibration_session(opt, games[i], dir, &records[i]); });
    std::vector<CalibrationRecord> all;
    for (size_t i = 0; i < games.size(); ++i) {
        for (const std::string &f : fails[i]) rep.error("knowledge", f);
        all.insert(all.end(), records[i].begin(), records[i].end());
    }
    calibration_report(rep, all);
}

}  // namespace rmcheck
