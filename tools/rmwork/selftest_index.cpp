#include "selftest_index.h"

#include "files.h"
#include "match_index.h"
#include "replay_cache.h"

static std::string index_diff(const rmwork::Match &a, const rmwork::Match &b);
static std::string log_diff(const std::vector<rmwork::LogLine> &a, const std::vector<rmwork::LogLine> &b);
static bool rewrite_index(const std::string &path, const std::function<void(std::string *body)> &change);

// The first difference between two indexes of one match ("" when none).
static std::string index_diff(const rmwork::Match &a, const rmwork::Match &b) {
    using namespace rmwork;
    if (a.decisions.size() != b.decisions.size()) return "the decision counts";
    for (size_t i = 0; i < a.decisions.size(); ++i) {
        const DecisionInfo &x = a.decisions[i], &y = b.decisions[i];
        bool same = x.game == y.game && x.sideboarding == y.sideboarding && x.decision == y.decision &&
                    x.decider == y.decider && x.turn == y.turn && x.phase == y.phase && x.header == y.header &&
                    x.menu_rows == y.menu_rows && x.reply == y.reply && x.reply_text == y.reply_text &&
                    x.life == y.life;
        for (int v = 0; v < VIEW_COUNT; ++v)
            same = same && x.frames[static_cast<size_t>(v)].offset == y.frames[static_cast<size_t>(v)].offset &&
                   x.frames[static_cast<size_t>(v)].size == y.frames[static_cast<size_t>(v)].size;
        if (!same) return "decision " + std::to_string(i + 1);
        if (b.find_decision(x.game, x.sideboarding, x.decision) != static_cast<int>(i))
            return "decision " + std::to_string(i + 1) + " by its key";
    }
    if (a.games.size() != b.games.size()) return "the games";
    for (size_t g = 0; g < a.games.size(); ++g)
        if (a.games[g].first != b.games[g].first || a.games[g].count != b.games[g].count ||
            a.games[g].label != b.games[g].label)
            return "game " + a.games[g].label;
    for (int v = 0; v < VIEW_COUNT; ++v) {
        const std::string d = log_diff(a.logs[static_cast<size_t>(v)], b.logs[static_cast<size_t>(v)]);
        if (!d.empty()) return std::string(view_name(v)) + "'s log: " + d;
        if (a.terminal[static_cast<size_t>(v)].offset != b.terminal[static_cast<size_t>(v)].offset ||
            a.terminal[static_cast<size_t>(v)].size != b.terminal[static_cast<size_t>(v)].size)
            return std::string(view_name(v)) + "'s terminal frame";
        if (!b.stream(v).ready()) return std::string(view_name(v)) + "'s stream is not fed";
    }
    if (a.narrative_mapped != b.narrative_mapped || a.stopped != b.stopped) return "the narrative or the stop";
    if (a.result.known != b.result.known || a.result.winner != b.result.winner || a.result.wins != b.result.wins ||
        a.result.game_winners != b.result.game_winners || a.result.text != b.result.text)
        return "the result";
    std::string err;
    for (size_t i : {size_t(0), a.decisions.size() / 2, a.decisions.size()})
        for (int v = 0; v < VIEW_COUNT && i <= a.decisions.size(); ++v)
            if ((i < a.decisions.size() || a.has_terminal()) && !b.frame(i, v, &err))
                return "frame " + std::to_string(i) + " does not decode: " + err;
    return "";
}

static std::string log_diff(const std::vector<rmwork::LogLine> &a, const std::vector<rmwork::LogLine> &b) {
    if (a.size() != b.size()) return std::to_string(a.size()) + " lines, read back " + std::to_string(b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].text != b[i].text || a[i].decision != b[i].decision || a[i].kind != b[i].kind ||
            a[i].seat != b[i].seat)
            return "line " + std::to_string(i + 1);
    return "";
}

// Applies `change` to the index's body (all but its trailing hash), then writes the body with its
// hash again, so the index reads as whole.
static bool rewrite_index(const std::string &path, const std::function<void(std::string *body)> &change) {
    std::string bytes, err;
    if (!rmwork::read_file(path, &bytes) || bytes.size() < 8) return false;
    std::string body = bytes.substr(0, bytes.size() - 8);
    change(&body);
    uint64_t h = rmwork::fnv1a64(body);
    for (int i = 0; i < 8; ++i) body += static_cast<char>((h >> (8 * i)) & 0xff);
    return rmwork::write_file_atomic(path, body, &err);
}

namespace rmwork {

void check_match_index(const CheckFn &check, App &app, const std::string &log, const std::function<bool()> &open) {
    if (!open()) return;
    const Match *m = app.replay().match();
    if (!m) return;
    check(m->from_cache && m->index_cached, "Index: a cached replay's second open indexes the dump again");
    const std::string dir = parent_dir(m->dump_path);
    const CacheEntry entry = cache_entry(parent_dir(dir), dir.substr(dir.rfind('/') + 1));
    check(file_exists(entry.index), "Index: no index beside the cached dump");
    // The index read back equals the one built from the dump.
    Match built;
    built.log_path = m->log_path;
    built.log = m->log;
    built.dump_path = m->dump_path;
    built.narrative_path = m->narrative_path;
    std::string err;
    check(index_match(&built, [](double) { return true; }, &err), "Index: the dump does not index: " + err);
    const std::string diff = index_diff(built, *m);
    check(diff.empty(), "Index: the index read back differs from the dump's in " + diff);

    // A corrupt index: the open indexes the dump and writes the index again.
    std::string bytes;
    read_file(entry.index, &bytes);
    bytes[bytes.size() / 2] = static_cast<char>(bytes[bytes.size() / 2] ^ 0x5a);
    check(write_file_atomic(entry.index, bytes, &err), "Index: cannot corrupt the index");
    app.replay().clear();
    if (open()) {
        check(app.replay().loaded() && !app.replay().match()->index_cached, "Index: a corrupt index is read");
        const std::string d = app.replay().loaded() ? index_diff(built, *app.replay().match()) : "";
        check(d.empty(), "Index: the rebuilt index differs in " + d);
    }
    app.replay().clear();
    if (open())
        check(app.replay().loaded() && app.replay().match()->index_cached, "Index: the rebuilt index is not read");

    // An index of an older version, whole otherwise.
    check(rewrite_index(entry.index,
              [](std::string *body) {
                  const uint32_t old = MATCH_INDEX_VERSION - 1;
                  for (size_t i = 0; i < 4; ++i) (*body)[8 + i] = static_cast<char>((old >> (8 * i)) & 0xff);
              }),
        "Index: cannot rewrite the index's version");
    Match probe;
    probe.log_path = built.log_path;
    probe.dump_path = built.dump_path;
    probe.narrative_path = built.narrative_path;
    std::string bytes_log;
    read_file(log, &bytes_log);
    check(!read_match_index(&probe, entry.index, fnv1a64(bytes_log), &err) && err == "an index of another version",
        "Index: an index of an older version is read (" + err + ")");
    app.replay().clear();
    if (open()) check(app.replay().loaded() && !app.replay().match()->index_cached, "Index: an old index is read");
    app.replay().clear();
    if (open())
        check(app.replay().loaded() && app.replay().match()->index_cached, "Index: the rewritten index is not read");
}

}  // namespace rmwork
