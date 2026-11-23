#include "loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "episodes.h"
#include "files.h"
#include "match_index.h"
#include "replay_cache.h"
#include "subprocess.h"

namespace {

// Counts the STATE frames of a dump the engine is still writing (it flushes after each decision):
// whole records only, from where the last scan stopped.
struct DumpScanner {
        uint64_t offset = 0;
        size_t states = 0;
        void scan(const std::string &path);
};

const int POLL_MS = 40;
const size_t TAIL_LINES = 40;

}  // namespace

static std::string base_name(const std::string &path);
static std::string tail_lines(const std::string &text, size_t n);
static size_t count_markers(const std::string &narrative);
static std::string shell_line(const std::vector<std::string> &argv, const std::string &cwd);
static std::string fatal_line(const std::string &stderr_text);

void DumpScanner::scan(const std::string &path) {
    const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    while (true) {
        char head[40];
        const ssize_t n = pread(fd, head, sizeof head, static_cast<off_t>(offset));
        if (n <= 0) break;
        const std::string h(head, static_cast<size_t>(n));
        const size_t nl = h.find('\n');
        if (nl == std::string::npos || h.compare(0, 8, "RMSTATE ") != 0) break;
        const unsigned long long size = std::strtoull(h.c_str() + 8, nullptr, 10);
        struct stat st;
        if (fstat(fd, &st) != 0 || offset + nl + 1 + size > static_cast<uint64_t>(st.st_size)) break;
        unsigned char kind = 0;
        if (pread(fd, &kind, 1, static_cast<off_t>(offset + nl + 1 + 6)) == 1 && kind == 3) ++states;
        offset += nl + 1 + size;
    }
    close(fd);
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string tail_lines(const std::string &text, size_t n) {
    size_t at = text.size();
    if (at > 0 && text[at - 1] == '\n') --at;
    size_t lines = 0;
    while (at > 0) {
        if (text[at - 1] == '\n' && ++lines == n) break;
        --at;
    }
    return text.substr(at);
}

static size_t count_markers(const std::string &narrative) {
    size_t n = 0;
    for (size_t at = narrative.find("(REPLAY) ["); at != std::string::npos; at = narrative.find("(REPLAY) [", at + 1))
        if (at == 0 || narrative[at - 1] == '\n') ++n;
    return n;
}

// "$ cd DIR && prog arg …", for the error dialog's details.
static std::string shell_line(const std::vector<std::string> &argv, const std::string &cwd) {
    std::string out = "$ cd " + cwd + " &&";
    for (const std::string &a : argv) out += " " + a;
    return out + "\n";
}

// The engine's last "FATAL: …" line, or "".
static std::string fatal_line(const std::string &stderr_text) {
    const size_t at = stderr_text.rfind("FATAL");
    if (at == std::string::npos || (at > 0 && stderr_text[at - 1] != '\n')) return "";
    const size_t nl = stderr_text.find('\n', at);
    return stderr_text.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
}

namespace rmwork {

std::string replay_engine_binary(const std::string &configured) {
    if (base_name(configured) != "robomage_nn") return configured;
    const std::string dir = parent_dir(configured);
    const std::string plain = (dir.empty() ? "." : dir) + "/robomage";
    return is_executable(plain) ? plain : configured;
}

Loader::~Loader() {
    if (thread_.joinable()) {
        cancel_.store(true);
        thread_.join();
    }
}

bool Loader::start(const LoadRequest &req) {
    if (running_) return false;
    if (thread_.joinable()) thread_.join();
    req_ = req;
    req_.log_path = absolute_path(req.log_path);
    out_ = LoadOutcome();
    cancel_.store(false);
    done_.store(false);
    running_ = true;
    set_phase("Opening " + base_name(req_.log_path), -1);
    thread_ = std::thread([this] { run(); });
    return true;
}

void Loader::progress(std::string *phase, double *fraction) const {
    std::lock_guard<std::mutex> lock(mu_);
    *phase = phase_;
    *fraction = fraction_;
}

bool Loader::take(LoadOutcome *out) {
    if (!running_ || !done_.load()) return false;
    thread_.join();
    *out = std::move(out_);
    out_ = LoadOutcome();
    running_ = false;
    return true;
}

void Loader::set_phase(const std::string &phase, double fraction) {
    std::lock_guard<std::mutex> lock(mu_);
    phase_ = phase;
    fraction_ = fraction;
}

void Loader::fail(LoadOutcome *out, const std::string &title, const std::string &message, const std::string &detail) {
    out->status = cancel_.load() ? LoadOutcome::Status::CANCELLED : LoadOutcome::Status::FAILED;
    out->match.reset();
    out->title = title;
    out->message = message;
    out->detail = detail;
}

void Loader::run() {
    LoadOutcome out;
    std::unique_ptr<Match> m(new Match());
    m->log_path = req_.log_path;
    std::string bytes, err;
    if (!read_file(req_.log_path, &bytes)) {
        fail(&out, "Decision log not read", "The decision log " + req_.log_path + " could not be read.",
            std::strerror(errno));
    } else if (!parse_rmlog(bytes, &m->log, &err)) {
        fail(&out, "Not a decision log",
            base_name(req_.log_path) + " is not a decision log the engine can replay (RMLOG v3).", err);
    } else if (replay(bytes, m.get(), &out)) {
        m->log_hash = fnv1a64(bytes);
        const bool ok = index(bytes, m.get(), &out, &err);
        if (!ok) {
            fail(&out, "Replay not decoded",
                "The frames the engine wrote while replaying " + base_name(req_.log_path) +
                    " could not be indexed, so the game is not shown.",
                "dump: " + m->dump_path + "\n" + err + "\n");
        } else {
            set_phase("Joining episodes", -1);
            join_episodes(m.get(), &out);
            out.status = LoadOutcome::Status::OK;
            out.match = std::move(m);
        }
    }
    if (cancel_.load() && out.status != LoadOutcome::Status::OK) out.status = LoadOutcome::Status::CANCELLED;
    out_ = std::move(out);
    done_.store(true);
}

// The cached replay of the log, else the engine's; the match's file paths set. False after fail().
bool Loader::replay(const std::string &log_bytes, Match *m, LoadOutcome *out) {
    const EngineIdentity id = engine_identity(req_.engine);
    if (!id.found || !is_executable(id.path)) {
        fail(out, "Engine not found",
            "The engine binary " + req_.engine +
                " is missing or not executable, so the decision log cannot be replayed. Set it under Edit → "
                "Settings → Engine, or build it with make.",
            req_.engine + "\n");
        return false;
    }
    m->engine = id.path;
    const CacheEntry entry = cache_entry(req_.cache_root, replay_cache_key(log_bytes, id));
    m->dump_path = entry.dump;
    m->narrative_path = entry.narrative;
    m->stderr_path = entry.stderr_path;
    index_path_ = entry.index;
    if (cache_entry_complete(entry)) {
        m->from_cache = true;
        utimensat(AT_FDCWD, entry.entry_json.c_str(), nullptr, 0);
        return true;
    }
    const std::string tmp = entry.dir + ".tmp-" + std::to_string(static_cast<long>(getpid()));
    remove_tree(tmp);
    if (!make_dirs(tmp)) {
        fail(out, "Replay cache not writable", "The replay cache directory could not be created.", tmp + "\n");
        return false;
    }
    const CacheEntry work = cache_entry(req_.cache_root, base_name(tmp));
    const std::vector<std::string> argv = {
        id.path, "--replay", req_.log_path, "--rmstate-dump", work.dump, "--narrative"};
    ChildProcess child;
    std::string err;
    if (!child.start(argv, req_.workdir, work.narrative, work.stderr_path, &err)) {
        remove_tree(tmp);
        fail(out, "Engine not started", "The engine could not be started.", shell_line(argv, req_.workdir) + err);
        return false;
    }
    const size_t total = m->log.replies.size();
    DumpScanner scanner;
    const std::string name = base_name(req_.log_path);
    set_phase("Replaying " + name, 0);
    while (child.poll()) {
        if (cancel_.load()) {
            child.kill_and_reap();
            remove_tree(tmp);
            fail(out, "Cancelled", "", "");
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_MS));
        scanner.scan(work.dump);
        const double done = total ? static_cast<double>(scanner.states / VIEW_COUNT) / static_cast<double>(total) : 0;
        set_phase("Replaying " + name + " (decision " + std::to_string(scanner.states / VIEW_COUNT) + " / " +
                      std::to_string(total) + ")",
            std::min(1.0, done));
    }
    if (!child.ok()) {
        std::string err_text, narrative;
        read_file(work.stderr_path, &err_text);
        read_file(work.narrative, &narrative);
        const size_t reached = count_markers(narrative);
        remove_tree(tmp);
        const std::string fatal = fatal_line(err_text);
        fail(out, "Engine exited",
            "The engine exited (" + child.status_text() + ") while replaying " + name + ", after decision " +
                std::to_string(reached) + " of " + std::to_string(total) + (fatal.empty() ? "" : ": " + fatal) +
                ". Nothing is shown for this log.",
            shell_line(argv, req_.workdir) + "--- stderr (last lines) ---\n" + tail_lines(err_text, TAIL_LINES) +
                "--- stdout (last lines) ---\n" + tail_lines(narrative, TAIL_LINES / 2));
        return false;
    }
    if (std::rename(tmp.c_str(), entry.dir.c_str()) != 0) {
        // Another workbench cached the same replay meanwhile: use its entry.
        remove_tree(tmp);
        if (!cache_entry_complete(entry)) {
            fail(out, "Replay cache not writable", "The replay could not be moved into the cache.",
                tmp + " -> " + entry.dir + ": " + std::strerror(errno) + "\n");
            return false;
        }
        return true;
    }
    if (!write_cache_entry(entry, req_.log_path, id, total, &err)) {
        fail(out, "Replay cache not writable", "The replay's cache entry could not be written.", err + "\n");
        return false;
    }
    out->replayed = true;
    prune_replay_cache(req_.cache_root, CACHE_MAX_BYTES, entry.dir);
    return true;
}

// The match's index: the cache entry's when it is current, else the dump indexed and the entry's
// index written (a cache that cannot take it is a note). False with `err`.
bool Loader::index(const std::string &log_bytes, Match *m, LoadOutcome *out, std::string *err) {
    const uint64_t log_hash = fnv1a64(log_bytes);
    std::string why;
    set_phase("Reading the index of " + base_name(req_.log_path), -1);
    if (read_match_index(m, index_path_, log_hash, &why)) return true;
    set_phase("Indexing " + base_name(req_.log_path), 0);
    const bool ok = index_match(
        m,
        [this](double f) {
            set_phase("Indexing " + base_name(req_.log_path), f);
            return !cancel_.load();
        },
        err);
    if (ok && !write_match_index(*m, index_path_, log_hash, &why))
        out->notes.push_back("the index was not cached: " + why);
    return ok;
}

// The log's own episodes, then those opened with it. A problem with an episode opened with the log
// is reported (the match still opens); one found by name is a note.
void Loader::join_episodes(Match *m, LoadOutcome *out) {
    std::vector<std::string> files;
    if (req_.find_episodes) files = episode_files_for_log(req_.log_path);
    for (const std::string &e : req_.episodes) {
        const std::string p = absolute_path(e);
        if (std::find(files.begin(), files.end(), p) == files.end()) files.push_back(p);
    }
    for (const std::string &f : files) {
        const bool explicit_one =
            std::find(req_.episodes.begin(), req_.episodes.end(), f) != req_.episodes.end() ||
            std::find(req_.episodes.begin(), req_.episodes.end(), absolute_path(f)) != req_.episodes.end();
        JoinReport report;
        std::string err;
        if (!join_episode(m, f, &report, &err)) {
            if (explicit_one) {
                out->title = "Episode not joined";
                out->message =
                    "The episode " + base_name(f) + " could not be joined to " + base_name(m->log_path) + ".";
                out->detail += err + "\n";
            } else {
                out->notes.push_back(base_name(f) + ": " + err);
            }
            continue;
        }
        out->notes.push_back(base_name(f) + ": " + std::to_string(report.joined) + " of " +
                             std::to_string(report.steps) + " decisions joined");
        if (!report.problems.empty() && explicit_one) {
            out->title = "Episode partly joined";
            out->message = "Some decisions of " + base_name(f) + " have no matching decision in the replay.";
            for (const std::string &p : report.problems) out->detail += p + "\n";
        }
    }
}

}  // namespace rmwork
