#include "play_session.h"

#include <signal.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>

#include "../../src/state_format/decode.h"
#include "files.h"
#include "loader.h"
#include "paths.h"
#include "start_data.h"

namespace {

const char *const OPPONENT_KEYS[rmwork::OPPONENT_COUNT] = {"model", "server", "explore", "random", "scripted"};
const char *const OPPONENT_LABELS[rmwork::OPPONENT_COUNT] = {"Model (nn:, a model export)",
    "Model server (nn:remote:, an inference server)", "Explore (the coverage fuzzer)", "Random",
    "Scripted (hard, through the teacher channel)"};
const char *const MODEL_READ_FILE = "Model's Read";

}  // namespace

static std::string base_name(const std::string &path);
static std::string shell_quote(const std::string &s);
static std::string game_stem(const std::string &dir, uint64_t seed);
static void append_quit(const std::string &log);

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// `s` as one /bin/sh word.
static std::string shell_quote(const std::string &s) {
    std::string out = "'";
    for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

// "<dir>/<timestamp>-<seed>", with a suffix when a game of that name exists.
static std::string game_stem(const std::string &dir, uint64_t seed) {
    const std::string base = dir + "/" + rmwork::file_timestamp() + "-" + std::to_string(seed);
    std::string stem = base;
    for (int i = 2; rmwork::file_exists(stem + ".rmlog"); ++i) stem = base + "-" + std::to_string(i);
    return stem;
}

// A game that ended without a result (stopped, or the engine gone): its decision log gets the QUIT
// line a human seat's quit writes, so it replays to the decision it stopped at (rmlog.h). Nothing
// for a log that is missing, does not read or already ends with QUIT.
static void append_quit(const std::string &log) {
    std::string text, err;
    rmwork::RmlogInfo info;
    if (log.empty() || !rmwork::read_file(log, &text) || !rmwork::parse_rmlog(text, &info, &err) || info.quit) return;
    FILE *f = std::fopen(log.c_str(), "ab");
    if (!f) return;
    if (!text.empty() && text.back() != '\n') std::fputc('\n', f);
    std::fputs("QUIT\n", f);
    std::fclose(f);
}

namespace rmwork {

const char *opponent_key(Opponent o) {
    return OPPONENT_KEYS[static_cast<int>(o)];
}

const char *opponent_label(Opponent o) {
    return OPPONENT_LABELS[static_cast<int>(o)];
}

int find_opponent(const std::string &key) {
    for (int i = 0; i < OPPONENT_COUNT; ++i)
        if (key == OPPONENT_KEYS[i]) return i;
    return -1;
}

bool play_config_from(const NewGameSettings &g, PlayConfig *out, std::string *why) {
    const int o = find_opponent(g.opponent);
    if (o < 0) {
        *why = "no opponent kind \"" + g.opponent + "\"";
        return false;
    }
    *out = PlayConfig();
    out->seat = g.seat == 1 ? 1 : 0;
    out->deck = g.deck;
    out->opponent = static_cast<Opponent>(o);
    out->model = out->opponent == Opponent::MODEL ? g.model : out->opponent == Opponent::SERVER ? g.server : "";
    out->opponent_deck = g.opponent_deck;
    out->bo3 = g.bo3;
    out->seed = g.seed;
    out->auto_mana = g.auto_mana;
    out->model_read = g.model_read;
    return true;
}

std::string play_engine_binary(const std::string &configured, Opponent o, std::string *why) {
    if (!opponent_needs_nn(o)) return replay_engine_binary(configured);
    if (base_name(configured) == "robomage_nn" && is_executable(configured)) return configured;
    const std::string beside = parent_dir(configured) + "/robomage_nn";
    if (is_executable(beside)) return beside;
    *why = std::string("A ") + opponent_key(o) +
           " opponent plays in robomage_nn (the engine with the nn: agent), and there is none at " + beside +
           ". Build it with `make actor`, or set Settings \xE2\x86\x92 Engine's binary to one.";
    return "";
}

std::string seat_agent_spec(const SeatAgent &a) {
    switch (a.kind) {
        case Opponent::MODEL:
            return "nn:" + a.model + a.options;
        case Opponent::SERVER:
            return "nn:remote:" + a.model + a.options;
        case Opponent::EXPLORE:
            return "explore";
        case Opponent::RANDOM:
            return "random";
        case Opponent::SCRIPTED:
            return "first";
    }
    return "first";
}

std::vector<std::string> agent_arguments(const SeatAgent *a, const SeatAgent *b, uint64_t seed,
    const std::string &repo, const std::string &python) {
    const SeatAgent *seats[2] = {a, b};
    std::vector<std::string> out, teachers;
    for (int s = 0; s < 2; ++s) {
        if (!seats[s]) continue;
        const std::string key = s == 0 ? "a" : "b";
        out.insert(out.end(), {"--agent-" + key, seat_agent_spec(*seats[s])});
        if (seats[s]->kind == Opponent::SCRIPTED)
            teachers.insert(teachers.end(), {"--teacher-" + key, shell_quote(python) + " " +
                                                                     shell_quote(repo + "/python/teacher.py") +
                                                                     " --tier hard --seed " + std::to_string(seed)});
    }
    if (!teachers.empty()) {
        out.insert(out.end(), teachers.begin(), teachers.end());
        out.insert(out.end(), {"--teacher-mix", "1"});
    }
    return out;
}

std::string opponent_agent(const PlayConfig &c) {
    SeatAgent a;
    a.kind = c.opponent;
    a.model = c.model;
    return seat_agent_spec(a);
}

std::vector<std::string> play_arguments(
    const PlayConfig &c, const std::string &log_path, const std::string &repo, const std::string &python) {
    const std::string seed = std::to_string(c.seed);
    std::vector<std::string> a = {"--machine", "--game-seed", seed, "--deck-a", c.seat == 0 ? c.deck : c.opponent_deck,
        "--deck-b", c.seat == 0 ? c.opponent_deck : c.deck};
    if (c.bo3) a.push_back("--bo3");
    if (!c.auto_mana) a.push_back("--no-auto-mana");
    a.insert(a.end(), {"--decision-log", log_path});
    SeatAgent opp;
    opp.kind = c.opponent;
    opp.model = c.model;
    const std::vector<std::string> agents =
        agent_arguments(c.seat == 1 ? &opp : nullptr, c.seat == 0 ? &opp : nullptr, c.seed, repo, python);
    a.insert(a.end(), agents.begin(), agents.end());
    // The setup flags after the agent's: a sculpted game's own --agent-<seat> wins.
    a.insert(a.end(), c.setup.begin(), c.setup.end());
    return a;
}

PlaySession::~PlaySession() {
    stop();
}

bool PlaySession::start(const PlayConfig &c, const PlayPaths &p, std::string *err) {
    stop();
    config_ = c;
    paths_ = p;
    state_ = State::NONE;
    view_.clear();
    starting_.reset();
    pending_ = -1;
    considered_ = false;
    queued_concede_ = 0;
    autopass_ = AutoPass();
    auto_answered_.clear();
    records_ = withheld_ = first_withheld_record_ = last_decision_record_ = 0;
    finished_event_ = failure_event_ = model_reported_ = model_failure_event_ = false;
    notes_.clear();
    {
        std::lock_guard<std::mutex> lock(mu_);
        inbox_.clear();
        stream_done_ = false;
        stream_error_.clear();
    }
    if (!make_dirs(p.games_dir)) {
        *err = "cannot create the games directory " + p.games_dir;
        return false;
    }
    const std::string stem = game_stem(p.games_dir, c.seed);
    log_path_ = stem + ".rmlog";
    stream_path_ = stem + (c.seat == 0 ? ".seat-a.rmstate" : ".seat-b.rmstate");
    stderr_path_ = stem + ".stderr";
    GameSidecar side;
    side.seat = c.seat;
    side.opponent = opponent_key(c.opponent);
    side.model = c.model;
    side.deck = c.deck;
    side.opponent_deck = c.opponent_deck;
    side.bo3 = c.bo3;
    side.seed = c.seed;
    if (!write_file_atomic(sidecar_path(log_path_), sidecar_json(side), err)) return false;
    starting_.reset(new Match());
    Match &m = *starting_;
    if (!begin_live_match(&m, c.seat, stream_path_, err)) return false;
    m.log_path = log_path_;
    m.stderr_path = stderr_path_;
    m.engine = p.engine;
    m.log.seed = c.seed;
    m.log.flags = std::string("machine") + (c.bo3 ? " bo3" : "") + (c.auto_mana ? "" : " no-auto-mana");
    m.log.deck_a = c.seat == 0 ? c.deck : c.opponent_deck;
    m.log.deck_b = c.seat == 0 ? c.opponent_deck : c.deck;
    m.agents[static_cast<size_t>(c.seat)] = "you";
    m.agents[static_cast<size_t>(1 - c.seat)] =
        c.opponent == Opponent::SCRIPTED ? "scripted:hard (teacher)" : opponent_agent(c);
    view_.set_view(c.seat);
    command_ = {p.engine};
    const std::vector<std::string> args = play_arguments(c, log_path_, p.repo, p.python);
    command_.insert(command_.end(), args.begin(), args.end());
    // A write to an engine that has gone must fail with EPIPE, not end the workbench.
    signal(SIGPIPE, SIG_IGN);
    int from_child = -1;
    if (!child_.start_piped(command_, p.workdir, stderr_path_, &to_child_, &from_child, err)) {
        starting_.reset();
        return false;
    }
    reader_ = std::thread([this, from_child] { reader_main(from_child); });
    state_ = State::RUNNING;
    if (c.model_read) {
        ModelReadConfig mr = p.model_read;
        mr.engine = p.engine;
        mr.workdir = p.workdir;
        mr.cards_path = workbench_cache_dir() + "/model_read/cards-" + std::to_string(getpid()) + ".rmst";
        model_.start(mr);
    }
    return true;
}

void PlaySession::reader_main(int fd) {
    FILE *f = fdopen(fd, "rb");
    if (!f) {
        close(fd);
        std::lock_guard<std::mutex> lock(mu_);
        stream_done_ = true;
        stream_error_ = std::string("cannot read the engine's output: ") + std::strerror(errno);
        return;
    }
    rmstate::RecordReader reader(f);
    std::string payload;
    rmstate::DecodeResult res;
    while (reader.next(&payload, &res)) {
        std::lock_guard<std::mutex> lock(mu_);
        inbox_.push_back(payload);
    }
    std::fclose(f);
    std::lock_guard<std::mutex> lock(mu_);
    stream_done_ = true;
    if (!res.ok()) stream_error_ = "record " + std::to_string(reader.count() + 1) + ": " + res.message;
}

void PlaySession::poll(const StopSettings &stops) {
    attach_values();
    if (state_ != State::RUNNING) return;
    take_records();
    if (state_ != State::RUNNING) return;
    answer_pending(stops);
    if (state_ != State::RUNNING) return;
    check_exit();
}

void PlaySession::take_records() {
    std::deque<std::string> got;
    {
        std::lock_guard<std::mutex> lock(mu_);
        got.swap(inbox_);
    }
    for (const std::string &payload : got) {
        add_record(payload);
        if (state_ != State::RUNNING) return;
    }
}

void PlaySession::add_record(const std::string &payload) {
    ++records_;
    // At the match's end the stream also carries the other seat's once-per-match and terminal
    // frames; they are dropped unread.
    if (payload.size() > 7 && static_cast<uint8_t>(payload[7]) != config_.seat) {
        if (!withheld_++) first_withheld_record_ = records_;
        return;
    }
    LiveFrame lf;
    std::string err;
    if (!append_live_frame(live(), payload, &lf, &err)) {
        fail("A frame from the engine could not be read",
            "Record " + std::to_string(records_) + " of the engine's RMSTATE stream: " + err +
                ". The game cannot go on; its decision log so far is " + log_path_ + ".");
        return;
    }
    if (config_.model_read) model_.feed(payload, lf.decision ? static_cast<long>(lf.index) : -1);
    if (lf.decision) {
        if (starting_) {
            view_.set_match(std::move(starting_), lf.index);
        } else {
            view_.go_to(lf.index);
        }
        pending_ = static_cast<long>(lf.index);
        considered_ = false;
        last_decision_record_ = records_;
    }
    if (view_.loaded()) view_.touch();
}

// The passing and a queued concession answer the pending decision as it arrives.
void PlaySession::answer_pending(const StopSettings &stops) {
    if (pending_ < 0 || considered_) return;
    considered_ = true;
    std::string err;
    if (queued_concede_ != 0) {
        const int r = queued_concede_;
        queued_concede_ = 0;
        reply(r, &err);
        return;
    }
    Match *m = live();
    const std::shared_ptr<const rmstate::DecodedFrame> f = m->frame(static_cast<size_t>(pending_), config_.seat, &err);
    if (!f) return;
    const PassMode mode = autopass_.mode;
    const AutoReply r =
        auto_reply(&autopass_, read_pass_facts(agents::DecisionView(m->stream(config_.seat), *f, config_.seat)), stops);
    if (!r.ended.empty())
        notes_.push_back(std::string(mode == PassMode::SKIP_TURN ? "F6" : "F2") + " stopped: " + r.ended);
    if (r.row < 0) return;
    const size_t index = static_cast<size_t>(pending_);
    if (reply(r.row, &err)) auto_answered_.push_back(index);
}

void PlaySession::attach_values() {
    if (!config_.model_read) return;
    const ModelReader::State ms = model_.state();
    if (!model_reported_ && ms == ModelReader::State::FAILED) {
        model_reported_ = model_failure_event_ = true;
    } else if (!model_reported_ && ms == ModelReader::State::NOT_CONNECTED) {
        model_reported_ = true;
        notes_.push_back("Model's Read: not connected (" + model_.error() + "); the game goes on without it");
    }
    Match *m = live();
    if (!m) return;
    bool any = false;
    for (ModelReadResult &r : model_.take()) {
        if (r.decision >= m->decisions.size() || r.policies.empty()) continue;
        DecisionInfo &d = m->decisions[r.decision];
        EpisodeStepInfo e;
        e.file = MODEL_READ_FILE;
        e.seat = config_.seat;
        e.chosen = d.reply == REPLY_PENDING ? -1 : d.reply;
        e.policy_version = static_cast<int64_t>(r.policies[0].version);
        e.policies = std::move(r.policies);
        m->episode_steps.push_back(std::move(e));
        d.episode_step = static_cast<int>(m->episode_steps.size()) - 1;
        any = true;
    }
    if (!any) return;
    if (m->episode_files.empty()) m->episode_files.push_back(MODEL_READ_FILE);
    if (view_.loaded()) view_.touch();
}

void PlaySession::check_exit() {
    const bool alive = child_.poll();
    bool drained = false;
    std::string broken;
    {
        std::lock_guard<std::mutex> lock(mu_);
        drained = stream_done_ && inbox_.empty();
        broken = stream_error_;
    }
    if (!broken.empty()) {
        fail("The engine's stream broke", "The engine's RMSTATE stream could not be read: " + broken + ".");
        return;
    }
    if (alive) return;
    // What the engine started (a teacher) may still hold the stream's pipe: its group goes too.
    child_.kill_and_reap();
    if (!drained) return;
    close_pipes();
    const Match *m = match();
    if (m && m->finished && child_.ok()) {
        state_ = State::FINISHED;
        finished_event_ = true;
        pending_ = -1;
        return;
    }
    const std::string at = m ? " after " + std::to_string(m->decisions.size()) + " of your decisions" : "";
    fail("The engine exited mid-game", "The engine stopped (" + child_.status_text() + ")" + at +
                                           " before the match ended. Its decision log so far is " + log_path_ + ".");
}

void PlaySession::fail(const std::string &title, const std::string &message) {
    state_ = State::FAILED;
    failure_event_ = true;
    pending_ = -1;
    fail_title_ = title;
    fail_message_ = message;
    child_.kill_and_reap();
    close_pipes();
    model_.stop();
    if (!match() || !match()->finished) append_quit(log_path_);
    std::string cmd;
    for (const std::string &a : command_) cmd += (cmd.empty() ? "" : " ") + a;
    fail_detail_ = "$ " + cmd + "\n" + stderr_tail();
    if (!fail_detail_.empty() && fail_detail_.back() != '\n') fail_detail_ += '\n';
    fail_detail_ += child_.status_text() + "\n";
}

void PlaySession::close_pipes() {
    if (to_child_ >= 0) {
        close(to_child_);
        to_child_ = -1;
    }
    if (reader_.joinable()) reader_.join();
}

void PlaySession::stop() {
    const bool was_running = state_ == State::RUNNING;
    if (was_running) state_ = State::STOPPED;
    child_.kill_and_reap();
    close_pipes();
    model_.stop();
    pending_ = -1;
    if (was_running && (!match() || !match()->finished)) append_quit(log_path_);
}

void PlaySession::close_game() {
    stop();
    view_.clear();
    starting_.reset();
    state_ = State::NONE;
    queued_concede_ = 0;
    autopass_ = AutoPass();
}

bool PlaySession::write_reply(int reply, std::string *err) {
    const std::string line = std::to_string(reply) + "\n";
    size_t done = 0;
    while (done < line.size()) {
        const ssize_t n = write(to_child_, line.data() + done, line.size() - done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            *err = std::string("cannot write to the engine: ") + std::strerror(errno);
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

bool PlaySession::reply(int reply, std::string *err) {
    if (pending_ < 0 || state_ != State::RUNNING) {
        *err = "no decision of yours is pending";
        return false;
    }
    Match *m = live();
    const size_t index = static_cast<size_t>(pending_);
    const int rows = m->decisions[index].menu_rows;
    if (reply < -3 || reply >= rows) {
        *err = "row " + std::to_string(reply) + " is not on the menu (" + std::to_string(rows) + " rows)";
        return false;
    }
    if (!write_reply(reply, err)) {
        fail("The engine is gone", "Your reply could not be sent: " + *err + ".");
        return false;
    }
    set_live_reply(m, index, reply);
    const int step = m->decisions[index].episode_step;
    if (step >= 0) m->episode_steps[static_cast<size_t>(step)].chosen = reply;
    pending_ = -1;
    if (view_.loaded()) view_.touch();
    return true;
}

void PlaySession::concede(bool match) {
    const int r = match ? -3 : -2;
    if (pending_ >= 0) {
        std::string err;
        reply(r, &err);
        return;
    }
    if (!running()) return;
    queued_concede_ = r;
    notes_.push_back(std::string("Conceding the ") + (match ? "match" : "game") + " at your next decision");
}

bool PlaySession::pass(PassMode mode, const StopSettings &stops, std::string *why) {
    if (!running()) {
        *why = "no game is running";
        return false;
    }
    PassFacts at;
    Match *m = live();
    if (m && !m->decisions.empty()) {
        const size_t newest = m->decisions.size() - 1;
        std::string err;
        const std::shared_ptr<const rmstate::DecodedFrame> f = m->frame(newest, config_.seat, &err);
        if (f) at = read_pass_facts(agents::DecisionView(m->stream(config_.seat), *f, config_.seat));
    }
    start_auto_pass(&autopass_, mode, at);
    if (pending_ >= 0) {
        considered_ = false;
        answer_pending(stops);
    }
    return true;
}

const Match *PlaySession::match() const {
    return starting_ ? starting_.get() : view_.match();
}

Match *PlaySession::live() {
    return starting_ ? starting_.get() : view_.match();
}

bool PlaySession::take_failure(std::string *title, std::string *message, std::string *detail) {
    if (!failure_event_) return false;
    failure_event_ = false;
    *title = fail_title_;
    *message = fail_message_;
    *detail = fail_detail_;
    return true;
}

bool PlaySession::take_finished() {
    const bool f = finished_event_;
    finished_event_ = false;
    return f;
}

bool PlaySession::take_model_failure(std::string *message) {
    if (!model_failure_event_) return false;
    model_failure_event_ = false;
    *message = model_.error();
    return true;
}

std::vector<std::string> PlaySession::take_notes() {
    std::vector<std::string> out;
    out.swap(notes_);
    return out;
}

std::string PlaySession::stderr_tail(size_t max_bytes) const {
    std::string text;
    if (stderr_path_.empty() || !read_file(stderr_path_, &text)) return "";
    if (text.size() <= max_bytes) return text;
    const size_t from = text.find('\n', text.size() - max_bytes);
    return text.substr(from == std::string::npos ? text.size() - max_bytes : from + 1);
}

}  // namespace rmwork
