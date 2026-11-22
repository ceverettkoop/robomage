#include "match.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../src/agents/decision_view.h"
#include "../../src/cli/board.h"
#include "../../src/state_format/decode.h"
#include "files.h"

namespace {

const char *const VIEW_NAMES[rmwork::VIEW_COUNT] = {"seat A", "seat B", "omniscient"};
const size_t FRAME_HEADER_BYTES = 24;

// The index pass's state: the decoded frames of the decision being read and the logs' turn
// headers so far.
struct Builder {
        rmwork::Match *m = nullptr;
        std::array<int, 2> last_turn{{-1, -1}};
        uint64_t event_kind_game_end = 0, event_kind_match_end = 0;
        bool have_event_kinds = false;
};

}  // namespace

static int seat_index(uint64_t seat);
static std::string seat_label(int seat);
static std::string humanize(const std::string &name);
static bool decode_ref(const rmwork::Match &m, const agents::SeatStream &stream, const rmwork::FrameRef &ref,
    rmstate::DecodedFrame *out, std::string *err);
static void read_key(const agents::Table &header, int *game, bool *sideboarding, int64_t *decision, bool *terminal);
static std::string phase_of(const agents::DecisionView &v, bool sideboarding);
static std::string reply_text(const agents::DecisionView &v, int reply);
static void fill_decision(rmwork::DecisionInfo *d, const agents::DecisionView &v);
static void add_seat_lines(Builder *b, int seat, const agents::DecisionView &v, int decision);
static void read_results(Builder *b, const agents::SeatStream &omni, const rmstate::DecodedFrame &f);
static bool process_triple(Builder *b, const std::array<rmwork::FrameRef, rmwork::VIEW_COUNT> &refs, std::string *err);
static void build_games(rmwork::Match *m);
static bool marker_line(const std::string &line);
static bool banner_line(const std::string &line);
static std::string header_text(const std::string &line);
static void build_narrative(rmwork::Match *m);
static size_t decided_count(const rmwork::Match &m);
static void finish_result(rmwork::Match *m);
static bool write_record(int fd, const std::string &payload, uint64_t *end, std::string *err);

// The `seat` enum's value (1 A, 2 B) as 0 / 1, -1 for none.
static int seat_index(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static std::string seat_label(int seat) {
    return seat == 0 ? "Player A" : seat == 1 ? "Player B" : "No player";
}

// FIRST_MAIN -> "First main".
static std::string humanize(const std::string &name) {
    std::string out;
    for (char c : name) out += c == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

static bool decode_ref(const rmwork::Match &m, const agents::SeatStream &stream, const rmwork::FrameRef &ref,
    rmstate::DecodedFrame *out, std::string *err) {
    std::string payload;
    if (!m.read_payload(ref, &payload, err)) return false;
    if (!stream.ready()) {
        *err = "a STATE frame before its viewer's SCHEMA frame";
        return false;
    }
    return stream.decode_state(payload, out, err);
}

static void read_key(const agents::Table &header, int *game, bool *sideboarding, int64_t *decision, bool *terminal) {
    *game = header.rows() ? static_cast<int>(header.u(0, header.col("game"))) : 0;
    *sideboarding = header.rows() && header.u(0, header.col("sideboarding")) != 0;
    *decision = header.rows() ? static_cast<int64_t>(header.u(0, header.col("decision"))) : 0;
    *terminal = header.rows() && header.u(0, header.col("terminal")) != 0;
}

// "Sideboarding", "Opening hand" (a keep or mulligan decision, or a London bottoming pick: a
// bottom-of-library pick while a player still owes cards to the bottom, which a scry-like effect's
// bottom pick is not), else the step.
static std::string phase_of(const agents::DecisionView &v, bool sideboarding) {
    if (sideboarding) return "Sideboarding";
    const agents::Table players = v.table("player");
    bool bottoming = false;
    for (size_t r = 0; r < players.rows(); ++r) bottoming = bottoming || players.u(r, players.col("to_bottom")) > 0;
    for (size_t i = 0; i < v.n(); ++i) {
        const std::string &c = v.category(i);
        if (c == "KEEP_HAND" || c == "MULLIGAN" || (c == "BOTTOM_DECK_CARD" && bottoming)) return "Opening hand";
    }
    const agents::Table turn = v.table("turn");
    if (!turn.rows()) return "";
    return humanize(v.stream().enum_name("step", turn.u(0, turn.col("step"))));
}

// The chosen row's description; -1 is the last row (an attacker or blocker confirm).
static std::string reply_text(const agents::DecisionView &v, int reply) {
    if (reply == -2) return "concedes the game";
    if (reply == -3) return "concedes the match";
    const size_t n = v.n();
    if (n == 0) return std::to_string(reply);
    const size_t row = reply == -1 ? n - 1 : static_cast<size_t>(reply);
    if (reply < -1 || row >= n) return "row " + std::to_string(reply) + " (outside the menu)";
    return v.description(row);
}

// The decision's turn, phase, board header, menu size and life totals, from a frame that shows the
// menu (`v`'s seat the decider).
static void fill_decision(rmwork::DecisionInfo *d, const agents::DecisionView &v) {
    const agents::Table turn = v.table("turn");
    d->turn = turn.rows() ? static_cast<int>(turn.u(0, turn.col("turn"))) : 0;
    d->phase = phase_of(v, d->sideboarding);
    d->header = cli::board_header(v);
    d->menu_rows = static_cast<int>(v.n());
    const agents::Table players = v.table("player");
    for (size_t r = 0; r < players.rows(); ++r) {
        const int seat = seat_index(players.u(r, players.col("player")));
        if (seat >= 0) d->life[static_cast<size_t>(seat)] = players.s(r, players.col("life"));
    }
}

// The seat's events at `decision`, each "T<n>: …" line under a turn header, a game event as a header.
static void add_seat_lines(Builder *b, int seat, const agents::DecisionView &v, int decision) {
    std::vector<rmwork::LogLine> &out = b->m->logs[static_cast<size_t>(seat)];
    int &last_turn = b->last_turn[static_cast<size_t>(seat)];
    for (const std::string &line : cli::event_lines(v)) {
        int turn = 0;
        std::string text;
        rmwork::LogLine l;
        l.decision = decision;
        if (!rmwork::split_turn_prefix(line, &turn, &text)) {
            l.kind = rmwork::LogLine::Kind::HEADER;
            l.text = line;
            last_turn = -1;
            out.push_back(l);
            continue;
        }
        if (turn != last_turn) {
            rmwork::LogLine h;
            h.decision = decision;
            h.kind = rmwork::LogLine::Kind::HEADER;
            h.text = "Turn " + std::to_string(turn);
            out.push_back(h);
            last_turn = turn;
        }
        l.text = text;
        out.push_back(l);
    }
}

// Game and match results from the omniscient frame's game_end and match_end events.
static void read_results(Builder *b, const agents::SeatStream &omni, const rmstate::DecodedFrame &f) {
    if (!b->have_event_kinds) {
        b->event_kind_game_end = static_cast<uint64_t>(omni.enum_value("event_kind", "GAME_END"));
        b->event_kind_match_end = static_cast<uint64_t>(omni.enum_value("event_kind", "MATCH_END"));
        b->have_event_kinds = true;
    }
    const agents::Table ev(f, omni.schema(), "event");
    const agents::Table ep(f, omni.schema(), "event_player");
    rmwork::MatchResult &r = b->m->result;
    for (size_t i = 0; i < ev.rows(); ++i) {
        const uint64_t kind = ev.u(i, ev.col("kind"));
        const int winner = seat_index(ev.u(i, ev.col("player")));
        if (kind == b->event_kind_game_end) {
            const size_t g = static_cast<size_t>(ev.u(i, ev.col("game")));
            if (r.game_winners.size() <= g) r.game_winners.resize(g + 1, -1);
            r.game_winners[g] = winner;
        } else if (kind == b->event_kind_match_end) {
            r.known = true;
            r.winner = winner;
            for (size_t p = 0; p < ep.rows(); ++p) {
                if (ep.u(p, ep.col("event")) != i) continue;
                const int seat = seat_index(ep.u(p, ep.col("player")));
                if (seat >= 0) r.wins[static_cast<size_t>(seat)] = static_cast<int>(ep.u(p, ep.col("wins")));
            }
        }
    }
}

// One decision's three STATE frames (or the terminal ones): decoded, checked, indexed, logged.
static bool process_triple(Builder *b, const std::array<rmwork::FrameRef, rmwork::VIEW_COUNT> &refs, std::string *err) {
    rmwork::Match &m = *b->m;
    std::array<rmstate::DecodedFrame, rmwork::VIEW_COUNT> frames;
    for (int v = 0; v < rmwork::VIEW_COUNT; ++v) {
        std::string why;
        if (!decode_ref(m, m.stream(v), refs[static_cast<size_t>(v)], &frames[static_cast<size_t>(v)], &why)) {
            *err = std::string("decision ") + std::to_string(m.decisions.size() + 1) + ", " + VIEW_NAMES[v] +
                   "'s frame: " + why;
            return false;
        }
    }
    int game = 0;
    bool sideboarding = false, terminal = false;
    int64_t decision = 0;
    read_key(agents::Table(frames[2], m.stream(2).schema(), "header"), &game, &sideboarding, &decision, &terminal);
    for (int v = 0; v < 2; ++v) {
        int g = 0;
        bool sb = false, term = false;
        int64_t d = 0;
        read_key(agents::Table(frames[static_cast<size_t>(v)], m.stream(v).schema(), "header"), &g, &sb, &d, &term);
        if (g != game || sb != sideboarding || d != decision || term != terminal) {
            *err = std::string("decision ") + std::to_string(m.decisions.size() + 1) + ": " + VIEW_NAMES[v] +
                   "'s frame is of another decision than the omniscient frame";
            return false;
        }
    }
    const size_t index = m.decisions.size();
    if (m.stopped) {
        *err = "the replay has frames after decision " + std::to_string(index) +
               ", where the run stopped without a reply";
        return false;
    }
    if (terminal) {
        m.terminal = refs;
    } else {
        if (index > m.log.replies.size()) {
            *err = "the replay has more decisions than the log's " + std::to_string(m.log.replies.size());
            return false;
        }
        rmwork::DecisionInfo d;
        d.game = game;
        d.sideboarding = sideboarding;
        d.decision = decision;
        d.frames = refs;
        for (int v = 0; v < 2; ++v)
            if (agents::Table(frames[static_cast<size_t>(v)], m.stream(v).schema(), "menu").rows() > 0) d.decider = v;
        const agents::DecisionView ov(m.stream(2), frames[2], d.decider < 0 ? 0 : d.decider);
        fill_decision(&d, ov);
        if (index < m.log.replies.size()) {
            d.reply = m.log.replies[index];
            d.reply_text = reply_text(ov, d.reply);
        } else {
            d.reply = rmwork::REPLY_PENDING;
            m.stopped = true;
        }
        m.decisions.push_back(d);
    }
    // A terminal frame's events join the last decision.
    const int log_at = static_cast<int>(terminal && index > 0 ? index - 1 : index);
    for (int v = 0; v < 2; ++v) {
        const agents::DecisionView sv(m.stream(v), frames[static_cast<size_t>(v)], v);
        add_seat_lines(b, v, sv, log_at);
    }
    read_results(b, m.stream(2), frames[2]);
    return true;
}

// The games and sideboarding phases in decision order.
static void build_games(rmwork::Match *m) {
    m->games.clear();
    for (size_t i = 0; i < m->decisions.size(); ++i) {
        const rmwork::DecisionInfo &d = m->decisions[i];
        if (!m->games.empty() && m->games.back().game == d.game && m->games.back().sideboarding == d.sideboarding) {
            ++m->games.back().count;
            continue;
        }
        rmwork::GameSpan s;
        s.game = d.game;
        s.sideboarding = d.sideboarding;
        s.first = i;
        s.count = 1;
        s.label = std::string(d.sideboarding ? "Sideboarding " : "Game ") + std::to_string(d.game + 1);
        m->games.push_back(s);
    }
}

// "(REPLAY) [T3 | First Main | Player A] Input: 1", the line the replay writes as it takes a decision.
static bool marker_line(const std::string &line) {
    return line.compare(0, 10, "(REPLAY) [") == 0 && line.find("] Input: ") != std::string::npos;
}

static bool banner_line(const std::string &line) {
    return line.compare(0, 9, "robomage ") == 0 || line.compare(0, 12, "REPLAY MODE:") == 0 ||
           line.compare(0, 11, "Using seed:") == 0;
}

// "-------- TURN 2 (Player B) --------" -> "TURN 2 (Player B)"; "" when not a header.
static std::string header_text(const std::string &line) {
    if (line.size() < 6 || line.compare(0, 3, "---") != 0) return "";
    const size_t a = line.find_first_not_of('-');
    const size_t b = line.find_last_not_of('-');
    if (a == std::string::npos || b < a) return "";
    std::string t = line.substr(a, b - a + 1);
    while (!t.empty() && t.front() == ' ') t.erase(t.begin());
    while (!t.empty() && t.back() == ' ') t.pop_back();
    return t;
}

static void build_narrative(rmwork::Match *m) {
    std::vector<rmwork::LogLine> &out = m->logs[rmwork::VIEW_OMNI];
    out.clear();
    std::string text;
    if (!rmwork::read_file(m->narrative_path, &text)) return;
    const size_t n = m->decisions.size();
    size_t markers = 0;
    size_t start = 0;
    while (start < text.size()) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) nl = text.size();
        const std::string line = text.substr(start, nl - start);
        start = nl + 1;
        if (line.empty() || line.find_first_not_of(' ') == std::string::npos || banner_line(line)) continue;
        rmwork::LogLine l;
        l.decision = static_cast<int>(n == 0 ? 0 : std::min(markers, n - 1));
        if (marker_line(line)) {
            if (markers < n) {
                const rmwork::DecisionInfo &d = m->decisions[markers];
                l.kind = rmwork::LogLine::Kind::CHOICE;
                l.seat = d.decider;
                l.text = "» " + seat_label(d.decider) + ": " + d.reply_text;
            } else {
                l.text = line;
            }
            ++markers;
            out.push_back(l);
            continue;
        }
        const std::string h = header_text(line);
        if (!h.empty()) {
            l.kind = rmwork::LogLine::Kind::HEADER;
            l.text = h;
        } else {
            l.text = line;
        }
        out.push_back(l);
    }
    m->narrative_mapped = markers == decided_count(*m);
}

// The decisions that were answered: all of them, less the one a stopped run did not decide.
static size_t decided_count(const rmwork::Match &m) {
    return m.decisions.size() - (m.stopped ? 1 : 0);
}

static void finish_result(rmwork::Match *m) {
    rmwork::MatchResult &r = m->result;
    if (!r.known) {
        r.text = m->stopped ? "no result: the run stopped at decision " + std::to_string(m->decisions.size()) +
                                  (m->log.quit ? " (quit)" : " (--max-decisions)")
                            : "no result";
        return;
    }
    if (r.winner < 0) {
        r.text = "a draw";
    } else {
        r.text = seat_label(r.winner) + " wins";
    }
    if (m->bo3()) r.text += " " + std::to_string(r.wins[0]) + "-" + std::to_string(r.wins[1]);
}

// Appends "RMSTATE <n>\n" + payload at `*end`, which moves past it; the payload starts at the new
// end less its size.
static bool write_record(int fd, const std::string &payload, uint64_t *end, std::string *err) {
    const std::string record = "RMSTATE " + std::to_string(payload.size()) + "\n" + payload;
    size_t done = 0;
    while (done < record.size()) {
        const ssize_t n = pwrite(fd, record.data() + done, record.size() - done, static_cast<off_t>(*end + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            *err = std::string("cannot write the stream file: ") + std::strerror(errno);
            return false;
        }
        done += static_cast<size_t>(n);
    }
    *end += record.size();
    return true;
}

namespace rmwork {

// A live match's index state: its logs' turn headers, the event kinds, the stream file's size.
struct LiveState {
        Builder builder;
        uint64_t end = 0;
};

bool split_turn_prefix(const std::string &line, int *turn, std::string *text) {
    if (line.size() < 4 || line[0] != 'T' || !std::isdigit(static_cast<unsigned char>(line[1]))) return false;
    const size_t colon = line.find(": ");
    if (colon == std::string::npos) return false;
    for (size_t i = 1; i < colon; ++i)
        if (!std::isdigit(static_cast<unsigned char>(line[i]))) return false;
    *turn = std::atoi(line.c_str() + 1);
    *text = line.substr(colon + 2);
    return true;
}

const char *view_name(int view) {
    return view >= 0 && view < VIEW_COUNT ? VIEW_NAMES[view] : "?";
}

Match::Match() = default;

Match::~Match() {
    if (fd_ >= 0) close(fd_);
}

bool Match::read_payload(const FrameRef &ref, std::string *out, std::string *err) const {
    if (!ref.valid() || fd_ < 0) {
        *err = "no such frame";
        return false;
    }
    out->resize(ref.size);
    size_t done = 0;
    while (done < ref.size) {
        const ssize_t n = pread(fd_, &(*out)[done], ref.size - done, static_cast<off_t>(ref.offset + done));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            *err = "cannot read " + dump_path + (n < 0 ? std::string(": ") + std::strerror(errno) : ": truncated");
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

std::shared_ptr<const rmstate::DecodedFrame> Match::frame(size_t index, int view, std::string *err) const {
    if (view < 0 || view >= VIEW_COUNT || index > decisions.size()) {
        *err = "no such decision";
        return nullptr;
    }
    const uint64_t key = static_cast<uint64_t>(index) * VIEW_COUNT + static_cast<uint64_t>(view);
    for (auto it = lru_.begin(); it != lru_.end(); ++it) {
        if (it->first != key) continue;
        lru_.splice(lru_.begin(), lru_, it);
        return lru_.front().second;
    }
    const FrameRef &ref = index == decisions.size() ? terminal[static_cast<size_t>(view)]
                                                    : decisions[index].frames[static_cast<size_t>(view)];
    auto f = std::make_shared<rmstate::DecodedFrame>();
    if (!decode_ref(*this, stream(view), ref, f.get(), err)) return nullptr;
    lru_.emplace_front(key, f);
    if (lru_.size() > LRU_CAPACITY) lru_.pop_back();
    return f;
}

int Match::span_of(size_t index) const {
    for (size_t g = 0; g < games.size(); ++g)
        if (index >= games[g].first && index < games[g].first + games[g].count) return static_cast<int>(g);
    return -1;
}

int Match::find_decision(int game, bool sideboarding, int64_t decision) const {
    const auto it = by_key_.find(std::make_tuple(game, sideboarding, decision));
    return it == by_key_.end() ? -1 : it->second;
}

std::string Match::title() const {
    return log.deck_a + " vs " + log.deck_b;
}

bool index_match(Match *m, const std::function<bool(double)> &progress, std::string *err) {
    for (auto &s : m->streams_) s.reset(new agents::SeatStream());
    m->decisions.clear();
    m->by_key_.clear();
    m->match_frames_.clear();
    m->lru_.clear();
    for (auto &l : m->logs) l.clear();
    m->result = MatchResult();
    m->terminal = {};
    m->stopped = false;
    if (m->fd_ >= 0) close(m->fd_);
    m->fd_ = open(m->dump_path.c_str(), O_RDONLY | O_CLOEXEC);
    FILE *f = std::fopen(m->dump_path.c_str(), "rb");
    if (m->fd_ < 0 || !f) {
        if (f) std::fclose(f);
        *err = "cannot open " + m->dump_path + ": " + std::strerror(errno);
        return false;
    }
    struct stat st;
    const double total = fstat(m->fd_, &st) == 0 && st.st_size > 0 ? static_cast<double>(st.st_size) : 1.0;
    Builder b;
    b.m = m;
    rmstate::RecordReader reader(f);
    std::array<FrameRef, VIEW_COUNT> triple;
    int filled = 0;
    bool ok = true;
    std::string payload;
    rmstate::DecodeResult res;
    while (ok) {
        if (!reader.next(&payload, &res)) {
            if (!res.ok()) {
                *err = "record " + std::to_string(reader.count() + 1) + " of the dump: " + res.message;
                ok = false;
            }
            break;
        }
        const long end = std::ftell(f);
        if (payload.size() < FRAME_HEADER_BYTES || end < 0) {
            *err = "record " + std::to_string(reader.count()) + " of the dump is not a frame";
            ok = false;
            break;
        }
        const uint8_t kind = static_cast<uint8_t>(payload[6]);
        const uint8_t viewer = static_cast<uint8_t>(payload[7]);
        if (viewer >= VIEW_COUNT) {
            *err = "record " + std::to_string(reader.count()) + " of the dump names viewer " + std::to_string(viewer);
            ok = false;
            break;
        }
        if (kind != static_cast<uint8_t>(rmstate::FrameKind::STATE)) {
            FrameRef ref;
            ref.offset = static_cast<uint64_t>(end) - payload.size();
            ref.size = static_cast<uint32_t>(payload.size());
            m->match_frames_.emplace_back(viewer, ref);
            if (!m->streams_[viewer]->feed_match_frame(payload, err)) {
                *err = std::string(VIEW_NAMES[viewer]) + "'s once-per-match frame: " + *err;
                ok = false;
            }
            continue;
        }
        if (viewer != filled) {
            *err = "record " + std::to_string(reader.count()) + ": a STATE frame of the " + VIEW_NAMES[viewer] +
                   " viewer where the " + VIEW_NAMES[filled] + " viewer's was expected";
            ok = false;
            break;
        }
        triple[viewer].offset = static_cast<uint64_t>(end) - payload.size();
        triple[viewer].size = static_cast<uint32_t>(payload.size());
        if (++filled < VIEW_COUNT) continue;
        filled = 0;
        if (!process_triple(&b, triple, err)) {
            ok = false;
            break;
        }
        if (!progress(static_cast<double>(end) / total)) {
            *err = "cancelled";
            ok = false;
        }
    }
    std::fclose(f);
    if (!ok) return false;
    if (filled != 0) {
        *err = "the dump ends inside a decision's frames";
        return false;
    }
    return finish_match_index(m, err);
}

bool finish_match_index(Match *m, std::string *err) {
    m->by_key_.clear();
    if (decided_count(*m) != m->log.replies.size()) {
        *err = "the replay produced " + std::to_string(m->decisions.size()) + " decisions; the log holds " +
               std::to_string(m->log.replies.size());
        return false;
    }
    for (size_t i = 0; i < m->decisions.size(); ++i) {
        const DecisionInfo &d = m->decisions[i];
        if (!m->by_key_.emplace(std::make_tuple(d.game, d.sideboarding, d.decision), static_cast<int>(i)).second) {
            *err = "two decisions share the key (game " + std::to_string(d.game) + ", decision " +
                   std::to_string(d.decision) + ")";
            return false;
        }
    }
    build_games(m);
    build_narrative(m);
    finish_result(m);
    return true;
}

bool begin_live_match(Match *m, int seat, const std::string &stream_path, std::string *err) {
    if (seat != VIEW_A && seat != VIEW_B) {
        *err = "a live match is a seat's";
        return false;
    }
    for (auto &st : m->streams_) st.reset(new agents::SeatStream());
    m->live = true;
    m->live_seat = seat;
    m->finished = false;
    m->dump_path = stream_path;
    m->live_state_.reset(new LiveState());
    m->live_state_->builder.m = m;
    if (m->fd_ >= 0) close(m->fd_);
    m->fd_ = open(stream_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (m->fd_ < 0) {
        *err = "cannot create " + stream_path + ": " + std::strerror(errno);
        return false;
    }
    m->result = MatchResult();
    finish_result(m);
    return true;
}

bool append_live_frame(Match *m, const std::string &payload, LiveFrame *out, std::string *err) {
    *out = LiveFrame();
    if (!m->live || !m->live_state_ || m->fd_ < 0) {
        *err = "not a live match";
        return false;
    }
    if (payload.size() < FRAME_HEADER_BYTES) {
        *err = "a record that is not a frame";
        return false;
    }
    const uint8_t kind = static_cast<uint8_t>(payload[6]);
    const uint8_t viewer = static_cast<uint8_t>(payload[7]);
    if (viewer != m->live_seat) {
        *err =
            std::string("a frame for the ") + view_name(viewer) + " viewer on " + view_name(m->live_seat) + "'s stream";
        return false;
    }
    LiveState &ls = *m->live_state_;
    if (!write_record(m->fd_, payload, &ls.end, err)) return false;
    agents::SeatStream &stream = *m->streams_[viewer];
    if (kind != static_cast<uint8_t>(rmstate::FrameKind::STATE)) {
        if (!stream.feed_match_frame(payload, err)) {
            *err = "a once-per-match frame: " + *err;
            return false;
        }
        FrameRef once;
        once.offset = ls.end - payload.size();
        once.size = static_cast<uint32_t>(payload.size());
        m->match_frames_.emplace_back(viewer, once);
        return true;
    }
    FrameRef ref;
    ref.offset = ls.end - payload.size();
    ref.size = static_cast<uint32_t>(payload.size());
    rmstate::DecodedFrame frame;
    if (!decode_ref(*m, stream, ref, &frame, err)) return false;
    out->state = true;
    int game = 0;
    bool sideboarding = false, terminal = false;
    int64_t decision = 0;
    read_key(agents::Table(frame, stream.schema(), "header"), &game, &sideboarding, &decision, &terminal);
    const agents::DecisionView v(stream, frame, viewer);
    const size_t index = m->decisions.size();
    if (terminal) {
        out->terminal = true;
        m->terminal[viewer] = ref;
        m->finished = true;
    } else if (v.n() > 0) {
        if (!m->by_key_.emplace(std::make_tuple(game, sideboarding, decision), static_cast<int>(index)).second) {
            *err = "two frames of decision " + std::to_string(decision) + " (game " + std::to_string(game + 1) + ")";
            return false;
        }
        DecisionInfo d;
        d.game = game;
        d.sideboarding = sideboarding;
        d.decision = decision;
        d.frames[viewer] = ref;
        d.decider = viewer;
        fill_decision(&d, v);
        d.reply = REPLY_PENDING;
        m->decisions.push_back(d);
        out->decision = true;
        out->index = index;
    }
    add_seat_lines(&ls.builder, viewer, v, static_cast<int>(terminal && index > 0 ? index - 1 : index));
    read_results(&ls.builder, stream, frame);
    build_games(m);
    finish_result(m);
    return true;
}

void set_live_reply(Match *m, size_t index, int reply) {
    if (index >= m->decisions.size()) return;
    DecisionInfo &d = m->decisions[index];
    d.reply = reply;
    std::string err;
    const std::shared_ptr<const rmstate::DecodedFrame> f = m->frame(index, d.decider, &err);
    if (!f) return;
    d.reply_text = reply_text(agents::DecisionView(m->stream(d.decider), *f, d.decider), reply);
    // The seat's log names its own choice after the decision's events.
    LogLine l;
    l.decision = static_cast<int>(index);
    l.kind = LogLine::Kind::CHOICE;
    l.seat = d.decider;
    l.text = "\xC2\xBB You: " + d.reply_text;
    m->logs[static_cast<size_t>(d.decider)].push_back(l);
}

}  // namespace rmwork
