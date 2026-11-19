#include "match_index.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

#include "files.h"
#include "replay_cache.h"

namespace {

const char MAGIC[8] = {'R', 'M', 'W', 'K', 'I', 'D', 'X', '\n'};

// Appends little-endian values and length-prefixed strings.
struct Writer {
        std::string out;
        void u8(uint8_t v) { out += static_cast<char>(v); }
        void u32(uint32_t v) {
            for (int i = 0; i < 4; ++i) out += static_cast<char>((v >> (8 * i)) & 0xff);
        }
        void u64(uint64_t v) {
            for (int i = 0; i < 8; ++i) out += static_cast<char>((v >> (8 * i)) & 0xff);
        }
        void i32(int32_t v) { u32(static_cast<uint32_t>(v)); }
        void i64(int64_t v) { u64(static_cast<uint64_t>(v)); }
        void str(const std::string &s) {
            u32(static_cast<uint32_t>(s.size()));
            out += s;
        }
        void frame(const rmwork::FrameRef &f) {
            u64(f.offset);
            u32(f.size);
        }
};

// Reads what Writer wrote; every read past the end fails the reader (ok false) and returns zero.
struct Reader {
        const std::string &in;
        size_t at = 0;
        bool ok = true;
        explicit Reader(const std::string &s) : in(s) {}
        bool take(size_t n) {
            if (!ok || in.size() - at < n) return ok = false;
            return true;
        }
        uint64_t le(int bytes) {
            if (!take(static_cast<size_t>(bytes))) return 0;
            uint64_t v = 0;
            for (int i = 0; i < bytes; ++i) v |= static_cast<uint64_t>(static_cast<uint8_t>(in[at + i])) << (8 * i);
            at += static_cast<size_t>(bytes);
            return v;
        }
        uint8_t u8() { return static_cast<uint8_t>(le(1)); }
        uint32_t u32() { return static_cast<uint32_t>(le(4)); }
        uint64_t u64() { return le(8); }
        int32_t i32() { return static_cast<int32_t>(u32()); }
        int64_t i64() { return static_cast<int64_t>(u64()); }
        std::string str() {
            const uint32_t n = u32();
            if (!take(n)) return "";
            std::string s = in.substr(at, n);
            at += n;
            return s;
        }
        rmwork::FrameRef frame() {
            rmwork::FrameRef f;
            f.offset = u64();
            f.size = u32();
            return f;
        }
        // A count of items each at least `item_bytes` long, refused when the rest cannot hold them.
        uint32_t count(size_t item_bytes) {
            const uint32_t n = u32();
            if (ok && static_cast<uint64_t>(n) * item_bytes > in.size() - at) ok = false;
            return ok ? n : 0;
        }
};

}  // namespace

namespace rmwork {

bool write_match_index(const Match &m, const std::string &path, uint64_t log_hash, std::string *err) {
    const FileStamp dump = file_stamp(m.dump_path);
    if (!dump.found) {
        *err = "no dump at " + m.dump_path;
        return false;
    }
    Writer w;
    w.out.append(MAGIC, sizeof MAGIC);
    w.u32(MATCH_INDEX_VERSION);
    w.u64(dump.size);
    w.i64(dump.mtime_ns);
    w.u64(log_hash);
    w.u32(static_cast<uint32_t>(m.match_frames_.size()));
    for (const auto &f : m.match_frames_) {
        w.u8(static_cast<uint8_t>(f.first));
        w.frame(f.second);
    }
    for (const FrameRef &f : m.terminal) w.frame(f);
    w.u8(m.stopped ? 1 : 0);
    w.u32(static_cast<uint32_t>(m.decisions.size()));
    for (const DecisionInfo &d : m.decisions) {
        w.i32(d.game);
        w.u8(d.sideboarding ? 1 : 0);
        w.i64(d.decision);
        for (const FrameRef &f : d.frames) w.frame(f);
        w.i32(d.decider);
        w.i32(d.turn);
        w.str(d.phase);
        w.str(d.header);
        w.i32(d.menu_rows);
        w.i32(d.reply);
        w.str(d.reply_text);
        w.i64(d.life[0]);
        w.i64(d.life[1]);
    }
    for (int v = 0; v < 2; ++v) {
        const std::vector<LogLine> &log = m.logs[static_cast<size_t>(v)];
        w.u32(static_cast<uint32_t>(log.size()));
        for (const LogLine &l : log) {
            w.str(l.text);
            w.i32(l.decision);
            w.u8(static_cast<uint8_t>(l.kind));
            w.i32(l.seat);
        }
    }
    const MatchResult &r = m.result;
    w.u8(r.known ? 1 : 0);
    w.i32(r.winner);
    w.i32(r.wins[0]);
    w.i32(r.wins[1]);
    w.u32(static_cast<uint32_t>(r.game_winners.size()));
    for (int g : r.game_winners) w.i32(g);
    w.u64(fnv1a64(w.out));
    return write_file_atomic(path, w.out, err);
}

bool read_match_index(Match *m, const std::string &path, uint64_t log_hash, std::string *err) {
    std::string bytes;
    if (!read_file(path, &bytes)) {
        *err = "no index";
        return false;
    }
    if (bytes.size() < sizeof MAGIC + 4 + 8 || std::memcmp(bytes.data(), MAGIC, sizeof MAGIC) != 0) {
        *err = "not an index";
        return false;
    }
    const std::string body = bytes.substr(0, bytes.size() - 8);
    Reader tail(bytes);
    tail.at = bytes.size() - 8;
    if (tail.u64() != fnv1a64(body)) {
        *err = "the index is corrupt";
        return false;
    }
    Reader r(body);
    r.at = sizeof MAGIC;
    if (r.u32() != MATCH_INDEX_VERSION) {
        *err = "an index of another version";
        return false;
    }
    const FileStamp dump = file_stamp(m->dump_path);
    const uint64_t size = r.u64();
    const int64_t mtime = r.i64();
    if (!dump.found || dump.size != size || dump.mtime_ns != mtime) {
        *err = "the index is of another dump";
        return false;
    }
    if (r.u64() != log_hash) {
        *err = "the index is of another log";
        return false;
    }

    for (auto &s : m->streams_) s.reset(new agents::SeatStream());
    m->decisions.clear();
    m->match_frames_.clear();
    m->lru_.clear();
    for (auto &l : m->logs) l.clear();
    m->result = MatchResult();
    m->terminal = {};
    const uint32_t frames = r.count(13);
    for (uint32_t i = 0; i < frames; ++i) {
        const int viewer = r.u8();
        const FrameRef f = r.frame();
        m->match_frames_.emplace_back(viewer, f);
    }
    for (FrameRef &f : m->terminal) f = r.frame();
    m->stopped = r.u8() != 0;
    const uint32_t n = r.count(64);
    m->decisions.reserve(n);
    for (uint32_t i = 0; i < n && r.ok; ++i) {
        DecisionInfo d;
        d.game = r.i32();
        d.sideboarding = r.u8() != 0;
        d.decision = r.i64();
        for (FrameRef &f : d.frames) f = r.frame();
        d.decider = r.i32();
        d.turn = r.i32();
        d.phase = r.str();
        d.header = r.str();
        d.menu_rows = r.i32();
        d.reply = r.i32();
        d.reply_text = r.str();
        d.life[0] = r.i64();
        d.life[1] = r.i64();
        m->decisions.push_back(std::move(d));
    }
    for (int v = 0; v < 2; ++v) {
        std::vector<LogLine> &log = m->logs[static_cast<size_t>(v)];
        const uint32_t lines = r.count(13);
        log.reserve(lines);
        for (uint32_t i = 0; i < lines && r.ok; ++i) {
            LogLine l;
            l.text = r.str();
            l.decision = r.i32();
            const uint8_t kind = r.u8();
            l.kind = kind == 1 ? LogLine::Kind::HEADER : kind == 2 ? LogLine::Kind::CHOICE : LogLine::Kind::EVENT;
            l.seat = r.i32();
            log.push_back(std::move(l));
        }
    }
    MatchResult &res = m->result;
    res.known = r.u8() != 0;
    res.winner = r.i32();
    res.wins[0] = r.i32();
    res.wins[1] = r.i32();
    const uint32_t games = r.count(4);
    for (uint32_t g = 0; g < games; ++g) res.game_winners.push_back(r.i32());
    if (!r.ok || r.at != body.size()) {
        *err = "the index is truncated";
        return false;
    }

    if (m->fd_ >= 0) close(m->fd_);
    m->fd_ = open(m->dump_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (m->fd_ < 0) {
        *err = "cannot open " + m->dump_path + ": " + std::strerror(errno);
        return false;
    }
    for (const auto &f : m->match_frames_) {
        if (f.first < 0 || f.first >= VIEW_COUNT) {
            *err = "the index names viewer " + std::to_string(f.first);
            return false;
        }
        std::string payload;
        if (!m->read_payload(f.second, &payload, err) ||
            !m->streams_[static_cast<size_t>(f.first)]->feed_match_frame(payload, err))
            return false;
    }
    if (!finish_match_index(m, err)) return false;
    m->index_cached = true;
    return true;
}

}  // namespace rmwork
