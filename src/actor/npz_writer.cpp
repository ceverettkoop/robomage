#include "npz_writer.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <string>
#include <vector>

#include "error.h"

namespace {

// Recursively create `dir` (like `mkdir -p`); no-op on existing dirs.
void mkdirs(const std::string& dir) {
    std::string acc;
    size_t i = 0;
    if (!dir.empty() && dir[0] == '/') {
        acc = "/";
        i = 1;
    }
    while (i <= dir.size()) {
        if (i == dir.size() || dir[i] == '/') {
            if (!acc.empty() && acc != "/") {
                if (::mkdir(acc.c_str(), 0777) != 0 && errno != EEXIST)
                    fatal_error("npz_writer: cannot create dir: " + acc);
            }
            if (i < dir.size()) acc += '/';
        } else {
            acc += dir[i];
        }
        i++;
    }
}

// ── standard CRC32 (IEEE 802.3, zlib polynomial 0xEDB88320) ────────────────
// Table-driven; matches zlib's crc32 so np.load's zip integrity check passes.
struct Crc32Table {
    uint32_t t[256];
    Crc32Table() {
        for (uint32_t n = 0; n < 256; n++) {
            uint32_t c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[n] = c;
        }
    }
};

// Fold `len` bytes into a running CRC state (internal, pre-final-xor form; seed
// with 0xFFFFFFFF and finalize with ^0xFFFFFFFF). Lets one CRC span the .npy
// header and its data across two calls.
uint32_t crc32_update(uint32_t crc, const void* data, size_t len) {
    static const Crc32Table table;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < len; i++)
        crc = table.t[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    return crc;
}

// ── little-endian scalar writers ───────────────────────────────────────────
void put_u16(std::vector<unsigned char>& b, uint16_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}
void put_u32(std::vector<unsigned char>& b, uint32_t v) {
    b.push_back(static_cast<unsigned char>(v & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
    b.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
}

// Build the shape tuple text, e.g. "(3, 6700)" or "(5,)" for a 1-D array.
std::string shape_tuple(const std::vector<size_t>& shape) {
    std::string s = "(";
    for (size_t i = 0; i < shape.size(); i++) s += std::to_string(shape[i]) + ", ";
    // s is now "(a, b, ... , ". numpy renders a 1-D shape as "(n,)" (keep the
    // comma, drop the space); multi-D as "(a, b)" (drop the trailing ", ").
    if (shape.size() == 1) {
        s.pop_back();  // "(n, " -> "(n,"
    } else if (!shape.empty()) {
        s.erase(s.size() - 2);  // "(a, b, " -> "(a, b"
    }
    s += ")";
    return s;
}

// Full .npy header block (magic + version + len + padded dict). 64-byte aligned.
std::string npy_header(const char* descr, const std::vector<size_t>& shape) {
    std::string dict = "{'descr': '";
    dict += descr;
    dict += "', 'fortran_order': False, 'shape': ";
    dict += shape_tuple(shape);
    dict += ", }";
    // preamble = 6 (magic) + 2 (version) + 2 (header-len u16) = 10 bytes; the
    // total up to and including the trailing '\n' must be a multiple of 64.
    size_t total = 10 + dict.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');

    std::string hdr;
    hdr.push_back('\x93');
    hdr += "NUMPY";
    hdr.push_back('\x01');  // version major 1
    hdr.push_back('\x00');  // version minor 0
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    hdr.push_back(static_cast<char>(hlen & 0xFF));
    hdr.push_back(static_cast<char>((hlen >> 8) & 0xFF));
    hdr += dict;
    return hdr;
}

}  // namespace

// ── NpzWriter ──────────────────────────────────────────────────────────────
NpzWriter::NpzWriter(const std::string& path)
    : final_path_(path), tmp_path_(path + ".tmp"), fp_(nullptr), offset_(0),
      finished_(false) {
    fp_ = std::fopen(tmp_path_.c_str(), "wb");
    if (!fp_) fatal_error("npz_writer: cannot open temp file: " + tmp_path_);
}

NpzWriter::~NpzWriter() {
    if (!finished_) finish();
}

void NpzWriter::add_float(const std::string& name, const float* data,
                          const std::vector<size_t>& shape) {
    add_member(name, "<f4", sizeof(float), data, shape);
}

void NpzWriter::add_bool(const std::string& name, const uint8_t* data,
                         const std::vector<size_t>& shape) {
    add_member(name, "|b1", sizeof(uint8_t), data, shape);
}

void NpzWriter::add_uint8(const std::string& name, const uint8_t* data,
                          const std::vector<size_t>& shape) {
    add_member(name, "|u1", sizeof(uint8_t), data, shape);
}

void NpzWriter::add_int8(const std::string& name, const int8_t* data,
                         const std::vector<size_t>& shape) {
    add_member(name, "|i1", sizeof(int8_t), data, shape);
}

void NpzWriter::add_int16(const std::string& name, const int16_t* data,
                          const std::vector<size_t>& shape) {
    add_member(name, "<i2", sizeof(int16_t), data, shape);
}

void NpzWriter::add_int32(const std::string& name, const int32_t* data,
                          const std::vector<size_t>& shape) {
    add_member(name, "<i4", sizeof(int32_t), data, shape);
}

void NpzWriter::add_int64(const std::string& name, const int64_t* data,
                          const std::vector<size_t>& shape) {
    add_member(name, "<i8", sizeof(int64_t), data, shape);
}

void NpzWriter::add_member(const std::string& name, const char* descr,
                           size_t elem_size, const void* data,
                           const std::vector<size_t>& shape) {
    size_t count = 1;
    for (size_t d : shape) count *= d;
    const std::string npy = npy_header(descr, shape);
    const size_t data_bytes = count * elem_size;
    const size_t uncompressed = npy.size() + data_bytes;

    // CRC over the full .npy payload (header + raw data).
    uint32_t crc = 0xFFFFFFFFu;
    crc = crc32_update(crc, npy.data(), npy.size());
    crc = crc32_update(crc, data, data_bytes);
    crc ^= 0xFFFFFFFFu;

    const std::string filename = name + ".npy";

    // Local file header (PK\x03\x04). STORE (method 0), fixed 1980-01-01 stamp.
    std::vector<unsigned char> lh;
    put_u32(lh, 0x04034b50u);
    put_u16(lh, 20);    // version needed
    put_u16(lh, 0);     // flags
    put_u16(lh, 0);     // method = store
    put_u16(lh, 0);     // mod time
    put_u16(lh, 0x21);  // mod date = 1980-01-01
    put_u32(lh, crc);
    put_u32(lh, static_cast<uint32_t>(uncompressed));  // compressed size
    put_u32(lh, static_cast<uint32_t>(uncompressed));  // uncompressed size
    put_u16(lh, static_cast<uint16_t>(filename.size()));
    put_u16(lh, 0);  // extra len
    lh.insert(lh.end(), filename.begin(), filename.end());

    // All zip size/offset fields are 32-bit (no Zip64 support): a member or
    // cumulative archive offset past 4 GiB would truncate silently and corrupt
    // the shard, so fail loudly instead. Unreachable at current shard sizes
    // (~110 MB); trips only after a large FLUSH_SAMPLES / obs-width change.
    if (uncompressed > UINT32_MAX ||
        static_cast<uint64_t>(offset_) + lh.size() + uncompressed > UINT32_MAX)
        fatal_error("npz_writer: member or archive exceeds 4 GiB (no Zip64 "
                    "support): " + filename);

    Member m;
    m.filename = filename;
    m.crc = crc;
    m.size = static_cast<uint32_t>(uncompressed);
    m.offset = offset_;

    if (std::fwrite(lh.data(), 1, lh.size(), fp_) != lh.size() ||
        std::fwrite(npy.data(), 1, npy.size(), fp_) != npy.size() ||
        std::fwrite(data, 1, data_bytes, fp_) != data_bytes)
        fatal_error("npz_writer: write failed for member " + filename);

    offset_ += static_cast<uint32_t>(lh.size() + uncompressed);
    members_.push_back(m);
}

void NpzWriter::finish() {
    if (finished_) return;
    finished_ = true;

    const uint32_t cd_start = offset_;
    std::vector<unsigned char> cd;
    for (const Member& m : members_) {
        put_u32(cd, 0x02014b50u);  // central dir signature
        put_u16(cd, 20);           // version made by
        put_u16(cd, 20);           // version needed
        put_u16(cd, 0);            // flags
        put_u16(cd, 0);            // method = store
        put_u16(cd, 0);            // mod time
        put_u16(cd, 0x21);         // mod date
        put_u32(cd, m.crc);
        put_u32(cd, m.size);       // compressed
        put_u32(cd, m.size);       // uncompressed
        put_u16(cd, static_cast<uint16_t>(m.filename.size()));
        put_u16(cd, 0);            // extra len
        put_u16(cd, 0);            // comment len
        put_u16(cd, 0);            // disk number start
        put_u16(cd, 0);            // internal attrs
        put_u32(cd, 0);            // external attrs
        put_u32(cd, m.offset);     // local-header offset
        cd.insert(cd.end(), m.filename.begin(), m.filename.end());
    }

    std::vector<unsigned char> eocd;
    put_u32(eocd, 0x06054b50u);
    put_u16(eocd, 0);  // this disk
    put_u16(eocd, 0);  // disk with central dir
    put_u16(eocd, static_cast<uint16_t>(members_.size()));
    put_u16(eocd, static_cast<uint16_t>(members_.size()));
    put_u32(eocd, static_cast<uint32_t>(cd.size()));
    put_u32(eocd, cd_start);
    put_u16(eocd, 0);  // comment len

    if (std::fwrite(cd.data(), 1, cd.size(), fp_) != cd.size() ||
        std::fwrite(eocd.data(), 1, eocd.size(), fp_) != eocd.size())
        fatal_error("npz_writer: write failed for central directory");
    if (std::fclose(fp_) != 0)
        fatal_error("npz_writer: close failed for " + tmp_path_);
    fp_ = nullptr;
    if (std::rename(tmp_path_.c_str(), final_path_.c_str()) != 0)
        fatal_error("npz_writer: rename failed: " + tmp_path_ + " -> " + final_path_);
}

// ── ShardAccumulator ───────────────────────────────────────────────────────
ShardAccumulator::ShardAccumulator(std::string out_dir, size_t flush_samples,
                                   size_t obs_width, size_t act_width)
    : out_dir_(std::move(out_dir)), flush_threshold_(flush_samples),
      obs_w_(obs_width), act_w_(act_width), buffered_(0), total_(0), shard_n_(0) {
    mkdirs(out_dir_);
}

void ShardAccumulator::add_sample(const float* obs, const float* pi, float z,
                                  const uint8_t* mask, float q, uint8_t explored,
                                  float td_q, const RootDiag* diag) {
    obs_.insert(obs_.end(), obs, obs + obs_w_);
    pi_.insert(pi_.end(), pi, pi + act_w_);
    z_.push_back(z);
    mask_.insert(mask_.end(), mask, mask + act_w_);
    q_.push_back(q);
    explored_.push_back(explored);
    td_q_.push_back(td_q);
    diags_.push_back(diag != nullptr ? *diag : RootDiag());
    buffered_++;
}

void ShardAccumulator::maybe_flush() {
    if (buffered_ >= flush_threshold_) flush();
}

void ShardAccumulator::flush_final() {
    if (buffered_ > 0) flush();
}

void ShardAccumulator::flush_match(const MatchReplayMeta& meta) {
    if (buffered_ == 0) return;
    // The sidecars read the buffers, so write them before the flush clears.
    const std::string path = next_shard_path();
    write_diag(path);
    write_replay(path, meta);
    flush_to(path);
}

std::string ShardAccumulator::next_shard_path() const {
    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
    return out_dir_ + "/shard_" + ts + "_" +
           std::to_string(static_cast<long>(getpid())) + "_" +
           std::to_string(shard_n_) + ".npz";
}

void ShardAccumulator::flush() {
    if (buffered_ > 0) flush_to(next_shard_path());
}

void ShardAccumulator::flush_to(const std::string& path) {
    {
        NpzWriter w(path);
        w.add_float("obs", obs_.data(), {buffered_, obs_w_});
        w.add_float("pi", pi_.data(), {buffered_, act_w_});
        w.add_float("z", z_.data(), {buffered_});
        w.add_bool("mask", mask_.data(), {buffered_, act_w_});
        w.add_float("q", q_.data(), {buffered_});
        w.add_uint8("explored", explored_.data(), {buffered_});
        w.add_float("td_q", td_q_.data(), {buffered_});
        w.finish();
    }

    shards_.push_back(path);
    total_ += buffered_;
    shard_n_++;
    clear_buffers();
}

void ShardAccumulator::clear_buffers() {
    obs_.clear();
    pi_.clear();
    z_.clear();
    mask_.clear();
    q_.clear();
    explored_.clear();
    td_q_.clear();
    diags_.clear();
    buffered_ = 0;
}

// The `.diag` sidecar: shard_record.DIAG_KEYS with pack_diag's exact dtypes
// and fixed widths (menu vectors MAX_ACTIONS wide, per-world vectors the
// widest row's world count, follow_path one column), row index == shard row.
// The extension is deliberately not .npz (readers glob shard_*.npz).
void ShardAccumulator::write_diag(const std::string& shard_path) const {
    const size_t n = buffered_;
    size_t w_max = 1;
    for (const RootDiag& d : diags_) {
        w_max = std::max(w_max, d.world_seeds.size());
        w_max = std::max(w_max, d.world_visits.size());
        w_max = std::max(w_max, d.world_values.size());
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<int8_t> kind(n, 0);
    std::vector<int32_t> visits(n * act_w_, 0);
    std::vector<float> priors(n * act_w_, 0.0f), q_act(n * act_w_, 0.0f),
        w_sum(n * act_w_, 0.0f);
    std::vector<float> root_value(n, 0.0f);
    std::vector<int32_t> sims_run(n, 0), sim_steps(n, 0), reused(n, 0), memo(n, 0);
    std::vector<uint8_t> stopped(n, 0);
    std::vector<float> tb(n, nan), tb_min(n, nan);
    std::vector<int16_t> n_worlds(n, 0);
    std::vector<int64_t> world_seeds(n * w_max, -1);
    std::vector<int32_t> world_visits(n * w_max, 0);
    std::vector<float> world_values(n * w_max, nan);
    std::vector<int32_t> origin_row(n, -1);
    std::vector<int32_t> follow_path(n, -1);
    std::vector<uint8_t> follow_worlds(n * w_max, 0);
    for (size_t i = 0; i < n; i++) {
        const RootDiag& d = diags_[i];
        if (d.kind == DIAG_KIND_NONE) continue;
        kind[i] = d.kind;
        const size_t nc = std::min(static_cast<size_t>(d.num_choices), act_w_);
        for (size_t k = 0; k < nc; k++) {
            if (k < d.visits.size()) visits[i * act_w_ + k] = d.visits[k];
            if (k < d.priors.size()) priors[i * act_w_ + k] = d.priors[k];
            if (k < d.q_act.size()) q_act[i * act_w_ + k] = d.q_act[k];
            if (k < d.w_sum.size()) w_sum[i * act_w_ + k] = d.w_sum[k];
        }
        root_value[i] = d.root_value;
        sims_run[i] = d.sims_run;
        sim_steps[i] = d.sim_steps;
        memo[i] = d.memo_hits;
        const size_t nw = std::max({d.world_seeds.size(), d.world_visits.size(),
                                    d.world_values.size()});
        n_worlds[i] = static_cast<int16_t>(nw);
        for (size_t w = 0; w < nw && w < w_max; w++) {
            if (w < d.world_seeds.size()) world_seeds[i * w_max + w] = d.world_seeds[w];
            if (w < d.world_visits.size())
                world_visits[i * w_max + w] = d.world_visits[w];
            if (w < d.world_values.size())
                world_values[i * w_max + w] = d.world_values[w];
        }
    }
    std::string stem = shard_path.substr(0, shard_path.size() - 4);  // strip .npz
    NpzWriter w(stem + ".diag");
    w.add_int8("kind", kind.data(), {n});
    w.add_int32("visits", visits.data(), {n, act_w_});
    w.add_float("priors", priors.data(), {n, act_w_});
    w.add_float("q_act", q_act.data(), {n, act_w_});
    w.add_float("w_sum", w_sum.data(), {n, act_w_});
    w.add_float("root_value", root_value.data(), {n});
    w.add_int32("sims_run", sims_run.data(), {n});
    w.add_int32("sim_steps", sim_steps.data(), {n});
    w.add_int32("reused_visits", reused.data(), {n});
    w.add_int32("memo_hits", memo.data(), {n});
    w.add_uint8("stopped_early", stopped.data(), {n});
    w.add_float("time_budget_s", tb.data(), {n});
    w.add_float("time_budget_min_s", tb_min.data(), {n});
    w.add_int16("n_worlds", n_worlds.data(), {n});
    w.add_int64("world_seeds", world_seeds.data(), {n, w_max});
    w.add_int32("world_visits", world_visits.data(), {n, w_max});
    w.add_float("world_values", world_values.data(), {n, w_max});
    w.add_int32("origin_row", origin_row.data(), {n});
    w.add_int32("follow_path", follow_path.data(), {n, 1});
    w.add_uint8("follow_worlds", follow_worlds.data(), {n, w_max});
    w.finish();
}

namespace {

std::string json_str(const std::string& s) {
    std::string out = "\"";
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out + "\"";
}

}  // namespace

// The `.rmplay` sidecar: gui_session_io.save_replay's document, which
// shard_replay.load_replay_sidecars pairs with the shard by stem. Written to
// a temp path and renamed, like the shard itself.
void ShardAccumulator::write_replay(const std::string& shard_path,
                                    const MatchReplayMeta& meta) const {
    std::string stem = shard_path.substr(0, shard_path.size() - 4);
    std::string path = stem + ".rmplay";
    std::string tmp = path + ".tmp";
    char ts[40];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S+00:00", std::gmtime(&now));

    std::string doc = "{\n";
    doc += " \"format\": \"robomage-play-replay\",\n \"version\": 1,\n";
    doc += " \"engine_seed\": " + std::to_string(meta.engine_seed) + ",\n";
    doc += " \"actions\": [";
    size_t n_actions = 0;
    if (meta.actions != nullptr) {
        n_actions = meta.actions->size();
        for (size_t i = 0; i < n_actions; i++) {
            if (i) doc += ", ";
            doc += std::to_string((*meta.actions)[i]);
        }
    }
    doc += "],\n";
    doc += " \"deck_a\": " + json_str(meta.deck_a) + ",\n";
    doc += " \"deck_b\": " + json_str(meta.deck_b) + ",\n";
    doc += " \"human_deck\": null,\n \"opp_deck\": null,\n";
    doc += std::string(" \"human_is_a\": ") + (meta.net_is_a ? "true" : "false") + ",\n";
    doc += std::string(" \"bo3\": ") + (meta.bo3 ? "true" : "false") + ",\n";
    doc += " \"opponent_spec\": null,\n \"analysis\": null,\n \"binary\": null,\n";
    doc += " \"engine_build\": {\"obs_size\": " + std::to_string(obs_w_) +
           ", \"max_actions\": " + std::to_string(act_w_) +
           ", \"state_size\": " + std::to_string(meta.state_size) + "},\n";
    doc += " \"history_len\": " + std::to_string(n_actions) + ",\n";
    doc += " \"final_obs_sha256\": null,\n \"in_progress\": false,\n";
    doc += std::string(" \"saved_at\": ") + json_str(ts) + ",\n";
    doc += " \"row_decision_idx\": [";
    for (size_t i = 0; i < buffered_; i++) {
        if (i) doc += ", ";
        doc += std::to_string(diags_[i].decision_idx);
    }
    doc += "],\n";
    doc += " \"search_provenance\": " +
           (meta.provenance_json.empty() ? std::string("null") : meta.provenance_json) +
           "\n}\n";

    std::FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (fp == nullptr) fatal_error("npz_writer: cannot open " + tmp);
    if (std::fwrite(doc.data(), 1, doc.size(), fp) != doc.size() ||
        std::fclose(fp) != 0)
        fatal_error("npz_writer: write failed: " + tmp);
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        fatal_error("npz_writer: rename failed: " + path);
}
