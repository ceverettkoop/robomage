#include "npz_writer.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
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
                                  const uint8_t* mask) {
    obs_.insert(obs_.end(), obs, obs + obs_w_);
    pi_.insert(pi_.end(), pi, pi + act_w_);
    z_.push_back(z);
    mask_.insert(mask_.end(), mask, mask + act_w_);
    buffered_++;
}

void ShardAccumulator::maybe_flush() {
    if (buffered_ >= flush_threshold_) flush();
}

void ShardAccumulator::flush_final() {
    if (buffered_ > 0) flush();
}

void ShardAccumulator::flush() {
    if (buffered_ == 0) return;
    char ts[32];
    std::time_t now = std::time(nullptr);
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&now));
    std::string path = out_dir_ + "/shard_" + ts + "_" +
                       std::to_string(static_cast<long>(getpid())) + "_" +
                       std::to_string(shard_n_) + ".npz";

    {
        NpzWriter w(path);
        w.add_float("obs", obs_.data(), {buffered_, obs_w_});
        w.add_float("pi", pi_.data(), {buffered_, act_w_});
        w.add_float("z", z_.data(), {buffered_});
        w.add_bool("mask", mask_.data(), {buffered_, act_w_});
        w.finish();
    }

    shards_.push_back(path);
    total_ += buffered_;
    shard_n_++;
    obs_.clear();
    pi_.clear();
    z_.clear();
    mask_.clear();
    buffered_ = 0;
}
