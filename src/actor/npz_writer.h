#ifndef ACTOR_NPZ_WRITER_H
#define ACTOR_NPZ_WRITER_H

// Minimal, dependency-free writer for the numpy .npz (uncompressed zip of .npy
// members) that train/az_train.py::load_window ingests. Two layers:
//
//   * NpzWriter — low-level: streams an uncompressed (STORE, method 0) zip whose
//     members are .npy arrays (magic \x93NUMPY, version 1.0, 64-byte-aligned
//     header). np.load reads it directly. Written to a temp path and rename()d on
//     finish() so a glob for shard_*.npz never sees a partial file.
//
//   * ShardAccumulator — high-level: buffers z-backfilled self-play samples
//     across games and flushes shard_{YYYYmmdd_HHMMSS}_{pid}_{n}.npz files under
//     an output dir, mirroring train/az_selfplay.py's per-game flush granularity
//     (a shard is flushed at a game boundary once the buffer reaches the sample
//     threshold; a game is never split across shards). Keys: obs (f32 n×obs_w),
//     pi (f32 n×act_w), z (f32 n), mask (bool n×act_w).

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// ── low-level uncompressed-npz writer ──────────────────────────────────────
class NpzWriter {
public:
    // Opens `path` + ".tmp" for writing; finish() renames it to `path`.
    explicit NpzWriter(const std::string& path);
    ~NpzWriter();

    NpzWriter(const NpzWriter&) = delete;
    NpzWriter& operator=(const NpzWriter&) = delete;

    // Add a float32 ('<f4') array member named `name` (".npy" appended) with the
    // given row-major shape and `data` pointing at prod(shape) floats.
    void add_float(const std::string& name, const float* data,
                   const std::vector<size_t>& shape);
    // Add a bool ('|b1') array member; `data` is prod(shape) bytes (0/1).
    void add_bool(const std::string& name, const uint8_t* data,
                  const std::vector<size_t>& shape);

    // Write the central directory + EOCD, close, and rename the temp file into
    // place. Safe to call once; the destructor calls it if not already done.
    void finish();

private:
    struct Member {
        std::string filename;  // e.g. "obs.npy"
        uint32_t crc;
        uint32_t size;         // uncompressed == compressed (STORE)
        uint32_t offset;       // local-header offset
    };

    void add_member(const std::string& name, const char* descr,
                    size_t elem_size, const void* data,
                    const std::vector<size_t>& shape);

    std::string final_path_;
    std::string tmp_path_;
    std::FILE* fp_;
    uint32_t offset_;  // running byte offset into the file
    std::vector<Member> members_;
    bool finished_;
};

// ── high-level self-play shard accumulator ─────────────────────────────────
class ShardAccumulator {
public:
    ShardAccumulator(std::string out_dir, size_t flush_samples, size_t obs_width,
                     size_t act_width);

    // Append one z-backfilled sample row.
    void add_sample(const float* obs, const float* pi, float z,
                    const uint8_t* mask);

    // Flush a shard if the buffer has reached the sample threshold. Call at a
    // GAME boundary (after adding a whole game's samples) so a game is never
    // split across shards — matching az_selfplay.py.
    void maybe_flush();

    // Flush any remaining buffered samples (call once at the end).
    void flush_final();

    size_t total_samples() const { return total_; }
    size_t shards_written() const { return shards_.size(); }
    const std::vector<std::string>& shard_paths() const { return shards_; }

private:
    void flush();

    std::string out_dir_;
    size_t flush_threshold_;
    size_t obs_w_;
    size_t act_w_;

    std::vector<float> obs_;
    std::vector<float> pi_;
    std::vector<float> z_;
    std::vector<uint8_t> mask_;
    size_t buffered_;  // rows currently buffered
    size_t total_;     // rows written across all shards
    int shard_n_;
    std::vector<std::string> shards_;
};

#endif /* ACTOR_NPZ_WRITER_H */
