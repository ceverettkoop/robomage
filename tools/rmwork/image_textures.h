#ifndef RMWORK_IMAGE_TEXTURES_H
#define RMWORK_IMAGE_TEXTURES_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "imgui.h"

// ── Image files as GPU textures: decoded off the UI thread, kept in a bounded LRU ───────────────
//
// get() asks for a file at a drawn width and returns its texture once it is there: a worker thread
// decodes the file with stb_image and halves it down to the level asked for (level L is the image
// at 1/2^L of its size: the smallest level at least as wide as the card is drawn, so a small card
// is not drawn from a large texture), and new_frame(), on the UI thread, hands at most
// UPLOADS_PER_FRAME decoded images a frame to ImGui as user textures (ImTextureData, which the
// renderer backend uploads, as table_backdrop.h's tiles are). Nothing waits: a texture not there
// yet is nullptr, and the caller draws something else this frame. A file that does not decode is
// remembered as failed until clear().
//
// The textures are bounded by count and bytes (TextureLru): past either bound new_frame() evicts
// the least recently drawn, never one drawn this frame or the last; an evicted texture the backend
// holds is destroyed through it (WantDestroy) and released once it is. Destroy the object while
// the ImGui context exists.

namespace rmwork {

// The LRU's bookkeeping, apart from any texture (the self-test checks it alone).
class TextureLru {
    public:
        // Records `key` (of `bytes`) as drawn at frame `frame`.
        void touch(const std::string &key, size_t bytes, uint64_t frame);
        void erase(const std::string &key);
        bool contains(const std::string &key) const { return entries_.count(key) > 0; }
        size_t count() const { return entries_.size(); }
        size_t bytes() const { return bytes_; }
        // The keys to evict, least recently drawn first, until at most `max_count` entries of at most
        // `max_bytes` remain; an entry drawn at `frame` or the frame before is never one.
        std::vector<std::string> over_budget(uint64_t frame, size_t max_count, size_t max_bytes) const;

    private:
        struct Entry {
                size_t bytes = 0;
                uint64_t frame = 0;
        };
        std::map<std::string, Entry> entries_;
        size_t bytes_ = 0;
};

// The level of an image `image_w` pixels wide that a card `drawn_w` pixels wide is drawn from: the
// largest L ≤ MAX_LEVEL with image_w / 2^L ≥ drawn_w (0 when the image is narrower).
int texture_level(int image_w, float drawn_w);
// `in` (RGBA, w × h) halved: each pixel the mean of a 2 × 2 block (an odd last row or column kept).
std::vector<uint32_t> halve_rgba(const std::vector<uint32_t> &in, int w, int h, int *out_w, int *out_h);

class ImageTextures {
    public:
        inline static constexpr int MAX_LEVEL = 3;
        inline static constexpr int UPLOADS_PER_FRAME = 12;
        inline static constexpr size_t DEFAULT_MAX_COUNT = 400;
        inline static constexpr size_t DEFAULT_MAX_BYTES = size_t(256) << 20;

        ImageTextures();
        ImageTextures(const ImageTextures &) = delete;
        ImageTextures &operator=(const ImageTextures &) = delete;
        ~ImageTextures();

        // Once per UI frame: uploads decoded images, evicts past the budget, releases what the
        // backend has destroyed.
        void new_frame();
        // `path`'s texture for drawing `drawn_w` pixels wide, nullptr until it is decoded and handed
        // over (it is asked for then) or when the file does not decode. `*w`, `*h`: its size.
        ImTextureData *get(const std::string &path, float drawn_w, int *w = nullptr, int *h = nullptr);
        bool failed(const std::string &path) const;
        // Requests are queued, being decoded or waiting to be handed over.
        bool busy() const;
        // Drops every texture and forgets failures and requests.
        void clear();
        void set_budget(size_t max_count, size_t max_bytes);
        const TextureLru &lru() const { return lru_; }
        // Textures handed to ImGui (live, not retired), decodes finished, evictions so far.
        size_t live() const { return textures_.size(); }
        uint64_t decoded() const { return decoded_; }
        uint64_t evicted() const { return evicted_; }

    private:
        struct Job {
                std::string key;
                std::string path;
                int level = 0;
                uint64_t epoch = 0;
        };
        struct Result {
                std::string key;
                std::string path;
                int w = 0, h = 0;
                int full_w = 0;  // level 0's width
                std::vector<uint32_t> pixels;
                bool ok = false;
                uint64_t epoch = 0;
        };
        struct Texture {
                std::unique_ptr<ImTextureData> data;
                int w = 0, h = 0;
        };
        void work();
        static Result decode(const Job &job);
        void evict(const std::string &key);
        void release(std::unique_ptr<ImTextureData> tex);
        static std::string key_of(const std::string &path, int level);

        // UI thread only.
        std::map<std::string, Texture> textures_;
        std::map<std::string, int> widths_;  // per path, its full width once one level decoded
        std::map<std::string, bool> failed_;
        std::map<std::string, bool> requested_;
        std::vector<std::unique_ptr<ImTextureData>> retired_;
        TextureLru lru_;
        uint64_t frame_ = 0;
        size_t max_count_ = DEFAULT_MAX_COUNT;
        size_t max_bytes_ = DEFAULT_MAX_BYTES;
        uint64_t decoded_ = 0;
        uint64_t evicted_ = 0;

        // Shared with the worker, under mutex_.
        mutable std::mutex mutex_;
        std::condition_variable wake_;
        std::deque<Job> jobs_;
        std::vector<Result> results_;
        bool working_ = false;  // the worker holds a job
        bool stop_ = false;
        uint64_t epoch_ = 0;  // clear() moves it: older results are dropped
        std::thread worker_;
};

}  // namespace rmwork

#endif /* RMWORK_IMAGE_TEXTURES_H */
