#include "image_textures.h"

#include <algorithm>

#include "imgui_internal.h"
#include "stb_image.h"

namespace {

// The width assumed of a file not decoded yet: a Scryfall `normal` image.
const int ASSUMED_WIDTH = 488;

}  // namespace

namespace rmwork {

void TextureLru::touch(const std::string &key, size_t bytes, uint64_t frame) {
    auto it = entries_.find(key);
    if (it == entries_.end()) {
        entries_[key] = Entry{bytes, frame};
        bytes_ += bytes;
        return;
    }
    bytes_ = bytes_ - it->second.bytes + bytes;
    it->second.bytes = bytes;
    it->second.frame = std::max(it->second.frame, frame);
}

void TextureLru::erase(const std::string &key) {
    auto it = entries_.find(key);
    if (it == entries_.end()) return;
    bytes_ -= it->second.bytes;
    entries_.erase(it);
}

std::vector<std::string> TextureLru::over_budget(uint64_t frame, size_t max_count, size_t max_bytes) const {
    std::vector<std::pair<uint64_t, std::string>> order;
    for (const auto &e : entries_) order.emplace_back(e.second.frame, e.first);
    std::sort(order.begin(), order.end());
    std::vector<std::string> out;
    size_t count = entries_.size(), bytes = bytes_;
    for (const auto &o : order) {
        if (count <= max_count && bytes <= max_bytes) break;
        if (o.first + 1 >= frame) break;
        out.push_back(o.second);
        --count;
        bytes -= entries_.at(o.second).bytes;
    }
    return out;
}

int texture_level(int image_w, float drawn_w) {
    int level = 0;
    while (level < ImageTextures::MAX_LEVEL && static_cast<float>(image_w >> (level + 1)) >= drawn_w) ++level;
    return level;
}

std::vector<uint32_t> halve_rgba(const std::vector<uint32_t> &in, int w, int h, int *out_w, int *out_h) {
    const int ow = std::max(1, w / 2), oh = std::max(1, h / 2);
    *out_w = ow;
    *out_h = oh;
    std::vector<uint32_t> out(static_cast<size_t>(ow) * static_cast<size_t>(oh));
    for (int y = 0; y < oh; ++y)
        for (int x = 0; x < ow; ++x) {
            const int x0 = std::min(w - 1, 2 * x), x1 = std::min(w - 1, 2 * x + 1);
            const int y0 = std::min(h - 1, 2 * y), y1 = std::min(h - 1, 2 * y + 1);
            const uint32_t px[4] = {in[static_cast<size_t>(y0 * w + x0)], in[static_cast<size_t>(y0 * w + x1)],
                in[static_cast<size_t>(y1 * w + x0)], in[static_cast<size_t>(y1 * w + x1)]};
            uint32_t v = 0;
            for (int c = 0; c < 4; ++c) {
                uint32_t sum = 0;
                for (uint32_t p : px) sum += (p >> (8 * c)) & 0xFF;
                v |= ((sum + 2) / 4) << (8 * c);
            }
            out[static_cast<size_t>(y * ow + x)] = v;
        }
    return out;
}

ImageTextures::ImageTextures() = default;

ImageTextures::~ImageTextures() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (worker_.joinable()) worker_.join();
    const bool live = ImGui::GetCurrentContext() != nullptr;
    for (auto &t : textures_)
        if (t.second.data && live) ImGui::UnregisterUserTexture(t.second.data.get());
    for (auto &t : retired_)
        if (live) ImGui::UnregisterUserTexture(t.get());
}

std::string ImageTextures::key_of(const std::string &path, int level) {
    return path + "#" + std::to_string(level);
}

ImageTextures::Result ImageTextures::decode(const Job &job) {
    Result r;
    r.key = job.key;
    r.path = job.path;
    r.epoch = job.epoch;
    int w = 0, h = 0, n = 0;
    unsigned char *data = stbi_load(job.path.c_str(), &w, &h, &n, 4);
    if (!data) return r;
    r.full_w = w;
    std::vector<uint32_t> px(static_cast<size_t>(w) * static_cast<size_t>(h));
    std::copy(data, data + px.size() * 4, reinterpret_cast<unsigned char *>(px.data()));
    stbi_image_free(data);
    for (int l = 0; l < job.level && w > 1 && h > 1; ++l) {
        int nw = 0, nh = 0;
        px = halve_rgba(px, w, h, &nw, &nh);
        w = nw;
        h = nh;
    }
    r.w = w;
    r.h = h;
    r.pixels = std::move(px);
    r.ok = true;
    return r;
}

void ImageTextures::work() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
            if (stop_) return;
            // The newest request first: what is on screen now.
            job = jobs_.back();
            jobs_.pop_back();
            working_ = true;
        }
        Result r = decode(job);
        std::lock_guard<std::mutex> lock(mutex_);
        working_ = false;
        if (r.epoch == epoch_) results_.push_back(std::move(r));
    }
}

void ImageTextures::release(std::unique_ptr<ImTextureData> tex) {
    if (!tex) return;
    if (tex->Status == ImTextureStatus_OK) {
        tex->SetStatus(ImTextureStatus_WantDestroy);
        retired_.push_back(std::move(tex));
    } else {
        ImGui::UnregisterUserTexture(tex.get());
    }
}

void ImageTextures::evict(const std::string &key) {
    auto it = textures_.find(key);
    if (it == textures_.end()) return;
    release(std::move(it->second.data));
    textures_.erase(it);
    lru_.erase(key);
    requested_.erase(key);
    ++evicted_;
}

void ImageTextures::new_frame() {
    ++frame_;
    for (auto it = retired_.begin(); it != retired_.end();) {
        if ((*it)->Status == ImTextureStatus_Destroyed) {
            ImGui::UnregisterUserTexture(it->get());
            it = retired_.erase(it);
        } else {
            ++it;
        }
    }
    // The backend keeps its own copy once a texture is uploaded.
    for (auto &t : textures_)
        if (t.second.data->Status == ImTextureStatus_OK && t.second.data->Pixels) t.second.data->DestroyPixels();
    std::vector<Result> ready;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t n = std::min(results_.size(), static_cast<size_t>(UPLOADS_PER_FRAME));
        ready.assign(std::make_move_iterator(results_.begin()), std::make_move_iterator(results_.begin() + n));
        results_.erase(results_.begin(), results_.begin() + static_cast<std::ptrdiff_t>(n));
    }
    for (Result &r : ready) {
        ++decoded_;
        if (!r.ok) {
            failed_[r.path] = true;
            requested_.erase(r.key);
            continue;
        }
        widths_[r.path] = r.full_w;
        Texture t;
        t.w = r.w;
        t.h = r.h;
        t.data = std::make_unique<ImTextureData>();
        t.data->Create(ImTextureFormat_RGBA32, r.w, r.h);
        std::copy(r.pixels.begin(), r.pixels.end(), reinterpret_cast<uint32_t *>(t.data->GetPixels()));
        t.data->UseColors = true;
        ImGui::RegisterUserTexture(t.data.get());
        lru_.touch(r.key, static_cast<size_t>(r.w) * static_cast<size_t>(r.h) * 4, frame_);
        textures_[r.key] = std::move(t);
    }
    for (const std::string &key : lru_.over_budget(frame_, max_count_, max_bytes_)) evict(key);
}

ImTextureData *ImageTextures::get(const std::string &path, float drawn_w, int *w, int *h) {
    if (failed_.count(path)) return nullptr;
    auto width = widths_.find(path);
    const int level = texture_level(width == widths_.end() ? ASSUMED_WIDTH : width->second, drawn_w);
    const std::string key = key_of(path, level);
    auto it = textures_.find(key);
    if (it == textures_.end() && !requested_.count(key)) {
        requested_[key] = true;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back(Job{key, path, level, epoch_});
        }
        if (!worker_.joinable()) worker_ = std::thread(&ImageTextures::work, this);
        wake_.notify_one();
    }
    // Until the level asked for is there, another level of the file stands in.
    for (int d = 0; it == textures_.end() && d <= MAX_LEVEL; ++d)
        for (int l : {level - d, level + d})
            if (l >= 0 && l <= MAX_LEVEL && it == textures_.end()) it = textures_.find(key_of(path, l));
    if (it == textures_.end()) return nullptr;
    lru_.touch(it->first, static_cast<size_t>(it->second.w) * static_cast<size_t>(it->second.h) * 4, frame_);
    if (w) *w = it->second.w;
    if (h) *h = it->second.h;
    return it->second.data.get();
}

bool ImageTextures::failed(const std::string &path) const {
    return failed_.count(path) > 0;
}

bool ImageTextures::busy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return !jobs_.empty() || working_ || !results_.empty();
}

void ImageTextures::clear() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        jobs_.clear();
        results_.clear();
        ++epoch_;
    }
    std::vector<std::string> keys;
    for (const auto &t : textures_) keys.push_back(t.first);
    for (const std::string &k : keys) evict(k);
    widths_.clear();
    failed_.clear();
    requested_.clear();
}

void ImageTextures::set_budget(size_t max_count, size_t max_bytes) {
    max_count_ = max_count;
    max_bytes_ = max_bytes;
}

}  // namespace rmwork
