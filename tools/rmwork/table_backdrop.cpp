#include "table_backdrop.h"

#include <algorithm>
#include <cmath>

#include "imgui_internal.h"
#include "widgets.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "stb_image.h"

uint32_t rmwork::TableBackdrop::hash2(int x, int y, uint32_t seed) {
    uint32_t v = static_cast<uint32_t>(x) * 374761393u + static_cast<uint32_t>(y) * 668265263u + seed * 2246822519u;
    v = (v ^ (v >> 13)) * 1274126177u;
    return v ^ (v >> 16);
}

// A value in [-1, 1].
float rmwork::TableBackdrop::noise(int x, int y, uint32_t seed) {
    return static_cast<float>(hash2(x, y, seed) & 0xFFFF) / 32767.5f - 1.0f;
}

uint32_t rmwork::TableBackdrop::rgba(float r, float g, float b, float a) {
    const auto c = [](float v) { return static_cast<uint32_t>(std::max(0.0f, std::min(255.0f, v))); };
    return IM_COL32(c(r), c(g), c(b), c(a));
}

// Two planks of 160 px with a dark seam, each a little different in tone, with wavy grain lines
// (periodic in the tile's height) and fine noise.
void rmwork::TableBackdrop::wood(std::vector<uint32_t> *px, int w, int h) {
    const float tone[2] = {1.0f, 0.9f};
    const float phase[2] = {0.0f, 2.1f};
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int plank = (x / 160) % 2;
            const int u = x % 160;
            const float t = 2.0f * IM_PI * static_cast<float>(y) / static_cast<float>(h);
            const float wave = 3.5f * std::sin(t * 2 + phase[plank]) + 1.5f * std::sin(t * 5 + phase[plank] * 1.7f);
            const float grain = 0.5f + 0.5f * std::sin((static_cast<float>(u) + wave) * 2.0f * IM_PI / 9.0f);
            const float streak = 0.5f + 0.5f * noise(u / 3, plank, 7);
            float k = tone[plank] * (0.86f + 0.10f * grain + 0.06f * streak) + 0.025f * noise(x, y, 3);
            if (u >= 158) k *= 0.5f;
            if (u == 0) k *= 1.12f;
            px->at(static_cast<size_t>(y * w + x)) = rgba(110 * k, 69 * k, 38 * k);
        }
    }
}

void rmwork::TableBackdrop::felt(std::vector<uint32_t> *px, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float n = 9.0f * noise(x, y, 11) + 4.0f * noise(x / 2, y / 2, 12);
            px->at(static_cast<size_t>(y * w + x)) = rgba(11 + n * 0.4f, 107 + n, 42 + n * 0.6f);
        }
}

// A weave: lighter threads every third row, darker every third column.
void rmwork::TableBackdrop::linen(std::vector<uint32_t> *px, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            float r = 185, g = 173, b = 146;
            if (y % 3 == 0) {
                r += (255 - r) * 0.12f;
                g += (255 - g) * 0.12f;
                b += (255 - b) * 0.12f;
            }
            if (x % 3 == 0) {
                r *= 0.91f;
                g *= 0.91f;
                b *= 0.91f;
            }
            const float n = 4.0f * noise(x, y, 21);
            px->at(static_cast<size_t>(y * w + x)) = rgba(r + n, g + n, b + n);
        }
}

void rmwork::TableBackdrop::tiles(std::vector<uint32_t> *px, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const bool odd = ((x / 16) + (y / 16)) % 2 != 0;
            const bool grout = x % 16 == 0 || y % 16 == 0;
            const float k = grout ? 0.85f : 1.0f;
            px->at(static_cast<size_t>(y * w + x)) = odd ? rgba(0, 123 * k, 123 * k) : rgba(0, 107 * k, 107 * k);
        }
}

// Translucent grain: dark and light specks, laid over the slate's gradient.
void rmwork::TableBackdrop::slate(std::vector<uint32_t> *px, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const float n = noise(x, y, 31) + 0.5f * noise(x / 3, y / 2, 32);
            px->at(static_cast<size_t>(y * w + x)) = n > 0 ? rgba(255, 255, 255, 18 * n) : rgba(0, 0, 0, -30 * n);
        }
}

// An image no larger than MAX_IMAGE_SIDE a side (box-filtered when it was larger).
std::vector<uint32_t> rmwork::TableBackdrop::scale_down(const unsigned char *in, int w, int h, int *out_w, int *out_h) {
    const int side = MAX_IMAGE_SIDE;
    const int f = std::max(1, (std::max(w, h) + side - 1) / side);
    *out_w = std::max(1, w / f);
    *out_h = std::max(1, h / f);
    std::vector<uint32_t> out(static_cast<size_t>(*out_w) * static_cast<size_t>(*out_h));
    for (int y = 0; y < *out_h; ++y)
        for (int x = 0; x < *out_w; ++x) {
            float acc[4] = {0, 0, 0, 0};
            for (int dy = 0; dy < f; ++dy)
                for (int dx = 0; dx < f; ++dx) {
                    const unsigned char *p = in + (static_cast<size_t>(y * f + dy) * static_cast<size_t>(w) +
                                                      static_cast<size_t>(x * f + dx)) *
                                                      4;
                    for (int c = 0; c < 4; ++c) acc[c] += p[c];
                }
            const float n = static_cast<float>(f * f);
            out[static_cast<size_t>(y * *out_w + x)] = rgba(acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n);
        }
    return out;
}

namespace rmwork {

TableBackdrop::~TableBackdrop() {
    const bool live = ImGui::GetCurrentContext() != nullptr;
    const auto drop = [live](std::unique_ptr<ImTextureData> &t) {
        if (t && live) ImGui::UnregisterUserTexture(t.get());
        t.reset();
    };
    for (auto &t : tiles_) drop(t.second.data);
    drop(image_.data);
    for (auto &t : retired_) drop(t);
}

std::vector<uint32_t> TableBackdrop::make_tile(const std::string &key, int *w, int *h) {
    std::vector<uint32_t> px;
    if (key == "felt") {
        *w = *h = 128;
    } else if (key == "linen") {
        *w = *h = 96;
    } else if (key == "tiles") {
        *w = *h = 32;
    } else if (key == "slate") {
        *w = *h = 192;
    } else {
        *w = *h = 320;
    }
    px.assign(static_cast<size_t>(*w) * static_cast<size_t>(*h), 0);
    if (key == "felt") {
        felt(&px, *w, *h);
    } else if (key == "linen") {
        linen(&px, *w, *h);
    } else if (key == "tiles") {
        tiles(&px, *w, *h);
    } else if (key == "slate") {
        slate(&px, *w, *h);
    } else {
        wood(&px, *w, *h);
    }
    return px;
}

std::unique_ptr<ImTextureData> TableBackdrop::make_texture(const std::vector<uint32_t> &pixels, int w, int h) {
    auto tex = std::make_unique<ImTextureData>();
    tex->Create(ImTextureFormat_RGBA32, w, h);
    std::copy(pixels.begin(), pixels.end(), reinterpret_cast<uint32_t *>(tex->GetPixels()));
    tex->UseColors = true;
    ImGui::RegisterUserTexture(tex.get());
    return tex;
}

void TableBackdrop::release(std::unique_ptr<ImTextureData> tex) {
    if (!tex) return;
    if (tex->Status == ImTextureStatus_OK) {
        tex->SetStatus(ImTextureStatus_WantDestroy);
        retired_.push_back(std::move(tex));
    } else {
        ImGui::UnregisterUserTexture(tex.get());
    }
}

TableBackdrop::Texture *TableBackdrop::texture(const std::string &key) {
    auto it = tiles_.find(key);
    if (it != tiles_.end()) return &it->second;
    Texture t;
    const std::vector<uint32_t> px = make_tile(key, &t.w, &t.h);
    t.data = make_texture(px, t.w, t.h);
    return &(tiles_[key] = std::move(t));
}

bool TableBackdrop::load_image(const std::string &path) {
    int w = 0, h = 0, n = 0;
    unsigned char *data = stbi_load(path.c_str(), &w, &h, &n, 4);
    if (!data) {
        message_ = "Table image " + path + " could not be read (" + stbi_failure_reason() + "); showing wood";
        return false;
    }
    int sw = 0, sh = 0;
    const std::vector<uint32_t> px = scale_down(data, w, h, &sw, &sh);
    stbi_image_free(data);
    release(std::move(image_.data));
    image_.w = sw;
    image_.h = sh;
    image_.data = make_texture(px, sw, sh);
    return true;
}

void TableBackdrop::draw_tiled(ImDrawList *dl, ImVec2 min, ImVec2 max, const std::string &key) {
    Texture *t = texture(key);
    if (key == "slate") {
        const ImU32 light = hex_color(0x3c4650), dark = hex_color(0x262c33);
        dl->AddRectFilledMultiColor(min, max, light, light, dark, dark);
    }
    const ImTextureRef ref = t->data->GetTexRef();
    for (float y = min.y; y < max.y; y += static_cast<float>(t->h))
        for (float x = min.x; x < max.x; x += static_cast<float>(t->w))
            dl->AddImage(ref, ImVec2(x, y), ImVec2(x + static_cast<float>(t->w), y + static_cast<float>(t->h)));
}

void TableBackdrop::draw(ImDrawList *dl, ImVec2 min, ImVec2 max, const TableBackground &bg) {
    for (auto it = retired_.begin(); it != retired_.end();) {
        if ((*it)->Status == ImTextureStatus_Destroyed) {
            ImGui::UnregisterUserTexture(it->get());
            it = retired_.erase(it);
        } else {
            ++it;
        }
    }
    dl->PushClipRect(min, max, true);
    if (bg.kind == TableBackground::Kind::COLOR) {
        ImU32 c = 0;
        if (!parse_hex_color(bg.value, &c)) c = hex_color(0x6e4526);
        dl->AddRectFilled(min, max, c);
    } else if (bg.kind == TableBackground::Kind::IMAGE) {
        if (bg.value != image_path_) {
            image_path_ = bg.value;
            image_failed_ = !load_image(bg.value);
        }
        if (image_failed_ || !image_.data) {
            draw_tiled(dl, min, max, "wood");
        } else {
            // Cover: scaled to fill, centred, the overflow cropped.
            const float pw = max.x - min.x, ph = max.y - min.y;
            const float iw = static_cast<float>(image_.w), ih = static_cast<float>(image_.h);
            const float s = std::max(pw / iw, ph / ih);
            const float uw = pw / (iw * s), uh = ph / (ih * s);
            const ImVec2 uv0((1 - uw) / 2, (1 - uh) / 2), uv1((1 + uw) / 2, (1 + uh) / 2);
            dl->AddRectFilled(min, max, hex_color(0x000000));
            dl->AddImage(image_.data->GetTexRef(), min, max, uv0, uv1);
        }
    } else {
        draw_tiled(dl, min, max, bg.value);
    }
    dl->PopClipRect();
}

std::string TableBackdrop::take_message() {
    std::string m;
    m.swap(message_);
    return m;
}

}  // namespace rmwork
