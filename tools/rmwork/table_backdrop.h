#ifndef RMWORK_TABLE_BACKDROP_H
#define RMWORK_TABLE_BACKDROP_H

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "imgui.h"
#include "settings.h"

// ── The Table's background: a colour, a texture or an image (Settings → Table) ─────────────────
//
// The five textures are tiles generated in memory the first time each is shown (wood with planks
// and grain, felt, linen, tiles, slate) and handed to ImGui as user textures, which the renderer
// uploads; they are drawn repeated over the panel. An image is read with stb_image (scaled down to
// at most MAX_IMAGE_SIDE a side) and drawn covering the panel, centred; a file that is missing or
// cannot be read shows the wood instead and says so once (take_message). Textures live until the
// backdrop is destroyed, which must happen while the ImGui context exists (an image replaced by
// another is retired: released once the renderer has destroyed its texture).

namespace rmwork {

class TableBackdrop {
    public:
        inline static constexpr int MAX_IMAGE_SIDE = 4096;

        TableBackdrop() = default;
        TableBackdrop(const TableBackdrop &) = delete;
        TableBackdrop &operator=(const TableBackdrop &) = delete;
        ~TableBackdrop();

        // Fills [min, max] with `bg`.
        void draw(ImDrawList *dl, ImVec2 min, ImVec2 max, const TableBackground &bg);
        // A message for the status bar ("" when none), once per failed image.
        std::string take_message();
        // The image setting's file could not be read (the wood is shown).
        bool image_failed() const { return image_failed_; }

        // Texture `key`'s tile (RGBA, `*w` × `*h`); an unknown key gives the wood's. Slate's tile is a
        // translucent grain laid over its gradient.
        static std::vector<uint32_t> make_tile(const std::string &key, int *w, int *h);

    private:
        struct Texture {
                std::unique_ptr<ImTextureData> data;
                int w = 0, h = 0;
        };
        Texture *texture(const std::string &key);
        bool load_image(const std::string &path);
        void draw_tiled(ImDrawList *dl, ImVec2 min, ImVec2 max, const std::string &key);
        static std::unique_ptr<ImTextureData> make_texture(const std::vector<uint32_t> &pixels, int w, int h);
        // The tiles' pixels.
        static uint32_t hash2(int x, int y, uint32_t seed);
        static float noise(int x, int y, uint32_t seed);  // in [-1, 1]
        static uint32_t rgba(float r, float g, float b, float a = 255.0f);
        static void wood(std::vector<uint32_t> *px, int w, int h);
        static void felt(std::vector<uint32_t> *px, int w, int h);
        static void linen(std::vector<uint32_t> *px, int w, int h);
        static void tiles(std::vector<uint32_t> *px, int w, int h);
        static void slate(std::vector<uint32_t> *px, int w, int h);
        static std::vector<uint32_t> scale_down(const unsigned char *in, int w, int h, int *out_w, int *out_h);
        void release(std::unique_ptr<ImTextureData> tex);

        std::map<std::string, Texture> tiles_;
        std::string image_path_;  // the image last asked for
        Texture image_;
        bool image_failed_ = false;
        std::string message_;
        std::vector<std::unique_ptr<ImTextureData>> retired_;
};

}  // namespace rmwork

#endif /* RMWORK_TABLE_BACKDROP_H */
