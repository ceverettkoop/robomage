#ifndef RMWORK_IMAGE_RENDERER_H
#define RMWORK_IMAGE_RENDERER_H

#include "card_images.h"
#include "card_renderer.h"

// ── The image renderer: a card drawn as its printing's Scryfall image ───────────────────────────
//
// A face the cache has an image of (CardImages::file_for) is drawn as that image, its corners
// rounded like the text renderer's card, turned a quarter for a tapped card and faded like it.
// Everything else goes to the text renderer: a card's back (a face-down or hidden card, with the
// note of what the viewer knows), a card or token the cache has no image of, and an image still
// being decoded (it is asked for, and drawn from the frame it is there).

namespace rmwork {

class ImageCardRenderer : public CardRenderer {
    public:
        explicit ImageCardRenderer(CardImages *images) : images_(images) {}
        void draw(ImDrawList *dl, const CardLook &look, ImVec2 slot_min, ImVec2 slot_max, bool rotated,
            float alpha) const override;
        bool other_face_differs(const CardLook &look) const override;
        // Whether `look` would be drawn as an image now (else by the text renderer).
        bool draws_image(const CardLook &look, float width) const;

    private:
        ImTextureData *texture(const CardLook &look, float width) const;
        CardImages *images_;
        TextCardRenderer text_;
};

}  // namespace rmwork

#endif /* RMWORK_IMAGE_RENDERER_H */
