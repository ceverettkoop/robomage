#ifndef RMWORK_SELFTEST_IMAGES_H
#define RMWORK_SELFTEST_IMAGES_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── The card images' checks: --selftest (a synthetic cache) and --selftest-data (the Table) ─────
//
// Never the network and never the user's cache: every cache is written under a temporary directory
// with tiny PNGs (stb_image_write).

namespace rmwork {

// The manifest's parsing and refusals, the printing shown (default, an override, an override the
// manifest does not list), the overrides file round trip, which image a face shows (a card, both
// faces of a double-faced card, a token by index and by name, an image not cached, a back), the
// cache's status, the LRU's bookkeeping and levels, then (in an ImGui context the caller has made
// headless) the decode thread and the texture hand-over, a file that does not decode, eviction past
// the budget, and the renderer drawing an image or falling back to text. `tmp` is a scratch
// directory.
void check_image_units(const CheckFn &check, const std::string &tmp);

// The loaded match's Table with images: a synthetic cache under the app's image cache directory
// with images for half the cards of the match's CARDS frame (both faces of a double-faced card,
// the tokens), then decisions drawn in every view: every card drawn resolves to a cached image or
// to the text renderer as the cache says, every image decodes; the cache is removed afterwards.
// `frames(n)` runs n UI frames.
void check_table_images(const CheckFn &check, App &app, const std::function<void(int)> &frames, const std::string &at);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_IMAGES_H */
