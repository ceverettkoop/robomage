#ifndef RMWORK_PRINTINGS_TAB_H
#define RMWORK_PRINTINGS_TAB_H

#include <memory>
#include <string>
#include <vector>

#include "card_images.h"
#include "settings.h"
#include "subprocess.h"

// ── Settings → Card Printings: card faces, the image cache and each card's printing ─────────────
//
// The top: "Card faces: images / text" (Settings::card_images) and the cache's status (vocab cards
// with images, tokens, what Scryfall lacked, the command that fetches the rest; a Reload). Below,
// a card list searched by name (cards and tokens in the manifest) and the chosen card's paper
// printings, oldest first: a thumbnail when its image is cached, the set's name and code, the
// collector number and year, which is the default (the oldest non-promo paper printing) and which
// is shown. Choosing a row overrides the default (printings.json); "Default (oldest)" resets the
// card, Reset All every card. A shown printing whose images are not cached says "not downloaded"
// with the exact fetch_images.py command and a Fetch button that runs it (the script does the
// network access, the workbench never does) and reloads the manifest when it exits.

namespace rmwork {

class PrintingsTab {
    public:
        // Draws the tab's body for `s` over `images` (nullptr: the images are not wired: a note).
        void draw(Settings *s, CardImages *images);
        // Chooses the first card whose name contains `name` (case-insensitive), as a click would.
        void select(const std::string &name) { want_select_ = name; }
        const std::string &selected() const { return selected_; }
        // The fetch subprocess runs.
        bool fetching() const { return fetch_ && fetching_; }
        // What the last fetch said ("" before one).
        const std::string &fetch_message() const { return fetch_message_; }

    private:
        void draw_status(Settings *s, CardImages &images);
        void draw_list(CardImages &images);
        void draw_card(CardImages &images, const ImageCard &card);
        void draw_printing_row(CardImages &images, const ImageCard &card, size_t i, bool shown);
        void start_fetch(CardImages &images, const ImageCard &card, const ImagePrinting &p);
        void poll_fetch(CardImages &images);
        static bool matches(const ImageCard &card, const std::string &query);

        char query_[128] = {0};
        std::string selected_;  // the chosen card's key
        std::string want_select_;
        bool scroll_shown_ = false;  // a card was just chosen: its shown printing's row is scrolled to
        std::string message_;
        bool message_error_ = false;
        std::unique_ptr<ChildProcess> fetch_;
        bool fetching_ = false;
        std::string fetch_key_;
        std::string fetch_err_path_;
        std::string fetch_message_;
};

}  // namespace rmwork

#endif /* RMWORK_PRINTINGS_TAB_H */
