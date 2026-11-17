#ifndef RMWORK_CARD_IMAGES_H
#define RMWORK_CARD_IMAGES_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "card_book.h"
#include "image_textures.h"

// ── Card images: the Scryfall cache, its manifest and the printing overrides ─────────────────────
//
// tools/scryfall/fetch_images.py fills the cache (default ~/.cache/robomage/scryfall, Settings →
// Card Printings) with each vocab card's and token's chosen printing's images and writes its
// manifest.json: per card (by vocab index; a token by its script stem) every paper printing and
// the one chosen by default (the oldest non-promo, non-digital paper printing), each face's image
// file. The workbench reads only the cache, never the network. The overrides
// (~/.config/robomage/printings.json: card key → set and collector number) change only in
// Settings → Card Printings.
//
// Which image a card face shows: the manifest card of its card id (a vocab index, or a token's in
// the token band; a token with no registered index by its name), the override's printing when the
// manifest lists it, else the chosen one; of a double-faced printing the face of the same name,
// else the back for a back face. No manifest, no such card, or an image file not in the cache: no
// image, so the text renderer draws the card.

namespace rmwork {

struct ImageFace {
        std::string name;
        std::string url;
        std::string file;  // relative to the cache
};

struct ImagePrinting {
        std::string set, set_name, number, released, set_type;
        bool promo = false;
        bool default_ok = false;  // passes the default printing's rule
        std::vector<ImageFace> faces;
};

struct ImageCard {
        std::string key;  // the vocab name, or "token:<stem>"
        int index = -1;   // the vocab index (-1: a token with no registered index)
        bool token = false;
        std::string stem;
        std::vector<std::string> names;  // every vocab name of the index (a token: its name)
        std::string scryfall_name;
        int chosen = 0;  // the default printing, into printings
        std::vector<ImagePrinting> printings;
        // The name shown for the card ("Delver of Secrets", "Clue token").
        std::string display() const;
};

class ImageManifest {
    public:
        // False with `err` on malformed JSON or a manifest of the wrong shape.
        bool parse(const std::string &text, std::string *err);
        // A missing file: false with `err` = "".
        bool load(const std::string &path, std::string *err);
        const std::vector<ImageCard> &cards() const { return cards_; }
        // The keys Scryfall had nothing for, each with why.
        const std::vector<std::pair<std::string, std::string>> &missing() const { return missing_; }
        const ImageCard *by_index(int index) const;
        const ImageCard *by_key(const std::string &key) const;
        // A token by its name (" Token" dropped, case- and accent-insensitive).
        const ImageCard *token_by_name(const std::string &name) const;
        // The manifest card a face is drawn from (nullptr: none).
        const ImageCard *card_for(const CardFace &face) const;

    private:
        std::vector<ImageCard> cards_;
        std::vector<std::pair<std::string, std::string>> missing_;
        std::map<int, size_t> by_index_;
        std::map<std::string, size_t> by_key_;
        std::map<std::string, size_t> token_names_;
};

// One override: the printing a card is drawn with instead of the default.
struct PrintingChoice {
        std::string set;
        std::string number;
        bool operator==(const PrintingChoice &o) const { return set == o.set && number == o.number; }
};
// By card key (ImageCard::key).
using PrintingOverrides = std::map<std::string, PrintingChoice>;

std::string printings_to_json(const PrintingOverrides &o);
// False with `err` on malformed JSON or a row without a set and number.
bool printings_from_json(const std::string &text, PrintingOverrides *out, std::string *err);
// A missing file gives no overrides (true).
bool load_printings(const std::string &path, PrintingOverrides *out, std::string *err);
bool save_printings(const std::string &path, const PrintingOverrides &o, std::string *err);
// <config>/robomage/printings.json, beside settings.json.
std::string printings_path(const std::string &config_dir);

// The printing `card` is drawn with: the override's when the manifest lists it, else the chosen one
// (nullptr when it has none). `*overridden` says whether the override's was taken.
const ImagePrinting *shown_printing(
    const ImageCard &card, const PrintingOverrides &overrides, bool *overridden = nullptr);
// The index of `p`'s printing `set`:`number`, -1 when none.
int find_printing(const ImageCard &card, const std::string &set, const std::string &number);
// The face of `p` that shows `face`: of two or more, the one of the same name, else the second for
// a back face; of one, that one. -1 when `p` has no faces.
int image_face_index(const ImagePrinting &p, const CardFace &face);

// "python3 tools/scryfall/fetch_images.py --cache <dir> --printing '<name>=<set>:<number>'".
std::string fetch_printing_command(const std::string &cache, const ImageCard &card, const ImagePrinting &p);
// The same as an argv (run with /usr/bin/env), the script at `repo`.
std::vector<std::string> fetch_printing_argv(
    const std::string &repo, const std::string &cache, const ImageCard &card, const ImagePrinting &p);
// "python3 tools/scryfall/fetch_images.py" with --cache when `cache` is not the default.
std::string fetch_all_command(const std::string &cache);

// How much of the vocab the cache covers.
struct ImageCacheStatus {
        bool manifest = false;
        std::string error;     // the manifest's, when it does not read
        int vocab = 0;         // vocab indices (cards, not tokens)
        int with_images = 0;   // of them, with their shown printing's images cached
        int tokens = 0;        // tokens in the manifest
        int token_images = 0;  // of them, with images cached
        int missing = 0;       // the manifest's missing list
};

// The cache, its manifest, the overrides and the textures of its images: what the image renderer
// and Settings → Card Printings read.
class CardImages {
    public:
        // The cache directory and the overrides file; re-read when either changes.
        void configure(const std::string &cache_dir, const std::string &printings_file);
        // Re-reads the manifest and the overrides, forgets which files exist and drops the textures.
        void reload();
        // Once per UI frame, before anything draws (ImageTextures::new_frame).
        void new_frame() { textures_.new_frame(); }

        const std::string &cache_dir() const { return cache_dir_; }
        const std::string &printings_file() const { return printings_file_; }
        bool has_manifest() const { return has_manifest_; }
        const std::string &manifest_error() const { return manifest_error_; }
        const ImageManifest &manifest() const { return manifest_; }
        const PrintingOverrides &overrides() const { return overrides_; }
        const std::string &overrides_error() const { return overrides_error_; }
        // Sets or clears (`choice` empty) a card's override and writes the file; false with `err`.
        bool set_override(const std::string &key, const PrintingChoice &choice, std::string *err);
        bool clear_overrides(std::string *err);

        // The absolute path of a manifest file.
        std::string path_of(const ImageFace &f) const;
        // Whether the face's image is in the cache (remembered until reload).
        bool cached(const ImageFace &f) const;
        bool printing_cached(const ImagePrinting &p) const;
        // The cached image file `face` is drawn from ("": none, the text renderer draws it).
        std::string file_for(const CardFace &face) const;
        ImageCacheStatus status() const;

        ImageTextures &textures() { return textures_; }
        const ImageTextures &textures() const { return textures_; }

    private:
        std::string cache_dir_;
        std::string printings_file_;
        bool configured_ = false;
        bool has_manifest_ = false;
        std::string manifest_error_;
        std::string overrides_error_;
        ImageManifest manifest_;
        PrintingOverrides overrides_;
        mutable std::map<std::string, bool> exists_;
        ImageTextures textures_;
};

}  // namespace rmwork

#endif /* RMWORK_CARD_IMAGES_H */
