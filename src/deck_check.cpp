#include "deck_check.h"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>

#include "card_db.h"
#include "card_vocab.h"
#include "classes/deck.h"
#include "parse.h"

namespace {

// One distinct card the decks name (by uid).
struct DeckCard {
        std::string name;  // as the first deck line naming it spells it
        int decks = 0;
        int copies = 0;
};

}  // namespace

static std::vector<std::string> deck_files(const std::string &dir);
static int highest_card_index();
static bool optional_dir(const std::string &dir);

// The .dk files directly in `dir`, sorted.
static std::vector<std::string> deck_files(const std::string &dir) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() > 3 && name.compare(name.size() - 3, 3, ".dk") == 0) out.push_back(dir + "/" + name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

// A default directory that may hold no deck: heldout/ before any deck is held out.
static bool optional_dir(const std::string &dir) { return dir == "heldout"; }

static int highest_card_index() {
    int highest = -1;
    for (int i = 0; i < CARD_VOCAB_SIZE; i++) highest = std::max(highest, card_vocab_entries[i].index);
    return highest;
}

int check_decks(const std::vector<std::string> &dirs) {
    const std::vector<std::string> wanted = dirs.empty() ? std::vector<std::string>{"", "league", "heldout"} : dirs;
    std::map<std::string, DeckCard> cards;  // uid -> card
    size_t n_decks = 0;
    size_t empty_dirs = 0;
    for (const std::string &dir : wanted) {
        const std::string path = RESOURCE_DIR + "/decks" + (dir.empty() ? "" : "/" + dir);
        const bool one_deck = path.size() > 3 && path.compare(path.size() - 3, 3, ".dk") == 0;
        const std::vector<std::string> files = one_deck ? std::vector<std::string>{path} : deck_files(path);
        if (!one_deck && files.empty() && optional_dir(dir) && dirs.empty()) continue;
        if (files.empty()) {
            std::printf("check-decks: ERROR no .dk files in %s\n", path.c_str());
            ++empty_dirs;
        }
        for (const std::string &file : files) {
            ++n_decks;
            const Deck deck(file);
            std::set<std::string> in_deck;
            for (const auto *section : {&deck.main_deck, &deck.sideboard})
                for (const auto &entry : *section) {
                    const std::string uid = name_to_uid(entry.second);
                    DeckCard &c = cards[uid];
                    if (c.name.empty()) c.name = entry.second;
                    c.copies += static_cast<int>(entry.first);
                    if (in_deck.insert(uid).second) ++c.decks;
                }
        }
    }
    struct Missing {
            const DeckCard *card;
            bool has_script;
    };
    std::vector<Missing> missing;
    for (const auto &kv : cards) {
        const CardData *cd = card_data_by_name(kv.second.name);
        if (cd && card_name_to_index(cd->name) >= 0) continue;
        missing.push_back({&kv.second, cd != nullptr});
    }
    std::stable_sort(missing.begin(), missing.end(), [](const Missing &a, const Missing &b) {
        if (a.card->decks != b.card->decks) return a.card->decks > b.card->decks;
        if (a.card->copies != b.card->copies) return a.card->copies > b.card->copies;
        return ascii_lower(a.card->name) < ascii_lower(b.card->name);
    });
    int next_index = highest_card_index() + 1;
    for (const Missing &m : missing) {
        std::string line = "check-decks: MISSING " + m.card->name + ": " +
                           (m.has_script ? "not in src/card_vocab.h" : "no script") + " (decks " +
                           std::to_string(m.card->decks) + ", copies " + std::to_string(m.card->copies);
        if (m.has_script) {
            if (next_index < TOKEN_VOCAB_BASE) line += ", next index " + std::to_string(next_index);
            ++next_index;
        }
        std::printf("%s)\n", line.c_str());
    }
    std::printf("check-decks: %zu deck(s), %zu distinct card(s), %zu missing\n", n_decks, cards.size(), missing.size());
    return missing.empty() && empty_dirs == 0 ? 0 : 1;
}
