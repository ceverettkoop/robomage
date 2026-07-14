#include "deck_state.h"

#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>

#include "../card_vocab.h"
#include "../error.h"
#include "../parse.h"  // name_to_uid
#include "deck.h"
#include "gamestate.h"  // DECKLIST_MAIN_SLOTS / DECKLIST_SIDE_SLOTS

std::vector<DecklistEntry> g_deck_main_a;
std::vector<DecklistEntry> g_deck_side_a;
std::vector<DecklistEntry> g_deck_main_b;
std::vector<DecklistEntry> g_deck_side_b;

// Resolve a DECK-FILE card name to its vocab index. Deck files strip punctuation
// (apostrophes, commas: "Dragons Rage Channeler", "Thalia Guardian of Thraben"),
// so the canonical card name in the vocab won't match by raw string. Canonicalize
// both through name_to_uid — exactly how the engine resolves a deck name to a card
// script (card_db.cpp) — so the comma/apostrophe-free deck form maps to the same
// card. Returns -1 for a name with no vocab entry.
static int deck_name_to_vocab(const std::string &name) {
    static const std::unordered_map<std::string, int> by_uid = [] {
        std::unordered_map<std::string, int> m;
        for (int i = 0; i < CARD_VOCAB_SIZE; i++)
            m[name_to_uid(card_vocab_entries[i].name)] = card_vocab_entries[i].index;
        return m;
    }();
    auto it = by_uid.find(name_to_uid(name));
    return it != by_uid.end() ? it->second : -1;
}

// Build a sorted (vocab_idx, count) list from a deck section, merging by vocab id.
// Fatal on an unregistered card name or on a distinct-name count exceeding max_slots.
static std::vector<DecklistEntry> build_block(
    const std::vector<std::pair<size_t, std::string>> &section, int max_slots,
    const char *block_name) {
    std::map<int, int> by_id;  // ordered ascending by vocab id
    for (const auto &entry : section) {
        int idx = deck_name_to_vocab(entry.second);
        if (idx < 0)
            fatal_error("deck_state: card \"" + entry.second +
                        "\" has no vocab index (add it to src/card_vocab.h) — "
                        "cannot serialize the " + block_name + " deck-identity block");
        by_id[idx] += static_cast<int>(entry.first);
    }
    if (static_cast<int>(by_id.size()) > max_slots)
        fatal_error("deck_state: " + std::string(block_name) + " has " +
                    std::to_string(by_id.size()) + " distinct card names, exceeds " +
                    std::to_string(max_slots) + " serialized slots");
    std::vector<DecklistEntry> out;
    out.reserve(by_id.size());
    for (const auto &kv : by_id) out.push_back({kv.first, kv.second});
    return out;
}

void deck_state_set(Zone::Ownership owner, const Deck &deck) {
    std::vector<DecklistEntry> main = build_block(deck.main_deck, DECKLIST_MAIN_SLOTS, "maindeck");
    std::vector<DecklistEntry> side = build_block(deck.sideboard, DECKLIST_SIDE_SLOTS, "sideboard");
    if (owner == Zone::PLAYER_A) {
        g_deck_main_a = std::move(main);
        g_deck_side_a = std::move(side);
    } else if (owner == Zone::PLAYER_B) {
        g_deck_main_b = std::move(main);
        g_deck_side_b = std::move(side);
    }
}

void deck_state_reset() {
    g_deck_main_a.clear();
    g_deck_side_a.clear();
    g_deck_main_b.clear();
    g_deck_side_b.clear();
}

const std::vector<DecklistEntry> &deck_state_main(Zone::Ownership owner) {
    return (owner == Zone::PLAYER_A) ? g_deck_main_a : g_deck_main_b;
}

const std::vector<DecklistEntry> &deck_state_side(Zone::Ownership owner) {
    return (owner == Zone::PLAYER_A) ? g_deck_side_a : g_deck_side_b;
}

void deck_state_capture(DeckStateSnapshot &out) {
    out.main_a = g_deck_main_a;
    out.side_a = g_deck_side_a;
    out.main_b = g_deck_main_b;
    out.side_b = g_deck_side_b;
}

void deck_state_restore(const DeckStateSnapshot &in) {
    g_deck_main_a = in.main_a;
    g_deck_side_a = in.side_a;
    g_deck_main_b = in.main_b;
    g_deck_side_b = in.side_b;
}
