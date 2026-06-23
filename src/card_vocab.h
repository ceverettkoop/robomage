#ifndef CARD_VOCAB_H
#define CARD_VOCAB_H

#include <string>
#include <unordered_map>

#include "machine_io.h"  // N_CARD_TYPES (embedding vocab size)

// Single source of truth for card name <-> vocab index mapping.
// To add a new card: append an entry to card_vocab_entries.
// N_CARD_TYPES in machine_io.h must be >= (highest index + 1).
struct CardVocabEntry {
    const char *name;
    int index;
};

inline constexpr CardVocabEntry card_vocab_entries[] = {
    {"Mountain", 0}, {"Forest", 1}, {"Lightning Bolt", 2}, {"Grizzly Bears", 3}, {"Volcanic Island", 4},
    {"Scalding Tarn", 5}, {"Flooded Strand", 6}, {"Polluted Delta", 7}, {"Wooded Foothills", 8}, {"Misty Rainforest", 9},
    {"Wasteland", 10}, {"Ponder", 11}, {"Force of Will", 12}, {"Daze", 13}, {"Soul Warden", 14}, {"Tundra", 15},
    {"Delver of Secrets", 16}, {"Insectile Aberration", 17}, {"Flying Men", 18}, {"Island", 19},
    {"Dragon's Rage Channeler", 20}, {"Air Elemental", 21}, {"Counterspell", 22}, {"Lightning Strike", 23},
    {"Brainstorm", 24}, {"Thundering Falls", 25},
    {"Murktide Regent", 26}, {"Mishra's Bauble", 27}, {"Cori-Steel Cutter", 28}, {"Unholy Heat", 29},
    {"Birds of Paradise", 30}, {"Collector Ouphe", 31}, {"Dryad Arbor", 32}, {"Endurance", 33},
    {"Gaea's Cradle", 34}, {"Green Sun's Zenith", 35}, {"Horizon Canopy", 36}, {"Icetill Explorer", 37},
    {"Ignoble Hierarch", 38}, {"Karakas", 39}, {"Keen-Eyed Curator", 40}, {"Knight of the Reliquary", 41},
    {"Noble Hierarch", 42}, {"Once Upon a Time", 43}, {"Plains", 44}, {"Savannah", 45},
    {"Scryb Ranger", 46}, {"Scythecat Cub", 47}, {"Swords to Plowshares", 48}, {"Sylvan Library", 49},
    {"Talon Gates of Madara", 50}, {"Thalia, Guardian of Thraben", 51}, {"Windswept Heath", 52},
    {"Doomsday", 53}, {"Thoughtseize", 54}, {"Dark Ritual", 55}, {"Lotus Petal", 56},
    {"Lion's Eye Diamond", 57}, {"Thassa's Oracle", 58}, {"Personal Tutor", 59}, {"Street Wraith", 60},
    {"Edge of Autumn", 61}, {"Swamp", 62}, {"Undercity Sewers", 63}, {"Underground Sea", 64},
    {"Bloodstained Mire", 65}, {"Verdant Catacombs", 66}, {"Cavern of Souls", 67},
    {"Consider", 68}, {"Duress", 69}, {"Deep Analysis", 70},
    {"Null Rod", 71}, {"Force of Negation", 72}, {"Force of Vigor", 73}, {"Faerie Macabre", 74},
    {"Abrade", 75}, {"Hydroblast", 76}, {"Pyroblast", 77}, {"Knight of Autumn", 78},
    {"Dismember", 79}, {"Meltdown", 80}, {"Deafening Silence", 81}, {"Choke", 82},
    {"Long Goodbye", 83}, {"Fatal Push", 84}, {"Doorkeeper Thrull", 85},
    {"Magus of the Moon", 86},
    {"Barrowgoyf", 87}, {"Dauthi Voidwalker", 88}, {"Stifle", 89},
    {"Life from the Loam", 90},
    {"Disruptor Flute", 91},
    {"Surgical Extraction", 92},
    {"Orcish Bowmasters", 93},
};

inline constexpr int CARD_VOCAB_SIZE = sizeof(card_vocab_entries) / sizeof(card_vocab_entries[0]);

// Slot N_CARD_TYPES - 1 is reserved as sentinel for all tokens in the ML observation.
static constexpr int TOKEN_SENTINEL = N_CARD_TYPES - 1;

// Maps a card name to a 0-based vocabulary index used to encode card identity in
// the machine-mode state vector.  Returns -1 for unregistered cards (encoded as
// the empty/unknown sentinel id).
inline int card_name_to_index(const std::string &name) {
    static const std::unordered_map<std::string, int> vocab = [] {
        std::unordered_map<std::string, int> m;
        for (int i = 0; i < CARD_VOCAB_SIZE; i++)
            m[card_vocab_entries[i].name] = card_vocab_entries[i].index;
        return m;
    }();
    auto it = vocab.find(name);
    return it != vocab.end() ? it->second : -1;
}

inline const char* card_index_to_name(int idx) {
    static const char** names = [] {
        static const char* table[N_CARD_TYPES]{};
        for (int i = 0; i < N_CARD_TYPES; i++) table[i] = "";
        table[TOKEN_SENTINEL] = "Token";
        for (int i = 0; i < CARD_VOCAB_SIZE; i++)
            table[card_vocab_entries[i].index] = card_vocab_entries[i].name;
        return table;
    }();
    if (idx < 0 || idx >= N_CARD_TYPES) return "???";
    return names[idx];
}

#endif /* CARD_VOCAB_H */
