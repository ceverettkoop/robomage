#ifndef CARD_VOCAB_H
#define CARD_VOCAB_H

#include <string>
#include <unordered_map>

// Maps a card name to a 0-based vocabulary index used for one-hot encoding in
// the machine-mode state vector.  Returns -1 for unregistered cards (encoded
// as all-zeros in the one-hot).
//
// To add a new card: append an entry here.  N_CARD_TYPES in machine_io.h must
// be >= (highest index + 1).
inline int card_name_to_index(const std::string &name) {
    static const std::unordered_map<std::string, int> vocab = {
        {"Mountain", 0}, {"Forest", 1}, {"Lightning Bolt", 2}, {"Grizzly Bears", 3}, {"Volcanic Island", 4},
        {"Scalding Tarn", 5}, {"Flooded Strand", 6}, {"Polluted Delta", 7}, {"Wooded Foothills", 8}, {"Misty Rainforest", 9},
        {"Wasteland", 10}, {"Ponder", 11}, {"Force of Will", 12}, {"Daze", 13}, {"Soul Warden", 14}, {"Tundra", 15},
        {"Delver of Secrets", 16}, {"Insectile Aberration", 17}, {"Flying Men", 18}, {"Island", 19},
        {"Dragon's Rage Channeler", 20}, {"Air Elemental", 21}, {"Counterspell", 22}, {"Lightning Strike", 23},
        {"Brainstorm", 24}, {"Thundering Falls", 25},
        {"Murktide Regent", 26}, {"Mishra's Bauble", 27}, {"Cori-Steel Cutter", 28}, {"Unholy Heat", 29}};
    auto it = vocab.find(name);
    return it != vocab.end() ? it->second : -1;
}

// Slot 127 (N_CARD_TYPES - 1) is reserved as sentinel for all tokens in the ML observation.
static constexpr int TOKEN_SENTINEL = 127;

inline const char* card_index_to_name(int idx) {
    static const char* names[] = {
        "Mountain", "Forest", "Lightning Bolt", "Grizzly Bears", "Volcanic Island",   // 0-4
        "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills", "Misty Rainforest", // 5-9
        "Wasteland", "Ponder", "Force of Will", "Daze", "Soul Warden", "Tundra",      // 10-15
        "Delver of Secrets", "Insectile Aberration", "Flying Men", "Island",           // 16-19
        "Dragon's Rage Channeler", "Air Elemental", "Counterspell", "Lightning Strike", // 20-23
        "Brainstorm", "Thundering Falls",                                               // 24-25
        "Murktide Regent", "Mishra's Bauble", "Cori-Steel Cutter", "Unholy Heat",     // 26-29
        // 30-126: reserved for future cards
        "","","","","","","","","","","","","","","","","","","","","", // 30-50
        "","","","","","","","","","","","","","","","","","","","","", // 51-71
        "","","","","","","","","","","","","","","","","","","","","", // 72-92
        "","","","","","","","","","","","","","","","","","","","","", // 93-113
        "","","","","","","","","","","","","",                         // 114-126
        "Token",                                                        // 127
    };
    if (idx < 0 || idx >= 128) return "???";
    return names[idx];
}

#endif /* CARD_VOCAB_H */
