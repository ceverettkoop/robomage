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
        {"Brainstorm", 24}, {"Thundering Falls", 25}};
    auto it = vocab.find(name);
    return it != vocab.end() ? it->second : -1;
}

inline const char* card_index_to_name(int idx) {
    static const char* names[] = {
        "Mountain", "Forest", "Lightning Bolt", "Grizzly Bears", "Volcanic Island",
        "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills", "Misty Rainforest",
        "Wasteland", "Ponder", "Force of Will", "Daze", "Soul Warden", "Tundra",
        "Delver of Secrets", "Insectile Aberration", "Flying Men", "Island",
        "Dragon's Rage Channeler", "Air Elemental", "Counterspell", "Lightning Strike",
        "Brainstorm", "Thundering Falls", "???", "???", "???", "???", "???",
    };
    if (idx < 0 || idx >= 32) return "???";
    return names[idx];
}

#endif /* CARD_VOCAB_H */
