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
        {"Mountain", 0}, {"Forest", 1}, {"Lightning Bolt", 2}, {"Grizzly Bears", 3}, {"Volcanic Island", 4}};
    auto it = vocab.find(name);
    return it != vocab.end() ? it->second : -1;
}

#endif /* CARD_VOCAB_H */
