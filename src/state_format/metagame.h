#ifndef STATE_FORMAT_METAGAME_H
#define STATE_FORMAT_METAGAME_H

#include <map>
#include <string>
#include <utility>
#include <vector>

// ── The reference metagame (docs/state_format.md §4, the METAGAME frame) ───────────────────────
//
// A set of named reference decklists, each with an archetype label, that the engine sends every
// viewer once per match. Source: bin/resources/decks/archetypes.json, the same file the Python
// side's train/archetypes.py reads:
//   - `_archetypes`: the canonical archetype labels in index order; the label list is these plus
//     the fallback `unknown` (archetypes.UNKNOWN_NAME) last;
//   - every other key not starting with `_`: a deck stem (relative to decks/) tagged with its
//     label. Each tagged deck whose .dk file exists, whose label is listed and whose every card has
//     a vocab entry is a reference list; the others are left out (`skipped` says why).
// Lists are in ascending stem order; a list's id is its index. Card ids are physical card ids
// (physical_card_of: a deck line naming a back face counts as its front face), as DECKLISTS has
// them. The registered decks of the match are not added: the metagame is the same for every match.

struct ReferenceList {
        std::string name;                          // the deck stem (as --deck-a / --deck-b name it)
        int archetype = 0;                         // index into ReferenceMetagame::archetypes
        std::map<int, std::pair<int, int>> cards;  // card id → (main copies, side copies), ascending
        int main = 0;
        int side = 0;
};

struct ReferenceMetagame {
        std::vector<std::string> archetypes;  // labels, archetype id order; `unknown` last
        std::vector<ReferenceList> lists;     // ascending name
        std::vector<std::string> skipped;     // "<stem>: <why>" per tagged deck left out
};

// The reference metagame, read from RESOURCE_DIR/decks/archetypes.json once per process (an
// unreadable or malformed file gives an empty metagame, with the reason in `skipped`).
const ReferenceMetagame &reference_metagame();

#endif /* STATE_FORMAT_METAGAME_H */
