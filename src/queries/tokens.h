#ifndef QUERIES_TOKENS_H
#define QUERIES_TOKENS_H

#include <set>
#include <string>

struct AbilityDef;
struct CardData;

// ── Token scripts a card can make (CR 111) ────────────────────────────────────

// The token script stems `def` can make — its Token effects' TokenScript$, the Clue an Investigate
// makes, an Amass's Army, a Mobilize's Warrior — with its sub-abilities, modes, floating triggers
// and granted abilities, added to `out`.
void ability_token_scripts(const AbilityDef &def, std::set<std::string> &out);

// The token script stems the keywords of `cd` (either face) make, added to `out`: the Warrior of a
// mobilize keyword (its trigger is built from the keyword during play, not parsed onto the card).
void keyword_token_scripts(const CardData &cd, std::set<std::string> &out);

#endif /* QUERIES_TOKENS_H */
