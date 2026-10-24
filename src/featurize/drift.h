#ifndef FEATURIZE_DRIFT_H
#define FEATURIZE_DRIFT_H

#include <cstdint>
#include <string>
#include <vector>

#include "card_tables.h"

// ── Table drift: a known card's inputs changed (docs/featurizer.md §6.2) ───────────────────────
//
// Standard library only. A change to the engine's parsing (or to a card's script) can change a
// known card's card, ability or part rows while every compatibility check (compat.h) still passes:
// the model then reads other inputs for that card. Drift makes such a change visible. A card's rows
// are its card-table row (id), its ability rows (id · A + i) and the part rows of those abilities,
// each as its nonzero values by column name (a reserved column that an append named, still 0,
// changes nothing). An id is described when its card row's `present` column is set.
//
//   card_digests  per described id an FNV-1a 64 of its rows' values (printed with 6 significant
//                 digits, so a last-bit difference of a platform's libm does not move it): the
//                 committed src/featurize/card_digest.json, which the gate's vocab tier checks and
//                 `make registry` rewrites;
//   digest_drift  the ids whose digest differs between two digest lists, and those only in one;
//   table_drift   the same between two tables (an old run's and today's: rmtrain retable), exact,
//                 naming each changed column and its old and new value.

namespace featurize {

struct CardDigest {
        uint64_t id = 0;
        std::string name;
        std::string digest;  // 16 lowercase hex digits
};

struct Drift {
        std::vector<std::string> changed, added, removed;  // one line per card
        bool empty() const { return changed.empty() && added.empty() && removed.empty(); }
};

// `names`: per id its name (CardTables::names, or a manifest's; "" unknown).
std::vector<CardDigest> card_digests(const CardTables &t, const std::vector<std::string> &names);
std::string digest_json(const std::vector<CardDigest> &digests);
bool parse_digest_json(const std::string &text, std::vector<CardDigest> *out, std::string *err);
Drift digest_drift(const std::vector<CardDigest> &base, const std::vector<CardDigest> &now);
Drift table_drift(const CardTables &before, const std::vector<std::string> &before_names, const CardTables &after,
    const std::vector<std::string> &after_names);

}  // namespace featurize

#endif /* FEATURIZE_DRIFT_H */
