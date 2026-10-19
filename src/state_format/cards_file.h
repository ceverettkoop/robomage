#ifndef STATE_FORMAT_CARDS_FILE_H
#define STATE_FORMAT_CARDS_FILE_H

#include <string>

#include "../featurize/setup.h"

// ── --cards-frame FILE: the whole-vocab CARDS frame (docs/featurizer.md §4) ───────────────────
//
// Plays no game: writes the omniscient viewer's SCHEMA frame, the CARDS frame of the
// --closed-decklist card set (every vocab card by its front face, each face a row, and every token
// those cards can make; read from the parse cache, no entity created) and the METAGAME frame to
// FILE as RMSTATE records ("RMSTATE <nbytes>\n<payload>", §9), in that order. The card tables and
// FEATURE_SPEC are built from it (rmfeat --export-tables / --spec). Exit status 0, or 2 when FILE
// cannot be written.
//
// The engine's own featurizer inputs (the nn: agent, --featurize-seats) are built in process from
// the same records and the committed text table (resources/embeddings/oracle_text.rmtb), so they
// equal what rmfeat builds from the file.

extern std::string cards_frame_path;  // --cards-frame FILE

// The file's bytes: the three records.
std::string cards_frame_records();
// Writes the file; the process exit status.
int write_cards_frame_file(const std::string &path);
// The featurizer's inputs from cards_frame_records() and the committed text table (the card tables
// built, not read). False with `err`.
bool load_engine_featurization(featurize::Featurization *out, std::string *err);

#endif /* STATE_FORMAT_CARDS_FILE_H */
