#ifndef TEXT_TABLE_H
#define TEXT_TABLE_H

#include <string>

#include "featurize/oracle_text.h"

// ── The oracle-text table's engine side (plan step 5c, docs/featurizer.md §5) ──────────────────
//
// The text entries come from the vocab (src/card_vocab.h): every card entry and registered token
// whose face has oracle text, one per face vocab id (a front face under its own id, a
// double-faced card's back face under the back face's entry; a split card, whose halves share one
// id, as both halves), read from the parse cache (card_data_by_name, parse_token_script), so no
// entity is created. The text is featurize::normalise_card_text of the face's type line (its
// Types: line in script order: supertypes, "Token" for a token, card types, then " — " and the
// subtypes), its oracle text and its names. Both modes play no game:
//
//   --text-table-check FILE --model-pin PIN
//       exit 0 when FILE is complete for the pin (featurize::text_table_status), printing one
//       "TEXT-TABLE: ok …" line; otherwise one line per header mismatch, one
//       "TEXT-TABLE: stale <id> <name>" per missing or stale entry, one "TEXT-TABLE: extra <id>"
//       per row that is no entry, and a summary, then exit 1. A missing or unreadable FILE makes
//       every entry stale.
//   --text-table-inputs FILE --model-pin PIN [--text-table TABLE]
//       writes the inputs file (featurize::text_inputs_jsonl) with the entries the check of TABLE
//       reports stale (every entry when TABLE is not given or missing).
//
// Exit 2 on an unreadable pin, an unwritable inputs file or a vocab with no text entry (card
// scripts not provisioned).

extern std::string text_table_check_path;   // --text-table-check FILE
extern std::string text_table_inputs_path;  // --text-table-inputs FILE
extern std::string text_table_path;         // --text-table TABLE
extern std::string model_pin_path;          // --model-pin PIN

// True when --text-table-check or --text-table-inputs was given.
bool text_table_requested();

// Runs the requested mode; the process exit status.
int run_text_table();

// What the text table must hold for `pin`: N_CARD_TYPES + 1 rows, the vocab hash, the pin's model
// and pooling, NORM_VERSION and the text entries in ascending id.
featurize::TextTableSpec text_table_spec(const featurize::ModelPin &pin);

#endif /* TEXT_TABLE_H */
