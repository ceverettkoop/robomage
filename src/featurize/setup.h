#ifndef FEATURIZE_SETUP_H
#define FEATURIZE_SETUP_H

#include <string>

#include "card_tables.h"
#include "feature_spec.h"
#include "frame_stream.h"
#include "rmtb.h"

// ── The featurizer's inputs, loaded (docs/featurizer.md §4, §6), standard library and the decoder only
//
// A whole-vocab cards file (robomage --cards-frame: SCHEMA, CARDS, METAGAME records) and the text
// table (its `vectors` row count is N_CARD_TYPES + 1, its vocab_hash the vocab's) give the card
// tables and the spec: the tables built from the cards file's CARDS frame, or read from their
// exported files (which must carry the spec's hash), and the spec from its SCHEMA and METAGAME
// frames and the tables. export_tables writes card.rmtb, ability.rmtb, part.rmtb and spec.json.

namespace featurize {

class Featurization {
    public:
        ViewerFrames source;  // the cards file's frames
        CardTables tables;
        FeatureSpec spec;

        // Loads the cards file and the text table; `tables_dir` "" builds the tables, else reads
        // its card.rmtb, ability.rmtb and part.rmtb. False with `err`.
        bool load(const std::string &cards_path, const std::string &text_table_path, const std::string &tables_dir,
            std::string *err);
        // The tables' files and spec.json into `dir` (created by the caller).
        bool export_tables(const std::string &dir, std::string *err) const;
        // The file names in a tables directory.
        static const char *const CARD_FILE;
        static const char *const ABILITY_FILE;
        static const char *const PART_FILE;
        static const char *const SPEC_FILE;

    private:
        static bool read_tables_dir(const std::string &dir, CardTables *out, uint64_t *spec_hash, std::string *err);
};

}  // namespace featurize

#endif /* FEATURIZE_SETUP_H */
