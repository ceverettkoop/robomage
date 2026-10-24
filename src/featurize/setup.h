#ifndef FEATURIZE_SETUP_H
#define FEATURIZE_SETUP_H

#include <string>

#include "card_tables.h"
#include "compat.h"
#include "feature_spec.h"
#include "frame_stream.h"
#include "rmtb.h"

// ── The featurizer's inputs, loaded (docs/featurizer.md §4, §6), standard library and the decoder only
//
// A whole-vocab cards file (robomage --cards-frame: SCHEMA, CARDS records) and the text
// table (its `vectors` row count is N_CARD_TYPES + 1, its vocab_hash the vocab's) give the card
// tables and the spec: the tables built from the cards file's CARDS frame, or read from their
// exported files (which must carry the spec's hash; the printed rows the object deltas compare
// with still come from the CARDS frame), and the spec from its SCHEMA frame and the tables, under
// `options` (the recent-event history, feature_spec.h), both at the positions of `registry` (the
// compiled-in layout registry unless the caller sets another before loading, registry.h).
// export_tables writes card.rmtb, ability.rmtb, part.rmtb, spec.json, manifest.json (the tables'
// model manifest, compat.h, without the registry) and registry.json (the layout registry).

namespace featurize {

class Featurization {
    public:
        ViewerFrames source;  // the cards file's frames
        CardTables tables;
        FeatureSpec spec;
        std::string text_model;                  // the text table's encoder (its `model`, "<repository>@<commit>")
        FeatureOptions options;                  // what the spec is built with (set before loading)
        Registry registry = builtin_registry();  // the layout (set before loading)

        // Loads the cards file and the text table; `tables_dir` "" builds the tables, else reads
        // its card.rmtb, ability.rmtb and part.rmtb. False with `err`.
        bool load(const std::string &cards_path, const std::string &text_table_path, const std::string &tables_dir,
            std::string *err);
        // The same from the cards file's bytes in memory.
        bool load_records(const std::string &records, const std::string &text_table_path, const std::string &tables_dir,
            std::string *err);
        // The tables' files, spec.json, manifest.json and registry.json into `dir` (created by the
        // caller).
        bool export_tables(const std::string &dir, std::string *err) const;
        // What a model trained on these tables carries (compat.h), the registry's text included.
        Manifest manifest() const;
        // The file names in a tables directory.
        static const char *const CARD_FILE;
        static const char *const ABILITY_FILE;
        static const char *const PART_FILE;
        static const char *const SPEC_FILE;
        static const char *const MANIFEST_FILE;
        static const char *const REGISTRY_FILE;

        // The card tables of an exported tables directory (no names or printed rows) and the spec hash
        // they carry; false with `err`.
        static bool read_tables_dir(const std::string &dir, CardTables *out, uint64_t *spec_hash, std::string *err);

    private:
        // The tables and the spec from `source` and the text table.
        bool build(const std::string &text_table_path, const std::string &tables_dir, std::string *err);
};

}  // namespace featurize

#endif /* FEATURIZE_SETUP_H */
