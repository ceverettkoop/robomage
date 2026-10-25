#ifndef FEATURIZE_ORACLE_TEXT_H
#define FEATURIZE_ORACLE_TEXT_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "rmtb.h"

// ── The oracle-text table's inputs and staleness (plan step 5c, docs/featurizer.md §5) ─────────
//
// Standard library only: the engine (--text-table-check / --text-table-inputs, text_table.h) builds
// the text entries from its cards and calls these; rmembed reads the inputs file and rewrites the
// table; the featurizer reuses the hashes.
//
// A text entry is one face vocab id with oracle text: its normalised text (normalise_card_text)
// and that text's hash. The table (an RMTB file of kind "text" whose one array `vectors` is f32
// [N_CARD_TYPES + 1, hidden], a row per vocab id) is complete when its model, pooling,
// norm_version and vocab_hash are the pin's and the vocab's, and its rows_meta lists every
// entry's id with the entry's hash and no other id. A row's vector depends on its text alone, so
// another vocab_hash (a vocab append) makes only the entries whose row is missing or of another
// hash stale; the other rows are kept and the header rewritten.
//
// The inputs file (--text-table-inputs, read by rmembed) is JSON lines: first a header
//   {"rows":<n>,"vocab_hash":"<16 hex>","model":"<repo>@<commit>","pooling":"<p>","norm_version":<v>,
//    "present":[{"id":<id>,"name":"<name>"},…]}
// (`present`: every text entry, ascending id; rows outside it are dropped from the table), then
// one line per entry to (re)encode: {"id":<id>,"name":"<name>","text":"<text>","hash":"<16 hex>"}.

namespace featurize {

// The text normalisation's version (§5); any change to normalise_card_text bumps it.
const int NORM_VERSION = 1;

// tools/embed/MODEL_PIN: `key value` lines (`#` comments) naming the encoder.
struct ModelPin {
        std::string repo;     // Hugging Face repository, "BAAI/bge-small-en-v1.5"
        std::string commit;   // full commit SHA
        std::string pooling;  // cls or mean
        std::string prefix;   // put before every text (a JSON string in the file)
        // "<repo>@<commit>", the table meta's `model`.
        std::string model() const { return repo + "@" + commit; }
};

// Reads a pin; false with `err` when unreadable, a required key (repo, commit, pooling) is
// missing, pooling is not cls or mean, or prefix is not a JSON string.
bool read_model_pin(const std::string &path, ModelPin *out, std::string *err);

// The pin's model directory under a cache root: <cache>/<repo, '/' as "__">/<commit>
// (tools/embed/fetch_model.py's layout).
std::string model_cache_dir(const std::string &cache_root, const ModelPin &pin);

// FNV-1a 64 of the normalised text's UTF-8 bytes (rows_meta's hash).
uint64_t text_hash(const std::string &normalised);

// FNV-1a 64 over the vocab's (id, name) pairs in ascending id: per pair the id as u32
// little-endian, the name's UTF-8 bytes and a 0 byte (the tables' vocab_hash).
uint64_t vocab_hash(const std::vector<std::pair<int, std::string>> &entries);

// One face's type line and oracle text.
struct FaceText {
        std::string type_line;
        std::string oracle;
};

// The normalised text of a vocab id (version NORM_VERSION):
//   prefix + "<type line>. <oracle>"
// for its face, or, when one id stands for both faces (a split card's halves), the faces joined
// with " // "; with every whole-word occurrence in an oracle of a name in `names` (the face's own
// name and the other face's, plus each one's part before its first ", ", the short name oracle
// text uses for a legendary card; longest first) replaced by CARDNAME, newlines and tabs made
// spaces, runs of spaces collapsed and the text trimmed.
std::string normalise_card_text(
    const std::vector<FaceText> &faces, const std::vector<std::string> &names, const std::string &prefix);

struct TextEntry {
        int64_t id = 0;
        std::string name;  // the vocab name
        std::string text;  // normalised
        uint64_t hash = 0;
};

// What the table must hold: its row count, the vocab hash, the pin's model and pooling, the
// normalisation version, and the text entries (ascending id).
struct TextTableSpec {
        uint64_t rows = 0;
        uint64_t vocab_hash = 0;
        std::string model;
        std::string pooling;
        int64_t norm_version = NORM_VERSION;
        std::vector<TextEntry> entries;
};

// How a table stands against a spec.
struct TextTableStatus {
        std::vector<std::string> mismatches;  // header differences (each makes every entry stale)
        std::string vocab;                    // the vocab_hash difference, when any (no entry stale by it)
        std::vector<size_t> stale;            // indices into spec.entries: missing or another hash
        std::vector<int64_t> extra;           // rows_meta ids that are no entry
        bool ok() const { return mismatches.empty() && vocab.empty() && stale.empty() && extra.empty(); }
};

// The text table's array of vectors.
const char *const TEXT_VECTORS = "vectors";

// `file`'s vectors array when it is f32 of rank 2, else nullptr.
const RmtbArray *text_vectors(const RmtbFile &file);

// The status of `table` (nullptr: no table, everything stale) against `spec`.
TextTableStatus text_table_status(const RmtbFile *table, const TextTableSpec &spec);

// The inputs file's text for `spec`: the header, then the entries at `which` (indices into
// spec.entries).
std::string text_inputs_jsonl(const TextTableSpec &spec, const std::vector<size_t> &which);

// Reads an inputs file: `spec` gets the header and the listed entries, `present` the header's
// present list. False with `err` (and the line) on anything malformed.
bool read_text_inputs(const std::string &path, TextTableSpec *spec,
    std::vector<std::pair<int64_t, std::string>> *present, std::string *err);

}  // namespace featurize

#endif /* FEATURIZE_ORACLE_TEXT_H */
