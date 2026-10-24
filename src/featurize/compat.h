#ifndef FEATURIZE_COMPAT_H
#define FEATURIZE_COMPAT_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// ── Model compatibility: what a model was trained with, and whether it runs on today's tables ──
//
// Standard library only (docs/featurizer.md §6.2; learn/rmlearn/spec.py `check_compatible` is the
// same rule in Python). A model — a learner checkpoint or an export — carries a Manifest of the
// tables it was trained with: the FEATURE_SPEC hash and the vocab hash (exact identity), the text
// table's encoder, the layout registry's layout_hash, the tables' digest (every card's card, ability
// and part rows: drift.h), the vocab manifest (the (card id, name) of every card-table row, the names
// the CARDS frame gives each face vocab id) and, for a checkpoint, the layout registry's text (its
// manifest of positions). check_compatible compares a model's
// manifest with the current tables':
//
//   - every model: the same layout_hash and text_model; every (id, name) of the model's vocab
//     present, unchanged, in the current vocab (a new id passes; a removed or renamed id is refused
//     naming it); when both carry a registry, no entry of the model's moved (registry_moves);
//   - exact (an export, which bakes the card rows in): the same spec_hash, vocab_hash and tables'
//     digest too (a known card's rows changed is drift: the baked rows are stale), and no current id
//     the model's vocab lacks — exports are rebuilt per vocab and tables from their checkpoints.
//
// A manifest with no layout_hash is of a model made before manifests: it passes only on tables of
// its own spec_hash, vocab_hash and text_model.

namespace featurize {

class Json;

using VocabManifest = std::vector<std::pair<uint64_t, std::string>>;  // (card id, name), ids ascending

struct Manifest {
        std::string spec_hash, vocab_hash, text_model, layout_hash;  // hashes as 16 lowercase hex digits
        std::string tables_digest;  // FNV-1a 64 of the tables' card digests (drift.h), 16 hex digits
        VocabManifest vocab;
        std::string registry;  // the layout registry's text ("": not carried)
};

// A refusal: `code` the first field that differs (layout_hash, text_model, vocab_hash, spec_hash,
// tables_digest: the inference protocol's error codes), "" when compatible.
struct Refusal {
        std::string code, message;
        bool refused() const { return !code.empty(); }
};

Refusal check_compatible(const Manifest &model, const Manifest &now, bool exact);

// The vocab manifest as a JSON array of [id, name] pairs, and back from its text or a parsed array
// (false with `err`).
std::string vocab_json(const VocabManifest &vocab);
bool parse_vocab_json(const std::string &text, VocabManifest *out, std::string *err);
bool vocab_from_json(const Json &j, VocabManifest *out, std::string *err);

// A manifest as a JSON object (manifest.json beside exported tables; the registry text only with
// `with_registry`), and back (a missing key is left empty; false with `err` when it is not an
// object or a key has the wrong type).
std::string manifest_json(const Manifest &m, bool with_registry);
bool parse_manifest_json(const std::string &text, Manifest *out, std::string *err);

}  // namespace featurize

#endif /* FEATURIZE_COMPAT_H */
