#ifndef ROBOMAGE_CHECK_EMBED_H
#define ROBOMAGE_CHECK_EMBED_H

#include <string>

#include "gate.h"

// ── robomage_check: the oracle-text generator's std-only parts (docs/featurizer.md §5) ────────

namespace rmcheck {

// The vocab tier's tokenizer check: src/featurize/wordpiece.cpp over the pinned model's vocab
// (.cache/embed/<repo>/<commit>/, tools/embed/fetch_model.py) against tests/embed/tokens_golden.jsonl
// (tools/embed/make_goldens.py tokens), id for id; and every current text entry (the engine's
// --text-table-inputs) tokenises without a refusal. Skipped, with an info line, when the model's
// vocab is not in the cache.
void embed_tokenizer_check(const Options &opt, Report &rep);

// The smoke tier's RMTB samples: checks src/featurize/rmtb.cpp's f16 rounding on fixed cases,
// writes two small files with it (one array of every dtype and rank 0-3 with a ragged values +
// offsets pair; a text table with every file-level key), checks that they decode back to the same
// bytes, and writes their meta and values as JSON for python/pycheck.py's rmtb part.
// Returns that JSON file's path ("" when writing failed, reported as an error).
std::string write_rmtb_samples(const Options &opt, Report &rep);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_EMBED_H */
