#ifndef RMWORK_EVAL_CACHE_H
#define RMWORK_EVAL_CACHE_H

#include <cstdint>
#include <string>
#include <vector>

#include "match.h"

// ── The re-evaluation cache: $XDG_CACHE_HOME/robomage/workbench/evaluations/ ─────────────────────
//
// One file per (decision log, model, seat): every decision of that seat in the match, each with the
// policy's value and log π over the menu rows, as a served policy returned them (reevaluation.h):
//
//   <root>/<log hash>-<model hash>-<a|b>.rmev
//
// The log hash is the log's bytes' FNV-1a 64 (Match::log_hash, as the replay cache keys it). The
// model hash is that of the model's identity (model_identity): the model file's resolved path, size
// and modification time (a zipped export's archive and member), so a rebuilt or replaced export is
// another entry. The file holds the policy version the server named; an entry whose version is not
// the version the server serves now is stale (reevaluation.h compares them once connected), and so
// is one whose decisions (count, keys, menu sizes) are not the match's. A file is written to
// `<name>.tmp` and renamed into place, only once every decision of the seat is evaluated, and ends
// with a hash of the whole: a truncated or damaged file reads as no entry. After each write the
// oldest entries (by modification time) are removed while the cache holds more than
// EVAL_CACHE_MAX_BYTES, never the one just written; `.tmp` files left by a write that never finished
// go with them.

namespace rmwork {

inline constexpr uint64_t EVAL_CACHE_MAX_BYTES = 512ull << 20;

// $XDG_CACHE_HOME/robomage/workbench/evaluations, else $HOME/.cache/….
std::string eval_cache_root();

// What identifies a model's weights: "<resolved path>\n<size>\n<mtime ns>" of its file (a zipped
// export, `<archive>.zip!<member>`: the archive's, then "\n<member>"); "missing\n<ref>" when the
// file is not there.
std::string model_identity(const std::string &model_ref);

// The entry of (log, model, seat) under `root`.
std::string eval_cache_path(const std::string &root, uint64_t log_hash, const std::string &identity, int seat);

struct CachedDecision {
        uint32_t index = 0;  // the match's decision
        int32_t game = 0;
        bool sideboarding = false;
        int64_t decision = 0;  // the state format's decision key
        float value = 0;
        std::vector<float> log_pi;
};

struct CachedEvals {
        uint64_t log_hash = 0;
        std::string identity;                   // model_identity of the model
        int seat = 0;                           // 0 A, 1 B
        uint32_t policy = 0;                    // the served id it was evaluated under (informative)
        uint64_t version = 0;                   // the version the server named
        std::vector<CachedDecision> decisions;  // the seat's decisions, in match order
};

// The file's bytes; parse_eval_cache is false with `err` on bytes that are not a whole entry.
std::string eval_cache_bytes(const CachedEvals &c);
bool parse_eval_cache(const std::string &bytes, CachedEvals *out, std::string *err);
bool write_eval_cache(const std::string &path, const CachedEvals &c, std::string *err);
bool read_eval_cache(const std::string &path, CachedEvals *out, std::string *err);

// The entry is `m`'s for `seat`: the log's hash, and exactly the seat's decisions (its deciding
// seat's) with their keys and menu sizes. False with `why`.
bool eval_cache_fits(const CachedEvals &c, const Match &m, int seat, std::string *why);

// Removes the oldest entries while `root` holds more than `max_bytes`, keeping `keep`, and every
// `.tmp` file older than ten minutes.
void prune_eval_cache(const std::string &root, uint64_t max_bytes, const std::string &keep);

}  // namespace rmwork

#endif /* RMWORK_EVAL_CACHE_H */
