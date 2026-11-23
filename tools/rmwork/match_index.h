#ifndef RMWORK_MATCH_INDEX_H
#define RMWORK_MATCH_INDEX_H

#include <cstdint>
#include <string>

#include "match.h"

// ── A replay's match index, kept beside its cached dump ─────────────────────────────────────────
//
// index_match decodes every frame of a dump once; a large replay (a bo3 of thousands of decisions)
// takes seconds. The replay cache keeps what that pass found in the entry's `index.bin`, so a
// cached replay opens without decoding the dump again:
//
//   "RMWKIDX\n", the format version (MATCH_INDEX_VERSION), the dump's size and modification time,
//   the decision log's hash; the once-per-match frames (viewer, offset, size); the terminal frames;
//   per decision its key, frames, decider, turn, phase, board header, menu size, reply and its
//   text, life totals; the seats' logs; the result; whether the run stopped; then the FNV-1a hash
//   of everything before it.
//
// Little-endian, strings as a 32-bit length and their bytes. read_match_index refuses a file of
// another version, of another dump (size or modification time) or log, a truncated or corrupt one
// (the hash), and an index that does not fit the log (finish_match_index); the caller then indexes
// the dump again and rewrites it. The once-per-match frames are read from the dump and fed to the
// viewers' streams, and the omniscient log is built from the narrative as index_match does. Play's
// live match has no index file.

namespace rmwork {

inline constexpr uint32_t MATCH_INDEX_VERSION = 1;

// Writes `m`'s index (a replay's, indexed by index_match) to `path`, atomically; `log_hash` is the
// decision log's fnv1a64. False with `err`.
bool write_match_index(const Match &m, const std::string &path, uint64_t log_hash, std::string *err);
// Reads the index at `path` into `m` (log_path, log, dump_path and narrative_path set by the
// caller), opening its dump; false with `err` when it is missing, stale or unreadable, `m` then
// left for index_match.
bool read_match_index(Match *m, const std::string &path, uint64_t log_hash, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_MATCH_INDEX_H */
