#ifndef RMWORK_REPLAY_CACHE_H
#define RMWORK_REPLAY_CACHE_H

#include <cstdint>
#include <string>

// ── The replay cache: $XDG_CACHE_HOME/robomage/workbench/replays/ (default ~/.cache/…) ─────────
//
// A replayed decision log's frames and narrative are kept so a second open skips the engine. An
// entry is a directory named by its key, the log's content hash and the engine binary's identity
// (its resolved path, size and modification time), so an engine rebuild replays again:
//
//   <root>/<key>/dump.rmst      --rmstate-dump: every viewer's frames
//   <root>/<key>/narrative.txt  the engine's stdout: the --narrative log and its result lines
//   <root>/<key>/stderr.txt     the engine's stderr
//   <root>/<key>/entry.json     written last: the entry is complete (the log, the engine, sizes)
//
// A replay writes into <key>.tmp-<pid> and renames it into place once the engine has exited 0.
// After each new entry the oldest entries (by entry.json's modification time) are removed while the
// cache holds more than CACHE_MAX_BYTES, never the newest.

namespace rmwork {

inline constexpr uint64_t CACHE_MAX_BYTES = 4ull << 30;

struct EngineIdentity {
        std::string path;  // resolved (realpath)
        uint64_t size = 0;
        int64_t mtime_ns = 0;
        bool found = false;
};

EngineIdentity engine_identity(const std::string &binary);
// $XDG_CACHE_HOME/robomage/workbench/replays, else $HOME/.cache/robomage/workbench/replays.
std::string replay_cache_root();
// 64-bit FNV-1a.
uint64_t fnv1a64(const std::string &data, uint64_t h = 0xcbf29ce484222325ull);
// "<16 hex of the log's hash>-<16 hex of the engine identity's hash>".
std::string replay_cache_key(const std::string &log_bytes, const EngineIdentity &engine);

struct CacheEntry {
        std::string dir, dump, narrative, stderr_path, entry_json;
};
CacheEntry cache_entry(const std::string &root, const std::string &key);
// The entry is complete (entry.json and the dump present).
bool cache_entry_complete(const CacheEntry &e);
// Writes entry.json last; false with `err`.
bool write_cache_entry(
    const CacheEntry &e, const std::string &log_path, const EngineIdentity &engine, size_t decisions, std::string *err);
// Removes the oldest complete entries (and stale .tmp directories of dead processes) while the
// cache is larger than `max_bytes`, keeping `keep`.
void prune_replay_cache(const std::string &root, uint64_t max_bytes, const std::string &keep);

}  // namespace rmwork

#endif /* RMWORK_REPLAY_CACHE_H */
