#ifndef FILE_UTIL_H
#define FILE_UTIL_H

#include <string>

// Shared file helpers (POSIX), C-style global functions (see CLAUDE.md).

// mkdir -p: creates `path` and its missing parents; true when it is a directory afterwards.
bool make_dirs(const std::string &path);

// Writes `data` to `path` whole: into `<path>.tmp`, then renamed over `path`, so a reader never sees
// part of it. False with `err` when it cannot.
bool write_file_atomic(const std::string &path, const std::string &data, std::string *err);

#endif /* FILE_UTIL_H */
