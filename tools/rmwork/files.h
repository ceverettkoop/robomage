#ifndef RMWORK_FILES_H
#define RMWORK_FILES_H

#include <cstdint>
#include <string>

// ── File helpers (POSIX) ─────────────────────────────────────────────────────────────────────────

namespace rmwork {

// mkdir -p: true when `path` is a directory afterwards.
bool make_dirs(const std::string &path);
// The directory part of `path` ("" for a bare name).
std::string parent_dir(const std::string &path);
bool file_exists(const std::string &path);

// A file's size and modification time (nanoseconds), following symbolic links.
struct FileStamp {
        bool found = false;
        bool regular = false;
        uint64_t size = 0;
        int64_t mtime_ns = 0;
};
FileStamp file_stamp(const std::string &path);

enum class PathKind {
    MISSING,
    FILE,
    DIRECTORY,
    SOCKET,
    OTHER
};
// What is at `path` (following symbolic links).
PathKind path_kind(const std::string &path);
// A regular file the process may execute.
bool is_executable(const std::string &path);
// The running executable's path ("" when it cannot be found).
std::string executable_path();
// The current directory ("" when it cannot be read).
std::string current_dir();
// `path` made absolute against the current directory (unchanged when absolute or empty).
std::string absolute_path(const std::string &path);
bool read_file(const std::string &path, std::string *out);
// Writes `data` to `<path>.tmp`, then renames it over `path`; false with `err` when it cannot.
bool write_file_atomic(const std::string &path, const std::string &data, std::string *err);
// A new directory under $TMPDIR (or /tmp) named `<prefix>XXXXXX`; "" when it cannot be made.
std::string make_temp_dir(const char *prefix);
// Removes `path` and everything under it.
void remove_tree(const std::string &path);
// "20261116-191502": the local time now, for the names of files made now.
std::string file_timestamp();

}  // namespace rmwork

#endif /* RMWORK_FILES_H */
