#ifndef ROBOMAGE_CHECK_UTIL_H
#define ROBOMAGE_CHECK_UTIL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// ── robomage_check: strings, files and paths ──────────────────────────────────────────────────
//
// Standard library and POSIX only, like the rest of tools/robomage_check/: the gate driver links no
// engine translation unit (it runs the engine as a subprocess).

namespace rmcheck {

bool starts_with(const std::string &s, const char *prefix);
bool ends_with(const std::string &s, const char *suffix);
bool contains(const std::string &s, const char *needle);
// `s` without leading and trailing ASCII whitespace.
std::string trim(const std::string &s);
// The lines of `text` split at '\n' (a trailing '\r' kept off each line); no final empty line for a
// text that ends with a newline.
std::vector<std::string> split_lines(const std::string &text);
// `s` split at `sep` (empty pieces kept).
std::vector<std::string> split(const std::string &s, char sep);
std::string join(const std::vector<std::string> &parts, const char *sep);
// A C-style quoted rendering of `s` (for failure messages, like Python's repr of a line).
std::string quoted(const std::string &s);

// The whole file, or false when it cannot be read.
bool read_file(const std::string &path, std::string *out);
bool write_file(const std::string &path, const std::string &data);
bool file_exists(const std::string &path);
// The first executable `name` on $PATH, else in /usr/bin, /usr/local/bin or /bin; "" when none.
std::string find_program(const char *name);
bool is_dir(const std::string &path);
// mkdir -p.
bool make_dirs(const std::string &path);
// rm -rf (a file or a directory tree).
void remove_tree(const std::string &path);
// The entries of directory `dir` (names, not paths) ending in `suffix`, sorted.
std::vector<std::string> list_dir(const std::string &dir, const char *suffix);
std::string path_join(const std::string &a, const std::string &b);
std::string dirname_of(const std::string &path);
std::string basename_of(const std::string &path);
// `path` without its last extension.
std::string strip_ext(const std::string &path);
// A new, empty directory under `parent` whose name starts with `prefix`.
std::string make_temp_dir(const std::string &parent, const std::string &prefix);

// The index of the first of `rxs` that does not compile as a std::regex (ECMAScript), or npos. With
// exceptions disabled a malformed pattern aborts the process that compiles it, so they are
// compiled in a forked child. Call it only while no other thread of this process runs (the child
// of a multithreaded fork may deadlock in malloc).
size_t first_bad_regex(const std::vector<std::string> &rxs);

// Seconds since an arbitrary fixed point (a monotonic clock).
double now_seconds();
// "%.1f".
std::string fmt1(double v);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_UTIL_H */
