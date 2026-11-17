#ifndef RMWORK_FILE_PICKER_H
#define RMWORK_FILE_PICKER_H

#include <cstdint>
#include <string>
#include <vector>

// ── A modal file browser (ImGui has none) ────────────────────────────────────────────────────────
//
// A path field, the places (home, the repository, its runs and replay corpus, the recent files),
// the current directory's listing (directories first, then the files the filter admits, in natural
// order: "a0-m2" before "a0-m10"), a filter, and Open / Cancel. For saving (open_save) a file name
// field and Save take Open's place: the chosen path is the directory joined with the name, and a
// file that exists asks first (Save becomes Replace). A file is chosen by double-clicking
// it or selecting it and pressing Open (or Enter); a directory is entered by double-clicking it, or
// chosen with "Choose this directory" when the filter is Directories. Generic: the caller says what
// it is picking (`purpose`, its own tag) and with which filter it starts.
//
// Listing reads names and kinds only (readdir); size and time are read for the rows drawn, so a
// directory of a run's 180 000 decision logs lists without stalling.

namespace rmwork {

enum class FileFilter {
    DECISION_LOGS,  // *.rmlog
    EPISODES,       // *.rmtraj
    LOGS_AND_EPISODES,
    DIRECTORIES,
    ALL
};
inline constexpr int FILE_FILTER_COUNT = 5;
const char *file_filter_label(FileFilter f);
// Whether a file named `name` passes `f` (directories pass every filter).
bool file_filter_admits(FileFilter f, const std::string &name);
// Natural order: digit runs compared as numbers.
bool natural_less(const std::string &a, const std::string &b);

struct PickerEntry {
        std::string name;
        bool dir = false;
        mutable bool stat_done = false;
        mutable uint64_t size = 0;
        mutable int64_t mtime = 0;
};

class FilePicker {
    public:
        // Opens at `dir` (the directory of `dir` when it names a file; home when it is missing).
        void open(const std::string &title, int purpose, FileFilter filter, const std::string &dir,
            const std::vector<std::string> &recent);
        // Opens for saving as `name` in `dir`.
        void open_save(const std::string &title, int purpose, FileFilter filter, const std::string &dir,
            const std::string &name, const std::vector<std::string> &recent);
        bool saving() const { return save_; }
        // Draws it while open (at the frame's top level); true in the frame a path was chosen.
        bool draw(std::string *chosen);
        bool is_open() const { return open_ || want_open_; }
        int purpose() const { return purpose_; }
        // The listing as shown (filtered and sorted), and its directory.
        const std::vector<PickerEntry> &entries() const { return entries_; }
        const std::string &directory() const { return dir_; }
        FileFilter filter() const { return filter_; }
        // Lists `dir` (false, the listing unchanged, when it cannot be read).
        bool navigate(const std::string &dir);
        void set_filter(FileFilter f);

    private:
        void list();
        void draw_places(std::string *chosen, bool *picked);
        void draw_listing(std::string *chosen, bool *picked);
        void up();
        void draw_save_name(std::string *chosen, bool *picked);
        // Chooses `<dir>/<name>` for saving, unless the file exists and Replace was not pressed yet.
        void try_save(const std::string &path, std::string *chosen, bool *picked);

        bool want_open_ = false, open_ = false;
        std::string title_;
        int purpose_ = 0;
        FileFilter filter_ = FileFilter::DECISION_LOGS;
        std::string dir_;
        std::vector<std::string> recent_;
        std::vector<PickerEntry> all_;      // the directory's entries
        std::vector<PickerEntry> entries_;  // shown
        std::string path_field_;
        std::string selected_;
        std::string error_;
        bool scroll_top_ = false;
        bool save_ = false;
        std::string save_name_;
        std::string replace_path_;  // the existing file Replace would overwrite
};

}  // namespace rmwork

#endif /* RMWORK_FILE_PICKER_H */
