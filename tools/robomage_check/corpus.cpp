#include "corpus.h"

#include "util.h"

static bool trace_menu_row(const std::string &line);
static bool trace_board_line(const std::string &line);

// A --trace-decisions menu row: ^\|   \d+: (as split_engine_output reads them).
static bool trace_menu_row(const std::string &line) {
    if (!rmcheck::starts_with(line, "|   ")) return false;
    size_t i = 4;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9') ++i;
    return i > 4 && line.compare(i, 2, ": ") == 0;
}

// A --trace-decisions line of the board (every trace line but the decision header, the menu, the
// choice and a one-row menu's "| auto: " line).
static bool trace_board_line(const std::string &line) {
    if (!rmcheck::starts_with(line, "| ")) return false;
    return !rmcheck::starts_with(line, "| decision ") && !rmcheck::starts_with(line, "| menu:") &&
           !trace_menu_row(line) && !rmcheck::starts_with(line, "| >> ") && !rmcheck::starts_with(line, "| auto: ");
}

namespace rmcheck {

std::vector<std::string> corpus_names(const Options &opt) {
    std::vector<std::string> out;
    for (const std::string &f : list_dir(opt.corpus_dir(), ".rmlog")) out.push_back(f.substr(0, f.size() - 6));
    return out;
}

std::string corpus_log(const Options &opt, const std::string &name) {
    return path_join(opt.corpus_dir(), name + ".rmlog");
}

std::string corpus_baseline(const Options &opt, const std::string &name) {
    return path_join(opt.corpus_dir(), name + ".replay");
}

ProcResult replay_corpus_game(const Options &opt, const std::string &name, const std::vector<std::string> &extra) {
    ProcSpec spec;
    spec.argv = {opt.binary, "--replay", corpus_log(opt, name), "--machine", "--narrative", "--trace-decisions"};
    spec.argv.insert(spec.argv.end(), extra.begin(), extra.end());
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 900;
    return run_process(spec);
}

std::string baseline_text(const std::string &out, const std::string &log_path, const std::string &name) {
    std::string result;
    const std::string banner = "REPLAY MODE: Using seed ";
    const std::string from = " from " + log_path;
    size_t start = 0;
    while (start < out.size()) {
        size_t nl = out.find('\n', start);
        const bool has_nl = nl != std::string::npos;
        if (!has_nl) nl = out.size();
        std::string line = out.substr(start, nl - start);
        start = nl + 1;
        if (trace_board_line(line)) continue;
        if (starts_with(line, banner.c_str()) && ends_with(line, from.c_str()))
            line = line.substr(0, line.size() - from.size()) + " from " + name + ".rmlog";
        result += line;
        if (has_nl) result += '\n';
    }
    return result;
}

std::string first_difference(const std::string &expected, const std::string &actual) {
    if (expected == actual) return "";
    const std::vector<std::string> e = split_lines(expected), a = split_lines(actual);
    for (size_t i = 0; i < std::max(e.size(), a.size()); ++i) {
        const std::string x = i < e.size() ? e[i] : "<EOF>", y = i < a.size() ? a[i] : "<EOF>";
        if (x != y)
            return "first diff at line " + std::to_string(i + 1) + ": expected " + quoted(x) + ", actual " + quoted(y);
    }
    return "the texts differ in line endings";
}

}  // namespace rmcheck
