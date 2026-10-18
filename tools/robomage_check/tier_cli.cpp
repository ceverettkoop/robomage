#include <string>
#include <vector>

#include "corpus.h"
#include "gate.h"
#include "json.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

// ── The cli tier: golden transcripts of the human CLI (tests/cli/) ──────────────────────────────
//
// Each session is <name>.json, a fixed engine command line and the lines a person types:
//
//   {"name": "<name>", "why": "<what it covers>", "args": ["--player", "A", ...], "input": ["0", ...]}
//
// and <name>.out, its golden transcript. The gate runs `robomage <args> --decision-log
// <out>/cli/<name>.rmlog` from bin/ with the input lines on stdin (then end of input) and compares
// its stdout, the decision log's path in the "Decision log: " line replaced by <name>.rmlog, with
// <name>.out byte for byte. The engine must exit 0 with nothing on stderr, and `robomage --replay`
// of the session's decision log must exit 0 with no error, ending at the session's quit (its QUIT
// line, "QUIT: ..." in the replay's output) when the session ended by quitting. --record writes the
// goldens instead (the replay is still checked).

namespace {

struct CliSession {
        std::string name, why;
        std::vector<std::string> args, input;
};

}  // namespace

static bool load_session(const std::string &path, CliSession *out, std::string *err);
static bool string_list(const rmcheck::Json &doc, const char *key, std::vector<std::string> *out, std::string *err);
static std::string golden_text(const std::string &out, const std::string &log_path, const std::string &name);
static std::string replay_problem(const rmcheck::Options &opt, const std::string &log);

// Reads and validates one session file.
static bool load_session(const std::string &path, CliSession *out, std::string *err) {
    std::string text;
    if (!rmcheck::read_file(path, &text)) {
        *err = "cannot read " + path;
        return false;
    }
    rmcheck::Json doc;
    if (!rmcheck::Json::parse(text, &doc, err)) return false;
    if (!doc.is_object()) {
        *err = "not a JSON object";
        return false;
    }
    for (const auto &m : doc.members())
        if (m.first != "name" && m.first != "why" && m.first != "args" && m.first != "input") {
            *err = "unknown key '" + m.first + "'";
            return false;
        }
    const rmcheck::Json *name = doc.get("name"), *why = doc.get("why");
    out->name = rmcheck::strip_ext(rmcheck::basename_of(path));
    if (!name || !name->is_string() || name->as_string() != out->name) {
        *err = "\"name\" must be the file's stem, \"" + out->name + "\"";
        return false;
    }
    if (!why || !why->is_string() || why->as_string().empty()) {
        *err = "\"why\" (what the session covers) is required";
        return false;
    }
    out->why = why->as_string();
    return string_list(doc, "args", &out->args, err) && string_list(doc, "input", &out->input, err);
}

// The member `key` of `doc` as a list of strings.
static bool string_list(const rmcheck::Json &doc, const char *key, std::vector<std::string> *out, std::string *err) {
    const rmcheck::Json *list = doc.get(key);
    if (!list || !list->is_array()) {
        *err = std::string("\"") + key + "\" must be a list of strings";
        return false;
    }
    for (const rmcheck::Json &item : list->items()) {
        if (!item.is_string()) {
            *err = std::string("\"") + key + "\" must be a list of strings";
            return false;
        }
        out->push_back(item.as_string());
    }
    return true;
}

// The session's stdout as the golden holds it: the decision log's path becomes <name>.rmlog.
static std::string golden_text(const std::string &out, const std::string &log_path, const std::string &name) {
    const std::string from = "Decision log: " + log_path + "\n";
    std::string text = out;
    const size_t at = text.find(from);
    if (at != std::string::npos) text.replace(at, from.size(), "Decision log: " + name + ".rmlog\n");
    return text;
}

// What is wrong with `robomage --replay` of a session's decision log; empty when it replays cleanly.
static std::string replay_problem(const rmcheck::Options &opt, const std::string &log) {
    std::string text;
    if (!rmcheck::read_file(log, &text)) return "no decision log " + log;
    const std::vector<std::string> lines = rmcheck::split_lines(text);
    const bool quit = !lines.empty() && rmcheck::trim(lines.back()) == "QUIT";
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--replay", log};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 120;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(r.out + r.err, &errors, nullptr);
    if (!r.ok() || !errors.empty() || rmcheck::contains(r.out + r.err, "FATAL")) {
        const std::vector<std::string> err_lines = rmcheck::split_lines(r.err);
        const std::string first = !errors.empty() ? errors[0].line : err_lines.empty() ? "" : err_lines[0];
        return "--replay failed (" + r.status_text() + ")" + (first.empty() ? "" : ": " + first);
    }
    if (quit && !rmcheck::contains(r.out, "QUIT: the recorded session quit at decision"))
        return "--replay did not stop at the session's quit";
    return "";
}

namespace rmcheck {

void tier_cli(const Options &opt, Report &rep) {
    const std::vector<std::string> files = list_dir(opt.cli_dir(), ".json");
    if (files.empty()) {
        rep.error("cli", "no session (*.json) in " + opt.cli_dir());
        return;
    }
    const std::string out_dir = opt.out("cli");
    std::vector<std::string> results(files.size());
    parallel_for(files.size(), opt.jobs, [&](size_t i) {
        CliSession s;
        std::string err;
        const std::string path = path_join(opt.cli_dir(), files[i]);
        if (!load_session(path, &s, &err)) {
            rep.error("cli", files[i] + ": " + err);
            return;
        }
        const std::string log = path_join(out_dir, s.name + ".rmlog");
        ProcSpec spec;
        spec.argv = {opt.binary};
        spec.argv.insert(spec.argv.end(), s.args.begin(), s.args.end());
        spec.argv.insert(spec.argv.end(), {"--decision-log", log});
        spec.cwd = opt.bin_dir;
        for (const std::string &line : s.input) spec.stdin_data += line + "\n";
        spec.timeout_s = 120;
        const ProcResult r = run_process(spec);
        std::vector<ScanLine> errors;
        scan_transcript(r.out + r.err, &errors, nullptr);
        if (!r.ok() || !errors.empty() || !r.err.empty()) {
            write_file(path_join(out_dir, s.name + ".out"), r.out + r.err);
            const std::vector<std::string> err_lines = split_lines(r.err);
            const std::string first = !errors.empty() ? errors[0].line : err_lines.empty() ? "" : err_lines[0];
            rep.error("cli", s.name + ": the session failed (" + r.status_text() + ")" +
                                 (first.empty() ? "" : ": " + first) + " (see " + path_join(out_dir, s.name + ".out") +
                                 ")");
            return;
        }
        const std::string replay = replay_problem(opt, log);
        if (!replay.empty()) {
            rep.error("cli", s.name + ": " + replay + " (log: " + log + ")");
            return;
        }
        const std::string text = golden_text(r.out, log, s.name);
        const std::string golden = path_join(opt.cli_dir(), s.name + ".out");
        if (opt.record) {
            std::string old;
            const bool had = read_file(golden, &old);
            if (!write_file(golden, text)) {
                rep.error("cli", s.name + ": cannot write " + golden);
                return;
            }
            results[i] = std::string(had && old == text ? "unchanged " : "recorded  ") + s.name + " (" +
                         std::to_string(split_lines(text).size()) + " lines)";
            return;
        }
        std::string expected;
        if (!read_file(golden, &expected)) {
            rep.error("cli", s.name + ": no golden " + golden + " (record it: robomage_check --tier cli --record)");
            return;
        }
        const std::string diff = first_difference(expected, text);
        if (!diff.empty()) {
            const std::string actual = path_join(out_dir, s.name + ".out");
            write_file(actual, text);
            rep.error("cli", "DRIFT " + s.name + ": " + diff + " (actual output: " + actual +
                                 "; re-record an intended change with robomage_check --tier cli --record)");
            return;
        }
        results[i] = "OK   " + s.name;
    });
    for (const std::string &f : list_dir(opt.cli_dir(), ".out")) {
        const std::string stem = f.substr(0, f.size() - 4);
        if (!file_exists(path_join(opt.cli_dir(), stem + ".json")))
            rep.error("cli", "golden " + f + " has no session " + stem + ".json");
    }
    size_t ok = 0;
    for (const std::string &r : results) {
        if (r.empty()) continue;
        ++ok;
        rep.info(r);
    }
    rep.info(std::string(opt.record ? "recorded " : "cli: ") + std::to_string(ok) + "/" + std::to_string(files.size()) +
             (opt.record ? " goldens" : " sessions byte-identical"));
}

}  // namespace rmcheck
