#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "../../src/featurize/rmtb.h"
#include "../../src/state_format/schema.h"
#include "actor.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

namespace {

// One robomage_nn match of the tier.
struct ActorMatch {
        std::string name;  // its files: <dir>/<name>.txt, .rmlog
        std::vector<std::string> args;
        rmcheck::ProcResult result;
};

const char *const TEXT_TABLE = "resources/embeddings/oracle_text.rmtb";

}  // namespace

static std::string file_stamp(const std::string &path);
static bool run_step(rmcheck::Report &rep, const char *tier, const std::string &what, const rmcheck::ProcSpec &spec,
    const std::string &log);
static std::string decisions_of(const std::string &rmlog);
static void check_match(rmcheck::Report &rep, const rmcheck::ActorSetup &a, const ActorMatch &m);
static void check_refusals(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a);
static void check_refeaturization(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &rmlog, const std::string &live);
static void check_exports(rmcheck::Report &rep, const rmcheck::ActorSetup &a, const std::string &live);

// The FNV-1a 64 of a file's bytes (16 hex digits), "" when it cannot be read.
static std::string file_stamp(const std::string &path) {
    std::string bytes;
    if (!rmcheck::read_file(path, &bytes)) return "";
    return featurize::hex16(rmstate::fnv1a64(bytes.data(), bytes.size()));
}

// Runs one setup step; on failure an error naming `what` and the output saved to `log`.
static bool run_step(rmcheck::Report &rep, const char *tier, const std::string &what, const rmcheck::ProcSpec &spec,
    const std::string &log) {
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    rmcheck::write_file(log, r.out + r.err);
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(r.out + r.err, &errors, nullptr);
    if (r.ok() && errors.empty()) return true;
    rep.error(
        tier, what + " " + r.status_text() + (errors.empty() ? "" : ": " + errors[0].line) + " (see " + log + ")");
    return false;
}

// A decision log's decisions (the lines after DECISIONS), "" when it cannot be read.
static std::string decisions_of(const std::string &rmlog) {
    std::string text;
    if (!rmcheck::read_file(rmlog, &text)) return "";
    const size_t at = text.find("DECISIONS\n");
    return at == std::string::npos ? "" : text.substr(at);
}

// A match ran to its MATCH_RESULT with no error or warning line.
static void check_match(rmcheck::Report &rep, const rmcheck::ActorSetup &a, const ActorMatch &m) {
    const std::string path = rmcheck::path_join(a.dir, m.name + ".txt");
    rmcheck::write_file(path, m.result.out + "\n=== engine stderr ===\n" + m.result.err);
    if (!m.result.ok()) {
        rep.error("actor", m.name + ": robomage_nn " + m.result.status_text() + " (see " + path + ")");
        return;
    }
    std::string result;
    bool drawn = false;
    for (const std::string &line : rmcheck::split_lines(m.result.out)) {
        if (rmcheck::starts_with(line, "MATCH_RESULT: ")) result = line;
        drawn = drawn || (rmcheck::starts_with(line, "GAME_RESULT: ") && rmcheck::ends_with(line, " draw"));
    }
    if (result.empty() || drawn) rep.error("actor", m.name + ": no match result, or a drawn game (see " + path + ")");
    std::vector<rmcheck::ScanLine> errs, warns;
    rmcheck::scan_transcript(m.result.out + m.result.err, &errs, &warns);
    if (!errs.empty()) rep.error("actor", m.name + ": " + rmcheck::quoted(errs[0].line) + " (see " + path + ")");
    if (!warns.empty()) rep.error("actor", m.name + ": " + rmcheck::quoted(warns[0].line) + " (see " + path + ")");
    rep.info(m.name + ": " + result + " -> " + path);
}

// The plain engine refuses nn: naming `make actor`; the actor refuses an unknown option.
static void check_refusals(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a) {
    const struct {
            std::string binary, spec;
            const char *needle;
    } cases[] = {
        {opt.binary, "nn:" + a.ts(), "make actor"},
        {a.nn_binary, "nn:" + a.ts() + ":tmep=2", "unknown option"},
        {a.nn_binary, "nn:remote:/tmp/none.sock", "part 2"},
    };
    for (const auto &c : cases) {
        rmcheck::ProcSpec spec;
        spec.argv = {c.binary, "--agent-a", c.spec, "--agent-b", "explore", "--game-seed", "1"};
        spec.cwd = opt.bin_dir;
        spec.merge_stderr = true;
        const rmcheck::ProcResult r = rmcheck::run_process(spec);
        if (r.ok() || !rmcheck::contains(r.out, c.needle))
            rep.error("actor", rmcheck::basename_of(c.binary) + " --agent-a " + c.spec +
                                   ": expected a refusal naming '" + c.needle + "', got " + r.status_text() + ": " +
                                   rmcheck::quoted(rmcheck::trim(r.out).substr(0, 200)));
    }
    rep.info("refusals: robomage refuses nn: (naming `make actor`); robomage_nn refuses an unknown key and nn:remote:");
}

// rmfeat --featurize over a --replay of the match's decision log equals what nn: featurized live.
static void check_refeaturization(const rmcheck::Options &opt, rmcheck::Report &rep, const rmcheck::ActorSetup &a,
    const std::string &rmlog, const std::string &live) {
    const std::string dump = rmcheck::path_join(a.dir, "match_replay.rmd");
    const std::string out = rmcheck::path_join(a.dir, "match_replay.rmtb");
    rmcheck::ProcSpec replay;
    replay.argv = {opt.binary, "--replay", rmlog, "--rmstate-dump", dump};
    replay.cwd = opt.bin_dir;
    if (!run_step(
            rep, "actor", "robomage --replay of the nn: match", replay, rmcheck::path_join(a.dir, "match_replay.txt")))
        return;
    rmcheck::ProcSpec feat;
    feat.argv = {
        opt.rmfeat, "--featurize", dump, "--viewer", "A", "--out", out, "--cards", a.cards, "--text-table", TEXT_TABLE};
    feat.cwd = opt.bin_dir;
    if (!run_step(rep, "actor", "rmfeat --featurize of the replay's dump", feat,
            rmcheck::path_join(a.dir, "match_replay_rmfeat.txt")))
        return;
    std::string want, got, err;
    featurize::RmtbFile f;
    if (!rmcheck::read_file(live, &want) || !rmcheck::read_file(out, &got) || !featurize::rmtb_decode(want, &f, &err)) {
        rep.error("actor", "re-featurization: cannot read " + live + " or " + out + " " + err);
        return;
    }
    const featurize::RmtbArray *viewer = f.find("viewer");
    const size_t n = viewer ? static_cast<size_t>(viewer->count()) : 0;
    if (want != got || n == 0) {
        rep.error("actor", "re-featurization: rmfeat over the replay (" + out +
                               ") is not the bytes nn: featurized live (" + live + ", " + std::to_string(n) +
                               " decisions)");
        return;
    }
    rep.info(
        "re-featurization: rmfeat --featurize over a --replay of the match equals nn:'s live tensors byte for "
        "byte (" +
        std::to_string(n) + " decisions, " + std::to_string(want.size()) + " bytes)");
}

// The reference policy's three exports agree on the match's decisions (make_reference_policy.py check).
static void check_exports(rmcheck::Report &rep, const rmcheck::ActorSetup &a, const std::string &live) {
    rmcheck::ProcSpec spec;
    spec.argv = {a.python, a.script, "check", "--out", a.prefix, "--tables", a.tables, "--tensors", live, "--decisions",
        "1000"};
    spec.timeout_s = 1800;
    spec.merge_stderr = true;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    const std::string log = rmcheck::path_join(a.dir, "check_exports.txt");
    rmcheck::write_file(log, r.out);
    bool ok = false;
    for (const std::string &line : rmcheck::split_lines(r.out)) {
        if (!rmcheck::starts_with(line, "NN-CHECK: ")) continue;
        rep.info(line.substr(10));
        ok = ok || rmcheck::starts_with(line, "NN-CHECK: ok");
    }
    if (!r.ok() || !ok)
        rep.error("actor",
            "the reference policy's exports disagree or the check failed (" + r.status_text() + ", see " + log + ")");
}

namespace rmcheck {

const char *const TORCH_PYTHON_ENV = "ROBOMAGE_TORCH_PYTHON";

bool actor_setup(const Options &opt, Report &rep, const char *tier, ActorSetup *out, std::string *skip) {
    ActorSetup &a = *out;
    a.nn_binary = path_join(dirname_of(opt.binary), "robomage_nn");
    if (access(a.nn_binary.c_str(), X_OK) != 0) {
        *skip = "no " + a.nn_binary + " (`make actor` builds it)";
        return false;
    }
    const char *py = std::getenv(TORCH_PYTHON_ENV);
    if (!py || !*py) {
        *skip = std::string("$") + TORCH_PYTHON_ENV +
                " is not set (a Python with torch, onnx, onnxscript and onnxruntime builds the reference policy)";
        return false;
    }
    skip->clear();
    a.python = py;
    a.script = path_join(opt.repo, "tools/nn/make_reference_policy.py");
    a.dir = opt.out("actor");
    a.cards = path_join(a.dir, "cards.rmst");
    a.tables = path_join(a.dir, "tables");
    a.prefix = path_join(a.dir, "ref");
    ProcSpec cards;
    cards.argv = {opt.binary, "--cards-frame", a.cards};
    cards.cwd = opt.bin_dir;
    if (!run_step(rep, tier, "robomage --cards-frame", cards, path_join(a.dir, "cards_frame.txt"))) return false;
    remove_tree(a.tables);
    make_dirs(a.tables);
    ProcSpec tables;
    tables.argv = {opt.rmfeat, "--export-tables", a.cards, a.tables, "--text-table", TEXT_TABLE};
    tables.cwd = opt.bin_dir;
    if (!run_step(rep, tier, "rmfeat --export-tables", tables, path_join(a.dir, "export_tables.txt"))) return false;
    const std::string stamp = file_stamp(path_join(a.tables, "spec.json")) + " " + file_stamp(a.script) + "\n";
    const std::string stamp_path = a.prefix + ".stamp";
    std::string old;
    if (read_file(stamp_path, &old) && old == stamp && file_exists(a.ts()) && file_exists(a.pt2()) &&
        file_exists(a.prefix + ".onnx")) {
        rep.info("reference policy: " + a.prefix + ".{ts.pt,pt2,onnx} (current)");
        return true;
    }
    std::remove(stamp_path.c_str());
    ProcSpec build;
    build.argv = {a.python, a.script, "build", "--tables", a.tables, "--out", a.prefix, "--text-table",
        path_join(opt.bin_dir, TEXT_TABLE)};
    build.timeout_s = 1800;
    build.merge_stderr = true;
    const ProcResult r = run_process(build);
    const std::string log = path_join(a.dir, "build_reference_policy.txt");
    write_file(log, r.out);
    if (!r.ok()) {
        rep.error(tier, "tools/nn/make_reference_policy.py build " + r.status_text() + " (see " + log + ")");
        return false;
    }
    write_file(stamp_path, stamp);
    rep.info("reference policy: built " + a.prefix + ".{ts.pt,pt2,onnx} in " + fmt1(r.seconds) + " s (" + log + ")");
    return true;
}

// The actor tier (docs/ci.md): robomage_nn and the reference policy (actor_setup; skipped without
// them). An nn: (TorchScript, sampled) vs explore bo3 match with no error or warning, its live
// tensors (nn:...:dump=) equal to rmfeat's over a --replay of its decision log, the .pt2 export
// playing the same argmax bo3 match as the TorchScript one, the refusals, and the three exports'
// agreement on the match's decisions.
void tier_actor(const Options &opt, Report &rep) {
    ActorSetup a;
    std::string skip;
    if (!actor_setup(opt, rep, "actor", &a, &skip)) {
        if (!skip.empty()) rep.info("actor: skipped: " + skip);
        return;
    }
    const std::string live = path_join(a.dir, "match_live.rmtb");
    const std::string seed = std::to_string(opt.seed + 7);
    std::vector<ActorMatch> matches = {
        {"match",
            {"--agent-a", "nn:" + a.ts() + ":dump=" + live, "--agent-b", "explore", "--deck-a", "league/reanimator",
                "--deck-b", "league/wrb_energy"},
            {}},
        {"argmax_ts",
            {"--agent-a", "nn:" + a.ts() + ":argmax", "--agent-b", "explore", "--deck-a", "league/ur_delver",
                "--deck-b", "league/bug"},
            {}},
        {"argmax_pt2",
            {"--agent-a", "nn:" + a.pt2() + ":argmax", "--agent-b", "explore", "--deck-a", "league/ur_delver",
                "--deck-b", "league/bug"},
            {}},
    };
    parallel_for(matches.size() + 1, opt.jobs, [&](size_t i) {
        if (i == matches.size()) {
            check_refusals(opt, rep, a);
            return;
        }
        ActorMatch &m = matches[i];
        ProcSpec spec;
        spec.argv = {a.nn_binary};
        spec.argv.insert(spec.argv.end(), m.args.begin(), m.args.end());
        spec.argv.insert(spec.argv.end(),
            {"--bo3", "--game-seed", seed, "--narrative", "--decision-log", path_join(a.dir, m.name + ".rmlog")});
        spec.cwd = opt.bin_dir;
        spec.timeout_s = 1800;
        m.result = run_process(spec);
    });
    for (const ActorMatch &m : matches) check_match(rep, a, m);
    const std::string ts = decisions_of(path_join(a.dir, "argmax_ts.rmlog"));
    const std::string pt2 = decisions_of(path_join(a.dir, "argmax_pt2.rmlog"));
    if (ts.empty() || ts != pt2)
        rep.error("actor",
            "the .pt2 export's argmax match is not the TorchScript one's (" + a.dir + "/argmax_{ts,pt2}.rmlog)");
    else
        rep.info("argmax: the .pt2 and TorchScript exports play the same bo3 match (" +
                 std::to_string(split_lines(ts).size() - 1) + " decisions)");
    if (!matches[0].result.ok()) return;
    check_refeaturization(opt, rep, a, path_join(a.dir, "match.rmlog"), live);
    check_exports(rep, a, live);
}

}  // namespace rmcheck
