#ifndef RMWORK_MODEL_MANAGER_H
#define RMWORK_MODEL_MANAGER_H

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../../src/featurize/compat.h"
#include "policy_catalog.h"

// ── The Model manager's data: the policies' order and filter, their compatibility (ImGui-free) ───
//
// The rows are policy_catalog.h's. A view of them is the rows the filter keeps (a substring of the
// run or the name, any case; one run; the hidden rows only with "Show all") sorted by a column,
// `latest` first under any sort.
//
// Compatibility (docs/featurizer.md §6.2): a row's model manifest against the engine's tables'.
// The model's is its run's tables/manifest.json (the tables its exports were built on), or latest's
// latest.json `manifest` object, which carries the hashes but neither the vocab manifest nor the
// layout registry: there the vocab is taken to be the engine's when the two vocab_hash values are
// equal (the same vocab), and an export with another vocab_hash is refused by that hash. A row that
// plays an export is checked exactly (an export bakes its tables in); a checkpoint-only row by the
// checkpoint rule. The engine's manifest comes from its whole-vocab cards file (`<engine>
// --cards-frame FILE` in its working directory) and the oracle-text table under its resources/
// (featurize::Featurization, as Model's Read loads them): computed once per engine binary (its path
// and modification time), on the checker's thread, with each manifest file's verdicts (kept by the
// file's path and modification time).

namespace rmwork {

enum class ModelColumn {
    DATE,
    RUN,
    NAME,
    STEPS,
    FORMATS,
    COMPAT
};
inline constexpr int MODEL_COLUMN_COUNT = 6;

struct ModelFilter {
        std::string text;      // a substring of the run or the name ("" any)
        std::string run;       // one run ("" every run)
        bool show_all = false;  // the hidden rows too
};

// The indices into `rows` that `f` keeps, sorted by `column` (ties: newest first), `latest` first.
std::vector<size_t> model_view(
    const std::vector<PolicyRow> &rows, const ModelFilter &f, ModelColumn column, bool ascending);
// The runs of `rows` (latest's and the zipped exports' included when they name one), each once, in
// natural order.
std::vector<std::string> model_runs(const std::vector<PolicyRow> &rows);

// A row's compatibility as the view shows it.
struct CompatVerdict {
        PolicyCompat compat = PolicyCompat::UNKNOWN;
        bool checking = false;  // not known yet: the engine's tables or the file are being read
        std::string why;        // the refusal, or why it is unknown ("" when compatible)
};

// "compatible", "incompatible", "unknown", "checking…".
const char *compat_verdict_text(const CompatVerdict &v);

// A model manifest from `path`: a tables directory's manifest.json, or latest.json's `manifest`
// object. False with `why` (no file, malformed, a latest.json without a manifest).
bool read_model_manifest(const std::string &path, featurize::Manifest *out, std::string *why);
// `model` against the engine's `now` (exact: an export), with the latest.json rule above.
CompatVerdict compat_verdict(featurize::Manifest model, const featurize::Manifest &now, bool exact);
// The hint an incompatible row shows: "rmtrain retable <run>".
std::string retable_hint(const PolicyRow &row);

// Where the engine's manifest comes from.
struct EngineTables {
        std::string engine;      // the engine binary
        std::string workdir;     // its working directory (resources/)
        std::string cards_path;  // where its cards file is written
};

class CompatChecker {
    public:
        CompatChecker() = default;
        CompatChecker(const CompatChecker &) = delete;
        CompatChecker &operator=(const CompatChecker &) = delete;
        ~CompatChecker();

        // The rows' manifests checked against `e`'s tables: the files not yet checked (or changed
        // since) and, for another engine binary, the engine's manifest first. One job at a time; a
        // request made while one runs waits for it.
        void request(const EngineTables &e, const std::vector<PolicyRow> &rows);
        // Takes a finished job's results (and starts the waiting request); true when something changed.
        bool poll();
        // Forgets every file's verdict (Refresh), keeping the engine's manifest.
        void forget_files();
        // A row's verdict from what is known now.
        CompatVerdict verdict(const PolicyRow &row) const;
        // Working: the engine's tables or manifest files being read.
        bool busy() const { return running_ != nullptr; }
        // The engine's manifest is known.
        bool engine_ready() const { return have_now_; }
        // Why the engine's manifest could not be had ("" otherwise).
        const std::string &engine_error() const { return engine_error_; }
        // Uses `now` as the engine's manifest for every engine (the self-test: no engine is run).
        void set_engine_manifest(const featurize::Manifest &now);

    private:
        struct FileVerdicts {
                int64_t mtime_ns = 0;
                bool ok = false;  // the manifest was read
                std::string error;
                CompatVerdict exact, loose;
        };
        struct Job {
                EngineTables tables;
                std::string engine_key;
                bool need_engine = false;
                featurize::Manifest now;
                std::vector<std::pair<std::string, int64_t>> files;  // path, modification time
                // results
                bool now_ok = false;
                std::string now_error;
                std::map<std::string, FileVerdicts> verdicts;
                std::atomic<bool> done{false};
                std::thread thread;
        };
        static void run(Job *job);
        void start(std::unique_ptr<Job> job);
        std::unique_ptr<Job> make_job(const EngineTables &e, const std::vector<PolicyRow> &rows) const;

        std::unique_ptr<Job> running_;
        std::unique_ptr<Job> waiting_;
        std::string engine_key_;  // the engine whose manifest now_ is (or whose failure engine_error_ is)
        bool have_now_ = false;
        bool fixed_now_ = false;
        featurize::Manifest now_;
        std::string engine_error_;
        std::map<std::string, FileVerdicts> verdicts_;
};

}  // namespace rmwork

#endif /* RMWORK_MODEL_MANAGER_H */
