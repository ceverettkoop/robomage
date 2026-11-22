#ifndef RMWORK_POLICY_CATALOG_H
#define RMWORK_POLICY_CATALOG_H

#include <cstdint>
#include <string>
#include <vector>

// ── The policies of the runs directory, newest first (docs/workbench_plan.md §4.6) ──────────────
//
// One flat list over the runs directory: a row is a policy (a run's `final`, its fixed points `m<k>`,
// its other named checkpoints and exports), an export and the checkpoint it was made from being one
// row with the formats it exists in. Where a run's files are (docs/rmtrain.md §4):
//
//   ckpt/fixed/m<k>.pt                      a fixed point                  row m<k>
//   ckpt/rnad-<name>.pt, ckpt/<name>.pt     final, seed, bc-final, bc-b1   row <name> (rnad- dropped)
//   ckpt/rnad-p<step>.pt, ckpt/bc-p<step>.pt  periodic checkpoints         hidden
//   exports/<name>.ts.pt / .pt2 / .onnx     the exports                    row <name>
//   exports/live/current.*                  the live export                row "live", hidden
//
// The seed is hidden too; hidden rows are listed, flagged, for a "Show all". Never listed: the old
// layouts' copies (ckpt.v1, ckpt.layout1, exports.v1, exports.layout1: no directory but those above
// is read), the evaluations' exports (exports/eval/) and temporary files.
//
// A row's date is when the policy was trained: the earliest metrics.jsonl line naming it (an
// `rnad_iteration` for m<k>, a `checkpoint` by its file, an `export` or `latest` by name), else the
// oldest modification time of its files (the live export's always). The metrics files are read once
// and then from where the last read stopped while they grow (a process-wide cache by path), and only
// lines holding one of those event names are parsed.
//
// The runs root's `latest/` (rmtrain's runs/latest: latest.json, model.pt, policy.ts.pt / .pt2 /
// .onnx) is never a run: it is the pinned first row, its run, name, learner steps and date from
// latest.json. Then the runs' rows, newest first (ties: run, then name in natural order, the later
// first); then the zipped exports of the models directory (model_zip.h), in archive then member order,
// undated (their archives' times are when they were copied, not when the policy was trained).
//
// The export a row plays is its TorchScript one, else its .pt2 (an AOTInductor package is compiled for
// the machine that exported it; TorchScript loads anywhere); a row with neither plays nothing.
// Compatibility with the engine's tables is not determined here: it needs the engine's tables
// (its cards frame), so every row is UNKNOWN; `tables_manifest` names the manifest to check against.

namespace rmwork {

enum PolicyFormat : unsigned {
    FORMAT_TS = 1u << 0,
    FORMAT_PT2 = 1u << 1,
    FORMAT_ONNX = 1u << 2,
    FORMAT_CHECKPOINT = 1u << 3,
};

enum class PolicyDateSource {
    METRICS,  // a metrics.jsonl line
    LATEST,   // latest.json's trained_at
    MTIME,    // the files' modification time
    NONE,     // a zipped export
};

enum class PolicyCompat {
    UNKNOWN,
    COMPATIBLE,
    INCOMPATIBLE,
};

struct PolicyRow {
        std::string run;      // the run directory's name ("league26-24h"); latest: latest.json's run
        std::string name;     // "m7", "final", "bc-final", "rnad-p1200", "seed", "live"
        bool latest = false;  // runs/latest
        std::string archive;  // a zipped export: the archive's file name ("" otherwise)
        bool hidden = false;  // periodic checkpoints, the seed, the live export
        double trained_at = 0;  // Unix seconds, 0 unknown
        PolicyDateSource dated_by = PolicyDateSource::NONE;
        int64_t learner_steps = -1;  // -1 unknown
        unsigned formats = 0;        // PolicyFormat bits
        std::string ts, pt2, onnx, checkpoint;  // each format's file (a zipped one: `<archive>.zip!<member>`)
        std::string play;                       // the export to play (ts, else pt2), "" none
        PolicyCompat compat = PolicyCompat::UNKNOWN;
        std::string tables_manifest;  // the run's tables/manifest.json, latest's latest.json ("" none)
        std::string label;            // "latest · league26-24h m7", "2026-11-21 · league26-24h/final"
};

// Every policy of the runs directory `runs` and the models directory `models_dir`'s archives, in the
// order above.
std::vector<PolicyRow> scan_policies(const std::string &runs, const std::string &models_dir);

// "ts, pt2, checkpoint".
std::string policy_formats_text(unsigned formats);
// "2026-11-21" (local time), "" for 0.
std::string policy_date_text(double seconds);

}  // namespace rmwork

#endif /* RMWORK_POLICY_CATALOG_H */
