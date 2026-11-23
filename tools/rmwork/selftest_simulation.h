#ifndef RMWORK_SELFTEST_SIMULATION_H
#define RMWORK_SELFTEST_SIMULATION_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── Simulate's checks (simulation.h, simulate_dialog.h, simulation_view.h; plan §2.9) ────────────
//
// --selftest, without ImGui (check_simulation_units): the Simulate settings' round trip and refusals;
// the configuration from the dialog's choices (paths resolved, the nn: options, the default jobs); the
// engine's arguments (the seats as played, swapped matches, a scripted seat's teacher, both seats
// scripted with one --teacher-mix, a model's options); a match judged from its output (a win, a draw,
// a bo3 match line, a FATAL, a nonzero exit, no result); the tally and its Wilson interval;
// simulation.json written and read back, a running match read as cancelled, the past simulations
// newest first. With ImGui (check_simulation_app): the dialog from File's action, its key and the
// start page's action, drawn ready to run, Esc closing it; the Simulations window over a synthetic
// past simulation: its summary lines, a finished row opening Replay's load, a failed row the error
// dialog.
//
// --selftest-data (check_simulation_data), with the engine `engine` in `workdir`: explore vs random,
// one bo1 match, opened in Replay when it ends; the scripted agent vs explore (when python3 runs);
// three bo1 matches with alternating seats and two jobs, in the Simulations window, every result
// what the engine's --replay of its log reports and the swapped match's log naming the entrants' decks
// the other way round, simulation.json read back equal, a row opened in Replay; a stand-in engine
// (the real one, but one seed sleeping and one failing with a FATAL line): the failure reported for
// its match alone, the others finished; a cancel mid-batch leaving the finished logs, no partial log
// and a summary that reads back as cancelled.
//
// --selftest-evals (check_simulation_evals): the reference policy against itself, one bo1 match in
// robomage_nn, its result what the plain engine's --replay of the log reports.

namespace rmwork {

struct SimulationTestEnv {
        std::string engine;   // the plain engine (robomage)
        std::string workdir;  // its working directory
        std::string tmp;      // a scratch directory
};

void check_simulation_units(const CheckFn &check, const std::string &tmp);
void check_simulation_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp);
void check_simulation_data(
    const CheckFn &check, App &app, const std::function<void(int)> &frames, const SimulationTestEnv &env);
void check_simulation_evals(const CheckFn &check, App &app, const std::function<void(int)> &frames,
    const SimulationTestEnv &env, const std::string &model);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_SIMULATION_H */
