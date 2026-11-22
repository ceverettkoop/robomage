#ifndef RMWORK_SELFTEST_REEVALUATION_H
#define RMWORK_SELFTEST_REEVALUATION_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── Re-evaluation's checks: --selftest (engine-free), --selftest-data (a stand-in server), and
//    --selftest-evals (rmserve with the reference policy) ────────────────────────────────────────

namespace rmwork {

// Engine-free, under `tmp`: the cache's entries round trip, a damaged, truncated or foreign entry
// refused, an entry that no longer fits its match (a decision's key or menu, the log), the model's
// identity following its file, the paths per log, model and seat, pruning (the oldest first, never
// the one kept, stale `.tmp` files); merging into a synthetic match (a series per seat beside the
// recorded one, a step of its own where a decision had none, replacing by id, the order, the label);
// the served policies; a re-evaluation from the cache alone, one with no server (failed, nothing
// merged beyond the cache) and one of Play's live match (refused).
void check_reevaluation_units(const CheckFn &check, const std::string &tmp);

// What the data checks drive the App with.
struct ReevalHooks {
        std::function<void(int)> frames;                   // headless frames
        std::function<void()> escape;                      // Esc pressed and released
        std::function<bool(const std::string &log)> open;  // opens a log and waits for the load
        std::string engine, workdir, tmp;                  // the engine, its directory, a short temporary dir
};

// On corpus log `log`, with a stand-in RMIP server (a thread speaking the protocol, its values a
// function of each decision's tensors, selftest_fake_server.h): no server (on open: nothing asked, no dialog; asked:
// the "No inference server" dialog); every decision of both seats evaluated in its deciding seat's own frame (its
// values those of that frame's tensors), in requests no larger than the server's batch; the tools on the merged
// evaluations (check_tools_values, and every decision in every view: a seat view sees only its own seat's); reopening
// from the cache with only a version check; a version the server changed, a damaged entry and a replaced model
// evaluated again, the rest from the cache; a cancel mid-way (no entry, no series); a refused handshake and a server
// lost mid-way (the error dialog, no entry) and a retry after it.
void check_reevaluation_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log);

// The App's open match (an nn: self-play match whose episodes recorded every decision's evaluation
// by `model`) re-evaluated through rmserve (beside `engine`) serving `model` as policy 0: every
// decision of both seats evaluated, each value and log π within 1e-5 of the episode's; the tools on
// the recorded and re-evaluated policies together; reopening it re-evaluates nothing. Starts and
// stops the server itself (a check fails when it does not start).
void check_reevaluation_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_REEVALUATION_H */
