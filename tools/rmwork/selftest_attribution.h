#ifndef RMWORK_SELFTEST_ATTRIBUTION_H
#define RMWORK_SELFTEST_ATTRIBUTION_H

#include <string>

#include "app.h"
#include "selftest_reevaluation.h"
#include "selftest_table.h"

// ── Attribution's checks: --selftest (engine-free), --selftest-data (the stand-in server) and
//    --selftest-evals (rmserve with the reference policy) ───────────────────────────────────────

namespace rmwork {

// Engine-free, under `tmp`: the metrics on known evaluations (tv, kl, flip, dlogp, dmargin, dv on either
// seat's scale, a frame without a menu V only), the argmax and runner-up; the cache's entries round trip,
// a damaged or truncated entry refused, the paths per log (a live game's under its log path), pruning;
// Shapley on an additive game (exact, efficient) and its permutations deterministic.
void check_attribution_units(const CheckFn &check, const std::string &tmp);

// On corpus log `log`, with the stand-in server (selftest_fake_server.h): no server and an empty cache
// (nothing attributed, the tool says why); a decision of each seat attributed in the omniscient view (the
// decider's frame): the unchanged evaluation equal to the re-evaluation's of that decision, every unit's
// evaluation that of the decision's tensors changed by the unit (deterministic), the second pass, a
// row's children asked on expanding it; the entry cached and the decision visited again with a version
// check only; a cancel mid-way leaving a whole entry or none and no `.tmp`; the Table layer's tints on the
// cards and player chips the units name (and none with the layer off); a hovered row linking its object;
// the State Tree's impact and its sort; the Tokens tool's heat on the cells evaluated; the seat-view rule
// (a seat view at the other seat's decision attributes its own frame, V only); Shapley's efficiency. The
// tools' open state is restored and the Table brought to the front.
void check_attribution_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log);

// A live Play game (sculpted, seat A against explore) explained at the human's decision through the
// stand-in server: the human seat's frame attributed, the entry under the live game's directory; then
// the game opened in Replay finds it at the same decision.
void check_attribution_play(const CheckFn &check, App &app, const ReevalHooks &h);

// The App's open match (the nn: self-play match `check_reevaluation_real` re-evaluated) through rmserve
// serving `model` as policy 0: the unchanged evaluation equal to the episode's recorded evaluation and the
// re-evaluation's at decisions of both seats; Shapley's efficiency within SHAPLEY_TOLERANCE; occlusion's
// cost per decision (the coarse pass and the second pass) measured and printed (ATTRIBUTION-COST lines).
void check_attribution_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_ATTRIBUTION_H */
