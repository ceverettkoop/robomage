#ifndef RMWORK_SELFTEST_POLICY_COMPARE_H
#define RMWORK_SELFTEST_POLICY_COMPARE_H

#include <string>

#include "app.h"
#include "selftest_reevaluation.h"
#include "selftest_table.h"

// ── Policy Compare's checks: --selftest (engine-free), --selftest-data (a stand-in server) and
//    --selftest-evals (rmserve with the reference policy) ────────────────────────────────────────

namespace rmwork {

// Engine-free: KL, TV and the argmax on made-up distributions against hand-computed values; a synthetic
// match's comparison (each decision's metrics, the reference's own entry, decisions the reference did not
// evaluate left out), its agreement overall, per seat and per turn, the ranked list (order, the policy filter,
// "argmax differs only", the shift) and the candidates and decisions per viewer.
void check_policy_compare_units(const CheckFn &check);

// On corpus log `log`, with a stand-in RMIP server whose policies' π differ (selftest_fake_server.h): the tool
// with no policy and with one; three re-evaluated policies, every decision's KL, TV and argmax against values
// computed here from the merged records, the agreement counts, the default reference (the last); the ranked list
// sorted, filtered to argmax differences and a row's jump moving the cursor; a decision panel row's hover linking
// its menu row; every viewer (a seat view only its seat's decisions, the other seat's decision not shown); the
// catalog popup listing a made-up runs directory's policies, and two of them brought in: served under new ids,
// the replay re-evaluated with them through the server, their series compared, the newest the reference.
void check_policy_compare_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log);

// The App's open match (an nn: self-play match recorded by `model`) re-evaluated through rmserve (beside
// `engine`) serving `model` twice (policies 0 and 1): the two re-evaluated policies, and the recorded one,
// agree at every decision (KL below 1e-6, every argmax the same). Starts and stops the server itself.
void check_policy_compare_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_POLICY_COMPARE_H */
