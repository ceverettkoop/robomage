#ifndef ROBOMAGE_CHECK_COVERAGE_CHECK_H
#define ROBOMAGE_CHECK_COVERAGE_CHECK_H

#include <string>
#include <vector>

#include "gate.h"

// ── The fuzz tier's fixture decks: proof that its games reached a path ────────────────────────
//
// A fixture is decks/tests/mech_<primitive>.dk (vocab cards only, dense in one primitive) with a
// sidecar mech_<primitive>.expect: one coverage key per line (`cost:Sac`; `#` comments), each of
// which its mirror games must hit (the engine's --coverage, src/coverage.h). docs/ci.md, fuzz.

namespace rmcheck {

// The fixtures a fuzz run plays, as specs relative to decks/ ("tests/mech_cost_paths"): every
// mech_* stem under decks/tests/ with a .dk or an .expect, or, when --matchups is an explicit list,
// the fixtures it names.
std::vector<std::string> fuzz_fixtures(const Options &opt);

// The block of seeds `fixture`'s games take, from its name: a new fixture moves no other's games.
long fixture_seed_slot(const std::string &fixture);

// `fixture`'s verdict: its deck checked by the engine's --check-decks, the coverage files of its
// `games` games (each a --coverage output) summed into <out>/coverage/<stem>.txt, and an error for
// each key of its .expect with no hits, for a game that wrote no file, and for an .expect that
// lists no key.
void check_fixture_coverage(
    const Options &opt, Report &rep, const std::string &fixture, const std::vector<std::string> &coverage_files);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_COVERAGE_CHECK_H */
