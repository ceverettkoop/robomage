#ifndef AGENTS_EXPLORE_H
#define AGENTS_EXPLORE_H

#include <set>
#include <string>

#include "agent.h"

// ── The explore tiers: the coverage fuzzer, python/explore_agent.py ported ────────────────────
//
// The decision rules are explore_agent.py's (its module docstring), applied to the same view
// (DecisionView: category names, card(i), the own hand, the CARDS frame's mana values) with the
// same weights in the same floating-point order, and the row drawn as Python's
// random.choices(range(n), weights, k=1) on the shared PyRandom. For the same frames and the
// same generator state it therefore picks the same row as the Python tier: a game seeded the same
// way (--game-seed N here; seed N on python/rm_driver.py) plays the same game.
//
//   1. A menu of one row (or none) is answered 0 with no draw.
//   2. A menu holding a MULLIGAN row starts a game: the novelty set is cleared.
//   3. Weight per row: the profile's category weight (default 2.0); a SEARCH_LIBRARY row with no
//      card x0.3; a CAST_SPELL / ACTIVATE_ABILITY row whose card is not yet chosen this game x3.0
//      (not an ACTIVATE_ABILITY row under patient); under patient a CAST_SPELL row with a card
//      x(1 + mana value), an ACTIVATE_ABILITY row whose card is in the own hand /(1 + mana value).
//   4. A two-row menu of two MULLIGAN rows gets [3, 1] (the engine's mulligan menu is KEEP_HAND +
//      MULLIGAN, so this never fires; kept for parity).
//   5. A total of 0 or less becomes uniform.
//   6. One weighted draw; a CAST_SPELL / ACTIVATE_ABILITY row's card joins the novelty set.

namespace agents {

class ExploreAgent : public Agent {
    public:
        ExploreAgent(PyRandom *rng, bool patient) : rng_(rng), patient_(patient) {}
        int decide(const Decision &d) override;
        std::string label() const override { return patient_ ? "explore:patient" : "explore"; }

    private:
        PyRandom *rng_;
        bool patient_;
        std::set<std::string> seen_;  // cards chosen this game (the novelty set)
};

}  // namespace agents

#endif /* AGENTS_EXPLORE_H */
