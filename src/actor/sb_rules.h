#ifndef ACTOR_SB_RULES_H
#define ACTOR_SB_RULES_H

// Dead-sideboard-card pruning — the C++ twin of train/mcts.py::sb_dead_mask.
// The two MUST stay in exact lockstep: the actor parity gate (ci_check --tier
// actor / train/test_mcts_parity.py) asserts bit-identical sideboard plan
// payloads, and a rule-evaluation divergence shifts which first picks are
// covered at every sb root. Any change here must land in mcts.py identically
// (and vice versa). The rule DATA is generated (src/gen/sb_rules_gen.h, from
// the hand-edited train/sb_dead_rules.json) so only the evaluator below is
// hand-mirrored.
//
// Semantics: dead[a] is true iff action a is a SIDEBOARD_IN of a ruled card
// and NO card of the opponent's REGISTERED 75 (the frozen OPP_DECK_MAIN/SIDE
// obs blocks) carries any of the rule's fact bits. Only SIDEBOARD_IN actions
// are ever masked — Done and OUT picks always stay available, so every
// sideboard menu keeps at least one live action. Pure obs -> bool, no engine
// reads, same lround-decode idiom as menu_merge.h.

#include <cmath>
#include <cstdint>
#include <vector>

#include "actor/obs_builder.h"   // ACT_CATS_START, ACT_IDS_START
#include "classes/action.h"      // ActionCategory, ACTION_CATEGORY_MAX
#include "gen/sb_rules_gen.h"    // SB_RULE_CARD/SB_RULE_MASK/SB_CARD_FACTS
#include "machine_io.h"          // N_CARD_TYPES, OPP_DECK_MAIN_START/SIDE_END

// is_sideboard_phase float within the match-context block (machine_io.h:
// game_number, self_wins, opp_wins, is_sideboard_phase) — env._IS_SIDEBOARD_IDX.
static constexpr int SB_IS_SIDEBOARD_IDX = MATCH_CTX_START + 3;

// Condition mask for a ruled card id, or 0 when the card has no rule.
// SB_RULE_CARD is sorted ascending (gen_sb_rules.py) — binary search.
inline uint32_t sb_rule_mask_for(int card_id) {
    int lo = 0, hi = N_SB_DEAD_RULES - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (SB_RULE_CARD[mid] == card_id) return SB_RULE_MASK[mid];
        if (SB_RULE_CARD[mid] < card_id) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0;
}

// Fill `out` with the dead-IN mask for the obs's legal menu. Mirror any
// change into mcts.py::sb_dead_mask.
inline void sb_dead_mask(const float* o, int nc, std::vector<bool>& out) {
    out.assign(static_cast<size_t>(nc), false);
    if (!(o[SB_IS_SIDEBOARD_IDX] > 0.5f)) return;
    // Union of fact bits over the opponent's registered main + side blocks.
    // Slots are (card_id, count) pairs; the empty-slot sentinel decodes to -1.
    uint32_t opp_facts = 0;
    for (int j = OPP_DECK_MAIN_START; j < OPP_DECK_SIDE_END; j += 2) {
        const int cid =
            static_cast<int>(std::lround(double(o[j]) * N_CARD_TYPES));
        if (cid >= 0) opp_facts |= SB_CARD_FACTS[cid];
    }
    for (int a = 0; a < nc; a++) {
        const int cat = static_cast<int>(
            std::lround(double(o[ACT_CATS_START + a]) * ACTION_CATEGORY_MAX));
        if (cat != static_cast<int>(ActionCategory::SIDEBOARD_IN)) continue;
        const int cid = static_cast<int>(
            std::lround(double(o[ACT_IDS_START + a]) * N_CARD_TYPES));
        const uint32_t rule = sb_rule_mask_for(cid);
        if (rule != 0 && (opp_facts & rule) == 0)
            out[static_cast<size_t>(a)] = true;
    }
}

#endif /* ACTOR_SB_RULES_H */
