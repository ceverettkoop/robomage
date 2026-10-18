#include "selfcheck_knowledge.h"

#include <cstdio>
#include <map>
#include <vector>

#include "../classes/deck_state.h"
#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/token.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../game_driver.h"
#include "../knowledge/k_rows.h"
#include "../knowledge/knowledge.h"
#include "../knowledge/truth.h"
#include "../queries/visibility.h"
#include "../queries/zones.h"
#include "decode.h"
#include "handles.h"
#include "knowledge_feed.h"
#include "schema.h"
#include "selfcheck.h"

namespace {

struct Stats {
        size_t decisions = 0;
        size_t facts = 0;
        size_t errors = 0;
};

}  // namespace

// ERROR: lines printed before the rest are only counted.
static const size_t MAX_ERROR_LINES = 40;

static Stats stats;
static std::string saved_state;  // the instances' state at the snapshot check's save
static bool have_saved_state = false;

static void report(const char *check, Zone::Ownership viewer, const std::string &detail);
static bool is_card_unit(Entity e);
static bool build_truth(Zone::Ownership viewer, knowledge::Truth *out);
static knowledge::Decklists registered_lists();
static void check_seat(Zone::Ownership viewer, const std::string &frame);

static void report(const char *check, Zone::Ownership viewer, const std::string &detail) {
    ++stats.errors;
    if (stats.errors > MAX_ERROR_LINES) return;
    // While the RMSTATE stream is on stdout is stderr's descriptor with a buffer of its own.
    std::fflush(stdout);
    const char *who = viewer == Zone::PLAYER_A ? "A" : viewer == Zone::PLAYER_B ? "B" : "all";
    non_fatal_error(std::string("selfcheck-knowledge ") + check + ": viewer " + who + ", " +
                    selfcheck_decision_context() + ": " + detail);
    if (stats.errors == MAX_ERROR_LINES) non_fatal_error("selfcheck-knowledge: further failures are only counted");
}

bool selfcheck_knowledge_enabled() {
    return selfcheck_knowledge && selfcheck_state_enabled();
}

// A card the knowledge model holds a unit for: a card object of a player (a token, a spell copy and
// an ability on the stack are none).
static bool is_card_unit(Entity e) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    if (global_coordinator.entity_has_component<Token>(e)) return false;
    if (global_coordinator.entity_has_component<Permanent>(e) && global_coordinator.GetComponent<Permanent>(e).is_token)
        return false;
    return !global_coordinator.entity_has_component<Spell>(e) || !global_coordinator.GetComponent<Spell>(e).is_copy;
}

// The true world from the engine state: every card unit's identity (card_identity, no visibility
// applied) by owner and zone, each library top first, and each face-down object by `viewer`'s
// handle. False when a library's positions are not 0..n-1.
static bool build_truth(Zone::Ownership viewer, knowledge::Truth *out) {
    *out = knowledge::Truth{};
    out->face_down_known = true;
    std::map<size_t, int> library[2];
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        if ((z.owner != Zone::PLAYER_A && z.owner != Zone::PLAYER_B) || !is_card_unit(e)) continue;
        const int p = z.owner == Zone::PLAYER_A ? 0 : 1;
        const int card = card_identity(e, Zone::UNKNOWN);
        ++out->owned[p][card];
        switch (z.location) {
            case Zone::HAND:
                ++out->hand[p][card];
                break;
            case Zone::SIDEBOARD:
                ++out->sideboard[p][card];
                break;
            case Zone::LIBRARY:
                library[p][z.distance_from_top] = card;
                break;
            default:
                break;
        }
        if (z.is_face_down) {
            const uint32_t h = handle_of(viewer, e);
            if (h != 0) out->face_down[h] = card;
        }
    }
    for (int p = 0; p < 2; ++p)
        for (const auto &kv : library[p]) {
            if (kv.first != out->library[p].size()) return false;
            out->library[p].push_back(kv.second);
        }
    return true;
}

// Both players' registered lists with their deck names, as an omniscient DECKLISTS frame has them
// (physical card ids).
static knowledge::Decklists registered_lists() {
    knowledge::Decklists lists;
    for (int p = 0; p < 2; ++p) {
        const Zone::Ownership owner = p == 0 ? Zone::PLAYER_A : Zone::PLAYER_B;
        lists.present[p] = true;
        lists.name[p] = p == 0 ? deck_a_name : deck_b_name;
        for (const DecklistEntry &e : deck_state_registered_main(owner))
            lists.cards.push_back({p, physical_card_of(e.vocab_idx), e.count, 0});
        for (const DecklistEntry &e : deck_state_registered_side(owner))
            lists.cards.push_back({p, physical_card_of(e.vocab_idx), 0, e.count});
    }
    return lists;
}

// One seat's frame: its k_* tables read back and checked against the truth.
static void check_seat(Zone::Ownership viewer, const std::string &frame) {
    const knowledge::Knowledge *know = knowledge_instance(ObsConsumer::SELFCHECK, viewer);
    if (!know || !know->started()) {
        report("soundness", viewer, "the self-check has no knowledge instance for the seat");
        return;
    }
    rmstate::DecodedFrame decoded;
    const rmstate::DecodeResult res = rmstate::decode_frame(frame, know->schema(), &decoded);
    knowledge::KRows rows;
    std::string err;
    if (!res.ok() || !knowledge::read_k_tables(decoded, know->codes(), &rows, &err)) {
        report("soundness", viewer, "the frame's k_* tables do not read back: " + (res.ok() ? err : res.message));
        return;
    }
    knowledge::Truth truth;
    if (!build_truth(viewer, &truth)) {
        report("soundness", viewer, "a library's positions are not 0..n-1");
        return;
    }
    const bool visible[2] = {
        !closed_decklist || viewer == Zone::PLAYER_A, !closed_decklist || viewer == Zone::PLAYER_B};
    knowledge::set_truth_lists(know->metagame(), registered_lists(), visible, &truth);
    const knowledge::Universe *const universe[2] = {&know->player(0).full.universe, &know->player(1).full.universe};
    knowledge::TruthCheck check;
    knowledge::check_truth(rows, truth, viewer == Zone::PLAYER_A ? 0 : 1, universe, &check);
    stats.facts += check.facts;
    for (const std::string &e : check.errors) report("soundness", viewer, e);
}

void selfcheck_knowledge_decision(const std::string &frame_a, const std::string &frame_b) {
    if (!selfcheck_knowledge_enabled()) return;
    ++stats.decisions;
    check_seat(Zone::PLAYER_A, frame_a);
    check_seat(Zone::PLAYER_B, frame_b);
}

void selfcheck_knowledge_saved() {
    if (!selfcheck_knowledge_enabled()) return;
    saved_state = knowledge_state_blob();
    have_saved_state = true;
}

void selfcheck_knowledge_restored() {
    if (!selfcheck_knowledge_enabled() || !have_saved_state) return;
    if (knowledge_state_blob() != saved_state)
        report("snapshot", Zone::UNKNOWN, "a knowledge instance's state differs from the one saved with the snapshot");
    have_saved_state = false;
    saved_state.clear();
}

void selfcheck_knowledge_match_over() {
    if (!selfcheck_knowledge_enabled()) return;
    std::fflush(stdout);
    std::fprintf(stderr, "SELFCHECK-KNOWLEDGE: decisions=%zu facts=%zu errors=%zu\n", stats.decisions, stats.facts,
        stats.errors);
    std::fflush(stderr);
}
