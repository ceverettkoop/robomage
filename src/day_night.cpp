#include "day_night.h"

#include <string>
#include <vector>

#include "cli_output.h"
#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "ecs/coordinator.h"
#include "game_queries.h"
#include "transform.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static bool keyword_present(const std::vector<std::string> &kws, const char *target);
static bool card_is_nightbound_dfc(const CardData &front);
static void apply_day_night_transforms();
static void set_day_night(Game::DayNight value);

// True if `target` appears in the keyword list (exact match).
static bool keyword_present(const std::vector<std::string> &kws, const char *target) {
    for (const auto &k : kws)
        if (k == target) return true;
    return false;
}

bool card_has_daybound(const CardData &front) {
    return keyword_present(front.keywords, "Daybound");
}

// True if the card's back face carries Nightbound (CR 702.145e — nightbound is on the back face of
// the same transforming DFC). A daybound/nightbound DFC always pairs the two (702.145a).
static bool card_is_nightbound_dfc(const CardData &front) {
    return front.backside && keyword_present(front.backside->keywords, "Nightbound");
}

// Perform the transforms the current day/night designation requires (CR 702.145c/f). Run
// immediately whenever the designation changes (these are not state-based actions, CR 702.145c).
static void apply_day_night_transforms() {
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!is_battlefield_permanent(e)) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        const CardData &cd = global_coordinator.GetComponent<CardData>(e);
        const bool back_face_up = global_coordinator.GetComponent<Permanent>(e).transformed;
        if (cur_game.day_night == Game::DN_NIGHT) {
            // 702.145c: a front-face-up daybound permanent transforms to its night (back) face.
            if (card_has_daybound(cd) && !back_face_up) transform_permanent(e);
        } else if (cur_game.day_night == Game::DN_DAY) {
            // 702.145f: a back-face-up nightbound permanent transforms to its day (front) face.
            if (card_is_nightbound_dfc(cd) && back_face_up) transform_permanent(e);
        }
    }
}

static void set_day_night(Game::DayNight value) {
    if (cur_game.day_night == value) return;  // already that designation (CR 731.1)
    cur_game.day_night = value;
    game_log(value == Game::DN_DAY ? "It becomes day.\n" : "It becomes night.\n");
    apply_day_night_transforms();
}

void become_day() { set_day_night(Game::DN_DAY); }
void become_night() { set_day_night(Game::DN_NIGHT); }

void day_night_untap_transition() {
    // CR 502.2 / 731.2: checked as the second part of the untap step, on the turn that just ended.
    // prev_turn_active_spell_count is the previous turn's active player's spell count during that
    // turn (captured at that turn's cleanup, before the per-turn reset).
    if (cur_game.day_night == Game::DN_DAY) {
        if (cur_game.prev_turn_active_spell_count == 0) become_night();      // 731.2a
    } else if (cur_game.day_night == Game::DN_NIGHT) {
        if (cur_game.prev_turn_active_spell_count >= 2) become_day();        // 731.2b
    }
    // DN_NEITHER: 731.2c — the check doesn't happen and it remains neither.
}
