#ifndef PERMANENT_H
#define PERMANENT_H

#include "../ecs/entity.h"
#include "zone.h"
#include "ability.h"
#include "static_ability.h"
#include "types.h"
#include <map>
#include <set>
#include <string>
#include <vector>

struct Permanent {
    std::string name;
    std::set<Type> types;
    bool is_token = false;
    bool is_tapped = false;
    bool has_summoning_sickness = true;
    std::vector<Ability> abilities;
    std::vector<StaticAbility> static_abilities;
    Zone::Ownership controller = Zone::UNKNOWN;
    size_t timestamp_entered_battlefield = 0;  // For ordering simultaneous ETBs
    size_t entered_on_turn = 0;  // cur_game.turn when this entered the battlefield (Ocelot Pride "entered this turn")
    bool transformed = false;  // true when DFC is showing its back face
    Entity equipped_to = 0;   // for equipment: which creature entity is equipped (0 = unattached)
    Entity equipped_by = 0;   // for creatures: which equipment is attached (0 = none)
    bool is_phased_out = false;
    // 122.1: typed counters on this permanent, keyed by counter type ("P1P1", "M1M1",
    // "LOYALTY", keyword counters). Single store for every counter kind (T2.4) — planeswalker
    // loyalty is just a LOYALTY counter (306.5c). Mutate via the get/add_counters helpers in
    // game_queries.h so +1/+1 and -1/-1 changes stay in sync with the creature's cached P/T
    // contribution (layer 7c, 613.4c). An entry is absent (not 0) when it has no counters.
    std::map<std::string, int> counters;
    // 606.3: a loyalty ability can be activated only if no loyalty ability of THIS
    // permanent has been activated this turn. The gate is per-permanent (across all its
    // loyalty abilities), not per-ability, so it can't reuse Ability::activations_this_turn.
    bool loyalty_ability_activated_this_turn = false;
    // 701.37b: monstrous is a per-permanent designation with no rules meaning of its own — a
    // marker the monstrosity action and "becomes monstrous" triggers identify. Once set it
    // stays until the permanent leaves the battlefield (so it resets naturally when a new
    // Permanent is created on re-entry). Internal state only — NOT exposed in the obs/state
    // vector. Set by the Monstrosity$ resolution in effect_put_counter.cpp; read by the
    // "NotMonstrous" activation gate (game_queries.h). Mutated nowhere else.
    bool is_monstrous = false;
    bool evoked = false;  // entered via its evoke alternate cost — fires the evoke sacrifice ETB trigger
    // This permanent entered because its spell was cast from the graveyard for its Escape cost
    // (CR 702.139): the permanent "escaped". Read by the Card.Self+escaped present condition
    // (Uro's "sacrifice it unless it escaped" ETB). Set when the Permanent is created from a
    // spell cast with escape; false for a normal hand cast or any non-escape entry. General —
    // reusable by any escape card with an "if it escaped" clause.
    bool cast_with_escape = false;
    bool entered_with_offspring = false;  // cast with its Offspring additional cost paid — fires the offspring token-copy ETB trigger (CR 702.171)
    // This permanent entered the battlefield as a spell its controller cast from their own
    // hand (CR 601 — a normal hand cast). Read by the Card.wasCastFromYourHandByYou condition
    // (Amped Raptor's dig-from-hand gate). Set when its Permanent is created from a spell that
    // was cast from the hand; false for permanents put onto the battlefield any other way
    // (reanimation, tokens, ChangeZone-to-battlefield, casts from exile/graveyard, etc.).
    bool cast_from_hand_by_controller = false;
    std::string chosen_type = "";  // creature type chosen on ETB (Cavern of Souls)
    std::string chosen_name = "";  // card name chosen on ETB (Disruptor Flute) — keys Card.NamedCard statics
    std::vector<Entity> exiled_with;  // entities exiled by this permanent (for Keen-Eyed Curator)

    // DB$ Animate | Duration$ Permanent (CR 613, the "becomes ..." continuous effects a
    // resolved ability bakes onto a permanent for the rest of the game). Stored on the
    // permanent itself (not as a battlefield-source static) because the source may leave;
    // the layer pass reapplies them each SBE so they survive the per-pass rebuild:
    //   * animate_added_types    — subtypes/types added in layer 4 (Guide of Souls: "Angel")
    //   * animate_added_keywords — keywords granted in layer 6 (e.g. Haste, when a later card needs it)
    // Extension points for the land-animation case (a later Earthbend card): set base P/T and
    // add a Creature component to a noncreature permanent. animate_set_pt drives layer-7b base
    // P/T; animate_make_creature requests the Creature-component bootstrap.
    std::vector<Type> animate_added_types;
    std::vector<std::string> animate_added_keywords;
    bool animate_set_pt = false;   // (extension) an Animate set this permanent's base P/T
    int animate_power = 0;         // (extension) base power Animate sets
    int animate_toughness = 0;     // (extension) base toughness Animate sets
    bool animate_make_creature = false;  // (extension) Animate turns a noncreature into a creature

    // DB$/AB$ Animate with the default "until end of turn" Duration (CR 613 continuous effect that
    // ends during the cleanup step, CR 514.2). Same role as the animate_* fields above but the
    // grant LAPSES at cleanup instead of persisting for the rest of the game. Reapplied each
    // static pass (so it survives the per-pass rebuild for the turn it is live) and then cleared
    // by the CLEANUP step in classes/game.cpp:
    //   * animate_added_types_eot   — types/subtypes added in layer 4 only for the rest of THIS
    //                                 turn. Only types NOT already on the permanent are recorded
    //                                 here, so the cleanup revert erases exactly what was granted
    //                                 (never a printed type — e.g. The Fantasticar keeps Artifact).
    //   * animate_make_creature_eot — this EOT Animate turned a noncreature permanent (a Vehicle)
    //                                 into a creature; cleanup strips the bootstrapped Creature/
    //                                 Damage components unless the permanent is a creature by some
    //                                 permanent means (printed creature face, or animate_make_creature).
    // A card opts in by giving its Animate ability no Duration$ (or Duration$ other than Permanent).
    std::vector<Type> animate_added_types_eot;
    bool animate_make_creature_eot = false;

    // AB$ AnimateAll | RemoveKeywords$ ... (Shadowspear: "Permanents your opponents control lose
    // hexproof and indestructible until end of turn"). Keyword(s) this permanent currently has
    // SUPPRESSED until end of turn by a mass continuous effect (CR 613, layer 6 keyword removal).
    // The effective-keyword accessors (permanent_has_keyword / is_indestructible in game_queries.h)
    // treat a keyword in this set as absent, so the loss has a real gameplay consequence — a
    // hexproof creature becomes targetable by opponents again, an indestructible one can be
    // destroyed / die to lethal damage — for the rest of the turn. Cleared at the cleanup step
    // (514.2). General: works for creatures (whose cr.keywords also drop it in gather_active_statics)
    // and for noncreature permanents (whose keywords the accessors read off CardData/Token).
    std::set<std::string> removed_keywords_eot;
};

#endif /* PERMANENT_H */
