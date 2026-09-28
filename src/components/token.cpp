#include "token.h"
#include "permanent.h"
#include "creature.h"
#include "damage.h"
#include "../classes/game.h"
#include "../ecs/coordinator.h"
#include "../cli_output.h"
#include "../queries/counters.h"
#include "../systems/replacement_effects.h"

extern Game cur_game;

void bootstrap_token_components(Entity tok_entity, const Token &tok,
                                Zone::Ownership controller, size_t &timestamp) {
    int etb_counters = 0;  // "enters with" counters (614.1c), added once the components exist
    std::string etb_counter_type = "P1P1";
    if (!global_coordinator.entity_has_component<Permanent>(tok_entity)) {
        Permanent perm;
        perm.name = tok.name;
        perm.types = tok.types;
        perm.is_token = true;
        perm.controller = controller;
        perm.has_summoning_sickness = true;
        // The replacement effects that shape how a permanent enters apply to a token as to a
        // card (CR 614.1c-d, 614.12): a token copy of a card has its self-replacements
        // (CR 707.2) — "enters tapped", "enters with counters".
        ReplacementEvent rev;
        rev.type = ReplacementEvent::ENTERS_BATTLEFIELD;
        rev.entity = tok_entity;
        rev.affected_player = controller;  // 616.1: the permanent's controller chooses
        rev.ask_inline = true;
        replacement::dispatch(rev);
        perm.is_tapped = rev.enters_tapped;
        if (perm.is_tapped) game_log("%s enters tapped.\n", tok.name.c_str());
        etb_counters = rev.etb_p1p1;
        etb_counter_type = rev.etb_counter_type;
        perm.timestamp_entered_battlefield = timestamp++;
        perm.entered_on_turn = cur_game.turn_state.turn;
        // Carry the token's intrinsic activated abilities onto the permanent so they are
        // offered as legal actions (a card's activated abilities are read from
        // Permanent::abilities, not the source component). The Eldrazi Spawn token's
        // "Sacrifice this creature: Add {C}." mana ability reaches the player this way.
        // Triggered abilities stay on Token::abilities (the trigger scan reads them there).
        for (const auto &ab : tok.abilities) {
            if (ab.ability_type != AbilityDef::ACTIVATED && ab.ability_type != AbilityDef::SPELL)
                continue;
            Ability copy(ab);
            copy.source = ObjectRef::of(tok_entity);
            perm.abilities.push_back(copy);
        }
        // Carry the token's continuous static abilities onto the permanent so the SBE static
        // pass (gather_active_statics) applies them — e.g. the Urza's Saga Construct token's
        // "+1/+1 for each artifact you control" self-buff (Affected$ Card.Self, AddPower$ X).
        perm.static_abilities = tok.static_abilities;
        global_coordinator.AddComponent(tok_entity, perm);
    }
    // Only creature tokens get Creature/Damage components and P/T. A noncreature token
    // (Powerstone, Treasure, Clue, Food, ...) has no power/toughness and must NOT acquire a
    // Creature component, or the zero-toughness state-based action (CR 704.5f) would destroy it
    // the instant it enters.
    bool is_creature_token = false;
    for (const auto &t : tok.types)
        if (t.name == "Creature") { is_creature_token = true; break; }
    if (is_creature_token && !global_coordinator.entity_has_component<Creature>(tok_entity)) {
        Creature creature;
        creature.base_power = static_cast<int>(tok.power);
        creature.base_toughness = static_cast<int>(tok.toughness);
        creature.keywords = tok.keywords;
        recompute_pt(creature);
        global_coordinator.AddComponent(tok_entity, creature);

        Damage damage;
        damage.damage_counters = 0;
        global_coordinator.AddComponent(tok_entity, damage);
    }
    if (etb_counters > 0) {
        add_counters(tok_entity, etb_counter_type, etb_counters);
        game_log("%s enters with %d %s counter(s).\n", tok.name.c_str(), etb_counters,
                 etb_counter_type.c_str());
    }
}
