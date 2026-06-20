#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "ability_params.h"
#include "zone.h"
#include <memory>
#include <string>
#include <variant>
#include <vector>

class Orderer;

struct Ability{

    enum AbilityType{
        TRIGGERED,
        ACTIVATED,
        SPELL,
    };

    AbilityType ability_type = SPELL;
    std::string category = "";
    std::string valid_tgts = "N_A";  // Value of ValidTgts$ param; "N_A" if no targeting required
    int target_min = 1;              // TargetMin$ 0 = optional targeting (can choose no target)
    int target_max = 1;             // TargetMax$ N — max number of targets (1 = single target)
    Entity source = 0;
    Entity target = 0;
    std::vector<Entity> targets;    // used when target_max > 1
    Zone::Ownership controller = Zone::PLAYER_A;  // set when pushed onto stack; stable even if source loses Permanent
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability
    std::vector<Colors> mana_choices;   // Produced$ Combo or Any — ordered list of selectable mana colors
    bool restrict_to_chosen_type_creature = false;  // RestrictValid$ Spell.Creature+ChosenType
    bool adds_no_counter = false;                    // AddsNoCounter$ True — spell can't be countered

    // Set by apply_land_abilities for mana abilities generated from land subtypes;
    // cleared and regenerated each SBE pass when type-changing effects are active.
    bool subtype_derived = false;

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string sac_cost_spec = "";     // Sac<1/Type;Type/> — type-based sac cost; empty = none
    std::string return_cost_type = "";  // Return<N/Type> — bounce a land of this subtype as cost
    int return_cost_count = 0;          // number of lands to return
    bool discard_hand_cost = false;     // Discard<0/Hand> — discard entire hand as activation cost (Lion's Eye Diamond)
    bool discard_self_cost = false;     // Discard<1/CARDNAME> — discard this card from hand as activation cost
    bool instant_speed = false;         // InstantSpeed$ True — activated ability that is NOT a mana ability; goes on stack
    int activation_limit = 0;           // ActivationLimit$ N — max activations per turn (0 = unlimited)
    int activation_zone = -1;           // ActivationZone$ Hand → Zone::HAND; -1 = default (battlefield)
    int activations_this_turn = 0;      // runtime counter, reset at UNTAP
    std::string change_type = "";        // ChangeType$ — comma-separated subtypes to search
    Zone::ZoneValue origin = Zone::LIBRARY;          // Origin$ — zone to search
    Zone::ZoneValue destination = Zone::BATTLEFIELD; // Destination$ — zone to move card to
    uint32_t trigger_on = 0;             // EventId that fires this ability; 0 = not event-triggered
    bool trigger_self_excluded = false;  // true when ValidCard$ has .Other — won't trigger for the source itself
    bool trigger_only_self = false;      // true when ValidCard$ Card.Self — only fires when the entering entity is the source itself
    bool trigger_valid_player_is_controller = false;  // true when ValidPlayer$ You
    bool mandatory = false;              // Mandatory$ True — player must choose; suppresses fail-to-find when zone non-empty
    bool may_shuffle = false;            // MayShuffle$ True — player may optionally shuffle after
    size_t unless_generic_cost = 0;      // UnlessCost$ N — target controller pays {N} to prevent counter
    std::string target_type = "";        // TargetType$ Spell — restricts targeting to stack spells

    // Delirium-conditional damage (Unholy Heat) now lives in DamageParams (params variant).
    std::string amount_svar = "";           // raw SVar key for non-numeric NumDmg$ (resolved at parse time)
    std::string dynamic_amount_expr = "";   // runtime SVar expression (e.g. "Count$Valid Creature.YouCtrl" or "Targeted$CardPower")
    bool defined_targeted_controller = false;  // Defined$ TargetedController — GainLife goes to target's controller
    bool defined_self = false;                  // Defined$ Self — ability moves its own source

    // DestroyAll filter (e.g. "Artifact.cmcLEX")
    std::string destroy_all_filter = "";

    // Effect-specific parameter blocks. As effects migrate off the flat
    // god-struct fields (Phase 3), their exclusive data moves into one of these
    // variant alternatives; shared fields stay as direct members. std::monostate
    // covers effects with no exclusive fields. See ability_params.h.
    std::variant<std::monostate, PumpParams, DamageParams> params;

    // Counter abilities (PutCounter category)
    std::string counter_type = "";          // "P1P1" for +1/+1 counters
    int counter_count = 0;                  // static number of counters; 0 when dynamic
    bool counter_count_from_delve = false;  // if true, counter_count = cur_game.delve_exiled.size() at resolve

    // Peek variant (Mishra's Bauble): look at target player's top card, skip reveal choice
    bool is_peek_no_reveal = false;

    // Delayed trigger (Mishra's Bauble)
    bool delayed_trigger_next_turn = false;  // NextTurn$ True

    // Zone-change trigger filters for CARD_CHANGED_ZONE (set by Mode$ ChangesZone triggers)
    int trigger_zone_origin = -1;       // Zone::ZoneValue origin filter; -1 = any
    int trigger_zone_destination = -1;  // Zone::ZoneValue destination filter; -1 = any
    bool trigger_valid_card_is_creature = false;        // ValidCard$ Creature
    bool trigger_valid_card_is_instant_or_sorcery = false;  // ValidCard$ Instant/Sorcery
    bool trigger_valid_card_is_land = false;            // ValidCard$ Land.*

    // Combat damage trigger (Barrowgoyf): damage amount stored at trigger fire time
    size_t trigger_damage_amount = 0;

    // Spell count trigger (Cori-Steel Cutter)
    size_t trigger_spell_count_eq = 0;  // ActivatorThisTurnCast$ EQN — fires on Nth spell

    // Token creation (Cori-Steel Cutter)
    std::string token_script = "";  // TokenScript$ w_1_1_monk_prowess

    // Attach / Equip sub-ability
    bool optional = false;           // Optional$ True — player may decline
    bool defined_remembered = false; // Defined$ Remembered — target is cur_game.remembered_entities[0]

    // Mill: remember milled cards in cur_game.remembered_entities
    bool remember_milled = false;    // RememberMilled$ True
    bool amount_from_damage = false; // NumCards$ DamageAmount — use trigger_damage_amount

    // Cleanup sub-ability
    bool clear_remembered = false;   // ClearRemembered$ True

    // RememberChanged$ — remember entities moved by this ChangeZone (for Doomsday)
    bool remember_changed = false;

    // Tapped$ True — searched card enters the battlefield tapped (Edge of Autumn)
    bool enters_tapped = false;

    // Multi-zone origin support (e.g. Origin$ Graveyard,Library)
    std::vector<Zone::ZoneValue> origins;  // populated when Origin$ has commas; origin holds first value

    // Dig ability (Once Upon a Time, Thassa's Oracle)
    size_t dig_num = 0;              // DigNum$ N — how many cards to look at from top of library
    std::string dig_num_expr = "";   // DigNum$ SVar — dynamic dig count (e.g. "Count$Devotion.Blue")
    std::string change_valid = "";   // ChangeValid$ — comma-separated filter like "Card.Creature,Card.Land"
    bool rest_random_order = false;  // RestRandomOrder$ True
    bool optional_choice = false;    // Optional$ True in Dig context — can choose nothing
    int dig_destination = -1;        // DestinationZone$ — where chosen card goes (-1 = HAND, Zone::LIBRARY etc.)
    int dig_library_position = -1;   // LibraryPosition$ — 0 = top, -1 = unset

    // Discard ability (Thoughtseize, Duress)
    std::string discard_valid = "";    // DiscardValid$ — filter for cards to discard (e.g. "Card.nonLand")
    std::string mode = "";             // Mode$ — e.g. "RevealYouChoose"

    // Conditional subability execution (Scythecat Cub, Thassa's Oracle)
    std::string condition_check_svar = "";   // ConditionCheckSVar$ — resolved expression e.g. "Count$ResolvedThisTurn"
    std::string condition_svar_compare = ""; // ConditionSVarCompare$ — e.g. "EQ2", "NE2", "GE1", or "LEX" with SVar RHS
    std::string condition_compare_svar_expr = "";  // when compare RHS is an SVar (e.g. LEX → "Count$Devotion.Blue")

    // Castability condition (Edge of Autumn): count permanents matching filter, compare to threshold
    std::string condition_present = "";   // ConditionPresent$ — e.g. "Land.YouCtrl"
    std::string condition_compare = "";   // ConditionCompare$ — e.g. "LE4", "GE3"

    // Delayed trigger params (Mishra's Bauble)
    std::string delayed_phase = "";         // Phase$ — phase name (e.g. "Upkeep", "Draw", "EndStep")
    std::string delayed_execute_svar = "";  // Execute$ — SVar name of ability to run when trigger fires
    std::string delayed_valid_player = "";  // ValidPlayer$ — "Player", "You", "Opponent"

    //for each AB on a card script there may be multiple SubAbility$, would get parsed into vector below
    std::vector<Ability> subabilities; // additional abilities resolved at same time this resolves, stored in order

    // Charm/modal spell choices — each entry is a fully-parsed sub-ability
    std::vector<Ability> charm_choices;
    std::vector<std::string> charm_choice_descriptions;  // SpellDescription$ for each choice
    int charm_num = 1;  // CharmNum$ — how many modes to pick (default 1)

    void resolve(std::shared_ptr<Orderer> orderer);
    bool identical_activated_ability(const Ability& other);
    // Single source of truth for target legality. Returns true if `cand` is a legal
    // target for this ability when controlled by `caster`. Used both to enumerate
    // legal targets (build_valid_targets) and to re-verify chosen targets at
    // resolution (is_target_valid).
    bool is_legal_target(Entity cand, Zone::Ownership caster) const;
private:
    // Per-effect resolution now lives in src/effects/effect_*.cpp, dispatched by
    // effects::handler_for(). resolve() keeps only target validity + condition
    // gating + subability chaining.
    bool is_target_valid() const;
    void fizzle(std::shared_ptr<Orderer> orderer);

};

// Returns the effect-param block of type P held in `ab.params`, default-
// constructing (and switching the variant to P) if it isn't already active.
// Use from parse hooks before writing effect-exclusive params. Resolution-time
// readers should use std::get_if<P>(&ab.params) and treat nullptr as "defaults",
// which is exception-free under -fno-exceptions.
template <typename P>
P& effect_params(Ability& ab) {
    if (!std::holds_alternative<P>(ab.params)) ab.params = P{};
    return std::get<P>(ab.params);
}

// Search a zone for cards matching the comma-separated type list in change_type
// (empty change_type matches all cards in the zone).
// When mandatory=true, "fail to find" is suppressed unless the zone is empty.
// Returns the chosen Entity, or 0 if the player fails to find / zone is empty.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
                   Zone::ZoneValue zone, const std::string& change_type,
                   bool mandatory = false,
                   Zone::ZoneValue destination = Zone::GRAVEYARD);

#endif /* ABILITY_H */
