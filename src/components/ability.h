#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "zone.h"
#include <memory>
#include <string>
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
    // TODO: support multiple targets (e.g. "deal 1 damage to each of up to two targets")
    std::string valid_tgts = "N_A";  // Value of ValidTgts$ param; "N_A" if no targeting required
    int target_min = 1;              // TargetMin$ 0 = optional targeting (can choose no target)
    Entity source = 0;
    Entity target = 0;
    Zone::Ownership controller = Zone::PLAYER_A;  // set when pushed onto stack; stable even if source loses Permanent
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability
    std::vector<Colors> mana_choices;   // Produced$ Combo or Any — ordered list of selectable mana colors

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string sac_cost_spec = "";     // Sac<1/Type;Type/> — type-based sac cost; empty = none
    std::string return_cost_type = "";  // Return<N/Type> — bounce a land of this subtype as cost
    int return_cost_count = 0;          // number of lands to return
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

    // Delirium-conditional damage (Unholy Heat)
    bool amount_is_delirium_scale = false;  // if true, use amount_delirium when delirium active
    size_t amount_delirium = 0;             // damage when delirium is active
    std::string amount_svar = "";           // raw SVar key for non-numeric NumDmg$ (resolved at parse time)
    std::string dynamic_amount_expr = "";   // runtime SVar expression (e.g. "Count$Valid Creature.YouCtrl" or "Targeted$CardPower")
    bool defined_targeted_controller = false;  // Defined$ TargetedController — GainLife goes to target's controller
    bool defined_self = false;                  // Defined$ Self — ability moves its own source

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

    // Spell count trigger (Cori-Steel Cutter)
    size_t trigger_spell_count_eq = 0;  // ActivatorThisTurnCast$ EQN — fires on Nth spell

    // Token creation (Cori-Steel Cutter)
    std::string token_script = "";  // TokenScript$ w_1_1_monk_prowess

    // Attach / Equip sub-ability
    bool optional = false;           // Optional$ True — player may decline
    bool defined_remembered = false; // Defined$ Remembered — target is cur_game.remembered_entity

    // Cleanup sub-ability
    bool clear_remembered = false;   // ClearRemembered$ True

    // Dig ability (Once Upon a Time)
    size_t dig_num = 0;              // DigNum$ N — how many cards to look at from top of library
    std::string change_valid = "";   // ChangeValid$ — comma-separated filter like "Card.Creature,Card.Land"
    bool rest_random_order = false;  // RestRandomOrder$ True
    bool optional_choice = false;    // Optional$ True in Dig context — can choose nothing

    // Discard ability (Thoughtseize, Duress)
    std::string discard_valid = "";    // DiscardValid$ — filter for cards to discard (e.g. "Card.nonLand")
    std::string mode = "";             // Mode$ — e.g. "RevealYouChoose"

    // Conditional subability execution (Scythecat Cub)
    std::string condition_check_svar = "";   // ConditionCheckSVar$ — resolved expression e.g. "Count$ResolvedThisTurn"
    std::string condition_svar_compare = ""; // ConditionSVarCompare$ — e.g. "EQ2", "NE2", "GE1"

    //for each AB on a card script there may be multiple SubAbility$, would get parsed into vector below
    std::vector<Ability> subabilities; // additional abilities resolved at same time this resolves, stored in order

    void resolve(std::shared_ptr<Orderer> orderer);
    bool identical_activated_ability(const Ability& other);
private:
    void resolve_change_zone(std::shared_ptr<Orderer> orderer);
    void resolve_destroy(std::shared_ptr<Orderer> orderer);
    void resolve_rearrange_top_of_library(std::shared_ptr<Orderer> orderer);
    void resolve_surveil(std::shared_ptr<Orderer> orderer);
    void resolve_put_counter();
    void resolve_token(std::shared_ptr<Orderer> orderer);
    void resolve_delayed_trigger();
    bool is_target_valid() const;
    void fizzle(std::shared_ptr<Orderer> orderer);

};

// Search a zone for cards matching the comma-separated type list in change_type
// (empty change_type matches all cards in the zone).
// When mandatory=true, "fail to find" is suppressed unless the zone is empty.
// Returns the chosen Entity, or 0 if the player fails to find / zone is empty.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
                   Zone::ZoneValue zone, const std::string& change_type,
                   bool mandatory = false,
                   Zone::ZoneValue destination = Zone::GRAVEYARD);

#endif /* ABILITY_H */
