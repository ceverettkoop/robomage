#ifndef STATIC_ABILITY_H
#define STATIC_ABILITY_H
#include <cstdint>
#include <string>

struct StaticAbility {
    std::string category  = "";  // "Continuous", "MustAttack"
    std::string condition = "";  // "Delirium", etc.; empty = always active
    // Continuous mode fields:
    int add_power     = 0;
    int add_toughness = 0;
    std::string add_power_svar = "";      // e.g. "Count$TypeInYourYard.Land" — evaluated at SBE time
    std::string add_toughness_svar = "";
    int last_applied_power = 0;           // tracks dynamic delta last applied (reset when condition lost)
    int last_applied_toughness = 0;
    std::string add_keyword = "";
    std::string affected = "";        // "EquippedBy" = apply buff to equipped creature, not source
    // ETB counter fields (category = "EtbCounter"):
    std::string counter_type = "";    // "P1P1" for +1/+1 counters, "CHARGE" for charge counters
    int counter_count = 0;            // literal count (etbCounter:M1M1:6 → 6); 0 when dynamic
    bool counter_count_from_delve = false;  // counter count = cur_game.delve_exiled.size()
    bool counter_count_from_xpaid = false;  // counter count = X paid at cast (Chalice of the Void: Count$xPaid)
    // RaiseCost fields (category = "RaiseCost"):
    int raise_cost = 0;               // generic mana added to cost of matching spells
    std::string raise_cost_filter = ""; // "nonCreature" = apply to non-creature spells
    // CantBeActivated fields (category = "CantBeActivated"):
    std::string cant_activate_card_filter = "";  // "Artifact" — card type whose activated abilities are suppressed
    // NamedCard fields (RaiseCost / CantBeActivated with ValidCard$ Card.NamedCard, e.g. Disruptor Flute):
    // when true the static applies only to cards whose name equals the source permanent's chosen_name.
    bool match_named_card = false;
    // Icetill Explorer statics (category = "Continuous"):
    int adjust_land_plays = 0;            // AdjustLandPlays$ N — additional land plays per turn
    bool may_play_from_graveyard = false; // MayPlay$ True with AffectedZone$ Graveyard
    // SVar-based condition (Keen-Eyed Curator)
    std::string check_svar_expr = "";   // resolved CheckSVar$ expression (e.g. "Count$ValidExile Card.ExiledWithSource$CardTypes")
    std::string svar_compare = "";      // SVarCompare$ value (e.g. "GE4")

    // CantBeCast fields (category = "CantBeCast"):
    std::string cant_cast_filter = "";      // "Card.nonCreature" — card type filter
    int cant_cast_limit_per_turn = 0;       // NumLimitEachTurn$ N — limit per player per turn
    bool cant_cast_by_opponent = false;     // Caster$ Opponent — restricts the controller's opponents
                                            // (Voice of Victory: "your opponents can't cast spells
                                            // during your turn", gated by Condition$ PlayerTurn).
    // Origin$ Graveyard,Library (Grafdigger's Cage): the restriction applies only to spells cast
    // from these zones (e.g. flashback). Both false = the restriction is zone-agnostic.
    bool cant_cast_from_graveyard = false;
    bool cant_cast_from_library = false;

    // Type-changing fields (category = "Continuous", layer 4):
    std::string add_type = "";              // AddType$ Mountain — land subtype to set
    bool remove_land_types = false;         // RemoveLandTypes$ True — strip existing land subtypes first

    // Ability-removal field (category = "Continuous", layer 6; Humility):
    bool remove_all_abilities = false;      // RemoveAllAbilities$ True — affected objects lose all abilities

    // Untap prevention fields (category = "Continuous" with AddHiddenKeyword):
    std::string hidden_keyword = "";        // "CARDNAME doesn't untap during your untap step."
    std::string affected_subtype = "";      // Affected$ Island — land subtype affected

    // DisableTriggers fields (category = "DisableTriggers"):
    std::string disable_triggers_cause = "";  // ValidCause$ Creature,Artifact
    std::string disable_triggers_mode = "";   // ValidMode$ ChangesZone

    // Characteristic-defining ability (Barrowgoyf): sets base P/T rather than additive
    bool characteristic_defining = false;
    std::string set_power_svar = "";      // SetPower$ X — SVar expression for base power
    std::string set_toughness_svar = "";  // SetToughness$ Y — SVar expression for base toughness

    // State tracking — lives in the Permanent's copy, not in CardData template:
    bool applied = false;             // true when the condition was met on the last SBE pass
    uint32_t last_applied_entity = 0; // for EquippedBy: entity that last received the buff
};

#endif /* STATIC_ABILITY_H */
