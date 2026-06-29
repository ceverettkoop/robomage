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
    // AddAbility$ <SVar> (Petrified Hamlet): a continuous static that GRANTS a full
    // activated ability (the SVar's resolved AB$ body) to every permanent the Affected$
    // filter designates — layer 6 (CR 613.1f). add_ability holds the resolved ability
    // line (e.g. "AB$ Mana | Cost$ T | Produced$ C"); the layer-6 grant pass parses it to
    // an Ability and attaches it to each affected permanent's ability list, de-duped per
    // source static so SBA repeats don't stack copies. Empty = no granted ability.
    std::string add_ability = "";
    std::string affected = "";        // "EquippedBy" = apply buff to equipped creature, not source
    // ETB counter fields (category = "EtbCounter"):
    std::string counter_type = "";    // "P1P1" for +1/+1 counters, "CHARGE" for charge counters
    int counter_count = 0;            // literal count (etbCounter:M1M1:6 → 6); 0 when dynamic
    bool counter_count_from_delve = false;  // counter count = cur_game.delve_exiled.size()
    bool counter_count_from_xpaid = false;  // counter count = X paid at cast (Chalice of the Void: Count$xPaid)
    // RaiseCost fields (category = "RaiseCost"):
    int raise_cost = 0;               // generic mana added to cost of matching spells
    std::string raise_cost_filter = ""; // "nonCreature" = apply to non-creature spells
    // Damping Sphere: "Each spell a player casts costs {1} more for each other spell that player
    // has cast this turn." A RELATIVE surcharge (Amount$ X | Relative$ True, X =
    // Count$ThisTurnCast_Card.YouCtrl) — the added generic is the spell's caster's count of
    // spells already cast this turn (the "other" spells), so it is computed per-cast against the
    // caster rather than as a fixed raise_cost.
    bool raise_cost_per_spell_cast = false;
    // ReduceCost fields (category = "ReduceCost"): the mirror of RaiseCost — a continuous
    // static that reduces the generic portion of matching spells' cost by reduce_cost
    // (CR 118.7 / 601.2f). reduce_cost_filter is the ValidCard$ spec (e.g. "Eldrazi.Colorless").
    // reduce_cost_you_only gates the reduction to the source's controller (Activator$ You).
    int reduce_cost = 0;
    std::string reduce_cost_filter = "";
    bool reduce_cost_you_only = false;
    // SetCost / cost-floor fields (category = "SetCost"; Trinisphere): a continuous static that
    // sets a MINIMUM total mana value on matching spells (RaiseTo$ True). When a spell's
    // effective total mana value — after every other increase/reduction is applied (CR 601.2f) —
    // is below set_cost_min, generic pips are added to bring the total up to set_cost_min.
    // set_cost_filter is the ValidCard$ spec the floor applies to ("Card"/empty = every spell).
    // Gated like any static through the IsPresent$ present-count condition (Trinisphere is active
    // only while its source is untapped). set_cost_raise_to distinguishes the floor form (the only
    // one implemented) from a hypothetical "set the cost to exactly N" form.
    int set_cost_min = 0;
    bool set_cost_raise_to = false;
    std::string set_cost_filter = "";
    // CantAttack fields (category = "CantAttack"; Ensnaring Bridge): a continuous combat
    // restriction — creatures matching cant_attack_filter can't be declared as attackers
    // (CR 509.1a). cant_attack_filter is the ValidCard$ spec (e.g. "Creature.powerGTX"); it is
    // populated only for the blanket "can't attack" form (no Target$ player restriction), so a
    // Target$-scoped "can't attack you" variant leaves it empty and is skipped. cant_attack_x_svar
    // holds the resolved SVar expression for a dynamic X referenced by a powerGTX/-style qualifier
    // (Ensnaring Bridge: "Count$ValidHand Card.YouOwn"), evaluated against the static SOURCE's
    // controller at declare-attackers time ("your hand" = the source controller's hand, applied to
    // every creature). Empty x_svar = the filter has no dynamic X.
    std::string cant_attack_filter = "";
    std::string cant_attack_x_svar = "";
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

    // Present-count condition (CR 604.3 / 611 continuous static gated on a zone count) —
    // the general IsPresent$/PresentZone$/PresentCompare$ gate (Elvish Reclaimer's +2/+2 when
    // there are 3+ land cards in your graveyard). When present_filter is non-empty the static is
    // active only while the count of cards/permanents matching present_filter in present_zone
    // satisfies present_compare. Re-evaluated every SBA pass so the static turns on/off as the
    // zone count changes. present_zone defaults to "Battlefield" (Forge convention) when the
    // script omits PresentZone$. present_compare defaults to "GE1" when omitted.
    std::string present_filter  = "";   // IsPresent$ filter (e.g. "Land.YouOwn")
    std::string present_zone    = "";   // PresentZone$ (Battlefield/Graveyard/Hand/Exile/Library)
    std::string present_compare = "";   // PresentCompare$ (e.g. "GE3", "EQ0", "LT2")

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

    // Color-changing fields (category = "Continuous", layer 5; CR 613.1e / 612). SetColor$ names
    // the color(s) the affected objects become — "Colorless" (CR 105.2c) or a White/Blue/Black/
    // Red/Green list. affected_zone is AffectedZone$ ("All" = every zone; otherwise a specific
    // zone). General: Mycosynth Lattice ("All cards ... are colorless", Affected$ Card |
    // SetColor$ Colorless | AffectedZone$ All) is the global-colorless case; an explicit color
    // list reuses the same field. Consumed by setcolor_override_for (state_manager_statics.cpp).
    std::string set_color = "";             // SetColor$ Colorless | White | "White Blue" | ...
    std::string affected_zone = "";         // AffectedZone$ All | Battlefield | Graveyard | ...

    // Mana-spending conversion (category = "ManaConvert"; CR 609.4 / 106.6). ManaConversion$
    // AnyType->AnyColor (Mycosynth Lattice's "Players may spend mana as though it were mana of
    // any color.") lets any mana pay any colored pip. Consumed by any_mana_as_any_color_active.
    std::string mana_conversion = "";       // ManaConversion$ AnyType->AnyColor

    // Self card-type animate static (Kaito, Bane of Nightmares' "During your turn ... he's a 3/4
    // Ninja creature with hexproof that's still a planeswalker"). When Affected$ references the
    // source itself (Permanent.Self) and add_type holds CARD types (e.g. "Creature & Ninja"), this
    // is a conditionally-active layer-4 effect that ADDS those types to the source while the
    // static's condition holds (CR 613 layer 4). If "Creature" is among them the source becomes a
    // creature: a Creature/Damage component is bootstrapped, its base P/T set by the 7b SetPower/
    // SetToughness setter, and any AddKeyword$ granted in layer 6 — all gated on condition_met.
    // remove_card_types mirrors RemoveCardTypes$ True: the original printed CARD types are dropped
    // EXCEPT Planeswalker (the Oracle's "that's still a planeswalker" — a planeswalker becoming a
    // creature keeps being a planeswalker, CR 306). self_counter_compare/self_counter_type carry the
    // per-source counter gate parsed off the Affected$ filter (counters_GE1_LOYALTY → "GE1"/"LOYALTY")
    // and AND into condition_met so the static is active only while the source has that many counters.
    bool remove_card_types = false;         // RemoveCardTypes$ True
    std::string self_counter_compare = "";  // e.g. "GE1" from Affected$ ...+counters_GE1_LOYALTY
    std::string self_counter_type    = "";  // e.g. "LOYALTY"

    // Ability-removal field (category = "Continuous", layer 6; Humility):
    bool remove_all_abilities = false;      // RemoveAllAbilities$ True — affected objects lose all abilities

    // Maximum-hand-size field (category = "Continuous"; CR 402.2 / 614 hand-size modifier). The
    // Tamiyo, Seasoned Scholar ultimate emblem grants "You have no maximum hand size."
    // (SetMaxHandSize$ Unlimited). 0 = the static does not touch max hand size; -1 = no maximum
    // (Unlimited); a positive N sets the maximum to N. Affected$ You scopes it to the static's
    // controller; read by the cleanup-step discard check (state_manager.cpp).
    int set_max_hand_size = 0;

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
