#ifndef ABILITY_PARAMS_H
#define ABILITY_PARAMS_H

#include <cstddef>
#include <string>
#include <vector>

// Per-effect parameter blocks held by Ability's `params` variant (see ability.h).
//
// Phase 3 of the effect-dispatch refactor decomposes the Ability god struct:
// fields that are exclusive to a single effect move into one of these structs;
// fields shared across effects (amount, target(s), controller, costs, valid_tgts,
// ...) stay as direct Ability members. Effects whose data is entirely shared use
// std::monostate.
//
// NOTE: the zone/library-manipulation family (ChangeZone, ChangeZoneAll, Counter,
// Dig, Rearrange) reads an overlapping set of fields that do NOT partition per
// kind (e.g. Counter reads `destination`, Rearrange reads `may_shuffle`,
// ChangeZoneAll reads `rest_random_order`/`dig_library_position`). Those fields
// therefore remain shared on Ability rather than being split into per-kind
// alternatives. Only cleanly-exclusive groups become variant alternatives.

struct PumpParams {
    int att = 0;  // NumAtt$ — power modifier (can be negative)
    int def = 0;  // NumDef$ — toughness modifier (can be negative)
    // NumAtt$/NumDef$ given as a count-SVar (e.g. "+X", X = Count$Valid Eldrazi.YouCtrl):
    // the static att/def above stay 0 and the magnitude is evaluated at resolution from
    // these Count$ expressions. The sign of the original "+X"/"-X" token is captured here.
    std::string att_expr = "";   // dynamic power magnitude expression (empty = use att)
    std::string def_expr = "";   // dynamic toughness magnitude expression (empty = use def)
    int att_sign = 1;            // +1 for "+X", -1 for "-X"
    int def_sign = 1;
    // KW$ — keyword(s) granted to the pumped creature until end of turn (e.g. Haste).
    std::vector<std::string> grant_keywords;
};

// Delirium-conditional damage (Unholy Heat). The base damage stays in the
// shared Ability::amount field; these capture only the delirium upgrade.
struct DamageParams {
    bool is_delirium_scale = false;  // use delirium_amount when delirium is active
    size_t delirium_amount = 0;      // damage dealt when the caster has delirium
};

// DestroyAll (e.g. Meltdown). Filter spec like "Artifact.cmcLEX".
struct DestroyAllParams {
    std::string filter = "";  // ValidCards$ — type/CMC filter for what to destroy
};

// Token creation (e.g. Cori-Steel Cutter). TokenScript$ string parsed at resolve.
struct TokenParams {
    std::string script = "";      // TokenScript$ e.g. "w_1_1_monk_prowess"
    bool owner_is_target = false;  // TokenOwner$ TargetedPlayer — tokens go to the targeted player
};

// PutCounter (e.g. Scythecat Cub landfall +1/+1). NOTE: this is the Ability
// variant only — the identically-named fields on StaticAbility (etbCounter) are
// a separate struct and out of scope.
struct CounterParams {
    std::string type = "";          // CounterType$ — "P1P1" for +1/+1 counters
    int count = 0;                  // CounterNum$ — static count; 0 when dynamic
};

// Discard (e.g. Thoughtseize/Duress/Cabal Therapy).
struct DiscardParams {
    std::string valid = "";  // DiscardValid$ — filter (e.g. "Card.nonLand", "Card.NamedCard")
    // Mode$ — "RevealYouChoose" (default): target player reveals hand, the ability's
    // controller picks ONE matching card to discard (Thoughtseize/Duress). Empty is
    // treated as this default. "RevealDiscardAll": target player reveals hand and discards
    // EVERY card matching `valid` (Cabal Therapy). "Random": the target player discards
    // NumCards$ cards chosen uniformly at random from their hand, no reveal, no choice by
    // anyone (Hymn to Tourach); count clamped to hand size (CR 701.8e/f).
    std::string mode = "";
};

// Peek-no-reveal variant (Mishra's Bauble): look at top card privately, no
// reveal choice. Distinguishes the peek path inside the PeekAndReveal handler.
struct PeekParams {
    bool no_reveal = false;  // NoReveal$ True
    int peek_amount = 1;     // PeekAmount$ N — how many top cards to look at (Birthing Ritual: 7)
};

// Amass (e.g. Orcish Bowmasters: "amass Orcs 1"). The counter count lives in the
// shared Ability::amount field; this captures the amassed creature subtype, which
// names both the Army's token script and the type the Army gains.
struct AmassParams {
    std::string subtype = "";  // Type$ — amassed creature subtype, e.g. "Orc"
};

// Delayed trigger registration (e.g. Mishra's Bauble: "draw a card at the
// beginning of your next upkeep").
struct DelayedTriggerParams {
    bool next_turn = false;       // NextTurn$ True — fire next turn vs this turn
    std::string phase = "";       // Phase$ — "Upkeep"/"Draw"/"EndStep" (empty = upkeep)
    std::string execute_svar = "";  // Execute$ — SVar name of the ability to fire
    std::string valid_player = "";  // ValidPlayer$ — "Player"/"You"/"Opponent"
};

#endif /* ABILITY_PARAMS_H */
