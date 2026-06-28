#ifndef ABILITY_PARAMS_H
#define ABILITY_PARAMS_H

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "../classes/colors.h"

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
    // KW$ Hexproof:Card.<Color>:<desc> — "hexproof from <color>" (Veil of Summer). Parsed out of
    // the keyword list into this color set; the Pump handler turns it into a turn-long player-
    // scoped grant (cur_game.hexproof_from_colors_this_turn) covering the controller and the
    // permanents they control, rather than a per-creature keyword (which couldn't protect the
    // player or non-creature permanents). Empty = no hexproof-from-color grant.
    std::set<Colors> grant_hexproof_from_colors;
    // KW$ Protection from everything with Defined$ You (The One Ring's ETB Pump) — "you gain
    // protection from everything". Parsed out of the keyword list into this flag; the Pump handler
    // turns it into a player-scoped grant (cur_game.player_protection_from_everything) for the
    // controller rather than a per-creature keyword. False = no player-protection grant.
    bool grant_protection_from_everything = false;
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
    // Dynamic mana-value bound for a cmcLE<SVar> filter whose threshold is not the X paid at
    // cast (the legacy cmcLEX path keys off cur_game.x_paid). Wrath of the Skies'
    // "cmcLEY" (Y = Count$ChosenNumber) resolves here: cmc_expr is the runtime Count$
    // expression and cmc_op the comparator ("LE"). Empty = no dynamic bound.
    std::string cmc_expr = "";
    std::string cmc_op = "";
    // UnlessCost$ PayEnergy<N> | UnlessPayer$ You | UnlessSwitched$ True (Wrath of the Skies):
    // the controller pays N energy ({E}); with UnlessSwitched the destroy happens ONLY IF the
    // energy is paid. energy_unless_expr is the runtime Count$ expression for N (Count$ChosenNumber).
    std::string energy_unless_expr = "";
    bool energy_unless_switched = false;  // UnlessSwitched$ True — invert to "do only if paid"
};

// Token creation (e.g. Cori-Steel Cutter). TokenScript$ string parsed at resolve.
struct TokenParams {
    std::string script = "";      // TokenScript$ e.g. "w_1_1_monk_prowess"
    bool owner_is_target = false;  // TokenOwner$ TargetedPlayer — tokens go to the targeted player
    bool owner_is_remembered = false;  // TokenOwner$ RememberedOwner — the token is owned and
                                       // controlled by the owner of the first remembered card
                                       // (Skyclave Apparition: the exiled card's owner gets it)
    // TokenPower$/TokenToughness$ as an SVar expression (e.g. Remembered$CardManaCost). Empty =
    // use the token script's printed P/T; otherwise the created token enters as an X/X with X
    // evaluated from this expression at creation time (Skyclave Apparition's MV-sized Illusion).
    std::string power_expr = "";
    std::string toughness_expr = "";
    // TokenOwner$ Promised (Gift keyword, CR 702.176): the token is created under the control of
    // the opponent who was promised the gift — i.e. the opponent of the ability's controller.
    bool owner_is_promised = false;
    // TokenOwner$ TargetedController (Cityscape Leveler: "destroy ... its controller creates a
    // tapped Powerstone token"): the token is owned/controlled by the controller of the ability's
    // target permanent (read via last-known info, since the target was just destroyed). With no
    // target chosen, no token is created — this implements the "if you do" gating.
    bool owner_is_targeted_controller = false;
    // TokenTapped$ True: the token enters the battlefield tapped (Into the Flood Maw's Fish).
    bool tapped = false;
};

// PutCounter (e.g. Scythecat Cub landfall +1/+1). NOTE: this is the Ability
// variant only — the identically-named fields on StaticAbility (etbCounter) are
// a separate struct and out of scope.
struct CounterParams {
    std::string type = "";          // CounterType$ — "P1P1" for +1/+1 counters
    int count = 0;                  // CounterNum$ — static count; 0 when dynamic
    // CounterNum$ given as an SVar that resolves to a runtime Count$ expression (Wrath of the
    // Skies: CounterNum$ X, X = Count$xPaid — the player gets X {E}). Empty = use the static
    // count above; otherwise the count is evaluated at resolution from this Count$ expression.
    std::string count_expr = "";
    // Optional SECOND counter kind placed by the same effect (PutCounterAll on Guide of
    // Souls: CounterType2$ Flying | CounterNum2$ 1 — also put a flying counter). Empty type2
    // = no second counter.
    std::string type2 = "";         // CounterType2$
    int count2 = 0;                 // CounterNum2$
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
    // Random$ True (Urza's Bauble): the controller looks at ONE card chosen uniformly at
    // random from the target/defined player's HAND. Used by the AB$/DB$ Reveal handler
    // (effect_reveal.cpp), not the PeekAndReveal library-top path above.
    bool random_from_hand = false;
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
    // RememberObjects$ RememberedLKI — at registration, snapshot the objects the immediately
    // preceding RememberChanged$ ChangeZone moved (cur_game.remembered_entities) and carry them
    // with the delayed trigger, so its Execute$ ability can act on those same objects when it
    // fires later (CR 603.7a — the delayed trigger references the objects as they were when it
    // was set up). Used by exile-and-return-at-end-of-turn cards (Flickerwisp, Phelia).
    bool remember_objects_lki = false;
};

#endif /* ABILITY_PARAMS_H */
