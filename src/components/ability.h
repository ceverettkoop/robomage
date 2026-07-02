#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "ability_params.h"
#include "static_ability.h"
#include "types.h"
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
    bool target_max_from_xpaid = false;  // TargetMax$ X (X = Count$xPaid): cap = X paid at cast (Kozilek's Command)
    bool target_min_from_xpaid = false;  // TargetMin$ X (X = Count$xPaid): lower bound = X paid — with the
                                         // matching max gives EXACTLY-X targeting (Candelabra, Hide on the Ceiling)
    std::string target_min_svar = "";    // raw TargetMin$ token when non-numeric (an SVar key); resolved post-parse
    std::string target_max_svar = "";    // raw TargetMax$ token when non-numeric (an SVar key); resolved post-parse
    // TargetMin$/TargetMax$ given as a count-SVar that resolves to a runtime Count$ expression
    // OTHER than Count$xPaid (e.g. Into the Flood Maw: TargetMin$ X = TargetMax$ X, X =
    // Count$PromisedGift.0.1). Evaluated by select_target at cast time (when the count's inputs —
    // e.g. the gift promise — are already decided), then stamped onto target_min/target_max. A
    // resolved count of 0 makes the ability target nothing and do nothing (CR: zero targets).
    std::string target_min_count_expr = "";
    std::string target_max_count_expr = "";
    Entity source = 0;
    Entity target = 0;
    std::vector<Entity> targets;    // used when target_max > 1
    Zone::Ownership controller = Zone::PLAYER_A;  // set when pushed onto stack; stable even if source loses Permanent
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability
    std::vector<Colors> mana_choices;   // Produced$ Combo or Any — ordered list of selectable mana colors
    // AB$ ManaReflected (Mox Amber): a mana ability whose producible colors are the UNION of the
    // colors of the permanents matching this filter (Valid$, controller-scoped). The set is
    // computed dynamically at mana-source enumeration from the live board (each matching
    // permanent's effective_colors), then expanded into per-color choices like mana_choices.
    // Empty = not a reflected-mana ability. CR 605 / "mana of any color among …".
    std::string reflected_mana_filter = "";
    bool restrict_to_chosen_type_creature = false;  // RestrictValid$ Spell.Creature+ChosenType
    bool restrict_to_creature = false;               // RestrictValid$ Spell.Creature (any creature spell)
    bool restrict_to_colorless_eldrazi = false;      // RestrictValid$ Spell.Eldrazi+Colorless (Eldrazi Temple)
    bool adds_no_counter = false;                    // AddsNoCounter$ True — spell can't be countered

    // Set by apply_land_abilities for mana abilities generated from land subtypes;
    // cleared and regenerated each SBE pass when type-changing effects are active.
    bool subtype_derived = false;

    // Layer-6 ability grant (CR 613.1f): a continuous AddAbility$ static (Petrified Hamlet)
    // attached this activated ability to the permanent. Holds the granting static's SOURCE
    // entity so the grant pass can de-dupe (one copy per source static) and remove the grant
    // when the static stops applying / leaves the battlefield. 0 = an intrinsic ability.
    Entity granted_by_static = 0;

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    bool activation_has_x = false;      // Cost$ contains {X}: choose X at activation, add X generic to the
                                        // activation mana cost; X = Count$xPaid (Candelabra of Tawnos)
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    bool life_cost_is_x = false;        // PayLife<X> — variable life cost: the life paid IS X (Count$xPaid),
                                        // chosen as an additional cost while casting (Toxic Deluge)
    int energy_cost = 0;                // PayEnergy<N> — energy ({E}) paid as part of the cost (CR 122.1c)
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string sac_cost_spec = "";     // Sac<1/Type;Type/> — type-based sac cost; empty = none
    // DB$ Sacrifice EFFECT (not a cost): sacrifice a chosen permanent you control matching
    // sac_valid (Birthing Ritual). Distinct from sac_self/sac_cost_spec which are activation costs.
    std::string sac_valid = "";         // SacValid$ — filter for the creature/permanent to sacrifice (e.g. "Creature")
    size_t sac_count = 1;               // number of permanents to sacrifice (Annihilator N: each sac is a separate choice; default 1)
    bool remember_sacrificed = false;   // RememberSacrificed$ True — push the sacrificed entity to remembered_entities
    std::string return_cost_type = "";  // Return<N/Type> — bounce a land of this subtype as cost
    int return_cost_count = 0;          // number of lands to return
    bool discard_hand_cost = false;     // Discard<0/Hand> — discard entire hand as activation cost (Lion's Eye Diamond)
    bool discard_self_cost = false;     // Discard<1/CARDNAME> — discard this card from hand as activation cost
    bool instant_speed = false;         // InstantSpeed$ True — activated ability that is NOT a mana ability; goes on stack
    bool sorcery_speed_only = false;    // SorcerySpeed$ True — activated only as a sorcery (CR 605.x / earthbend); gated in the legal-action enumeration
    // Activation$ <condition> — a named "activate only if <condition>" gate (CR 602.5). The
    // ability is an illegal (not-offered, not-payable) activation unless the named condition
    // holds for its controller at activation time. "Metalcraft" = you control 3+ artifacts
    // (CR 702.46). General: future gated activations (Threshold, Delirium, …) add their name
    // here and a case in activation_condition_met(). Empty = no activation gate.
    std::string activation_condition = "";
    // Monstrosity$ N on an AB$/DB$ PutCounter (CR 701.37): "Monstrosity N" = if this permanent
    // isn't monstrous, put N +1/+1 counters on it and it becomes monstrous. Parsed by
    // parse_put_counter into CounterParams (P1P1 / count N) plus this flag; the flag also installs
    // the "NotMonstrous" activation gate so the ability is illegal once the source is monstrous.
    // At resolution put_counter() places the counters, sets Permanent::is_monstrous, and fires the
    // BECAME_MONSTROUS event (firing the BecomeMonstrous triggers). General over any Monstrosity card.
    bool is_monstrosity = false;
    bool tap_on_etb = false;            // ETB$ True on a DB$ Tap — taps Defined$ Self as it enters the battlefield
    // K:Ninjutsu:<cost> (CR 702.49): a hand-activated ability usable only during the declare-
    // blockers step (after blockers are declared) while its controller has an unblocked attacker.
    // Activating it returns one unblocked attacker to hand and pays the ninjutsu mana cost
    // (activation_mana_cost), then puts THIS card from hand onto the battlefield tapped and
    // attacking the defender the returned attacker was attacking. Handled by process_ninjutsu;
    // the offer is gated to DECLARE_BLOCKERS in determine_legal_actions.
    bool is_ninjutsu = false;
    int activation_limit = 0;           // ActivationLimit$ N — max activations per turn (0 = unlimited)
    // Loyalty abilities (planeswalkers). is_loyalty_ability is the load-bearing flag;
    // loyalty_cost == 0 is still a valid loyalty ability (e.g. Jace "0:" Brainstorm), so
    // never infer loyalty-ness from loyalty_cost. Cost$ AddCounter<N/LOYALTY> → +N,
    // SubCounter<N/LOYALTY> → -N. Paid by modifying the source's own loyalty at activation.
    bool is_loyalty_ability = false;    // Planeswalker$ True
    int loyalty_cost = 0;               // +N (AddCounter) or -N (SubCounter); loyalty counters added/removed as the cost
    // X loyalty cost (Cost$ SubCounter<X/LOYALTY> — Chandra, Flamecaller's [-X] ultimate). The
    // magnitude is chosen at activation (X, bounded by current loyalty for a minus ability) and
    // recorded as cur_game.x_paid so the effect's Count$xPaid (e.g. NumDmg$ X) reads it; loyalty_cost
    // then carries only the SIGN (-1 minus / +1 plus). Never stoi("X") at parse time.
    bool loyalty_cost_is_x = false;
    int activation_zone = -1;           // ActivationZone$ Hand → Zone::HAND; -1 = default (battlefield)
    int activations_this_turn = 0;      // runtime counter, reset at UNTAP
    // ReduceCost$ on an ACTIVATED ability (Eiganjo's Channel: "ReduceCost$ X",
    // X = Count$Valid Creature.Legendary+YouCtrl): the GENERIC portion of
    // activation_mana_cost is reduced by this amount at activation time (CR 601.2f —
    // cost reductions reduce only generic mana, colored pips are never reduced). Either a
    // literal integer (stored verbatim, e.g. "1") or a runtime Count$/SVar expression
    // resolved via evaluate_dynamic_amount with the activating controller as "you". Empty
    // = no reduction. effective_activation_mana_cost() applies it for BOTH the affordability
    // check and the actual payment so the two never diverge.
    std::string reduce_cost_expr = "";
    std::string change_type = "";        // ChangeType$ — comma-separated subtypes to search
    // ChangeType filter with a dynamic mana-value bound (Aether Vial: "Creature.cmcEQX",
    // X = Count$CardCounters.CHARGE). Holds the resolved runtime Count$ expression and the
    // two-letter comparator ("EQ"/"LE"); evaluated against this ability's source at
    // resolution and applied as a per-card mana-value gate in the zone search. Empty = no
    // dynamic cmc filter (the legacy cmcLEX path keys off cur_game.x_paid instead).
    std::string change_type_cmc_expr = "";
    std::string change_type_cmc_op = "";
    Zone::ZoneValue origin = Zone::LIBRARY;          // Origin$ — zone to search
    Zone::ZoneValue destination = Zone::BATTLEFIELD; // Destination$ — zone to move card to
    // RememberTargets$ / RememberObjects$ Targeted — at resolution, push the target(s)
    // into cur_game.remembered_entities so chained ChangeType$ Remembered.sameName
    // subabilities can reference the card whose name to match (Surgical Extraction).
    bool remember_targeted = false;
    // TgtZone$ Graveyard — this spell/ability targets a card in a graveyard even though
    // its category isn't ChangeZone (e.g. Surgical Extraction's SP$ Pump vehicle). Lets
    // target enumeration offer graveyard cards.
    bool target_in_graveyard = false;
    uint32_t trigger_on = 0;             // EventId that fires this ability; 0 = not event-triggered
    // Additional EventIds this ability also fires on (a phase trigger listing multiple phases,
    // e.g. Carpet of Flowers' Phase$ Main1,Main2 binds FIRST_MAIN_BEGAN here-plus SECOND_MAIN_BEGAN).
    // The general battlefield trigger scan matches trigger_on OR any entry here.
    std::vector<uint32_t> trigger_on_extra;
    // Static$ True bookkeeping trigger (Carpet of Flowers' cleanup / leave-battlefield resets):
    // Forge's internal "static" triggers do not use the stack — they resolve immediately and
    // off-stack (CR 605.1a-style) the instant their event fires. The trigger scan resolves the
    // Execute effect inline instead of placing a PendingTrigger. Distinct from
    // trigger_taps_for_mana_static (which the mana system resolves) — this covers phase/zone-change
    // static triggers. General over any Static$ True phase/ChangesZone trigger.
    bool trigger_static_offstack = false;
    // Per-permanent stored-SVar trigger gate (Carpet of Flowers' CheckSVar$ CarpetX | SVarCompare$
    // EQ0 — "if you haven't added mana with this ability this turn"). When set, the trigger fires
    // only if the SOURCE permanent's stored_svars[gate_name] (default 0) satisfies gate_compare
    // (e.g. "EQ0"). Like an intervening-if (CR 603.4) it is re-checked at resolution. Distinct
    // from the existing CheckSVar path (which resolves the SVar to a Count$ board expression);
    // this reads a Number$ scratch int latched by DB$ StoreSVar. General over the StoreSVar/
    // CheckSVar latch idiom.
    std::string stored_svar_gate_name = "";
    std::string stored_svar_gate_compare = "";
    bool trigger_self_excluded = false;  // true when ValidCard$ has .Other — won't trigger for the source itself
    bool trigger_only_self = false;      // true when ValidCard$ Card.Self — only fires when the entering entity is the source itself
    // true when ValidCard$ has the wasCastByYou qualifier (The One Ring's "enters, if you cast
    // it" ETB) — an intervening cast-condition: the trigger fires only when its source permanent
    // entered the battlefield by being cast (Permanent::entered_by_cast).
    bool trigger_requires_entered_by_cast = false;
    // This triggered ability is a Saga chapter ability (CR 714.3). When it finishes resolving and
    // leaves the stack, StackManager::resolve_top decrements its source Saga's
    // saga_chapters_in_flight so the 714.4 sacrifice SBA can fire. Set when the chapter ability is
    // built from CardData::saga_chapters in the trigger system.
    bool is_saga_chapter = false;
    bool is_evoke_sacrifice = false;     // synthetic ETB self-trigger from K:Evoke — only fires when the permanent was evoked
    bool is_offspring_token = false;     // synthetic ETB self-trigger from K:Offspring — only fires when the permanent was cast with offspring; creates a 1/1 token copy
    bool trigger_valid_player_is_controller = false;  // true when ValidPlayer$ You
    bool mandatory = false;              // Mandatory$ True — player must choose; suppresses fail-to-find when zone non-empty
    bool shuffle_after = false;          // Shuffle$ True — shuffle the destination library after a ChangeZoneAll move (Emrakul death trigger: shuffle graveyard into library)
    bool may_shuffle = false;            // MayShuffle$ True — player may optionally shuffle after
    size_t unless_generic_cost = 0;      // UnlessCost$ N — target controller pays {N} to prevent counter
    bool unless_cost_is_life = false;    // when true, unless_generic_cost is paid as N life rather than {N} mana (Ward—Pay life, CR 702.21)
    // UnlessCost$ Discard<N/Card> (Reality Smasher: "counter ... unless its controller discards a
    // card"). The payer discards `unless_generic_cost` card(s) of their choice from hand to prevent
    // the counter; if they can't (or decline) the spell is countered. General "counter unless
    // discard" path (CR 701.8 discard); reuses the same run_unless_loop choice machinery.
    bool unless_cost_is_discard = false;
    // UnlessCost$ PayEnergy<N> on a non-DestroyAll effect (Static Prison: "sacrifice CARDNAME
    // unless you pay {E}"). The payer may pay `unless_generic_cost` energy ({E}, CR 122.1c) to
    // prevent the effect; if they can't or decline, the effect happens. (DestroyAll's energy
    // unless-cost rides on DestroyAllParams::energy_unless_expr instead — see parse.cpp.)
    bool unless_cost_is_energy = false;
    // UnlessPayer$ TriggeredSourceSAController — the payer of the unless-cost is the controller of
    // the triggering spell/ability (the opponent who targeted the permanent), NOT the trigger's own
    // controller. Bound at trigger-fire time into unless_payer (UNKNOWN ⇒ default to the countered
    // spell's controller, as Ward does). General for any "unless its controller pays/discards".
    bool unless_payer_is_triggered_source_sa_ctrl = false;
    Zone::Ownership unless_payer = Zone::UNKNOWN;  // resolved payer for the unless-cost; UNKNOWN ⇒ default
    std::string target_type = "";        // TargetType$ Spell — restricts targeting to stack spells

    // Delirium-conditional damage (Unholy Heat) now lives in DamageParams (params variant).
    std::string amount_svar = "";           // raw SVar key for non-numeric NumDmg$ (resolved at parse time)
    std::string dynamic_amount_expr = "";   // runtime SVar expression (e.g. "Count$Valid Creature.YouCtrl" or "Targeted$CardPower")
    // Raw Defined$/DefinedPlayer$ token verbatim from the script (e.g. "Targeted", "ParentTarget",
    // "You", "Opponent", "Remembered", "TargetedController", "Self", ...). Empty when the ability
    // declared no Defined$. Kept alongside the specific bools above so sub-ability target
    // resolution can read the script's stated intent (CR 608.2c) instead of relying on a blanket
    // N_A sentinel; the specific bools remain authoritative for their effects.
    std::string defined = "";
    bool defined_targeted_controller = false;  // Defined$ TargetedController — GainLife goes to target's controller
    // Chooser$ You — for a search/move ChangeZone over a player's hidden zone, the SELECTION is
    // made by the ability's controller, not the searched zone's owner. Thought-Knot Seer: the
    // targeted opponent reveals their hand (DefinedPlayer$ Targeted routes the search to that
    // opponent's hand) and YOU (the controller) choose which revealed nonland card to exile. The
    // searched cards are already public (revealed), so the controller may see them when choosing.
    // Default (false) = the zone's owner chooses (the normal tutor case).
    bool chooser_is_controller = false;
    bool defined_self = false;                  // Defined$ Self — ability moves its own source
    // Defined$ TriggeredAttacker / TriggeredAttackerLKICopy — the effect (a Pump) acts on the
    // creature that triggered this ability by attacking (Tamiyo, Seasoned Scholar: the attacking
    // creature "gets -1/-0"). Bound as this ability's target at trigger-fire time from the
    // CREATURE_ATTACKED event's ENTITY. The LKICopy form would use the attacker's last-known
    // info if it has since left combat/play; the trigger resolves during the same declare-
    // attackers step (the attacker is still on the battlefield), so the live entity is used.
    bool defined_triggered_attacker_lki = false;
    // K:Unearth (CR 702.84): this ChangeZone is the unearth return (Graveyard -> Battlefield).
    // When it resolves and the source enters the battlefield, the permanent is marked "unearthed"
    // (gains haste, is exiled at the next end step, and is exiled instead of leaving the
    // battlefield). General over any unearth card.
    bool is_unearth = false;
    bool defined_each_opponent = false;         // Defined$ Player.Opponent — effect applies to each opponent (no target)
    bool defined_you = false;                   // Defined$ You — effect's player is the source's controller (e.g. Ancient Tomb pain)
    // Defined$ TriggeredActivator — the effect's player is the player who caused the trigger
    // (the caster of the triggering spell / the activator of the triggering event), CR 603.x.
    // Set at parse time; the actual player is captured into `triggered_activator` when the
    // trigger fires (from the event's PLAYER param). Used by Mai, Scornful Striker (the player
    // who cast the noncreature spell loses 2 life), but general to any effect reading a
    // Defined player.
    bool defined_triggered_activator = false;
    // The player who caused this triggered ability to fire (the triggering event's PLAYER).
    // Populated at trigger-fire time when defined_triggered_activator is set; UNKNOWN until then.
    Zone::Ownership triggered_activator = Zone::UNKNOWN;

    // DB$ Animate (Guide of Souls): the Types$ list (classified into TYPE/SUBTYPE/SUPERTYPE
    // at parse time) to add to the targeted permanent (e.g. "Angel"), and whether the grant
    // is permanent (Duration$ Permanent). The Animate handler bakes these onto the target's
    // Permanent so the layer system reapplies them. Only the type-add path is exercised today;
    // base-P/T/keyword/creature extension points live on Permanent (see permanent.h).
    std::vector<Type> animate_types;
    // DB$ Animate | Abilities$ <svar>[,<svar>...] (Urza's Saga chapters I & II): activated
    // ability(ies) the Animate GRANTS to the target permanent for the effect's Duration (CR
    // 613.1f — a lasting "gains [activated ability]"). Each entry is a fully-parsed AB$ ability
    // (e.g. "{T}: Add {C}." or "{2}, {T}: Create a Construct token"). The Animate handler pushes
    // them onto the permanent's activated-ability list for a Duration$ Permanent grant. Empty when
    // the Animate grants no abilities (Guide of Souls' type-only animate).
    std::vector<Ability> animate_granted_abilities;
    bool animate_duration_permanent = false;
    // Duration$ UntilYourNextTurn (Karn, the Great Creator +1): the grant lasts until the start
    // of the animating player's NEXT turn — longer than the Forge default (until end of turn)
    // but not the rest of the game (Duration$ Permanent). Reverted at that player's untap step.
    bool animate_duration_until_your_next_turn = false;
    // Duration$ UntilYourNextTurn on a non-Animate effect (The One Ring's ETB Pump that grants the
    // controller protection from everything): the effect lasts until the start of the controller's
    // next turn rather than the Forge default (until end of turn). Reverted at that player's untap.
    bool duration_until_your_next_turn = false;
    // AB$ Animate | Power$/Toughness$ (Karn +1: "power and toughness each equal to its mana
    // value"). A numeric value sets the base P/T directly; an SVar token (e.g. Power$ X with
    // SVar:X:Targeted$CardManaCost) is resolved post-parse into animate_power_expr /
    // animate_toughness_expr — a runtime dynamic_amount expr evaluated against the animate TARGET
    // (a snapshot of its mana value taken at resolution). animate_has_pt gates the whole path.
    bool animate_has_pt = false;
    int animate_base_power = 0;
    int animate_base_toughness = 0;
    std::string animate_power_token;       // raw Power$ token, pre-SVar-resolution
    std::string animate_toughness_token;   // raw Toughness$ token, pre-SVar-resolution
    std::string animate_power_expr;        // resolved runtime expr (empty when numeric)
    std::string animate_toughness_expr;

    // Filter naming which permanents a mass effect affects (DestroyAll / SacrificeAll /
    // PutCounterAll): the ValidCards$ spec, e.g. "Cat.YouCtrl" or
    // "Permanent.nonLand+OppCtrl+nonChosenCard". A plain member (not in the params
    // variant) so PutCounterAll can carry it alongside CounterParams.
    std::string valid_cards_filter = "";

    // AB$ AnimateAll | RemoveKeywords$ Hexproof & Indestructible (Shadowspear): the keyword(s)
    // a mass continuous effect REMOVES from every permanent matching valid_cards_filter, until
    // end of turn (CR 613 layer 6 keyword removal). Empty for effects that grant rather than
    // remove. The add direction (AddKeyword$/Keywords$) reuses the same effect via add_keywords.
    std::vector<std::string> remove_keywords;
    std::vector<std::string> add_keywords;  // AnimateAll grant direction (until end of turn)

    // Condition$ Blessing (CopyPermanent on Ocelot Pride): the effect body runs only if the
    // ability's controller has the city's blessing (702.131). Like other condition gates, a
    // false condition skips the body but still chains subabilities.
    bool condition_city_blessing = false;

    // Effect-specific parameter blocks. As effects migrate off the flat
    // god-struct fields (Phase 3), their exclusive data moves into one of these
    // variant alternatives; shared fields stay as direct members. std::monostate
    // covers effects with no exclusive fields. See ability_params.h.
    std::variant<std::monostate, PumpParams, DamageParams, DestroyAllParams, TokenParams,
                 DelayedTriggerParams, CounterParams, DiscardParams, PeekParams, AmassParams> params;

    // Counter abilities (PutCounter category) now live in CounterParams (params variant).

    // Peek variant (Mishra's Bauble) now lives in PeekParams (params variant).

    // Delayed trigger params (Mishra's Bauble) now live in DelayedTriggerParams (params variant).

    // Zone-change trigger filters for CARD_CHANGED_ZONE (set by Mode$ ChangesZone triggers)
    int trigger_zone_origin = -1;       // Zone::ZoneValue origin filter; -1 = any
    int trigger_zone_destination = -1;  // Zone::ZoneValue destination filter; -1 = any
    bool trigger_valid_card_is_creature = false;        // ValidCard$ Creature
    bool trigger_valid_card_is_instant_or_sorcery = false;  // ValidCard$ Instant/Sorcery
    bool trigger_valid_card_is_land = false;            // ValidCard$ Land.*
    bool trigger_valid_card_is_artifact = false;        // ValidCard$ Artifact.* (Kappa Cannoneer)
    // ValidCard$ ...+untapped — the changing card must be an UNTAPPED battlefield permanent
    // at trigger time (Mystic Sanctuary: "When this land enters untapped"). The enters-tapped
    // replacement (T2.2) has already set Permanent::is_tapped by the time triggers are scanned.
    bool trigger_valid_card_untapped = false;
    // ValidCard$ Card.nonCreature combined with an ActivatorThisTurnCast$ count (The Fantasticar's
    // "your fourth noncreature spell each turn"): the cast spell must be NONCREATURE. Bound to the
    // SPELL_CAST event (fired after the per-cast counters are bumped) rather than the dedicated
    // NONCREATURE_SPELL_CAST event (fired before), so a count gate sees the current cast counted.
    bool trigger_valid_card_non_creature = false;
    // ValidCard$ ...+!token — the changing card must be a real card, not a token (CR 110.1 /
    // 111.7). Moonshadow's "permanent cards put into your graveyard" excludes tokens.
    bool trigger_valid_card_non_token = false;
    // ValidCard$ Permanent — the changing card must be a permanent card (CR 110.4a: artifact,
    // battle, creature, enchantment, land, planeswalker), excluding instants/sorceries.
    bool trigger_valid_card_is_permanent = false;
    // ValidCard$ ...+Colorless — the cast spell (SpellCast) or changing card (ChangesZone)
    // must be colorless (CR 105.2c / 202.2). Used by Glaring Fleshraker (Card.Colorless
    // SpellCast trigger; Creature.Other+Colorless+YouCtrl ChangesZone trigger).
    bool trigger_valid_card_colorless = false;
    // ValidCard(s)$ <Subtype> — the changing card must have this subtype (e.g. Ajani's
    // "Cat.Other+YouCtrl"). Empty = no subtype filter. Matched against CardData/Token types.
    std::string trigger_valid_card_subtype = "";
    // OptionalDecider$ You — a "you may ..." triggered ability; its controller is prompted
    // to accept or decline as it resolves (Mode$ ChangesZoneAll on Ajani).
    bool trigger_optional = false;

    // Drawn trigger (Orcish Bowmasters): Mode$ Drawn fires on PLAYER_DREW_CARD.
    bool trigger_valid_card_opp_own = false;       // ValidCard$ Card.OppOwn — the drawn card is owned by an opponent of the source's controller
    bool trigger_exclude_first_draw_step = false;  // FirstCardInDrawStep$ False — ignore the first card the player draws in each of their draw steps
    // Number$ N on a Mode$ Drawn trigger (CR 603.2, Tamiyo, Inquisitive Student: "whenever you
    // draw your THIRD card in a turn"): fire only when the firing draw is the Nth card that player
    // has drawn this turn. Matched against the per-draw running count carried on the
    // PLAYER_DREW_CARD event (Params::AMOUNT, 1-based). 0 = no Nth-draw gate (every draw fires).
    size_t trigger_draw_number_eq = 0;

    // Combat damage trigger (Barrowgoyf): damage amount stored at trigger fire time
    size_t trigger_damage_amount = 0;

    // Spell count trigger (Cori-Steel Cutter)
    size_t trigger_spell_count_eq = 0;  // ActivatorThisTurnCast$ EQN — fires on Nth spell
    // The Nth-spell count above counts only NONCREATURE spells (The Fantasticar) rather than all
    // spells cast this turn — compare against Player::noncreature_spells_cast_this_turn.
    bool trigger_spell_count_noncreature = false;

    // Kicker-linked SpellCast trigger (CR 702.33e/f, Wastescape Battlemage): a
    // "When you cast this spell, if it was kicked with its [N] kicker, ..." trigger.
    // ValidCard$ Card.Self+kicked N — fires on SPELL_CAST of the source spell itself only
    // when its (N)th kicker was paid. 0 = no kicker requirement; N >= 1 requires the spell's
    // kicked[N-1] flag. Matched against the cast spell's Spell::kicked in the dedicated
    // self-cast trigger scan (the spell is on the stack, not the battlefield, when it fires).
    int trigger_kicked_index = 0;

    // SpellCast trigger with a dynamic mana-value filter on the cast spell
    // (Chalice of the Void: ValidCard$ Card.cmcEQY, Y = Count$CardCounters.CHARGE — fires
    // when a spell's mana value equals the source's charge-counter count). Empty expr = no
    // such filter. The expr is the resolved Count$ expression; op is "EQ"/"LE"/"GE"/...
    std::string trigger_cmc_expr = "";
    std::string trigger_cmc_op = "";

    // Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl (Reality Smasher): the trigger fires only
    // when the targeting object is a SPELL controlled by an opponent of the source's controller
    // ("a spell an opponent controls"). Matched at trigger-fire time against the targeting object
    // (BECAME_TARGET event's ENTITY/PLAYER). ValidTarget$ Card.Self reuses trigger_only_self (the
    // permanent that became a target must be this source). General over becomes-target triggers.
    bool trigger_source_must_be_spell = false;    // ValidSource$ Spell — targeting object is a spell
    bool trigger_source_opp_ctrl = false;         // ValidSource$ ...OppCtrl — controlled by an opponent of the source's controller

    // Mode$ DamageAll | ValidSource$ Creature.YouCtrl | ValidTarget$ Player | CombatDamage$ True —
    // "Whenever one or more creatures you control deal combat damage to one or more players" (Forth
    // Eorlingas!'s floating monarch trigger). Fires on COMBAT_DAMAGE_TO_PLAYER when the damaging
    // creature (the event's ENTITY) is controlled by this ability's controller. Used by floating
    // triggers (no source permanent), so it cannot rely on trigger_only_self/source scans.
    bool trigger_damage_source_youctrl = false;   // ValidSource$ Creature.YouCtrl on a DamageAll combat-damage trigger

    // Mode$ Attacks | ValidCard$ Creature.OppCtrl (Tamiyo, Seasoned Scholar's +2 hosted trigger):
    // the attacking creature (CREATURE_ATTACKED's ENTITY) must be controlled by an opponent of
    // this trigger's controller. Matched at fire time (the trigger is hosted on a command-zone
    // Effect with no source permanent to self-reference).
    bool trigger_attacker_opp_ctrl = false;
    // Attacked$ You,Planeswalker.YouCtrl — the attack must be against the trigger's controller or a
    // planeswalker they control. In the two-player engine the only defender an opponent's attacker
    // can have IS this controller (or their planeswalker), so this is satisfied whenever an
    // opponent's creature attacks; the flag records the script's stated intent (CR 508.1).
    bool trigger_attacked_defender_you = false;

    // TriggerZones$ Graveyard (Arclight Phoenix): the triggered ability functions from
    // the graveyard, not the battlefield (CR 113.6 / 603.6). When set, the trigger scan
    // matches the source while it is in its owner's graveyard.
    bool trigger_from_graveyard = false;

    // Mode$ TapsForMana | Static$ True (Badgermole Cub): "whenever you tap a creature for mana,
    // add an additional {G}." A mana-additional triggered ability that resolves immediately as
    // part of the mana tap (CR 605.1a) — it never uses the stack. Handled directly by the mana
    // system, not the stack-trigger scan. The Execute$ SVar's AddMana effect lives in subabilities
    // (the produced color/amount); ValidCard$ Creature gates which tapped source it watches.
    bool trigger_taps_for_mana_static = false;

    // Token creation (Cori-Steel Cutter) now lives in TokenParams (params variant).

    // Attach / Equip sub-ability
    bool optional = false;           // Optional$ True — player may decline
    bool defined_remembered = false; // Defined$ Remembered — target is cur_game.remembered_entities[0]
    // True for the sub-ability a DB$ DelayedTrigger named in its Execute$ (vs. a trailing
    // SubAbility$ cleanup). delayed_trigger() fires this one and chains the rest after it.
    bool from_delayed_execute = false;
    bool defined_triggered_spell = false; // Defined$ TriggeredSpellAbility — target is the spell that triggered this ability (Chalice of the Void counters it)
    // Defined$ TriggeredSourceSA — target is the spell/ability that targeted the source (the
    // "triggering source spell ability" of a Mode$ BecomesTarget trigger, Reality Smasher). Bound
    // at trigger-fire time from the BECAME_TARGET event's ENTITY (the targeting object). The Counter
    // effect counters that specific spell. Distinct from TriggeredSpellAbility (the spell whose cast
    // fired a SpellCast trigger) — here the trigger is "became the target of", not "was cast".
    bool defined_triggered_source_sa = false;

    // RepeatEach over players (Price of Progress): RepeatPlayers$ Player makes the effect
    // loop once per player, setting cur_game.remembered_entities to that player's entity
    // before resolving the RepeatSubAbility (parsed into subabilities). Empty = not a
    // per-player repeat.
    std::string repeat_players = "";  // RepeatPlayers$ — currently "Player" (each player)

    // Mill: remember milled cards in cur_game.remembered_entities
    bool remember_milled = false;    // RememberMilled$ True
    bool amount_from_damage = false; // NumCards$ DamageAmount — use trigger_damage_amount

    // Leaves-the-battlefield ability that operates on the cards its source had exiled
    // (Skyclave Apparition's TrigToken): the trigger-firing code snapshots the source's
    // exiled_with here (from the live Permanent, or its last-known info if already stripped),
    // and resolve() restores it into cur_game.remembered_entities so the body's
    // Remembered$CardManaCost (token P/T), TokenOwner$ RememberedOwner, and
    // ConditionPresent$ Card.ExiledWithSource gate all read the exiled card. Empty = no restore.
    std::vector<Entity> restore_remembered_exiled_with;

    // Cleanup sub-ability
    bool clear_remembered = false;   // ClearRemembered$ True
    bool clear_chosen = false;       // ClearChosenCard$ True — clears cur_game.chosen_cards

    // ChooseCard ChooseEach$ "Type & Type & ..." (Ajani -4): each affected player chooses
    // one permanent of each listed type from among their matching permanents to keep
    // (recorded in cur_game.chosen_cards). Empty = the legacy single-pick ChooseCard.
    std::string choose_each = "";

    // SP$/AB$ Vote VoteCard$ <filter> (Council's Judgment): the permanent filter the vote
    // chooses among (e.g. "Permanent.nonLand+YouDontCtrl"). In the two-player engine the vote
    // reduces to the controller choosing one matching permanent; the winner is fed to the
    // VoteSubAbility (parsed into subabilities) via the remembered set. See effect_vote.cpp.
    std::string vote_card_filter = "";

    // RememberChanged$ — remember entities moved by this ChangeZone (for Doomsday)
    bool remember_changed = false;

    // RememberLKI$ — remember the moved object's last-known information (Boomerang Basics:
    // remember the bounced permanent so a paired ConditionDefined$ RememberedLKI gate can
    // read the controller it had as it left the battlefield, CR 608.2g). Like remember_changed
    // it stashes the moved entity in cur_game.remembered_entities; the condition reads its LKI.
    bool remember_lki = false;

    // RememberRevealed$ True (Cloak and Dagger's DBRevealHand): after a RevealHand resolves,
    // stash every revealed card into cur_game.remembered_entities so a later Defined$ Remembered
    // effect can act on them. Starts a fresh remembered set (clears it first) — it is the first
    // link of the chain that records the candidates a subsequent exile picks from.
    bool remember_revealed = false;

    // RememberPumped$ True (Cloak and Dagger's DBPump): a Pump used purely as an (optional)
    // target-selector — it applies no stat change, it just APPENDS its chosen creature to
    // cur_game.remembered_entities so it joins the candidate pool of a later exile.
    bool remember_pumped = false;

    // Duration$ UntilHostLeavesPlay on a ChangeZone | Destination$ Exile (CR 603.6e linked
    // exile-and-return: Cloak and Dagger, Sheltered by Ghosts, Static Prison). Each card moved
    // to exile is recorded against the ability's host (source) so that WHEN THE HOST LEAVES THE
    // BATTLEFIELD the card RETURNS to the zone it came from — a hand card to its owner's hand, a
    // battlefield permanent onto the battlefield under its owner's control. Implemented as a
    // per-card delayed trigger watching the host's departure (effects::register_exile_until_host_leaves).
    bool duration_until_host_leaves = false;

    // DB$ Effect — a transient continuous effect created by an ability (CR 611). The
    // StaticAbilities$ list names the continuous static(s) it grants; RememberObjects$ Self
    // means the effect tracks its own source. Kappa Cannoneer's "it can't be blocked this
    // turn" is StaticAbilities$ Unblockable + RememberObjects$ Self — applied as an
    // until-end-of-turn "can't be blocked" mark on the source rather than a stack object.
    std::string effect_static_ability = "";  // StaticAbilities$ value (e.g. "Unblockable")
    bool effect_remember_self = false;        // RememberObjects$ Self

    // Emblem (CR 114): an AB$ Effect with StaticAbilities$ <SVar> + Duration$ Permanent that gives
    // its controller an emblem carrying the named continuous static ("Ninjas you control get
    // +1/+1." on Kaito's [+1]). The resolved permanent static(s) are stored here at parse time; the
    // GrantCast handler creates a player-owned Emblem (cur_game.emblems) carrying them — an
    // unremovable, zoneless source whose statics are gathered into g_active_statics each SBA pass.
    std::vector<StaticAbility> effect_emblem_statics;

    // DB$ Effect | StaticAbilities$ <SVar> where the named static grants MayPlay$ True +
    // MayPlayWithoutManaCost$ True with AffectedZone$ Exile (Ugin, Eye of the Storms' -11:
    // "Until end of turn, you may cast those cards without paying their mana costs"). The
    // StaticAbilities$ SVar is resolved at parse time; when it carries that grant this flag is
    // set, and the GrantCast handler records a free (no-mana-cost) cast-from-exile permission
    // for each currently-remembered exiled card (cur_game.remembered_entities), good until end
    // of turn. CR 113.3 / 601.3e / 118.9 (cast without paying mana cost).
    bool effect_grant_free_cast_from_exile = false;

    // DB$ Effect | Triggers$ <SVar> — a transient until-end-of-turn floating triggered ability
    // (Forth Eorlingas!'s "Whenever one or more creatures you control deal combat damage to one
    // or more players this turn, you become the monarch"). The named trigger SVar is parsed into
    // a full TRIGGERED Ability and held here; the GrantCast handler registers a copy in
    // cur_game.floating_triggers (controller bound) so the trigger scan fires it through the
    // normal trigger system, then it lapses at cleanup. Empty subabilities vector = no floating
    // trigger. General over any DB$ Effect that names a Triggers$ SVar.
    std::vector<Ability> effect_floating_triggers;

    // DB$ Effect | ReplacementEffects$ <SVar> where the named SVar is a CR 614.13/CantHappen
    // "Event$ Counter | ValidSA$ Spell.YouCtrl | Layer$ CantHappen" (Veil of Summer:
    // "Spells you control can't be countered this turn"). Set at parse time; the GrantCast
    // handler records the effect's controller in cur_game.cant_counter_spells_of for the rest
    // of the turn (a turn-long, sourceless can't-be-countered grant — distinct from Hexing
    // Squelcher's battlefield static).
    bool effect_spells_uncounterable_this_turn = false;

    // Tapped$ True — searched card enters the battlefield tapped (Edge of Autumn)
    bool enters_tapped = false;

    // Transformed$ True — the moved card enters the battlefield showing its DFC back
    // face (Ajani's exile-and-return-transformed). Consumed at permanent creation.
    bool enters_transformed = false;

    // Multi-zone origin support (e.g. Origin$ Graveyard,Library)
    std::vector<Zone::ZoneValue> origins;  // populated when Origin$ has commas; origin holds first value

    // Dig ability (Once Upon a Time, Thassa's Oracle)
    size_t dig_num = 0;              // DigNum$ N — how many cards to look at from top of library
    std::string dig_num_expr = "";   // DigNum$ SVar — dynamic dig count (e.g. "Count$Devotion.Blue")
    std::string change_valid = "";   // ChangeValid$ — comma-separated filter like "Card.Creature,Card.Land"
    bool rest_random_order = false;  // RestRandomOrder$ True
    bool optional_choice = false;    // Optional$ True in Dig context — can choose nothing
    bool change_num_any = false;     // ChangeNum$ Any — may take any number (0..pool) of looked-at cards (Fateseal)
    int change_num = -1;             // ChangeNum$ <N> for Dig — exact take count incl. 0 (-1 = unset); honored over amount so "take 0" works (Birthing Ritual DBDigBis)
    int dig_destination = -1;        // DestinationZone$ — where chosen card goes (-1 = HAND, Zone::LIBRARY etc.)
    int dig_library_position = -1;   // LibraryPosition$ — 0 = top, -1 = unset
    int dig_rest_library_position = -1;  // LibraryPosition2$ — where unchosen cards go: 0 = top, -1 = bottom (default)

    // DB$ DigUntil (Amped Raptor): exile from the top of the library until a card matches
    // change_valid (Valid$). dig_until_found_dest is where the matching card goes,
    // dig_until_revealed_dest where the non-matching cards passed over go (both Exile for
    // Amped Raptor). dig_until_remember_found stores the matching card in
    // cur_game.remembered_entities (RememberFound$) for a chained DB$ Play.
    int dig_until_found_dest = Zone::HAND;      // FoundDestination$ — zone the matching card goes to
    int dig_until_revealed_dest = Zone::LIBRARY; // RevealedDestination$ — zone the skipped cards go to
    bool dig_until_remember_found = false;       // RememberFound$ True

    // DB$ Play (Amped Raptor): cast a Defined$ card from its current zone, paying an
    // alternative RESOURCE cost (PlayCost$) instead of its mana cost. play_cost_resource is
    // the resource paid (energy or life); play_cost_expr is the amount — either a literal int
    // (as a string) or "ConvertedManaCost" (the cast card's mana value). play_valid_sa
    // restricts to nonland spells (ValidSA$ Spell). The optionality is carried by
    // optional_choice (Optional$ True). General over the resource so a future Bolas's Citadel
    // ("pay life equal to mana value") reuses this path with play_cost_resource = LIFE.
    enum PlayCostResource { PLAY_COST_ENERGY, PLAY_COST_LIFE };
    PlayCostResource play_cost_resource = PLAY_COST_ENERGY;
    std::string play_cost_expr = "";  // amount: "ConvertedManaCost" or a literal int string
    bool play_valid_sa_spell = false; // ValidSA$ Spell — only a castable nonland spell

    // Conditional amount (Flow State): the effective count is `cond_amount_if_true`
    // when the summed runtime counts in `cond_amount_exprs` satisfy
    // `cond_amount_compare`, otherwise `amount` (the false/default value).
    // Generalizes "Count$Compare <SVar-sum> <op><n>.<true>.<false>" where the SVar
    // is itself a sum of capped graveyard counts (SVar$Z1/Plus.Z2 with /LimitMax).
    // Currently consumed by the Dig effect as its take-count.
    bool cond_amount_active = false;
    std::vector<std::string> cond_amount_exprs;  // runtime Count$ exprs to sum (LHS of compare)
    std::string cond_amount_compare = "";        // e.g. "GE2"
    size_t cond_amount_if_true = 0;              // count when the compare passes

    // Discard ability (Thoughtseize, Duress) now lives in DiscardParams (params variant).

    // Conditional subability execution (Scythecat Cub, Thassa's Oracle)
    std::string condition_check_svar = "";   // ConditionCheckSVar$ — resolved expression e.g. "Count$ResolvedThisTurn"
    std::string condition_svar_compare = ""; // ConditionSVarCompare$ — e.g. "EQ2", "NE2", "GE1", or "LEX" with SVar RHS
    std::string condition_compare_svar_expr = "";  // when compare RHS is an SVar (e.g. LEX → "Count$Devotion.Blue")

    // Castability condition (Edge of Autumn): count permanents matching filter, compare to threshold
    std::string condition_present = "";   // ConditionPresent$ — e.g. "Land.YouCtrl"
    std::string condition_compare = "";   // ConditionCompare$ — e.g. "LE4", "GE3"
    // ConditionNotPresent$ — the condition is INVERTED: the gated body runs only when the
    // condition_present check is FALSE (Uro: "sacrifice it unless it escaped" →
    // ConditionNotPresent$ Card.Self+escaped sacrifices unless the permanent escaped). Applied
    // by evaluate_present_condition, which negates its result when this is set. General — works
    // with any condition_present filter.
    bool condition_negate = false;
    // ConditionDefined$ Targeted — the condition applies to the chosen target at
    // resolution (e.g. Fatal Push "destroy if its mana value <= X"), NOT to board state
    // at cast time. Such abilities can target anything legal; the conditional effect is
    // enforced when they resolve, so cast-time legality must NOT gate on the condition.
    bool condition_on_target = false;
    // ConditionDefined$ Remembered — condition_present/condition_compare are evaluated over
    // cur_game.remembered_entities (count of remembered cards) rather than battlefield
    // permanents (Birthing Ritual: the dig only happens if a creature was sacrificed). Gated
    // at resolution in Ability::resolve(): on failure the body is skipped, subabilities chain.
    bool condition_on_remembered = false;
    // ConditionDefined$ TriggeredCard — condition_present is a property check on the ability's
    // SOURCE object (the card that triggered this ability), not a board-presence count. Used by
    // ConditionPresent$ Card.wasCastFromYourHandByYou (Amped Raptor: the dig only happens if the
    // creature that entered was cast from its controller's own hand). Gated at resolution in
    // Ability::resolve(): on failure the body is skipped, subabilities still chain.
    bool condition_on_triggered_card = false;

    // Intervening-if (rule 603.4) for a TRIGGERED ability: condition_present/condition_compare
    // are checked BOTH when the trigger would go on the stack (check_triggered_abilities) AND
    // again as it resolves; if false at resolution the ability does nothing (no subabilities).
    // Set from a trigger line's IsPresent$/PresentCompare$. Distinct from condition_present used
    // for spell castability, which is checked only at cast time.
    bool intervening_if = false;

    // IsCurse$ True (Carpet of Flowers' DB$ Pump): a Pump used purely as a targeting vehicle to
    // establish "target opponent" for a chained sub-ability — it applies no P/T and grants no
    // keyword. The Pump effect short-circuits (keeping ab.target = the chosen opponent player so
    // a sub-ability's TargetedPlayerCtrl count can read it) instead of re-entering creature target
    // selection. General over any IsCurse$ targeting-only Pump.
    bool is_curse = false;

    // DB$ StoreSVar (Carpet of Flowers): write `stored_svar_set_value` into the SOURCE permanent's
    // stored_svars[stored_svar_set_name]. The general per-permanent named-integer scratch store
    // (see Permanent::stored_svars). Parsed from SVar$ NAME / Expression$ N (Type$ Number).
    std::string stored_svar_set_name = "";
    int stored_svar_set_value = 0;

    // (delayed-trigger Phase$/Execute$/ValidPlayer$ moved to DelayedTriggerParams)

    //for each AB on a card script there may be multiple SubAbility$, would get parsed into vector below
    std::vector<Ability> subabilities; // additional abilities resolved at same time this resolves, stored in order

    // Gift (CR 702.176): the gift effect(s) of a spell cast with the Gift keyword (Into the
    // Flood Maw: DB$ Token making a tapped 1/1 Fish for the promised opponent). Copied onto the
    // resolving spell's primary ability at cast; if the gift was promised (Spell::gift_promised),
    // these resolve at the START of resolution — the opponent gets the gift "before its other
    // effects" (CR 702.176c) — and are skipped otherwise. Empty for a non-gift ability.
    std::vector<Ability> gift_abilities;

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
// reveal=true marks every offered card choice as public knowledge (revealed
// tutors), so observers may show the chosen card's name even into a hidden zone.
// cmc_bound (>= 0) plus cmc_op ("EQ"/"LE"/...) additionally gate candidate cards by
// mana value (Aether Vial: MV == charge-counter count, resolved by the caller); -1 = none.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
                   Zone::ZoneValue zone, const std::string& change_type,
                   bool mandatory = false,
                   Zone::ZoneValue destination = Zone::GRAVEYARD,
                   bool reveal = false,
                   int cmc_bound = -1, const std::string& cmc_op = "");

#endif /* ABILITY_H */
