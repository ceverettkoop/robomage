#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "ability_params.h"
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
    Entity source = 0;
    Entity target = 0;
    std::vector<Entity> targets;    // used when target_max > 1
    Zone::Ownership controller = Zone::PLAYER_A;  // set when pushed onto stack; stable even if source loses Permanent
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability
    std::vector<Colors> mana_choices;   // Produced$ Combo or Any — ordered list of selectable mana colors
    bool restrict_to_chosen_type_creature = false;  // RestrictValid$ Spell.Creature+ChosenType
    bool restrict_to_creature = false;               // RestrictValid$ Spell.Creature (any creature spell)
    bool restrict_to_colorless_eldrazi = false;      // RestrictValid$ Spell.Eldrazi+Colorless (Eldrazi Temple)
    bool adds_no_counter = false;                    // AddsNoCounter$ True — spell can't be countered

    // Set by apply_land_abilities for mana abilities generated from land subtypes;
    // cleared and regenerated each SBE pass when type-changing effects are active.
    bool subtype_derived = false;

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    int energy_cost = 0;                // PayEnergy<N> — energy ({E}) paid as part of the cost (CR 122.1c)
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string sac_cost_spec = "";     // Sac<1/Type;Type/> — type-based sac cost; empty = none
    // DB$ Sacrifice EFFECT (not a cost): sacrifice a chosen permanent you control matching
    // sac_valid (Birthing Ritual). Distinct from sac_self/sac_cost_spec which are activation costs.
    std::string sac_valid = "";         // SacValid$ — filter for the creature/permanent to sacrifice (e.g. "Creature")
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
    bool tap_on_etb = false;            // ETB$ True on a DB$ Tap — taps Defined$ Self as it enters the battlefield
    int activation_limit = 0;           // ActivationLimit$ N — max activations per turn (0 = unlimited)
    // Loyalty abilities (planeswalkers). is_loyalty_ability is the load-bearing flag;
    // loyalty_cost == 0 is still a valid loyalty ability (e.g. Jace "0:" Brainstorm), so
    // never infer loyalty-ness from loyalty_cost. Cost$ AddCounter<N/LOYALTY> → +N,
    // SubCounter<N/LOYALTY> → -N. Paid by modifying the source's own loyalty at activation.
    bool is_loyalty_ability = false;    // Planeswalker$ True
    int loyalty_cost = 0;               // +N (AddCounter) or -N (SubCounter); loyalty counters added/removed as the cost
    int activation_zone = -1;           // ActivationZone$ Hand → Zone::HAND; -1 = default (battlefield)
    int activations_this_turn = 0;      // runtime counter, reset at UNTAP
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
    bool trigger_self_excluded = false;  // true when ValidCard$ has .Other — won't trigger for the source itself
    bool trigger_only_self = false;      // true when ValidCard$ Card.Self — only fires when the entering entity is the source itself
    bool is_evoke_sacrifice = false;     // synthetic ETB self-trigger from K:Evoke — only fires when the permanent was evoked
    bool is_offspring_token = false;     // synthetic ETB self-trigger from K:Offspring — only fires when the permanent was cast with offspring; creates a 1/1 token copy
    bool trigger_valid_player_is_controller = false;  // true when ValidPlayer$ You
    bool mandatory = false;              // Mandatory$ True — player must choose; suppresses fail-to-find when zone non-empty
    bool may_shuffle = false;            // MayShuffle$ True — player may optionally shuffle after
    size_t unless_generic_cost = 0;      // UnlessCost$ N — target controller pays {N} to prevent counter
    bool unless_cost_is_life = false;    // when true, unless_generic_cost is paid as N life rather than {N} mana (Ward—Pay life, CR 702.21)
    // UnlessCost$ Discard<N/Card> (Reality Smasher: "counter ... unless its controller discards a
    // card"). The payer discards `unless_generic_cost` card(s) of their choice from hand to prevent
    // the counter; if they can't (or decline) the spell is countered. General "counter unless
    // discard" path (CR 701.8 discard); reuses the same run_unless_loop choice machinery.
    bool unless_cost_is_discard = false;
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
    bool animate_duration_permanent = false;

    // Filter naming which permanents a mass effect affects (DestroyAll / SacrificeAll /
    // PutCounterAll): the ValidCards$ spec, e.g. "Cat.YouCtrl" or
    // "Permanent.nonLand+OppCtrl+nonChosenCard". A plain member (not in the params
    // variant) so PutCounterAll can carry it alongside CounterParams.
    std::string valid_cards_filter = "";

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

    // Combat damage trigger (Barrowgoyf): damage amount stored at trigger fire time
    size_t trigger_damage_amount = 0;

    // Spell count trigger (Cori-Steel Cutter)
    size_t trigger_spell_count_eq = 0;  // ActivatorThisTurnCast$ EQN — fires on Nth spell

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

    // Cleanup sub-ability
    bool clear_remembered = false;   // ClearRemembered$ True
    bool clear_chosen = false;       // ClearChosenCard$ True — clears cur_game.chosen_cards

    // ChooseCard ChooseEach$ "Type & Type & ..." (Ajani -4): each affected player chooses
    // one permanent of each listed type from among their matching permanents to keep
    // (recorded in cur_game.chosen_cards). Empty = the legacy single-pick ChooseCard.
    std::string choose_each = "";

    // RememberChanged$ — remember entities moved by this ChangeZone (for Doomsday)
    bool remember_changed = false;

    // DB$ Effect — a transient continuous effect created by an ability (CR 611). The
    // StaticAbilities$ list names the continuous static(s) it grants; RememberObjects$ Self
    // means the effect tracks its own source. Kappa Cannoneer's "it can't be blocked this
    // turn" is StaticAbilities$ Unblockable + RememberObjects$ Self — applied as an
    // until-end-of-turn "can't be blocked" mark on the source rather than a stack object.
    std::string effect_static_ability = "";  // StaticAbilities$ value (e.g. "Unblockable")
    bool effect_remember_self = false;        // RememberObjects$ Self

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

    // (delayed-trigger Phase$/Execute$/ValidPlayer$ moved to DelayedTriggerParams)

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
