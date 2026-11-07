#include "trigger_modes.h"

#include <algorithm>
#include <array>
#include <cstring>

#include "components/carddata.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "ecs/events.h"
#include "mana_system.h"
#include "parse.h"
#include "queries/characteristics.h"
#include "queries/filters.h"
#include "queries/players.h"
#include "queries/types.h"
#include "str_util.h"
#include "svar_eval.h"
#include "systems/rules_modifying.h"

extern Coordinator global_coordinator;

static void bind_changes_zone(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_phase(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_spell_cast(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_damage_done(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_damage_all(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_drawn(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_land_played(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_attacks(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_attackers_declared(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_taps_for_mana(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_becomes_target(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_become_monstrous(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static void bind_always(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
static bool card_read_move(const std::string &value, bool floating);
static bool card_read_spell_cast(const std::string &value, bool floating);
static bool card_read_drawn(const std::string &value, bool floating);
static bool card_read_attacks(const std::string &value, bool floating);
static bool card_read_self(const std::string &value, bool floating);
static bool card_read_creature(const std::string &value, bool floating);
static bool source_read_self(const std::string &value);
static bool source_read_your_creature(const std::string &value);
static bool source_read_targeting(const std::string &value);
static bool filter_reads_only(const std::string &spec, const std::vector<std::string> &tokens);
static bool op_holds(int value, const std::string &op, int bound);
static bool zone_move_matches(
    const TriggerEvent &fired, Entity moved, Zone::ZoneValue origin, Zone::ZoneValue destination);
static bool match_changes_zone(const TriggerEvent &fired);
static bool match_land_played(const TriggerEvent &fired);
static bool match_spell_cast(const TriggerEvent &fired);
static bool match_drawn(const TriggerEvent &fired);
static bool match_becomes_target(const TriggerEvent &fired);
static bool match_attacks(const TriggerEvent &fired);
static bool match_combat_damage(const TriggerEvent &fired);
static void capture_spell_cast(Ability &ab, const Event &ev);
static void capture_combat_damage(Ability &ab, const Event &ev);
static void capture_attacks(Ability &ab, const Event &ev);
static void capture_becomes_target(Ability &ab, const Event &ev);

static void bind_changes_zone(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // All ChangesZone triggers use CARD_CHANGED_ZONE; origin/destination/type filters applied
    // at match time. ChangesZone fires once per matching card. ChangesZoneAll ("whenever one
    // or more cards ...") is a single batch trigger (CR 603.2c): it fires exactly ONCE for a
    // group of simultaneous zone changes, no matter how many cards matched, so the trigger
    // scan dedupes it to a single firing per batch (Moonshadow: milling 3 permanent cards
    // removes ONE -1/-1 counter, not three).
    ability.trigger_on = mode.event;
    ability.trigger_zone_origin = t.origin;
    ability.trigger_zone_destination = t.destination;
    // ValidCard$ (types, control, ownership, …) is matched against the moving object itself
    // through ability.trigger_valid_card, not against the event's player.
    ability.trigger_valid_card_untapped = t.valid_card_untapped;
    ability.trigger_batch_zone_all = (std::strcmp(mode.name, "ChangesZoneAll") == 0);
    if (t.valid_card_self) ability.trigger_only_self = true;
    // A Destination$ Battlefield trigger gated by IsPresent$ Card.Self (the source must
    // already be on the battlefield) is Forge's idiom for "Whenever ANOTHER permanent
    // enters" — the source's own entry must not satisfy it (Kappa Cannoneer's Oracle text
    // reads "another artifact you control"). Exclude the source from this trigger.
    if (t.destination == Zone::BATTLEFIELD && t.intervening_if && t.condition_present == "Card.Self")
        ability.trigger_self_excluded = true;
}

static void bind_damage_done(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever CARDNAME deals combat damage to a player" — Barrowgoyf
    if (t.combat_damage_only) {
        ability.trigger_on = mode.event;
        ability.trigger_only_self = true;  // ValidSource$ Card.Self
    }
}

static void bind_damage_all(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever one or more creatures you control deal combat damage to one or more players" —
    // Forth Eorlingas!'s floating monarch trigger (Mode$ DamageAll | ValidSource$ Creature.YouCtrl
    // | ValidTarget$ Player | CombatDamage$ True). Fires on COMBAT_DAMAGE_TO_PLAYER when the
    // damaging creature is controlled by this trigger's controller (matched at fire time, since
    // the floating trigger has no source permanent to self-reference).
    if (t.combat_damage_only) {
        ability.trigger_on = mode.event;
        ability.trigger_damage_source_youctrl = t.source_creature_youctrl;
        ability.trigger_batch_zone_all = true;
    }
}

static void bind_drawn(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "whenever a player draws a card" — Orcish Bowmasters (Mode$ Drawn)
    ability.trigger_on = mode.event;
    ability.trigger_valid_card_opp_own = t.valid_card_opp_own;
    ability.trigger_exclude_first_draw_step = t.exclude_first_draw_step;
    ability.trigger_draw_number_eq = t.draw_number_eq;
    // ValidCard$ Card.YouCtrl on a Drawn trigger ("whenever YOU draw ...", Tamiyo): the drawer
    // must be the source's controller. Reuse the controller-is-event-player gate (the
    // PLAYER_DREW_CARD event's PLAYER is the drawer), set by YouCtrl in the ValidCard parse.
    ability.trigger_valid_player_is_controller = t.player_is_you;
}

static void bind_land_played(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "When you play another land, ..." — City of Traitors (Mode$ LandPlayed | ValidCard$
    // Land.Other+YouCtrl). Fires when a land is PLAYED (CR 305.1), not when an effect puts one
    // onto the battlefield (CR 305.4). ValidCard$ is matched against the played land on the
    // battlefield through trigger_valid_card (its YouCtrl: the player who played it, CR
    // 305.2 / 110.2a); Other is the trigger_self_excluded identity gate, Self trigger_only_self.
    // Origin$ restricts the zone the land was played from.
    ability.trigger_on = mode.event;
    ability.trigger_zone_origin = t.origin;
    if (t.valid_card_self) ability.trigger_only_self = true;
}

static void bind_attacks(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever CARDNAME attacks, ..." — Phelia (Mode$ Attacks | ValidCard$ Card.Self). Fires
    // once for this creature each time it is declared as an attacker (CR 508.2). ValidCard$
    // Card.Self → trigger_only_self matches the attacking ENTITY against the source.
    ability.trigger_on = mode.event;
    if (t.valid_card_self) ability.trigger_only_self = true;
    // ValidCard$ Creature.OppCtrl | Attacked$ You,Planeswalker.YouCtrl (Tamiyo, Seasoned
    // Scholar's +2 hosted trigger): an opponent's creature attacking you/your planeswalker.
    // Matched at fire time against the attacker's controller (the trigger has no source perm).
    ability.trigger_attacker_opp_ctrl = t.valid_card_opp_ctrl;
}

static void bind_attackers_declared(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever you attack" — Guide of Souls (Mode$ AttackersDeclared | AttackingPlayer$ You).
    // Fires once per combat when the source's controller declares one or more attackers.
    ability.trigger_on = mode.event;
    ability.trigger_valid_player_is_controller = t.attacking_player_is_you;
}

static void bind_taps_for_mana(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever you tap a creature for mana, add an additional {G}." — Badgermole Cub
    // (Mode$ TapsForMana | ValidCard$ Creature | Activator$ You | Static$ True). A
    // mana-additional triggered ability resolved immediately by the mana system (off-stack,
    // CR 605.1a) rather than placed on the stack.
    ability.trigger_on = mode.event;
    ability.trigger_valid_card_is_creature = t.valid_card_creature;
    ability.trigger_valid_player_is_controller = t.player_is_you;
    ability.trigger_taps_for_mana_static = t.is_static;
}

static void bind_becomes_target(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "Whenever CARDNAME becomes the target of a spell an opponent controls, ..." — Reality
    // Smasher (Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl | ValidTarget$ Card.Self). Fires
    // when this permanent becomes the target of a matching spell (CR 603.2c). ValidTarget$
    // Card.Self reuses trigger_only_self (the targeted permanent must be the source).
    ability.trigger_on = mode.event;
    ability.trigger_source_must_be_spell = t.source_is_spell;
    ability.trigger_source_opp_ctrl = t.source_opp_ctrl;
    if (t.target_self) ability.trigger_only_self = true;
}

static void bind_become_monstrous(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // "When CARDNAME becomes monstrous, ..." — Mode$ BecomeMonstrous (CR 701.37). Fired by the
    // resolving Monstrosity$ ability (effect_put_counter.cpp) with ENTITY = the permanent that
    // became monstrous, so ValidCard$ Card.Self reuses the standard trigger_only_self ENTITY check.
    // TriggerZones$ Battlefield is the default functioning zone; no extra handling needed.
    ability.trigger_on = mode.event;
    if (t.valid_card_self) ability.trigger_only_self = true;
}

static void bind_always(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    // Mode$ Always — a state-triggered ability (CR 603.8). Its trigger condition is a game STATE
    // (the IsPresent$ intervening-if bind_trigger_line put into condition_present/intervening_if), not a
    // game event, so it has no trigger_on; the dedicated state-trigger scan in
    // collect_triggered_abilities evaluates the condition each SBA pass and fires once when it
    // becomes true. Dark Depths: IsPresent$ Card.Self+counters_EQ0_ICE ("when this has no ice
    // counters on it"). trigger_only_self is set so the source is the permanent whose counters
    // are checked. parse_triggered_abilities keeps this ability despite trigger_on == 0.
    ability.trigger_state_condition = true;
    ability.trigger_only_self = true;
}

// Mode$ Phase: "at the beginning of [your] <step>". The first listed phase (in the order below)
// binds trigger_on and any further one is an additional event the same trigger fires on (Carpet
// of Flowers' Phase$ Main1,Main2 — "each of your main phases").
static void bind_phase(const TriggerMode &, const TriggerLine &t, AbilityDef &ability) {
    static const struct {
            const char *phase;
            EventId event;
    } kPhaseEvents[] = {
        {"Upkeep", Events::UPKEEP_BEGAN},
        // Forge writes the end step as either "EndStep" or "End of Turn".
        {"EndStep", Events::END_STEP_BEGAN},
        {"End of Turn", Events::END_STEP_BEGAN},
        {"Draw", Events::DRAW_STEP_BEGAN},
        {"BeginCombat", Events::BEGIN_COMBAT_BEGAN},
        // Forge writes the (pre-combat) first main phase as "Main1", the post-combat one as "Main2".
        {"Main1", Events::FIRST_MAIN_BEGAN},
        {"Main2", Events::SECOND_MAIN_BEGAN},
        // With no ValidPlayer$ You a Cleanup trigger fires at every cleanup.
        {"Cleanup", Events::CLEANUP_BEGAN},
    };
    for (const std::string &phase : t.phases) {
        bool known = false;
        for (const auto &pe : kPhaseEvents) known = known || phase == pe.phase;
        if (!known) parse_finding("trigger_phase", phase, "no step event: the phase is ignored");
    }
    for (const auto &pe : kPhaseEvents) {
        if (std::find(t.phases.begin(), t.phases.end(), pe.phase) == t.phases.end()) continue;
        if (ability.trigger_on == 0)
            ability.trigger_on = pe.event;
        else if (ability.trigger_on != pe.event &&
                 std::find(ability.trigger_on_extra.begin(), ability.trigger_on_extra.end(), pe.event) ==
                     ability.trigger_on_extra.end())
            ability.trigger_on_extra.push_back(pe.event);
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
}

// Mode$ SpellCast: which cast event the trigger binds to, and its gates on the cast spell.
static void bind_spell_cast(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability) {
    if (t.valid_card_non_creature) {
        ability.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "Whenever you cast a colorless spell, ..." — Glaring Fleshraker
    // (Mode$ SpellCast | ValidCard$ Card.Colorless | ValidActivatingPlayer$ You). A plain
    // SpellCast with a colorless filter on the cast spell; matched at trigger time against the
    // spell's colorlessness (CR 105.2c). Keyed on the general Colorless tag, not this card.
    if (t.valid_card_colorless) {
        ability.trigger_on = mode.event;
        ability.trigger_valid_card_colorless = true;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "whenever you cast your Nth spell" — Cori-Steel Cutter; or "your Nth NONCREATURE spell each
    // turn" — The Fantasticar. Bind to SPELL_CAST (fired AFTER the per-cast spell counters bump,
    // unlike NONCREATURE_SPELL_CAST which fires before) so the count gate sees the current cast.
    if (t.cast_count_eq > 0) {
        ability.trigger_on = mode.event;
        ability.trigger_valid_player_is_controller = t.player_is_you;
        ability.trigger_spell_count_eq = t.cast_count_eq;
        if (t.valid_card_non_creature) {
            // Count only noncreature spells, and only fire on a noncreature cast (the SPELL_CAST
            // event carries every spell, so filter the triggering card to noncreature too).
            ability.trigger_valid_card_non_creature = true;
            ability.trigger_spell_count_noncreature = true;
        }
    }
    // "Whenever a player casts a spell with mana value equal to ..." — Chalice of the Void
    // (Mode$ SpellCast | ValidCard$ Card.cmcEQY | ValidActivatingPlayer$ Player). A dynamic
    // mana-value filter on any player's spell. The cmc match is checked at trigger time.
    if (!ability.trigger_cmc_expr.empty()) {
        ability.trigger_on = mode.event;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
    // "When you cast this spell, [if it was kicked with its [N] kicker,] ..." — Wastescape
    // Battlemage (Mode$ SpellCast | ValidCard$ Card.Self[+kicked N]). A linked self-cast
    // trigger that fires while the spell is on the stack (CR 702.33e/f). trigger_only_self
    // restricts it to the source spell; trigger_kicked_index (>0) additionally gates on the
    // Nth kicker having been paid. Handled by the dedicated self-cast SPELL_CAST scan.
    if (t.valid_card_self) {
        ability.trigger_on = mode.event;
        ability.trigger_only_self = true;
        ability.trigger_kicked_index = t.kicked_index;
    }
    // General "whenever you cast a spell, ..." — Paradox Engine
    // (Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ You). A plain, unfiltered
    // SpellCast trigger that fires on EVERY spell the source's controller casts. None of the
    // specialized SpellCast bindings above (noncreature/colorless/count/cmc/self) applied, so
    // bind to SPELL_CAST and gate on the caster being this source's controller. The general
    // battlefield trigger scan fires it (no extra ValidCard$ filter ⇒ any spell). Keyed on the
    // bare SpellCast mode, not this card.
    if (ability.trigger_on == 0) {
        ability.trigger_on = mode.event;
        ability.trigger_valid_player_is_controller = t.player_is_you;
    }
}

// The move modes match ValidCard$ whole against the moving object (zone_change_object_matches), but
// read Self and Other off the whole list; the others read only the tokens their binding names.
static bool card_read_move(const std::string &value, bool) {
    return value.find(',') == std::string::npos ||
           (!filter_names_token(value, "Self") && !filter_names_token(value, "Other"));
}

static bool card_read_spell_cast(const std::string &value, bool) {
    // bind_spell_cast binds one gate: a noncreature spell, or a colour or mana value.
    if (filter_names_token(value, "nonCreature") &&
        (filter_names_token(value, "Colorless") || value.find("cmc") != std::string::npos))
        return false;
    return filter_reads_only(
        value, {"Card", "Permanent", "Colorless", "nonCreature", "Self", "YouCtrl", "cmc*", "kicked*"});
}

static bool card_read_drawn(const std::string &value, bool) {
    return filter_reads_only(value, {"Card", "OppOwn", "YouCtrl"});
}

static bool card_read_attacks(const std::string &value, bool floating) {
    return value == "Card.Self" || (floating && value == "Creature.OppCtrl");
}

static bool card_read_self(const std::string &value, bool) {
    return value == "Card.Self";
}

static bool card_read_creature(const std::string &value, bool) {
    return value == "Creature";
}

static bool source_read_self(const std::string &value) {
    return value == "Card.Self";
}

static bool source_read_your_creature(const std::string &value) {
    return value == "Creature.YouCtrl";
}

static bool source_read_targeting(const std::string &value) {
    return filter_reads_only(value, {"Spell", "Card", "OppCtrl"});
}

// Does the filter spec name nothing but `tokens`, in a single alternative? A token ending in '*'
// stands for any token with that prefix.
static bool filter_reads_only(const std::string &spec, const std::vector<std::string> &tokens) {
    if (spec.find_first_of(",;") != std::string::npos) return false;
    size_t p = 0;
    while (p <= spec.size()) {
        const size_t nx = std::min(spec.find_first_of(".+", p), spec.size());
        const std::string tok = spec.substr(p, nx - p);
        p = nx + 1;
        bool known = false;
        for (const std::string &t : tokens)
            known = known || tok == t || (t.back() == '*' && tok.rfind(t.substr(0, t.size() - 1), 0) == 0);
        if (!known) return false;
    }
    return true;
}

// `value` against `bound` by a Forge comparator (EQ, NE, LE, GE, LT, GT); any other reads as EQ.
static bool op_holds(int value, const std::string &op, int bound) {
    return (op == "LE")   ? value <= bound
           : (op == "GE") ? value >= bound
           : (op == "LT") ? value < bound
           : (op == "GT") ? value > bound
           : (op == "NE") ? value != bound
                          : value == bound;
}

// A zone-change trigger's filters against one move of object `moved` (0 = none named) from
// `origin` to `destination`: its Origin$ / Destination$ zones and its ValidCard$, matched with
// the source's controller as "you" (zone_change_object_matches: the object as it exists after the
// move, or as it last existed on the battlefield for a departure — CR 603.6a / 603.10a).
static bool zone_move_matches(
    const TriggerEvent &fired, Entity moved, Zone::ZoneValue origin, Zone::ZoneValue destination) {
    const AbilityDef &ab = fired.ab;
    if (ab.trigger_zone_origin >= 0 && origin != static_cast<Zone::ZoneValue>(ab.trigger_zone_origin)) return false;
    if (ab.trigger_zone_destination >= 0 && destination != static_cast<Zone::ZoneValue>(ab.trigger_zone_destination))
        return false;
    if (ab.trigger_valid_card.empty() || moved == 0) return true;
    MatchCtx vctx;
    vctx.controller = fired.controller;
    vctx.source = fired.source;
    return zone_change_object_matches(moved, origin, destination, ab.trigger_valid_card, vctx);
}

// CARD_CHANGED_ZONE: origin, destination and the ValidCard$ filter (Guide of Souls'
// Creature.Other+YouCtrl, Kappa Cannoneer's Artifact.YouCtrl, landfall's Land.YouCtrl, Ajani's
// Cat.Other+YouCtrl, Moonshadow's Permanent.YouOwn+!token, Murktide's Instant/Sorcery) on the moving
// object — its live types (an animated land is a creature, a Clue token isn't, an
// entering-transformed face is its back face) and its controller (a creature reanimated from an
// opponent's graveyard is YouCtrl for the reanimator).
static bool match_changes_zone(const TriggerEvent &fired) {
    const Event &ev = fired.ev;
    const Entity moved = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
    const Zone::ZoneValue destination = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
    // DisableTriggers check (Doorkeeper Thrull): suppress ETB triggers caused by matching card types
    if (destination == Zone::BATTLEFIELD && rules_mod::etb_triggers_suppressed(moved)) return false;
    return zone_move_matches(fired, moved, ev.GetParam<Zone::ZoneValue>(Params::ORIGIN), destination);
}

// LAND_PLAYED (City of Traitors' Land.Other+YouCtrl): the played land's move from the zone it was
// played from onto the battlefield.
static bool match_land_played(const TriggerEvent &fired) {
    const Event &ev = fired.ev;
    return zone_move_matches(
        fired, ev.GetParam<Entity>(Params::ENTITY), ev.GetParam<Zone::ZoneValue>(Params::ORIGIN), Zone::BATTLEFIELD);
}

// SPELL_CAST: the gates on the spell cast and on how many have been cast.
static bool match_spell_cast(const TriggerEvent &fired) {
    const AbilityDef &ab = fired.ab;
    const Event &ev = fired.ev;
    const bool has_spell = ev.HasParam(Params::ENTITY);
    const Entity spell = has_spell ? ev.GetParam<Entity>(Params::ENTITY) : 0;
    // ValidCard$ Card.nonCreature (The Fantasticar): the cast spell must NOT be a creature spell.
    if (ab.trigger_valid_card_non_creature && has_spell && type_set_has(effective_types(spell), "Creature"))
        return false;
    // ValidCard$ Permanent, ValidSA$ Spell.ManaFromCard.StrictlySelf (The Myriad Pools): a permanent
    // spell, cast with mana this source made.
    if (has_spell) {
        if (ab.trigger_valid_card_permanent && !is_permanent_type_line(effective_types(spell))) return false;
        if (ab.trigger_mana_from_self &&
            (!global_coordinator.entity_has_component<Spell>(spell) ||
                mana_from(global_coordinator.GetComponent<Spell>(spell).chosen.mana_paid, fired.source) == 0))
            return false;
    }
    // Colorless filter (Glaring Fleshraker): the cast spell must be colorless (CR 105.2c).
    if (ab.trigger_valid_card_colorless && has_spell && !is_colorless(spell)) return false;
    // Spell count filter (Cori-Steel Cutter)
    if (ab.trigger_spell_count_eq > 0 && ev.HasParam(Params::PLAYER)) {
        Entity ev_player = ev.GetParam<Entity>(Params::PLAYER);
        if (!global_coordinator.entity_has_component<Player>(ev_player)) return false;
        auto &pl = global_coordinator.GetComponent<Player>(ev_player);
        // The Fantasticar counts only noncreature spells; Cori-Steel Cutter counts all.
        size_t cast_count =
            ab.trigger_spell_count_noncreature ? pl.noncreature_spells_cast_this_turn : pl.spells_cast_this_turn;
        if (cast_count != ab.trigger_spell_count_eq) return false;
    }
    // Dynamic mana-value filter on the cast spell (Chalice of the Void: ValidCard$ Card.cmcEQY,
    // Y = Count$CardCounters.CHARGE). Compare the cast spell's mana value to the count resolved
    // against this source permanent.
    if (!ab.trigger_cmc_expr.empty()) {
        if (!has_spell || !global_coordinator.entity_has_component<CardData>(spell)) return false;
        const int spell_mv = object_mana_value(spell, global_coordinator.GetComponent<CardData>(spell));
        const int bound = evaluate_svar(ab.trigger_cmc_expr, fired.controller, fired.source);
        if (!op_holds(spell_mv, ab.trigger_cmc_op, bound)) return false;
    }
    // ValidSA$ Spell.ManaSpent <op><n> filter (Roiling Vortex: "if no mana was spent to cast that
    // spell" = ManaSpent EQ0). Compare the cast spell's recorded ChosenCost::mana_spent (CR
    // 106/601.2g) to the trigger's bound. The spell is still on the stack when SPELL_CAST fires, so
    // its Spell component is present.
    if (!ab.trigger_mana_spent_op.empty()) {
        if (!has_spell || !global_coordinator.entity_has_component<Spell>(spell)) return false;
        const int spent = global_coordinator.GetComponent<Spell>(spell).chosen.mana_spent();
        if (!op_holds(spent, ab.trigger_mana_spent_op, ab.trigger_mana_spent_val)) return false;
    }
    return true;
}

// PLAYER_DREW_CARD (Orcish Bowmasters).
static bool match_drawn(const TriggerEvent &fired) {
    const AbilityDef &ab = fired.ab;
    const Event &ev = fired.ev;
    // ValidCard$ Card.OppOwn — the drawn card must be owned by an opponent of the source's
    // controller (drawer != controller).
    if (ab.trigger_valid_card_opp_own && ev.HasParam(Params::PLAYER) &&
        ev.GetParam<Entity>(Params::PLAYER) == get_player_entity(fired.controller))
        return false;
    // FirstCardInDrawStep$ False — ignore the first card drawn in the drawer's draw step (the
    // turn-based draw).
    if (ab.trigger_exclude_first_draw_step && ev.HasParam(Params::FIRST_IN_STEP) &&
        ev.GetParam<int>(Params::FIRST_IN_STEP) == 1)
        return false;
    // Number$ N (Tamiyo, Inquisitive Student: "your THIRD card in a turn") — fire only when this
    // draw is the drawer's Nth this turn. AMOUNT is the per-draw running ordinal stamped on the
    // event (1-based), so exactly the Nth draw fires.
    if (ab.trigger_draw_number_eq > 0 &&
        (!ev.HasParam(Params::AMOUNT) || ev.GetParam<uint32_t>(Params::AMOUNT) != ab.trigger_draw_number_eq))
        return false;
    return true;
}

// BECAME_TARGET (Reality Smasher): the trigger fires only for the permanent that became a target
// and only when the targeting object is a spell controlled by an opponent (ValidSource$
// Spell.OppCtrl).
static bool match_becomes_target(const TriggerEvent &fired) {
    const AbilityDef &ab = fired.ab;
    const Event &ev = fired.ev;
    // ValidTarget$ Card.Self: the permanent that became a target must be this one.
    const Entity targeted = ev.HasParam(Params::TARGET) ? ev.GetParam<Entity>(Params::TARGET) : 0;
    if (ab.trigger_only_self && targeted != fired.source) return false;
    const Entity targeting = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
    // ValidSource$ Spell — the targeting object must be a spell on the stack.
    if (ab.trigger_source_must_be_spell && !global_coordinator.entity_has_component<Spell>(targeting)) return false;
    // ValidSource$ ...OppCtrl — controlled by an opponent of this source's controller.
    if (ab.trigger_source_opp_ctrl && ev.HasParam(Params::PLAYER) &&
        ev.GetParam<Entity>(Params::PLAYER) == get_player_entity(fired.controller))
        return false;
    return true;
}

// CREATURE_ATTACKED: ValidCard$ Creature.OppCtrl (Tamiyo, Seasoned Scholar's +2), the attacker
// controlled by an opponent of the trigger's controller.
static bool match_attacks(const TriggerEvent &fired) {
    return !fired.ab.trigger_attacker_opp_ctrl || !fired.ev.HasParam(Params::ENTITY) ||
           source_controller(fired.ev.GetParam<Entity>(Params::ENTITY)) != fired.controller;
}

// COMBAT_DAMAGE_TO_PLAYER: ValidSource$ Creature.YouCtrl (Forth Eorlingas!), the damaging creature
// controlled by the trigger's controller.
static bool match_combat_damage(const TriggerEvent &fired) {
    if (!fired.ab.trigger_damage_source_youctrl) return true;
    return fired.ev.HasParam(Params::ENTITY) &&
           source_controller(fired.ev.GetParam<Entity>(Params::ENTITY)) == fired.controller;
}

static void capture_spell_cast(Ability &ab, const Event &ev) {
    if (ev.HasParam(Params::ENTITY)) ab.cast_spell = ObjectRef::of(ev.GetParam<Entity>(Params::ENTITY));
}

// For combat damage triggers, capture the damage amount.
static void capture_combat_damage(Ability &ab, const Event &ev) {
    if (ev.HasParam(Params::AMOUNT)) ab.trigger_damage_amount = ev.GetParam<uint32_t>(Params::AMOUNT);
}

// Defined$ TriggeredDefendingPlayer — bind the defending player of the attack (Goblin Guide's Dig
// acts on the DEFENDER's library). CREATURE_ATTACKED carries PLAYER = the attacker's controller
// (active player); in a two-player game the defender is that player's opponent. Bind it as the
// ability's target (a player entity), which the Dig handler reads as the library owner.
// Defined$ TriggeredAttacker(LKICopy) binds the attacker the same way, for the effect's target
// (Tamiyo, Seasoned Scholar's +2).
static void capture_attacks(Ability &ab, const Event &ev) {
    if (ab.def->defined.is(ObjectSpec::TRIGGERED_ATTACKER) && ev.HasParam(Params::ENTITY))
        ab.target = ObjectRef::of(ev.GetParam<Entity>(Params::ENTITY));
    if (!ab.def->defined.is(ObjectSpec::TRIGGERED_DEFENDING_PLAYER) || !ev.HasParam(Params::PLAYER)) return;
    const Entity attacker_player = ev.GetParam<Entity>(Params::PLAYER);
    ab.target = ObjectRef::of(get_player_entity(opponent_of(seat_of_player(attacker_player))));
}

// Defined$ TriggeredSourceSA — the Counter effect acts on the spell that targeted this permanent.
// Bind it as the ability's target from the event's ENTITY (the targeting object). UnlessPayer$
// TriggeredSourceSAController binds the payer of the unless-cost to that spell's controller (the
// opponent), captured from PLAYER.
static void capture_becomes_target(Ability &ab, const Event &ev) {
    if (ab.def->defined.is(ObjectSpec::TRIGGERED_SOURCE_SA) && ev.HasParam(Params::ENTITY))
        ab.target = ObjectRef::of(ev.GetParam<Entity>(Params::ENTITY));
    if (ab.def->unless_payer_is_triggered_source_sa_ctrl && ev.HasParam(Params::PLAYER))
        ab.unless_payer = seat_of_player(ev.GetParam<Entity>(Params::PLAYER));
}

static const TriggerMode kTriggerModes[] = {
    {"ChangesZone", Events::CARD_CHANGED_ZONE, bind_changes_zone, "Origin Destination ValidCard ValidCards",
        card_read_move, nullptr, nullptr, match_changes_zone},
    {"ChangesZoneAll", Events::CARD_CHANGED_ZONE, bind_changes_zone, "Origin Destination ValidCard ValidCards",
        card_read_move, nullptr, nullptr, match_changes_zone},
    {"Phase", 0, bind_phase, "Phase ValidPlayer"},
    {"SpellCast", Events::SPELL_CAST, bind_spell_cast, "ValidCard ValidActivatingPlayer ValidSA ActivatorThisTurnCast",
        card_read_spell_cast, nullptr, nullptr, match_spell_cast, capture_spell_cast},
    {"DamageDone", Events::COMBAT_DAMAGE_TO_PLAYER, bind_damage_done, "ValidSource ValidTarget CombatDamage", nullptr,
        source_read_self, "Player", match_combat_damage, capture_combat_damage},
    {"DamageAll", Events::COMBAT_DAMAGE_TO_PLAYER, bind_damage_all, "ValidSource ValidTarget CombatDamage", nullptr,
        source_read_your_creature, "Player", match_combat_damage, capture_combat_damage},
    {"Drawn", Events::PLAYER_DREW_CARD, bind_drawn, "ValidCard FirstCardInDrawStep Number", card_read_drawn, nullptr,
        nullptr, match_drawn},
    {"LandPlayed", Events::LAND_PLAYED, bind_land_played, "ValidCard Origin", card_read_move, nullptr, nullptr,
        match_land_played},
    {"Attacks", Events::CREATURE_ATTACKED, bind_attacks, "ValidCard Attacked", card_read_attacks, nullptr, nullptr,
        match_attacks, capture_attacks},
    {"AttackersDeclared", Events::ATTACKERS_DECLARED, bind_attackers_declared, "AttackingPlayer"},
    {"TapsForMana", Events::TAPPED_FOR_MANA, bind_taps_for_mana, "ValidCard Activator", card_read_creature},
    {"BecomesTarget", Events::BECAME_TARGET, bind_becomes_target, "ValidSource ValidTarget", nullptr,
        source_read_targeting, "Card.Self", match_becomes_target, capture_becomes_target, true},
    {"BecomeMonstrous", Events::BECAME_MONSTROUS, bind_become_monstrous, "ValidCard", card_read_self},
    {"Always", 0, bind_always, ""},
};

const TriggerMode *trigger_mode_named(const std::string &name) {
    for (const TriggerMode &mode : kTriggerModes)
        if (name == mode.name) return &mode;
    return nullptr;
}

const TriggerMode *trigger_mode_on(EventId event) {
    static const auto by_event = [] {
        std::array<const TriggerMode *, 64> rows{};
        for (const TriggerMode &mode : kTriggerModes)
            if (mode.event != 0 && !rows[mode.event]) rows[mode.event] = &mode;
        return rows;
    }();
    return event < by_event.size() ? by_event[event] : nullptr;
}

bool trigger_mode_reads_key(const TriggerMode &mode, const std::string &key) {
    const std::vector<std::string> keys = split(mode.reads, ' ');
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}
