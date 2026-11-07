#include "trigger_modes.h"

#include <algorithm>
#include <cstring>

#include "components/zone.h"
#include "ecs/events.h"
#include "parse.h"
#include "queries/filters.h"
#include "str_util.h"

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
        if (ability.trigger_on == 0) ability.trigger_on = pe.event;
        else if (ability.trigger_on != pe.event &&
                 std::find(ability.trigger_on_extra.begin(), ability.trigger_on_extra.end(),
                           pe.event) == ability.trigger_on_extra.end())
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
    return filter_reads_only(value, {"Card", "Permanent", "Colorless", "nonCreature", "Self", "YouCtrl", "cmc*", "kicked*"});
}

static bool card_read_drawn(const std::string &value, bool) {
    return filter_reads_only(value, {"Card", "OppOwn", "YouCtrl"});
}

static bool card_read_attacks(const std::string &value, bool floating) {
    return value == "Card.Self" || (floating && value == "Creature.OppCtrl");
}

static bool card_read_self(const std::string &value, bool) { return value == "Card.Self"; }

static bool card_read_creature(const std::string &value, bool) { return value == "Creature"; }

static bool source_read_self(const std::string &value) { return value == "Card.Self"; }

static bool source_read_your_creature(const std::string &value) { return value == "Creature.YouCtrl"; }

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


static const TriggerMode kTriggerModes[] = {
    {"ChangesZone", Events::CARD_CHANGED_ZONE, bind_changes_zone, "Origin Destination ValidCard ValidCards", card_read_move},
    {"ChangesZoneAll", Events::CARD_CHANGED_ZONE, bind_changes_zone, "Origin Destination ValidCard ValidCards", card_read_move},
    {"Phase", 0, bind_phase, "Phase ValidPlayer"},
    {"SpellCast", Events::SPELL_CAST, bind_spell_cast, "ValidCard ValidActivatingPlayer ValidSA ActivatorThisTurnCast",
     card_read_spell_cast},
    {"DamageDone", Events::COMBAT_DAMAGE_TO_PLAYER, bind_damage_done, "ValidSource ValidTarget CombatDamage", nullptr,
     source_read_self, "Player"},
    {"DamageAll", Events::COMBAT_DAMAGE_TO_PLAYER, bind_damage_all, "ValidSource ValidTarget CombatDamage", nullptr,
     source_read_your_creature, "Player"},
    {"Drawn", Events::PLAYER_DREW_CARD, bind_drawn, "ValidCard FirstCardInDrawStep Number", card_read_drawn},
    {"LandPlayed", Events::LAND_PLAYED, bind_land_played, "ValidCard Origin", card_read_move},
    {"Attacks", Events::CREATURE_ATTACKED, bind_attacks, "ValidCard Attacked", card_read_attacks},
    {"AttackersDeclared", Events::ATTACKERS_DECLARED, bind_attackers_declared, "AttackingPlayer"},
    {"TapsForMana", Events::TAPPED_FOR_MANA, bind_taps_for_mana, "ValidCard Activator", card_read_creature},
    {"BecomesTarget", Events::BECAME_TARGET, bind_becomes_target, "ValidSource ValidTarget", nullptr,
     source_read_targeting, "Card.Self"},
    {"BecomeMonstrous", Events::BECAME_MONSTROUS, bind_become_monstrous, "ValidCard", card_read_self},
    {"Always", 0, bind_always, ""},
};

const TriggerMode *trigger_mode_named(const std::string &name) {
    for (const TriggerMode &mode : kTriggerModes)
        if (name == mode.name) return &mode;
    return nullptr;
}

bool trigger_mode_reads_key(const TriggerMode &mode, const std::string &key) {
    const std::vector<std::string> keys = split(mode.reads, ' ');
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}
