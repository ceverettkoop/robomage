#ifndef TRIGGER_MODES_H
#define TRIGGER_MODES_H

#include <cstddef>
#include <string>
#include <vector>

#include "components/ability.h"
#include "components/zone.h"
#include "ecs/entity.h"
#include "ecs/event.h"

// A trigger line's params: what the parser read of a T: line (or an Effect's Triggers$ SVar) that
// the event binding depends on. The filters and gates that need no binding go straight onto the
// ability (read_trigger_line, parse.cpp).
struct TriggerLine {
    std::string mode;                 // Mode$
    std::vector<std::string> phases;  // Phase$ list (Mode$ Phase)
    int origin = -1;                  // Origin$ / Destination$ zone filter (Mode$ ChangesZone); -1 = any
    int destination = -1;
    bool player_is_you = false;       // the event's player is the source's controller (ValidPlayer$ You, ...)
    bool valid_card_creature = false;
    bool valid_card_self = false;
    bool valid_card_non_creature = false;
    bool valid_card_untapped = false;
    bool valid_card_opp_own = false;
    bool valid_card_opp_ctrl = false;
    int kicked_index = 0;             // ValidCard$ ...+kicked N — fires only if the Nth kicker was paid
    bool source_is_spell = false;     // ValidSource$ Spell...
    bool source_opp_ctrl = false;     // ValidSource$ ...OppCtrl
    bool source_creature_youctrl = false;  // ValidSource$ Creature.YouCtrl
    bool target_self = false;         // ValidTarget$ Card.Self
    bool is_static = false;           // Static$ True
    bool attacking_player_is_you = false;
    bool exclude_first_draw_step = false;
    bool optional = false;            // OptionalDecider$
    int resolved_limit = 0;           // ResolvedLimit$ N
    int turn_limit = 0;               // ActivationLimit$ N
    bool no_resolving_check = false;  // NoResolvingCheck$ True
    bool combat_damage_only = false;  // CombatDamage$ True
    bool from_graveyard = false;      // TriggerZones$ Graveyard
    size_t draw_number_eq = 0;        // Number$ N on a Mode$ Drawn trigger (Nth-draw gate)
    size_t cast_count_eq = 0;         // ActivatorThisTurnCast$ EQN
    // 603.4 intervening-if (IsPresent$/PresentCompare$ or a CheckSVar$ count gate).
    bool intervening_if = false;
    std::string condition_present;
    std::string condition_compare;
    std::string condition_compare_expr;
};

// One event met by one triggered ability of `source`, controlled by `controller`.
struct TriggerEvent {
    const AbilityDef &ab;
    Entity source;
    Zone::Ownership controller;
    const Event &ev;
};

// One row per trigger Mode$: a new mode is a row here. The binding the parser asks of it, what the
// trigger scan matches, and what --parse-report calls read come from the same row, so a mode can't
// be known to one and not the others.
struct TriggerMode {
    const char *name;  // Mode$
    // The event it fires on. 0 when its binder chooses among several (Phase) or it has none
    // (Always, a state trigger); a binder may choose another (SpellCast's noncreature event), or
    // none for a form that binds nothing.
    EventId event;
    // Writes the line's event and that mode's filters onto `ability`.
    void (*bind)(const TriggerMode &mode, const TriggerLine &t, AbilityDef &ability);
    // The params it acts on, space separated, beyond the ones every mode reads
    // (trigger_form_findings). Any other param on it reads as nothing.
    const char *reads;
    // Does its reader honour ValidCard$ `value`? `floating`: the trigger sits on an Effect. nullptr
    // when it reads none.
    bool (*card_read)(const std::string &value, bool floating);
    // Does its reader honour ValidSource$ `value`, and the one ValidTarget$ it reads.
    bool (*source_read)(const std::string &value);
    const char *valid_target;
    // Its filters on one event of its event type, past the gates every mode shares
    // (match_event_triggers): false when the ability doesn't trigger. nullptr when it has none.
    bool (*match)(const TriggerEvent &fired);
    // What it takes from the event onto the triggered ability. nullptr when it takes nothing.
    void (*capture)(Ability &ab, const Event &ev);
    // The event's ENTITY and PLAYER are the targeting object and its controller, not the subject
    // of the trigger, so the shared self and player gates don't read them.
    bool names_target;
};

// The row of Mode$ `name`, or nullptr when no event binding exists for it.
const TriggerMode *trigger_mode_named(const std::string &name);
// The row reading `event`, for the trigger scan, or nullptr when no row does. Rows sharing an event
// share what they read of it, and a trigger the engine builds has no Mode$, so it goes by the event.
const TriggerMode *trigger_mode_on(EventId event);
// Does `mode` act on the param `key`?
bool trigger_mode_reads_key(const TriggerMode &mode, const std::string &key);

#endif /* TRIGGER_MODES_H */
