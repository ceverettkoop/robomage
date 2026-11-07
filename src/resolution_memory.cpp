#include "resolution_memory.h"

#include <algorithm>
#include <vector>

#include "classes/game.h"
#include "components/ability.h"
#include "components/permanent.h"
#include "coverage.h"
#include "ecs/coordinator.h"
#include "queries/affected.h"

extern Game cur_game;

// Which values of a row's param add its op.
enum class ParamForm {
    NONE,            // no param: the effect kind makes the write
    IS_TRUE,         // True
    NAMES_TARGETED,  // any value naming Targeted (RememberObjects$ Targeted, ThisTargetedCard)
    NAMES_OBJECTS,   // a Defined$ name, which the op takes as its objects
};

// One memory write: asked for by `param`, or with a null param made by every effect of `kind`.
struct MemoryRow {
        const char *param;
        ParamForm form;
        EffectKind kind;
        MemoryOp op;
};

// Every memory write of an effect. A target is remembered as the ability starts to resolve, for
// its own effect to read (a DelayedTrigger's watched object); the rest as the effect finishes.
static const MemoryRow kMemoryRows[] = {
    {"RememberTargets", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TARGETS}},
    {"RememberObjects", ParamForm::NAMES_TARGETED, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TARGETS}},
    {"RememberChanged", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::MOVED}},
    {"RememberLKI", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::MOVED_LKI}},
    {"RememberSacrificed", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::SACRIFICED}},
    {"RememberDiscarded", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::DISCARDED}},
    {"RememberMilled", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::MILLED}},
    {"RememberRevealed", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::REVEALED_HAND}},
    {"RememberPumped", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::PUMPED}},
    {"RememberFound", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::FOUND}},
    {"RememberChosen", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::CHOSEN | EffectObjects::CHOSEN_IMPRINTED}},
    {"Imprint", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::APPEND, EffectObjects::MOVED}},
    {"ImprintCards", ParamForm::NAMES_TARGETED, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::APPEND, EffectObjects::TARGETS}},
    {"ImprintRevealed", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::REPLACE, EffectObjects::REVEALED_TOP}},
    {"ImprintPlayed", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::APPEND, EffectObjects::PLAYED}},
    {"ClearRemembered", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::CLEANED}},
    {"ClearChosenCard", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::CHOSEN_CARDS, MemoryAction::REPLACE, EffectObjects::CLEANED}},
    {"ClearImprinted", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::REPLACE, EffectObjects::CLEANED}},
    {"ClearNamedCard", ParamForm::IS_TRUE, EffectKind::None,
        {MemorySlot::NAMED_CARD, MemoryAction::REPLACE, EffectObjects::CLEANED}},
    {"ForgetObjects", ParamForm::NAMES_OBJECTS, EffectKind::None,
        {MemorySlot::REMEMBERED, MemoryAction::FORGET, EffectObjects::FINISHED}},
    {"ForgetImprinted", ParamForm::NAMES_OBJECTS, EffectKind::None,
        {MemorySlot::IMPRINTED, MemoryAction::FORGET, EffectObjects::FINISHED}},
    {nullptr, ParamForm::NONE, EffectKind::Token,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TOKENS}},
    {nullptr, ParamForm::NONE, EffectKind::Investigate,
        {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TOKENS}},
    {nullptr, ParamForm::NONE, EffectKind::Vote, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::VOTED}},
    {nullptr, ParamForm::NONE, EffectKind::ChooseCard,
        {MemorySlot::CHOSEN_CARDS, MemoryAction::REPLACE, EffectObjects::CHOSEN}},
    {nullptr, ParamForm::NONE, EffectKind::ChooseCard,
        {MemorySlot::CHOSEN_CARDS, MemoryAction::APPEND, EffectObjects::KEPT}},
};

// Adds `op` to `def`, into the op it shares a slot and an action with if there is one.
static void add_memory_op(AbilityDef &def, const MemoryOp &op);
// Does `form` take `value`?
static bool form_takes(ParamForm form, const std::string &value);
// Does a row of `def` read any of `sources`?
static bool reads_report(const AbilityDef &def, uint32_t sources);
// The Permanent of `ab`'s source, while it is that object on the battlefield; else null. What it
// imprints lasts on it past the resolution.
static Permanent *imprinting_permanent(const Ability &ab);
// Empties `slot`.
static void clear_slot(MemorySlot slot, const Ability &ab);
// Adds `ref` to `slot`.
static void add_to_slot(MemorySlot slot, const ObjectRef &ref, const Ability &ab);
// Takes `e` out of `slot`.
static void forget_from_slot(MemorySlot slot, Entity e, const Ability &ab);

static void add_memory_op(AbilityDef &def, const MemoryOp &op) {
    for (MemoryOp &held : def.memory_ops) {
        if (held.slot != op.slot || held.action != op.action || held.objects != op.objects) continue;
        held.sources |= op.sources;
        return;
    }
    def.memory_ops.push_back(op);
}

static bool form_takes(ParamForm form, const std::string &value) {
    switch (form) {
        case ParamForm::NONE:
            return false;
        case ParamForm::IS_TRUE:
            return value == "True";
        case ParamForm::NAMES_TARGETED:
            return value.find("Targeted") != std::string::npos;
        case ParamForm::NAMES_OBJECTS:
            return !parse_object_spec(value).is(ObjectSpec::UNREAD);
    }
    return false;
}

static bool reads_report(const AbilityDef &def, uint32_t sources) {
    for (const MemoryOp &op : def.memory_ops)
        if (op.sources & sources) return true;
    return false;
}

static Permanent *imprinting_permanent(const Ability &ab) {
    const Entity source = ab.source.get();
    if (source == 0 || !global_coordinator.entity_has_component<Permanent>(source)) return nullptr;
    return &global_coordinator.GetComponent<Permanent>(source);
}

static void clear_slot(MemorySlot slot, const Ability &ab) {
    switch (slot) {
        case MemorySlot::REMEMBERED:
            cur_game.resolution.memory.remembered.clear();
            break;
        case MemorySlot::IMPRINTED:
            cur_game.resolution.memory.imprinted.clear();
            if (Permanent *host = imprinting_permanent(ab)) host->imprinted.clear();
            break;
        case MemorySlot::CHOSEN_CARDS:
            cur_game.resolution.memory.chosen_cards.clear();
            break;
        case MemorySlot::NAMED_CARD:
            cur_game.resolution.memory.named_card.clear();
            break;
    }
}

static void add_to_slot(MemorySlot slot, const ObjectRef &ref, const Ability &ab) {
    switch (slot) {
        case MemorySlot::REMEMBERED:
            cur_game.resolution.memory.remembered.push_back(ref);
            break;
        case MemorySlot::IMPRINTED:
            cur_game.resolution.memory.imprinted.push_back(ref);
            if (Permanent *host = imprinting_permanent(ab)) {
                host->imprinted.push_back(ref);
                coverage_row("imprint_on_source");
            }
            break;
        case MemorySlot::CHOSEN_CARDS:
            cur_game.resolution.memory.chosen_cards.insert(ref);
            break;
        case MemorySlot::NAMED_CARD:
            break;
    }
}

static void forget_from_slot(MemorySlot slot, Entity e, const Ability &ab) {
    if (slot == MemorySlot::REMEMBERED) {
        forget_remembered(e);
        return;
    }
    if (slot != MemorySlot::IMPRINTED) return;
    const auto names_e = [e](const ObjectRef &ref) { return ref.lki_entity() == e; };
    std::vector<ObjectRef> &held = cur_game.resolution.memory.imprinted;
    held.erase(std::remove_if(held.begin(), held.end(), names_e), held.end());
    if (Permanent *host = imprinting_permanent(ab)) {
        host->imprinted.erase(std::remove_if(host->imprinted.begin(), host->imprinted.end(), names_e),
                              host->imprinted.end());
        coverage_row("forget_imprinted");
    }
}

bool read_memory_param(AbilityDef &def, const std::string &key, const std::string &value) {
    bool known = false;
    for (const MemoryRow &row : kMemoryRows) {
        if (!row.param || key != row.param) continue;
        // A name with no row stays unread, for --parse-report to say so.
        if (row.form == ParamForm::NAMES_OBJECTS && !form_takes(row.form, value)) continue;
        known = true;
        if (!form_takes(row.form, value)) continue;
        MemoryOp op = row.op;
        if (row.form == ParamForm::NAMES_OBJECTS) op.objects = parse_object_spec(value);
        add_memory_op(def, op);
    }
    return known;
}

void bind_memory_ops(AbilityDef &def) {
    for (const MemoryRow &row : kMemoryRows)
        if (!row.param && row.kind == def.kind) add_memory_op(def, row.op);
}

void report_effect(Ability &ab, uint32_t sources) {
    if (reads_report(*ab.def, sources)) ab.report.reported |= sources;
}

void report_object(Ability &ab, uint32_t sources, Entity e) {
    if (!reads_report(*ab.def, sources)) return;
    ab.report.reported |= sources;
    ab.report.objects.push_back({sources, ObjectRef::of(e)});
}

void report_targets(Ability &ab) {
    report_effect(ab, EffectObjects::TARGETS);
    if (!ab.targets.empty()) {
        for (const ObjectRef &t : ab.targets)
            if (Entity te = t.get()) report_object(ab, EffectObjects::TARGETS, te);
    } else if (Entity te = ab.target.get()) {
        report_object(ab, EffectObjects::TARGETS, te);
    }
}

bool reported_object(const Ability &ab, Entity e) {
    if (e == 0) return false;
    for (const ReportedObject &object : ab.report.objects)
        if (object.ref.e == e && object.ref.get() == e) return true;
    return false;
}

void forget_remembered(Entity e) {
    std::vector<ObjectRef> &remembered = cur_game.resolution.memory.remembered;
    remembered.erase(std::remove_if(remembered.begin(), remembered.end(),
                                    [e](const ObjectRef &ref) { return ref.get() == e; }),
                     remembered.end());
}

void apply_memory_ops(Ability &ab) {
    if (ab.report.reported == 0) return;
    for (const MemoryOp &op : ab.def->memory_ops) {
        if (!(op.sources & ab.report.reported)) continue;
        if (op.action == MemoryAction::FORGET) {
            for (Entity e : resolve_objects(op.objects, ResolveCtx::of(ab))) forget_from_slot(op.slot, e, ab);
            continue;
        }
        if (op.action == MemoryAction::REPLACE) clear_slot(op.slot, ab);
        for (const ReportedObject &object : ab.report.objects)
            if (object.sources & op.sources) add_to_slot(op.slot, object.ref, ab);
    }
    ab.report = EffectReport{};
}
