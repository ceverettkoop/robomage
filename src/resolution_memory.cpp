#include "resolution_memory.h"

#include <vector>

#include "classes/game.h"
#include "components/ability.h"

extern Game cur_game;

// Which values of a row's param add its op.
enum class ParamForm {
    IS_TRUE,         // True
    NAMES_TARGETED,  // any value naming Targeted (RememberObjects$ Targeted, ThisTargetedCard)
};

struct MemoryRow {
    const char *param;
    ParamForm form;
    MemoryOp op;
};

// Every memory write an effect's params ask for.
static const MemoryRow kMemoryRows[] = {
    {"RememberTargets", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TARGETS}},
    {"RememberObjects", ParamForm::NAMES_TARGETED, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::TARGETS}},
    {"RememberChanged", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::MOVED}},
    {"RememberLKI", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::MOVED_LKI}},
    {"RememberSacrificed", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::SACRIFICED}},
    {"RememberMilled", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::MILLED}},
    {"RememberRevealed", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::REVEALED_HAND}},
    {"RememberPumped", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::APPEND, EffectObjects::PUMPED}},
    {"RememberFound", ParamForm::IS_TRUE, {MemorySlot::REMEMBERED, MemoryAction::REPLACE, EffectObjects::FOUND}},
    {"ImprintRevealed", ParamForm::IS_TRUE, {MemorySlot::IMPRINTED, MemoryAction::REPLACE, EffectObjects::REVEALED_TOP}},
};

// Adds `op` to `def`, into the op it shares a slot and an action with if there is one.
static void add_memory_op(AbilityDef &def, const MemoryOp &op);
// Does `form` take `value`?
static bool form_takes(ParamForm form, const std::string &value);
// Does a row of `def` read any of `sources`?
static bool reads_report(const AbilityDef &def, uint32_t sources);
// Empties `slot`.
static void clear_slot(MemorySlot slot);
// Adds `ref` to `slot`.
static void add_to_slot(MemorySlot slot, const ObjectRef &ref);

static void add_memory_op(AbilityDef &def, const MemoryOp &op) {
    for (MemoryOp &held : def.memory_ops) {
        if (held.slot != op.slot || held.action != op.action) continue;
        held.sources |= op.sources;
        return;
    }
    def.memory_ops.push_back(op);
}

static bool form_takes(ParamForm form, const std::string &value) {
    switch (form) {
        case ParamForm::IS_TRUE: return value == "True";
        case ParamForm::NAMES_TARGETED: return value.find("Targeted") != std::string::npos;
    }
    return false;
}

static bool reads_report(const AbilityDef &def, uint32_t sources) {
    for (const MemoryOp &op : def.memory_ops)
        if (op.sources & sources) return true;
    return false;
}

static void clear_slot(MemorySlot slot) {
    switch (slot) {
        case MemorySlot::REMEMBERED: cur_game.resolution.memory.remembered.clear(); break;
        case MemorySlot::IMPRINTED: cur_game.resolution.memory.imprinted.clear(); break;
    }
}

static void add_to_slot(MemorySlot slot, const ObjectRef &ref) {
    switch (slot) {
        case MemorySlot::REMEMBERED: cur_game.resolution.memory.remembered.push_back(ref); break;
        case MemorySlot::IMPRINTED: cur_game.resolution.memory.imprinted.push_back(ref); break;
    }
}

bool read_memory_param(AbilityDef &def, const std::string &key, const std::string &value) {
    bool known = false;
    for (const MemoryRow &row : kMemoryRows) {
        if (key != row.param) continue;
        known = true;
        if (form_takes(row.form, value)) add_memory_op(def, row.op);
    }
    return known;
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

void apply_memory_ops(Ability &ab) {
    if (ab.report.reported == 0) return;
    for (const MemoryOp &op : ab.def->memory_ops) {
        if (!(op.sources & ab.report.reported)) continue;
        if (op.action == MemoryAction::REPLACE) clear_slot(op.slot);
        for (const ReportedObject &object : ab.report.objects)
            if (object.sources & op.sources) add_to_slot(op.slot, object.ref);
    }
    ab.report = EffectReport{};
}
