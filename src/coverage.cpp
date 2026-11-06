#include "coverage.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>

#include "classes/action.h"
#include "components/ability.h"
#include "effects/effect_kind.h"
#include "file_util.h"
#include "state_format/enum_names.h"

static void count_hit(const char *kind, const std::string &name);
static void write_coverage();
static const char *cost_kind_name(CostPart::Kind kind);

static bool counting = false;
static std::string coverage_path;

// Sorted by `<kind> <name>`, the order the file lists them. Allocated once and never freed, since
// the write runs at exit, after static destructors may have started.
static std::map<std::string, uint64_t> &hits() {
    static auto *store = new std::map<std::string, uint64_t>();
    return *store;
}

static void count_hit(const char *kind, const std::string &name) {
    ++hits()[std::string(kind) + " " + name];
}

static void write_coverage() {
    std::string text;
    for (const auto &[key, count] : hits()) text += key + " " + std::to_string(count) + "\n";
    std::string err;
    if (!write_file_atomic(coverage_path, text, &err)) std::fprintf(stderr, "ERROR: --coverage: %s\n", err.c_str());
}

void coverage_open(const std::string &path) {
    coverage_path = path;
    if (!counting) std::atexit(write_coverage);
    counting = true;
}

void coverage_row(const char *name) {
    if (counting) count_hit("row", name);
}

void coverage_effect(EffectKind kind) {
    if (counting) count_hit("effect", effect_kind_name(kind));
}

void coverage_trigger(const Ability &ab) {
    if (!counting) return;
    const AbilityDef &def = *ab.def;
    std::string mode = !def.trigger_mode.empty()  ? def.trigger_mode
                       : ab.delayed_link.seq != 0 ? "Delayed"
                       : !def.ref_key.empty()     ? def.ref_key
                                                  : "Reflexive";
    std::replace(mode.begin(), mode.end(), ' ', '_');
    count_hit("trigger", mode);
}

void coverage_replacement(const char *kind) {
    if (counting) count_hit("replacement", kind);
}

void coverage_decision(ActionCategory category) {
    if (counting) count_hit("category", action_category_enum_name(category));
}

static const char *cost_kind_name(CostPart::Kind kind) {
    switch (kind) {
        case CostPart::MANA:
            return "MANA";
        case CostPart::X_MANA:
            return "X_MANA";
        case CostPart::TAP:
            return "TAP";
        case CostPart::LIFE:
            return "LIFE";
        case CostPart::ENERGY:
            return "ENERGY";
        case CostPart::LOYALTY:
            return "LOYALTY";
        case CostPart::EXILE:
            return "EXILE";
        case CostPart::SACRIFICE:
            return "SACRIFICE";
        case CostPart::DISCARD:
            return "DISCARD";
        case CostPart::RETURN:
            return "RETURN";
        case CostPart::EXILE_FROM_HAND:
            return "EXILE_FROM_HAND";
        case CostPart::EXILE_FROM_GRAVEYARD:
            return "EXILE_FROM_GRAVEYARD";
        case CostPart::MANDATORY:
            break;
    }
    return "MANDATORY";
}

void coverage_cost(CostPart::Kind kind, const char *form) {
    if (counting) count_hit("cost", std::string(cost_kind_name(kind)) + (form ? std::string(".") + form : ""));
}

void coverage_cast(const char *mode) {
    if (counting) count_hit("cast", mode);
}
