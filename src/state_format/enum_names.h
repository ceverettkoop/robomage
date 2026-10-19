#ifndef STATE_FORMAT_ENUM_NAMES_H
#define STATE_FORMAT_ENUM_NAMES_H

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../components/types.h"
#include "schema.h"

// ── The engine's enums as state-format string tables (docs/state_format.md §9, Appendix A.7) ──
//
// schema.h (standard library only) defines the format's own enums; the enums the engine defines
// reach the SCHEMA frame from here, each built from the engine's own definition of it: a name
// function that switches over every enumerator (a missing one fails the build, see
// enum_names.cpp), the effect_kinds.def X-macro, the type lists in src/constants.cpp and the
// parser's keyword table. Nothing here repeats a list by hand.

// Every engine enum the format uses, in a fixed order: step, action_category, action_type, zone,
// seat, color, mandatory_choice, day_night, pending_tag, window_kind, window_end, fire_kind,
// effect_kind, card_type, subtype, supertype, keyword.
std::vector<rmstate::EnumTable> engine_enums();

// The format's enums followed by the engine's: the string tables of the SCHEMA frame.
std::vector<rmstate::EnumTable> all_enums();

// An action category's name, as the action_category string table holds it.
std::string action_category_enum_name(ActionCategory v);

// Indices into the type enums (the sorted all_types / all_subtypes / all_supertypes lists); -1
// for a name not in the list.
int card_type_index(const std::string &name);
int subtype_index(const std::string &name);
int supertype_index(const std::string &name);
// A type line as the format's masks and subtype rows hold it.
uint32_t card_type_mask(const std::set<Type> &types);
uint8_t supertype_mask(const std::set<Type> &types);
std::vector<int> subtype_indices(const std::set<Type> &types);

// The keyword enum value of a keyword as an object holds it ("Flying", "Ward:2",
// "Protection from blue"): the longest enum name the text starts with, followed by its end, a ':'
// or a space; -1 for none.
int keyword_index(const std::string &keyword);

#endif /* STATE_FORMAT_ENUM_NAMES_H */
