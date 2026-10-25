#ifndef TYPE_CONSTANTS_H
#define TYPE_CONSTANTS_H

#include <set>
#include <string>

extern const std::set<std::string> all_types;
extern const std::set<std::string> all_subtypes;
extern const std::set<std::string> all_supertypes;

// The all_subtypes entry `name` spells (a curly apostrophe read as the straight one the entries
// use), or nullptr. A multi-word subtype (Time Lord, the planar types) is one name with its spaces.
const std::string *subtype_entry(const std::string &name);
// The most words one subtype has (a type line's tokens are matched longest first: CR 205.3m).
size_t subtype_max_words();

#endif /* TYPE_CONSTANTS_H */
