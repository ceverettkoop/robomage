#ifndef ERROR_H
#define ERROR_H

#include <string>
#include "ecs/entity.h"

void warning(std::string err);
void non_fatal_error(std::string err);
// Prints "FATAL: <err>" and exit(1)s — it never returns. Marked [[noreturn]] so
// every guard that calls it (e.g. the deck-identity distinct-name caps in
// deck_state.cpp / machine_io.cpp) is provably terminal and can never fall
// through to the code below it.
[[noreturn]] void fatal_error(std::string err);

#ifndef NDEBUG
void dump_entity(Entity e);
#endif

#endif /* ERROR_H */
