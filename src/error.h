#ifndef ERROR_H
#define ERROR_H

#include <string>
#include "ecs/entity.h"

void non_fatal_error(std::string err);
void fatal_error(std::string err);

#ifndef NDEBUG
void dump_entity(Entity e);
#endif

#endif /* ERROR_H */
