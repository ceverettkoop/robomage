#ifndef DEBUG_H
#define DEBUG_H

#include "components/zone.h"
#include "systems/orderer.h"

void print_library(std::shared_ptr<Orderer> orderer, Zone::Ownership owner);

#endif /* DEBUG_H */
