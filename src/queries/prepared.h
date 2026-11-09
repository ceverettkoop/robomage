#ifndef QUERIES_PREPARED_H
#define QUERIES_PREPARED_H

#include <memory>

#include "../ecs/entity.h"
#include "../ecs/entity_set.h"

class Orderer;

// ── The prepared designation (CR 722.3) ─────────────────────────────────────────
// Keeps the copies in exile true to the permanents: a permanent that has become prepared, or phased
// in prepared, gets its copy (CR 722.3c), and a copy goes as soon as its permanent is gone, phased
// out or no longer prepared, or it is no longer in exile (CR 704.5e). Run with the state-based pass.
void sync_prepared_copies(const EntitySet &entities, const std::shared_ptr<Orderer> &orderer);

// The cast of the copy `card` became cast: its permanent loses the designation (CR 722.3c).
void prepared_copy_cast(Entity card);

#endif /* QUERIES_PREPARED_H */
