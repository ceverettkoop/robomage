#ifndef ROBOMAGE_CHECK_REGISTRY_CHECK_H
#define ROBOMAGE_CHECK_REGISTRY_CHECK_H

#include "gate.h"

// ── robomage_check: the layout registry is current and append-only (docs/featurizer.md §6.1) ──
//
// Part of the vocab tier: rmfeat --registry-check on src/featurize/registry.json with the engine's
// whole-vocab cards file (robomage --cards-frame): the file parses and is in its canonical form, the
// binaries were built with it, the cards file's schema and tables need no entry it lacks (else `make
// registry`), and against the committed file (`git show HEAD:src/featurize/registry.json`, when git
// and a committed file are there) no entry moved unless layout_version was raised.

namespace rmcheck {

void registry_check_part(const Options &opt, Report &rep);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_REGISTRY_CHECK_H */
