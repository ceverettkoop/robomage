#ifndef PREPARED_COPY_H
#define PREPARED_COPY_H

#include "../object_ref.h"

// The copy a prepared permanent makes of its prepare spell in exile (CR 722.3c): the object has this
// while it waits there, and is no card.
struct PreparedCopy {
    ObjectRef permanent;  // the prepared permanent
};

#endif /* PREPARED_COPY_H */
