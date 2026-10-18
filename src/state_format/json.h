#ifndef STATE_FORMAT_JSON_H
#define STATE_FORMAT_JSON_H

#include <string>

#include "decode.h"
#include "schema.h"

// ── --state-dump json: a decoded frame as one line of JSON (docs/state_format.md §9) ──────────
//
// Rendered from the decoded binary (decode.h), never from the engine, with no whitespace and a
// fixed key order, so the same frame always renders to the same bytes (python/rmstate.py renders
// it identically):
//
//   {"frame":<kind>,"viewer":<viewer>,"version":<n>,"schema_hash":"<16 hex digits>",
//    "tables":{"<table>":[{"<column>":<value>,…},…],…}[,"skipped":[<id>,…]]}
//
// Tables in ascending id, the string pool left out (text cells carry their strings); rows in
// order; columns in schema order. "skipped" (the ids of blocks the schema does not know) only when
// there are some. A value renders by its column's storage and semantic tag:
//   - text: a JSON string (", \ and control characters escaped; other bytes as sent);
//   - enum:<e>: all ones of the width → null, else the value's name in <e>'s string table, else
//     the number;
//   - mask:<e>: the list of set bits in ascending order, each as its name in <e>'s string table,
//     else its bit number;
//   - bool: 0 → false, 1 → true, else the number;
//   - card, index, row:<t>: all ones of the width → null, else the number;
//   - anything else (handle, count, signed, player-rel, …): the number (sign-extended for the
//     signed storages; f32 as printf "%.9g", null when not finite).
// The frame kind and viewer render as their names in the frame_kind / viewer string tables.

namespace rmstate {

std::string frame_json(const DecodedFrame &frame, const Schema &schema);

}  // namespace rmstate

#endif /* STATE_FORMAT_JSON_H */
