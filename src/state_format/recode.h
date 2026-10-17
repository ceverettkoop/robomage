#ifndef STATE_FORMAT_RECODE_H
#define STATE_FORMAT_RECODE_H

#include <string>

// ── --rmstate-read FILE: offline check of an RMSTATE stream (docs/state_format.md §9) ─────────
//
// Standard library, decode.h and json.h only. Reads every record of FILE ("-" = stdin), builds the
// decoder from the stream's SCHEMA frames (each one replaces the schema in use), decodes each
// frame, re-encodes it with the frame writer and compares the bytes. A record that fails to
// decode, or re-encodes to other bytes, prints an ERROR: line on stderr; the summary
//   RMSTATE_READ: records=<n> identical=<n> errors=<n>
// goes to stderr too. With `json` (--state-dump json) each decoded frame is printed on stdout as
// one JSON line (json.h), so stdout is exactly the stream's JSON rendering.
// Returns the process exit status: 0 when every record decoded and re-encoded identically.
int rmstate_read_stream(const std::string &path, bool json);

#endif /* STATE_FORMAT_RECODE_H */
