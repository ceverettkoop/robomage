#ifndef RMWORK_SELFTEST_INFERENCE_SERVER_H
#define RMWORK_SELFTEST_INFERENCE_SERVER_H

#include <string>

#include "selftest_table.h"

// ── The server the workbench starts: --selftest (stand-in servers) ───────────────────────────────

namespace rmwork {

// In a new temporary directory (short, for the sockets' paths), with shell scripts in rmserve's
// place: a started server STARTING until its socket listens, then RUNNING, its arguments (--socket
// and the numeric policy, a non-numeric id left out), stopped with SIGTERM and reaped; one that exits
// reported FAILED with its stderr; a server already listening taken as EXTERNAL, not started and not
// stopped; a missing binary, no numeric policy, an unreadable zipped policy and a socket path too long
// refused; start_server's settings round trip and the default socket it gives.
void check_inference_server_units(const CheckFn &check);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_INFERENCE_SERVER_H */
