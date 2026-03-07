#include "error.h"
#include "cli_output.h"

void non_fatal_error(std::string err) { cli_error(err); }
void fatal_error(std::string err) { cli_fatal_error(err); }
