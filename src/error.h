#ifndef ERROR_H
#define ERROR_H

#define FAILED_TO_OPEN_STREAM -1
#define SCRIPT_TOO_LONG -2

#include <string>

void non_fatal_error(std::string err);

#endif /* ERROR_H */
