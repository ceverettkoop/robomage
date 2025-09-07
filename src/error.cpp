#include "error.h"
#include <cstdio>

void non_fatal_error(std::string err) {
    err += '\n';
    printf("%s", err.c_str());
    return;
}