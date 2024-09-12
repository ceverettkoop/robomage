#include <cstdlib>
#include <cstdio>

#ifndef VERSION_NUMBER
#define VERSION_NUMBER "0.001"
#endif

/*
extern "C" {
#include "gui.h"
}
*/

int main(int argc, char const *argv[]) {
    printf("robomage %s\n", VERSION_NUMBER);

    return 0;
}
