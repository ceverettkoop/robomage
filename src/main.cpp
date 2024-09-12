#include <cstdlib>

extern "C" {
#include "gui.h"
}

int main(int argc, char const *argv[]) {
    bool should_quit = false;

    while (!should_quit) {
        should_quit = gui_loop();
    }

    return 0;
}
