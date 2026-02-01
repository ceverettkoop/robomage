#include "cli.h"

#include <cstdio>

#include "classes/action.h"

int get_int_input() {
    int choice = -1;
    int c;
    if (scanf("%d", &choice) != 1) {
        //error / clear buffer return -1
        int c;
        while ((c = getchar()) != '\n' && c != EOF);
        return -1;
    }
    //success
    while ((c = getchar()) != '\n' && c != EOF);
    return choice;
}
