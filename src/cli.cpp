#include "cli.h"

#include <cstdio>

#include "classes/action.h"

int get_int_input() {
    int c;
    while ((c = getchar()) == ' ' || c == '\t');
    if (c == EOF || c == '\n') return -1;
    if (c == 'q' || c == 'Q') {
        printf("Quitting.\n");
        exit(0);
    }
    if (c == 'z' || c == 'Z') {
        while ((c = getchar()) != '\n' && c != EOF);
        return PASS_TURN_CMD;
    }
    ungetc(c, stdin);
    int choice = -1;
    if (scanf("%d", &choice) != 1) {
        while ((c = getchar()) != '\n' && c != EOF);
        return -1;
    }
    while ((c = getchar()) != '\n' && c != EOF);
    return choice;
}
