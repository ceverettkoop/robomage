#include "colors.h"

std::string mana_symbol(Colors color){
    switch (color) {
        case WHITE:
            return "W";
            break;
        case BLUE:
            return "U";
            break;
        case BLACK:
            return "B";
            break;
        case RED:
            return "R";
            break;
        case GREEN:
            return "G";
            break;
        case COLORLESS:
            return "C";
            break;
        default:
            return "?";
            break;
    }
}