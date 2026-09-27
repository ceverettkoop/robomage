#include "zone.h"

Zone::Zone() {
}

Zone::Zone(ZoneValue in_loc, Ownership in_owner, Ownership in_controller) {
    location = in_loc;
    owner = in_owner;
    controller = in_controller;
    //distance defaults to 0, managed by shuffle system
}

bool Zone::from_script_name(const std::string &name, ZoneValue &out) {
    if (name == "Library")     { out = LIBRARY;     return true; }
    if (name == "Hand")        { out = HAND;        return true; }
    if (name == "Graveyard")   { out = GRAVEYARD;   return true; }
    if (name == "Exile")       { out = EXILE;       return true; }
    if (name == "Sideboard")   { out = SIDEBOARD;   return true; }
    if (name == "Stack")       { out = STACK;       return true; }
    if (name == "Battlefield") { out = BATTLEFIELD; return true; }
    return false;
}
