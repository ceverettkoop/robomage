#include "zone.h"

Zone::Zone() {
}

Zone::Zone(ZoneValue in_loc, Ownership in_owner, Ownership in_controller) {
    location = in_loc;
    owner = in_owner;
    controller = in_controller;
    //distance defaults to 0, managed by shuffle system
}
