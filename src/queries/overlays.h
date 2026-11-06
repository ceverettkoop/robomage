#ifndef QUERIES_OVERLAYS_H
#define QUERIES_OVERLAYS_H

#include "../components/cast_option.h"
#include "../components/zone.h"

// ── What a cast changes on the spell ────────────────────────────────────────────
// A SpellOverlay is a kind and a row of the table in overlays.cpp; a row of the cast table
// (cast_options.cpp) names one, and the option it makes carries it. Nothing else branches on a kind.
struct SpellOverlayRow {
        SpellOverlay kind;
        const char *name;  // what the log and --coverage call it
        // Where the spell goes as it finishes resolving (CR 608.2n) whatever else would send it
        // elsewhere; GRAVEYARD leaves that to what moves it.
        Zone::ZoneValue resolved_to;
};

const SpellOverlayRow &spell_overlay(SpellOverlay kind);

#endif /* QUERIES_OVERLAYS_H */
