#ifndef QUERIES_OVERLAYS_H
#define QUERIES_OVERLAYS_H

#include <cstdint>

#include "../components/cast_option.h"
#include "../components/zone.h"

struct CardData;

// ── What a cast changes on the spell ────────────────────────────────────────────
// A SpellOverlay is a kind and a row of the table in overlays.cpp; a row of the cast table
// (cast_options.cpp) names one, and the option it makes carries it. Nothing else branches on a kind.
struct SpellOverlayRow {
        SpellOverlay kind;
        const char *name;  // what the log and --coverage call it
        int variant;       // its rmstate::CastVariant, -1 with none
        // Where the spell goes as it finishes resolving (CR 608.2n) whatever else would send it
        // elsewhere; GRAVEYARD leaves that to what moves it.
        Zone::ZoneValue resolved_to;
        // The characteristics it gives a card: whether the card has it, and what it changes on a
        // copy of the card's own. Null for an overlay that reads a face the card has anyway.
        bool (*has)(const CardData &card);
        void (*change)(CardData &face);
};

const SpellOverlayRow &spell_overlay(SpellOverlay kind);

// The cast_variant bit `kind` sets on a menu row and a stack row; 0 for none.
uint32_t spell_overlay_variant_mask(SpellOverlay kind);

// Gives the parsed `card` the faces of the overlays it has.
void build_overlay_faces(CardData &card);

#endif /* QUERIES_OVERLAYS_H */
