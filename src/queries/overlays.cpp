#include "overlays.h"

#include <cstddef>

static const SpellOverlayRow kSpellOverlayRows[] = {
    // kind              name  resolved_to
    {SpellOverlay::NONE, "",   Zone::GRAVEYARD},
};

const SpellOverlayRow &spell_overlay(SpellOverlay kind) { return kSpellOverlayRows[static_cast<size_t>(kind)]; }
