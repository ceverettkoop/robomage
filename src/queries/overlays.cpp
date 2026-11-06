#include "overlays.h"

#include <cstddef>
#include <memory>

#include "../components/ability.h"
#include "../components/carddata.h"
#include "../state_format/schema.h"
#include "affected.h"
#include "filters.h"

static bool has_overload(const CardData &card);
static void target_reads_each(AbilityDef &ability);
static void overload_spell(CardData &face);

static const SpellOverlayRow kSpellOverlayRows[] = {
    // kind                  name        variant                                            resolved_to      has           change
    {SpellOverlay::NONE,     "",         -1,                                                Zone::GRAVEYARD, nullptr,      nullptr},
    {SpellOverlay::OVERLOAD, "overload", static_cast<int>(rmstate::CastVariant::OVERLOAD),  Zone::GRAVEYARD, has_overload, overload_spell},
};

static bool has_overload(const CardData &card) { return card.alt_cost.is_overload; }

// CR 702.96a: the text-changing effect. Each instruction that targets acts on every permanent its
// target spec matches instead, chooses nothing, and so can reach what couldn't be targeted
// (CR 702.96b).
static void target_reads_each(AbilityDef &ability) {
    if (ability.valid_tgts != "N_A") {
        ability.defined = parse_object_spec("Valid " + target_filter(ability.valid_tgts).text());
        ability.valid_tgts = "N_A";
    }
    for (AbilityDef &sub : ability.subabilities) target_reads_each(sub);
}

static void overload_spell(CardData &face) {
    for (const AbilityDef *&ability : face.abilities)
        if (ability->ability_type == AbilityDef::SPELL)
            ability = derived_ability_def(ability, "overload", 0, target_reads_each);
}

const SpellOverlayRow &spell_overlay(SpellOverlay kind) { return kSpellOverlayRows[static_cast<size_t>(kind)]; }

uint32_t spell_overlay_variant_mask(SpellOverlay kind) {
    const int variant = spell_overlay(kind).variant;
    return variant < 0 ? 0 : 1u << variant;
}

void build_overlay_faces(CardData &card) {
    for (const SpellOverlayRow &row : kSpellOverlayRows) {
        if (!row.has || !row.has(card)) continue;
        auto face = std::make_shared<CardData>(card);
        row.change(*face);
        card.overlay_faces.push_back({row.kind, face});
    }
}
