#include "overlays.h"

#include <cstddef>
#include <memory>
#include <set>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../state_format/schema.h"
#include "affected.h"
#include "characteristics.h"
#include "filters.h"
#include "types.h"

static bool has_overload(const CardData &card);
static void target_reads_each(AbilityDef &ability);
static void overload_spell(CardData &face);
static std::shared_ptr<const CardData> combined_halves(const CardData &left, const CardData &right);
static void on_an_adventure(Entity card, Zone::Ownership caster);
static bool has_bestow(const CardData &card);
static void bestow_aura(CardData &face);

static const SpellOverlayRow kSpellOverlayRows[] = {
    // kind                   name         variant                                            resolved_to      has           change          resolved         stays  copy
    {SpellOverlay::NONE,      "",          -1,                                                Zone::GRAVEYARD, nullptr,      nullptr,        nullptr,         false, false},
    {SpellOverlay::OVERLOAD,  "overload",  static_cast<int>(rmstate::CastVariant::OVERLOAD),  Zone::GRAVEYARD, has_overload, overload_spell, nullptr,         false, false},
    {SpellOverlay::ADVENTURE, "adventure", static_cast<int>(rmstate::CastVariant::ADVENTURE), Zone::EXILE,     nullptr,      nullptr,        on_an_adventure, false, false},
    {SpellOverlay::BESTOW,    "bestow",    static_cast<int>(rmstate::CastVariant::BESTOW),    Zone::GRAVEYARD, has_bestow,   bestow_aura,    nullptr,         true,  false},
    {SpellOverlay::PREPARE,   "prepared",  static_cast<int>(rmstate::CastVariant::PREPARED),  Zone::GRAVEYARD, nullptr,      nullptr,        nullptr,         false, true},
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
            ability = derived_ability_def(ability, "overload", 0, [ability](AbilityDef &d) {
                target_reads_each(d);
                d.printed_as = ability;
            });
}

// CR 709.4: the left half with what the right half adds, one spell ability doing the left half's
// instructions and then the right's (CR 702.102d).
static std::shared_ptr<const CardData> combined_halves(const CardData &left, const CardData &right) {
    auto both = std::make_shared<CardData>(left);
    both->name = whole_card_name(left);
    both->backside = nullptr;
    both->is_split = false;
    both->mana_cost.insert(right.mana_cost.begin(), right.mana_cost.end());
    both->phyrexian_mana.insert(both->phyrexian_mana.end(), right.phyrexian_mana.begin(), right.phyrexian_mana.end());
    both->hybrid_mana.insert(both->hybrid_mana.end(), right.hybrid_mana.begin(), right.hybrid_mana.end());
    both->types.insert(right.types.begin(), right.types.end());
    both->keywords.insert(both->keywords.end(), right.keywords.begin(), right.keywords.end());
    const AbilityDef *right_spell = nullptr;
    for (const AbilityDef *ability : right.abilities) {
        if (ability->ability_type == AbilityDef::SPELL && !right_spell)
            right_spell = ability;
        else
            both->abilities.push_back(ability);
    }
    for (const AbilityDef *&ability : both->abilities)
        if (ability->ability_type == AbilityDef::SPELL && right_spell) {
            const AbilityDef *left_spell = ability;
            ability = derived_ability_def(left_spell, "fused", 0, [left_spell, right_spell](AbilityDef &d) {
                d.subabilities.push_back(*right_spell);
                d.printed_as = left_spell;
            });
            break;
        }
    return both;
}

// CR 715.3d: for as long as the card remains exiled, the player who cast the Adventure may play it.
static void on_an_adventure(Entity card, Zone::Ownership caster) {
    Game::ImpulseCastPermission grant;
    grant.resource = Game::ImpulseCastPermission::NORMAL;
    grant.caster = caster;
    grant.allow_land = true;
    grant.on_an_adventure = true;
    cur_game.resolved_effects.impulse_cast_permission[card] = grant;
}

static bool has_bestow(const CardData &card) { return card.alt_cost.is_bestow; }

// CR 702.103b: an Aura enchantment with enchant creature, no longer a creature, so without the
// creature types (CR 205.1b).
static void bestow_aura(CardData &face) {
    std::set<Type> types;
    for (const Type &type : face.types)
        if (type.kind == SUPERTYPE || (type.kind == TYPE && type.name != "Creature")) types.insert(type);
    types.insert(Type{TYPE, "Enchantment"});
    types.insert(Type{SUBTYPE, "Aura"});
    face.types = types;
    face.types_text = "Enchantment Aura";
    face.enchant_filter = "Creature";
}

const SpellOverlayRow &spell_overlay(SpellOverlay kind) { return kSpellOverlayRows[static_cast<size_t>(kind)]; }

uint32_t spell_overlay_variant_mask(SpellOverlay kind) {
    const int variant = spell_overlay(kind).variant;
    return variant < 0 ? 0 : 1u << variant;
}

// A characteristic-defining ability's power or toughness: a literal is written on the face, anything
// else is counted as it is read (card_power_toughness).
static void read_defined_pt(const std::string &value, uint32_t &field, bool &counted) {
    if (value.empty()) return;
    if (value.find_first_not_of("0123456789") == std::string::npos) field = static_cast<uint32_t>(std::stoul(value));
    else counted = true;
}

// What `card`'s characteristic-defining abilities make it wherever it isn't a permanent (CR 604.3).
static void build_off_battlefield_face(CardData &card) {
    for (const StaticAbility &sa : card.static_abilities) {
        if (!sa.characteristic_defining) continue;
        if (!card.off_battlefield) card.off_battlefield = std::make_shared<CardData>(card);
        CardData &face = *card.off_battlefield;
        for (const Type &t : types_added_by_static(sa)) face.types.insert(t);
        read_defined_pt(sa.set_power_svar, face.power, card.has_counted_pt);
        read_defined_pt(sa.set_toughness_svar, face.toughness, card.has_counted_pt);
    }
    if (card.off_battlefield) card.off_battlefield->has_counted_pt = card.has_counted_pt;
}

void build_overlay_faces(CardData &card) {
    for (const SpellOverlayRow &row : kSpellOverlayRows) {
        if (!row.has || !row.has(card)) continue;
        auto face = std::make_shared<CardData>(card);
        row.change(*face);
        card.overlay_faces.push_back({row.kind, face});
    }
    if (card.is_split && card.backside) card.combined = combined_halves(card, *card.backside);
    build_off_battlefield_face(card);
}
