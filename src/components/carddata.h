#ifndef CARDDATA_H
#define CARDDATA_H

#include <memory>
#include <string>
#include <set>
#include <cstdint>
#include <vector>
#include <unordered_set>
#include "../components/types.h"
#include "../classes/colors.h"
#include "ability.h"
#include "effect.h"
#include "static_ability.h"

struct AltCost {
    bool has_alt_cost = false;
    int life_cost = 0;
    int energy_cost = 0;  // PayEnergy<N> — energy ({E}) paid as part of this cost (CR 122.1c)
    int exile_from_hand_count = 0;
    Colors exile_from_hand_color = NO_COLOR;  // color of card to exile (BLUE, GREEN, etc.)
    int return_to_hand_count = 0;
    std::string return_to_hand_type = "";
    ManaValue mana_cost;                // mana portion of the alt cost (e.g. Evoke:R)
    std::string condition_svar = "";    // e.g. "Count$YouCastThisGame" — condition checked before allowing alt cost
    std::string condition_compare = ""; // e.g. "EQ0" — comparison for condition_svar
    std::string condition_is_present = "";  // IsPresent$ <filter> — controller must control a matching permanent (e.g. "Swamp.YouCtrl" for Snuff Out)
    bool is_free = false;               // Cost$ 0 — no mana required
    bool condition_not_your_turn = false;  // Condition$ NotPlayerTurn
    bool is_evoke = false;              // K:Evoke — when paid, the permanent sacrifices itself on ETB
    std::string sac_cost_spec = "";     // Sac<N/Type> portion of an alt/flashback cost (e.g. "Creature" for Cabal Therapy's Flashback—Sacrifice a creature)
    // ExileFromGrave<X/<filter>/<label>> — exile any number of OTHER cards from the caster's
    // graveyard as an additional cost (CR 601.2f), constrained so the chosen set collectively
    // has at least `exile_grave_min_types` distinct card types (Escape: Nethergoyf). 0 = no such cost.
    int exile_grave_min_types = 0;
};

//this is the underlying card, not a permanent or spell
struct CardData{
    std::string uid;
    std::string name;
    std::string oracle_text;
    std::set<Type> types;
    std::multiset<Colors> mana_cost;
    std::vector<Colors> phyrexian_mana;  // Phyrexian mana symbols: each can be paid with color OR 2 life
    uint32_t power = 0;
    uint32_t toughness = 0;
    int starting_loyalty = 0;  // Loyalty: line — printed loyalty a planeswalker enters with (306.5b)
    std::vector<Ability> abilities;
    AltCost alt_cost;
    std::vector<std::string> keywords;
    std::vector<StaticAbility> static_abilities;
    std::vector<Effect::Replacement> replacement_effects;  // parsed from R: lines TODO expand this to parse SVAR below
    std::shared_ptr<CardData> backside;  // populated for DFCs; nullptr for normal cards
    bool has_delve = false;              // K:Delve — exile from graveyard to reduce generic cost
    bool has_improvise = false;          // K:Improvise — tap untapped artifacts you control to pay {1} each (CR 702.126)
    int ward_cost = 0;                   // K:Ward:N — opponent's spell/ability targeting this is countered unless they pay {N} (CR 702.21)
    bool ward_is_life = false;           // K:Ward:PayLife<N> — the ward cost is paid as N life rather than {N} mana (CR 702.21)
    bool affinity_artifact = false;      // K:Affinity:Artifact — costs {1} less to cast per artifact you control (CR 702.41)
    bool has_x_cost = false;             // ManaCost contains X — variable generic cost chosen at cast time
    bool shuffle_into_library = false;   // card shuffles into library instead of going to graveyard on resolution
    bool has_flashback = false;          // K:Flashback — can cast from graveyard for flashback cost, then exile
    bool has_etb_choose_creature_type = false;  // K:ETBReplacement:Other:ChooseCT — choose creature type on ETB
    bool has_etb_name_card = false;             // K:ETBReplacement:Other:DBNameCard — choose a card name on ETB (Disruptor Flute)
    int dredge = 0;                      // K:Dredge:N — replace a draw by milling N and returning this from graveyard to hand
    ManaValue flashback_mana_cost;       // mana portion of flashback cost
    AltCost flashback_alt_cost;          // non-mana costs (e.g. PayLife<3> for Deep Analysis)
    bool has_escape = false;             // K:Escape — cast from graveyard for the escape cost (CR 702.139)
    ManaValue escape_mana_cost;          // mana portion of the escape cost (e.g. {2}{B})
    AltCost escape_alt_cost;             // additional escape costs (e.g. ExileFromGrave group-type cost)
    bool is_equipment = false;           // has K:Equip line
    ManaValue equip_cost;                // parsed from K:Equip:cost
    bool has_offspring = false;          // K:Offspring:cost — optional additional cost (CR 702.171)
    ManaValue offspring_cost;            // mana paid in addition to the spell's cost for Offspring
    // K:Kicker:<cost1>[:<cost2>...] — one or more OPTIONAL ADDITIONAL costs (CR 702.33).
    // "Kicker [A] and/or [B]" is Forge-encoded as two colon-separated costs and means
    // "Kicker [A], kicker [B]" (CR 702.33b): each may be paid independently as the spell is
    // cast. kicker_costs[i] is the mana paid for the (i+1)th kicker; the spell is "kicked
    // with its (i+1)th kicker" iff that cost was paid. The list is multikicker-ready (any
    // count >= 1). Empty = the card has no kicker. Linked "if it was kicked with its [N]
    // kicker" triggers read the per-Spell kicked flags by index (see Spell::kicked).
    std::vector<ManaValue> kicker_costs;
    std::set<Colors> explicit_colors;    // Colors: field override (e.g. Dryad Arbor)
};

#endif /* CARD_H */
