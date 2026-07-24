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
    // K:Impending:N:cost — Impending (CR 702.175, Duskmourn). An ALTERNATIVE casting cost: the
    // spell may be cast for `mana_cost` instead of its normal mana cost. When cast this way the
    // permanent enters with `impending_count` time counters and ISN'T A CREATURE until the last
    // time counter is removed (it is on the battlefield only as its other types). At the
    // beginning of its controller's end step one time counter is removed (CR 702.175e); it
    // becomes a creature when the last is gone. Cast for the NORMAL cost, none of this applies.
    bool is_impending = false;
    int impending_count = 0;            // N time counters the permanent enters with
    // K:Spectacle:<cost> — Spectacle (CR 702.107). An ALTERNATIVE casting cost: the spell may
    // be cast for `mana_cost` (the spectacle cost) instead of its normal mana cost, but only if
    // an opponent of the caster lost life this turn (CR 702.107a). Gated in can_afford_alt on
    // the opponent's Player::life_lost_this_turn; paid via the shared alt-cost path.
    bool is_spectacle = false;
    std::string sac_cost_spec = "";     // Sac<N/Type> portion of an alt/flashback cost (e.g. "Creature" for Cabal Therapy's Flashback—Sacrifice a creature)
    // Sac<N/Type> count of an ALTERNATIVE casting cost (CR 118.9, e.g. Fireblast: "sacrifice two
    // Mountains rather than pay this spell's mana cost"). sac_cost_spec holds the "Type" filter,
    // this holds N. 0 = the alt cost has no sacrifice component. (The flashback sac path always
    // sacrifices exactly one and doesn't read this; it's used only by the alt-cost cast path.)
    int sac_cost_count = 0;
    // ExileFromGrave<X/<filter>/<label>> — exile any number of OTHER cards from the caster's
    // graveyard as an additional cost (CR 601.2f), constrained so the chosen set collectively
    // has at least `exile_grave_min_types` distinct card types (Escape: Nethergoyf). 0 = no such cost.
    int exile_grave_min_types = 0;
    // ExileFromGrave<N/<filter>/<label>> with a literal count N (no withTypesGE constraint):
    // exile exactly N OTHER cards from the caster's graveyard as an additional cost (CR 601.2f),
    // e.g. Escape: Uro "Exile five other cards from your graveyard" → 5. 0 = no such cost.
    int exile_grave_count = 0;
};

// A hybrid mana pip (CR 107.4b/107.4e, 202.3f). Stored parallel to mana_cost (which holds only
// the NON-hybrid pips), the same way phyrexian_mana is kept separate. Two forms:
//   - Color hybrid {W/U}: `colors` holds the two color options (one mana of either); generic_alt
//     is 0. Mana value contribution = 1 (CR 202.3f: greatest component MV).
//   - Monocolored hybrid / "twobrid" {2/W}: `colors` holds the single color option (one mana of
//     that color) and generic_alt is the generic alternative (2). Mana value contribution = 2.
// `mana_value` caches the CMC contribution so consumers don't re-derive it (Phyrexian {W/P} is
// NOT modeled here — it stays in phyrexian_mana — so this is purely color-hybrid / twobrid).
struct HybridPip {
    std::vector<Colors> colors;  // color alternatives (color-hybrid: 2; twobrid: 1)
    int generic_alt = 0;         // generic-mana alternative count (twobrid: 2; color-hybrid: 0)
    int mana_value = 1;          // contribution to mana value (CMC)
};

//this is the underlying card, not a permanent or spell
struct CardData{
    std::string uid;
    std::string name;
    std::string oracle_text;
    std::set<Type> types;
    std::multiset<Colors> mana_cost;
    std::vector<Colors> phyrexian_mana;  // Phyrexian mana symbols: each can be paid with color OR 2 life
    std::vector<HybridPip> hybrid_mana;  // hybrid pips ({W/U}, {2/W}); kept out of mana_cost like phyrexian
    uint32_t power = 0;
    uint32_t toughness = 0;
    int starting_loyalty = 0;  // Loyalty: line — printed loyalty a planeswalker enters with (306.5b)
    std::vector<Ability> abilities;
    AltCost alt_cost;
    std::vector<std::string> keywords;
    std::vector<StaticAbility> static_abilities;
    std::vector<Effect::Replacement> replacement_effects;  // parsed from R: lines TODO expand this to parse SVAR below
    std::shared_ptr<CardData> backside;  // populated for DFCs; nullptr for normal cards
    // AlternateMode:Modal — a MODAL double-faced card (MDFC, CR 712.x): the player chooses,
    // when playing it from hand, which face to cast/play (front spell OR back face). Unlike a
    // transform DFC (Delver/Ajani), an MDFC is not a transforming permanent — the chosen face
    // is what enters and it does not flip in play. The back face is its own complete CardData in
    // `backside`. False for single-faced cards and for transform DFCs.
    bool is_modal_dfc = false;
    bool has_delve = false;              // K:Delve — exile from graveyard to reduce generic cost
    bool has_improvise = false;          // K:Improvise — tap untapped artifacts you control to pay {1} each (CR 702.126)
    int ward_cost = 0;                   // K:Ward:N — opponent's spell/ability targeting this is countered unless they pay {N} (CR 702.21)
    bool ward_is_life = false;           // K:Ward:PayLife<N> — the ward cost is paid as N life rather than {N} mana (CR 702.21)
    bool affinity_artifact = false;      // K:Affinity:Artifact — costs {1} less to cast per artifact you control (CR 702.41)
    std::string enchant_filter;          // K:Enchant:<ValidTgts> — for Auras, the object this can enchant (CR 303.4); the aura's spell targets it at cast and attaches on resolution
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
    // Reconfigure (CR 702.151): an Equipment keyword on a creature card. It is parsed like
    // Equip (is_equipment + equip_cost both set), but is_reconfigure additionally (a) lets the
    // attach ability target only a creature you control, (b) offers an "unattach" ability while
    // attached, and (c) makes the permanent stop being a creature while it is attached.
    bool is_reconfigure = false;         // has K:Reconfigure:<cost> line
    bool has_offspring = false;          // K:Offspring:cost — optional additional cost (CR 702.171)
    ManaValue offspring_cost;            // mana paid in addition to the spell's cost for Offspring
    // K:Gift — the Gift keyword (CR 702.176). As the spell is cast its controller MAY promise the
    // gift to an opponent (an optional choice, not a cost); if promised, the opponent receives the
    // gift as the spell resolves, before its other effects. gift_abilities holds the parsed gift
    // effect (Into the Flood Maw: a DB$ Token making a tapped 1/1 Fish for the promised opponent),
    // from the card's GiftAbility SVar. gift_description is the printed name of the gift (display).
    bool has_gift = false;
    std::vector<Ability> gift_abilities;
    std::string gift_description = "";
    // K:Kicker:<cost1>[:<cost2>...] — one or more OPTIONAL ADDITIONAL costs (CR 702.33).
    // "Kicker [A] and/or [B]" is Forge-encoded as two colon-separated costs and means
    // "Kicker [A], kicker [B]" (CR 702.33b): each may be paid independently as the spell is
    // cast. kicker_costs[i] is the mana paid for the (i+1)th kicker; the spell is "kicked
    // with its (i+1)th kicker" iff that cost was paid. The list is multikicker-ready (any
    // count >= 1). Empty = the card has no kicker. Linked "if it was kicked with its [N]
    // kicker" triggers read the per-Spell kicked flags by index (see Spell::kicked).
    std::vector<ManaValue> kicker_costs;
    // K:Replicate:<cost> — an OPTIONAL ADDITIONAL cost that may be paid ANY NUMBER OF TIMES as
    // the spell is cast (CR 702.x / Consign to Memory). Each payment increments the spell's
    // "replicate count"; when the spell is cast it is copied that many times, and the copies
    // may choose new targets (the copy mechanism, not a recast). has_replicate gates it; the
    // per-instance mana paid for each replication is replicate_cost. The chosen count is stored
    // per-Spell (Spell::replicate_count) so the on-cast copy effect knows how many copies to make.
    bool has_replicate = false;
    ManaValue replicate_cost;
    std::set<Colors> explicit_colors;    // Colors: field override (e.g. Dryad Arbor)
    // K:Companion:<grouping>:<restriction>:<desc> — the Companion keyword (CR 702.139). A card
    // with this keyword may begin the game in the sideboard ("outside the game"); if the chosen
    // companion's deckbuilding restriction is met by the starting deck, its controller may pay
    // {3} once per game, at sorcery speed, to put it from the sideboard into their hand.
    // companion_restriction is the Forge restriction token (e.g. "DeckSizePlus20") evaluated by
    // deck_meets_companion_restriction in companion.cpp. False/empty for non-companion cards.
    bool is_companion = false;
    std::string companion_restriction;
    // K:Chapter:<final>:<svar1>,<svar2>,... — a Saga's chapter abilities (CR 714). saga_chapters[i]
    // is the triggered ability (a DB$ ability body) run when this Saga's lore counters reach chapter
    // (i+1) (CR 714.2b/714.3); the vector size is the Saga's final chapter number (CR 714.2d).
    // Multiple chapters may share the same ability (Summon: Bahamut I & II both destroy). Empty for
    // a non-Saga card. The Saga lifecycle (lore counters, chapter triggers, sacrifice SBA) lives in
    // src/saga.{h,cpp}.
    std::vector<Ability> saga_chapters;
    // K:MayEffectFromOpeningHand:<SVar>[:!PlayFirst] — "If this card is in your opening hand,
    // you may [effect]" (CR 103.6b; the Leylines' "begin the game with it on the battlefield").
    // The named SVar's body is parsed into opening_hand_abilities (Leyline of the Void:
    // DB$ ChangeZone | Defined$ Self | Origin$ Hand | Destination$ Battlefield). After mulligans
    // resolve, players are offered these in APNAP order — starting player first — before the
    // first turn begins (the pregame gate's OPENING_ACTIONS stage, game_driver.cpp). An
    // optional !PlayFirst field
    // (Gemstone Caverns) restricts the offer to a player who is NOT the starting player.
    // Empty for cards without the keyword.
    std::vector<Ability> opening_hand_abilities;
    bool opening_hand_not_first = false;
};

#endif /* CARD_H */
