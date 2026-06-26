#ifndef EFFECTS_H
#define EFFECTS_H

#include <memory>
#include <string>

#include "../components/ability.h"
#include "effect_kind.h"

class Orderer;

// ── Effect handler dispatch ────────────────────────────────────────────────
//
// Each effect category resolves through a free-function handler with this
// signature. The bool return is "run the standard subability-chaining loop
// afterward?" — almost every effect returns true; the handful that manage their
// own subability resolution or short-circuit the game (Charm, WinsGame, the
// non-peek PeekAndReveal path) return false to suppress it, exactly as the
// legacy if/else chain did via early `return`.
namespace effects {

using EffectHandler = bool (*)(Ability &, std::shared_ptr<Orderer>);

// Returns the handler for `kind`, or nullptr if no resolve-time handler exists
// (None / not-yet-migrated). Ability::resolve() falls back to its legacy chain
// when this returns nullptr.
EffectHandler handler_for(EffectKind kind);

// Per-effect handlers (defined one per src/effects/effect_*.cpp).
bool deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer);
bool draw(Ability &ab, std::shared_ptr<Orderer> orderer);
bool gain_life(Ability &ab, std::shared_ptr<Orderer> orderer);
bool lose_life(Ability &ab, std::shared_ptr<Orderer> orderer);
bool mill(Ability &ab, std::shared_ptr<Orderer> orderer);
bool untap(Ability &ab, std::shared_ptr<Orderer> orderer);
bool cleanup(Ability &ab, std::shared_ptr<Orderer> orderer);
bool multiply_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool phases(Ability &ab, std::shared_ptr<Orderer> orderer);
bool wins_game(Ability &ab, std::shared_ptr<Orderer> orderer);
bool prowess_bonus(Ability &ab, std::shared_ptr<Orderer> orderer);
bool exalted_bonus(Ability &ab, std::shared_ptr<Orderer> orderer);
bool attach(Ability &ab, std::shared_ptr<Orderer> orderer);
bool choose_card(Ability &ab, std::shared_ptr<Orderer> orderer);
bool destroy_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool destroy(Ability &ab, std::shared_ptr<Orderer> orderer);
bool token(Ability &ab, std::shared_ptr<Orderer> orderer);
bool surveil(Ability &ab, std::shared_ptr<Orderer> orderer);
bool delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer);
bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer);
bool change_zone(Ability &ab, std::shared_ptr<Orderer> orderer);
bool change_zone_all(Ability &ab, std::shared_ptr<Orderer> orderer);
// ChangeType$ Remembered.sameName / Targeted.sameName mover, shared by change_zone
// (force_all=false) and change_zone_all (force_all=true).
bool change_zone_same_name(Ability &ab, std::shared_ptr<Orderer> orderer, bool force_all);
bool counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool charm(Ability &ab, std::shared_ptr<Orderer> orderer);
// Pyroblast/Hydroblast: ConditionPresent$ <type>.<Color> gates the EFFECT (not the
// target's legality). Returns true if there is no color requirement, or the target
// has the required color. Non-color ConditionPresent specs (e.g. cmcLEX) return true.
bool target_color_condition_met(const Ability &ab, Entity target);
bool add_mana(Ability &ab, std::shared_ptr<Orderer> orderer);
bool discard(Ability &ab, std::shared_ptr<Orderer> orderer);
bool pump(Ability &ab, std::shared_ptr<Orderer> orderer);
bool peek_and_reveal(Ability &ab, std::shared_ptr<Orderer> orderer);
bool dig(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sylvan_library(Ability &ab, std::shared_ptr<Orderer> orderer);
bool amass(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sacrifice(Ability &ab, std::shared_ptr<Orderer> orderer);
bool put_counter_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sacrifice_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer);

// True if battlefield permanent `e` matches a Forge ValidCards$/ConditionPresent$
// filter spec: a type/subtype head plus '+'-delimited qualifiers (YouCtrl, OppCtrl,
// Other, nonLand, nonToken, nonChosenCard, and the five colors). `controller` is the
// effect's controller (resolves You/Opp), `source` resolves Other. Unknown qualifiers
// fail closed so a mass effect never over-selects. Shared by PutCounterAll,
// SacrificeAll, and the ImmediateTrigger condition gate.
bool permanent_matches_cards_filter(Entity e, const std::string &spec,
                                    Zone::Ownership controller, Entity source);

// ── Effect-specific parse hooks ─────────────────────────────────────────────
//
// Co-located with each effect's resolve handler: each hook owns the card-script
// param keys that are exclusive to that effect, returning true if it consumed
// the (key, value). The parser's generic apply_param_to_ability handles the
// shared keys and delegates anything left over to apply_parse_hook(), which
// tries each hook in turn. Keys are partitioned so at most one hook claims any
// given key — relocation is therefore byte-identical to the old flat parser.
bool apply_parse_hook(Ability &ab, const std::string &key, const std::string &value);

bool parse_deal_damage(Ability &ab, const std::string &key, const std::string &value);
bool parse_pump(Ability &ab, const std::string &key, const std::string &value);
bool parse_token(Ability &ab, const std::string &key, const std::string &value);
bool parse_add_mana(Ability &ab, const std::string &key, const std::string &value);
bool parse_destroy_all(Ability &ab, const std::string &key, const std::string &value);
bool parse_change_zone(Ability &ab, const std::string &key, const std::string &value);
bool parse_put_counter(Ability &ab, const std::string &key, const std::string &value);
bool parse_dig(Ability &ab, const std::string &key, const std::string &value);
bool parse_delayed_trigger(Ability &ab, const std::string &key, const std::string &value);
bool parse_discard(Ability &ab, const std::string &key, const std::string &value);
bool parse_mill(Ability &ab, const std::string &key, const std::string &value);
bool parse_peek_and_reveal(Ability &ab, const std::string &key, const std::string &value);
bool parse_amass(Ability &ab, const std::string &key, const std::string &value);

}  // namespace effects

// ── Shared resolution helpers ───────────────────────────────────────────────
// De-static'd from ability.cpp so effect handlers in separate translation units
// can reuse them. Definitions still live in ability.cpp for now.
// (search_zone is declared in ability.h.)
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target);
Entity search_multi_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    const std::vector<Zone::ZoneValue> &zones, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal = false);
bool run_unless_loop(size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for);

#endif /* EFFECTS_H */
