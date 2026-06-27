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
bool damage_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool destroy(Ability &ab, std::shared_ptr<Orderer> orderer);
bool token(Ability &ab, std::shared_ptr<Orderer> orderer);
bool surveil(Ability &ab, std::shared_ptr<Orderer> orderer);
bool scry(Ability &ab, std::shared_ptr<Orderer> orderer);
bool delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer);
bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool remove_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
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
bool reveal(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ RevealHand (Thought-Knot Seer): the targeted player (ValidTgts$ Opponent) reveals their
// hand to all players (CR 701.16) — logged and recorded in the belief state. General over
// Duress/Thoughtseize-style "target player reveals their hand". See effect_reveal_hand.cpp.
bool reveal_hand(Ability &ab, std::shared_ptr<Orderer> orderer);
bool dig(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sylvan_library(Ability &ab, std::shared_ptr<Orderer> orderer);
bool amass(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sacrifice(Ability &ab, std::shared_ptr<Orderer> orderer);
bool put_counter_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool sacrifice_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer);
bool copy_permanent(Ability &ab, std::shared_ptr<Orderer> orderer);
// Mobilize N (702.176): create N tapped+attacking 1/1 red Warrior tokens and register a
// delayed end-step sacrifice of exactly those tokens.
bool mobilize(Ability &ab, std::shared_ptr<Orderer> orderer);
// Delayed end-step sacrifice fired by Mobilize: sacrifice each entity in ab.targets that is
// still on the battlefield (the tokens created when the creature attacked).
bool sacrifice_tokens(Ability &ab, std::shared_ptr<Orderer> orderer);
// RepeatEach over players (Price of Progress): resolve the RepeatSubAbility once per
// player, with cur_game.remembered_entities set to that player's entity each iteration.
bool repeat_each(Ability &ab, std::shared_ptr<Orderer> orderer);
// AB$ Effect granting "you may cast that card this turn" (Emry): records the targeted
// graveyard card in cur_game.may_cast_this_turn so the casting path offers it this turn.
bool grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer);
// SP$/DB$ NameCard (Cabal Therapy): the ability's controller names a card (CR 201.4); the
// chosen name is stored in cur_game.named_card so a chained Card.NamedCard sub-ability
// (here a RevealDiscardAll discard) can reference it.
bool name_card(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ Animate (Guide of Souls): the targeted permanent "becomes ..." — bakes the
// Duration$ Permanent continuous effect (added types/subtypes, and the extension-point
// base-P/T / keyword / creature grants) onto the permanent so the layer system reapplies
// it each SBE pass. See effect_animate.cpp.
bool animate(Ability &ab, std::shared_ptr<Orderer> orderer);
// Bootstrap (or refresh) the Creature/Damage components on a permanent the Animate extension
// points (animate_make_creature + animate_set_pt + animate_added_keywords) turned into a
// creature — e.g. an earthbended land. Idempotent; safe to call each SBA pass. Defined in
// effect_animate.cpp.
void apply_animate_creature_bootstrap(Entity e);
// DB$/AB$ Earthbend (Badgermole Cub, Ba Sing Se): the targeted land you control becomes a 0/0
// creature with haste that's still a land (via the Animate extension), gets ab.amount +1/+1
// counters, and a "when it leaves the battlefield, return it tapped" delayed trigger is
// registered. See effect_earthbend.cpp.
bool earthbend(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ Tap (Ba Sing Se LandTapped SVar / generic): tap Defined$ Self or the target. The
// conditional "enters tapped" case is handled by the ENTERS_TAPPED replacement; this resolve
// handler covers a tap that reaches the stack. See effect_tap.cpp.
bool tap(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ ChooseNumber (Wrath of the Skies): the resolving controller chooses an integer in
// [0, Max], where Max is the runtime Count$ expression in ab.dynamic_amount_expr (e.g.
// Count$YourCountersEnergy = their current energy). The pick is recorded in
// cur_game.chosen_number for a chained Count$ChosenNumber reference. General over "choose a
// number up to N" cards. See effect_choose_number.cpp.
bool choose_number(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ DigUntil (Amped Raptor): exile from the top of the library until a card matches Valid$;
// skipped cards go to RevealedDestination$, the match to FoundDestination$ (both Exile here);
// RememberFound$ remembers the match for a chained DB$ Play. See effect_dig_until.cpp.
bool dig_until(Ability &ab, std::shared_ptr<Orderer> orderer);
// DB$ Play (Amped Raptor): grant a one-shot permission to cast a Defined$ card from its
// current zone this turn, paying an alternative RESOURCE cost (PlayCost$) instead of mana.
// See effect_play.cpp.
bool play(Ability &ab, std::shared_ptr<Orderer> orderer);


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
bool parse_reveal(Ability &ab, const std::string &key, const std::string &value);
bool parse_amass(Ability &ab, const std::string &key, const std::string &value);
bool parse_choose_number(Ability &ab, const std::string &key, const std::string &value);
bool parse_dig_until(Ability &ab, const std::string &key, const std::string &value);
bool parse_play(Ability &ab, const std::string &key, const std::string &value);

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
// The unless-cost payment kind for run_unless_loop: pay {N} generic mana (default), pay N life
// (Ward—Pay life, CR 702.21), or discard N card(s) from hand (Reality Smasher, CR 701.8). Returns
// true if the spell should be countered (payer declined or couldn't pay).
enum class UnlessPayKind { MANA, LIFE, DISCARD };
bool run_unless_loop(size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
                     UnlessPayKind kind = UnlessPayKind::MANA);

#endif /* EFFECTS_H */
