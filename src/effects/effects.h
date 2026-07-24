#ifndef EFFECTS_H
#define EFFECTS_H

#include <memory>
#include <string>

#include "../components/ability.h"
#include "../resolution_frame.h"
#include "effect_kind.h"

class Orderer;

// ── Effect handler dispatch ────────────────────────────────────────────────
//
// Each effect category resolves through a free-function handler with this
// signature. DONE_RUN_SUBS is "run the standard subability-chaining loop
// afterward" — almost every effect; the handful that manage their own
// subability resolution or short-circuit the game (Charm, WinsGame, the
// non-peek PeekAndReveal path) return DONE_NO_SUBS to suppress it, exactly as
// the legacy if/else chain did via early `return`. SUSPENDED means the handler
// parked a decision through `ctx.ask` (see resolution_frame.h) and must be
// re-entered with the latched answer; it propagates as ResolveStatus::SUSPENDED
// up through resolve_top.
namespace effects {

using EffectHandler = HandlerResult (*)(Ability &, std::shared_ptr<Orderer>, FrameCtx &);

// Returns the handler for `kind`, or nullptr if no resolve-time handler exists
// (None / not-yet-migrated). Ability::resolve() falls back to its legacy chain
// when this returns nullptr.
EffectHandler handler_for(EffectKind kind);

// Per-effect handlers (defined one per src/effects/effect_*.cpp).
HandlerResult deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult draw(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Draw `total` cards for `owner` one at a time, offering the dredge
// draw-replacement (CR 702.52a) before each draw through `ctx` (suspendable).
// `done` is the caller's persisted progress counter (a field of its frame rt),
// so a resume re-enters the parked dredge question mid-batch. Returns false
// when a dredge ask suspended (the caller returns SUSPENDED mutating nothing);
// true when every remaining draw completed (or the game ended mid-batch,
// mirroring Orderer::draw's per-draw ended bail). Blocking contexts prompt
// inline, byte-identical to Orderer::draw. Shared by effects::draw and
// sylvan_library's draw-2. Defined in effect_draw.cpp.
bool draw_n_with_replacements(FrameCtx &ctx, std::shared_ptr<Orderer> orderer,
                              Zone::Ownership owner, size_t &done, size_t total);
HandlerResult gain_life(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult lose_life(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult mill(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult untap(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$/AB$ UntapAll (Paradox Engine): untap every battlefield permanent matching ValidCards$
// (controller-scoped). Mass counterpart of single-target Untap. See effect_untap_all.cpp.
HandlerResult untap_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult cleanup(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult multiply_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult phases(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult wins_game(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult prowess_bonus(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult exalted_bonus(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult attach(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult choose_card(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult destroy_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult damage_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult destroy(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult token(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult investigate(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult surveil(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult scry(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult put_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult remove_counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult change_zone(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult change_zone_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// ChangeType$ Remembered.sameName / Targeted.sameName mover, shared by change_zone
// (force_all=false) and change_zone_all (force_all=true).
bool change_zone_same_name(Ability &ab, std::shared_ptr<Orderer> orderer, bool force_all);
HandlerResult counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult charm(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Pyroblast/Hydroblast: ConditionPresent$ <type>.<Color> gates the EFFECT (not the
// target's legality). Returns true if there is no color requirement, or the target
// has the required color. Non-color ConditionPresent specs (e.g. cmcLEX) return true.
bool target_color_condition_met(const Ability &ab, Entity target);
HandlerResult add_mana(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult discard(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult pump(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ PumpAll (Lorehold Charm's pump mode): mass +P/+T and keyword grant, until end of turn,
// to every battlefield permanent matching ValidCards$ (controller-scoped by YouCtrl/OppCtrl).
// General over "creatures you control get +X/+Y and gain <keyword> until end of turn". Reuses
// the single-target Pump application + amount-resolution helpers below. See effect_pump_all.cpp.
HandlerResult pump_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Apply one creature's until-end-of-turn +P/+T and granted keyword(s) (CR 514.2/611.2b cleanup
// bucket). Shared by single-target Pump and mass PumpAll. Defined in effect_pump.cpp.
void apply_pump_to_creature(Entity target, int pump_att, int pump_def, const PumpParams *pp);
// Resolve a PumpParams' NumAtt$/NumDef$ (static literal or count-SVar) at resolution. Shared by
// single-target Pump and mass PumpAll. Defined in effect_pump.cpp.
void resolve_pump_amounts(const PumpParams *pp, Zone::Ownership ctrl,
                          std::shared_ptr<Orderer> orderer, Entity target,
                          int &out_att, int &out_def);
HandlerResult peek_and_reveal(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult reveal(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ RevealHand (Thought-Knot Seer): the targeted player (ValidTgts$ Opponent) reveals their
// hand to all players (CR 701.16) — logged and recorded in the belief state. General over
// Duress/Thoughtseize-style "target player reveals their hand". See effect_reveal_hand.cpp.
HandlerResult reveal_hand(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult dig(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult sylvan_library(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult amass(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult sacrifice(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult put_counter_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult sacrifice_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult immediate_trigger(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
HandlerResult copy_permanent(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// AB$ Clone (Thespian's Stage): the SOURCE permanent becomes a copy of target land in place —
// the same object acquires the target's copiable characteristics (CR 706.2), keeping its own
// counters/tapped state/attachments. GainThisAbility$ True retains this Clone ability on the copy.
HandlerResult clone(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Mobilize N (702.176): create N tapped+attacking 1/1 red Warrior tokens and register a
// delayed end-step sacrifice of exactly those tokens.
HandlerResult mobilize(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Delayed end-step sacrifice fired by Mobilize: sacrifice each entity in ab.targets that is
// still on the battlefield (the tokens created when the creature attacked).
HandlerResult sacrifice_tokens(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Delayed end-of-combat exile fired by AtEOT$ ExileCombat (Geist of Saint Traft): exile each
// entity in ab.targets still on the battlefield (the token(s) created when the creature attacked).
HandlerResult exile_tokens(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// RepeatEach over players (Price of Progress): resolve the RepeatSubAbility once per
// player, with cur_game.remembered_entities set to that player's entity each iteration.
HandlerResult repeat_each(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// AB$ Effect granting "you may cast that card this turn" (Emry): records the targeted
// graveyard card in cur_game.may_cast_this_turn so the casting path offers it this turn.
HandlerResult grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ BecomeMonarch (CR 725, Forth Eorlingas!): the ability's controller becomes the monarch.
HandlerResult become_monarch(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// SP$/AB$ Vote (Council's Judgment, CR 701.32 will-of-the-council): voters choose among the
// VoteCard$ permanents and the VoteSubAbility$ effect (a DB$ ChangeZone Battlefield→Exile) is
// applied to the winner(s). The engine is two-player only (CLAUDE.md scope), so the vote
// reduces to the spell/ability's controller CHOOSING one permanent matching VoteCard$ — a
// choice, not a target, so it ignores shroud/hexproof. See effect_vote.cpp.
HandlerResult vote(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// K:Storm (CR 702.40): the synthesized self-cast triggered ability resolves here. ab.amount holds
// the storm count (spells cast before the storm spell this turn, by either player), locked in when
// the trigger fired. Puts that many copies of the source spell on the stack (each may choose new
// targets) via the shared run_copy_spell machine (suspendable). See effect_storm.cpp.
HandlerResult storm(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ CopySpellAbility (Chain Lightning): after the parent spell's effect, the target-or-its-
// controller may pay an UnlessCost$ ({R}{R}) to copy the parent spell and choose new targets for
// the copy (UnlessSwitched$ True = copy ONLY IF paid). Reuses run_unless_loop + the shared
// run_copy_spell machine (suspendable). See effect_copy_spell.cpp. CR 707.10 / 707.12.
HandlerResult copy_spell_ability(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// SP$/DB$ NameCard (Cabal Therapy): the ability's controller names a card (CR 201.4); the
// chosen name is stored in cur_game.named_card so a chained Card.NamedCard sub-ability
// (here a RevealDiscardAll discard) can reference it.
HandlerResult name_card(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ Animate (Guide of Souls): the targeted permanent "becomes ..." — bakes the
// Duration$ Permanent continuous effect (added types/subtypes, and the extension-point
// base-P/T / keyword / creature grants) onto the permanent so the layer system reapplies
// it each SBE pass. See effect_animate.cpp.
HandlerResult animate(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// AB$ AnimateAll (Shadowspear): a mass until-end-of-turn continuous effect that removes
// (RemoveKeywords$) or grants keyword(s) on every battlefield permanent matching ValidCards$.
// General over "permanents matching <filter> lose/gain <keyword> until end of turn" (CR 613
// layer 6). See effect_animate_all.cpp.
HandlerResult animate_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Bootstrap (or refresh) the Creature/Damage components on a permanent the Animate extension
// points (animate_make_creature + animate_set_pt + animate_added_keywords) turned into a
// creature — e.g. an earthbended land. Idempotent; safe to call each SBA pass. Defined in
// effect_animate.cpp.
void apply_animate_creature_bootstrap(Entity e);
// Lapse every Duration$ UntilYourNextTurn Animate (Karn, the Great Creator +1) created by
// `active_player`; call from that player's untap step. Defined in effect_animate.cpp.
void revert_until_turn_animates(Zone::Ownership active_player);
// DB$/AB$ Earthbend (Badgermole Cub, Ba Sing Se): the targeted land you control becomes a 0/0
// creature with haste that's still a land (via the Animate extension), gets ab.amount +1/+1
// counters, and a "when it leaves the battlefield, return it tapped" delayed trigger is
// registered. See effect_earthbend.cpp.
HandlerResult earthbend(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ Tap (Ba Sing Se LandTapped SVar / generic): tap Defined$ Self or the target. The
// conditional "enters tapped" case is handled by the ENTERS_TAPPED replacement; this resolve
// handler covers a tap that reaches the stack. See effect_tap.cpp.
HandlerResult tap(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ ChooseNumber (Wrath of the Skies): the resolving controller chooses an integer in
// [0, Max], where Max is the runtime Count$ expression in ab.dynamic_amount_expr (e.g.
// Count$YourCountersEnergy = their current energy). The pick is recorded in
// cur_game.chosen_number for a chained Count$ChosenNumber reference. General over "choose a
// number up to N" cards. See effect_choose_number.cpp.
HandlerResult choose_number(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ DigUntil (Amped Raptor): exile from the top of the library until a card matches Valid$;
// skipped cards go to RevealedDestination$, the match to FoundDestination$ (both Exile here);
// RememberFound$ remembers the match for a chained DB$ Play. See effect_dig_until.cpp.
HandlerResult dig_until(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ Play (Amped Raptor): grant a one-shot permission to cast a Defined$ card from its
// current zone this turn, paying an alternative RESOURCE cost (PlayCost$) instead of mana.
// See effect_play.cpp.
HandlerResult play(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ AddTurn (Emrakul, the Aeons Torn): the Defined$ player (You = the ability's controller)
// takes ab.amount (NumTurns$, default 1) extra turns after the current one, queued onto
// cur_game.extra_turns and consulted at turn hand-off (CR 500.7 / 720). General over any "take
// an extra turn" effect. See effect_add_turn.cpp.
HandlerResult add_turn(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// DB$ StoreSVar (Carpet of Flowers): latch ab.stored_svar_set_value into the SOURCE permanent's
// Permanent::stored_svars[ab.stored_svar_set_name] — the per-permanent named-integer scratch store
// read back by a CheckSVar trigger gate. No-op if the source is not a battlefield permanent (e.g.
// the leave-battlefield reset, whose Permanent is already gone). See effect_store_svar.cpp.
HandlerResult store_svar(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
// Suspend upkeep tick (CR 702.62a, second/third abilities): remove one suspend time counter from
// ab.source (an exiled suspended card, tracked in cur_game.suspend_time_counters — an exiled card
// is not a permanent, so its counters can't live in Permanent::counters). When the last counter is
// removed, grant its owner a FREE from_suspend impulse-cast permission so it may be cast without
// paying its mana cost (the third suspend ability). General over any Suspend card. See
// effect_suspend_tick.cpp.
HandlerResult suspend_tick(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);


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
// AB$ AnimateAll RemoveKeywords$/AddKeyword$/ValidCards$ (Shadowspear). See effect_animate_all.cpp.
bool parse_animate_all(Ability &ab, const std::string &key, const std::string &value);
// DB$ StoreSVar SVar$/Expression$/Type$ (Carpet of Flowers). See effect_store_svar.cpp.
bool parse_store_svar(Ability &ab, const std::string &key, const std::string &value);

}  // namespace effects

// ── Shared resolution helpers ───────────────────────────────────────────────
// De-static'd from ability.cpp so effect handlers in separate translation units
// can reuse them. Definitions still live in ability.cpp for now.
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target,
    Entity source = 0);
// Search a zone for cards matching the comma-separated type list in change_type
// (empty change_type matches all cards in the zone).
// When mandatory=true, "fail to find" is suppressed unless the zone is empty.
// Returns the chosen Entity, or 0 if the player fails to find / zone is empty.
// reveal=true marks every offered card choice as public knowledge (revealed
// tutors), so observers may show the chosen card's name even into a hidden zone.
// cmc_bound (>= 0) plus cmc_op ("EQ"/"LE"/...) additionally gate candidate cards by
// mana value (Aether Vial: MV == charge-counter count, resolved by the caller); -1 = none.
// The pick asks through `ctx` (with `decision_source` as the pending-decision
// context — the calling handler's ab.source); if the ask suspends, `suspended`
// is set and 0 is returned mutating nothing — the caller must propagate
// SUSPENDED. The candidate scan and menu rebuild identically on resume.
// `chain_target` is the resolving chain's card target for a ChangeType `targetedBy` filter
// alternative (Cloak and Dagger, Entwined); 0 = none.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    Zone::ZoneValue zone, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal, int cmc_bound, const std::string &cmc_op,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target = 0);
Entity search_multi_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    const std::vector<Zone::ZoneValue> &zones, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target = 0);
// The unless-cost payment kind for run_unless_loop: pay {N} generic mana (default), pay N life
// (Ward—Pay life, CR 702.21), discard N card(s) from hand (Reality Smasher, CR 701.8), or pay N
// energy ({E}, CR 122.1c — Static Prison's "unless you pay {E}"). Returns true if the prevented
// effect should still happen (payer declined or couldn't pay). The LIFE/ENERGY yes-no and the
// DISCARD flow ask through `ctx` and may suspend (`suspended` set, return value meaningless —
// check it FIRST); the MANA tap-for-mana loop is still a blocking prompt (Shape C, Batch 7).
enum class UnlessPayKind { MANA, LIFE, DISCARD, ENERGY };
// `cost_pips` (MANA kind only) overrides the default generic {cost} requirement with exact colored
// pips (Chain Lightning: {R}{R}); when null/empty the cost is `cost` generic mana as before.
bool run_unless_loop(size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
                     FrameCtx &ctx, bool &suspended, UnlessPayKind kind = UnlessPayKind::MANA,
                     const ManaValue *cost_pips = nullptr);

#endif /* EFFECTS_H */
