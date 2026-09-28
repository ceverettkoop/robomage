#ifndef EFFECTS_H
#define EFFECTS_H

#include <memory>
#include <set>
#include <string>

#include "../choice_labels.h"
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
// non-peek PeekAndReveal path) return DONE_NO_SUBS to suppress it. SUSPENDED means the handler
// parked a decision through `ctx.ask` (see resolution_frame.h) and must be
// re-entered with the latched answer; it propagates as ResolveStatus::SUSPENDED
// up through resolve_top.
namespace effects {

using EffectHandler = HandlerResult (*)(Ability &, std::shared_ptr<Orderer>, FrameCtx &);

// Returns the handler for `kind`, or nullptr if no resolve-time handler exists
// (None, or an EFFECT_KIND_ELSEWHERE category). resolve_ability() just chains the
// subabilities when this returns nullptr.
EffectHandler handler_for(EffectKind kind);

// The per-effect handlers, one per src/effects/effect_*.cpp (documented in effect_kinds.def).
#define EFFECT_KIND(kind, category, handler) \
    HandlerResult handler(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx);
#include "effect_kinds.def"

// Draw `total` cards for `owner` one at a time, offering the dredge
// draw-replacement (CR 702.52a) before each draw through `ctx` (suspendable).
// `done` is the caller's persisted progress counter (a field of its frame rt),
// so a resume re-enters the parked dredge question mid-batch. Returns false
// when a dredge ask suspended (the caller returns SUSPENDED mutating nothing);
// true when every remaining draw completed (or the game ended mid-batch,
// mirroring Orderer::draw's per-draw ended bail). Blocking contexts prompt
// inline, byte-identical to Orderer::draw. Shared by effects::draw and
// sylvan_library's draw-2. `decision_source` (the resolving ability's source) is
// the pending-decision context of each dredge question. Defined in effect_draw.cpp.
bool draw_n_with_replacements(FrameCtx &ctx, std::shared_ptr<Orderer> orderer,
                              Zone::Ownership owner, size_t &done, size_t total,
                              Entity decision_source);
// CR 702.26b/g: phase permanent `e` out directly — it is removed from combat (CR 506.4) and every
// Aura/Equipment attached to it phases out indirectly with it. `entities` is the iterating
// system's mEntities. See effect_phases.cpp.
void phase_out(Entity e, const std::set<Entity> &entities);
// CR 702.26c/g: phase permanent `e` in, together with everything that phased out indirectly
// along with it. Called from the untap step for each permanent that phased out directly under
// the active player's control. See effect_phases.cpp.
void phase_in(Entity e, const std::set<Entity> &entities);
// Where the cards a look-and-split step doesn't keep on top go: the library bottom (scry) or the
// graveyard (surveil).
enum class LookSplitRest { LIBRARY_BOTTOM, GRAVEYARD };
// The choice step shared by scry (CR 701.22a) and surveil (CR 701.25a) over the looked-at cards in
// rt.remaining: one interactive loop in which `looker` picks any remaining card and sends it to
// the top of their library or to `rest`, until none remain. The order cards are sent to the top
// fixes the final library order (the FIRST one ends up topmost); each is put at its final depth
// the moment it is chosen, so it sits on the known-top cache for the remaining choices. Cards sent
// to the bottom go under the ones bottomed before them, so both piles are in any order the looker
// wants. Menu: "Put X on top of library" (TOP_LIBRARY, option_ordinal = the depth it will sit at)
// for each remaining card, then one `rest` option per card (BOTTOM_DECK_CARD / CHOOSE_CARD) — the
// two share the card entity, so they differ by category for the semantic action resolver.
// Returns SUSPENDED when a pick parked. Defined in effect_surveil.cpp.
HandlerResult look_and_split(LookSplitRt &rt, Zone::Ownership looker, LookSplitRest rest,
                             std::shared_ptr<Orderer> orderer, FrameCtx &ctx, Entity source);
// CounterNum$ at resolution: the static count, or its dynamic Count$ expression (CounterNum$ X)
// evaluated for the ability's controller. Shared by PutCounter, PutCounterAll and RemoveCounter.
// Defined in effect_put_counter.cpp.
int resolve_counter_num(const Ability &ab, const CounterParams &cp, std::shared_ptr<Orderer> orderer);
// Put card `e` onto the battlefield by an effect rather than by resolving as a spell — the one
// entry every effect that "puts a card onto the battlefield" shares with ChangeZone: a
// nonpermanent card can't enter (CR 110.4a), and an Aura first chooses what it will enchant, or
// with no legal object stays where it is (CR 303.4f/g). `fctx` carries that pick (see
// change_zone_move). Returns the zone the card is in afterwards. Defined in effect_change_zone.cpp.
Zone::ZoneValue put_onto_battlefield(const std::shared_ptr<Orderer> &orderer, FrameCtx fctx, Entity e);
// ChangeType$ Remembered.sameName / Targeted.sameName mover, shared by change_zone
// (force_all=false) and change_zone_all (force_all=true).
bool change_zone_same_name(Ability &ab, std::shared_ptr<Orderer> orderer, bool force_all);
// Pyroblast/Hydroblast: ConditionPresent$ <type>.<Color> gates the EFFECT (not the
// target's legality). Returns true if there is no color requirement, or the target
// has the required color. Non-color ConditionPresent specs (e.g. cmcLEX) return true.
bool target_color_condition_met(const Ability &ab, Entity target);
// Apply one creature's until-end-of-turn +P/+T and granted keyword(s) (CR 514.2/611.2b cleanup
// bucket). Shared by single-target Pump and mass PumpAll. Defined in effect_pump.cpp.
void apply_pump_to_creature(Entity target, int pump_att, int pump_def, const PumpParams *pp);
// Resolve a PumpParams' NumAtt$/NumDef$ (static literal or count-SVar) at resolution. Shared by
// single-target Pump and mass PumpAll. Defined in effect_pump.cpp.
void resolve_pump_amounts(const PumpParams *pp, Zone::Ownership ctrl,
                          std::shared_ptr<Orderer> orderer, Entity target,
                          int &out_att, int &out_def);
// Bootstrap (or refresh) the Creature/Damage components on a permanent the Animate extension
// points (animate_make_creature + animate_set_pt + animate_added_keywords) turned into a
// creature — e.g. an earthbended land. Idempotent; safe to call each SBA pass. Defined in
// effect_animate.cpp.
void apply_animate_creature_bootstrap(Entity e);
// Lapse every Duration$ UntilYourNextTurn Animate (Karn, the Great Creator +1) created by
// `active_player`; call from that player's untap step. Defined in effect_animate.cpp.
void revert_until_turn_animates(Zone::Ownership active_player);



// ── Effect-specific parse hooks ─────────────────────────────────────────────
//
// Co-located with each effect's resolve handler: each hook owns the card-script
// param keys that are exclusive to that effect, returning true if it consumed
// the (key, value). The parser's generic apply_param_to_ability handles the
// shared keys and delegates anything left over to apply_parse_hook(), which
// tries each hook in turn. Keys are partitioned so at most one hook claims any
// given key — relocation is therefore byte-identical to the old flat parser.
bool apply_parse_hook(AbilityDef &ab, const std::string &key, const std::string &value);

bool parse_deal_damage(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_pump(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_token(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_add_mana(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_destroy_all(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_change_zone(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_put_counter(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_dig(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_delayed_trigger(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_discard(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_mill(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_peek_and_reveal(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_reveal(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_amass(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_choose_number(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_dig_until(AbilityDef &ab, const std::string &key, const std::string &value);
bool parse_play(AbilityDef &ab, const std::string &key, const std::string &value);
// AB$ AnimateAll RemoveKeywords$/AddKeyword$/ValidCards$ (Shadowspear). See effect_animate_all.cpp.
bool parse_animate_all(AbilityDef &ab, const std::string &key, const std::string &value);
// DB$ StoreSVar SVar$/Expression$/Type$ (Carpet of Flowers). See effect_store_svar.cpp.
bool parse_store_svar(AbilityDef &ab, const std::string &key, const std::string &value);
// DB$ SetState Mode$ <mode> (The Creation of Avacyn). See effect_set_state.cpp.
bool parse_set_state(AbilityDef &ab, const std::string &key, const std::string &value);

}  // namespace effects

#endif /* EFFECTS_H */
