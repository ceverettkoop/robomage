#ifndef ACTION_PROCESSOR_H
#define ACTION_PROCESSOR_H

#include <memory>
#include <vector>
#include "classes/action.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Forward declarations
class Orderer;

// Process a legal action selected by the user
void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer);

// Handle the current mandatory choice (declare attackers, blockers, etc.)
void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer);

// Cast during resolution (CR 608.2g): a resolving effect lets `caster` cast `card` (in exile) as
// part of its resolution, for the cost `grant` puts in place of the card's mana cost (FREE =
// "without paying its mana cost"). Offers "cast it / don't" when the cast is possible (timing
// ignored; targets, prohibitions and the cost checked by exile_grant_castable), then runs the
// ordinary cast flow: the spell becomes the topmost object on the stack, above the still-resolving
// ability, and no player receives priority after it is cast. Suspendable: the offer is asked
// through `ctx`, and a cast prompt parks as a CAST query that the main loop resumes before
// re-entering the resolution — the calling handler returns SUSPENDED on SUSPENDED and calls this
// again with the same `rt` when re-entered. CAST when the spell was cast, DECLINED when it
// wasn't (declined, not castable, or the cast was cancelled).
enum class ResolutionCastStatus { CAST, DECLINED, SUSPENDED };
ResolutionCastStatus cast_during_resolution(Entity card, Zone::Ownership caster,
                                            Game::ImpulseCastPermission grant,
                                            ResolutionCastRt& rt, FrameCtx& ctx,
                                            std::shared_ptr<Orderer> orderer);

// Loop-top dispatcher entry for a parked combat target sub-prompt (PendingQuery
// tags ATTACK_TARGET / BLOCK_TARGET): commits the latched answer onto the
// creature persisted in Game::pending_attacker / pending_blocker and clears the
// pending query. Called from the main loop's pending-query branch.
void resume_combat_target_choice(Game& game);

// Loop-top dispatcher entry for a parked cast-time prompt (PendingQuery tag
// CAST): consumes the latched answer and re-enters run_cast_flow — the
// persisted CAST_SPELL state machine in Game::pending_cast. The resume may arm
// the NEXT cast prompt (the caller must loop back to the pending-query branch
// while pending_query.active), cancel the cast (payment rewind), or complete it
// (spell on the stack + game.take_action(), exactly the blocking branch's end).
void resume_cast_flow(Game& game, std::shared_ptr<Orderer> orderer);

// Loop-top dispatcher entry for a parked activated-ability prompt (PendingQuery
// tag ACTIVATION): consumes the latched answer and re-enters
// run_activation_flow — the persisted ACTIVATE_ABILITY state machine in
// Game::pending_activation. The resume may arm the NEXT activation prompt (the
// caller must loop back to the pending-query branch while pending_query.active),
// cancel the activation (payment rewind), or complete it (mana produced
// off-stack, or the ability on the stack + game.take_action(), exactly the
// blocking branch's end).
void resume_activation_flow(Game& game, std::shared_ptr<Orderer> orderer);

// Loop-top dispatcher entry for a parked combat damage-assignment pick
// (PendingQuery tag DAMAGE_ASSIGN): applies the latched answer to the in-flight
// attacker persisted in Game::pending_damage, then either arms the next pick's
// query (same or next attacker — the caller must loop back to the pending-query
// branch when pending_query.active is still set) or completes the assignment,
// after which process_turn_based_actions proceeds to deal_combat_damage.
void resume_damage_assignment(Game& game, std::shared_ptr<Orderer> orderer);

// T3.10: true if some attacker this combat-damage step needs its controller to divide damage
// among 2+ blockers it cannot all kill (and hasn't already been asked). When true, the combat
// step requests ASSIGN_COMBAT_DAMAGE_CHOICE before dealing damage.
bool any_attacker_needs_damage_assignment(Game& game, std::shared_ptr<Orderer> orderer,
                                          bool first_strike_only);

// Returns true if the ability has no targeting requirement or at least one legal target exists.
// "Requirement" is the ability's minimum target count (CR 601.2c) as effective_target_min reads it
// before X is announced: an X-driven minimum counts as 0, since X may legally be 0.
bool has_legal_targets(const Ability& ability, std::shared_ptr<Orderer> orderer);

// The transient targeting ability an Aura's enchant ability defines (CR 303.4a): its legal
// objects are those its Enchant filter names (CardData::enchant_filter), judged from `chooser`'s
// perspective (the filter is controller-relative: Sheltered by Ghosts' Creature.YouCtrl), and a
// graveyard-card filter (Animate Dead) searches graveyards. One builder for every enchant pick —
// the cast-offer gate, the cast-time target, the choice as an uncast Aura enters, and the
// resolution re-check — so they never disagree about what the Aura may enchant.
Ability enchant_target_ability(Entity aura, const CardData &cd, Zone::Ownership chooser);

// True if `aura`'s recorded pending_aura_target is still a legal object for it to enchant
// (CR 608.2b / 608.3b): still the same object (CR 400.7) and still matching its enchant ability
// for `controller`. False with no recorded target.
bool pending_aura_target_legal(Entity aura, Zone::Ownership controller);

// CR 601.2c cast-legality target check across a spell's reachable modes. Returns true if every
// required target (of the primary spell ability and any targeting sub-ability) can be legally
// chosen for at least one reachable set of choices — in particular, for a Gift spell, the
// not-promised OR the promised mode (which switch which ability actually requires a target).
bool spell_has_castable_targets(const Ability& primary, std::shared_ptr<Orderer> orderer,
                                Zone::Ownership caster, bool has_gift);

// Stamp the real casting source/controller onto a spell-ability TEMPLATE (from CardData::abilities,
// whose source is 0) and its targeting sub-abilities, returning the stamped copy for the
// cast-legality target probe. Cast-time source-dependent target restrictions — protection from the
// spell's color (CR 702.16e; e.g. Emrakul vs a white spell, Scryb Ranger vs a blue spell), and the
// OppCtrl/YouCtrl perspective — are evaluated off the ability's source/controller. Without this the
// gate probes with source 0, so has_protection_from(cand, 0) is vacuously false and a protected
// creature is offered as a legal target; select_target then re-checks with the real source, finds
// none, and aborts (CR 601.2c). `card_entity` is the card being cast, which is the SAME entity that
// becomes the spell on the stack, so its source matches select_target's exactly.
Ability cast_gate_probe(const Ability& tmpl, Entity card_entity, Zone::Ownership caster);

// Prompts the active player to choose a target and sets ability.target.
// Targets are presented opponent-first so action index 0 always refers to an
// opponent entity (player or permanent), regardless of which player is casting.
// Caller must ensure has_legal_targets() is true before calling.
void select_target(Ability& ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);

// Suspension-aware form of select_target (the shared target sub-machine; see
// TargetSelectRT / TargetAsker in resolution_frame.h). One call per resume:
// stamps the dynamic min/max bounds once (rt.active), then asks one pick at a
// time through `asker`; SUSPENDED means an ask parked a pending query — the
// caller returns without further mutation and re-enters with the SAME rt when
// the answer is latched. select_target is this machine run with a blocking
// asker; the caller must have seated priority at the choosing player.
TargetStatus run_target_select(Ability& ability, TargetSelectRT& rt, TargetAsker& asker,
                               std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);

// General "copy a spell on the stack" machine (CR 707.10 / 707.12), resumable (Batch 10).
// Creates `count` independent copies of the spell entity `original` on top of the stack,
// controlled by `controller`. Each copy is a copy of the spell's characteristics
// (CardData/color/Ability), is NOT cast (pays no costs, fires no cast triggers), and may CHOOSE
// NEW TARGETS — each copy re-runs target selection through `asker` (copies with no legal
// required target are simply not created). The copies are marked Spell::is_copy so they cease
// to exist on resolution. Reusable by any copy-spell effect (Replicate drives it at cast
// FINISH with the CAST-tag asker; Storm at resolution with the ResolutionTargetAsker).
// copy_spell_begin seeds the rt (left inactive — a no-op run — when count <= 0 or the original
// has no CardData, the old early-outs); run_copy_spell drives it to DONE or returns SUSPENDED
// with a copy's target pick parked — re-enter with the SAME rt once the answer is latched (the
// partially built copy entity and its in-flight ability persist in the rt across suspension).
void copy_spell_begin(CopySpellRT& rt, Entity original, int count, Zone::Ownership controller);
TargetStatus run_copy_spell(CopySpellRT& rt, TargetAsker& asker, std::shared_ptr<Orderer> orderer);

// The single "targets were chosen" hook (CR 601.2c / 602.2b / 603.3d / 707.10): call it once
// the stack object `targeting_entity` (a spell, a copy of a spell, or an activated or triggered
// ability, controlled by `controller`) is on the stack with all of its targets chosen. Every
// object it targets — through its own "target", any chosen mode, any chained sub-ability, or an
// Aura spell's enchant ability (CR 115.1b) — becomes its target once: each Ward triggers
// (CR 702.21a), queued to go on the stack with the next trigger placement (CR 603.3b), and a
// BECAME_TARGET event fires for each targeted permanent (Mode$ BecomesTarget, CR 603.2c). No-op
// for an object with no targets.
void fire_targeting_hooks(Entity targeting_entity, Zone::Ownership controller);

// Evaluates ability.condition_present against ability.condition_compare for `controller`.
// Domain is battlefield permanents matching the filter's type and YouCtrl/OppCtrl qualifier,
// unless ability.condition_on_remembered is set, in which case it counts the remembered
// entities (cur_game.remembered_entities). An empty condition_present returns true; an empty
// condition_compare defaults to ">= 1". Shared by spell castability, trigger intervening-ifs
// (603.4), and ConditionDefined$ Remembered subability gates.
bool evaluate_present_condition(const Ability& ability, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);

#endif
