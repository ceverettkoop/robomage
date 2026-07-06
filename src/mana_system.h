#ifndef MANA_SYSTEM_H
#define MANA_SYSTEM_H

#include <cstddef>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "classes/action.h"
#include "classes/colors.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;
struct Permanent;
struct Ability;
struct HybridPip;
struct Player;

// Get player entity from ownership
Entity get_player_entity(Zone::Ownership player);

// True if `ab` is a mana ability (CR 605): one that adds mana and resolves at activation
// without using the stack. Covers the ordinary AddMana producers AND AB$ ManaReflected
// (Mox Amber), whose producible colors are computed dynamically. Single source so every
// "is this a mana ability?" gate (action processor, legal-action enumeration, mana-source
// collection) classifies ManaReflected consistently and none push it onto the stack.
bool ability_is_mana(const Ability &ab);

// Bump the per-turn activation counter for the ability on `perm` that matches `ability`,
// if that ability is activation-limited (no-op when activation_limit == 0). Mana abilities
// (category "AddMana") are keyed by tap-cost + color, since one permanent can expose several
// AddMana abilities of different colors; every other activated ability is keyed by its
// return-to-hand cost (Return<N/Type>). Shared by the spell/ability processor and both mana
// payers so the (formerly four-times duplicated) match rule lives in one place.
void increment_activation_count(Permanent &perm, const Ability &ability);

// Narrative style for the mana-production line each activation path emits. The three
// call paths historically log slightly different strings (verb, printed amount, symbol
// braces); the style keeps each path's output byte-identical while the logic is shared.
enum class ManaLogStyle {
    ACTIVATED,      // "%s activated %s for %zu(%s)\n"  — auto-payer / interactive payer
    TAPPED_AMOUNT,  // "%s tapped %s for %zu(%s)\n"     — priority-menu activation
    TAPPED_SYMBOL,  // "%s tapped %s for {%s}\n"        — pay-unless payment loop
};

// The production half of a mana-ability activation, assuming ALL costs are already
// paid: evaluate the produced amount (dynamic amounts included), apply ProduceMana
// replacement effects, add the mana to `pool`, resolve TapsForMana mana-additional
// triggers, set the uncounterable flag for adds_no_counter sources, log (commit only),
// resolve SubAbility$ riders (commit only), and bump the activation counter.
// Replacements and triggers run in simulate mode (!commit) too, so affordability
// (can_pay_mana) and the real payment always agree on the mana produced.
void produce_mana_from_ability(Entity source, const Ability& ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer, ManaValue& pool,
                               bool commit, ManaLogStyle log_style);

// Activate one mana source: pay its activation mana cost from the working `pool`,
// tap/sacrifice it, pay its life cost, then produce via produce_mana_from_ability.
// Pool changes (activation cost paid, mana produced) always apply to the working
// `pool`; the write-only ECS side effects are skipped when !commit (simulate mode).
// Returns false — with NO side effects (no tap, no sacrifice, no mana produced, pool
// untouched) — when the ability's activation mana cost (Talon Gates' {1}{T}) cannot be
// paid from the working pool. The cost is paid FIRST, before any other effect, so a
// refusal cancels cleanly. Shared by the auto-payer (commit per simulate/real), the
// interactive payer, and the pay-unless loop (both always commit, with pool == the
// player's real mana pool).
bool activate_mana_source(Entity source, const Ability& ab, Zone::Ownership controller,
                          std::shared_ptr<Orderer> orderer, ManaValue& pool, Player& player,
                          bool commit, ManaLogStyle log_style);

// Check if a given mana pool can afford a cost (does not read player state)
bool can_afford_pool(const std::multiset<Colors>& pool, const std::multiset<Colors>& cost);

// Check if player can afford a mana cost
bool can_afford(Zone::Ownership player, const std::multiset<Colors>& cost);

// Check if player can afford a cost considering current pool plus all activatable mana sources
// exclude_entity: if nonzero, skip this entity when counting available sources (used when
// the ability being checked will tap the source as part of its own cost)
// paid_for: if nonzero, only count restricted mana sources (Cavern of Souls) when the
// spell matches the restriction (creature of chosen type)
// NOTE: this is a hypothetical-pool HEURISTIC (flexible-count accounting for multi-color
// sources), not the simulate-mode payer. can_pay_mana is the exact predicate; this one is
// kept for the cost-bearing-source gates, where changing it would shift legality menus.
bool can_afford_with_sources(Zone::Ownership player, const std::multiset<Colors>& cost,
                             std::shared_ptr<Orderer> orderer, Entity exclude_entity = 0,
                             Entity paid_for = 0);

// Return the maximum total mana producible (pool + untapped sources) minus colored base cost
// obligations; used for computing max X value
// NOTE: a coarse upper bound — ignores color feasibility of the base cost's pips and counts
// one ability per source entity. An overestimated X fails at payment and is absorbed by the
// payment_fail_counts rewind, so the bound is deliberately cheap rather than exact.
size_t max_available_mana(Zone::Ownership player, const ManaValue& base_cost,
                          std::shared_ptr<Orderer> orderer);

// Spend mana from player's pool (assumes can_afford was checked)
// paid_for: the entity being paid for (used for diagnostics on insufficient mana)
void spend_mana(Zone::Ownership player, const std::multiset<Colors>& cost, Entity paid_for);

// Add mana to player's pool
void add_mana(Zone::Ownership player, Colors mana_color, size_t amount);

// Pay as much of cost as possible from player's pool, return the unpaid remainder
ManaValue pay_partial(Zone::Ownership player, const ManaValue& cost);

// Empty player's mana pool (called at step transitions)
void empty_mana_pool(Zone::Ownership player);

// Snapshot of mana-related state for rewind on payment failure
struct ManaPaymentSnapshot {
    std::multiset<Colors> player_mana;
    std::vector<std::pair<Entity, bool>> tapped_state;  // entity, was_tapped
    std::vector<std::tuple<Entity, size_t, int>> activation_counts;  // entity, ability_idx, old count
    std::vector<Entity> delve_exiled;  // snapshot of cur_game.delve_exiled for delve rewind
};

ManaPaymentSnapshot snapshot_mana_state(Zone::Ownership player, std::shared_ptr<Orderer> orderer);
void restore_mana_state(Zone::Ownership player, const ManaPaymentSnapshot& snap,
                        std::shared_ptr<Orderer> orderer);

// The activation mana cost of `ab` AFTER applying its ReduceCost$ (CR 601.2f). When the
// ability declares a ReduceCost$ (a literal integer or a Count$/SVar expression such as
// Eiganjo's "Count$Valid Creature.Legendary+YouCtrl"), the resolved amount is subtracted from
// the GENERIC portion of activation_mana_cost (floored at 0); colored pips are never reduced.
// `controller` is the counting "you" for the dynamic reduction. Returns the unmodified cost
// when no ReduceCost$ is set. The single source of truth used by BOTH the affordability check
// and the actual payment so legality and payment can never diverge.
ManaValue effective_activation_mana_cost(const Ability &ab, Zone::Ownership controller,
                                         std::shared_ptr<Orderer> orderer);

// Map a mana color to its ActionCategory (MANA_W, MANA_U, etc.)
ActionCategory mana_action_category(Colors color);

// Collect legal actions for all activatable mana sources a player controls.
// Each action has a color-specific ActionCategory (MANA_W, MANA_U, etc.).
// Sources with activation_mana_cost are included only if the cost is affordable.
// at_priority includes instant-speed mana abilities (e.g. Lion's Eye Diamond), which may
// only be activated when their controller holds priority — never mid-cost-payment.
std::vector<LegalAction> collect_mana_legal_actions(
    Zone::Ownership player, std::shared_ptr<Orderer> orderer, Entity paid_for = 0,
    bool at_priority = false);

// Can the controller pay `cost`? Runs the real machine-mode payment algorithm in a
// side-effect-free simulate mode, so legality and actual payment can never disagree.
// has_delve folds in graveyard instant/sorcery exiling; paid_for gates restricted mana
// (Cavern of Souls) the same way the payer does.
bool can_pay_mana(Zone::Ownership controller, const std::multiset<Colors>& cost,
                  Entity paid_for, std::shared_ptr<Orderer> orderer, bool has_delve = false,
                  bool has_improvise = false);

// Resolve a card's HYBRID pips (CR 107.4) against the caster's available mana. Each color-hybrid
// pip ({W/U}) may be paid by one mana of either listed color; each twobrid pip ({2/W}) by one
// mana of its color OR its generic-mana alternative. Enumerates the assignments (colored option
// tried first, so a payable colored choice is preferred) and returns true on the FIRST assignment
// that, added to `base_flat_cost`, is fully payable per can_pay_mana — writing that concrete flat
// cost into *out_resolved when non-null. With an empty `hybrids` this is exactly
// can_pay_mana(base_flat_cost) (and copies it to *out_resolved). Single source shared by the
// cast-legality gate and the machine/auto payment path so castability and payment agree on the
// hybrid choice.
bool resolve_hybrid_cost(Zone::Ownership caster, const std::multiset<Colors>& base_flat_cost,
                         const std::vector<HybridPip>& hybrids, Entity paid_for,
                         std::shared_ptr<Orderer> orderer, bool has_delve = false,
                         bool has_improvise = false, std::multiset<Colors>* out_resolved = nullptr);

// Prompt the player to activate mana abilities to pay a cost. Returns true if cost was fully paid.
// On false, caller must restore from snapshot.
// If has_delve is true, delve exile options are included alongside mana abilities for generic costs.
bool prompt_mana_payment(Zone::Ownership controller, const ManaValue& cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve = false, bool has_improvise = false);

// Interactive Delve (CR 702.66 / 601.2h): the caster first chooses HOW MANY graveyard
// cards to exile (a CHOOSE_X count menu constrained to counts whose remaining
// cost is still payable, so any offered pick is safe), then WHICH cards, one CHOOSE_CARD
// pick at a time (each exile removes that card from the next menu). Exiled cards are
// recorded in cur_game.delve_exiled (for etbCounter counts, e.g. Murktide Regent) and each
// exile removes one GENERIC pip from `cost`. Call before prompt_mana_payment and then pay
// the reduced cost with has_delve=false. No-op when the cost has no generic portion or the
// graveyard offers nothing to exile.
void prompt_delve_exiles(Zone::Ownership controller, ManaValue& cost, Entity paid_for,
                         std::shared_ptr<Orderer> orderer, bool has_improvise = false);

#endif
