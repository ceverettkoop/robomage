#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include "../components/static_ability.h"
#include "../classes/action.h"
#include "../classes/colors.h"
#include <vector>
#include <string>
#include <memory>
#include <set>

struct CardData;

struct Deck;
struct Game;

class Orderer;
class StackManager;

// Cached snapshot of an active static ability on the battlefield.
// Rebuilt each SBE pass by gather_active_statics() (the continuous-effects engine
// preamble); queried by determine_legal_actions, check_triggered_abilities,
// mana_system, game.cpp untap, etc.
struct ActiveStatic {
    Entity            entity = 0;
    StaticAbility    *sa = nullptr;
    Zone::Ownership   controller = Zone::PLAYER_A;
    bool              condition_met = false;  // evaluated once per gather pass; read by every layer applier
    bool              suppressed = false;     // an ability-removal effect (Humility) removed this static's source's abilities; every layer applier skips it
};

// Global cached list of active static abilities on battlefield permanents.
// Rebuilt every SBE pass. Consumers read this instead of scanning all permanents.
extern std::vector<ActiveStatic> g_active_statics;

// Resolve the set of battlefield permanents a continuous static's Affected$ filter
// designates (CR 611/613 — the objects a continuous effect applies to). The filter is
// evaluated through the shared permanent_matches_filter so the full qualifier grammar
// (colors, Colorless, subtypes, P/T, …) is honoured; YouCtrl/OppCtrl are controller-scoped
// to the static's controller and `+Other` excludes the source permanent itself. Returns an
// empty list for the EquippedBy / Self forms (those resolve to a single creature in the
// applier) or for an empty filter. This is the single, reusable "which permanents does this
// static affect" step — the additive-P/T anthem path (It That Heralds the End) uses it, and
// future Affected$-filter statics (AddAbility$, AddKeyword$ anthems) reuse it too.
std::vector<Entity> affected_permanents_for_static(const ActiveStatic &as,
                                                   const std::set<Entity> &entities);

// Total generic mana that active RaiseCost statics add to the cost of casting `card_data`.
// Honours the nonCreature filter and the NamedCard filter (Disruptor Flute): a NamedCard
// RaiseCost applies only when the spell's name equals its source's chosen_name. Shared by
// determine_legal_actions (affordability) and action_processor (payment).
int active_raise_cost_for(const CardData &card_data);

// Total generic mana that active ReduceCost statics remove from `card_data`'s cost for
// player `caster` (the mirror of active_raise_cost_for). An Activator$ You static (Eye of
// Ugin) only reduces spells cast by its own controller. The generic portion is clamped at
// zero and colored pips are never reduced (CR 118.7 / 601.2f); applied in effective_base_cost.
int active_reduce_cost_for(const CardData &card_data, Zone::Ownership caster);

// `card_data.mana_cost` with the active RaiseCost generic surcharge folded in (but
// NOT the X-cost choice, which is resolved interactively at cast time). The single
// effective-base-cost builder shared by determine_legal_actions (affordability) and
// action_processor (payment) so the two cannot disagree on the surcharge.
//
// `caster` (when not UNKNOWN) additionally folds in cost reductions that depend on the
// casting player's board — currently Affinity for artifacts (CR 702.41): the generic
// portion is reduced by {1} per artifact the caster controls. CR 601.2f applies all
// such reductions after additions; the generic total never drops below zero.
ManaValue effective_base_cost(const CardData &card_data,
                              Zone::Ownership caster = Zone::UNKNOWN);

class StateManager : public System {

public:
    static void init();
    void state_based_effects(Game& game, std::shared_ptr<Orderer> orderer);
    void process_turn_based_actions(Game& game, std::shared_ptr<Orderer> orderer);
    std::vector<LegalAction> determine_legal_actions(const Game& game, std::shared_ptr<Orderer> orderer,
                                                      std::shared_ptr<StackManager> stack_manager);

private:
    void apply_permanent_components(Game& game, std::shared_ptr<Orderer> orderer);
    void update_city_blessing(Game& game);  // 702.131: grant ascend's city's blessing at 10+ permanents
    void apply_land_abilities(Entity entity);
    void apply_keyword_abilities(Entity entity);
    void deal_combat_damage(Game& game, bool first_strike_only);
    void check_triggered_abilities(Game& game, std::shared_ptr<Orderer> orderer);

    // Continuous-effects engine (rule 613). apply_continuous_effects is the single
    // ordered driver (src/systems/state_manager_layers.cpp); the per-layer appliers
    // and the gather preamble live in state_manager_statics.cpp where the keyword
    // helpers are defined. See continuous_effects.h for the data model.
    void apply_continuous_effects(Game& game);    // layer-ordered driver
    void gather_active_statics(Game& game);       // preamble: reset bonuses (incl. keyword rebuild-from-base), build g_active_statics, eval conditions
    void suppress_removed_statics(Game& game);    // layer 6 (613.1f): mark statics on objects that lose all abilities (Humility) as suppressed
    void apply_type_changing_effects();           // layer 4 (613.1d)
    void apply_layer6_ability_effects();          // layer 6 (613.1f): keyword grant/removal
    void recompute_abilities(Game& game);         // layer 6 (613.1f): erase activated/keyword abilities removed by an ability-removal effect
    void apply_layer7_pt_effects();               // layer 7 (613.4): CDA set + additive P/T
    void apply_rules_modifying_effects();         // 613.11: MustAttack and other rules-modifiers
    void recompute_battlefield_pt();              // flush layer-7 result into cached P/T
};

#endif /* STATE_MANAGER_H */
