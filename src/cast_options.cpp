#include "cast_options.h"

#include <cstddef>
#include <string>

#include "action_processor.h"
#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/player.h"
#include "ecs/coordinator.h"
#include "mana_system.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/costs.h"
#include "queries/permissions.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "queries/spells.h"
#include "queries/types.h"
#include "svar_eval.h"
#include "systems/orderer.h"
#include "systems/rules_modifying.h"
#include "systems/state_manager.h"
#include "targeting.h"

static bool aura_enchant_target_available(const CardData &card_data, Zone::Ownership caster,
                                          std::shared_ptr<Orderer> orderer);
static void offer_modal_back_face_casts(std::vector<LegalAction> &actions, Zone::Ownership priority_player,
                                        const std::vector<Entity> &hand, std::shared_ptr<Orderer> orderer,
                                        bool sorcery_window);
static bool spell_timing_ok(const CardData &face, Zone::Ownership caster, bool sorcery_window);
static bool can_cast_now(const CardData &face, Entity card_entity, Zone::Ownership caster,
                         Zone::ZoneValue from_zone, bool sorcery_window, bool ignore_timing,
                         std::shared_ptr<Orderer> orderer);
static LegalAction cast_action(Entity card_entity, const std::string &desc, int option_ordinal);
static CastCost alternative_cost_kind(const AltCost &alt_cost);
static ManaValue regular_cast_cost(
    const CardData &face, Entity card_entity, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static CostCheck total_cost_check(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                  const std::shared_ptr<Orderer> &orderer);
static bool regular_cast_payable(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                 const ManaValue &mana, std::shared_ptr<Orderer> orderer);
static void offer_free_cast(std::vector<LegalAction> &actions, const LegalAction &cast, const CardData &face,
    Zone::ZoneValue from_zone, Zone::Ownership caster, bool sorcery_window, std::shared_ptr<Orderer> orderer);
static bool can_afford_alt(const CardData &card_data, const AltCost &alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer);

// May `caster` cast a spell with `face`'s characteristics now, as far as timing goes (CR 601.3,
// 307.1)? An instant, a Flash card, or a spell a cast-with-flash permission covers (Teferi, Time
// Raveler's +1, CR 702.8) may be cast any time its caster holds priority; any other spell only in
// the caster's sorcery window. An opponent's sorcery-speed lock (Teferi's static "each opponent can
// cast spells only any time they could cast a sorcery") overrides both, so under it every spell —
// even an instant, or one the caster's own permission grants flash — waits for the sorcery window.
// Every cast-offering path shares this one rule: hand, split / modal DFC back face, flashback,
// escape, the graveyard and exile cast permissions, and the suspend special action.
static bool spell_timing_ok(const CardData &face, Zone::Ownership caster, bool sorcery_window) {
    if (sorcery_window) return true;
    if (rules_mod::opponent_sorcery_speed_locked(caster)) return false;
    return card_has_type(face, "Instant") || card_has_keyword(face, "Flash") ||
           rules_mod::cast_with_flash_active(caster, face);
}

// The cast-legality gates every cast-offering path shares (CR 601.2c, 601.3): the spell's timing
// (spell_timing_ok, unless `ignore_timing`), a legal target for every required target of some
// reachable mode, an Aura's enchant target, no CantBeCast prohibition on a spell cast from
// `from_zone`, and the payment-failure guard. `face` is the face being cast (a split / modal DFC
// back face, or the card itself). Costs differ per path (normal, alternative, flashback, escape,
// resource grants), so each caller checks its own.
static bool can_cast_now(const CardData &face, Entity card_entity, Zone::Ownership caster,
                         Zone::ZoneValue from_zone, bool sorcery_window, bool ignore_timing,
                         std::shared_ptr<Orderer> orderer) {
    if (!ignore_timing && !spell_timing_ok(face, caster, sorcery_window)) return false;
    // A mandatory additional cost applies however the spell is cast (CR 118.9d, 601.2f): it must be
    // payable, one of its forms if it has several, with the mana held to the cast's own gates.
    {
        CostCheck check(caster, card_entity, orderer);
        check.unchecked = CostParts::kind_bit(CostPart::MANA);
        if (!can_pay(spell_additional_cost(face), check)) return false;
        if (!face.additional_choices.empty() && payable_choices(face.additional_choices, check).empty()) return false;
    }
    // Target legality (CR 601.2c), mode-aware: for a Gift spell the required target type switches
    // on the gift promise (Into the Flood Maw: a creature without the gift, a nonland permanent
    // with it) and a modal spell needs enough choosable modes, so the spell is castable iff a legal
    // target exists for at least one reachable mode. Probed with the real cast source/controller
    // (card_entity) so source-dependent target restrictions — protection from this spell's color,
    // OppCtrl — match select_target and a protected-only target (Emrakul vs white, Scryb Ranger vs
    // blue) is not offered. A ConditionPresent$ "if ..." clause is checked only at resolution
    // (CR 608.2c), so it never gates the cast.
    for (const auto &ab : face.abilities) {
        if (ab->ability_type != AbilityDef::SPELL) continue;
        if (!spell_has_castable_targets(cast_gate_probe(Ability(ab), card_entity, caster), orderer, caster,
                                        face.has_gift))
            return false;
        break;
    }
    // Aura enchant-target gate (CR 303.4 / 601.2c) — see aura_enchant_target_available.
    if (!aura_enchant_target_available(face, caster, orderer)) return false;
    if (rules_mod::cast_prohibited(caster, face, from_zone)) return false;
    return !payment_blocked(card_entity);
}

// A CAST_SPELL action for `card_entity`; `option_ordinal` tells the cast variants apart
// (0 normal, 1 alternate / impending cost, 2 offspring, 3 modal-DFC back face, 4 flashback,
// 5 escape, 6 cast-from-graveyard permission, 7 impulse / free cast from exile, 8 without paying
// its mana cost under a free-cast static).
static LegalAction cast_action(Entity card_entity, const std::string &desc, int option_ordinal) {
    LegalAction la(CAST_SPELL, card_entity, desc);
    la.category = ActionCategory::CAST_SPELL;
    la.option_ordinal = option_ordinal;
    return la;
}

// The CastCost of `alt_cost`, the alternative cost of a card.
static CastCost alternative_cost_kind(const AltCost &alt_cost) {
    if (alt_cost.is_evoke) return CastCost::EVOKE;
    if (alt_cost.is_impending) return CastCost::IMPENDING;
    if (alt_cost.is_warp) return CastCost::WARP;
    return CastCost::ALTERNATIVE;
}

// The regular cost of casting `face` from `card_entity` for `caster` that the cast-legality gates
// check: effective_base_cost, less the most a ValidTarget$-conditioned reduction takes off for one
// of the spell's legal targets (best_target_cost_reduction; Mystical Dispute targeting a blue
// spell). The cast then offers only the targets whose cost can be paid.
static ManaValue regular_cast_cost(
    const CardData &face, Entity card_entity, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    ManaValue cost = effective_base_cost(face, caster);
    reduce_generic_cost(cost, best_target_cost_reduction(face, card_entity, caster, orderer), face);
    return cost;
}

// A check of a cost of casting `face`: its Delve and TapToPay act on the total cost, whichever cost
// that was made from (CR 702.66a, 702.126a).
static CostCheck total_cost_check(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                  const std::shared_ptr<Orderer> &orderer) {
    CostCheck check(caster, card_entity, orderer);
    check.delve = face.has_delve;
    check.tap_to_pay = face.tap_to_pay;
    return check;
}

// Can `caster` cast `face` for its mana cost, `mana` after reductions, with the hybrid and Phyrexian
// pips folded in (castable iff SOME assignment is payable), and pay its additional costs?
static bool regular_cast_payable(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                 const ManaValue &mana, std::shared_ptr<Orderer> orderer) {
    CostCheck check(caster, card_entity, orderer);
    check.mana = &mana;
    check.spell = &face;
    check.hand_reserved = fewest_hand_cards(face.additional_choices, check);
    return can_pay(spell_additional_cost(face), check);
}

// Free-cast statics (Omniscience, Aluren; rules_mod::free_cast_grant): beside `cast`, the card's
// cast from `from_zone` for its own costs, offer the cast without paying its mana cost — the
// alternative cost of CR 118.9, so never combined with another one (118.9a). It keeps the
// spell's own timing, or any time its caster holds priority when the grant gives flash (CR
// 601.3c; an opponent's sorcery-speed lock still applies), and the other cast gates. Its mana is
// what cost increases and the SetCost floor add to {0} (CR 118.9d); the spell's own additional
// sacrifice must be payable, and with Offspring a second row pays that additional cost too.
static void offer_free_cast(std::vector<LegalAction> &actions, const LegalAction &cast, const CardData &face,
    Zone::ZoneValue from_zone, Zone::Ownership caster, bool sorcery_window, std::shared_ptr<Orderer> orderer) {
    const Entity card = cast.source_entity;
    const rules_mod::FreeCastGrant grant = rules_mod::free_cast_grant(card, face, caster);
    if (!grant.granted) return;
    const bool timing_ok = grant.with_flash ? sorcery_window || !rules_mod::opponent_sorcery_speed_locked(caster)
                                            : spell_timing_ok(face, caster, sorcery_window);
    if (!timing_ok || !can_cast_now(face, card, caster, from_zone, sorcery_window, /*ignore_timing=*/true, orderer))
        return;
    const ManaValue mana = floored_alt_mana_cost(face, ManaValue{}, caster);
    CostCheck check = total_cost_check(face, card, caster, orderer);
    check.mana = &mana;
    const CostParts &additional = spell_additional_cost(face);
    if (!can_pay(additional, check)) return;
    LegalAction free_la = cast;
    free_la.option.cost = CastCost::NO_MANA_COST;
    free_la.option_ordinal = 8;
    free_la.description = cast.description + " without paying its mana cost";
    actions.push_back(free_la);
    if (!face.has_offspring) return;
    ManaValue with_offspring = mana;
    for (Colors c : face.offspring_cost.mana()) with_offspring.insert(c);
    check.mana = &with_offspring;
    if (!can_pay(additional, check)) return;
    free_la.option.offspring = true;
    free_la.description += " (offspring)";
    actions.push_back(free_la);
}

// See declaration in cast_options.h.
bool exile_grant_castable(Entity card, Zone::Ownership caster, bool sorcery_window,
                          std::shared_ptr<Orderer> orderer) {
    const Game::ImpulseCastPermission *grant = cur_game.resolved_effects.impulse_cast_permission.find(card);
    if (grant == nullptr) return false;
    const Game::ImpulseCastPermission &perm_grant = *grant;
    const CardData &ecd = global_coordinator.GetComponent<CardData>(card);

    // A cast made during a resolution (CR 608.2g) ignores the card's type-based timing.
    // Aura enchant-target gate: the concrete crash it fixes here is Animate Dead reanimating
    // the opponent's Amped Raptor (emptying the graveyard), the Raptor's impulse exiling a
    // SECOND Animate Dead and granting this energy-cast permission — which must not be
    // offered while no creature card is in any graveyard.
    if (!can_cast_now(ecd, card, caster, Zone::EXILE, sorcery_window,
                      /*ignore_timing=*/perm_grant.during_resolution, orderer))
        return false;

    // Affordability of the alternative resource cost.
    Entity pe = get_player_entity(caster);
    if (!global_coordinator.entity_has_component<Player>(pe)) return false;
    auto &ppl = global_coordinator.GetComponent<Player>(pe);
    bool is_normal_play = (perm_grant.resource == Game::ImpulseCastPermission::NORMAL);
    if (perm_grant.resource == Game::ImpulseCastPermission::FREE) {
        // No cost to pay (Ugin -11 grant) — always affordable.
    } else if (is_normal_play) {
        // Play a nonland card for its NORMAL mana cost (Light Up the Stage): affordable iff
        // the full (cost-increase-adjusted, hybrid- and Phyrexian-resolved) base cost can be
        // paid.
        if (!regular_cast_payable(ecd, card, caster, regular_cast_cost(ecd, card, caster, orderer), orderer))
            return false;
    } else if (perm_grant.resource == Game::ImpulseCastPermission::ENERGY) {
        if (player_energy(ppl) < perm_grant.amount) return false;
    } else {  // LIFE — must be able to pay without the cost itself being lethal is not a
              // legality bar in MTG, but a player won't be forced; require enough life so
              // the optional cast is sensibly offered.
        if (!can_pay_life(ppl, perm_grant.amount)) return false;
    }

    // Cost-increase / SetCost-floor statics apply to alternative costs too (CR 118.9d /
    // 601.2f): an impulse/free cast substitutes a {0} mana cost, but an active Trinisphere
    // floor pads that up to its minimum ({3}) and Thalia adds its surcharge — payable ON TOP
    // of the energy/life resource cost. Require the floored mana; empty (no floor/increase)
    // means no extra mana and this gate is a no-op. NORMAL plays already pay the full base
    // cost above, so this alt-cost floor doesn't apply to them.
    // A LIFE grant's life is paid before this mana, so the mana may not spend it either.
    if (!is_normal_play) {
        int grant_life = perm_grant.resource == Game::ImpulseCastPermission::LIFE
                             ? perm_grant.amount : 0;
        ManaValue floor_mana = floored_alt_mana_cost(ecd, ManaValue{}, caster);
        if (!floor_mana.empty() &&
            !can_pay_mana(caster, floor_mana, card, orderer, ecd.has_delve,
                          ecd.tap_to_pay, /*exclude_entity=*/0,
                          /*life_reserve=*/grant_life))
            return false;
    }
    return true;
}

// See declaration in cast_options.h.
bool miracle_castable(Entity card, Zone::Ownership owner, std::shared_ptr<Orderer> orderer) {
    if (!global_coordinator.entity_has_component<CardData>(card) ||
        !global_coordinator.entity_has_component<Zone>(card))
        return false;
    const auto &z = global_coordinator.GetComponent<Zone>(card);
    if (z.location != Zone::HAND || z.owner != owner) return false;
    const CardData &cd = global_coordinator.GetComponent<CardData>(card);
    if (!can_cast_now(cd, card, owner, Zone::HAND, /*sorcery_window=*/false,
                      /*ignore_timing=*/true, orderer))
        return false;
    const ManaValue alt_mana = floored_alt_mana_cost(cd, cd.alt_cost.cost.mana(), owner);
    CostCheck check = total_cost_check(cd, card, owner, orderer);
    check.mana = &alt_mana;
    check.unchecked = kAllButMana;
    return can_pay(cd.alt_cost.cost, check);
}

// An Aura (CR 303.4 / 601.2c) targets the object it will enchant as it is cast, so EVERY
// cast-offering path — hand, modal back face, flashback, escape, a cast-from-graveyard
// permission (Emry), an impulse/free cast from exile (Amped Raptor, suspend) — may only offer
// it while a legal object matching its Enchant restriction exists. Builds the same transient
// targeting ability the cast flow's AURA_TARGET step uses, so the offer gate and the cast-time
// target menu can never disagree. The enchant filter is controller-relative (Sheltered by
// Ghosts: Enchant Creature.YouCtrl), so the transient ability MUST carry the actual caster as
// controller — has_legal_targets evaluates YouCtrl/OppCtrl from ability_perspective_player,
// which defaults to PLAYER_A when left unset. Returns true for non-Auras.
// (This gate was originally inlined only in the hand-cast loop; an Amped Raptor impulse grant
// exiling Animate Dead off an empty-graveyard board then offered the cast ungated and crashed
// AURA_TARGET's empty target menu — CR 601.2c requires the gate on every offering path.)
static bool aura_enchant_target_available(const CardData &card_data, Zone::Ownership caster,
                                          std::shared_ptr<Orderer> orderer) {
    if (card_data.enchant_filter.empty()) return true;  // not an Aura
    return has_legal_targets(enchant_target_ability(0, card_data, caster), orderer);
}

static bool can_afford_alt(const CardData& card_data, const AltCost& alt_cost,
                           Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Miracle (CR 702.94) is never a normal priority-menu cast: it is castable ONLY through the
    // dedicated immediate "cast it for its miracle cost / do not cast" decision presented right when
    // its linked triggered ability resolves (proc_mandatory_choice's miracle-cast branch, gated on
    // its own affordability check). So the alternate-cost cast option is never offered here.
    if (alt_cost.is_miracle) return false;

    // Spectacle (CR 702.107a): the spectacle cost may be paid only if an opponent of the
    // caster lost life this turn. Two-player game — the sole opponent is the other seat.
    if (alt_cost.is_spectacle) {
        Zone::Ownership opp = opponent_of(priority_player);
        if (global_coordinator.GetComponent<Player>(get_player_entity(opp)).life_lost_this_turn <= 0)
            return false;
    }

    // Check SVar condition (e.g. Once Upon a Time: free only if first spell this game)
    if (!alt_cost.condition_svar.empty()) {
        // The condition's SVar (Once Upon a Time: Count$YouCastThisGame EQ0; Mindbreak Trap:
        // PlayerCountOpponents$ConditionGE3 SpellsCastThisTurn — an opponent cast three or more
        // spells this turn), compared as scripted, else at least 1 (Forge's default comparator).
        const std::string compare =
            alt_cost.condition_compare.empty() ? std::string("GE1") : alt_cost.condition_compare;
        if (!compare_svar(evaluate_svar(alt_cost.condition_svar, priority_player), compare))
            return false;
    }

    // IsPresent$ <type>[.YouCtrl] — the alt cost is only available while the
    // caster controls a matching permanent (e.g. Snuff Out: control a Swamp).
    if (!alt_cost.condition_is_present.empty()) {
        std::string filter = alt_cost.condition_is_present;
        size_t dot = filter.find('.');
        std::string type_name = (dot == std::string::npos) ? filter : filter.substr(0, dot);
        bool found = false;
        for (auto e : orderer->mEntities) {
            if (!is_battlefield_permanent(e, priority_player)) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            for (auto &t : perm.types) {
                if (t.name == type_name) { found = true; break; }
            }
            if (found) break;
        }
        if (!found) return false;
    }

    // Condition: not your turn (Force of Negation, Force of Vigor).
    if (alt_cost.condition_not_your_turn) {
        bool is_my_turn = (priority_player == Zone::PLAYER_A) ? cur_game.turn_state.player_a_turn
                                                              : !cur_game.turn_state.player_a_turn;
        if (is_my_turn) return false;
    }

    // The mana portion of the alt cost (e.g. Evoke:R) with the active SetCost floor
    // (Trinisphere) folded in — CR 601.2f applies the floor AFTER the alternative cost is
    // substituted, so even a Cost$ 0 / pitch cast must be able to pay up to the floor.
    const ManaValue alt_mana = floored_alt_mana_cost(card_data, alt_cost.cost.mana(), priority_player,
                                                     static_cast<int>(alt_cost.cost.hybrid().size()));
    CostCheck check = total_cost_check(card_data, card_entity, priority_player, orderer);
    check.mana = &alt_mana;
    return can_pay(alt_cost.cost, check);
}

// Modal DFC whose BACK face is a NONLAND spell (Tergrid, God of Fright // Tergrid's Lantern):
// offer a CAST_SPELL that casts the BACK face — paying the back's mana cost and using the back
// face's characteristics/abilities (CR 712.8). The front face is offered as a normal cast by the
// main hand loop, and the LAND-back case is offered there as a PLAY_LAND; only the nonland-back
// CAST is added here. Kept in its own hand pass so a prohibition/`continue` on the front face
// doesn't suppress the back-face option (the two faces are cast independently).
static void offer_modal_back_face_casts(std::vector<LegalAction> &actions, Zone::Ownership priority_player,
                                        const std::vector<Entity> &hand, std::shared_ptr<Orderer> orderer,
                                        bool sorcery_window) {
    for (auto card_entity : hand) {
        auto &front = global_coordinator.GetComponent<CardData>(card_entity);
        // Offer the back half for both modal DFCs (nonland back) and split cards (CR 709): both
        // store the second face in `backside` and cast it via the shared cast_back_face path.
        if ((!front.is_modal_dfc && !front.is_split) || !front.backside) continue;
        const CardData &back = *front.backside;
        if (is_land_card(back)) continue;  // land back is a PLAY_LAND, handled in the main loop

        LegalAction la = cast_action(card_entity, "Cast " + back.name, 3);
        la.option.face = CastFace::BACK;
        // Timing, targets and prohibitions are the back face's own (Gone is an instant).
        if (can_cast_now(back, card_entity, priority_player, Zone::HAND, sorcery_window,
                /*ignore_timing=*/false, orderer) &&
            regular_cast_payable(back, card_entity, priority_player,
                                 regular_cast_cost(back, card_entity, priority_player, orderer), orderer))
            actions.push_back(la);
        offer_free_cast(actions, la, back, Zone::HAND, priority_player, sorcery_window, orderer);
    }
}

// See declaration in cast_options.h.
void offer_cast_options(std::vector<LegalAction> &actions, const CastWindow &window, const std::vector<Entity> &hand,
                        const std::vector<std::pair<Entity, unsigned>> &zone_play_routes) {
    const Zone::Ownership priority_player = window.caster;
    const bool sorcery_window = window.sorcery_window;
    const std::shared_ptr<Orderer> &orderer = window.orderer;
    // checking for spells to cast from hand
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        if (card_has_type(card_data, "Land")) continue;  // can't cast land
        // Timing alone also gates the suspend special action below.
        bool timing_ok = spell_timing_ok(card_data, priority_player, sorcery_window);
        if (timing_ok && can_cast_now(card_data, card_entity, priority_player, Zone::HAND,
                                      sorcery_window, /*ignore_timing=*/true, orderer)) {
            LegalAction la = cast_action(card_entity, "Cast " + card_data.name, 0);

            ManaValue effective_cost = regular_cast_cost(card_data, card_entity, priority_player, orderer);

            // X-cost spells: base cost (without X) is enough to be castable;
            // X value is chosen at cast time in action_processor. Additional Sacrifice-a-<type>
            // cost on the spell itself (Natural Order: "As an additional cost to cast this spell,
            // sacrifice a green creature"): the spell can't be cast unless a matching permanent is
            // available to sacrifice (CR 601.2f). General: any spell whose SPELL ability Cost$
            // carries a Sac<...> token, matched (incl. color qualifier) like an activation cost.
            bool can_regular = regular_cast_payable(card_data, card_entity, priority_player, effective_cost, orderer);

            bool can_alt = can_afford_alt(card_data, card_data.alt_cost, priority_player, card_entity, orderer);

            if (can_regular) actions.push_back(la);
            if (can_alt) {
                LegalAction alt_la = la;
                alt_la.option.cost = alternative_cost_kind(card_data.alt_cost);
                alt_la.option_ordinal = 1;  // cast variant: 1 = alternate/impending cost
                alt_la.description = "Cast " + card_data.name +
                    (card_data.alt_cost.is_impending ? " (impending)"
                     : card_data.alt_cost.is_spectacle ? " (spectacle)"
                     : card_data.alt_cost.is_miracle   ? " (miracle)"
                     : card_data.alt_cost.is_warp      ? " (warp)"
                                                        : " (alternate cost)");
                actions.push_back(alt_la);
            }
            // Offspring (CR 702.171): optional ADDITIONAL cost. Offer a separate cast option
            // that must pay the base cost plus the offspring cost together.
            if (card_data.has_offspring) {
                ManaValue offspring_total = effective_cost;
                for (Colors c : card_data.offspring_cost.mana()) offspring_total.insert(c);
                if (regular_cast_payable(card_data, card_entity, priority_player, offspring_total, orderer)) {
                    LegalAction off_la = la;
                    off_la.option.offspring = true;
                    off_la.option_ordinal = 2;  // cast variant: 2 = offspring
                    off_la.description = "Cast " + card_data.name + " (offspring)";
                    actions.push_back(off_la);
                }
            }
        }
        offer_free_cast(actions, cast_action(card_entity, "Cast " + card_data.name, 0), card_data, Zone::HAND,
            priority_player, sorcery_window, orderer);
        // SUSPEND (CR 702.62a, first ability): from the hand, at the timing you could begin to cast
        // the card (`timing_ok` — sorcery speed for a sorcery), its owner may instead pay the
        // suspend cost and exile it with N time counters. This is a special action (doesn't use the
        // stack). Its targets are chosen only later, when the last counter is removed and it is cast
        // for free, so no legal target is required now (702.62 casts it then, not here); the card
        // just must not be under a cast prohibition (702.62c) and the suspend mana cost must be
        // affordable. General over any Suspend card. Offered independently of the normal-cast block.
        CostCheck suspend_check(priority_player, card_entity, orderer);
        suspend_check.payee = ManaPayee(card_entity, ManaPurpose::OTHER);
        if (card_data.has_suspend && timing_ok &&
            !rules_mod::cast_prohibited(priority_player, card_data) &&
            can_pay(card_data.suspend_cost, suspend_check) &&
            !payment_blocked(card_entity)) {
            LegalAction sus_la(SPECIAL_ACTION, card_entity, "Suspend " + card_data.name);
            sus_la.category = ActionCategory::CAST_SPELL;  // cast-adjacent "play this card" action
            sus_la.suspend_action = true;
            actions.push_back(sus_la);
        }
    }
    // Modal DFC nonland back faces: offer the BACK face as a CAST_SPELL (the front face's normal
    // cast and the land-back PLAY_LAND were handled in the hand loop above).
    offer_modal_back_face_casts(actions, priority_player, hand, orderer, sorcery_window);
    // checking graveyard for flashback spells (the FLASHBACK route)
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & permission_bit(Permission::FLASHBACK))) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        // A graveyard-cast static (Grafdigger's Cage: Origin$ Graveyard) prohibits flashback.
        if (!can_cast_now(gcd, gy_entity, priority_player, Zone::GRAVEYARD, sorcery_window,
                          /*ignore_timing=*/false, orderer))
            continue;

        // Check affordability: the cost with its mana floored — flashback is an alternative cost
        // (CR 702.34a), so an active SetCost floor applies to it too. Cabal Therapy's Flashback—
        // Sacrifice a creature can't be cast unless a matching permanent is available to sacrifice
        // (CR 601.2f / 601.3a).
        const ManaValue fb_mana = floored_alt_mana_cost(gcd, gcd.flashback_cost.mana(), priority_player);
        CostCheck fb_check = total_cost_check(gcd, gy_entity, priority_player, orderer);
        fb_check.mana = &fb_mana;
        if (!can_pay(gcd.flashback_cost, fb_check)) continue;

        LegalAction fb_la = cast_action(gy_entity, "Cast " + gcd.name + " (flashback)", 4);
        fb_la.option.zone = Zone::GRAVEYARD;
        fb_la.option.cost = CastCost::FLASHBACK;
        fb_la.option.permission = Permission::FLASHBACK;
        fb_la.option.destination = Zone::EXILE;
        actions.push_back(fb_la);
    }
    // ESCAPE (CR 702.139): a card in its owner's graveyard may be cast from there for its
    // escape cost (mana + additional cost). Same sorcery-speed timing flashback uses unless the
    // card is an instant. Nethergoyf's additional cost is ExileFromGrave with a "≥N card types
    // among the chosen cards" constraint, so the cast is only legal when OTHER cards in the
    // caster's graveyard collectively cover those N types (otherwise the cost is unpayable).
    // The permission itself is the ESCAPE route.
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & permission_bit(Permission::ESCAPE))) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        if (!can_cast_now(gcd, gy_entity, priority_player, Zone::GRAVEYARD, sorcery_window,
                          /*ignore_timing=*/false, orderer))
            continue;

        // Escape is an alternative cost (CR 702.139a): fold in any active SetCost floor.
        const ManaValue esc_mana = floored_alt_mana_cost(gcd, gcd.escape_cost.mana(), priority_player);
        CostCheck esc_check = total_cost_check(gcd, gy_entity, priority_player, orderer);
        esc_check.mana = &esc_mana;
        if (!can_pay(gcd.escape_cost, esc_check)) continue;

        LegalAction esc_la = cast_action(gy_entity, "Cast " + gcd.name + " (escape)", 5);
        esc_la.option.zone = Zone::GRAVEYARD;
        esc_la.option.cost = CastCost::ESCAPE;
        esc_la.option.permission = Permission::ESCAPE;
        actions.push_back(esc_la);
    }
    // CAST-FROM-GRAVEYARD PERMISSIONS (Emry's AB$ Effect): a card the priority player has
    // been granted permission to cast this turn (CR 601.3e). It is cast from the graveyard
    // for its normal cost, at the timing its type allows. The GRAVEYARD_CAST route covers
    // only the graveyard owner and never a land (601.1).
    for (const auto &[gy_entity, routes] : zone_play_routes) {
        if (!(routes & permission_bit(Permission::GRAVEYARD_CAST))) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        LegalAction gy_la = cast_action(gy_entity, "Cast " + gcd.name + " (from graveyard)", 6);
        gy_la.option.zone = Zone::GRAVEYARD;
        gy_la.option.permission = Permission::GRAVEYARD_CAST;
        if (can_cast_now(gcd, gy_entity, priority_player, Zone::GRAVEYARD, sorcery_window,
                /*ignore_timing=*/false, orderer) &&
            regular_cast_payable(gcd, gy_entity, priority_player,
                                 regular_cast_cost(gcd, gy_entity, priority_player, orderer), orderer))
            actions.push_back(gy_la);
        offer_free_cast(actions, gy_la, gcd, Zone::GRAVEYARD, priority_player, sorcery_window, orderer);
    }
    // PLAY-FROM-EXILE PERMISSIONS (Light Up the Stage, Ugin -11, Dauthi Voidwalker, warp): an
    // exiled card its grant lets this player play, for its normal cost, without paying its mana
    // cost, or for an alternative resource cost (CR 118.9). Cast from EXILE at the timing its type
    // allows; only the granted player may cast it, and only if they can pay the cost. The
    // EXILE_GRANT route covers a card still in exile whose grant names this player (and a land
    // only under a "play" grant that allows lands).
    for (const auto &[ex_entity, routes] : zone_play_routes) {
        if (!(routes & permission_bit(Permission::EXILE_GRANT))) continue;
        const auto &perm_grant = *cur_game.resolved_effects.impulse_cast_permission.find(ex_entity);
        auto &ecd = global_coordinator.GetComponent<CardData>(ex_entity);

        // A LAND among the exiled cards: only a "play" permission (Light Up the Stage's "you may
        // PLAY those cards", Dauthi Voidwalker's "you may play it") may play it — a land play,
        // sorcery-timing, own main phase, empty stack, and a land drop remaining (CR 305.1 /
        // 305.2). A "cast" grant (Ugin -11) can't play a land (601.1), and play_permissions
        // reports no route for a land under one.
        if (is_land_card(ecd)) {
            if (!sorcery_window) continue;
            // Same shared land-drop expression the hand loop above uses.
            if (rules_mod::land_drops_remaining(priority_player) <= 0) continue;
            LegalAction land_la(SPECIAL_ACTION, ex_entity, "Play " + ecd.name + " (from exile)");
            land_la.category = ActionCategory::PLAY_LAND;
            actions.push_back(land_la);
            continue;
        }

        // A grant to cast the card for its normal cost leaves its cost open to a free-cast
        // static's alternative cost; any other grant is an alternative cost itself (CR 118.9a).
        const bool normal_grant =
            perm_grant.resource == Game::ImpulseCastPermission::NORMAL && !perm_grant.during_resolution;
        const bool castable = exile_grant_castable(ex_entity, priority_player, sorcery_window, orderer);
        if (!castable && !normal_grant) continue;

        const char *imp_suffix = (perm_grant.resource == Game::ImpulseCastPermission::FREE)
                                     ? " (from exile, no cost)"
                                     : " (from exile)";
        LegalAction imp_la = cast_action(ex_entity, "Cast " + ecd.name + imp_suffix, 7);
        imp_la.option.zone = Zone::EXILE;
        imp_la.option.permission = Permission::EXILE_GRANT;
        if (perm_grant.resource != Game::ImpulseCastPermission::NORMAL) imp_la.option.cost = CastCost::GRANT;
        if (castable) actions.push_back(imp_la);
        if (normal_grant) offer_free_cast(actions, imp_la, ecd, Zone::EXILE, priority_player, sorcery_window, orderer);
    }
}
