#include "cast_options.h"

#include <cstddef>
#include <string>

#include "action_processor.h"
#include "classes/game.h"
#include "components/carddata.h"
#include "components/permanent.h"
#include "components/player.h"
#include "coverage.h"
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

// One card's turn in a section: what its rows ask about, each gate they share worked out once.
struct CastOffer {
        Entity card;
        const CardData &face;  // the face cast
        Zone::ZoneValue zone;
        const CastWindow &window;
        int timing = -1;  // spell_timing_ok, -1 until asked
        int gates = -1;   // can_cast_now without the timing, -1 until asked
};

// A way of casting: its coverage name, what it pays, what labels it (LegalAction::option_ordinal
// and what follows "Cast <face>" in the menu), where it sends the spell, whether it is offered, and
// what it changes on the spell.
struct CastRow {
        const char *name;
        CastCost cost;
        bool offspring;
        int ordinal;
        const char *label;
        Zone::ZoneValue destination;
        bool (*offered)(CastOffer &offer);
        SpellOverlay overlay = SpellOverlay::NONE;
};

// The rows of one zone, permission and face, offered to each card that has the permission. `covers`
// (none: every card) says whether the card is cast this way at all, and `beside` adds what stands
// among a section's rows without being a cast.
struct CastSection {
        Zone::ZoneValue zone;
        Permission permission;
        CastFace face;
        bool (*covers)(const CardData &card);
        void (*beside)(CastOffer &offer, std::vector<LegalAction> &actions);
        std::vector<CastRow> rows;
};

static bool aura_enchant_target_available(const CardData &card_data, Zone::Ownership caster,
                                          std::shared_ptr<Orderer> orderer);
static bool spell_timing_ok(const CardData &face, Zone::Ownership caster, bool sorcery_window);
static bool can_cast_now(const CardData &face, Entity card_entity, Zone::Ownership caster,
                         Zone::ZoneValue from_zone, bool sorcery_window, bool ignore_timing,
                         std::shared_ptr<Orderer> orderer);
static ManaValue regular_cast_cost(
    const CardData &face, Entity card_entity, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static CostCheck total_cost_check(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                  const std::shared_ptr<Orderer> &orderer);
static bool regular_cast_payable(const CardData &face, Entity card_entity, Zone::Ownership caster,
                                 const ManaValue &mana, std::shared_ptr<Orderer> orderer);
static bool alt_svar_condition_holds(const std::string &svar, const std::string &compare, Zone::Ownership player);
static bool can_afford_alt(const CardData &card_data, const AltCost &alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer);
static bool offer_timing_ok(CastOffer &offer);
static bool offer_gates_ok(CastOffer &offer);
static bool offer_castable(CastOffer &offer);
static bool mana_cost_offered(CastOffer &offer);
static bool alternative_offered(CastOffer &offer);
static bool evoke_offered(CastOffer &offer);
static bool impending_offered(CastOffer &offer);
static bool spectacle_offered(CastOffer &offer);
static bool warp_offered(CastOffer &offer);
static bool plain_alternative_offered(CastOffer &offer);
static bool offspring_offered(CastOffer &offer);
static bool free_cast_offered(CastOffer &offer);
static bool free_offspring_offered(CastOffer &offer);
static bool replacing_cost_payable(const CastOffer &offer, const CostParts &cost);
static bool flashback_offered(CastOffer &offer);
static bool escape_offered(CastOffer &offer);
static bool grant_cost_payable(Entity card, const CardData &ecd, const Game::ImpulseCastPermission &grant,
                               Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static const Game::ImpulseCastPermission &offer_grant(const CastOffer &offer);
static bool grant_offered(CastOffer &offer);
static bool exile_mana_cost_offered(CastOffer &offer);
static bool exile_resource_offered(CastOffer &offer);
static bool exile_no_cost_offered(CastOffer &offer);
static bool exile_free_offered(CastOffer &offer);
static bool exile_free_offspring_offered(CastOffer &offer);
static bool cast_from_exile(const CardData &card);
static void offer_exile_land(CastOffer &offer, std::vector<LegalAction> &actions);
static bool cast_from_front(const CardData &card);
static bool cast_from_back(const CardData &front);
static void offer_suspend(CastOffer &offer, std::vector<LegalAction> &actions);
static LegalAction row_action(const CastRow &row, const CastSection &section, Entity card, const std::string &desc);
static void enumerate_cast_options(Entity card, const CastSection &section, const CastWindow &window,
                                   std::vector<LegalAction> &actions);

// The cast rows of a priority menu, in menu order: a section at a time, card by card, each card's
// rows in turn. A new way of casting is a row, with what offers it.
static const CastSection kCastSections[] = {
    {Zone::HAND, Permission::NONE, CastFace::FRONT, cast_from_front, offer_suspend, {
        // name             cost                   offspring ordinal label                     destination      offered
        {"hand",            CastCost::MANA_COST,    false, 0, "",                      Zone::GRAVEYARD, mana_cost_offered},
        {"evoke",           CastCost::EVOKE,        false, 1, " (alternate cost)",     Zone::GRAVEYARD, evoke_offered},
        {"impending",       CastCost::IMPENDING,    false, 1, " (impending)",          Zone::GRAVEYARD, impending_offered},
        {"spectacle",       CastCost::ALTERNATIVE,  false, 1, " (spectacle)",          Zone::GRAVEYARD, spectacle_offered},
        {"warp",            CastCost::WARP,         false, 1, " (warp)",               Zone::GRAVEYARD, warp_offered},
        {"alternative",     CastCost::ALTERNATIVE,  false, 1, " (alternate cost)",     Zone::GRAVEYARD, plain_alternative_offered},
        {"offspring",       CastCost::MANA_COST,    true,  2, " (offspring)",          Zone::GRAVEYARD, offspring_offered},
        {"hand_free",       CastCost::NO_MANA_COST, false, 8, " without paying its mana cost", Zone::GRAVEYARD, free_cast_offered},
        {"hand_free_offspring", CastCost::NO_MANA_COST, true, 8, " without paying its mana cost (offspring)", Zone::GRAVEYARD, free_offspring_offered},
    }},
    {Zone::HAND, Permission::NONE, CastFace::BACK, cast_from_back, nullptr, {
        {"back_face",       CastCost::MANA_COST,    false, 3, "",                      Zone::GRAVEYARD, mana_cost_offered},
        {"back_face_free",  CastCost::NO_MANA_COST, false, 8, " without paying its mana cost", Zone::GRAVEYARD, free_cast_offered},
        {"back_face_free_offspring", CastCost::NO_MANA_COST, true, 8, " without paying its mana cost (offspring)", Zone::GRAVEYARD, free_offspring_offered},
    }},
    {Zone::GRAVEYARD, Permission::FLASHBACK, CastFace::FRONT, nullptr, nullptr, {
        {"flashback",       CastCost::FLASHBACK,    false, 4, " (flashback)",          Zone::EXILE,     flashback_offered},
    }},
    {Zone::GRAVEYARD, Permission::ESCAPE, CastFace::FRONT, nullptr, nullptr, {
        {"escape",          CastCost::ESCAPE,       false, 5, " (escape)",             Zone::GRAVEYARD, escape_offered},
    }},
    // CAST-FROM-GRAVEYARD PERMISSIONS (Emry's AB$ Effect): a card the priority player has
    // been granted permission to cast this turn (CR 601.3e). It is cast from the graveyard
    // for its normal cost, at the timing its type allows. The GRAVEYARD_CAST route covers
    // only the graveyard owner and never a land (601.1).
    {Zone::GRAVEYARD, Permission::GRAVEYARD_CAST, CastFace::FRONT, nullptr, nullptr, {
        {"graveyard_cast",  CastCost::MANA_COST,    false, 6, " (from graveyard)",     Zone::GRAVEYARD, mana_cost_offered},
        {"graveyard_free",  CastCost::NO_MANA_COST, false, 8, " (from graveyard) without paying its mana cost", Zone::GRAVEYARD, free_cast_offered},
        {"graveyard_free_offspring", CastCost::NO_MANA_COST, true, 8, " (from graveyard) without paying its mana cost (offspring)", Zone::GRAVEYARD, free_offspring_offered},
    }},
    // PLAY-FROM-EXILE PERMISSIONS (Light Up the Stage, Ugin -11, Dauthi Voidwalker, warp): an
    // exiled card its grant lets this player play, for its normal cost, without paying its mana
    // cost, or for an alternative resource cost (CR 118.9). Cast from EXILE at the timing its type
    // allows; only the granted player may cast it, and only if they can pay the cost. The
    // EXILE_GRANT route covers a card still in exile whose grant names this player (and a land
    // only under a "play" grant that allows lands).
    {Zone::EXILE, Permission::EXILE_GRANT, CastFace::FRONT, cast_from_exile, offer_exile_land, {
        {"exile_mana_cost", CastCost::MANA_COST,    false, 7, " (from exile)",         Zone::GRAVEYARD, exile_mana_cost_offered},
        {"exile_resource",  CastCost::GRANT,        false, 7, " (from exile)",         Zone::GRAVEYARD, exile_resource_offered},
        {"exile_no_cost",   CastCost::GRANT,        false, 7, " (from exile, no cost)", Zone::GRAVEYARD, exile_no_cost_offered},
        {"exile_free",      CastCost::NO_MANA_COST, false, 8, " (from exile) without paying its mana cost", Zone::GRAVEYARD, exile_free_offered},
        {"exile_free_offspring", CastCost::NO_MANA_COST, true, 8, " (from exile) without paying its mana cost (offspring)", Zone::GRAVEYARD, exile_free_offspring_offered},
    }},
};

// The casts a resolution offers (CR 608.2g), not priority: miracle_castable and exile_grant_castable
// say whether they are offered, so they aren't walked.
static const CastSection kMiracleCast = {Zone::HAND, Permission::NONE, CastFace::FRONT, nullptr, nullptr, {
    {"miracle",             CastCost::ALTERNATIVE,  false, 1, " (miracle)",            Zone::GRAVEYARD, nullptr},
}};
static const CastSection kResolutionGrantCast = {Zone::EXILE, Permission::EXILE_GRANT, CastFace::FRONT, nullptr, nullptr, {
    {"exile_resolution",    CastCost::GRANT,        false, -1, "",                     Zone::GRAVEYARD, nullptr},
}};

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
    return grant_cost_payable(card, ecd, perm_grant, caster, orderer);
}

// Can `caster` pay what `perm_grant` asks of `card`?
static bool grant_cost_payable(Entity card, const CardData &ecd, const Game::ImpulseCastPermission &perm_grant,
                               Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
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

// An alternative cost's condition: its SVar (Once Upon a Time: Count$YouCastThisGame EQ0; Mindbreak
// Trap: PlayerCountOpponents$ConditionGE3 SpellsCastThisTurn — an opponent cast three or more spells
// this turn), compared as scripted, else at least 1 (Forge's default comparator). No SVar: it holds.
static bool alt_svar_condition_holds(const std::string &svar, const std::string &compare, Zone::Ownership player) {
    if (svar.empty()) return true;
    return compare_svar(evaluate_svar(svar, player), compare.empty() ? std::string("GE1") : compare);
}

static bool can_afford_alt(const CardData& card_data, const AltCost& alt_cost,
                           Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Check SVar conditions (e.g. Once Upon a Time: free only if first spell this game)
    if (!alt_svar_condition_holds(alt_cost.condition_svar, alt_cost.condition_compare, priority_player) ||
        !alt_svar_condition_holds(alt_cost.condition_second_svar, alt_cost.condition_second_compare, priority_player))
        return false;

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

// The spell's timing, once.
static bool offer_timing_ok(CastOffer &offer) {
    if (offer.timing < 0)
        offer.timing = spell_timing_ok(offer.face, offer.window.caster, offer.window.sorcery_window);
    return offer.timing != 0;
}

// can_cast_now without the timing, once.
static bool offer_gates_ok(CastOffer &offer) {
    if (offer.gates < 0)
        offer.gates = can_cast_now(offer.face, offer.card, offer.window.caster, offer.zone, offer.window.sorcery_window,
                                   /*ignore_timing=*/true, offer.window.orderer);
    return offer.gates != 0;
}

static bool offer_castable(CastOffer &offer) { return offer_timing_ok(offer) && offer_gates_ok(offer); }

// X-cost spells: base cost (without X) is enough to be castable;
// X value is chosen at cast time in action_processor. Additional Sacrifice-a-<type>
// cost on the spell itself (Natural Order: "As an additional cost to cast this spell,
// sacrifice a green creature"): the spell can't be cast unless a matching permanent is
// available to sacrifice (CR 601.2f). General: any spell whose SPELL ability Cost$
// carries a Sac<...> token, matched (incl. color qualifier) like an activation cost.
static bool mana_cost_offered(CastOffer &offer) {
    const CastWindow &w = offer.window;
    return offer_castable(offer) &&
           regular_cast_payable(offer.face, offer.card, w.caster,
                                regular_cast_cost(offer.face, offer.card, w.caster, w.orderer), w.orderer);
}

static bool alternative_offered(CastOffer &offer) {
    return offer_castable(offer) && can_afford_alt(offer.face, offer.face.alt_cost, offer.window.caster, offer.card,
                                                   offer.window.orderer);
}

static bool evoke_offered(CastOffer &offer) { return offer.face.alt_cost.is_evoke && alternative_offered(offer); }

static bool impending_offered(CastOffer &offer) {
    return offer.face.alt_cost.is_impending && alternative_offered(offer);
}

// Spectacle (CR 702.107a): the spectacle cost may be paid only if an opponent of the
// caster lost life this turn. Two-player game — the sole opponent is the other seat.
static bool spectacle_offered(CastOffer &offer) {
    if (!offer.face.alt_cost.is_spectacle) return false;
    Zone::Ownership opp = opponent_of(offer.window.caster);
    if (global_coordinator.GetComponent<Player>(get_player_entity(opp)).life_lost_this_turn <= 0) return false;
    return alternative_offered(offer);
}

static bool warp_offered(CastOffer &offer) { return offer.face.alt_cost.is_warp && alternative_offered(offer); }

// Any other alternative cost. Miracle (CR 702.94) is never a normal priority-menu cast: it is
// castable ONLY through the dedicated immediate "cast it for its miracle cost / do not cast" decision
// presented right when its linked triggered ability resolves (proc_mandatory_choice's miracle-cast
// branch, gated on its own affordability check). So the alternate-cost cast option is never offered
// here.
static bool plain_alternative_offered(CastOffer &offer) {
    const AltCost &alt = offer.face.alt_cost;
    if (alt.is_evoke || alt.is_impending || alt.is_spectacle || alt.is_warp || alt.is_miracle) return false;
    return alternative_offered(offer);
}

// Offspring (CR 702.171): optional ADDITIONAL cost. Offer a separate cast option
// that must pay the base cost plus the offspring cost together.
static bool offspring_offered(CastOffer &offer) {
    if (!offer.face.has_offspring || !offer_castable(offer)) return false;
    const CastWindow &w = offer.window;
    ManaValue offspring_total = regular_cast_cost(offer.face, offer.card, w.caster, w.orderer);
    for (Colors c : offer.face.offspring_cost.mana()) offspring_total.insert(c);
    return regular_cast_payable(offer.face, offer.card, w.caster, offspring_total, w.orderer);
}

// Free-cast statics (Omniscience, Aluren; rules_mod::free_cast_grant): beside the card's cast for
// its own costs, offer the cast without paying its mana cost — the
// alternative cost of CR 118.9, so never combined with another one (118.9a). It keeps the
// spell's own timing, or any time its caster holds priority when the grant gives flash (CR
// 601.3c; an opponent's sorcery-speed lock still applies), and the other cast gates. Its mana is
// what cost increases and the SetCost floor add to {0} (CR 118.9d); the spell's own additional
// sacrifice must be payable, and with Offspring a second row pays that additional cost too.
static bool free_cast_offered(CastOffer &offer) {
    const CastWindow &w = offer.window;
    const rules_mod::FreeCastGrant grant = rules_mod::free_cast_grant(offer.card, offer.face, w.caster);
    if (!grant.granted) return false;
    const bool timing_ok = grant.with_flash ? w.sorcery_window || !rules_mod::opponent_sorcery_speed_locked(w.caster)
                                            : offer_timing_ok(offer);
    if (!timing_ok || !offer_gates_ok(offer)) return false;
    const ManaValue mana = floored_alt_mana_cost(offer.face, ManaValue{}, w.caster);
    CostCheck check = total_cost_check(offer.face, offer.card, w.caster, w.orderer);
    check.mana = &mana;
    return can_pay(spell_additional_cost(offer.face), check);
}

static bool free_offspring_offered(CastOffer &offer) {
    if (!offer.face.has_offspring || !free_cast_offered(offer)) return false;
    const CastWindow &w = offer.window;
    ManaValue with_offspring = floored_alt_mana_cost(offer.face, ManaValue{}, w.caster);
    for (Colors c : offer.face.offspring_cost.mana()) with_offspring.insert(c);
    CostCheck check = total_cost_check(offer.face, offer.card, w.caster, w.orderer);
    check.mana = &with_offspring;
    return can_pay(spell_additional_cost(offer.face), check);
}

// A cost that replaces the mana cost, with its mana floored: flashback and escape are alternative
// costs (CR 702.34a, 702.139a), so an active SetCost floor applies to them too.
static bool replacing_cost_payable(const CastOffer &offer, const CostParts &cost) {
    const CastWindow &w = offer.window;
    const ManaValue mana = floored_alt_mana_cost(offer.face, cost.mana(), w.caster);
    CostCheck check = total_cost_check(offer.face, offer.card, w.caster, w.orderer);
    check.mana = &mana;
    return can_pay(cost, check);
}

// A graveyard-cast static (Grafdigger's Cage: Origin$ Graveyard) prohibits flashback. Cabal
// Therapy's Flashback— Sacrifice a creature can't be cast unless a matching permanent is available
// to sacrifice (CR 601.2f / 601.3a).
static bool flashback_offered(CastOffer &offer) {
    return offer_castable(offer) && replacing_cost_payable(offer, offer.face.flashback_cost);
}

// ESCAPE (CR 702.139): a card in its owner's graveyard may be cast from there for its
// escape cost (mana + additional cost). Same sorcery-speed timing flashback uses unless the
// card is an instant. Nethergoyf's additional cost is ExileFromGrave with a "≥N card types
// among the chosen cards" constraint, so the cast is only legal when OTHER cards in the
// caster's graveyard collectively cover those N types (otherwise the cost is unpayable).
static bool escape_offered(CastOffer &offer) {
    return offer_castable(offer) && replacing_cost_payable(offer, offer.face.escape_cost);
}

static const Game::ImpulseCastPermission &offer_grant(const CastOffer &offer) {
    return *cur_game.resolved_effects.impulse_cast_permission.find(offer.card);
}

static bool grant_offered(CastOffer &offer) {
    return offer_castable(offer) &&
           grant_cost_payable(offer.card, offer.face, offer_grant(offer), offer.window.caster, offer.window.orderer);
}

// A NORMAL grant casts the card for its own costs, which leaves them open to a free-cast static's
// alternative cost; any other grant is an alternative cost itself (CR 118.9a).
static bool exile_mana_cost_offered(CastOffer &offer) {
    return offer_grant(offer).resource == Game::ImpulseCastPermission::NORMAL && grant_offered(offer);
}

static bool exile_resource_offered(CastOffer &offer) {
    const Game::ImpulseCastPermission::Resource resource = offer_grant(offer).resource;
    return (resource == Game::ImpulseCastPermission::ENERGY || resource == Game::ImpulseCastPermission::LIFE) &&
           grant_offered(offer);
}

static bool exile_no_cost_offered(CastOffer &offer) {
    return offer_grant(offer).resource == Game::ImpulseCastPermission::FREE && grant_offered(offer);
}

static bool exile_free_offered(CastOffer &offer) {
    return offer_grant(offer).resource == Game::ImpulseCastPermission::NORMAL && free_cast_offered(offer);
}

static bool exile_free_offspring_offered(CastOffer &offer) {
    return offer_grant(offer).resource == Game::ImpulseCastPermission::NORMAL && free_offspring_offered(offer);
}

// A land among the exiled cards is played, not cast (offer_exile_land).
static bool cast_from_exile(const CardData &card) { return !is_land_card(card); }

// A LAND among the exiled cards: only a "play" permission (Light Up the Stage's "you may
// PLAY those cards", Dauthi Voidwalker's "you may play it") may play it — a land play,
// sorcery-timing, own main phase, empty stack, and a land drop remaining (CR 305.1 /
// 305.2). A "cast" grant (Ugin -11) can't play a land (601.1), and play_permissions
// reports no route for a land under one.
static void offer_exile_land(CastOffer &offer, std::vector<LegalAction> &actions) {
    if (!is_land_card(offer.face) || !offer.window.sorcery_window) return;
    // Same shared land-drop expression the hand loop uses.
    if (rules_mod::land_drops_remaining(offer.window.caster) <= 0) return;
    LegalAction land_la(SPECIAL_ACTION, offer.card, "Play " + offer.face.name + " (from exile)");
    land_la.category = ActionCategory::PLAY_LAND;
    actions.push_back(land_la);
    coverage_cast_offer("exile_land");
}

// A land isn't cast (CR 601.1).
static bool cast_from_front(const CardData &card) { return !card_has_type(card, "Land"); }

// Modal DFC whose BACK face is a NONLAND spell (Tergrid, God of Fright // Tergrid's Lantern):
// a CAST_SPELL that casts the BACK face — paying the back's mana cost and using the back
// face's characteristics/abilities (CR 712.8). Timing, targets and prohibitions are the back
// face's own (Gone is an instant), so the two faces are cast independently.
static bool cast_from_back(const CardData &front) {
    // Offer the back half for both modal DFCs (nonland back) and split cards (CR 709): both
    // store the second face in `backside`.
    if ((!front.is_modal_dfc && !front.is_split) || !front.backside) return false;
    return !is_land_card(*front.backside);  // land back is a PLAY_LAND
}

// SUSPEND (CR 702.62a, first ability): from the hand, at the timing you could begin to cast
// the card (`timing_ok` — sorcery speed for a sorcery), its owner may instead pay the
// suspend cost and exile it with N time counters. This is a special action (doesn't use the
// stack). Its targets are chosen only later, when the last counter is removed and it is cast
// for free, so no legal target is required now (702.62 casts it then, not here); the card
// just must not be under a cast prohibition (702.62c) and the suspend mana cost must be
// affordable. General over any Suspend card. Offered independently of the normal-cast block.
static void offer_suspend(CastOffer &offer, std::vector<LegalAction> &actions) {
    const CardData &card_data = offer.face;
    if (!card_data.has_suspend || !cast_from_front(card_data) || !offer_timing_ok(offer)) return;
    const Zone::Ownership caster = offer.window.caster;
    CostCheck suspend_check(caster, offer.card, offer.window.orderer);
    suspend_check.payee = ManaPayee(offer.card, ManaPurpose::OTHER);
    if (rules_mod::cast_prohibited(caster, card_data) || !can_pay(card_data.suspend_cost, suspend_check) ||
        payment_blocked(offer.card))
        return;
    LegalAction sus_la(SPECIAL_ACTION, offer.card, "Suspend " + card_data.name);
    sus_la.category = ActionCategory::CAST_SPELL;  // cast-adjacent "play this card" action
    sus_la.suspend_action = true;
    actions.push_back(sus_la);
}

// The menu row of `row`, offered for `card`. The ordinal tells the cast variants apart (0 normal,
// 1 alternate / impending cost, 2 offspring, 3 modal-DFC back face, 4 flashback, 5 escape,
// 6 cast-from-graveyard permission, 7 impulse / free cast from exile, 8 without paying its mana cost
// under a free-cast static).
static LegalAction row_action(const CastRow &row, const CastSection &section, Entity card, const std::string &desc) {
    LegalAction la(CAST_SPELL, card, desc);
    la.category = ActionCategory::CAST_SPELL;
    la.option_ordinal = row.ordinal;
    la.option = CastOption{section.face, section.zone, row.cost, section.permission, row.offspring, row.destination, row.name, row.overlay};
    return la;
}

// `card`'s turn in `section`: the rows offered to it, in the table's order.
static void enumerate_cast_options(Entity card, const CastSection &section, const CastWindow &window,
                                   std::vector<LegalAction> &actions) {
    const CardData &printed = global_coordinator.GetComponent<CardData>(card);
    const bool covered = !section.covers || section.covers(printed);
    CastOption how;
    how.face = covered ? section.face : CastFace::FRONT;
    CastOffer offer{card, cast_face(printed, how), section.zone, window};
    if (covered)
        for (const CastRow &row : section.rows) {
            // A row that overlays the spell asks about the spell it makes (CR 702.103d), which the
            // rows without one don't share.
            how.overlay = row.overlay;
            CastOffer overlaid{card, cast_face(printed, how), section.zone, window};
            CastOffer &asked = row.overlay == SpellOverlay::NONE ? offer : overlaid;
            if (row.offered(asked)) {
                actions.push_back(row_action(row, section, card, "Cast " + asked.face.name + row.label));
                coverage_cast_offer(row.name);
            }
        }
    if (section.beside) section.beside(offer, actions);
}

// See declaration in cast_options.h.
LegalAction miracle_cast_action(Entity card, const std::string &name) {
    const CastRow &row = kMiracleCast.rows.front();
    return row_action(row, kMiracleCast, card, "Cast " + name + row.label);
}

// See declaration in cast_options.h.
LegalAction exile_grant_cast_action(Entity card, const std::string &name) {
    const CastRow &row = kResolutionGrantCast.rows.front();
    return row_action(row, kResolutionGrantCast, card, "Cast " + name + row.label);
}

// See declaration in cast_options.h.
void offer_cast_options(std::vector<LegalAction> &actions, const CastWindow &window, const std::vector<Entity> &hand,
                        const std::vector<std::pair<Entity, unsigned>> &routes) {
    for (const CastSection &section : kCastSections) {
        if (section.permission == Permission::NONE) {
            for (Entity card : hand) enumerate_cast_options(card, section, window, actions);
            continue;
        }
        for (const auto &[card, sources] : routes)
            if (sources & permission_bit(section.permission)) enumerate_cast_options(card, section, window, actions);
    }
}
