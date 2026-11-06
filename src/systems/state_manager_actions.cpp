#include "state_manager.h"
#include "state_manager_internal.h"
#include "rules_modifying.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
#include "../cast_options.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../queries/activation.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/costs.h"
#include "../queries/counters.h"
#include "../queries/filters.h"
#include "../queries/lki.h"
#include "../queries/permissions.h"
#include "../queries/player_resources.h"
#include "../queries/players.h"
#include "../queries/spells.h"
#include "../queries/types.h"
#include "../queries/zones.h"
#include "../game_driver.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "orderer.h"
#include "../targeting.h"

static bool present_condition_raw(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);
static std::string loyalty_cost_label(const Ability &ab);
static std::vector<Entity> stack_removal_targets(std::shared_ptr<Orderer> orderer);
static bool sorcery_timing_ok(const Game &game, Zone::Ownership seat, bool stack_empty);
static bool can_activate_now(const Ability &ab, Entity source, Zone::Ownership activator,
                             bool sorcery_window, std::shared_ptr<Orderer> orderer);
static LegalAction activate_action(Entity source, const Ability &ab, const std::string &desc,
                                   int ability_index);
// Offer the abilities of `cards`, in `zone`, that their owner activates from there.
static void offer_zone_activations(std::vector<LegalAction> &actions, const std::vector<Entity> &cards,
                                   Zone::ZoneValue zone, Zone::Ownership player, bool sorcery_window,
                                   std::shared_ptr<Orderer> orderer);
static std::string zone_activation_label(const AbilityDef &def, const std::string &name, Zone::ZoneValue zone);

// The sorcery-timing window for `seat` (CR 307.1): its own turn, a main phase, and an empty
// stack. Casting a sorcery, playing a land (CR 305.2), activating Equip (CR 702.6a), a loyalty
// ability (CR 606.3) or any "activate only as a sorcery" ability, and the companion special
// action (CR 702.139a) all wait for it.
static bool sorcery_timing_ok(const Game &game, Zone::Ownership seat, bool stack_empty) {
    return stack_empty && (game.turn_state.step == FIRST_MAIN || game.turn_state.step == SECOND_MAIN) &&
           game.turn_state.player_a_turn == (seat == Zone::PLAYER_A);
}

// The activation-legality gates every non-mana activated-ability offer shares (CR 602.2, 602.5):
// no CantBeActivated prohibition on the source, "activate only as a sorcery" timing, the source's
// state (Activation$ condition, activation limit), the cost with its mana after ReduceCost$ (can_pay),
// a legal target for every required target, and the payment-failure guard. `source` is the object the ability is activated
// from: a battlefield permanent, or a card in hand (channel, ninjutsu) or graveyard (unearth).
// Loyalty and ninjutsu windows stay with their callers.
static bool can_activate_now(const Ability &ab, Entity source, Zone::Ownership activator,
                             bool sorcery_window, std::shared_ptr<Orderer> orderer) {
    if (rules_mod::activation_prohibited(source)) return false;
    // SorcerySpeed$ True (Ba Sing Se's earthbend, unearth): activatable only any time its
    // controller could cast a sorcery (CR 602.5d).
    if (ab.def->sorcery_speed_only && !sorcery_window) return false;
    if (!activation_source_ready(ab, source, activator, orderer->mEntities)) return false;
    // Gate on the post-ReduceCost$ cost (Eiganjo's Channel is cheaper per legendary creature you
    // control) so legality matches what payment will charge.
    const ManaValue mana = effective_activation_mana_cost(ab, activator, orderer);
    CostCheck check(activator, source, orderer);
    check.payee = ManaPayee(source, ManaPurpose::ACTIVATE);
    check.mana = &mana;
    if (!can_pay(ab.def->cost, check)) return false;
    // Target existence (CR 602.2b / 601.2c), probed with the real source and activator so
    // .OppCtrl / .YouCtrl are read from the activating seat (Boseiju's Channel is never offered
    // against its own controller's nonbasic lands), over the sub-abilities too: The Stone Brain's
    // target is its first sub-ability's, and a spell's gate reads the same chain.
    if (!spell_has_castable_targets(cast_gate_probe(ab, source, activator), orderer, activator, /*has_gift=*/false))
        return false;
    return !payment_blocked(source);
}

// An ACTIVATE_ABILITY action for `ab` of `source`. `ability_index` is the ability's stable
// position in its source's ability list, emitted as the action's option_ordinal so the ML
// observation can tell same-source activations apart (e.g. a planeswalker's loyalty abilities,
// which are otherwise feature-identical).
static LegalAction activate_action(Entity source, const Ability &ab, const std::string &desc,
                                   int ability_index) {
    LegalAction la(ACTIVATE_ABILITY, source, ab, desc);
    la.category = ActionCategory::ACTIVATE_ABILITY;
    la.option_ordinal = ability_index;
    return la;
}

static void offer_zone_activations(std::vector<LegalAction> &actions, const std::vector<Entity> &cards,
                                   Zone::ZoneValue zone, Zone::Ownership player, bool sorcery_window,
                                   std::shared_ptr<Orderer> orderer) {
    for (Entity card : cards) {
        if (!global_coordinator.entity_has_component<CardData>(card)) continue;
        const CardData &card_data = global_coordinator.GetComponent<CardData>(card);
        int ability_index = -1;  // ordinal: see the battlefield loop in determine_legal_actions
        for (const auto &ab : card_data.abilities) {
            ++ability_index;
            if (ab->ability_type != AbilityDef::ACTIVATED || ab->activation_zone != zone) continue;
            const Ability zone_ab(ab);
            if (ability_is_mana(zone_ab)) continue;  // collect_mana_legal_actions offers these
            if (!can_activate_now(zone_ab, card, player, sorcery_window, orderer)) continue;
            actions.push_back(
                activate_action(card, zone_ab, zone_activation_label(*ab, card_data.name, zone), ability_index));
        }
    }
}

static std::string zone_activation_label(const AbilityDef &def, const std::string &name, Zone::ZoneValue zone) {
    if (def.is_ninjutsu) return "Ninjutsu " + name;
    if (def.is_unearth) return "Unearth " + name;
    return "Activate " + name + " from " + (zone == Zone::GRAVEYARD ? "graveyard" : "hand") + " (" + def.category + ")";
}

// Chosen targets of every stack object that would destroy or exile a battlefield
// permanent (category "Destroy", or "ChangeZone" battlefield->exile). Player-entity
// targets may be included; callers only membership-test against permanent entities.
static std::vector<Entity> stack_removal_targets(std::shared_ptr<Orderer> orderer) {
    std::vector<Entity> tgts;
    for (Entity e : orderer->get_stack()) {
        if (!global_coordinator.entity_has_component<Ability>(e)) continue;
        const Ability &ab = stack_object_ability(e);
        bool exiles_permanent =
            ab.def->kind == EffectKind::ChangeZone && ab.def->destination == Zone::EXILE &&
            (ab.def->origin == Zone::BATTLEFIELD ||
             std::find(ab.def->origins.begin(), ab.def->origins.end(), Zone::BATTLEFIELD) != ab.def->origins.end());
        if (ab.def->kind != EffectKind::Destroy && !exiles_permanent) continue;
        if (Entity t = ab.target.get()) tgts.push_back(t);
        for (Entity t : live_entities(ab.targets)) tgts.push_back(t);
    }
    return tgts;
}

// A planeswalker loyalty ability's cost as an MTG-notation suffix (" [+1]",
// " [0]", " [-3]", " [-X]"); empty for a non-loyalty ability. Shown in the
// action menu so the player sees each ability's loyalty cost, not just its
// effect category.
static std::string loyalty_cost_label(const Ability &ab) {
    if (!ab.def->is_loyalty_ability) return "";
    if (ab.def->cost.loyalty_is_x()) return ab.def->cost.loyalty() < 0 ? " [-X]" : " [+X]";
    if (ab.def->cost.loyalty() == 0) return " [0]";
    int magnitude = ab.def->cost.loyalty() < 0 ? -ab.def->cost.loyalty() : ab.def->cost.loyalty();
    std::string sign = ab.def->cost.loyalty() < 0 ? "-" : "+";
    return " [" + sign + std::to_string(magnitude) + "]";
}

// Check ConditionPresent$ / ConditionCompare$ condition (rule-603.4 intervening-if and the
// CR 608.2c resolution-time "if" gate, including ConditionDefined$ Remembered, share this).
// Counts battlefield permanents matching the filter (or remembered cards when
// condition_on_remembered) and compares against the threshold (default ">= 1").
// Filter format: "Type.YouCtrl" or "Type.OppCtrl" (e.g. "Land.YouCtrl"); "Card" matches any.
static bool present_condition_raw(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    if (ab.def->condition_present.empty()) return true;
    // Empty compare means the bare "if you control a <thing>" form → at least one.
    std::string compare = ab.def->condition_compare.empty() ? "GE1" : ab.def->condition_compare;

    // ConditionDefined$ ExiledWith (The Creation of Avacyn II & III): the condition is a property
    // check on the card the source Saga exiled face down — is it a Creature card? Match the card's
    // PRINTED characteristics against condition_present (card_matches_filter is battlefield-agnostic;
    // the exiled card sits in exile). Absent card ⇒ 0 matches (condition unmet).
    if (ab.def->condition_on_exiled_with) {
        Entity ew = exiled_with_card(ab.source.get());
        int matches = 0;
        if (ew != 0 && global_coordinator.entity_has_component<CardData>(ew)) {
            MatchCtx ctx;
            ctx.controller = caster;
            ctx.source = ab.source.lki_entity();
            if (card_matches_filter(ew, ab.def->condition_present, ctx)) matches = 1;
        }
        return compare_svar(matches, compare);
    }

    // ConditionDefined$ Remembered: count remembered cards, not battlefield permanents.
    if (ab.def->condition_on_remembered) {
        const std::vector<ObjectRef> &held = ab.def->condition_on_imprinted ? cur_game.resolution.memory.imprinted
                                                                            : cur_game.resolution.memory.remembered;
        // ConditionPresent$ Card.ExiledWithSource (Skyclave Apparition's TrigToken): only the
        // remembered cards that are STILL exiled (currently in the exile zone) count. CR 707/the
        // card's reminder text: when Skyclave leaves, the token is made only if the exiled card
        // is still exiled — if it has already returned to another zone, no token (and a card that
        // can't be found / is gone yields none either).
        if (ab.def->condition_present == "Card.ExiledWithSource") {
            size_t still_exiled = 0;
            for (Entity e : live_entities(held))
                if (global_coordinator.GetComponent<Zone>(e).location == Zone::EXILE) still_exiled++;
            return compare_svar(static_cast<int>(still_exiled), compare);
        }
        // A specific filter (other than the bare "Card") counts only the remembered cards that
        // match it as battlefield permanents. Phelia's "If it entered under your control, put a
        // +1/+1 counter on it" is ConditionDefined$ Imprinted | ConditionPresent$
        // Card.YouCtrl+ThisTurnEntered — count the returned card iff it is now a permanent the
        // ability's controller controls that entered this turn (CR 122 / the card text). Phelia's
        // counter gate checks the card the delayed trigger just put back onto the battlefield.
        if (!ab.def->condition_present.empty() && ab.def->condition_present != "Card") {
            MatchCtx ctx;
            ctx.controller = caster;
            ctx.source = ab.source.lki_entity();
            size_t matching = 0;
            for (Entity e : lki_entities(held)) {
                if (permanent_matches_filter(e, ab.def->condition_present, ctx)) {
                    matching++;
                    continue;
                }
                // A card returned to the battlefield earlier in THIS resolution (Phelia's
                // exile-and-return) has its Zone set to BATTLEFIELD with its controller assigned,
                // but its Permanent component is created by the deferred SBA pass — so it is not
                // yet a battlefield permanent for permanent_matches_filter. Fall back to Zone data:
                // it entered this turn by definition (it just returned), so the YouCtrl/OppCtrl
                // clause is checked against Zone.controller.
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                const auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location != Zone::BATTLEFIELD) {
                    // A remembered card in a non-battlefield zone (a revealed top-of-library
                    // card, or a remembered card in hand/graveyard/exile) is matched by its
                    // PRINTED characteristics — permanent_matches_filter is battlefield-only, so
                    // fall back to card_matches_filter for the type/color portion of the filter
                    // (a YouCtrl/OppCtrl qualifier is a no-op off the battlefield there).
                    if (!card_matches_filter(e, ab.def->condition_present, ctx)) continue;
                    // A controller qualifier on a card that LEFT the battlefield (Boomerang
                    // Basics: "If you controlled that permanent" against the bounced permanent) is
                    // resolved from last-known information — the controller it had as it left
                    // (CR 608.2g) — captured in cur_game.last_known_info when it left play.
                    bool youctrl = ab.def->condition_present.text().find("YouCtrl") != std::string::npos;
                    bool oppctrl = ab.def->condition_present.text().find("OppCtrl") != std::string::npos;
                    if (youctrl || oppctrl) {
                        const LastKnownInfo *lki = lki_for(e);
                        Zone::Ownership lc = lki ? lki->controller : Zone::UNKNOWN;
                        bool ctrl_ok = youctrl ? (lc == caster)
                                               : (lc != caster && lc != Zone::UNKNOWN);
                        if (!ctrl_ok) continue;
                    }
                    matching++;
                    continue;
                }
                if (global_coordinator.entity_has_component<Permanent>(e)) continue;  // handled above
                bool youctrl = ab.def->condition_present.text().find("YouCtrl") != std::string::npos;
                bool oppctrl = ab.def->condition_present.text().find("OppCtrl") != std::string::npos;
                bool ctrl_ok = youctrl ? (z.controller == caster)
                             : oppctrl ? (z.controller != caster)
                                       : true;
                if (ctrl_ok) matching++;
            }
            return compare_svar(static_cast<int>(matching), compare);
        }
        return compare_svar(static_cast<int>(held.size()), compare);
    }

    // ConditionPresent$ Card.wasCastFromYourHandByYou (Amped Raptor): the ability's source —
    // the card that triggered it — must have entered the battlefield as a spell its controller
    // cast from their own hand. Read the persisted flag off its Permanent (set when the
    // permanent was created from a hand cast). A non-hand entry leaves the flag false. If the
    // source has already left the battlefield before this trigger resolves (e.g. Amped Raptor
    // killed in response to its own ETB trigger), the Permanent is gone; fall back to the
    // last-known-information snapshot captured as it left play (CR 603.10 / 608.2h) so the
    // exile-cast clause is not silently lost.
    if (ab.def->condition_present == "Card.wasCastFromYourHandByYou") {
        const Entity self = ab.source.get();
        if (self != 0 && global_coordinator.entity_has_component<Permanent>(self))
            return global_coordinator.GetComponent<Permanent>(self).cast_from_hand_by_controller;
        const LastKnownInfo *lki = departed_lki_for(ab.source.lki_entity());
        return lki && lki->cast_from_hand_by_controller;
    }

    // IsPresent$ Card.Self: the source must itself be on the battlefield (Kappa Cannoneer's
    // "Whenever another artifact you control enters" only functions while Kappa is in play).
    if (ab.def->condition_present == "Card.Self") {
        return ability_source_on_battlefield(ab);
    }

    // IsPresent$ Card.StrictlySelf (Animate Dead's ETB: "When CARDNAME enters, if it's on the
    // battlefield, ..."): strictly this object itself must be on the battlefield when the
    // trigger would go on the stack AND when it resolves (CR 603.4 — either check failing means
    // the trigger does nothing). Unlike Card.Self this is NOT Forge's "another permanent
    // enters" idiom, so the source's own entry satisfies it (no trigger_self_excluded in
    // parse.cpp). Must not fall through to the generic presence scan below, which would read
    // the type token "Card" as any-permanent and pass vacuously whenever anything is in play.
    if (ab.def->condition_present == "Card.StrictlySelf") {
        return ability_source_on_battlefield(ab);
    }

    // Card.Self+escaped: the source permanent entered because its spell was cast from the
    // graveyard for its Escape cost (CR 702.139). Read the persisted flag off its Permanent.
    // Uro's TrigSac uses ConditionNotPresent$ Card.Self+escaped ("sacrifice it unless it
    // escaped"), so condition_negate inverts this in the wrapper below. General — any escape
    // card with an "if it escaped" clause reuses it.
    if (ab.def->condition_present == "Card.Self+escaped") {
        const Entity self = ab.source.get();
        return self != 0 && global_coordinator.entity_has_component<Permanent>(self) &&
               global_coordinator.GetComponent<Permanent>(self).cast_with_escape;
    }

    // IsPresent$ Card.Self+counters_<OP><N>_<TYPE>: the source must be on the battlefield AND its
    // count of counters of the given type satisfy the comparison <OP><N> (Forge spells the op as
    // GE/LE/EQ/NE/GT/LT). Moonshadow: "while this creature has a -1/-1 counter on it" →
    // counters_GE1_M1M1; Dark Depths: "when this has no ice counters on it" → counters_EQ0_ICE.
    // CR 122.1/603.4/603.8 — the counter count is re-checked whenever the condition is evaluated
    // (trigger placement, resolution, and each state-based check for a Mode$ Always state trigger).
    if (ab.def->condition_present.text().rfind("Card.Self+counters_", 0) == 0) {
        if (!ability_source_on_battlefield(ab)) return false;
        std::string rest = ab.def->condition_present.text().substr(std::string("Card.Self+counters_").size());
        // rest is "<OP><N>_<TYPE>", e.g. "EQ0_ICE" / "GE1_M1M1".
        std::string op = rest.substr(0, 2);          // two-letter comparator
        std::string after = rest.substr(2);          // "<N>_<TYPE>"
        size_t us = after.find('_');
        std::string num = (us != std::string::npos) ? after.substr(0, us) : after;
        std::string ctype = (us != std::string::npos) ? after.substr(us + 1) : "M1M1";
        return compare_svar(get_counters(ab.source.get(), ctype), op + num);
    }

    // Count$<...> dynamic intervening-if (Ocelot Pride's life gained this turn, Arclight
    // Phoenix's instant/sorcery spells cast this turn, ...). These are counts over game history
    // / player state, NOT board presence, so route every Count$ condition through the shared
    // SVar evaluator and compare, rather than falling through to the permanent-presence scan
    // below — where the raw "Count$..." string would be read as a permanent type name.
    if (ab.def->condition_present.text().rfind("Count$", 0) == 0)
        return compare_svar(evaluate_svar(ab.def->condition_present, caster, ab.source.lki_entity()),
                            compare);

    // A board-presence condition (Birthing Ritual's Creature.YouCtrl, Edge of Autumn's
    // Land.YouCtrl, Permanent.Red+YouCtrl+Other): count the battlefield permanents matching the
    // whole filter by their current characteristics (a Clue token is not a creature; an animated
    // manland is), relative to this ability's controller and source.
    int count = count_battlefield_matching(ab.def->condition_present, caster, ab.source.lki_entity());
    return compare_svar(count, compare);
}

// Public entry point: evaluate the present condition, applying ConditionNotPresent$ negation.
// An empty condition_present is "no condition" → always satisfied (the negate flag is never set
// in that case, since ConditionNotPresent always carries a filter). Used by the intervening-if
// trigger checks (CR 603.4) and the resolution-time condition gate (CR 608.2c) alike.
bool evaluate_present_condition(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    bool raw = present_condition_raw(ab, caster, orderer);
    return ab.def->condition_negate ? !raw : raw;
}


std::vector<LegalAction> StateManager::determine_legal_actions(
    const Game &game, std::shared_ptr<Orderer> orderer, std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;          // return value

    // Determine whose turn/priority it is
    Zone::Ownership priority_player = priority_seat();
    Entity priority_player_entity = get_player_entity(priority_player);
    bool stack_empty = stack_manager->is_empty();
    bool sorcery_window = sorcery_timing_ok(game, priority_player, stack_empty);

    // Graveyard / exile cards the priority player has some play route for (the shared
    // play_permissions predicate, which ignores timing and cost), in ascending entity
    // order, with their route bits. The graveyard land rows and offer_cast_options walk this
    // list and apply their own timing, cost and target gates.
    const std::vector<std::pair<Entity, unsigned>> zone_play_routes = play_routes(orderer->mEntities, priority_player);

    // PASS PRIORITY
    LegalAction la(PASS_PRIORITY, "Pass priority");
    la.category = ActionCategory::PASS_PRIORITY;
    actions.push_back(la);

    // LAND FROM HAND — requires empty stack (sorcery-speed)
    if (sorcery_window && global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        // Effective land play allowance (base 1 + AdjustLandPlays statics) minus the
        // lands already played, through the shared rules_mod expression the ML
        // observation's mana-development block reports.
        if (rules_mod::land_drops_remaining(priority_player) > 0) {
            // Check hand for lands
            auto hand = orderer->get_hand(priority_player);
            for (auto card_entity : hand) {
                auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
                // A modal DFC whose BACK face is a land (Witch Enchanter // Witch-Blessed
                // Meadow): playing the back face is a land play, subject to the same one-
                // land-per-turn drop (CR 712.x / 305.2). The front face is still offered as a
                // cast in the spell loop below. (A modal DFC whose back is a nonland spell
                // would instead be a cast — keyed on the back face's card type.)
                if (!card_playable_as_land(card_data)) continue;
                bool back_face = !is_land_card(card_data);
                std::string desc =
                    "Play " + (back_face ? card_data.backside->name : card_data.name);
                LegalAction la(SPECIAL_ACTION, card_entity, desc);
                la.category = ActionCategory::PLAY_LAND;
                if (back_face) la.option.face = CastFace::BACK;
                actions.push_back(la);
            }
            // Lands in the graveyard while a play-lands-from-graveyard static applies
            // (the GRAVEYARD_LAND route).
            for (const auto &[gy_e, routes] : zone_play_routes) {
                if (!(routes & permission_bit(Permission::GRAVEYARD_LAND))) continue;
                auto &gcd = global_coordinator.GetComponent<CardData>(gy_e);
                std::string desc = "Play " + gcd.name + " (from graveyard)";
                LegalAction la(SPECIAL_ACTION, gy_e, desc);
                la.category = ActionCategory::PLAY_LAND;
                actions.push_back(la);
            }
        }
    }
    // COMPANION (CR 702.139) — at sorcery speed, the player may pay {3} ONCE per game to put their
    // chosen companion from the sideboard ("outside the game") into their hand. The chosen companion
    // and the deckbuilding-restriction gate were resolved at game start (setup_companions); here we
    // only offer the special action while the companion is still in the sideboard, it hasn't been
    // used yet, and {3} is affordable. Mirrors the play-land special action's timing gate.
    if (sorcery_window && global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        auto &player = global_coordinator.GetComponent<Player>(priority_player_entity);
        Entity comp = player.chosen_companion;
        if (comp != 0 && !player.companion_brought_to_hand &&
            global_coordinator.entity_has_component<Zone>(comp) &&
            global_coordinator.GetComponent<Zone>(comp).location == Zone::SIDEBOARD &&
            global_coordinator.entity_has_component<CardData>(comp)) {
            ManaValue three = {GENERIC, GENERIC, GENERIC};
            if (can_pay_mana(priority_player, three, ManaPayee(comp, ManaPurpose::OTHER), orderer) &&
                !payment_blocked(comp)) {
                auto &cd = global_coordinator.GetComponent<CardData>(comp);
                LegalAction la(SPECIAL_ACTION, comp,
                               "Companion: pay {3}, put " + cd.name + " into your hand");
                la.category = ActionCategory::COMPANION;
                la.companion_to_hand = true;
                la.card_is_public = true;
                actions.push_back(la);
            }
        }
    }

    const auto hand = orderer->get_hand(priority_player);
    offer_cast_options(actions, CastWindow{priority_player, sorcery_window, orderer}, hand, zone_play_routes);
    // checking permanents for activated abilities
    // mana abilities parsed last
    // Simple tap-only mana sources collected via shared function
    std::vector<LegalAction> legal_mana_abilities =
        collect_mana_legal_actions(priority_player, orderer, 0, /*at_priority=*/true);
    for (auto entity : orderer->mEntities) {
        if (!is_battlefield_permanent(entity, priority_player)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);

        // Check if any CantBeActivated static suppresses this permanent's abilities.
        // (Mana abilities are collected separately above, so they remain usable — this
        // matches Disruptor Flute's ValidSA$ Activated.!ManaAbility.)
        if (rules_mod::activation_prohibited(entity)) continue;

        // ability_index: the ability's stable position in this permanent's ability
        // list, emitted as the action's option_ordinal so the ML observation can
        // tell same-permanent activations apart (e.g. a planeswalker's loyalty
        // abilities, which are otherwise feature-identical). Counted over the FULL
        // list (including skipped/mana abilities) so an ability keeps one ordinal
        // regardless of which of its siblings happen to be legal right now.
        int ability_index = -1;
        for (const auto &ab : permanent.abilities) {
            ++ability_index;
            if (ab.def->ability_type != AbilityDef::ACTIVATED) continue;
            if (ab.def->activation_zone != -1) continue;  // functions from another zone (CR 113.6b): hand, graveyard
            // Loyalty abilities (606.3): sorcery-speed only, once per turn per permanent across
            // all its loyalty abilities.
            if (ab.def->is_loyalty_ability) {
                if (!sorcery_window) continue;
                if (permanent.loyalty_ability_activated_this_turn) continue;
            }
            // All mana abilities — including InstantSpeed$ ones (e.g. LED) and AB$ ManaReflected
            // (Mox Amber) — are collected via collect_mana_legal_actions above and resolve
            // off-stack. None go on the stack.
            if (ability_is_mana(ab)) continue;
            // Non-mana activated ability (e.g. ChangeZone for fetch lands, Destroy for Wasteland).
            if (!can_activate_now(ab, entity, priority_player, sorcery_window, orderer)) continue;
            std::string desc = !ab.def->keyword_label.empty()
                                   ? ab.def->keyword_label + " " + entity_name(entity)
                                   : "Activate " + entity_name(entity) + loyalty_cost_label(ab) +
                                         " (" + ab.def->category + ")";
            actions.push_back(activate_action(entity, ab, desc, ability_index));
        }
    }
    // Abilities that function from the hand or the graveyard (CR 113.6b): Talon Gates of Madara,
    // ninjutsu, Unearth (CR 702.84), Runehorn Hellkite.
    offer_zone_activations(actions, hand, Zone::HAND, priority_player, sorcery_window, orderer);
    offer_zone_activations(actions, orderer->get_graveyard(priority_player), Zone::GRAVEYARD, priority_player,
                           sorcery_window, orderer);

    // With auto-mana off every collected mana ability is listed, so mana can be floated
    // manually whether or not it contributes to a castable spell.
    std::vector<Entity> removal_tgts;
    if (auto_mana && !legal_mana_abilities.empty()) removal_tgts = stack_removal_targets(orderer);
    for (auto &ma : legal_mana_abilities) {
        // With auto-mana on, normal mana sources stay hidden and are auto-paid during cost
        // payment, with three exceptions offered at priority so mana can still be floated:
        // instant-speed sources (e.g. LED, only activatable here), a card's ability from the hand
        // (Simian Spirit Guide, a card the player may spend whether or not a payment needs it), and any source that is
        // a chosen target of a Destroy/exile effect on the stack (float in response to
        // removal, e.g. Wasteland, before the source leaves the battlefield). A rider that gives
        // a keyword is a fourth: the payer never chooses it, its mana being no more than the tap's
        // (Arena of Glory).
        const ManaRider *rider = ma.ability.def->mana_rider;
        if (auto_mana && !ma.ability.def->instant_speed && ma.ability.def->activation_zone == -1 &&
            !(rider && !rider->keyword.empty()) &&
            std::find(removal_tgts.begin(), removal_tgts.end(), ma.source_entity) ==
                removal_tgts.end())
            continue;
        actions.push_back(ma);
    }
    return actions;
}