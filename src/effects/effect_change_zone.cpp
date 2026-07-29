#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../classes/action.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static bool search_reveals_card(const Ability &ab);
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest, bool exile_face_down = false);
static void register_exile_until_host_leaves(Entity host, Entity card, Zone::ZoneValue origin);
static HandlerResult each_player_put_from_hand(Ability &ab, std::shared_ptr<Orderer> orderer,
                                               FrameCtx &fctx);

// Name an object for a log line / action label without assuming it is a card: a token has a
// Permanent (and Token) but no CardData, so reading CardData on it crashes. Delegates to the
// shared resolver (game_queries.h), which also names lingering tokens and ability entities.
static std::string object_display_name(Entity e) { return entity_name(e); }

// Move a card for a ChangeZone effect and report the zone it actually landed in. A
// replacement effect can divert the move during add_to_zone — Containment Priest redirects an
// uncast creature to exile (614.1a), Grafdigger's Cage prevents the entry so the card stays in
// its origin zone (614.13) — so the result may differ from `dest`. Callers gate their
// battlefield-only bookkeeping (controller / enters-tapped / enters-transformed) and their
// "moved to <dest>" log on the returned zone; when it differs from `dest` the replacement
// dispatcher has already logged the reason for the divert, so no generic line is emitted.
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest, bool exile_face_down) {
    // CR 110.4a / 712.10: only permanents exist on the battlefield. An effect that would put a
    // non-permanent card onto the battlefield can't — the card stays in its current zone. The
    // case that reaches here is a double-faced card returning from exile via a flicker (e.g.
    // Flickerwisp/Phelia on a modal DFC like Fell the Profane // Fell Mire): a DFC returns with
    // its FRONT face up (the card already shows its front face once off the battlefield), so the
    // entity's current CardData decides. If that front face is an instant/sorcery it can't enter,
    // and the card remains exiled. Permanent-faced cards and tokens (always permanents) are
    // unaffected; like a Grafdigger's Cage divert, the caller's battlefield bookkeeping/log is
    // gated on the returned zone, so nothing is logged as having entered.
    if (dest == Zone::BATTLEFIELD &&
        global_coordinator.entity_has_component<CardData>(e) &&
        !is_permanent_card(global_coordinator.GetComponent<CardData>(e))) {
        return global_coordinator.GetComponent<Zone>(e).location;
    }
    orderer->add_to_zone(false, e, dest, /*top_seen_by_owner=*/true, exile_face_down);
    return global_coordinator.GetComponent<Zone>(e).location;
}

// CR 603.6e linked exile-and-return ("exile ... until [host] leaves the battlefield"). Records
// that `host` exiled `card` from `origin` under a Duration$ UntilHostLeavesPlay, and registers a
// delayed trigger that RETURNS the card when the host leaves the battlefield. The return goes
// back to `origin`: a HAND card to its owner's hand, a BATTLEFIELD permanent onto the battlefield
// under its owner's control (a fresh object — re-entry triggers fire normally). Reused by Cloak
// and Dagger (a revealed hand card or the chosen creature) and by the single-target-permanent
// exilers Sheltered by Ghosts / Static Prison (origin BATTLEFIELD). One trigger is registered per
// exiled card, so each card carries its own origin even when a host exiles several.
static void register_exile_until_host_leaves(Entity host, Entity card, Zone::ZoneValue origin) {
    if (host == 0 || card == 0) return;
    // Track the exiled card on the host's Permanent (snapshotted into last-known info when the
    // host leaves; also the channel Keen-Eyed Curator-style "cards exiled with this" effects read).
    if (global_coordinator.entity_has_component<Permanent>(host))
        global_coordinator.GetComponent<Permanent>(host).exiled_with.push_back(card);

    // The fire ability returns this one card from exile to its origin zone. Seeding
    // restore_remembered_exiled_with makes resolve() set the remembered set to exactly this card,
    // and the Defined$ Remembered move (source = card) sends it to fire_ab.destination under its
    // owner's control (owner = the card's Zone.owner).
    Ability fire_ab;
    fire_ab.ability_type = Ability::TRIGGERED;
    fire_ab.category = "ChangeZone";
    fire_ab.defined_remembered = true;
    fire_ab.restore_remembered_exiled_with = {card};
    fire_ab.source = card;
    fire_ab.origin = Zone::EXILE;
    fire_ab.destination = origin;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::CARD_CHANGED_ZONE;
    dt.owner_entity = get_player_entity(source_controller(host));
    dt.fire_on_turn = cur_game.turn;
    dt.watch_entity = host;
    dt.fire_on_leave_battlefield = true;
    cur_game.delayed_triggers.push_back(dt);
}

// A library search reveals the chosen card when it must satisfy a restriction
// more specific than "any card" — the searcher proves the card qualifies (e.g.
// Personal Tutor: "search for a sorcery card, reveal it"). An unrestricted
// "Card" search (Vampiric/Demonic Tutor, Doomsday) reveals nothing. The
// sideboard ("outside the game") is likewise hidden, and wish effects say
// "reveal it" (Karn, the Great Creator -2), so it reveals the same way.
// Searches of other zones are public already.
static bool search_reveals_card(const Ability &ab) {
    bool from_hidden = (ab.origin == Zone::LIBRARY || ab.origin == Zone::SIDEBOARD);
    for (auto z : ab.origins) {
        if (z == Zone::LIBRARY || z == Zone::SIDEBOARD) from_hidden = true;
    }
    bool specific_type = !ab.change_type.empty() && ab.change_type != "Card";
    return from_hidden && specific_type;
}

// Show and Tell family: DefinedPlayer$ Player — EACH player may put one matching card from THEIR
// OWN hand onto the battlefield, in APNAP order (CR 101.4 / 405.6: active player first). It is a
// "may" per player (each may decline), and each player chooses privately from their own hand. A
// card put this way enters under its owner's control (CR 110.2a; no mana is paid — big creatures
// like Griselbrand simply enter, they are not cast), and its ETB fires via the normal SBA pass.
// Suspend-safe: only the player index persists (EachPlayerPutRt); each player's candidate menu is
// re-derived from their live hand every pass (a card already put has left the hand).
static HandlerResult each_player_put_from_hand(Ability &ab, std::shared_ptr<Orderer> orderer,
                                               FrameCtx &fctx) {
    Zone::Ownership active = cur_game.player_a_active ? Zone::PLAYER_A : Zone::PLAYER_B;
    Zone::Ownership nonactive = (active == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    Zone::Ownership order[2] = {active, nonactive};

    EachPlayerPutRt local_rt;
    EachPlayerPutRt &rt = fctx.can_suspend() ? fctx.rt<EachPlayerPutRt>() : local_rt;

    MatchCtx mctx;
    for (; rt.player_idx < 2; rt.player_idx++) {
        Zone::Ownership p = order[rt.player_idx];
        mctx.controller = p;
        // Candidates: this player's own hand cards matching the ChangeType filter
        // (Creature,Artifact,Enchantment,Land). Re-derived every pass so a resume rebuilds the
        // identical menu (and a card already put no longer appears).
        std::vector<Entity> cands;
        for (auto e : orderer->get_hand(p)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            // A hand card is matched by its PRINTED characteristics (card_matches_any), not the
            // battlefield-permanent matcher — the card is not on the battlefield yet.
            if (card_matches_any(e, ab.change_type, mctx)) cands.push_back(e);
        }
        if (cands.empty()) continue;  // no eligible card — this player puts nothing

        std::vector<LegalAction> picks;
        for (auto e : cands) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, std::string("Put ") + cd.name + " onto the battlefield");
            la.category = ActionCategory::CHOOSE_CARD;
            picks.push_back(la);
        }
        LegalAction none(PASS_PRIORITY, std::string("Put nothing"));
        none.category = ActionCategory::CHOOSE_CARD;
        picks.push_back(none);

        int choice = fctx.ask(std::move(picks), p, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (choice < 0 || choice >= static_cast<int>(cands.size())) continue;  // declined
        Entity chosen = cands[static_cast<size_t>(choice)];
        std::string cname = global_coordinator.GetComponent<CardData>(chosen).name;
        Zone::ZoneValue landed = change_zone_move(orderer, chosen, Zone::BATTLEFIELD);
        if (landed == Zone::BATTLEFIELD) {
            global_coordinator.GetComponent<Zone>(chosen).controller = p;  // owner's control (110.2a)
            game_log("%s puts %s onto the battlefield\n", player_name(p).c_str(), cname.c_str());
        }
    }
    return HandlerResult::DONE_RUN_SUBS;
}

HandlerResult change_zone(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &fctx) {
    PendingDecisionScope pending_scope(ab.source);
    // Same-name search/move (Surgical Extraction, Infernal Tutor, Secret Salvage, Pack
    // Hunt, ...): ChangeType$ Remembered.sameName / Targeted.sameName.
    if (ab.change_type.find("sameName") != std::string::npos)
        return change_zone_same_name(ab, orderer, /*force_all=*/false)
                   ? HandlerResult::DONE_RUN_SUBS
                   : HandlerResult::DONE_NO_SUBS;

    // The owning/searching player for this ChangeZone. Normally the source's Zone.owner, but
    // the source may have ceased to exist by the time the ability resolves (CR 608.2g/h): a
    // token that created an ability still on the stack is DestroyEntity'd the instant it leaves
    // the battlefield, taking its Zone with it. Fall back to the last-known controller (== owner
    // for a token, which is always owned by its controller), the same last-known-information the
    // target reads below already rely on.
    Zone::Ownership owner = global_coordinator.entity_has_component<Zone>(ab.source)
                                ? global_coordinator.GetComponent<Zone>(ab.source).owner
                                : last_known_controller(ab.source);

    // DefinedPlayer$ TargetedController (Erode / White Orchid Phantom's DBSearch: "Its
    // controller may search their library for a basic land card, put it onto the battlefield
    // tapped, then shuffle."): the searching/owning player for a search-based ChangeZone is
    // the targeted card's controller, not the spell's caster (CR 109.5). The target may
    // already have left the battlefield (the parent Destroy ran first), but Zone.controller
    // persists through the move to the graveyard, so it still names the last controller. This
    // only redirects the search path; the targeted-move branch below is skipped because such
    // a sub-ability carries no target of its own (ValidTgts$ N_A).
    //
    // The defined player is derived FROM the parent's chosen target, so with no target the
    // definition names no one and the effect does nothing (White Orchid Phantom's "up to one
    // target" trigger with zero targets picked: no land was destroyed, so nobody searches —
    // the search must not fall back to the caster's own library).
    if (ab.defined_targeted_controller) {
        Zone::Ownership tc =
            ab.target != 0 ? last_known_controller(ab.target)  // CR 608.2g/h, robust to post-move reads
                           : Zone::UNKNOWN;
        if (tc == Zone::UNKNOWN) return HandlerResult::DONE_RUN_SUBS;  // no targeted object → no defined player; still chain subs
        owner = tc;
    }

    // DefinedPlayer$ Targeted — the searched/owning zone belongs to the chain's targeted PLAYER,
    // so a Hand search picks from THEIR hand. Usually that player is this sub-ability's own bound
    // target (Thought-Knot Seer: bind_sub_target inherited the opponent the parent RevealHand
    // targeted). When an intervening sub retargeted a CARD instead (Cloak and Dagger, Entwined:
    // DBPump chose the opponent's creature, so this sub inherited that creature), the player is
    // read from targeted_player — the nearest player target up the chain, propagated by
    // bind_sub_target the same way Forge walks ancestors. The targeted-move branch below is
    // skipped in both cases because this sub-ability carries no target of its own (ValidTgts$
    // N_A); the bound target/targeted_player only name the searched player here.
    if (ab.defined == "Targeted") {
        Entity ptgt = (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
                          ? ab.target
                          : ab.targeted_player;
        if (ptgt != 0 && global_coordinator.entity_has_component<Player>(ptgt))
            owner = (ptgt == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    }

    // DefinedPlayer$ Player — EACH player may put a matching card from THEIR OWN hand onto the
    // battlefield (Show and Tell), in APNAP order. Not a search (no filter reveal / library
    // shuffle) and not a caster-only put: it is a per-player "may". Routed here before the generic
    // caster-scoped search path below, which would otherwise put only the caster's card.
    if (ab.defined == "Player" && ab.origin == Zone::HAND && ab.destination == Zone::BATTLEFIELD &&
        ab.valid_tgts == "N_A")
        return each_player_put_from_hand(ab, orderer, fctx);

    const char *dest_str = ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.destination == Zone::LIBRARY   ? "top of library"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                                                               : "exile";

    // Targeted ChangeZone (e.g. Swords to Plowshares, Faerie Macabre, Life from the
    // Loam): move target(s) directly. A targeted ability with no chosen targets (e.g.
    // "up to three" with zero picked) does nothing rather than searching.
    if (ab.valid_tgts != "N_A") {
        std::vector<Entity> to_move;
        if (!ab.targets.empty()) {
            to_move = ab.targets;
        } else if (ab.target != 0) {
            to_move.push_back(ab.target);
        }
        // ConditionDefined$ Targeted with a dynamic mana-value bound (Prismatic Ending: exile the
        // target only if its mana value <= Converge, the number of colors of mana spent — cmcLEY,
        // Y = Count$Converge). CR 702.90 / 608.2b: the gate is enforced HERE at resolution (cast
        // legality does not gate on condition_on_target), so a target whose MV exceeds the bound is
        // simply not moved (the spell does nothing to it). General to any targeted ChangeZone whose
        // ConditionPresent carries a cmcLE<svar> threshold resolved into dynamic_amount_expr.
        int cmc_threshold = -1;
        if (ab.condition_on_target && !ab.dynamic_amount_expr.empty() &&
            ab.condition_present.find("cmcLE") != std::string::npos)
            cmc_threshold = static_cast<int>(
                evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer, ab.target, ab.source));

        for (auto tgt : to_move) {
            if (!global_coordinator.entity_has_component<Zone>(tgt)) continue;
            if (cmc_threshold >= 0 && global_coordinator.entity_has_component<CardData>(tgt)) {
                int tgt_cmc = card_mana_value(global_coordinator.GetComponent<CardData>(tgt));
                if (tgt_cmc > cmc_threshold) {
                    game_log("%s has mana value %d (Converge %d) — not exiled\n",
                             entity_name(tgt).c_str(), tgt_cmc, cmc_threshold);
                    continue;
                }
            }
            // Pre-move origin zone — captured before the move for a Duration$ UntilHostLeavesPlay
            // exile so the return knows where the card came from (Sheltered by Ghosts / Static
            // Prison target a battlefield permanent: origin BATTLEFIELD).
            Zone::ZoneValue tgt_origin = global_coordinator.GetComponent<Zone>(tgt).location;
            std::string tname = entity_name(tgt);
            // A target that is a spell/ability on the stack (Mindbreak Trap: "exile any
            // number of target spells") is being removed from the stack. Strip its Spell/
            // Ability components — like effects::counter does — so the stack no longer
            // treats it as a live object to resolve (CR 701.5a / 702.59c). A standalone
            // ability entity (no card / CardData) is destroyed after leaving the stack.
            bool on_stack = global_coordinator.GetComponent<Zone>(tgt).location == Zone::STACK;
            bool standalone_ability = on_stack &&
                                      !global_coordinator.entity_has_component<CardData>(tgt) &&
                                      global_coordinator.entity_has_component<Ability>(tgt);
            if (on_stack) {
                if (global_coordinator.entity_has_component<Spell>(tgt))
                    global_coordinator.RemoveComponent<Spell>(tgt);
                if (global_coordinator.entity_has_component<Ability>(tgt))
                    global_coordinator.RemoveComponent<Ability>(tgt);
            }
            // Targeted reanimation (Lorehold Charm: graveyard→battlefield). A permanent
            // entering this way comes under the controller's control (CR 608.2; the spell's
            // controller is the one returning it from their own graveyard). enters_tapped/
            // enters_transformed honour the same flags the search/defined paths use.
            if (ab.destination == Zone::BATTLEFIELD && ab.origin != Zone::BATTLEFIELD) {
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(tgt);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(tgt);
            }
            Zone::ZoneValue landed = change_zone_move(orderer, tgt, ab.destination);
            if (landed == Zone::BATTLEFIELD && ab.origin != Zone::BATTLEFIELD)
                global_coordinator.GetComponent<Zone>(tgt).controller = ab.controller;
            if (standalone_ability) {
                global_coordinator.DestroyEntity(tgt);
                game_log("%s is exiled from the stack\n", tname.c_str());
                continue;
            }
            if (ab.destination == Zone::EXILE && ab.source != 0 &&
                global_coordinator.entity_has_component<Permanent>(ab.source)) {
                // Duration$ UntilHostLeavesPlay: register the linked return (which also records
                // exiled_with). Otherwise just record the exile for "cards exiled with this"
                // readers (Skyclave Apparition / Keen-Eyed Curator).
                if (ab.duration_until_host_leaves)
                    register_exile_until_host_leaves(ab.source, tgt, tgt_origin);
                else
                    global_coordinator.GetComponent<Permanent>(ab.source).exiled_with.push_back(tgt);
            }
            // RememberChanged$ True (Skyclave Apparition's TrigExile): stash the moved card in
            // the remembered set so a later SVar (Remembered$CardManaCost) and a paired
            // leaves-the-battlefield ability (TrigToken sizing/owning the Illusion) can read it.
            if (ab.remember_changed || ab.remember_lki) cur_game.remembered_entities.push_back(tgt);
            if (landed == ab.destination)
                game_log("%s is moved to %s\n", tname.c_str(), dest_str);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ Enchanted — move the object this Aura enchants (CR 303.4 / 608.2c). Animate Dead's
    // ETB reanimation: the aura was cast "enchanting" a creature card in a graveyard (recorded at
    // cast in pending_aura_target since the card had no Permanent to attach to), and this trigger
    // returns THAT card to the battlefield under the aura controller's control (GainControl$) and
    // remembers it (RememberChanged$) so the chained DB$ Attach re-attaches the aura to it. For a
    // normal already-attached aura reading "enchanted", fall back to its live attachment link.
    if (ab.defined == "Enchanted" && ab.source != 0) {
        Entity enchanted = 0;
        auto pat = cur_game.pending_aura_target.find(ab.source);
        if (pat != cur_game.pending_aura_target.end())
            enchanted = pat->second;
        else if (global_coordinator.entity_has_component<Permanent>(ab.source))
            enchanted = global_coordinator.GetComponent<Permanent>(ab.source).equipped_to;
        if (enchanted != 0 && global_coordinator.entity_has_component<Zone>(enchanted)) {
            std::string ename = entity_name(enchanted);
            if (ab.enters_tapped && ab.destination == Zone::BATTLEFIELD)
                cur_game.pending_enters_tapped.insert(enchanted);
            Zone::ZoneValue landed = change_zone_move(orderer, enchanted, ab.destination);
            if (landed == Zone::BATTLEFIELD)
                // GainControl$ True: the card enters under the aura controller's control (CR 110.2a
                // / 608.2). Without the gain-control flag it would enter under its owner's control.
                global_coordinator.GetComponent<Zone>(enchanted).controller =
                    ab.gain_control ? ab.controller
                                    : global_coordinator.GetComponent<Zone>(enchanted).owner;
            if (ab.remember_changed || ab.remember_lki)
                cur_game.remembered_entities.push_back(enchanted);
            if (landed == ab.destination)
                game_log("%s is moved to %s\n", ename.c_str(), dest_str);
        }
        // The reanimation is done; drop the pending marker so the "unattached aura" state-based
        // action resumes governing this aura (the immediately following apply_permanent_components
        // pass finalizes the DB$ Attach that this chain queues, before that SBA runs).
        cur_game.pending_aura_target.erase(ab.source);
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ Self — move the source card directly (e.g. Talon Gates putting itself onto battlefield from hand)
    if (ab.defined_self && ab.source != 0) {
        std::string sname = entity_name(ab.source);
        if (ab.enters_tapped && ab.destination == Zone::BATTLEFIELD)
            cur_game.pending_enters_tapped.insert(ab.source);
        Zone::ZoneValue landed = change_zone_move(orderer, ab.source, ab.destination);
        // RememberChanged$ True — record the moved card so a later chained sub-ability can act on
        // it via Card.IsRemembered / Count$RememberedSize (Triumph of Saint Katherine's self-exile
        // head of its recursion pile, counted by the GE7 shuffle-back condition).
        if (ab.remember_changed) cur_game.remembered_entities.push_back(ab.source);
        if (landed == Zone::BATTLEFIELD) {
            global_coordinator.GetComponent<Zone>(ab.source).controller = owner;
            // Unearth (CR 702.84): flag the returning permanent so its Permanent is created
            // unearthed (haste + delayed end-step exile + leaves-the-battlefield exile).
            if (ab.is_unearth) cur_game.pending_unearthed.insert(ab.source);
        }
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", sname.c_str(), dest_str);
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ ExiledWith — move the card the source Saga chapter I exiled face down (The Creation
    // of Avacyn chapter III: "You may put the exiled card onto the battlefield if it's a creature
    // card. If you don't, put it into its owner's hand."). The card is tracked in the Saga source's
    // Permanent::exiled_with. This runs as a two-leg chain: DBReturn (Optional$, Destination$
    // Battlefield, gated on ConditionPresent$ Creature) then DBChangeZone (Destination$ Hand). Each
    // leg acts only while the card is still in its Origin$ (Exile) zone, so once one leg moves it the
    // other naturally no-ops — declining the battlefield put (or a noncreature skipping the gated
    // first leg) leaves it in exile for the hand leg.
    if (ab.defined_exiled_with) {
        Entity card = exiled_with_card(ab.source);
        if (card == 0 || !global_coordinator.entity_has_component<Zone>(card))
            return HandlerResult::DONE_RUN_SUBS;
        Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(card).location;
        bool in_origin = ab.origins.empty() ? (loc == ab.origin) : false;
        for (auto z : ab.origins) if (z == loc) in_origin = true;
        if (!in_origin) return HandlerResult::DONE_RUN_SUBS;  // already moved by the other leg

        std::string cname = entity_name(card);
        if (ab.optional_choice) {
            std::vector<LegalAction> yn;
            LegalAction decline(PASS_PRIORITY, std::string("Leave ") + cname + " in exile");
            decline.category = ActionCategory::OPTIONAL_YESNO;
            decline.option_ordinal = 0;
            yn.push_back(decline);
            LegalAction accept(PASS_PRIORITY, std::string("Put ") + cname + " onto the battlefield");
            accept.category = ActionCategory::OPTIONAL_YESNO;
            accept.option_ordinal = 1;
            yn.push_back(accept);
            int yc = fctx.ask(std::move(yn), ab.controller, ab.source);
            if (yc < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
            if (yc == 0) return HandlerResult::DONE_RUN_SUBS;  // declined — stays in exile for the hand leg
        }
        if (ab.enters_tapped && ab.destination == Zone::BATTLEFIELD)
            cur_game.pending_enters_tapped.insert(card);
        Zone::ZoneValue landed = change_zone_move(orderer, card, ab.destination);
        if (landed == Zone::BATTLEFIELD)
            // The exiled card enters under its OWNER's control (CR 110.2a).
            global_coordinator.GetComponent<Zone>(card).controller =
                global_coordinator.GetComponent<Zone>(card).owner;
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", cname.c_str(), dest_str);
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ Remembered with a bounded/optional selection (Cloak and Dagger's DBChangeZone:
    // "you MAY exile up to one of the remembered candidates"). Unlike the blanket defined_remembered
    // move below (Ajani, which moves EVERY remembered object), this PRESENTS A CHOICE of which
    // remembered object(s) to move: ChangeNum$ N caps the count and Optional$ True permits
    // declining. Candidates are the remembered objects currently in an eligible Origin zone (Hand
    // or Battlefield); nonland-only, honoring the oracle "exile a nonland card from their hand or
    // the chosen creature" (the creature is itself nonland). Only reached with an explicit
    // count/optionality, so the Ajani blanket path is unaffected.
    if (ab.defined_remembered && (ab.optional_choice || ab.change_num >= 0)) {
        int cap = (ab.change_num >= 0) ? ab.change_num
                                       : (ab.amount > 0 ? static_cast<int>(ab.amount) : 1);
        bool optional = ab.optional_choice;
        // Live-menu loop (Shape C): only the pick counter persists — the
        // candidate pool, and hence the menu, is rebuilt from the remembered
        // set's current zones at every arm AND at every consume (that rebuild
        // is what maps the latched answer back to an entity; ask's size assert
        // guards drift). The asks are seated on the controller through
        // fctx.ask, replacing the old manual seat swap around the whole loop.
        ChangeZoneRememberedRt local_rt;
        ChangeZoneRememberedRt &rt =
            fctx.can_suspend() ? fctx.rt<ChangeZoneRememberedRt>() : local_rt;
        for (; rt.picked < cap; rt.picked++) {
            // Rebuild the candidate list each pick (a card already moved leaves the eligible zones).
            std::vector<Entity> cands;
            for (auto e : cur_game.remembered_entities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(e).location;
                bool zone_ok = false;
                for (auto z : ab.origins) if (z == loc) zone_ok = true;
                if (!zone_ok) continue;
                if (is_land_card(global_coordinator.GetComponent<CardData>(e))) continue;  // nonland (oracle)
                cands.push_back(e);
            }
            if (cands.empty()) break;
            std::vector<LegalAction> picks;
            for (auto e : cands) {
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(e).location;
                const char *where = (loc == Zone::BATTLEFIELD) ? " (the chosen creature)" : " (from hand)";
                LegalAction la(PASS_PRIORITY, e, std::string("Exile ") + cd.name + where);
                la.category = ActionCategory::CHOOSE_CARD;
                la.card_is_public = true;
                picks.push_back(la);
            }
            if (optional) {
                LegalAction none(PASS_PRIORITY, std::string("Exile nothing"));
                none.category = ActionCategory::CHOOSE_CARD;
                picks.push_back(none);
            }
            // Arm-only pre-ask log: a resume re-enters this same pick with the
            // announcement already printed at arm time.
            if (!fctx.resuming())
                game_log("%s may exile a card until %s leaves the battlefield:\n",
                    player_name(ab.controller).c_str(),
                    global_coordinator.entity_has_component<CardData>(ab.source)
                        ? global_coordinator.GetComponent<CardData>(ab.source).name.c_str() : "the source");
            int choice = fctx.ask(std::move(picks), ab.controller, ab.source);
            if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
            if (choice < 0 || choice >= static_cast<int>(cands.size())) break;  // declined
            Entity chosen = cands[static_cast<size_t>(choice)];
            Zone::ZoneValue card_origin = global_coordinator.GetComponent<Zone>(chosen).location;
            Zone::Ownership card_owner = global_coordinator.GetComponent<Zone>(chosen).owner;
            std::string cname = global_coordinator.GetComponent<CardData>(chosen).name;
            change_zone_move(orderer, chosen, ab.destination);
            mark_card_revealed(chosen, card_owner);
            game_log("%s exiles %s\n", player_name(ab.controller).c_str(), cname.c_str());
            if (ab.duration_until_host_leaves)
                register_exile_until_host_leaves(ab.source, chosen, card_origin);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Defined$ Remembered — move the remembered card(s) directly. Ajani's exile-and-
    // return chain uses this: TrigExile remembers the exiled permanent, DBReturn brings
    // that same card back to the battlefield (transformed) under its owner's control.
    if (ab.defined_remembered) {
        for (auto e : cur_game.remembered_entities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            // Stale remembered object: only move a card that is actually in the ability's
            // declared Origin$ zone. A delayed return (Phelia/Flickerwisp's TrigBounce,
            // Origin$ Exile) must not "return" a card that already left exile — or a
            // recycled entity id — since the phantom move would emit a false
            // entered-the-battlefield event (falsely firing ETB watchers like Guide of Souls).
            if (!ab.origins.empty()) {
                Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(e).location;
                bool in_origin = false;
                for (auto z : ab.origins)
                    if (z == loc) in_origin = true;
                if (!in_origin) continue;
            }
            std::string nm = entity_name(e);
            Zone::ZoneValue landed = change_zone_move(orderer, e, ab.destination);
            if (landed == Zone::BATTLEFIELD) {
                // Forge default for ChangeZone Destination$ Battlefield: the card enters
                // under its OWNER's control unless GainControl$ True (CR 110.2a). The
                // delayed exile-returns that use this branch (Phelia, Flickerwisp, Hide on
                // the Ceiling, Yorion, Ajani) all return the card to its owner — assigning
                // the ability controller here let an attacker steal an opponent's permanent.
                auto &ezone = global_coordinator.GetComponent<Zone>(e);
                ezone.controller = ezone.owner;
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(e);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(e);
            }
            if (landed == ab.destination)
                game_log("%s is moved to %s\n", nm.c_str(), dest_str);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // A ChangeZone leaving the battlefield with no target and no search filter operates
    // on the ability's own source (Forge's Defined$ Self default; e.g. Ajani's TrigExile
    // exiles himself). search_zone can't enumerate the battlefield, so a self-move is the
    // only sensible reading of a battlefield origin here.
    if (ab.origin == Zone::BATTLEFIELD && ab.change_type.empty() && ab.source != 0 &&
        global_coordinator.entity_has_component<Zone>(ab.source)) {
        std::string nm = entity_name(ab.source);
        Zone::ZoneValue landed = change_zone_move(orderer, ab.source, ab.destination);
        if (ab.remember_changed) cur_game.remembered_entities.push_back(ab.source);
        if (landed == Zone::BATTLEFIELD) {
            global_coordinator.GetComponent<Zone>(ab.source).controller = owner;
            if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(ab.source);
        }
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", nm.c_str(), dest_str);
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Non-targeted player-driven battlefield multi-select (Yorion, Sky Nomad's blink: "exile any
    // number of OTHER nonland permanents you own and control"). The ability has no ValidTgts$
    // (selection is not targeting, CR 701.x) — it filters the battlefield by ChangeType$ and lets
    // the controller pick how many to move. ChangeNum$ X here is just Count$ of the matching set
    // ("any number"), so the cap is effectively "all matching": re-derive candidates after each
    // pick and offer a "done" to stop early. RememberChanged$ records the moved cards for a paired
    // delayed-return SubAbility (DelTrig → return at the next end step). Distinct from the
    // empty-filter self-move branch above and from ChangeZoneAll (which moves the whole set with no
    // choice).
    if (ab.origin == Zone::BATTLEFIELD && !ab.change_type.empty()) {
        MatchCtx ctx;
        ctx.controller = owner;
        ctx.source = ab.source;
        // Live-menu loop (Shape C) with no persisted progress at all: the
        // candidate pool is the live battlefield filtered by ChangeType$, so
        // every pass — arm, consume, and post-pick re-arm alike — re-derives it
        // from components (each exile removes its pick; the loop runs until
        // "done" or the pool empties). The consume-side rebuild maps the
        // latched answer back to an entity; ask's size assert guards drift.
        // Asks are seated on `owner` (the selecting player) through fctx.ask,
        // replacing the old manual seat swap around the whole loop.
        while (true) {
            std::vector<Entity> cands;
            for (auto e : battlefield_permanents(orderer->mEntities)) {
                if (permanent_matches_any(e, ab.change_type, ctx)) cands.push_back(e);
            }
            if (cands.empty()) break;
            std::vector<LegalAction> picks;
            for (auto e : cands) {
                // A token among the candidates (e.g. an Orc Army you control when Yorion blinks)
                // has no CardData — name it from its Permanent instead of crashing.
                LegalAction la(PASS_PRIORITY, e, std::string("Exile ") + object_display_name(e));
                la.category = ActionCategory::CHOOSE_CARD;
                la.card_is_public = true;
                picks.push_back(la);
            }
            LegalAction done(PASS_PRIORITY, "Done (exile no more)");
            done.category = ActionCategory::CHOOSE_CARD;
            picks.push_back(done);
            int choice = fctx.ask(std::move(picks), owner, ab.source);
            if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
            if (choice < 0 || choice >= static_cast<int>(cands.size())) break;  // declined / done
            Entity chosen = cands[static_cast<size_t>(choice)];
            std::string cname = object_display_name(chosen);
            change_zone_move(orderer, chosen, ab.destination);
            if (ab.remember_changed) cur_game.remembered_entities.push_back(chosen);
            game_log("%s exiles %s\n", player_name(owner).c_str(), cname.c_str());
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Search-based ChangeZone (e.g. fetch lands, Green Sun's Zenith). The number to move is
    // normally a fixed numeric ChangeNum$. A dynamic count-SVar ChangeNum (Ugin, Eye of the
    // Storms' -11: "Search ... for any number of colorless nonland cards", ChangeNum$ X, X =
    // Count of those cards you own) means "any number up to all": resolve the cap from the
    // count expression and let a fail-to-find stop the search early (CR 701.19, a "search for
    // any number" lets the player choose fewer). dynamic_amount_expr is only set when ChangeNum$
    // itself is a count-SVar, so fixed-count fetches (and Green Sun's Zenith, whose X bounds the
    // filter, not the count) are unaffected.
    bool multi_zone = ab.origins.size() > 1;
    bool reveal = search_reveals_card(ab);

    // The resolved pick count and dynamic filter bound persist in the frame rt:
    // earlier picks mutate the state they were counted from (cards moved, the
    // remembered set grown — Doomsday's ChangeNum$ 5 over a shrinking pool), so
    // a resume must NOT re-evaluate them. The loop index persists likewise so a
    // resume re-enters the suspended pick.
    ChangeZoneSearchRt local_rt;
    ChangeZoneSearchRt &rt = fctx.can_suspend() ? fctx.rt<ChangeZoneSearchRt>() : local_rt;
    if (!rt.init) {
        if (!ab.dynamic_amount_expr.empty())
            rt.num_to_move = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, 0);
        else
            rt.num_to_move = (ab.amount > 0) ? ab.amount : 1;
        // Dynamic mana-value bound on the search filter (Aether Vial: "Creature.cmcEQX",
        // X = charge counters on this Aether Vial). Resolve against the ability's source so
        // the hand search only offers creatures of the matching mana value (CR 122.1).
        rt.cmc_bound = -1;
        if (!ab.change_type_cmc_expr.empty())
            rt.cmc_bound = evaluate_sa_svar(ab.change_type_cmc_expr, owner, ab.source);
        rt.prev_priority = cur_game.player_a_has_priority;
        rt.init = true;
    }

    // Seat the search/pick prompt on the player who actually makes the choice. By default that
    // is the searched zone's owner — normally also the ability's controller, but a DefinedPlayer$
    // TargetedController search (White Orchid Phantom / Erode: the destroyed land's controller
    // may search THEIR library) belongs to that player, not the caster whose trigger is resolving
    // (the resolve seat is the controller's, set in stack_manager). The input/BQUERY seat follows
    // cur_game.player_a_has_priority, so set it here and restore rt.prev_priority after the loop
    // (the set is idempotent on resume: a suspension persisted priority at this same seat).
    //
    // Chooser$ You overrides: the ability's CONTROLLER makes the selection from `owner`'s zone
    // (Thought-Knot Seer: you pick a nonland card from the targeted opponent's revealed hand to
    // exile). The searched cards are public knowledge there (the hand was revealed by the parent
    // RevealHand), so the picks carry card_is_public — flag reveal so the chosen card's identity
    // is shown even into a hidden destination and recorded in the belief state.
    if (ab.chooser_is_controller) {
        cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
        reveal = true;
    } else {
        cur_game.player_a_has_priority = (owner == Zone::PLAYER_A);
    }

    // The chain's targeted CARD, for a ChangeType `targetedBy` filter alternative (Cloak and
    // Dagger, Entwined: the DBPump-chosen creature). A player-entity target names the searched
    // player above, not a card, so it is excluded here.
    Entity chain_target =
        (ab.target != 0 && !global_coordinator.entity_has_component<Player>(ab.target)) ? ab.target
                                                                                        : 0;

    for (; rt.iter < rt.num_to_move; rt.iter++) {
        bool suspended = false;
        Entity chosen = 0;
        if (multi_zone) {
            chosen = search_multi_zone(orderer, owner, ab.origins, ab.change_type, ab.mandatory, ab.destination,
                reveal, fctx, ab.source, suspended, chain_target);
        } else {
            chosen = search_zone(orderer, owner, ab.origin, ab.change_type, ab.mandatory, ab.destination,
                reveal, rt.cmc_bound, ab.change_type_cmc_op, fctx, ab.source, suspended, chain_target);
        }
        // Suspended: the seat stays persisted at the chooser (the parked query
        // holds it); rt.prev_priority is restored by the completion epilogue.
        if (suspended) return HandlerResult::SUSPENDED;

        // after we have chosen but before we place it where it goes, if we messed with library shuffle it
        if (ab.origin == Zone::LIBRARY) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library\n", player_name(owner).c_str());
        }

        if (chosen != 0) {
            auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
            auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
            // Pre-move origin, for a Duration$ UntilHostLeavesPlay exile's linked return.
            Zone::ZoneValue chosen_origin = chosen_zone.location;
            // ExileFaceDown$ True (CR 708): thread the face-down intent into the move so add_to_zone
            // both stamps Zone::is_face_down and withholds the card from the owner's public revealed
            // multi-hot (a face-down exile is not public knowledge, CR 708.2).
            Zone::ZoneValue landed = change_zone_move(orderer, chosen, ab.destination,
                /*exile_face_down=*/ab.exile_face_down && ab.destination == Zone::EXILE);
            if (landed == Zone::BATTLEFIELD) {
                chosen_zone.controller = owner;
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(chosen);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(chosen);
            }
            // Duration$ UntilHostLeavesPlay on a search-based exile (Cloak and Dagger,
            // Entwined): register the linked return, like the targeted branch above.
            if (ab.duration_until_host_leaves && landed == Zone::EXILE)
                register_exile_until_host_leaves(ab.source, chosen, chosen_origin);
            // Record the exiled card against the source's Permanent (the "cards exiled with this"
            // association a Saga reads for its later chapters — The Creation of Avacyn's Defined$
            // ExiledWith), mirroring the targeted-exile branch above.
            if (landed == Zone::EXILE && ab.source != 0 &&
                global_coordinator.entity_has_component<Permanent>(ab.source))
                global_coordinator.GetComponent<Permanent>(ab.source).exiled_with.push_back(chosen);
            // The move above (via change_zone_move → add_to_zone's exile_face_down param) already
            // stamped Zone::is_face_down for a face-down exile; this just mirrors the condition for
            // the private-vs-public logging below.
            bool face_down = (ab.exile_face_down && landed == Zone::EXILE);
            if (ab.remember_changed) {
                cur_game.remembered_entities.push_back(chosen);
            }
            bool dest_public =
                (ab.destination == Zone::BATTLEFIELD || ab.destination == Zone::GRAVEYARD || ab.destination == Zone::EXILE);
            if (landed != ab.destination) {
                // A replacement effect diverted the move (Containment Priest → exile,
                // Grafdigger's Cage → prevented) and already logged its reason; emit nothing.
            } else if (face_down) {
                // A face-down exile (CR 708.4): the searcher knows the card; the opponent sees only
                // that a card was exiled face down. Report it privately, not as public knowledge.
                game_log_private(owner, "%s exiles %s face down\n", player_name(owner).c_str(),
                                 chosen_cd.name.c_str());
                game_log_redacted(owner, "%s exiles a card face down\n", player_name(owner).c_str());
            } else if (dest_public) {
                game_log("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
            } else if (reveal) {
                // Hidden destination, but the card was revealed — it's public knowledge.
                // The orderer reveal hook only fires for public zones, so mark it here.
                mark_card_revealed(chosen, owner);
                game_log("%s reveals %s and puts it to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(),
                    dest_str);
            } else {
                game_log_private(
                    owner, "%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
                game_log_redacted(owner, "%s puts a card to %s\n", player_name(owner).c_str(), dest_str);
            }
        } else {
            game_log("%s fails to find\n", player_name(owner).c_str());
            break;
        }
    }
    cur_game.player_a_has_priority = rt.prev_priority;
    return HandlerResult::DONE_RUN_SUBS;
}

// Owns the zone-movement param keys shared by the ChangeZone family (ChangeZone
// and ChangeZoneAll both consume origin/destination/etc. at resolve).
bool parse_change_zone(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "ChangeType") {
        ab.change_type = value;
        return true;
    } else if (key == "RememberChanged") {
        ab.remember_changed = (value == "True");
        return true;
    } else if (key == "Origin") {
        // Handle comma-separated origins (e.g. "Graveyard,Library")
        auto parse_zone = [](const std::string &s) -> Zone::ZoneValue {
            if (s == "Library")        return Zone::LIBRARY;
            if (s == "Hand")           return Zone::HAND;
            if (s == "Graveyard")      return Zone::GRAVEYARD;
            if (s == "Exile")          return Zone::EXILE;
            if (s == "Sideboard")      return Zone::SIDEBOARD;
            if (s == "Stack")          return Zone::STACK;
            if (s == "Battlefield")    return Zone::BATTLEFIELD;
            return Zone::LIBRARY;
        };
        ab.origins.clear();
        size_t zp = 0;
        while (true) {
            size_t comma = value.find(',', zp);
            if (comma == std::string::npos) {
                ab.origins.push_back(parse_zone(value.substr(zp)));
                break;
            }
            ab.origins.push_back(parse_zone(value.substr(zp, comma - zp)));
            zp = comma + 1;
        }
        ab.origin = ab.origins[0];  // backward compat
        return true;
    } else if (key == "Destination") {
        if (value == "Battlefield")    ab.destination = Zone::BATTLEFIELD;
        else if (value == "Library")   ab.destination = Zone::LIBRARY;
        else if (value == "Hand")      ab.destination = Zone::HAND;
        else if (value == "Graveyard") ab.destination = Zone::GRAVEYARD;
        else if (value == "Exile")     ab.destination = Zone::EXILE;
        return true;
    } else if (key == "MayShuffle") {
        ab.may_shuffle = (value == "True");
        return true;
    } else if (key == "Tapped") {
        ab.enters_tapped = (value == "True");
        return true;
    } else if (key == "Transformed") {
        ab.enters_transformed = (value == "True");
        return true;
    } else if (key == "ExileFaceDown") {
        // ExileFaceDown$ True — the card moved to exile is exiled face down (CR 708). The
        // face-down flag is stamped on its Zone after the move (The Creation of Avacyn I).
        ab.exile_face_down = (value == "True");
        return true;
    }
    return false;
}

// Shared handler for ChangeType$ Remembered.sameName / Targeted.sameName (Surgical
// Extraction, Extirpate, Infernal Tutor, Secret Salvage, Pack Hunt, ...). Builds the set
// of cards sharing a name with the referenced card across the ability's Origin zone(s),
// belonging to the caster (default) or the targeted card's owner (DefinedPlayer$ /
// Defined$ TargetedController), and moves up to the allowed number to the Destination.
//
// Quantity follows the agreed simplification: an "up to N" / count-SVar / ChangeZoneAll
// quantity moves the maximum available (force_all, an empty amount_svar with a numeric
// cap moves min(cap, found)). Searching an opponent's hand/library reveals those zones,
// recorded into the match-scoped belief state (mark_card_revealed) — the only
// opponent-card-identity channel in the observation.
bool change_zone_same_name(Ability &ab, std::shared_ptr<Orderer> orderer, bool force_all) {
    // The reference card whose name we match: the target (Targeted.sameName) or the
    // remembered object (Remembered.sameName, set by RememberTargets/RememberObjects).
    Entity ref = 0;
    if (ab.change_type.find("Targeted") != std::string::npos)
        ref = !ab.targets.empty() ? ab.targets[0] : ab.target;
    else
        ref = !cur_game.remembered_entities.empty() ? cur_game.remembered_entities[0]
              : (!ab.targets.empty() ? ab.targets[0] : ab.target);
    if (ref == 0 || !global_coordinator.entity_has_component<CardData>(ref)) return true;
    std::string name = global_coordinator.GetComponent<CardData>(ref).name;

    // Whose zones to search: the caster by default, or the owner of the referenced card
    // (DefinedPlayer$/Defined$ TargetedController). Derive from `ref` (the remembered/
    // targeted card) rather than ab.target, which a Pump vehicle may have overwritten.
    Zone::Ownership caster = global_coordinator.GetComponent<Zone>(ab.source).owner;
    Zone::Ownership searched = caster;
    if (ab.defined_targeted_controller && global_coordinator.entity_has_component<Zone>(ref))
        searched = global_coordinator.GetComponent<Zone>(ref).owner;

    std::vector<Zone::ZoneValue> zones =
        ab.origins.size() > 1 ? ab.origins : std::vector<Zone::ZoneValue>{ ab.origin };

    bool searched_library = false;
    std::vector<Entity> matches;
    auto collect = [&](const std::vector<Entity> &cards) {
        for (auto e : cards) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (global_coordinator.GetComponent<CardData>(e).name == name) matches.push_back(e);
        }
    };
    for (auto z : zones) {
        if (z == Zone::LIBRARY)        { searched_library = true; collect(orderer->get_library_contents(searched)); }
        else if (z == Zone::HAND)      collect(orderer->get_hand(searched));
        else if (z == Zone::GRAVEYARD) collect(orderer->get_graveyard(searched));
    }

    // Cap: ChangeZoneAll / count-SVar / unset → all matches; numeric ChangeNum → min(N, found).
    size_t cap = matches.size();
    if (!force_all && ab.amount_svar.empty() && ab.amount > 0)
        cap = std::min(static_cast<size_t>(ab.amount), matches.size());

    const char *dest_str = ab.destination == Zone::EXILE       ? "exile"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                           : ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                                                                 : "library";
    for (size_t i = 0; i < cap; i++) {
        Zone::ZoneValue landed = change_zone_move(orderer, matches[i], ab.destination);
        if (landed == Zone::BATTLEFIELD)
            global_coordinator.GetComponent<Zone>(matches[i]).controller = caster;
        mark_card_revealed(matches[i], searched);  // moved card is now public / revealed
    }
    game_log("%s moves %zu card(s) named %s from %s's %s to %s\n", player_name(caster).c_str(),
        cap, name.c_str(), player_name(searched).c_str(),
        zones.size() > 1 ? "zones" : "zone", dest_str);

    // Searching an opponent's hidden zones reveals their full contents to the searcher;
    // record them into the belief state.
    if (searched != caster) {
        for (auto z : zones) {
            if (z == Zone::HAND)    for (auto e : orderer->get_hand(searched))             mark_card_revealed(e, searched);
            if (z == Zone::LIBRARY) for (auto e : orderer->get_library_contents(searched)) mark_card_revealed(e, searched);
        }
    }

    if (searched_library) {
        orderer->shuffle_library(searched);
        game_log("%s shuffles their library\n", player_name(searched).c_str());
    }
    return true;
}

}  // namespace effects
