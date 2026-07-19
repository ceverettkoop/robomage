#ifndef GAME_H
#define GAME_H

#define ACTION_HISTORY_SIZE 128
#define KNOWN_TOP_LIBRARY_SIZE 5

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Step {
    UNTAP,
    UPKEEP,
    DRAW,
    FIRST_MAIN,
    BEGIN_COMBAT,
    DECLARE_ATTACKERS,
    DECLARE_BLOCKERS,
    FIRST_STRIKE_DAMAGE,
    COMBAT_DAMAGE,
    END_OF_COMBAT,
    SECOND_MAIN,
    END_STEP,
    CLEANUP
}Step;

#ifdef __cplusplus
}  // end extern "C"
#endif

#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include <string>

#include "../ecs/entity.h"
#include "../components/ability.h"
#include "../mana_system.h"
#include "../pending_query.h"
#include "../resolution_frame.h"
#include "colors.h"

struct Deck;
struct Game;
struct DelayedTrigger;

extern Game cur_game;

struct DelayedTrigger {
    Ability ability;        // what to push onto the stack when it fires
    uint32_t fire_on;       // event ID (e.g. Events::UPKEEP_BEGAN)
    Entity owner_entity;    // player entity who controls it
    size_t fire_on_turn;    // game.turn value at which to fire (cur_game.turn + 1 at registration)
    // Phase restriction (script ValidPlayer$): the player whose phase this trigger may fire on,
    // or 0 = any player's phase. "At the beginning of the next turn's upkeep" / "the next end
    // step" fires at the NEXT occurrence of that phase whoever's turn it is (Mishra's Bauble,
    // Flickerwisp), so most delayed triggers leave this 0; only an explicit "your upkeep"-style
    // ValidPlayer$ You restricts it. Independent of owner_entity, which is always the ability's
    // controller (the Bauble draw is the controller's even on the opponent's upkeep).
    Entity restrict_player = 0;
    // "When THIS specific permanent leaves the battlefield" delayed trigger (CR 603.6e), set up
    // by the earthbend resolution: fire_on is CARD_CHANGED_ZONE, and the trigger fires only when
    // `watch_entity` is the card that left the battlefield (origin BATTLEFIELD). General over any
    // "when X leaves, do Y" delayed trigger; 0 = not entity-watched (the phase-based default).
    Entity watch_entity = 0;          // the specific permanent whose departure fires this trigger
    bool fire_on_leave_battlefield = false;  // true: match watch_entity leaving the battlefield, not a phase
    // Destination filter for fire_on_leave_battlefield triggers: when non-empty, the trigger
    // fires only if the watched entity moved from the battlefield TO one of these zones (e.g.
    // earthbend's "when it dies or is exiled" = {GRAVEYARD, EXILE} — a bounce to hand or a
    // shuffle into the library must NOT fire it). Empty = any departure ("leaves the
    // battlefield", e.g. the exile-until-host-leaves triggers).
    std::vector<Zone::ZoneValue> fire_dest_zones;
    // RememberObjects$ RememberedLKI (CR 603.7a): objects this delayed trigger captured when it
    // was set up (the cards the preceding RememberChanged$ ChangeZone moved, e.g. the permanent
    // Flickerwisp/Phelia exiled). Restored into cur_game.remembered_entities before the fire
    // ability resolves so its Defined$ DelayTriggerRememberedLKI acts on those same objects.
    std::vector<Entity> remembered_objects;
};

// Last-known information (CR 608.2h / 112.7a): a permanent's effective characteristics —
// after all continuous effects and counters — captured the instant it leaves the battlefield.
// An effect that references the object after it has changed zones (e.g. Swords to Plowshares'
// "its controller gains life equal to its power", read from a creature it just exiled) uses
// these last-known values rather than the printed base. While the object is still in its
// expected zone, effective characteristics are read live from its components instead.
struct LastKnownInfo {
    int power = 0;
    int toughness = 0;
    std::vector<std::string> type_names;   // type/subtype/supertype names
    std::set<Colors> colors;               // effective colors
    Zone::Ownership controller = Zone::UNKNOWN;  // last controller (CR 608.2g): "that permanent's controller"
    std::vector<Entity> exiled_with;       // Permanent::exiled_with snapshot — the cards this permanent
                                           // had exiled, so a leaves-the-battlefield ability (Skyclave
                                           // Apparition's TrigToken) can still find them after the
                                           // Permanent component is stripped by the SBA pass.
    // How-it-entered markers (Permanent flags), snapshotted so an ETB trigger of a permanent that
    // entered and then LEFT again before trigger collection (legend-rule keep-other, 0-toughness
    // SBA death) can still be gated correctly by the 603.10 look-back scan: the trigger fired when
    // the permanent entered (CR 603.3a), even though its Permanent component is gone by now.
    bool entered_by_cast = false;          // "if you cast it" gate (The One Ring)
    bool evoked = false;                   // evoke self-sacrifice gate
    bool entered_with_offspring = false;   // offspring token-copy gate
    bool transformed = false;              // which DFC face was active (CR 712.4 ability selection)
    bool abilities_removed = false;        // Permanent::abilities_removed snapshot: a permanent that
                                           // had its abilities removed (Humility, layer 6 / CR 613.1f)
                                           // has no triggered abilities, so its own leaves/dies
                                           // look-back trigger (CR 603.10) must not fire either.
    bool cast_from_hand_by_controller = false;  // "if you cast it from your hand" gate (Amped
                                                // Raptor's Card.wasCastFromYourHandByYou), read via
                                                // LKI when the source left play before the trigger
                                                // resolved (CR 603.10 / 608.2h).
};

enum MandatoryChoice {
    NONE,
    DECLARE_ATTACKERS_CHOICE,
    DECLARE_BLOCKERS_CHOICE,
    CLEANUP_DISCARD,
    CHOOSE_ENTITY,  // Legend rule, replacement effect, choose card name, choose permanent
    ASSIGN_COMBAT_DAMAGE_CHOICE  // T3.10: attacker divides damage among 2+ blockers it can't all kill
};

struct ActionHistoryEntry {
    int category;        // ActionCategory value
    int card_vocab_idx;  // -1 for non-card entities
    bool player_a;       // true if Player A took this action
    int turn;            // cur_game.turn when the action was taken
};

// An emblem (CR 114): a continuous-effect source owned by a player that exists outside any zone
// and can't be removed. Created by an AB$ Effect with StaticAbilities$ + Duration$ Permanent
// (Kaito's [+1] "Ninjas you control get +1/+1."). Its statics are gathered into g_active_statics
// every SBA pass with the owner as their controller, so they apply through the normal layer
// engine without the emblem being a real (targetable, counted, destructible) permanent. Persists
// for the rest of the game; a fresh Game (new game of a match) starts with none.
struct Emblem {
    Zone::Ownership controller = Zone::PLAYER_A;
    std::vector<StaticAbility> statics;
};

struct Game {
        Game() {};
        Game(size_t _seed) {
            seed = _seed;
            gen = std::mt19937(seed);
        };
        // Day/Night designation the game itself can have (CR 731.1). Starts at "neither" and, once
        // set, is always exactly one of day/night. Driven by the daybound/nightbound subsystem
        // (src/day_night.*); read by day-/night-conditional effects. A fresh Game starts neither.
        enum DayNight { DN_NEITHER, DN_DAY, DN_NIGHT };
        DayNight day_night = DN_NEITHER;
        // Spells cast by the previous turn's active player DURING that turn (CR 502.2 / 731.2),
        // read by the untap-step day/night turn-based check on the following turn. Snapshotted at
        // cleanup from Player::spells_cast_this_turn before that per-turn counter is reset. Because
        // cleanup resets BOTH players' spells_cast_this_turn to 0 (so a player's instants cast on
        // the opponent's turn never leak into their own-turn count), the snapshot is just the active
        // player's spells_cast_this_turn at cleanup.
        int prev_turn_active_spell_count = 0;
        size_t seed;
        size_t timestamp = 0;
        size_t turn = 0;
        Step cur_step = UNTAP;
        Entity player_a_entity;
        Entity player_b_entity;
        std::mt19937 gen;
        bool ended = false;
        int winner = 0;  // 0=none, 1=PLAYER_A, 2=PLAYER_B (Zone::Ownership values)
        bool player_a_active = true;
        bool player_a_turn = true;
        bool player_a_has_priority = true;
        bool a_has_passed = false;
        bool b_has_passed = false;
        MandatoryChoice pending_choice = NONE;
        bool attackers_declared = false;
        bool blockers_declared = false;
        bool combat_damage_dealt = false;
        bool has_first_strikers = false;
        // T3.10: attacker -> (blocker -> damage assigned). Populated by assign_combat_damage()
        // only for attackers that required a controller choice this strike step; deal_combat_damage()
        // reads it and auto-assigns any attacker absent from the map. Cleared at handler entry
        // (per strike step) and in the END_OF_COMBAT cleanup.
        std::map<Entity, std::map<Entity, uint32_t>> combat_damage_assignment;
        // Pending extra turns (CR 500.7 / 720). Each entry is the player who will take an extra
        // turn, treated as a LIFO stack: the most recently added extra turn is taken first (CR
        // 500.7, "The most recently created turn will be taken first"). Consulted at turn hand-off
        // (advance_step's cleanup → next turn) before flipping the active player; a non-empty stack
        // makes the player on top take the next turn instead of passing to the opponent. General
        // over any "take an extra turn" effect (effects::add_turn pushes onto it). Persists across
        // turns; empty in a fresh Game.
        std::vector<Zone::Ownership> extra_turns;
        std::vector<DelayedTrigger> delayed_triggers;
        // Floating triggered abilities (CR 603.7e-style "this turn" triggers) created by a
        // transient DB$ Effect | Triggers$ <SVar> (e.g. Forth Eorlingas!'s become-monarch-on-
        // combat-damage). Each is a fully-parsed TRIGGERED Ability with its controller bound;
        // the trigger scan (check_triggered_abilities) tests them against drained events just like
        // a permanent's triggered ability. Cleared at the cleanup step so they last only their
        // turn of creation. General over any until-end-of-turn floating triggered ability.
        std::vector<Ability> floating_triggers;
        // Emblems the players have (CR 114). Each carries permanent continuous statics gathered
        // into g_active_statics every SBA pass (see gather_active_statics). Persists for the game.
        std::vector<Emblem> emblems;
        // The monarch (CR 725). MAX_ENTITIES = no monarch (none until an effect makes a player
        // the monarch). Serialized into the state vector's global-extras block (per-player
        // is_monarch flags; see machine_io.h).
        Entity monarch_entity = MAX_ENTITIES;
        std::vector<Entity> delve_exiled;   // entities exiled during current delve cast; cleared after ETB
        size_t x_paid = 0;                  // X value chosen at cast time for X-cost spells
        std::map<Entity, LastKnownInfo> last_known_info;  // effective characteristics captured as a
                                            // permanent leaves the battlefield (CR 608.2h); read by the
                                            // effective_* accessors when the object is no longer in play
        std::vector<Entity> remembered_entities;  // Defined$ Remembered — used by Attach sub-ability, Doomsday remember-changed
        std::map<Entity, int> ability_resolution_counts;  // Count$ResolvedThisTurn: incremented per triggered-ability resolve
        std::map<Entity, int> payment_fail_counts;  // machine mode: block casting after 2 failed payments
        bool pending_cant_be_countered = false;  // set during mana payment when Cavern restricted mana used
        bool pending_gift_promised = false;  // Gift (CR 702.176): the spell currently being cast promised its gift; read by Count$PromisedGift while its targets are chosen
        // The source entity of the spell/ability currently making a mid-resolution choice
        // (target select, dig/scry/surveil pick, search, discard, modal, ...). Serialized into
        // the state vector's pending-decision context block so the ML observation shows WHAT is
        // asking for the current choice — the source may not be on the stack yet, since targets
        // are announced before the spell moves there (CR 601.2b/c). Managed exclusively via
        // PendingDecisionScope; 0 = no ability-driven choice pending.
        Entity pending_decision_source = 0;
        // Suspension framework (see pending_query.h / resolution_frame.h): a
        // mid-flow decision parked for the main loop to emit, and the persisted
        // resolve() continuation it belongs to. Value members so a cur_game
        // copy (snapshot_save) covers the whole suspended state for free.
        PendingQuery pending_query;
        ResolutionFrame resolution;
        // Combat sub-prompt suspension state (pending_query tags ATTACK_TARGET /
        // BLOCK_TARGET): the creature whose target sub-prompt is currently parked.
        // Set when declare_attackers/declare_blockers suspends on the target menu,
        // consumed and cleared by the loop-top resume. Value members so a snapshot
        // covers the parked selection. 0 = no sub-prompt parked.
        Entity pending_attacker = 0;
        Entity pending_blocker = 0;
        // Combat damage-assignment suspension state (pending_query tag
        // DAMAGE_ASSIGN): the attacker whose lethal-order division is mid-prompt,
        // plus the former inner-loop locals of assign_combat_damage (remaining
        // power to assign, blockers not yet assigned lethal, last blocker picked
        // — the 510.1a leftover-dump target). Value member so a snapshot covers
        // the in-flight division. active == true iff a DAMAGE_ASSIGN query is
        // parked; completed attackers are tracked by their (possibly partial)
        // combat_damage_assignment map entries, the outer scan's re-entrancy guard.
        struct PendingDamageAssign {
            bool active = false;
            Entity attacker = 0;
            uint32_t remaining = 0;
            std::vector<Entity> pool;  // blockers not yet assigned lethal
            Entity last_assigned = 0;
        };
        PendingDamageAssign pending_damage;
        // Trigger-placement suspension state (pending_query tag TRIGGER_PLACE):
        // the APNAP-flattened queue of collected triggers still to be put on
        // the stack, plus the front trigger's in-flight target selection. Set
        // by place_triggers_apnap, driven by resume_trigger_placement, cleared
        // at placement completion (which also restores saved_priority and
        // clears lk_battlefield_types). Value member so a snapshot covers the
        // parked placement. See resolution_frame.h.
        TriggerPlacementRT trigger_placement;
        // Cast-time suspension state (pending_query tag CAST): the persisted
        // state machine of process_action's CAST_SPELL branch (run_cast_flow,
        // action_processor.cpp). The branch's former locals — the accumulating
        // cost, per-kicker flags, deferred payment pieces, the half-built
        // primary Ability — live here BY VALUE so a cur_game copy covers the
        // whole in-flight cast. Batch 9 converts the LINEAR prompts (kicker /
        // replicate / X ladder / phyrexian pips / life-X / spell-sac / gift);
        // the announce and deferred-payment steps still pass through blocking
        // (Batches 10-11). active == true from the CAST_SPELL action until the
        // spell reaches the stack (FINISH) or the payment cancel rewinds.
        struct PendingCast {
            // Where the flow resumes. Steps run in today's exact statement
            // order; unconverted steps pass through synchronously. Values not
            // yet driven by run_cast_flow are reserved for Batches 10-11
            // (announce targeting, delve/deferred payment, replicate copies).
            enum Step {
                COST,             // cost-branch dispatch (flashback/escape/impulse/alt vs regular)
                ALT_PITCH,        // (Batch 11) alt-cost exile-from-hand picks
                ALT_RETURN,       // (Batch 11) alt-cost return-to-hand picks
                KICKER,           // per-kicker optional-additional-cost y/n
                REPLICATE,        // repeated replicate-cost y/n loop
                CHOOSE_X,         // X-cost ladder
                HYBRID,           // hybrid pips (machine auto-resolve; interactive stays blocking)
                PHYREXIAN_PIP,    // per-pip colored-mana-or-2-life
                LIFE_X,           // variable life X (Toxic Deluge's PayLife<X>)
                SPELL_SAC,        // spell's own additional sacrifice cost (Natural Order)
                GIFT,             // gift promise y/n (incl. the forced-promise no-prompt path)
                CHARM_MODE,       // (Batch 10) modal mode announcement
                CHARM_TARGET,     // (Batch 10) per-mode target announcement
                PRIMARY_TARGET,   // (Batch 10) primary spell target
                SUB_TARGET,       // (Batch 10) targeting sub-ability targets
                AURA_TARGET,      // (Batch 10) aura enchant target
                ANNOUNCE,         // modes + targets + aura (blocking pass-through this batch)
                DELVE_COUNT,      // (Batch 11) delve exile count
                DELVE_PICK,       // (Batch 11) delve exile picks
                MANA_PAY,         // deferred payment + non-mana pieces (blocking pass-through)
                DEF_SAC,          // (Batch 11) deferred flashback sacrifice
                DEF_EXILE_TYPES,  // (Batch 11) escape exile-by-types picks
                DEF_EXILE_COUNT,  // (Batch 11) escape exile-by-count picks
                COPY_TARGETS,     // (Batch 10) replicate/storm copy retargeting
                FINISH            // spell to stack + cast events + copies + take_action
            };
            bool active = false;
            Step step = COST;
            // Cast identity: reconstructs the consumed LegalAction across the
            // suspension gap.
            Entity spell_entity = 0;
            bool caster_is_a = true;
            bool use_flashback = false;
            bool use_escape = false;
            bool use_alt_cost = false;
            bool use_offspring = false;
            bool impulse_cast = false;
            bool cast_back_face = false;
            // Snapshot of mana state for rewind on payment failure (taken in
            // process_action before the flow starts, consumed by the MANA_PAY
            // cancel path).
            ManaPaymentSnapshot mana_snap;
            // The regular-cost accumulation (base + offspring/kicker/replicate/
            // X/hybrid/phyrexian folds), deferred into deferred_mana_cost once
            // the cost is fully resolved.
            ManaValue cost_to_pay;
            // Kicker (CR 702.33): per-kicker "paid?" flags, populated in the regular-cost
            // branch and copied onto the Spell so linked "if it was kicked with its [N]
            // kicker" triggers can read them. Empty unless the card has K:Kicker.
            std::vector<bool> kicked_flags;
            size_t kicker_idx = 0;     // next kicker pip to offer/apply
            // Replicate (CR 702.x): how many times the replicate additional cost was paid in
            // the regular-cost branch. Drives the on-cast copy effect. 0 unless the card
            // has K:Replicate and the caster chose to pay it.
            int replicate_count = 0;
            size_t phyrexian_idx = 0;  // next phyrexian pip to offer/apply
            bool gift_promised = false;
            // CR 601.2 orders target choice (601.2c) BEFORE paying costs (601.2h). The regular
            // mana payment is therefore captured here and deferred until after targets are chosen.
            // This matters when the mana is paid by sacrificing a permanent for mana
            // (Lotus Petal / Lion's Eye Diamond): paying first could remove the spell's only legal
            // target before it is chosen, crashing on an empty target menu. With the payment
            // deferred, the target is chosen while the would-be mana source is still on the
            // battlefield; sacrificing it for mana then merely makes the target illegal, so the
            // spell fizzles at resolution (CR 608.2b) instead of being offered with no legal target.
            ManaValue deferred_mana_cost;
            bool deferred_mana_pending = false;
            bool deferred_delve = false;
            bool deferred_improvise = false;
            // Non-mana alternative-cost pieces (flashback life/sacrifice, escape
            // exile-from-graveyard) are deferred the same way: targets first (601.2c),
            // then every cost (601.2g/h). They are paid only after the deferred mana
            // payment commits, so a cancelled payment never costs life or a creature.
            int deferred_life_cost = 0;
            std::string deferred_sac_spec;
            int deferred_exile_min_types = 0;
            int deferred_exile_count = 0;
            // The half-built primary spell ability. The ENTITY's Ability
            // component is added at the same point as today (right after
            // announce_spell_targets), so component state at every prompt
            // matches the blocking flow's exactly.
            Ability ability;
            bool have_ability = false;
        };
        PendingCast pending_cast;

        // Turn-long "spells you control can't be countered" grant created by a resolving spell/
        // ability (Veil of Summer's DB$ Effect | ReplacementEffects$ AntiMagic, CR 614.13/
        // CantHappen). A player here means every spell that player controls can't be countered
        // this turn — unlike Hexing Squelcher's battlefield static, this form belongs to no
        // permanent (the instant is in the graveyard), so it is recorded here and cleared at
        // cleanup. Consulted at counter-resolution time (effects::counter).
        std::set<Zone::Ownership> cant_counter_spells_of;
        // Turn-long "hexproof from <color(s)>" grant for a player and the permanents they control
        // (Veil of Summer: "You and permanents you control gain hexproof from blue and from black
        // until end of turn", CR 702.11e). Each entry protects `player` (and any permanent they
        // control) from being targeted by spells/abilities an opponent controls whose source is
        // one of `colors`. Player-scoped (rather than a per-permanent keyword grant) so it can
        // also protect the player object, and so every permanent the player controls is covered;
        // cleared at cleanup. Consulted in Ability::is_legal_target.
        struct HexproofFromColors {
            Zone::Ownership player = Zone::UNKNOWN;
            std::set<Colors> colors;
        };
        std::vector<HexproofFromColors> hexproof_from_colors_this_turn;
        // Player-scoped "protection from everything" grant (CR 702.16; The One Ring's ETB: "you
        // gain protection from everything until your next turn"). While active the protected
        // `player` can't be the target of a spell/ability an opponent controls, and isn't dealt
        // damage by any source an opponent controls. `until_your_next_turn` selects the duration:
        // when true the grant is reverted at the start of the protected player's next turn (their
        // untap step); when false it lapses at cleanup (end of turn). Consulted in
        // Ability::is_legal_target and deal_damage_to_player.
        struct PlayerProtectionFromEverything {
            Zone::Ownership player = Zone::UNKNOWN;
            bool until_your_next_turn = false;
        };
        std::vector<PlayerProtectionFromEverything> player_protection_from_everything;
        bool revolt_player_a = false;  // a permanent Player A controlled left the battlefield this turn
        bool revolt_player_b = false;  // a permanent Player B controlled left the battlefield this turn
        std::set<Entity> void_countered;  // entities exiled with void counters (Dauthi Voidwalker)
        std::set<Entity> may_cast_this_turn;  // cards a permission effect (Emry's AB$ Effect) lets their owner cast from the graveyard this turn (CR 601.3e); cleared each cleanup
        std::set<Entity> chosen_cards;  // permanents chosen/kept by a ChooseCard effect (Ajani -4); read by SacrificeAll's nonChosenCard filter, cleared by Cleanup ClearChosenCard$
        std::string named_card = "";  // card name chosen by a resolving SP$/DB$ NameCard effect (CR 201.4, Cabal Therapy); read by a chained Card.NamedCard discard, cleared after the spell finishes resolving
        int chosen_number = 0;  // integer chosen by a resolving DB$ ChooseNumber effect (Wrath of the Skies: "pay any amount of {E}"); read downstream via Count$ChosenNumber (e.g. the cmc bound and PayEnergy unless-cost of the chained DestroyAll)
        std::map<Entity, std::vector<std::string>> lk_battlefield_types;  // last-known type/subtype names of a permanent captured as it leaves the battlefield (603.10 look-back), so a "dies"/leaves trigger can match a token that has already ceased to exist by the time triggers are checked
        std::set<Entity> pending_enters_tapped;  // one-shot: a ChangeZone effect put this card onto the battlefield tapped; consumed when its Permanent is created
        std::map<Entity, Entity> pending_enters_attacking;  // one-shot: {ninja -> attack target} a K:Ninjutsu (CR 702.49e) put this card onto the battlefield attacking; consumed when its Creature component is created (a non-creature ninja, e.g. a planeswalker, drops the mark — it can't be a combatant)
        std::set<Entity> pending_enters_transformed;  // one-shot: a ChangeZone effect (Transformed$ True) put this card onto the battlefield showing its DFC back face; consumed when its Permanent is created
        std::set<Entity> pending_evoked;  // one-shot: a spell cast for its evoke cost is resolving; consumed when its Permanent is created (sets Permanent::evoked)
        std::set<Entity> pending_offspring;  // one-shot: a spell cast with its Offspring additional cost is resolving; consumed when its Permanent is created (sets Permanent::entered_with_offspring)
        std::set<Entity> pending_escaped;  // one-shot: a spell cast from the graveyard for its Escape cost is resolving; consumed when its Permanent is created (sets Permanent::cast_with_escape) — Uro's "sacrifice it unless it escaped"
        std::set<Entity> pending_unearthed;  // one-shot: an Unearth ChangeZone (CR 702.84) returned this card to the battlefield; consumed when its Permanent is created (sets Permanent::unearthed → haste + delayed end-step exile + leaves→exile replacement)
        std::set<Entity> pending_impending;  // one-shot: a spell cast for its Impending alternate cost (CR 702.175) is resolving; consumed when its Permanent is created (puts impending_count TIME counters on it → not a creature until they shed)
        std::set<Entity> cast_to_battlefield;  // one-shot: a cast spell is resolving from the stack onto the battlefield (it "was cast", CR 614.12 / Containment Priest); consumed when its Permanent is created
        std::set<Entity> cast_from_hand;  // one-shot: a spell now resolving onto the battlefield was cast from its controller's own hand (a normal CR 601 hand cast); consumed when its Permanent is created → Permanent::cast_from_hand_by_controller (Amped Raptor's Card.wasCastFromYourHandByYou gate)
        // Impulse-cast permission (CR 707 "impulsive draw" / 118.9 alternative cost): a card a
        // resolving DB$ Play effect (Amped Raptor) lets its controller cast from EXILE this turn,
        // paying an alternative RESOURCE cost (energy or life) instead of its mana cost. Keyed by
        // the card entity; cleared each cleanup. Generalizes the alt-cost-cast over the resource
        // so the same path serves energy ({E}) and life (a future Bolas's Citadel "pay life =
        // mana value"). The casting path reads this to compute the cost and skip mana payment.
        struct ImpulseCastPermission {
            // FREE = cast without paying any cost (Ugin, Eye of the Storms' -11: "cast those
            // cards without paying their mana costs", CR 118.9 / 601.2f). ENERGY/LIFE pay an
            // alternative resource cost equal to `amount` (Amped Raptor's DB$ Play).
            enum Resource { ENERGY, LIFE, FREE } resource = ENERGY;
            int amount = 0;            // resolved cost (e.g. the card's mana value); 0 when FREE
            Zone::Ownership caster = Zone::UNKNOWN;  // who may cast it (its controller)
        };
        std::map<Entity, ImpulseCastPermission> impulse_cast_permission;
        std::map<Entity, int> pending_etb_xpaid;  // one-shot: X paid for an X-cost permanent spell now resolving, used by an "enters with X counters" replacement (Chalice of the Void); consumed when its Permanent is created
        std::map<Entity, Entity> pending_attach;  // one-shot: {creature -> equipment} a DB$ Attach resolved onto a creature whose Permanent did not exist yet (reanimate-then-attach, Pre-War Formalwear); the equip link is finalized when the creature's Permanent is created
        std::map<Entity, Entity> pending_aura_target;  // one-shot: {aura -> enchanted object} an Aura spell chose its enchant target at cast (CR 303.4); the attach link (aura.equipped_to) is finalized when the aura's Permanent is created

        // Recent action history ring buffer for ML observation
        ActionHistoryEntry action_history[ACTION_HISTORY_SIZE] = {};
        int action_history_write = 0;  // next write position (circular)
        int action_history_count = 0;  // total entries written (capped at ACTION_HISTORY_SIZE)

        // Known top-of-library cards (one array per player). Index 0 is the top of the
        // library. -1 = unknown (default). Updated when a card is placed on top of a
        // library or when a card is removed from the top; cleared to all -1 on shuffle.
        int known_top_library_a[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};
        int known_top_library_b[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};

        // CR 725: make `player` the monarch. The previous monarch (if any) ceases to be the
        // monarch (725.3). No-op if `player` is already the monarch. Sourceless inherent monarch
        // triggers (end-step draw, steal-on-combat-damage) are fired by check_triggered_abilities.
        void set_monarch(Entity player_entity);

        void record_action(int category, int card_vocab_idx, bool player_a);
        void clear_known_top_library(bool player_a_owner);
        void known_top_library_push(bool player_a_owner, int card_vocab_idx);
        void known_top_library_remove_pos(bool player_a_owner, int pos);

        bool ready_to_resolve();
        bool is_mandatory_choice_pending() const;
        void generate_players(const Deck &deck_a, const Deck &deck_b);
        bool advance_step(std::shared_ptr<class StackManager> stack_manager, std::shared_ptr<class Orderer> orderer);
        void pass_priority();
        void take_action();  // resets last_player_passed since an action was taken

    private:
        Entity gen_player(const Deck &deck);
};

// RAII marker for Game::pending_decision_source: constructed at the top of an effect/target
// handler that is about to prompt a choice on behalf of a spell/ability, so every get_input
// within the scope serializes that ability's source card into the observation's
// pending-decision context. Saves/restores the previous value, so nested choices (a
// sub-ability's target chosen during a parent's resolution) unwind correctly.
struct PendingDecisionScope {
    Entity prev_source;
    explicit PendingDecisionScope(Entity source) : prev_source(cur_game.pending_decision_source) {
        cur_game.pending_decision_source = source;
    }
    ~PendingDecisionScope() { cur_game.pending_decision_source = prev_source; }
};

// True while a suspended decision is parked for the main loop to emit (see
// pending_query.h). Later batches gate cooperative early-returns on it
// (`if (decision_suspended()) return;` in suspendable callees).
inline bool decision_suspended() { return cur_game.pending_query.active; }

#endif // __cplusplus

#endif /* GAME_H */
