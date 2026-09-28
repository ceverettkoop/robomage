#ifndef GAME_H
#define GAME_H

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
    size_t fire_on_turn;    // game.turn_state.turn value at which to fire (cur_game.turn_state.turn + 1 at registration)
    // Phase restriction (script ValidPlayer$): the player whose phase this trigger may fire on,
    // or 0 = any player's phase. "At the beginning of the next turn's upkeep" / "the next end
    // step" fires at the NEXT occurrence of that phase whoever's turn it is (Mishra's Bauble,
    // Flickerwisp), so most delayed triggers leave this 0; only an explicit "your upkeep"-style
    // ValidPlayer$ You restricts it. Independent of owner_entity, which is always the ability's
    // controller (the Bauble draw is the controller's even on the opponent's upkeep).
    Entity restrict_player = 0;
    // "When THIS specific permanent leaves the battlefield" delayed trigger (CR 603.6e), set up
    // by the earthbend resolution: fire_on is CARD_CHANGED_ZONE, and the trigger fires only when
    // `watched` is the object that left the battlefield (origin BATTLEFIELD). General over any
    // "when X leaves, do Y" delayed trigger; empty = not object-watched (the phase-based default).
    ObjectRef watched;                // the specific permanent whose departure fires this trigger
    bool fire_on_leave_battlefield = false;  // true: match `watched` leaving the battlefield, not a phase
    // Destination filter for fire_on_leave_battlefield triggers: when non-empty, the trigger
    // fires only if the watched entity moved from the battlefield TO one of these zones (e.g.
    // earthbend's "when it dies or is exiled" = {GRAVEYARD, EXILE} — a bounce to hand or a
    // shuffle into the library must NOT fire it). Empty = any departure ("leaves the
    // battlefield", e.g. the exile-until-host-leaves triggers).
    std::vector<Zone::ZoneValue> fire_dest_zones;
    // ThisTurn$ True (Searing Blood's "When that creature dies this turn"): a leave-battlefield
    // delayed trigger bounded to the turn it was registered. If the watched object has not left
    // the battlefield by that turn's cleanup, the trigger is dropped unfired (CR 603.7b). The
    // phase-based earthbend/exile-until-host-leaves triggers leave this false (they persist).
    bool expires_end_of_turn = false;
    // RememberObjects$ RememberedLKI (CR 603.7a): objects this delayed trigger captured when it
    // was set up (the cards the preceding RememberChanged$ ChangeZone moved, e.g. the permanent
    // Flickerwisp/Phelia exiled). Restored into cur_game.remembered_entities before the fire
    // ability resolves so its Defined$ DelayTriggerRememberedLKI acts on those same objects.
    std::vector<ObjectRef> remembered_objects;
};

// Last-known information (CR 608.2h / 112.7a): a permanent's effective characteristics —
// after all continuous effects and counters — captured the instant it leaves the battlefield.
// An effect that references the object after it has changed zones (e.g. Swords to Plowshares'
// "its controller gains life equal to its power", read from a creature it just exiled) uses
// these last-known values rather than the printed base. While the object is still in its
// expected zone, effective characteristics are read live from its components instead.
struct CardData;

struct LastKnownInfo {
    uint32_t issue = 0;                    // the holder of the entity id it was captured for
                                           // (Coordinator::GetIssueCount): a later holder of the id
                                           // has no last-known information from this one
    std::string name;                      // Permanent name as it left play: a token that then ceases
                                           // to exist (CR 111.7) keeps its identity for a prompt or
                                           // observation that still refers to it
    bool is_token = false;                 // it was a token (CR 111.1)
    std::string token_script;              // a token's script stem (Token::script_name), the key of its
                                           // token-band vocab index
    int power = 0;
    int toughness = 0;
    std::vector<std::string> keywords;     // effective keywords (permanent_keywords): a damage source
                                           // that left play deals damage with them (CR 702.15c)
    std::set<Type> types;                  // its type line (types, subtypes, supertypes)
    std::set<Colors> colors;               // effective colors
    Zone::Ownership controller = Zone::UNKNOWN;  // last controller (CR 608.2g): "that permanent's controller"
    std::vector<ObjectRef> exiled_with;    // Permanent::exiled_with snapshot — the cards this permanent
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
    std::map<std::string, int> counters;   // typed-counter counts snapshotted as the permanent left
                                           // play, so an effect that counts counters on a source
                                           // sacrificed as part of its own activation cost (Blast
                                           // Zone: "MV equal to the number of charge counters on it")
                                           // reads the last-known count (CR 608.2h).
    bool superseded = false;               // the card is a new object to every later reader (CR 400.7):
                                           // the resolution that moved it is over, or it moved again
                                           // (see lki_for). General reads (lki_for) no longer see this
                                           // snapshot; only look-backs at the departed object itself
                                           // (departed_lki_for) still do (CR 608.2h).
    std::shared_ptr<const CardData> copied_card;  // the copied characteristics of a permanent that
                                                  // left play as an in-place copy (Thespian's
                                                  // Stage): the card itself reverts to its printed
                                                  // CardData on leaving (CR 400.7), so the 603.10
                                                  // look-back reads the copy's abilities from here.
                                                  // Null for a permanent that was not a copy.
};

enum MandatoryChoice {
    NONE,
    DECLARE_ATTACKERS_CHOICE,
    DECLARE_BLOCKERS_CHOICE,
    CLEANUP_DISCARD,
    CHOOSE_ENTITY,  // Legend rule, replacement effect, choose card name, choose permanent
    ASSIGN_COMBAT_DAMAGE_CHOICE  // T3.10: attacker divides damage among 2+ blockers it can't all kill
};
// Width of the MandatoryChoice one-hot in the serialized state vector. Derived
// from the enum's last value, so adding a choice kind widens the one-hot (and
// shifts every later block via machine_io.h's offset chain) automatically.
// Mirrored into Python by train/gen_enums.py, which counts the enum's members.
static constexpr int N_MANDATORY_CHOICES = ASSIGN_COMBAT_DAMAGE_CHOICE + 1;

// An emblem (CR 114): a continuous-effect source owned by a player that exists outside any zone
// and can't be removed. Created by an AB$ Effect with StaticAbilities$ + Duration$ Permanent
// (Kaito's [+1] "Ninjas you control get +1/+1."). Its statics are gathered into g_active_statics
// every SBA pass with the owner as their controller, so they apply through the normal layer
// engine without the emblem being a real (targetable, counted, destructible) permanent. Persists
// for the rest of the game; a fresh Game (new game of a match) starts with none.
struct Emblem {
    Zone::Ownership controller = Zone::PLAYER_A;
    std::vector<StaticAbility> statics;
    // The card whose ability created the emblem (Kaito, Bane of Nightmares; Tamiyo, Seasoned
    // Scholar) and its vocab idx captured at creation. The emblem's identity in the observation's
    // player-effects block. Display-only (the vocab idx is what the observation reads), so a plain
    // Entity.
    Entity source = 0;
    int source_vocab_idx = -1;
};

struct Game {
        Game() {};
        Game(size_t _seed) {
            rng.seed = _seed;
            rng.engine = std::mt19937(rng.seed);
        };
        // Day/Night designation the game itself can have (CR 731.1). Starts at "neither" and, once
        // set, is always exactly one of day/night. Driven by the daybound/nightbound subsystem
        // (src/day_night.*); read by day-/night-conditional effects. A fresh Game starts neither.
        enum DayNight { DN_NEITHER, DN_DAY, DN_NIGHT };
        DayNight day_night = DN_NEITHER;
        size_t timestamp = 0;
        // The game's seeded random number generator (every shuffle and random choice).
        struct Rng {
            size_t seed;
            std::mt19937 engine;
        };
        Rng rng;
        // Object identity (CR 400.7, object_ref.h).
        struct ObjectIdentity {
            // Monotonic source for Zone::obj_gen (CR 400.7 object identity). Handed out and
            // post-incremented on every Orderer::add_to_zone, so each zone entry gets a
            // globally-unique stamp that no recycled entity id can collide with. Starts at 1 so
            // 0 stays reserved for "never stamped". Purely internal (not serialized into the ML
            // observation); deterministic because the add_to_zone call sequence is deterministic.
            uint64_t next_obj_gen = 1;
            // CR 400.7j follow window (object_ref.h): open while one effect resolves; maps each object
            // the effect moved to the identity stamp it had before its first move in the window.
            bool follow_window_open = false;
            std::map<Entity, uint64_t> follow_window_origins;
        };
        ObjectIdentity identity;
        Entity player_a_entity;
        Entity player_b_entity;
        bool ended = false;
        int winner = 0;  // 0=none (or a draw once `ended`), 1=PLAYER_A, 2=PLAYER_B (Zone::Ownership values)
        // The turn: its number, step and active player, the extra turns owed, and the progress of
        // its cleanup step.
        struct TurnState {
            size_t turn = 0;
            Step step = UNTAP;
            bool player_a_turn = true;  // Player A is the active player
            // Pending extra turns (CR 500.7 / 720). Each entry is the player who will take an extra
            // turn, treated as a LIFO stack: the most recently added extra turn is taken first (CR
            // 500.7, "The most recently created turn will be taken first"). Consulted at turn hand-off
            // (advance_step's cleanup → next turn) before flipping the active player; a non-empty stack
            // makes the player on top take the next turn instead of passing to the opponent. General
            // over any "take an extra turn" effect (effects::add_turn pushes onto it). Persists across
            // turns; empty in a fresh Game.
            std::vector<Zone::Ownership> extra_turns;
            // Cleanup step (CR 514): the 514.2 actions (damage removal, "until end of turn" and
            // "this turn" effects end) have happened in this cleanup step; a state-based action was
            // performed in it; and players received priority in it (CR 514.3a), so another cleanup
            // step follows once the stack is empty and all players pass in succession.
            bool cleanup_effects_ended = false;
            bool cleanup_sba_performed = false;
            bool cleanup_priority_round = false;
            // Spells cast by the previous turn's active player DURING that turn (CR 502.2 / 731.2),
            // read by the untap-step day/night turn-based check on the following turn. Snapshotted at
            // cleanup from Player::spells_cast_this_turn before that per-turn counter is reset. Because
            // cleanup resets BOTH players' spells_cast_this_turn to 0 (so a player's instants cast on
            // the opponent's turn never leak into their own-turn count), the snapshot is just the active
            // player's spells_cast_this_turn at cleanup. -1 until a turn has ended: on the game's first
            // turn there is no previous turn to check (CR 731.2).
            int prev_turn_active_spell_count = -1;
        };
        TurnState turn_state;
        // Priority (CR 117): who holds it and who has passed in succession since the last action
        // (both passing resolves the top of the stack or ends the step).
        struct PriorityState {
            bool player_a_has_priority = true;
            bool a_has_passed = false;
            bool b_has_passed = false;
            // Machine mode: block casting a spell or activating an ability after 2 failed payments
            // for it since the last action taken. Keyed by the card or source as the object it is at
            // its origin: a cancelled cast returns the card there as its restored object.
            ObjectMap<int> payment_fail_counts;
        };
        PriorityState priority;
        // The current combat's progress (CR 506-511).
        struct CombatState {
            bool attackers_declared = false;
            bool blockers_declared = false;
            bool damage_dealt = false;
            bool has_first_strikers = false;
            // T3.10: attacker -> (blocker -> damage assigned). Populated by assign_combat_damage()
            // only for attackers that required a controller choice this strike step; deal_combat_damage()
            // reads it and auto-assigns any attacker absent from the map. Cleared at handler entry
            // (per strike step) and in the END_OF_COMBAT cleanup.
            std::map<Entity, std::map<Entity, uint32_t>> damage_assignment;
        };
        CombatState combat;
        std::vector<DelayedTrigger> delayed_triggers;
        // Next DelayedTriggerLink::seq register_delayed_trigger hands out (monotonic per game,
        // starting at 1 so 0 stays "not a delayed trigger").
        uint32_t next_delayed_seq = 1;
        // The monarch (CR 725). MAX_ENTITIES = no monarch (none until an effect makes a player
        // the monarch). Serialized into the state vector's global-extras block (per-player
        // is_monarch flags; see machine_io.h).
        Entity monarch_entity = MAX_ENTITIES;
        std::vector<ObjectRef> delve_exiled;  // cards exiled during current delve cast; cleared after ETB
        size_t x_paid = 0;                  // X value chosen at cast time for X-cost spells
        // Converge (CR 702.90): the number of distinct COLORS of mana spent to cast the spell
        // currently resolving. Restored from the resolving Spell::colors_spent by StackManager
        // before its ability runs (mirrors x_paid), so a Count$Converge amount/condition bound
        // (Prismatic Ending's cmcLEY exile threshold) reads this spell's value. 0 for anything not
        // cast (abilities, copies) or cast with no colored mana.
        int converge = 0;
        std::map<Entity, LastKnownInfo> last_known_info;  // effective characteristics captured as a
                                            // permanent leaves the battlefield (CR 608.2h); read by the
                                            // effective_* accessors when the object is no longer in play.
                                            // A card's entry is superseded once the card is a new
                                            // object (see lki_for); an entry is unread once its
                                            // entity id is issued again (LastKnownInfo::issue)
        std::vector<ObjectRef> remembered_entities;  // Defined$ Remembered — used by Attach sub-ability, Doomsday remember-changed
        ObjectMap<int> ability_resolution_counts;  // Count$ResolvedThisTurn: incremented per triggered-ability resolve of its source
        bool pending_cant_be_countered = false;  // set during mana payment when Cavern restricted mana used
        bool pending_gift_promised = false;  // Gift (CR 702.176): the spell currently being cast promised its gift; read by Count$PromisedGift while its targets are chosen
        // The persisted resolve() continuation of the resolving spell or ability
        // (resolution_frame.h): a resolution that parked a decision resumes from it.
        ResolutionFrame resolution;
        // Combat damage-assignment suspension state (pending.query tag
        // DAMAGE_ASSIGN): the attacker whose lethal-order division is mid-prompt,
        // plus the former inner-loop locals of assign_combat_damage (remaining
        // power to assign, blockers not yet assigned lethal, last blocker picked
        // — the 510.1a leftover-dump target). Value member so a snapshot covers
        // the in-flight division. active == true iff a DAMAGE_ASSIGN query is
        // parked; completed attackers are tracked by their (possibly partial)
        // combat.damage_assignment map entries, the outer scan's re-entrancy guard.
        struct PendingDamageAssign {
            bool active = false;
            Entity attacker = 0;
            uint32_t remaining = 0;
            std::vector<Entity> pool;  // blockers not yet assigned lethal
            Entity last_assigned = 0;
        };
        // Triggered abilities that have triggered (CR 603.2) and wait to be put on the stack the
        // next time a player would receive priority (CR 603.3): those the trigger scan collected
        // from events (collect_triggered_abilities, run before each state-based-action check)
        // and those that triggered outside it — Ward when a spell or ability targets (CR
        // 702.21a), a reflexive "when you do" trigger created by a resolving ability (CR 603.12),
        // queued through queue_trigger. place_waiting_triggers puts them all on the stack in
        // APNAP order (CR 603.3b).
        std::vector<PendingTriggerRT> waiting_triggers;
        // Cast-time suspension state (pending.query tag CAST): the persisted
        // state machine of process_action's CAST_SPELL branch (run_cast_flow,
        // action_processor.cpp). The branch's former locals — the accumulating
        // cost, per-kicker flags, deferred payment pieces, the half-built
        // primary Ability — live here BY VALUE so a cur_game copy covers the
        // whole in-flight cast. Batch 9 converted the LINEAR prompts (kicker /
        // replicate / X ladder / phyrexian pips / life-X / spell-sac / gift);
        // Batch 10 the announce stages (charm modes + their targets, the
        // primary/sub/aura targets — via the shared TargetSelectRT machine and
        // the CAST-tag asker) and the replicate copy retargeting (CopySpellRT);
        // Batch 11 the deferred-payment picks (delve count/picks, flashback
        // sacrifice, escape exile-from-graveyard, alt-cost pitch/return); only
        // the interactive mana payment and the interactive hybrid pips remain
        // blocking (machine mode auto-resolves both with zero decisions).
        // active == true from the CAST_SPELL action until the spell becomes
        // cast (COPY_TARGETS end) or the proposal is reversed (rewind_cast).
        // The card is on the stack for that whole span (CR 601.2a).
        struct PendingCast {
            // Where the flow resumes. The enum is listed in RUN order, which follows
            // CR 601.2: announce (601.2b) -> targets (601.2c) -> pay every cost
            // (601.2f-h). Everything from ALT_PITCH to PAY_APPLY is the single payment
            // phase: it CHOOSES each non-mana cost item first, then floats and pays
            // mana, then applies the chosen items and every other cost (CR 601.2h).
            // Nothing irreversible happens before MANA_PAY commits, so a proposal
            // reversed at any step before PAY_APPLY (CR 601.5 / 733.1) is undone by
            // rewind_cast: the card returns to cast_origin, mana_snap restores the
            // payment's mana abilities, and the chosen items are simply dropped.
            enum Step {
                COST,             // cost-branch dispatch (flashback/escape/impulse/alt vs regular)
                KICKER,           // per-kicker optional-additional-cost y/n
                REPLICATE,        // repeated replicate-cost y/n loop
                CHOOSE_X,         // X-cost ladder
                HYBRID,           // hybrid pips (machine auto-resolve; interactive stays blocking)
                PHYREXIAN_PIP,    // per-pip colored-mana-or-2-life
                LIFE_X,           // variable life X ANNOUNCEMENT (Toxic Deluge's PayLife<X>;
                                  // CR 601.2b announces the value, the life is paid at PAY_APPLY)
                GIFT,             // gift promise y/n (incl. the forced-promise no-prompt path)
                CHARM_MODE,       // modal mode announcement (one pick per entry, CR 601.2b)
                CHARM_TARGET,     // the just-picked mode's targets (before the next mode pick)
                PRIMARY_TARGET,   // primary spell target
                SUB_TARGET,       // targeting chained sub-abilities' targets (ends with AddComponent)
                AURA_TARGET,      // aura enchant target (pc.enchant_ab)
                ANNOUNCE,         // primary-template setup; dispatches into the announce steps
                // ── payment phase (CR 601.2f-h), all of it after targets ──
                ALT_PITCH,        // alt-cost exile-from-hand picks (Force of Will's pitch)
                ALT_RETURN,       // alt-cost return-to-hand picks (Daze)
                ALT_SAC,          // alt-cost sacrifice picks (Fireblast: sacrifice two Mountains)
                SPELL_SAC,        // spell's own additional sacrifice cost (Natural Order)
                DELVE_COUNT,      // delve exile count menu (CR 702.66)
                DELVE_PICK,       // delve exile picks, one card at a time
                DEF_SAC,          // deferred flashback sacrifice (Cabal Therapy)
                DEF_EXILE_TYPES,  // escape exile-by-types picks (CR 702.139)
                DEF_EXILE_COUNT,  // escape exile-by-count picks (Uro)
                MANA_PAY,         // float doomed permanents, then pay mana (interactive payer stays blocking)
                PAY_APPLY,        // apply the chosen cost items + life + energy, once the mana committed
                COPY_TARGETS,     // replicate copy retargeting (pc.copy_rt) + take_action LAST
                FINISH            // the spell becomes cast (CR 601.2i): cast events; seeds COPY_TARGETS
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
            // The card's Zone before the cast moved it to the stack (CR 601.2a):
            // where a reversed proposal returns it (Orderer::rewind_cast_move) and
            // the origin of the zone change reported once it becomes cast.
            Zone cast_origin;
            // cur_game.x_paid before this cast announced an X, restored by a rewind.
            size_t x_paid_before = 0;
            // Snapshot of mana state taken as MANA_PAY begins (the first step that
            // can activate a mana ability), restored when the proposal is reversed
            // after it (mana_snap_taken).
            ManaPaymentSnapshot mana_snap;
            bool mana_snap_taken = false;
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
            // Total mana pips actually paid to cast this spell (CR 106/601.2g). Captured from
            // deferred_mana_cost at the MANA_PAY step (after any delve/improvise reduction) and
            // copied onto the resulting Spell::mana_spent at FINISH. 0 for a free / no-mana
            // alternative cost. Read by a SpellCast trigger's ValidSA$ Spell.ManaSpent filter.
            int mana_spent = 0;
            // Converge (CR 702.90): the exact COLORS of mana actually spent to cast this spell,
            // accumulated by the payment (prompt_mana_payment's spent-sink) as pips leave the pool.
            // The distinct real colors (WHITE..GREEN) are copied onto Spell::colors_spent at FINISH,
            // then restored into cur_game.converge at resolution. Empty for a free / no-mana cast.
            ManaValue mana_spent_colors;
            bool deferred_delve = false;
            bool deferred_improvise = false;
            // Non-mana alternative-cost pieces (flashback life/sacrifice, escape
            // exile-from-graveyard) are deferred the same way: targets first (601.2c),
            // then every cost (601.2g/h). They are paid only after the deferred mana
            // payment commits, so a cancelled payment never costs life or a creature.
            // An exile grant's LIFE resource cost (CR 118.9) adds to it too.
            int deferred_life_cost = 0;
            // Life paid for Phyrexian pips (CR 107.4f). Paid as each pip is announced, so the
            // choice shows in the life total at the prompts that follow; a rewind gives it back.
            int phyrexian_life_paid = 0;
            // An exile grant's ENERGY resource cost (Amped Raptor, CR 118.9 / 107.14), paid at
            // PAY_APPLY with the rest of the non-mana costs.
            int deferred_energy_cost = 0;
            // Variable life X (Toxic Deluge's PayLife<X>): the value ANNOUNCED at LIFE_X
            // (CR 601.2b), paid at PAY_APPLY. Kept apart from deferred_life_cost only so
            // the narrative can still say "pays N life (X = N)". -1 = no such cost.
            int life_x_announced = -1;
            std::string deferred_sac_spec;
            int deferred_exile_min_types = 0;
            int deferred_exile_count = 0;
            // Cost items that have been CHOSEN but not yet applied. Every non-mana cost
            // that moves a card (the alt cost's pitch/bounce/sacrifice, a spell's
            // additional sacrifice, flashback's sacrifice, escape's graveyard exiles,
            // delve's graveyard exiles) is picked before the mana payment and applied
            // only after it commits, at PAY_APPLY. That split is what makes a reversed
            // proposal a CLEAN rewind: before PAY_APPLY nothing irreversible has happened,
            // so rewind_cast restores the whole game (CR 601.2h lets the costs be paid in
            // any order, and CR 733.1 wants the reversal to leave no trace). It also lets
            // the payment SEE the doomed permanents — MANA_PAY floats their mana before
            // spending, since they are still on the battlefield at that point.
            struct CostRemoval {
                Entity entity = 0;
                Zone::ZoneValue dest = Zone::GRAVEYARD;
                // The narrative line, formatted at PICK time (while the card is still
                // where it was) and emitted at PAY_APPLY, where the move happens. Held
                // whole so each cost keeps its own wording ("sacrifices X", "returns X
                // to hand", "exiles X from their graveyard").
                std::string log;
                // A delve exile (CR 702.66a): recorded in cur_game.delve_exiled as it moves.
                bool delve = false;
            };
            std::vector<CostRemoval> cost_removals;
            // The half-built primary spell ability. The ENTITY's Ability
            // component is added at the same point as the blocking flow did
            // (at the end of SUB_TARGET, once every announce target is
            // chosen), so component state at every prompt matches the blocking
            // flow's exactly: absent at the announce prompts, present from the
            // payment steps on.
            Ability ability;
            bool have_ability = false;
            // ── Announce-stage progress (Batch 10) ──
            // Charm announcement: completed mode iterations (a mode counts
            // once its targets are chosen). The picked modes themselves
            // persist in ability.charm_chosen, which reconstructs the taken[]
            // menu filter on resume.
            int charm_picks_done = 0;
            // Which chained sub-ability's target selection is in flight.
            size_t sub_idx = 0;
            // The shared in-flight target pick (charm-mode / primary / sub /
            // aura — one at a time, reset between). Member field per the
            // Batch 4 finding (never its own EffectRuntime alternative).
            TargetSelectRT tsel;
            // AURA cast (CR 303.4): the transient targeting ability built from
            // the Enchant filter — a former local of the blocking flow,
            // persisted so a suspended enchant-target pick resumes it.
            Ability enchant_ab;
            // Replicate copies' retargeting machine (COPY_TARGETS; see
            // CopySpellRT / effect_copy_spell.cpp).
            CopySpellRT copy_rt;
            // ── Deferred-payment progress (Batch 11) ──
            // Alt-cost pick loops (Force of Will's pitch, Daze's return):
            // completed picks.
            int alt_pitch_done = 0;
            int alt_return_done = 0;
            int alt_sac_done = 0;  // alt-cost Sac<N/Type> sacrifices completed (Fireblast)
            // Delve (CR 702.66): the chosen exile count, completed picks, and
            // the pre-delve priority seat (the blocking prompt seated both
            // delve stages on the caster and restored the seat afterwards;
            // the arm-time repoint persists its prev value here instead, and
            // delve_seat_held says it is still to be restored).
            size_t delve_exile_ct = 0;
            size_t delve_picks_done = 0;
            bool delve_prev_priority_a = true;
            bool delve_seat_held = false;
            // Escape ExileFromGrave progress (CR 702.139): distinct card types
            // exiled so far (min-types form) / cards exiled so far (Uro's
            // literal-count form). Candidate lists are re-derived from the
            // live graveyard at each arm, exactly like the blocking loops.
            std::set<std::string> escape_exiled_types;
            int escape_exiled_count = 0;
        };
        // Activated-ability suspension state (pending.query tag ACTIVATION):
        // the persisted state machine of process_action's ACTIVATE_ABILITY
        // branch (run_activation_flow, action_processor.cpp). The activated
        // ability, the in-flight targeted copy, the chosen X and the chosen
        // cost items live here BY VALUE so a cur_game copy covers the whole
        // in-flight activation. Every machine-mode prompt in the family is a
        // loop-top pending decision: the X-activation and loyalty-X ladders,
        // the pre-cost target selection (battlefield and hand/graveyard
        // activation zones), and the sacrifice/return cost picks (including
        // ninjutsu's return-an-attacker pick). The interactive mana payment
        // stays blocking (machine mode auto-pays with zero decisions).
        // active == true from the ACTIVATE_ABILITY action until the ability
        // resolves off-stack (mana ability), becomes activated, or the
        // proposal is reversed (rewind_activation). A non-mana ability is on
        // the stack for that whole span (CR 602.2a).
        struct PendingActivation {
            // Where the flow resumes, in CR 602.2 order: announce and choose (X,
            // targets, the cost items to sacrifice/return), then pay every cost
            // (602.2b -> 601.2g-h). Nothing is paid before PAY, and PAY_APPLY applies
            // the non-mana costs only once the mana committed, so a proposal reversed
            // at any step before PAY_APPLY (CR 733.1) leaves no trace.
            enum Step {
                ZONE_TARGET,       // hand/graveyard activation: pre-cost target select
                X_LADDER,          // X activation cost (Candelabra of Tawnos)
                LOYALTY_X,         // X loyalty cost (Chandra, Flamecaller's [-X])
                TARGET,            // battlefield pre-cost select_target
                COST_SAC,          // type-based sacrifice-cost pick (chosen, not yet moved)
                COST_RETURN,       // return-to-hand-cost pick (chosen, not yet moved)
                PAY,               // tap cost + mana payment (failure rewinds; sync)
                PAY_APPLY,         // loyalty/life/energy/sacrifice/return/discard costs (sync)
                FINISH             // mana production, or the ability becomes activated + take_action
            };
            bool active = false;
            Step step = X_LADDER;
            // Activation identity: reconstructs the consumed LegalAction
            // across the suspension gap.
            Entity source_entity = 0;
            bool activator_is_a = true;
            // Hand/graveyard ActivationZone$ path (no Permanent component):
            // routes FINISH to the zone path's completion (auto-consume to
            // graveyard) instead of the battlefield mana/stack completion.
            bool zone_path = false;
            // The activated ability as the consumed LegalAction carried it
            // (costs are read from here), and the in-flight copy the target
            // selection writes into (the stack object's Ability once it
            // becomes activated) — the branch's former `ability` / `stack_ab`
            // pair.
            Ability ability;
            Ability stack_ab;
            // The ability's object on the stack (CR 602.2a), created as the
            // activation is proposed; 0 for a mana ability (CR 605.3b).
            Entity stack_entity = 0;
            // X chosen at the X_LADDER step (added as generic pips to the
            // PAY cost). The loyalty-X choice lives only in cur_game.x_paid.
            size_t x_activation = 0;
            // cur_game.x_paid before this activation announced an X, restored by
            // a rewind.
            size_t x_paid_before = 0;
            // The permanents chosen at COST_SAC / COST_RETURN, moved at PAY_APPLY
            // (0 = none).
            Entity sac_choice = 0;
            Entity return_choice = 0;
            // Snapshot of mana state taken as PAY begins, restored when the payment
            // fails (mana_snap_taken).
            ManaPaymentSnapshot mana_snap;
            bool mana_snap_taken = false;
            // The shared in-flight target pick (ZONE_TARGET / TARGET). Member
            // field per the Batch 4 finding (never its own EffectRuntime
            // alternative).
            TargetSelectRT tsel;
        };
        // Pre-game phase state (Family F): mulligans and CR 103.6b opening-hand
        // actions run as loop-top decisions driven by the main loop's pregame
        // gate (run_pregame_step, game_driver.cpp) instead of a synchronous
        // pre-loop block. All stage/progress state lives here BY VALUE so a
        // cur_game copy (snapshot_save) covers the whole in-flight pregame —
        // a RESTORE targeting a keep/mulligan or bottoming root re-derives the
        // decision at the gate even after the real line progressed into the
        // main game (only a main-loop gate can bounce control back there).
        struct PregameState {
            enum Stage {
                MULL_DECIDE,      // one keep/mulligan decision per gate call (CR 103.4/103.5)
                MULL_BOTTOM,      // one London bottoming pick per gate call (CR 103.4a)
                FIAT_SETUP,       // promptless: test-harness presets, companions, preplaced SBE
                OPENING_ACTIONS,  // CR 103.6b opening-hand y/n offers, one per gate call
                DONE              // pregame over
            };
            // Default DONE: a mid-game snapshot restores as non-pregame (and
            // pre-existing snapshots/back-compat states never re-enter the
            // gate). play_single_game initializes it to MULL_DECIDE at game
            // start.
            Stage stage = DONE;
            bool a_goes_first = true;
            bool a_kept = false;
            bool b_kept = false;
            int mulls_a = 0;
            int mulls_b = 0;
            // CR 103.5 round bookkeeping: whether the seat deciding first this
            // round has already had its keep/mulligan decision this round.
            bool first_seat_decided_this_round = false;
            // MULL_BOTTOM: whose kept hand is being bottomed, and how many
            // cards remain to bottom.
            Zone::Ownership bottoming_owner = Zone::UNKNOWN;
            int bottom_remaining = 0;
            // FIAT_SETUP: presets/companions already placed (a resumed setup
            // must never re-place), and the preplaced battlefield entities
            // awaiting their summoning-sickness clear once the SBE pass
            // settles (cleared after use so the resume path is idempotent).
            bool fiat_placed = false;
            std::vector<Entity> preplaced;
            // OPENING_ACTIONS iteration state: which player (0/1 in play
            // order), the up-front snapshot of their kept hand (CR 103.5 —
            // no new card can join the opening hand mid-phase), and the next
            // hand card to consider.
            int oh_player_idx = 0;
            bool oh_hand_init = false;
            std::vector<Entity> oh_hand;
            size_t oh_card_idx = 0;
            bool oh_any_ran = false;
        };
        PregameState pregame;
        // Turn-based draw suspension state (pending.query tag TURN_DRAW): the
        // draw step's draw batch with the dredge draw-replacement question
        // (CR 702.52a) parked as a loop-top decision. advance_step's
        // UPKEEP→DRAW case arms it and calls resume_pending_draws; while a
        // dredge question is parked the step-change epilogue is deferred
        // (pass flags stay true, mana pools un-emptied — exactly the state
        // the blocking prompt read) and the loop-top TURN_DRAW dispatch runs
        // it via finish_suspended_turn_draw once the batch completes. Value
        // member so a snapshot covers the parked batch. active == true only
        // while draws remain (or a dredge query is parked); cleared when the
        // batch completes or the game ends mid-batch.
        struct PendingDrawRT {
            bool active = false;
            Zone::Ownership player = Zone::UNKNOWN;
            int remaining = 0;
        };
        // Suspended decisions (see pending_query.h / resolution_frame.h): the decision parked for
        // the main loop to emit, and the persisted state of each flow that can park one midway.
        // Value members so a cur_game copy (snapshot_save) covers the whole suspended state for
        // free.
        struct PendingDecisions {
            // The mandatory choice (a decision other than priority) the next decision is.
            MandatoryChoice choice = NONE;
            PendingQuery query;  // the decision parked for the main loop to emit
            // The source entity of the spell/ability currently making a mid-resolution choice
            // (target select, dig/scry/surveil pick, search, discard, modal, ...). Serialized into
            // the state vector's pending-decision context block so the ML observation shows WHAT is
            // asking for the current choice (a mid-resolution choice's source need not be on the
            // stack). Managed exclusively via PendingDecisionScope; 0 = no ability-driven choice
            // pending.
            Entity decision_source = 0;
            // Combat sub-prompt suspension state (pending.query tags ATTACK_TARGET /
            // BLOCK_TARGET): the creature whose target sub-prompt is currently parked.
            // Set when declare_attackers/declare_blockers suspends on the target menu,
            // consumed and cleared by the loop-top resume. Value members so a snapshot
            // covers the parked selection. 0 = no sub-prompt parked.
            Entity attacker = 0;
            Entity blocker = 0;
            PendingDamageAssign damage;  // pending.query tag DAMAGE_ASSIGN
            // Trigger-placement suspension state (pending.query tag TRIGGER_PLACE):
            // the APNAP-flattened queue of collected triggers still to be put on
            // the stack, plus the front trigger's in-flight target selection. Set
            // by place_triggers_apnap, driven by resume_trigger_placement, cleared
            // at placement completion (which also restores saved_priority). Value
            // member so a snapshot covers the
            // parked placement. See resolution_frame.h.
            TriggerPlacementRT trigger_placement;
            PendingCast cast;  // pending.query tag CAST
            PendingActivation activation;  // pending.query tag ACTIVATION
            PendingDrawRT draw;  // pending.query tag TURN_DRAW
            // Miracle (CR 702.94): miracle_reveal is a first-of-turn miracle card just drawn,
            // awaiting its owner's PRIVATE reveal decision — the "you may reveal it as you draw it"
            // special action (off the stack, hidden from the opponent until they choose to reveal),
            // riding the mandatory-choice channel (proc_mandatory_choice). Set in
            // Orderer::perform_draw; 0 when nothing is pending, and cleared each cleanup. On reveal
            // the card becomes public and the linked "you may cast it" triggered ability (category
            // MiracleCast) is put on the stack; the owner decides whether to cast it as that trigger
            // resolves (effect_miracle.cpp, CR 608.2g). Miracle is never offered as a normal
            // priority-menu cast (can_afford_alt returns false for it).
            Entity miracle_reveal = 0;
        };
        PendingDecisions pending;

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
        // Player-scoped "protection from everything" grant (CR 702.16; The One Ring's ETB: "you
        // gain protection from everything until your next turn"). While active the protected
        // `player` can't be the target of a spell/ability an opponent controls, and isn't dealt
        // damage by any source an opponent controls. `until_your_next_turn` selects the duration:
        // when true the grant is reverted at the start of the protected player's next turn (their
        // untap step); when false it lapses at cleanup (end of turn). Consulted in
        // Ability::is_legal_target and deal_damage.
        struct PlayerProtectionFromEverything {
            Zone::Ownership player = Zone::UNKNOWN;
            bool until_your_next_turn = false;
        };
        // Cast-timing permission "you may cast <filter> spells as though they had flash" (CR 702.8 /
        // 601.3a; Teferi, Time Raveler's +1 "Until your next turn, you may cast sorcery spells as
        // though they had flash."). While an entry is active, `controller` may cast a spell matching
        // `filter` (e.g. "Sorcery") ignoring the sorcery-speed timing restriction. `until_your_next_turn`
        // selects the duration: when true the grant lapses at the controller's next untap step; when
        // false it lapses at cleanup (end of turn). A sourceless grant (the Effect belongs to no
        // permanent). Consulted by the cast-speed gate (rules_mod::cast_with_flash_active).
        struct CastWithFlashPermission {
            Zone::Ownership controller = Zone::UNKNOWN;
            std::string filter = "";  // ValidCard$ filter the flash permission applies to
            bool until_your_next_turn = false;
        };
        // Turn-scoped combat-damage prevention shield (CR 615) created by a resolving DB$ Effect |
        // ReplacementEffects$ <Event$ DamageDone | Prevent$ True | IsCombat$ True | ValidSource$/
        // ValidTarget$ Card.IsRemembered> (Maze of Ith's "Prevent all combat damage that would be
        // dealt to and dealt by that creature this turn"). Each shield remembers one `creature`;
        // while active, all COMBAT damage that creature would DEAL (prevent_as_source) and/or all
        // combat damage that would be dealt TO it (prevent_as_target) is prevented. A sourceless
        // turn-long grant (the Effect belongs to no permanent); consulted by deal_combat_damage
        // via combat_damage_prevented() and cleared at cleanup. General over any DamageDone/Prevent
        // Effect keyed on a remembered object (reusable by future fog/prevention cards).
        struct CombatDamagePreventionShield {
            ObjectRef creature;
            bool prevent_as_source = false;  // ValidSource$ Card.IsRemembered — damage BY the creature
            bool prevent_as_target = false;  // ValidTarget$ Card.IsRemembered — damage TO the creature
        };
        // Cards revealed while in a library (CR 701.20a: shown to all players for as long as the
        // revealing effect needs them). A revealed card the effect puts into a hand stays known to
        // the opponent there (Zone::identity_known). An entry ends when the card changes zones or
        // its library is shuffled (CR 701.20d), and every entry ends when the resolution finishes.
        ObjectSet revealed_in_library;
        ObjectSet chosen_cards;  // cards chosen by a resolving ChooseCard effect (Ajani -4's kept permanents, read by SacrificeAll's nonChosenCard filter; Dauthi Voidwalker's exiled card, read by a RememberObjects$ ChosenCard Effect); cleared by Cleanup ClearChosenCard$ and when the resolution finishes
        std::string named_card = "";  // card name chosen by a resolving SP$/DB$ NameCard effect (CR 201.4, Cabal Therapy); read by a chained Card.NamedCard discard, cleared after the spell finishes resolving
        int chosen_number = 0;  // integer chosen by a resolving DB$ ChooseNumber effect (Wrath of the Skies: "pay any amount of {E}"); read downstream via Count$ChosenNumber (e.g. the cmc bound and PayEnergy unless-cost of the chained DestroyAll)
        std::vector<ObjectRef> imprinted_entities;  // the set of cards "imprinted" (recorded) by a resolving DB$ PeekAndReveal | ImprintRevealed$ True (Atraxa, Grand Unifier: the top-N revealed cards); read by a chained Card.IsImprinted filter (RepeatTypesFrom$ / ChooseCard / ChangeZoneAll) and cleared by Cleanup ClearImprinted$. Distinct from remembered_entities (which holds the chosen cards taken to hand).
        std::string chosen_type = "";  // the current card type set by a DB$ RepeatEach | RepeatTypesFrom$ loop (Atraxa: iterated per card type present among the imprinted cards); read by a ChooseCard Choices$ Card.ChosenType filter, cleared when the loop ends
        // Play-from-exile permission: a card in EXILE that an effect lets a player play — at
        // priority for as long as the grant lasts (Light Up the Stage, Ugin -11, Dauthi
        // Voidwalker, warp; cleared at cleanup unless it lasts longer), or during a resolution
        // only (Amped Raptor's DB$ Play, suspend's last time counter; during_resolution, CR
        // 608.2g). Keyed by the exiled card as an object: it lapses once the card becomes a new
        // object, except that it follows the card onto the stack as it is cast (CR 400.7g). The
        // casting path reads it for the cost that replaces the card's mana cost (CR 118.9) and
        // consumes it as the card is cast.
        struct ImpulseCastPermission {
            // FREE = cast without paying its mana cost (Ugin, Eye of the Storms' -11: "cast those
            // cards without paying their mana costs"; Dauthi Voidwalker: "play it ... without
            // paying its mana cost", CR 118.9 / 601.2f). ENERGY/LIFE pay an alternative resource
            // cost equal to `amount` (Amped Raptor's DB$ Play). NORMAL = PLAY the card for its
            // NORMAL cost (Light Up the Stage's "you may play those cards").
            enum Resource { ENERGY, LIFE, FREE, NORMAL } resource = ENERGY;
            int amount = 0;            // resolved cost (e.g. the card's mana value); 0 when FREE/NORMAL
            Zone::Ownership caster = Zone::UNKNOWN;  // who may cast it (its controller)
            bool allow_land = false;   // a "play" grant may also play a LAND card from exile (CR 305.1)
            // persist_until_end_of_next_turn: the permission survives the cleanup of the turn it was
            // granted; it is removed at the caster's NEXT turn's cleanup (CR "until the end of your
            // next turn"). grant_turn records cur_game.turn_state.turn at grant so game.cpp can detect that
            // next-turn cleanup. Default (false) = the Forge default "this turn" (cleared every cleanup).
            bool persist_until_end_of_next_turn = false;
            size_t grant_turn = 0;
            // during_resolution: granted by a resolving effect for a cast made as part of that
            // resolution (CR 608.2g; cast_during_resolution — Suspend's last time counter, CR
            // 702.62a). The cast ignores the card's type-based timing, and the permission is never
            // offered at priority: it is consumed by that cast or dropped when it isn't made.
            // Ordinary free casts (Ugin) leave this false and keep their type-based timing.
            bool during_resolution = false;
            // warp: granted when a warp-cast object is exiled at the next end step. Unlike the
            // per-turn grants above, a warp permission persists across turns FOR AS LONG AS the
            // card remains in exile — it lapses only once the card leaves exile (cast, or moved
            // by another effect). The recast keeps the card's normal sorcery/instant timing (a
            // creature is recast at sorcery speed), so during_resolution stays false. See
            // effect_warp.cpp.
            bool warp = false;
        };
        // Continuous effects and floating triggered abilities that belong to no permanent: created
        // by resolved spells and abilities (CR 611.2), and emblems (CR 114). Each lasts as its
        // entry says ("this turn", "until your next turn", for the game); end_cleanup_effects and
        // the untap step end them.
        struct ResolvedEffects {
            // Turn-long "spells you control can't be countered" grant created by a resolving spell/
            // ability (Veil of Summer's DB$ Effect | ReplacementEffects$ AntiMagic, CR 614.13/
            // CantHappen). A player here means every spell that player controls can't be countered
            // this turn — unlike Hexing Squelcher's battlefield static, this form belongs to no
            // permanent (the instant is in the graveyard), so it is recorded here and cleared at
            // cleanup. Consulted at counter-resolution time (effects::counter).
            std::set<Zone::Ownership> cant_counter_spells_of;
            // Turn-long "can't gain life" prohibition (CR 119.x) created by a resolving activated
            // ability (Roiling Vortex's {R}: AB$ Effect | StaticAbilities$ Mode$ CantGainLife |
            // ValidPlayer$ Player.Opponent — "your opponents can't gain life this turn"). Each player
            // in the set has all life gain replaced with nothing until cleanup. A sourceless turn-long
            // grant (the effect belongs to no permanent), consulted centrally in player_gain_life and
            // cleared at cleanup. General over any CantGainLife effect.
            std::set<Zone::Ownership> cant_gain_life_this_turn;
            std::vector<HexproofFromColors> hexproof_from_colors_this_turn;
            std::vector<PlayerProtectionFromEverything> player_protection_from_everything;
            std::vector<CastWithFlashPermission> cast_with_flash_permissions;
            std::vector<CombatDamagePreventionShield> combat_damage_prevention_shields;
            ObjectSet may_cast_this_turn;  // cards a permission effect (Emry's AB$ Effect) lets their owner cast from the graveyard this turn (CR 601.3e); cleared each cleanup
            ObjectMap<ImpulseCastPermission> impulse_cast_permission;
            // Floating triggered abilities (CR 603.7e-style "this turn" triggers) created by a
            // transient DB$ Effect | Triggers$ <SVar> (e.g. Forth Eorlingas!'s become-monarch-on-
            // combat-damage). Each is a fully-parsed TRIGGERED Ability with its controller bound;
            // the trigger scan (collect_triggered_abilities) tests them against drained events just like
            // a permanent's triggered ability. Cleared at the cleanup step so they last only their
            // turn of creation. General over any until-end-of-turn floating triggered ability.
            std::vector<Ability> floating_triggers;
            // Emblems the players have (CR 114). Each carries permanent continuous statics gathered
            // into g_active_statics every SBA pass (see gather_active_statics). Persists for the game.
            std::vector<Emblem> emblems;
        };
        ResolvedEffects resolved_effects;
        // Cards whose Permanent the running apply_permanent_components pass has created (the pass
        // can suspend and resume): they entered the battlefield together, so none of them is on the
        // battlefield yet when another one's "as it enters" condition is checked (CR 614.12, e.g.
        // "enters tapped unless you control a basic land"). Cleared when the pass completes.
        std::set<Entity> entering_together;

        // Known top-of-library cards of one library, per player who knows them. Index 0 is the
        // top of the library; -1 = unknown (default). `by_owner` is what the library's owner
        // knows (their Brainstorm put-backs, a scry keep, Delver's look), `by_opponent` what the
        // owner's opponent knows (their fateseal or Mishra's Bauble look at it); a card revealed
        // there (CR 701.20a) or put there from a public zone is known to both. The entries follow
        // the cards as cards are placed on or removed from the top, and are cleared on shuffle.
        struct KnownLibraryTop {
            int by_owner[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};
            int by_opponent[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};
        };
        KnownLibraryTop known_top_library_a;  // Player A's library
        KnownLibraryTop known_top_library_b;  // Player B's library

        // CR 725: make `player` the monarch. The previous monarch (if any) ceases to be the
        // monarch (725.3). No-op if `player` is already the monarch. Sourceless inherent monarch
        // triggers (end-step draw, steal-on-combat-damage) are fired by collect_triggered_abilities.
        void set_monarch(Entity player_entity);

        // Queue a triggered ability that just triggered to be put on the stack with the other
        // waiting triggers (waiting_triggers). `ab` carries its source and controller; its targets are
        // chosen as it is put on the stack (CR 603.3d). `log_line` is narrated as it is placed;
        // the scan labels it for the ordering choice.
        void queue_trigger(const Ability &ab, const std::string &log_line);

        // The one place a game ends (CR 104.1): marks the game over, records `result` as the
        // winner (Zone::PLAYER_A / Zone::PLAYER_B, or Zone::UNKNOWN for a draw — CR 104.4a) and
        // prints the one result line "\n<reason> - Player X wins!" / "\n<reason> - the game is
        // a draw!". Every game-ending path goes through it: the state-based-action losses
        // (players_lose), an effect that says a player wins (CR 104.2b), and conceding (CR
        // 104.3a, see concede_current_game in game_driver.h). The first game-ending event
        // decides the game: a call on an already-ended game changes nothing.
        void end_game(Zone::Ownership result, const std::string &reason);
        // A player LOSING decides the game for their opponent (a two-player game, so the last
        // player standing wins — CR 104.2a); see end_game.
        void player_loses(Zone::Ownership loser, const std::string &reason);
        // State-based-action losses found in one check (CR 704.3): one loser's opponent wins; if
        // both players lose simultaneously the game is a draw (CR 104.4a). No-op when neither
        // player loses.
        void players_lose(bool a_loses, bool b_loses, const std::string &reason);

        // The known-top record of `library_owner`'s library.
        KnownLibraryTop &known_top_library(Zone::Ownership library_owner);
        const KnownLibraryTop &known_top_library(Zone::Ownership library_owner) const;
        // What `viewer` knows about the top of `library_owner`'s library (KNOWN_TOP_LIBRARY_SIZE
        // vocab ids, -1 = unknown).
        const int *known_top_library_seen_by(Zone::Ownership library_owner,
                                             Zone::Ownership viewer) const;
        void clear_known_top_library(Zone::Ownership library_owner);
        // A card was placed on top of the library: every known entry moves one deeper, and the new
        // top is known to the owner as `owner_idx` and to the opponent as `opp_idx` (-1 = unseen).
        void known_top_library_push(Zone::Ownership library_owner, int owner_idx, int opp_idx);
        // The card at `pos` left the library: the entries below it move up one.
        void known_top_library_remove_pos(Zone::Ownership library_owner, int pos);
        // `knower` learned the card sitting at `pos` (no card moves). No-op outside the window.
        void known_top_library_note(Zone::Ownership library_owner, int pos, int card_vocab_idx,
                                    Zone::Ownership knower);
        // The card at `from` moved to depth `to` (the cards in between shift to close the gap),
        // carrying both players' knowledge of it. Positions outside the window are unknown.
        void known_top_library_move(Zone::Ownership library_owner, int from, int to);

        bool ready_to_resolve();
        // CR 514.2: all damage marked on permanents is removed and all "until end of turn" and
        // "this turn" effects end, simultaneously. Run once per cleanup step, after the 514.1
        // discard (process_turn_based_actions).
        void end_cleanup_effects();
        // Begin the end of combat step (CR 511) for the active player `active_player_entity`.
        void begin_end_of_combat_step(Entity active_player_entity);
        // Begin a cleanup step (CR 514) for the active player `active_player_entity`.
        void begin_cleanup_step(Entity active_player_entity);
        // CR 615: is this combat damage prevented by an active combat-damage prevention shield?
        // True when `source` is a shielded creature under a prevent-as-source shield (damage it
        // would deal), or `target` is a shielded creature under a prevent-as-target shield (damage
        // it would be dealt). Consulted at each combat-damage assignment in deal_combat_damage.
        bool combat_damage_prevented(Entity source, Entity target) const;
        // True when `creature` is the creature of any active combat-damage prevention shield,
        // in either direction (damage it would deal or be dealt).
        bool combat_damage_shielded(Entity creature) const;
        bool is_mandatory_choice_pending() const;
        // The turn's end resets everything counted "this turn": both players' counts
        // (Player::reset_turn_counters), the per-source resolution counts and a lapsed miracle
        // reveal. `active_player_entity` is the ending turn's active player.
        void reset_turn_counters(Entity active_player_entity);
        void generate_players(const Deck &deck_a, const Deck &deck_b);
        bool advance_step(std::shared_ptr<class StackManager> stack_manager, std::shared_ptr<class Orderer> orderer);
        // The step-change epilogue advance_step deferred when the turn-based
        // draw suspended on a dredge question: seat priority with the active
        // player, reset pass tracking, empty the mana pools. Called by the
        // loop-top TURN_DRAW dispatch once resume_pending_draws completes.
        void finish_suspended_turn_draw();
        void pass_priority();
        void take_action();  // resets last_player_passed since an action was taken

    private:
        Entity gen_player(const Deck &deck);
};

// RAII marker for Game::pending.decision_source: constructed at the top of an effect/target
// handler that is about to prompt a choice on behalf of a spell/ability, so every get_input
// within the scope serializes that ability's source card into the observation's
// pending-decision context. Saves/restores the previous value, so nested choices (a
// sub-ability's target chosen during a parent's resolution) unwind correctly.
struct PendingDecisionScope {
    Entity prev_source;
    explicit PendingDecisionScope(Entity source) : prev_source(cur_game.pending.decision_source) {
        cur_game.pending.decision_source = source;
    }
    ~PendingDecisionScope() { cur_game.pending.decision_source = prev_source; }
};

// True while a suspended decision is parked for the main loop to emit (see
// pending_query.h). Later batches gate cooperative early-returns on it
// (`if (decision_suspended()) return;` in suspendable callees).
inline bool decision_suspended() { return cur_game.pending.query.active; }

// Drive the turn-based draw batch (Game::pending.draw): consume a latched
// TURN_DRAW answer if one is parked, then draw / dredge one card at a time
// until the batch completes or the next dredge question arms a fresh query
// (tag TURN_DRAW). Called synchronously by advance_step's UPKEEP→DRAW case
// (promptless when no dredge applies) and by the loop-top TURN_DRAW dispatch.
void resume_pending_draws(Game &game, std::shared_ptr<class Orderer> orderer);

#endif // __cplusplus

#endif /* GAME_H */
