#include "affected.h"

#include <algorithm>
#include <cstring>

#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/entry_info.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "entry.h"
#include "filters.h"
#include "players.h"
#include "zones.h"

extern Coordinator global_coordinator;
extern Game cur_game;

using ObjectsFn = void (*)(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
using PlayersFn = void (*)(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
using ReadByFn = bool (*)(const AbilityDef &ability);

enum class Refers : uint8_t { OBJECT, PLAYER, OBJECT_OR_PLAYER };

// One Defined$ name. A name ending in a space is a prefix whose remainder is the spec's filter.
struct DefinedRow {
    const char *name;  // null for the kinds no script spells
    ObjectSpec::Kind kind;
    Refers refers;
    ObjectsFn objects = nullptr;   // what it resolves to as objects; null: none, or its one reader's own
    PlayersFn players = nullptr;   // what it resolves to as players; null: none, or its one reader's own
    ReadByFn read_by = nullptr;    // the abilities that act on it; null for every ability
    bool inherits_target = false;  // a sub-ability takes it from its parent's chosen target
    bool statics = false;          // a static's AffectedDefined$ may name it
};

static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out);
static void objects_source(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_targets(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_remembered(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_exiled_with(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_enchanted(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_equipped(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void objects_valid(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out);
static void players_you(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_opponent(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_each(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_chain_target(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_targeted_controller(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_triggered_activator(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_triggered_player(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static void players_triggered_card_owner(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out);
static bool read_by_change_zone(const AbilityDef &ability);
static bool read_by_change_zone_all(const AbilityDef &ability);
static bool read_by_copy_permanent(const AbilityDef &ability);
static bool read_by_pump(const AbilityDef &ability);
static bool read_by_each_player(const AbilityDef &ability);
static bool read_by_no_ability(const AbilityDef &ability);
static const DefinedRow *row_named(const std::string &value);
static const DefinedRow &row_of(ObjectSpec::Kind kind);
static bool row_read_by(const DefinedRow &row, const AbilityDef *ability);
static bool names_parent_target(const ObjectSpec &spec);

// The table the parser, --parse-report and the resolvers read. A Defined$ name is one row here.
static const DefinedRow kDefinedRows[] = {
    // Forge's default: an ability that names nothing acts on its source.
    {nullptr, ObjectSpec::DEFAULT, Refers::OBJECT, objects_source, nullptr, nullptr, false, true},
    {nullptr, ObjectSpec::UNREAD, Refers::OBJECT},

    {"Self", ObjectSpec::SELF, Refers::OBJECT, objects_source, nullptr, nullptr, false, true},
    // TriggeredNewCard(LKICopy) / TriggeredCard(LKICopy) — the object that changed zones and fired
    // this trigger, referenced in its NEW zone. For a Card.Self ChangesZone trigger (CR 603.6e) the
    // triggering object IS the ability's own source (e.g. Triumph of Saint Katherine's dies-trigger,
    // which exiles itself from the graveyard as the head of a "Descend"-style recursion), so bind
    // it to the source. The LKICopy variant is a last-known snapshot of that same card; moving the
    // real card in its new zone is equivalent here. General over self-referential
    // graveyard-recursion cards.
    {"TriggeredNewCardLKICopy", ObjectSpec::SELF, Refers::OBJECT, objects_source},
    {"TriggeredNewCard", ObjectSpec::SELF, Refers::OBJECT, objects_source},
    {"TriggeredCardLKICopy", ObjectSpec::SELF, Refers::OBJECT, objects_source},
    {"TriggeredCard", ObjectSpec::SELF, Refers::OBJECT, objects_source},

    {"Remembered", ObjectSpec::REMEMBERED, Refers::OBJECT, objects_remembered},
    // The objects a DB$ DelayedTrigger captured at registration (RememberObjects$ RememberedLKI).
    // delayed_trigger() restores them into cur_game.resolution.memory.remembered before the fire
    // ability resolves, so this resolves exactly like Defined$ Remembered (Flickerwisp / Phelia
    // return the exiled card).
    {"DelayTriggerRememberedLKI", ObjectSpec::DELAY_REMEMBERED, Refers::OBJECT, objects_remembered},

    // The target its parent chose, which may be an object or a player (CR 608.2c). As players these
    // name the nearest player target up the chain; ChangeZone routes a searched zone by Targeted
    // alone, the one reader that tells the names apart.
    {"Targeted", ObjectSpec::TARGETED, Refers::OBJECT_OR_PLAYER, objects_targets, players_chain_target, nullptr,
     true},
    {"ParentTarget", ObjectSpec::PARENT_TARGET, Refers::OBJECT_OR_PLAYER, objects_targets, players_chain_target,
     nullptr, true},
    {"Parent", ObjectSpec::PARENT_TARGET, Refers::OBJECT_OR_PLAYER, objects_targets, players_chain_target, nullptr,
     true},

    // TriggeredAttacker(LKICopy) — the effect acts on the creature whose attack fired this trigger
    // (Tamiyo, Seasoned Scholar: the attacking creature gets -1/-0). Bound as the ability's target
    // at trigger-fire time from the CREATURE_ATTACKED event. The LKICopy form would use the
    // attacker's last-known info if it has since left combat/play; the trigger resolves during the
    // same declare-attackers step (the attacker is still on the battlefield), so the live entity
    // is used.
    {"TriggeredAttacker", ObjectSpec::TRIGGERED_ATTACKER, Refers::OBJECT, objects_targets},
    {"TriggeredAttackerLKICopy", ObjectSpec::TRIGGERED_ATTACKER, Refers::OBJECT, objects_targets},
    // TriggeredSpellAbility — the effect acts on the spell that fired this trigger (Chalice of the
    // Void: "counter that spell"). Set at trigger fire time.
    {"TriggeredSpellAbility", ObjectSpec::TRIGGERED_SPELL, Refers::OBJECT, objects_targets},
    {"TriggeredSpell", ObjectSpec::TRIGGERED_SPELL, Refers::OBJECT, objects_targets},
    // TriggeredSourceSA — the effect acts on the spell/ability that targeted the source (Reality
    // Smasher: "counter that spell"). Bound at trigger fire time from the BECAME_TARGET event's
    // ENTITY (the targeting object). Distinct from TriggeredSpellAbility (the spell whose cast fired
    // a SpellCast trigger): here the trigger is "became the target of", not "was cast".
    {"TriggeredSourceSA", ObjectSpec::TRIGGERED_SOURCE_SA, Refers::OBJECT, objects_targets},

    // ExiledWith — the effect acts on the card this permanent's Saga chapter I exiled face down,
    // tracked in Permanent::exiled_with on the source (The Creation of Avacyn chapters II & III). A
    // SetState turns it face up (chapter II); a ChangeZone moves it out of exile (chapter III).
    {"ExiledWith", ObjectSpec::EXILED_WITH, Refers::OBJECT, objects_exiled_with},
    {"Enchanted", ObjectSpec::ENCHANTED, Refers::OBJECT, objects_enchanted, nullptr, read_by_change_zone, false, true},
    {"Equipped", ObjectSpec::EQUIPPED, Refers::OBJECT, objects_equipped, nullptr, read_by_no_ability, false, true},
    // ReplacedCard — a ReplaceWith$ move (Mox Diamond, Leyline of the Void): the replacement
    // dispatcher applies it to its event's own object, so no resolver reads this.
    {"ReplacedCard", ObjectSpec::REPLACED_CARD, Refers::OBJECT},
    // Valid <filter> (CopyPermanent's "for each token you control that entered this turn").
    {"Valid ", ObjectSpec::VALID, Refers::OBJECT, objects_valid, nullptr, read_by_copy_permanent},

    // You — the effect's player is the source's controller (CR 109.5). Used by self-pain riders
    // like Ancient Tomb's "deals 2 damage to you" sub-ability.
    {"You", ObjectSpec::YOU, Refers::PLAYER, nullptr, players_you},
    // Player.Opponent — the effect's player is "each opponent" (no chosen target). CR 109.5 /
    // 102.1: in a two-player game this is the single opponent.
    {"Player.Opponent", ObjectSpec::OPPONENT, Refers::PLAYER, nullptr, players_opponent},
    {"Opponent", ObjectSpec::OPPONENT, Refers::PLAYER, nullptr, players_opponent},
    {"Player", ObjectSpec::EACH_PLAYER, Refers::PLAYER, nullptr, players_each, read_by_each_player},
    // TargetedController — the effect's player is the last-known controller of the target the
    // sub-ability inherits (CR 608.2g/h): its parent may already have moved it.
    {"TargetedController", ObjectSpec::TARGETED_CONTROLLER, Refers::PLAYER, nullptr, players_targeted_controller,
     nullptr, true},
    // TriggeredActivator — the effect acts on the player who caused the trigger (the caster of the
    // triggering spell). The actual player is bound at trigger-fire time from the event's PLAYER
    // param. CR 603.x.
    {"TriggeredActivator", ObjectSpec::TRIGGERED_ACTIVATOR, Refers::PLAYER, nullptr, players_triggered_activator},
    // TriggeredPlayer — the effect acts on the player whose event fired this trigger (Roiling
    // Vortex: each player's upkeep damages THAT player). The actual player is bound at trigger-fire
    // time from the event's PLAYER param.
    {"TriggeredPlayer", ObjectSpec::TRIGGERED_PLAYER, Refers::PLAYER, nullptr, players_triggered_player},
    // TriggeredCardController — the effect's player is the controller of the card whose zone change
    // fired the (Mode$ ChangesZone delayed) trigger (Searing Blood: the dead creature's controller).
    // Bound from the watched card's last-known controller at trigger-fire time (CR 608.2g); the card
    // is already in the graveyard by then. Shares triggered_player storage.
    {"TriggeredCardController", ObjectSpec::TRIGGERED_CARD_CONTROLLER, Refers::PLAYER, nullptr,
     players_triggered_player},
    {"TriggeredCardOwner", ObjectSpec::TRIGGERED_CARD_OWNER, Refers::PLAYER, nullptr, players_triggered_card_owner,
     read_by_change_zone_all},
    // TriggeredDefendingPlayer — the effect acts on the defending player of the attack that fired
    // this trigger (Goblin Guide). Bound as the ability's target at trigger-fire time from the
    // CREATURE_ATTACKED event (the opponent of the attacker's controller in a 2-player game), which
    // its reader, Dig, takes for the library's owner.
    {"TriggeredDefendingPlayer", ObjectSpec::TRIGGERED_DEFENDING_PLAYER, Refers::PLAYER},
    // Veil of Summer's Pump grants its hexproof to the controller by its own KW$, whatever this
    // value names.
    {"You & Valid Permanent.YouCtrl", ObjectSpec::YOU_AND_PERMANENTS, Refers::PLAYER, nullptr, nullptr, read_by_pump},
};

static void append_chosen_targets(const Ability &ab, std::vector<Entity> &out) {
    if (!ab.targets.empty()) {
        for (Entity e : live_entities(ab.targets)) out.push_back(e);
    } else if (Entity t = ab.target.get()) {
        out.push_back(t);
    }
}

static void objects_source(const ObjectSpec &, const ResolveCtx &ctx, std::vector<Entity> &out) {
    if (Entity e = ctx.source.get()) out.push_back(e);
}

static void objects_targets(const ObjectSpec &, const ResolveCtx &ctx, std::vector<Entity> &out) {
    if (ctx.ability) append_chosen_targets(*ctx.ability, out);
}

static void objects_remembered(const ObjectSpec &, const ResolveCtx &, std::vector<Entity> &out) {
    for (Entity e : live_entities(cur_game.resolution.memory.remembered)) out.push_back(e);
}

static void objects_exiled_with(const ObjectSpec &, const ResolveCtx &ctx, std::vector<Entity> &out) {
    if (Entity e = exiled_with_card(ctx.source.get())) out.push_back(e);
}

// The object the source Aura enchants: the card it was put onto the battlefield enchanting (Animate
// Dead's reanimation, recorded at cast in EntryInfo::aura_target since the card had no Permanent to
// attach to), only while it is still that object, as one that left its zone in response is a new
// object (CR 400.7) that this Aura does not enchant; else its live attachment link.
static void objects_enchanted(const ObjectSpec &, const ResolveCtx &ctx, std::vector<Entity> &out) {
    Entity enchanted = 0;
    const Entity aura = ctx.source.get();
    const EntryInfo *aura_entry = find_entry_info(ctx.source.lki_entity());
    if (aura_entry && !aura_entry->aura_target.empty()) {
        enchanted = aura_entry->aura_target.get();
    } else if (aura != 0 && global_coordinator.entity_has_component<Permanent>(aura)) {
        enchanted = global_coordinator.GetComponent<Permanent>(aura).equipped_to.get();
    }
    if (enchanted != 0) out.push_back(enchanted);
}

static void objects_valid(const ObjectSpec &spec, const ResolveCtx &ctx, std::vector<Entity> &out) {
    const MatchCtx match{ctx.controller, ctx.source.lki_entity()};
    for (Entity e : zoned_entities())
        if (permanent_matches_filter(e, spec.filter, match)) out.push_back(e);
}

static void objects_equipped(const ObjectSpec &, const ResolveCtx &ctx, std::vector<Entity> &out) {
    const Entity equipment = ctx.source.get();
    if (equipment == 0 || !global_coordinator.entity_has_component<Permanent>(equipment)) return;
    if (Entity e = global_coordinator.GetComponent<Permanent>(equipment).equipped_to.get()) out.push_back(e);
}

static void players_you(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    out.push_back(ctx.controller);
}

static void players_opponent(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    out.push_back(opponent_of(ctx.controller));
}

static void players_each(const ResolveCtx &, std::vector<Zone::Ownership> &out) {
    out.push_back(active_seat());
    out.push_back(opponent_of(active_seat()));
}

// The nearest player target up the resolving chain (Ability::player_target_for_subs).
static void players_chain_target(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    if (!ctx.ability) return;
    const Entity player = ctx.ability->player_target_for_subs();
    if (player != 0 && global_coordinator.entity_has_component<Player>(player))
        out.push_back(seat_of_player(player));
}

static void players_targeted_controller(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    if (ctx.ability && !ctx.ability->target.empty())
        out.push_back(last_known_controller(ctx.ability->target.lki_entity()));
}

static void players_triggered_activator(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    out.push_back(ctx.triggered_activator);
}

static void players_triggered_player(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    out.push_back(ctx.triggered_player);
}

// The owner of the card that triggered the ability, its source: ownership never changes (CR 108.3).
static void players_triggered_card_owner(const ResolveCtx &ctx, std::vector<Zone::Ownership> &out) {
    const Entity card = ctx.source.lki_entity();
    if (card != 0 && global_coordinator.entity_has_component<Zone>(card))
        out.push_back(global_coordinator.GetComponent<Zone>(card).owner);
}

static bool read_by_change_zone(const AbilityDef &ability) { return ability.category == "ChangeZone"; }
static bool read_by_change_zone_all(const AbilityDef &ability) { return ability.category == "ChangeZoneAll"; }
static bool read_by_copy_permanent(const AbilityDef &ability) { return ability.category == "CopyPermanent"; }
static bool read_by_pump(const AbilityDef &ability) { return ability.category == "Pump"; }

static bool read_by_each_player(const AbilityDef &ability) {
    return ability.category == "Vote" || (ability.category == "ChangeZone" && ability.origin == Zone::HAND &&
                                          ability.destination == Zone::BATTLEFIELD);
}

static bool read_by_no_ability(const AbilityDef &) { return false; }

static const DefinedRow *row_named(const std::string &value) {
    for (const DefinedRow &row : kDefinedRows) {
        if (!row.name) continue;
        const size_t len = std::strlen(row.name);
        const bool prefix = row.name[len - 1] == ' ';
        if (prefix ? value.compare(0, len, row.name) == 0 : value == row.name) return &row;
    }
    return nullptr;
}

static const DefinedRow &row_of(ObjectSpec::Kind kind) {
    for (const DefinedRow &row : kDefinedRows)
        if (row.kind == kind) return row;
    return kDefinedRows[1];
}

// `ability` null for a static.
static bool row_read_by(const DefinedRow &row, const AbilityDef *ability) {
    return ability ? !row.read_by || row.read_by(*ability) : row.statics;
}

// Defined$ Targeted / ParentTarget / Parent: the ability refers to the target its parent chose.
static bool names_parent_target(const ObjectSpec &spec) {
    return row_of(spec.kind).refers == Refers::OBJECT_OR_PLAYER;
}

ResolveCtx ResolveCtx::of(const Ability &ab) {
    ResolveCtx ctx;
    ctx.controller = ab.controller;
    ctx.source = ab.source;
    ctx.ability = &ab;
    ctx.triggered_activator = ab.triggered_activator;
    ctx.triggered_player = ab.triggered_player;
    return ctx;
}

ResolveCtx ResolveCtx::of_static(Zone::Ownership controller, Entity source) {
    ResolveCtx ctx;
    ctx.controller = controller;
    ctx.source = ObjectRef::of(source);
    return ctx;
}

ObjectSpec parse_object_spec(const std::string &value) {
    if (value.empty()) return ObjectSpec();
    const DefinedRow *row = row_named(value);
    if (!row) return ObjectSpec(ObjectSpec::UNREAD);
    ObjectSpec spec(row->kind);
    const size_t len = std::strlen(row->name);
    if (value.size() > len) spec.filter = value.substr(len);
    return spec;
}

ObjectSpec parse_affected_defined(const std::string &value) {
    const DefinedRow *row = row_named(value);
    return row && row->statics ? ObjectSpec(row->kind) : ObjectSpec(ObjectSpec::UNREAD);
}

bool defined_name_read(const AbilityDef &ability, const std::string &value) {
    const DefinedRow *row = row_named(value);
    return row && row_read_by(*row, &ability);
}

std::vector<Entity> resolve_objects(const ObjectSpec &spec, const ResolveCtx &ctx) {
    std::vector<Entity> out;
    const DefinedRow &row = row_of(spec.kind);
    if (row.objects && row_read_by(row, ctx.ability ? ctx.ability->def : nullptr)) row.objects(spec, ctx, out);
    return out;
}

std::vector<Zone::Ownership> resolve_players(const ObjectSpec &spec, const ResolveCtx &ctx) {
    std::vector<Zone::Ownership> out;
    const DefinedRow &row = row_of(spec.kind);
    if (row.players && row_read_by(row, ctx.ability ? ctx.ability->def : nullptr)) row.players(ctx, out);
    out.erase(std::remove(out.begin(), out.end(), Zone::UNKNOWN), out.end());
    return out;
}

bool inherits_parent_target(const ObjectSpec &spec) { return row_of(spec.kind).inherits_target; }

bool names_defined_player(const Ability &ab) {
    const DefinedRow &row = row_of(ab.def->defined.kind);
    return row.refers == Refers::PLAYER && row.players && row_read_by(row, ab.def);
}

Zone::Ownership affected_player(const Ability &ab) {
    if (names_defined_player(ab)) {
        const std::vector<Zone::Ownership> who = resolve_players(ab.def->defined, ResolveCtx::of(ab));
        if (!who.empty()) return who.front();
    }
    if (ab.def->valid_tgts != "N_A" || names_parent_target(ab.def->defined)) {
        const Entity t = ab.target.get();
        if (t != 0 && global_coordinator.entity_has_component<Player>(t)) return seat_of_player(t);
    }
    return ab.controller;
}

std::vector<Entity> affected_objects(const Ability &ab) {
    if (ab.def->valid_tgts != "N_A") {
        std::vector<Entity> out;
        append_chosen_targets(ab, out);
        return out;
    }
    return resolve_objects(ab.def->defined, ResolveCtx::of(ab));
}
