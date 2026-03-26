#include "error.h"
#include "cli_output.h"

#ifndef NDEBUG
#include <execinfo.h>
#include <cstdio>
#include <cstdlib>
#include "ecs/coordinator.h"
#include "components/carddata.h"
#include "components/zone.h"
#include "components/permanent.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/player.h"
#include "components/ability.h"
#include "components/spell.h"
#include "components/effect.h"
#include "components/token.h"
#include "components/color_identity.h"
#include "classes/colors.h"

extern Coordinator global_coordinator;

static const char *zone_str(Zone::ZoneValue z);
static const char *owner_str(Zone::Ownership o);
static const char *ability_type_str(Ability::AbilityType t);
static void print_backtrace();

static const char *zone_str(Zone::ZoneValue z) {
    switch (z) {
    case Zone::LIBRARY:     return "LIBRARY";
    case Zone::BATTLEFIELD: return "BATTLEFIELD";
    case Zone::HAND:        return "HAND";
    case Zone::STACK:       return "STACK";
    case Zone::GRAVEYARD:   return "GRAVEYARD";
    case Zone::EXILE:       return "EXILE";
    case Zone::SIDEBOARD:   return "SIDEBOARD";
    }
    return "?";
}

static const char *owner_str(Zone::Ownership o) {
    switch (o) {
    case Zone::UNKNOWN:  return "UNKNOWN";
    case Zone::PLAYER_A: return "PLAYER_A";
    case Zone::PLAYER_B: return "PLAYER_B";
    }
    return "?";
}

static const char *ability_type_str(Ability::AbilityType t) {
    switch (t) {
    case Ability::TRIGGERED: return "TRIGGERED";
    case Ability::ACTIVATED: return "ACTIVATED";
    case Ability::SPELL:     return "SPELL";
    }
    return "?";
}

void dump_entity(Entity e) {
    fprintf(stderr, "=== dump_entity(%u) ===\n", e);

    if (global_coordinator.entity_has_component<CardData>(e)) {
        auto &c = global_coordinator.GetComponent<CardData>(e);
        fprintf(stderr, "  CardData:\n");
        fprintf(stderr, "    uid=%s  name=%s\n", c.uid.c_str(), c.name.c_str());
        fprintf(stderr, "    oracle_text=%s\n", c.oracle_text.c_str());
        fprintf(stderr, "    power=%u  toughness=%u\n", c.power, c.toughness);
        fprintf(stderr, "    mana_cost:");
        for (auto col : c.mana_cost) fprintf(stderr, " %s", mana_symbol(col).c_str());
        fprintf(stderr, "\n");
        fprintf(stderr, "    types:");
        for (auto &t : c.types) fprintf(stderr, " %s", t.name.c_str());
        fprintf(stderr, "\n");
        fprintf(stderr, "    keywords:");
        for (auto &k : c.keywords) fprintf(stderr, " %s", k.c_str());
        fprintf(stderr, "\n");
        fprintf(stderr, "    abilities: %zu\n", c.abilities.size());
        fprintf(stderr, "    has_delve=%d  has_x_cost=%d  is_equipment=%d\n",
                c.has_delve, c.has_x_cost, c.is_equipment);
        if (c.backside) fprintf(stderr, "    backside=%s\n", c.backside->name.c_str());
    }

    if (global_coordinator.entity_has_component<Zone>(e)) {
        auto &z = global_coordinator.GetComponent<Zone>(e);
        fprintf(stderr, "  Zone:\n");
        fprintf(stderr, "    location=%s  distance_from_top=%zu\n",
                zone_str(z.location), z.distance_from_top);
        fprintf(stderr, "    owner=%s  controller=%s\n",
                owner_str(z.owner), owner_str(z.controller));
    }

    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &p = global_coordinator.GetComponent<Permanent>(e);
        fprintf(stderr, "  Permanent:\n");
        fprintf(stderr, "    name=%s  controller=%s\n", p.name.c_str(), owner_str(p.controller));
        fprintf(stderr, "    is_token=%d  is_tapped=%d  has_summoning_sickness=%d\n",
                p.is_token, p.is_tapped, p.has_summoning_sickness);
        fprintf(stderr, "    transformed=%d  is_phased_out=%d\n", p.transformed, p.is_phased_out);
        fprintf(stderr, "    equipped_to=%u  equipped_by=%u\n", p.equipped_to, p.equipped_by);
        fprintf(stderr, "    abilities: %zu  static_abilities: %zu\n",
                p.abilities.size(), p.static_abilities.size());
        fprintf(stderr, "    types:");
        for (auto &t : p.types) fprintf(stderr, " %s", t.name.c_str());
        fprintf(stderr, "\n");
    }

    if (global_coordinator.entity_has_component<Creature>(e)) {
        auto &c = global_coordinator.GetComponent<Creature>(e);
        fprintf(stderr, "  Creature:\n");
        fprintf(stderr, "    power=%u  toughness=%u\n", c.power, c.toughness);
        fprintf(stderr, "    is_attacking=%d  attack_target=%u\n", c.is_attacking, c.attack_target);
        fprintf(stderr, "    is_blocking=%d  blocking_target=%u\n", c.is_blocking, c.blocking_target);
        fprintf(stderr, "    must_attack=%d  plus_one_counters=%d  prowess_bonus=%d\n",
                c.must_attack, c.plus_one_counters, c.prowess_bonus);
        fprintf(stderr, "    keywords:");
        for (auto &k : c.keywords) fprintf(stderr, " %s", k.c_str());
        fprintf(stderr, "\n");
    }

    if (global_coordinator.entity_has_component<Damage>(e)) {
        auto &d = global_coordinator.GetComponent<Damage>(e);
        fprintf(stderr, "  Damage:\n");
        fprintf(stderr, "    damage_counters=%zu\n", d.damage_counters);
    }

    if (global_coordinator.entity_has_component<Player>(e)) {
        auto &p = global_coordinator.GetComponent<Player>(e);
        fprintf(stderr, "  Player:\n");
        fprintf(stderr, "    life_total=%d  otp=%d\n", p.life_total, p.otp);
        fprintf(stderr, "    poison=%u  energy=%u  lands_played=%u\n",
                p.poison_counters, p.energy_counters, p.lands_played_this_turn);
        fprintf(stderr, "    spells_cast_turn=%zu  spells_cast_game=%zu\n",
                p.spells_cast_this_turn, p.spells_cast_this_game);
        fprintf(stderr, "    mana_pool:");
        for (auto col : p.mana) fprintf(stderr, " %s", mana_symbol(col).c_str());
        fprintf(stderr, "\n");
    }

    if (global_coordinator.entity_has_component<Ability>(e)) {
        auto &a = global_coordinator.GetComponent<Ability>(e);
        fprintf(stderr, "  Ability:\n");
        fprintf(stderr, "    type=%s  category=%s\n",
                ability_type_str(a.ability_type), a.category.c_str());
        fprintf(stderr, "    source=%u  target=%u  controller=%s\n",
                a.source, a.target, owner_str(a.controller));
        fprintf(stderr, "    amount=%zu  color=%s\n", a.amount, mana_symbol(a.color).c_str());
        fprintf(stderr, "    valid_tgts=%s  tap_cost=%d  sac_self=%d  life_cost=%d\n",
                a.valid_tgts.c_str(), a.tap_cost, a.sac_self, a.life_cost);
        fprintf(stderr, "    subabilities: %zu\n", a.subabilities.size());
    }

    if (global_coordinator.entity_has_component<Spell>(e)) {
        auto &s = global_coordinator.GetComponent<Spell>(e);
        fprintf(stderr, "  Spell:\n");
        fprintf(stderr, "    caster=%s\n", owner_str(s.caster));
    }

    if (global_coordinator.entity_has_component<Effect>(e)) {
        auto &ef = global_coordinator.GetComponent<Effect>(e);
        fprintf(stderr, "  Effect:\n");
        fprintf(stderr, "    category=%s  amount=%d  replacements=%zu\n",
                ef.category.c_str(), ef.amount, ef.replacements.size());
    }

    if (global_coordinator.entity_has_component<Token>(e)) {
        auto &t = global_coordinator.GetComponent<Token>(e);
        fprintf(stderr, "  Token:\n");
        fprintf(stderr, "    name=%s  power=%u  toughness=%u\n",
                t.name.c_str(), t.power, t.toughness);
        fprintf(stderr, "    types:");
        for (auto &ty : t.types) fprintf(stderr, " %s", ty.name.c_str());
        fprintf(stderr, "\n");
    }

    if (global_coordinator.entity_has_component<ColorIdentity>(e)) {
        auto &ci = global_coordinator.GetComponent<ColorIdentity>(e);
        fprintf(stderr, "  ColorIdentity:");
        for (auto col : ci.colors) fprintf(stderr, " %s", mana_symbol(col).c_str());
        fprintf(stderr, "\n");
    }

    fprintf(stderr, "=== end dump_entity(%u) ===\n", e);
}

static void print_backtrace() {
    void *frames[64];
    int n = backtrace(frames, 64);
    char **syms = backtrace_symbols(frames, n);
    if (!syms) return;
    fprintf(stderr, "Backtrace:\n");
    for (int i = 1; i < n; ++i) // skip frame 0 (print_backtrace itself)
        fprintf(stderr, "  %s\n", syms[i]);
    free(syms);
}
#endif

void warning(std::string err) {
#ifndef NDEBUG
    cli_warning(err);
#endif
}

void non_fatal_error(std::string err) {
    cli_error(err);
#ifndef NDEBUG
    print_backtrace();
#endif
}

void fatal_error(std::string err) {
#ifndef NDEBUG
    print_backtrace();
#endif
    cli_fatal_error(err);
}
