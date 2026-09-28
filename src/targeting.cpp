#include "targeting.h"

#include <set>
#include <string>
#include <vector>

#include "classes/game.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "ecs/coordinator.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/filters.h"
#include "queries/keywords.h"
#include "queries/players.h"
#include "queries/spells.h"
#include "str_util.h"
#include "svar_eval.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// ── Stack-object target matching (TargetType$) ──────────────────────────────
// TargetType$ is a comma-separated list of OR alternatives, each restricting the chosen
// target to a kind of object ON THE STACK: a "Spell[.quals]" alternative matches a spell
// (with optional color / type qualifiers), an "Activated"/"Triggered" alternative matches a
// standalone ability of that kind (CR 113.7). Consign to Memory's
// "Spell.Colorless,Triggered" is one such disjunction. Forward-declared per CLAUDE.md.
static bool stack_spell_alt_matches(const std::string &alt, Entity cand);
static bool stack_ability_alt_matches(const std::string &alt, Entity cand);
static bool target_type_matches_stack_object(const std::string &target_type, Entity cand);
static bool target_has_color_hexproof(Entity cand, Entity source, Zone::Ownership caster);
static bool player_has_protection_from_everything(Entity cand, Zone::Ownership caster);
static Zone::Ownership ability_perspective_player(const Ability &ability);

// Does one "Spell[.quals]" TargetType alternative match a spell on the stack? Checks the
// stack-spell preconditions plus the alternative's own qualifiers: type negations
// (nonCreature / Instant|Sorcery-only), a positive color restriction (.Blue — Red Elemental
// Blast, CR 115.1), and Colorless (Consign to Memory: a spell with no color). `alt` is the
// single alternative (e.g. "Spell.Colorless"). Returns false for a candidate that is not a
// spell on the stack.
static bool stack_spell_alt_matches(const std::string &alt, Entity cand) {
    if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
    if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
    if (!global_coordinator.entity_has_component<Spell>(cand)) return false;
    bool non_creature_only = alt.find("nonCreature") != std::string::npos;
    bool instant_sorcery_only =
        (alt.find("Instant") != std::string::npos || alt.find("Sorcery") != std::string::npos) &&
        alt.find("Creature") == std::string::npos;
    if ((non_creature_only || instant_sorcery_only) &&
        global_coordinator.entity_has_component<CardData>(cand)) {
        auto &cd = global_coordinator.GetComponent<CardData>(cand);
        bool is_creature = false, is_instant = false, is_sorcery = false;
        for (auto &t : cd.types) {
            if (t.name == "Creature") is_creature = true;
            if (t.name == "Instant") is_instant = true;
            if (t.name == "Sorcery") is_sorcery = true;
        }
        if (non_creature_only && is_creature) return false;
        if (instant_sorcery_only && !is_instant && !is_sorcery) return false;
    }
    // Positive color restriction (.Blue — Red Elemental Blast, CR 115.1).
    if (global_coordinator.entity_has_component<CardData>(cand) &&
        !color_set_passes(alt, effective_colors(cand)))
        return false;
    // Colorless restriction (Consign to Memory): the spell must have NO color (CR 105.2c).
    if (alt.find("Colorless") != std::string::npos && !is_colorless(cand))
        return false;
    return true;
}

// Does one "Activated"/"Triggered" TargetType alternative match a standalone ability on the
// stack (Stifle, Consign to Memory)? Spells have a Spell component and are excluded.
static bool stack_ability_alt_matches(const std::string &alt, Entity cand) {
    if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
    if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
    if (global_coordinator.entity_has_component<Spell>(cand)) return false;  // spells aren't abilities
    if (!global_coordinator.entity_has_component<Ability>(cand)) return false;
    auto &ab = global_coordinator.GetComponent<Ability>(cand);
    if (alt.find("Activated") != std::string::npos && ab.def->ability_type == AbilityDef::ACTIVATED)
        return true;
    if (alt.find("Triggered") != std::string::npos && ab.def->ability_type == AbilityDef::TRIGGERED)
        return true;
    return false;
}

// TargetType$ is an OR list of stack-object alternatives — split on commas and accept the
// candidate if ANY alternative matches it (CR 115.1 target restrictions are satisfied by any
// one named kind). Drives counterspells (Spell), Stifle (Activated,Triggered) and Consign to
// Memory (Spell.Colorless,Triggered) off one matcher.
static bool target_type_matches_stack_object(const std::string &target_type, Entity cand) {
    for (const std::string &alt : split(target_type, ',', /*skip_empty=*/true)) {
        bool is_ability_alt = (alt.find("Activated") != std::string::npos ||
                               alt.find("Triggered") != std::string::npos);
        bool is_spell_alt = (alt.find("Spell") != std::string::npos);
        if (is_ability_alt && stack_ability_alt_matches(alt, cand)) return true;
        if (is_spell_alt && stack_spell_alt_matches(alt, cand)) return true;
    }
    return false;
}

// "Hexproof from <color>" (CR 702.11e): a candidate protected by a turn-long
// cur_game.resolved_effects.hexproof_from_colors_this_turn grant can't be targeted by a spell/ability an OPPONENT
// of the protected player controls whose SOURCE is one of the granted colors. Covers both the
// protected player (the player object) and any permanent that player controls. `source` is the
// targeting object (a spell card on the stack, or an ability's source permanent) whose
// effective colors give the color of the spell/ability for the comparison; `caster` is its
// controller. Returns true when the candidate is protected (so the target is illegal).
static bool target_has_color_hexproof(Entity cand, Entity source, Zone::Ownership caster) {
    if (cur_game.resolved_effects.hexproof_from_colors_this_turn.empty()) return false;
    const std::set<Colors> src_colors = effective_colors(source);
    for (const auto &h : cur_game.resolved_effects.hexproof_from_colors_this_turn) {
        // Only protects against an opponent's spell/ability (two-player: caster != protected player).
        if (caster == h.player) continue;
        // The candidate must be the protected player, or a permanent that player controls.
        bool is_protected_player = (cand == get_player_entity(h.player));
        bool is_protected_perm = global_coordinator.entity_has_component<Permanent>(cand) &&
                                 is_battlefield_permanent(cand, h.player);
        if (!is_protected_player && !is_protected_perm) continue;
        // The source must be one of the granted colors.
        for (Colors c : h.colors)
            if (src_colors.count(c)) return true;
    }
    return false;
}

// "Protection from everything" for a player (CR 702.16; The One Ring). A player covered by a
// cur_game.resolved_effects.player_protection_from_everything grant can't be the target of a spell/ability an
// OPPONENT controls. `caster` is the targeting object's controller. Returns true when the
// candidate is the protected player and the targeting object belongs to their opponent.
static bool player_has_protection_from_everything(Entity cand, Zone::Ownership caster) {
    if (cur_game.resolved_effects.player_protection_from_everything.empty()) return false;
    for (const auto &p : cur_game.resolved_effects.player_protection_from_everything) {
        if (caster == p.player) continue;  // own spells/abilities can still target the player
        if (cand == get_player_entity(p.player)) return true;
    }
    return false;
}

// Single source of truth for target legality (see header). build_valid_targets
// enumerates candidates and filters them through this; targets_still_legal re-runs the
// chosen target(s) through it at resolution. Keeping both on one predicate is what
// prevents the enumeration and re-verification rules from drifting apart.
bool is_legal_target(const Ability &ab, Entity cand, Zone::Ownership caster) {
    if (cand == 0) return false;

    // Hexproof from <color> (Veil of Summer) — applies to players and permanents alike, so it is
    // checked up front before the type-specific branches below.
    if (target_has_color_hexproof(cand, ab.source.lki_entity(), caster)) return false;

    // Protection from everything for a player (The One Ring) — the protected player can't be
    // targeted by an opponent's spell/ability (CR 702.16e). Checked up front like hexproof.
    if (player_has_protection_from_everything(cand, caster)) return false;

    // ValidTgts$ ...Other (e.g. Solitude/Flickerwisp "another"/"other" target): the source
    // of the ability cannot be chosen as its own target (CR 115.1; "other" is a target
    // restriction). Enforced uniformly here so it applies to every target type. Match
    // "Other" only as a complete dot/plus-delimited qualifier token (Creature.Other,
    // Permanent.Other+nonLand), never as a substring of a longer subtype/name (so a future
    // "Brotherhood"/"Otherworldly" token can't spuriously forbid self-targeting).
    if (cand == ab.source.get()) {
        for (size_t p = ab.def->valid_tgts.find("Other"); p != std::string::npos;
             p = ab.def->valid_tgts.find("Other", p + 1)) {
            bool left_ok = (p > 0) && (ab.def->valid_tgts[p - 1] == '.' || ab.def->valid_tgts[p - 1] == '+');
            size_t end = p + 5;  // length of "Other"
            bool right_ok = (end == ab.def->valid_tgts.size()) ||
                            ab.def->valid_tgts[end] == '.' || ab.def->valid_tgts[end] == '+';
            if (left_ok && right_ok) return false;
        }
    }

    // NOTE: Pyroblast/Hydroblast (ValidTgts$ Card + ConditionPresent$ <type>.<Color>)
    // intentionally do NOT restrict target legality by color — they may target any
    // spell/permanent and their counter/destroy effect is conditional on the target's
    // color (enforced in effects::counter / effects::destroy via
    // target_color_condition_met). By contrast Red Elemental Blast (ValidTgts$ Card.Blue /
    // Permanent.Blue) bakes blue into the target restriction itself, enforced below through
    // the shared filter evaluator on both the stack-spell and battlefield-permanent paths
    // (CR 115.1: target restrictions are checked when chosen, against the candidate).

    const std::string &vt = ab.def->valid_tgts;

    // Stack-object targets (counterspells, Stifle, Consign to Memory): TargetType$ names one or
    // more kinds of stack object — "Spell[.quals]", "Activated", "Triggered" — as an OR list.
    if (!ab.def->target_type.empty() &&
        (ab.def->target_type.find("Spell") != std::string::npos ||
         ab.def->target_type.find("Activated") != std::string::npos ||
         ab.def->target_type.find("Triggered") != std::string::npos)) {
        if (!target_type_matches_stack_object(ab.def->target_type, cand)) return false;
        // A bare "TargetType$ Spell" carries its restriction in ValidTgts$, written against the
        // CARD on the stack — Red Elemental Blast "Card.Blue", Force of Negation
        // "Card.nonCreature", Flusterstorm "Instant,Sorcery" (CR 115.1: the restriction is part
        // of choosing the target, and re-checked at resolution via targets_still_legal). Match it
        // through the shared comma-OR card filter. A standalone ability entity on the stack (an
        // Activated/Triggered alternative the script names explicitly — Stifle, Consign to
        // Memory) is not a card, so a card-shaped ValidTgts ("Card,Emblem") never filters it.
        // A static numeric cmc qualifier (Spell Snare: ValidTgts$ Card.cmcEQ2) is deferred by
        // the filter evaluator to ctx.cmc_bound, so seed it here or the comparator is a no-op
        // and the counterspell would target any spell.
        MatchCtx spell_ctx{caster, ab.source.lki_entity()};
        extract_static_cmc_bound(vt, spell_ctx);
        if (vt != "N_A" && !vt.empty() &&
            global_coordinator.entity_has_component<Spell>(cand) &&
            !card_matches_any(cand, vt, spell_ctx))
            return false;
        return true;
    }

    // Card in a non-battlefield zone (e.g. Faerie Macabre targeting graveyard cards)
    if (vt == "Card" && ab.def->kind == EffectKind::ChangeZone && ab.def->origin == Zone::GRAVEYARD) {
        return global_coordinator.entity_has_component<Zone>(cand) &&
               global_coordinator.GetComponent<Zone>(cand).location == Zone::GRAVEYARD;
    }

    int cmc_le = -1;
    {
        size_t cmc_pos = vt.find("cmcLE");
        if (cmc_pos != std::string::npos) {
            std::string bound = vt.substr(cmc_pos + 5);
            // Trim to the leading token (stop at the next '.'/'+' qualifier).
            size_t end = bound.find_first_of(".+");
            if (end != std::string::npos) bound = bound.substr(0, end);
            if (!bound.empty() && std::isdigit(static_cast<unsigned char>(bound[0])))
                cmc_le = std::stoi(bound);
            else
                // cmcLEX (Kozilek's Command): the bound is the X paid at cast time.
                cmc_le = current_x_paid();
        }
    }
    // The ValidTgts spec in the shared matcher's grammar: "YouDontCtrl" is Forge's spelling of
    // OppCtrl (a permanent you don't control), and "Any" as a permanent target is "any creature
    // or planeswalker" (CR 115.4 / 306.7; players are handled in their own branch below).
    MatchCtx ctx;
    ctx.controller = caster;
    ctx.source = ab.source.lki_entity();
    if (ab.targeted_player != 0) ctx.targeted_player = seat_of_player(ab.targeted_player);
    // The mana-value bound (cmcLE<n>/cmcLEX, e.g. Abrupt Decay's cmcLE3) is parsed above into
    // cmc_le; feed it to the evaluator so the bound is actually enforced.
    if (cmc_le >= 0) { ctx.cmc_bound = cmc_le; ctx.cmc_op = "LE"; }
    std::string spec = (vt == "Any") ? std::string("Creature,Planeswalker") : vt;
    for (size_t pos = spec.find("YouDontCtrl"); pos != std::string::npos;
         pos = spec.find("YouDontCtrl", pos))
        spec.replace(pos, std::string("YouDontCtrl").size(), "OppCtrl");

    // Card in a graveyard targeted by a ChangeZone with a type filter. Covers both
    // graveyard→non-battlefield moves (e.g. Life from the Loam: ValidTgts$ Land.YouCtrl,
    // Origin$ Graveyard, Destination$ Hand) and targeted reanimation graveyard→battlefield
    // (e.g. Lorehold Charm's "return target artifact or creature with mana value 2 or less
    // from your graveyard to the battlefield"). In every case the target is the card sitting
    // in the graveyard, matched there by its card characteristics regardless of destination; a
    // card's controller there is its owner (CR 108.4a), so YouCtrl/OppCtrl mean YouOwn/OppOwn.
    if (ab.def->target_in_graveyard ||
        (ab.def->kind == EffectKind::ChangeZone && ab.def->origin == Zone::GRAVEYARD)) {
        if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
        if (global_coordinator.GetComponent<Zone>(cand).location != Zone::GRAVEYARD) return false;
        return card_matches_filter(cand, owner_relative_filter(spec), ctx);
    }

    // Player target
    if (global_coordinator.entity_has_component<Player>(cand))
        return player_matches_target_spec(vt, cand, caster);

    // Battlefield permanent target (phased-out permanents can't be targeted, 702.26b)
    if (!is_battlefield_permanent(cand)) return false;

    // Match the ValidTgts spec against the permanent through the shared filter evaluator
    // (queries/filters), with the normalized spec and mana-value bound built above.
    if (!permanent_matches_filter(cand, spec, ctx)) return false;

    // Protection (CR 702.16e): a creature with protection from the source's color/quality can't
    // be targeted by it. The filter evaluator doesn't model protection, so check it separately.
    if (global_coordinator.entity_has_component<Creature>(cand)) {
        const Creature &cand_cr = global_coordinator.GetComponent<Creature>(cand);
        if (has_protection_from(cand_cr, ab.source.lki_entity())) return false;
        // Protection from colored spells (Emrakul: K:Protection:Spell.nonColorless, CR 702.16b/e):
        // a creature with this protection can't be the target of a SPELL that is one or more
        // colors. The "is a spell" half is known here from this Ability's type (a SPELL ability,
        // not an activated/triggered ability), and the "is colored" half from the source's
        // effective colors — so a colorless spell, or any ability, may still target it.
        if (ab.def->ability_type == AbilityDef::SPELL && has_protection_from_colored_spells(cand_cr) &&
            !is_colorless(ab.source.lki_entity()))
            return false;
    }

    // Shroud (CR 702.18e) and Hexproof (CR 702.11b): targeting restrictions read off the
    // candidate's EFFECTIVE keyword set (printed + granted via Pump/effects/keyword counter,
    // through permanent_has_keyword), so a creature granted Shroud (Sylvan Safekeeper) is
    // untargetable while the grant lasts.
    //   • Shroud: can't be the target of ANY spell or ability (yours OR opponents').
    //   • Hexproof: can't be the target of spells/abilities an OPPONENT controls — legal for
    //     the controller of the targeting spell/ability, illegal for the other player. `caster`
    //     is the controller of this ability/spell; the candidate's controller is its Permanent.
    if (permanent_has_keyword(cand, "Shroud")) return false;
    if (permanent_has_keyword(cand, "Hexproof") &&
        global_coordinator.entity_has_component<Permanent>(cand) &&
        global_coordinator.GetComponent<Permanent>(cand).controller != caster)
        return false;
    return true;
}

bool targets_still_legal(const Ability &ab) {
    // Optional targeting: no target chosen is valid
    if (ab.target.empty() && ab.targets.empty() && ab.target_min == 0) return true;

    // Multi-target abilities (target_max > 1) populate `targets`. CR 608.2b: the spell or
    // ability is countered on resolution only if ALL of its targets are illegal; with at
    // least one still-legal target it resolves, affecting only the legal ones (resolve_ability()
    // prunes the illegal targets before dispatching to the effect handler). A target that
    // changed zones since selection is a new object (CR 400.7) and is illegal even if a same-id
    // incarnation now looks legal: its ObjectRef no longer resolves.
    if (!ab.targets.empty()) {
        for (const ObjectRef &t : ab.targets)
            if (is_legal_target(ab, t.get(), ab.controller)) return true;
        return false;
    }

    return is_legal_target(ab, ab.target.get(), ab.controller);
}

std::vector<Entity> build_valid_targets(const Ability &ability, std::shared_ptr<Orderer> orderer,
                                        Zone::Ownership priority_player) {
    std::vector<Entity> valid_targets;
    const std::string &vt = ability.def->valid_tgts;

    // Stack targets: spells (counterspells) or standalone abilities (Stifle)
    if (ability.def->target_type == "Spell" ||
        ability.def->target_type.find("Activated") != std::string::npos ||
        ability.def->target_type.find("Triggered") != std::string::npos) {
        for (auto e : orderer->get_stack()) {
            // A spell/ability can't target itself (CR 115.5) — a spell choosing its targets
            // as it is cast (CR 601.2a/c), a modal spell (Pyroblast/Hydroblast) that picks its
            // target at resolution, or an activated ability being activated (CR 602.2a).
            if (e == ability.source.get()) continue;
            if (cur_game.pending.activation.active && e == cur_game.pending.activation.stack_entity)
                continue;
            if (is_legal_target(ability, e, priority_player)) valid_targets.push_back(e);
        }
        return valid_targets;
    }

    Zone::Ownership opp = opponent_of(priority_player);

    // Target cards in a graveyard (e.g. Faerie Macabre targeting any graveyard card,
    // Life from the Loam targeting Land.YouCtrl, or targeted reanimation graveyard→
    // battlefield like Lorehold Charm): opponent's graveyard first, then own.
    // is_legal_target applies the type/owner/MV filter, so YouOwn effects only keep the
    // caster's own cards. The destination is irrelevant to where the candidate sits, so
    // a graveyard-origin ChangeZone enumerates the graveyard regardless of destination.
    // target_in_graveyard covers spells that target a graveyard card via a non-ChangeZone
    // vehicle (Surgical Extraction's SP$ Pump with TgtZone$ Graveyard).
    if (ability.def->target_in_graveyard ||
        (ability.def->kind == EffectKind::ChangeZone && ability.def->origin == Zone::GRAVEYARD)) {
        for (int pass = 0; pass < 2; pass++) {
            Zone::Ownership slot_owner = (pass == 0) ? opp : priority_player;
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                if (global_coordinator.GetComponent<Zone>(e).owner != slot_owner) continue;
                if (is_legal_target(ability, e, priority_player)) valid_targets.push_back(e);
            }
        }
        return valid_targets;
    }

    // Players: opponent first, self second (is_legal_target applies the spec's player clause)
    if (target_spec_names_players(vt)) {
        for (Zone::Ownership seat : {opp, priority_player})
            if (is_legal_target(ability, get_player_entity(seat), priority_player))
                valid_targets.push_back(get_player_entity(seat));
    }

    // Permanents: two passes — opponent's first, then own (entity-ID order within each group)
    for (int pass = 0; pass < 2; pass++) {
        Zone::Ownership slot_owner = (pass == 0) ? opp : priority_player;
        for (auto entity : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
            if (global_coordinator.GetComponent<Permanent>(entity).controller != slot_owner) continue;
            if (is_legal_target(ability, entity, priority_player)) valid_targets.push_back(entity);
        }
    }
    return valid_targets;
}

// Perspective player for an ability's target search. Ownership-restricted targets
// (.YouOwn/.YouCtrl/.OppOwn — e.g. Emry's "target artifact card in YOUR graveyard") are
// relative to the activating/controlling player, so the existence check must use that player
// rather than a hardcoded placeholder. Derive it from the ability's source: a battlefield
// permanent's controller, else its owning zone, else the ability's stored controller.
static Zone::Ownership ability_perspective_player(const Ability &ability) {
    Entity src = ability.source.get();
    if (src != 0) {
        if (global_coordinator.entity_has_component<Permanent>(src))
            return global_coordinator.GetComponent<Permanent>(src).controller;
        if (global_coordinator.entity_has_component<Zone>(src))
            return global_coordinator.GetComponent<Zone>(src).owner;
    }
    return ability.controller;
}

bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer) {
    if (is_modal(ability))
        return has_choosable_modes(ability, orderer, ability_perspective_player(ability), false);
    if (ability.def->valid_tgts == "N_A") return true;
    // Ordering doesn't affect existence for symmetric targets, but ownership-restricted
    // targets must be evaluated from the controlling player's perspective (see above), or a
    // ".YouOwn" ability could be offered with no legal target and crash on an empty target menu.
    Zone::Ownership perspective = ability_perspective_player(ability);
    // optional targeting always has "legal targets"
    if (effective_target_min(ability, perspective, orderer, /*x_announced=*/false) <= 0) return true;
    return !build_valid_targets(ability, orderer, perspective).empty();
}

int effective_target_min(const Ability &ab, Zone::Ownership perspective,
                         std::shared_ptr<Orderer> orderer, bool x_announced) {
    if (ab.def->target_min_from_xpaid) return x_announced ? current_x_paid() : 0;
    if (!ab.def->target_min_count_expr.empty())
        return static_cast<int>(evaluate_amount(ab.def->target_min_count_expr, perspective, ab.source.lki_entity()));
    return ab.target_min;
}

bool is_modal(const Ability &ab) { return ab.def->kind == EffectKind::Charm; }

bool mode_choosable(const Ability &modal, size_t idx, std::shared_ptr<Orderer> orderer,
                    Zone::Ownership chooser, bool x_announced) {
    const Ability &mode = modal.charm_choices[idx];
    if (mode.def->valid_tgts == "N_A") return true;
    // Probed as the modal object's own: its source and controller decide protection and the
    // YouCtrl/OppCtrl perspective of the mode's targets.
    Ability probe = mode;
    probe.source = modal.source;
    probe.controller = chooser;
    if (effective_target_min(probe, chooser, orderer, x_announced) <= 0) return true;
    return !build_valid_targets(probe, orderer, chooser).empty();
}

bool has_choosable_modes(const Ability &modal, std::shared_ptr<Orderer> orderer,
                         Zone::Ownership chooser, bool x_announced) {
    const int needed = modal.def->charm_num < 1 ? 1 : modal.def->charm_num;
    int choosable = 0;
    for (size_t i = 0; i < modal.charm_choices.size(); i++)
        if (mode_choosable(modal, i, orderer, chooser, x_announced) && ++choosable >= needed)
            return true;
    return false;
}
