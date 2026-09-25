#include "effects.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../game_queries.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../stable_rng.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static Zone::ZoneValue dig_chosen_destination(const Ability &ab);
static bool dig_chosen_on_bottom(const Ability &ab);
static Zone::ZoneValue dig_rest_destination(const Ability &ab);
static bool dig_rest_on_bottom(const Ability &ab);

// Where the chosen cards go: DestinationZone$, else the hand.
static Zone::ZoneValue dig_chosen_destination(const Ability &ab) {
    return ab.dig_destination >= 0 ? static_cast<Zone::ZoneValue>(ab.dig_destination) : Zone::HAND;
}

// Whether chosen cards put into a library go on the bottom (LibraryPosition$ 0 = top). Only
// meaningful with DestinationZone$ set.
static bool dig_chosen_on_bottom(const Ability &ab) {
    return ab.dig_destination >= 0 && ab.dig_library_position != 0;
}

// Where the unchosen rest go: DestinationZone2$, else the library.
static Zone::ZoneValue dig_rest_destination(const Ability &ab) {
    return ab.dig_rest_destination >= 0 ? static_cast<Zone::ZoneValue>(ab.dig_rest_destination)
                                        : Zone::LIBRARY;
}

// Whether the unchosen rest go on the bottom of the library (LibraryPosition2$ 0 keeps them on
// top — Fateseal).
static bool dig_rest_on_bottom(const Ability &ab) { return ab.dig_rest_library_position != 0; }

namespace effects {

HandlerResult dig(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    // Look at top N cards, player picks one matching filter, rest go to bottom.
    // When the ability targets a player (Fateseal, e.g. Jace +2), the dug library is
    // the TARGET player's, not the controller's.
    Zone::Ownership dig_owner = ab.controller;
    if (ab.target != 0 && global_coordinator.entity_has_component<Player>(ab.target))
        dig_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    // The LOOKER (who sees the cards and makes the choices) is always the ability's controller.
    // For a fateseal on an opponent's library (Jace +2) that differs from dig_owner: the owner
    // must NOT learn any card placed back on top, and private card-name logs are pinned to the
    // looker and redacted from the owner.
    Zone::Ownership looker = ab.controller;
    bool owner_sees = (dig_owner == looker);
    // Narrative actor: the ability's controller when the dug player is a chosen target (Jace
    // +2 looks at the target's library), else the dug player acting on their own library
    // (Goblin Guide: the defending player reveals and takes the card). Zones are the dig owner's
    // ("their" when the actor owns them, else "Player B's").
    Zone::Ownership actor = (ab.valid_tgts != "N_A") ? looker : dig_owner;
    const std::string actor_name = player_name(actor);
    const std::string owner_poss = owner_possessive(actor, dig_owner);

    // The revealed slice, the filtered pool, and the resolved take count are
    // computed ONCE (frozen) and persist in the frame rt so a suspended pick
    // resumes against the identical pool; the whole slice is pinned against
    // determinize by pinned_entities() (revealed cards must stay in place for
    // the pool AND the to-bottom epilogue below).
    DigRt local_rt;
    DigRt &rt = ctx.can_suspend() ? ctx.rt<DigRt>() : local_rt;
    if (!rt.init) {
        // Resolve dynamic dig count (e.g. Count$Devotion.Blue)
        size_t effective_dig_num = ab.dig_num;
        if (!ab.dig_num_expr.empty()) {
            effective_dig_num = evaluate_dynamic_amount(ab.dig_num_expr, dig_owner, orderer, 0);
        }
        rt.lib = orderer->get_library_top(dig_owner, effective_dig_num);

        // Parse change_valid filters (comma-separated "Card.Creature,Card.Land" etc.)
        std::vector<std::string> filters;
        if (!ab.change_valid.empty()) {
            size_t fp = 0;
            while (true) {
                size_t comma = ab.change_valid.find(',', fp);
                if (comma == std::string::npos) {
                    filters.push_back(ab.change_valid.substr(fp));
                    break;
                }
                filters.push_back(ab.change_valid.substr(fp, comma - fp));
                fp = comma + 1;
            }
        }

        // A filter like "Creature.cmcLEX" or "Card.cmcLEX" carries a mana-value bound X, resolved
        // from dynamic_amount_expr (e.g. Birthing Ritual: 1 + sacrificed creature's mana value).
        bool has_cmc_le = !ab.change_valid.empty() && ab.change_valid.find("cmcLE") != std::string::npos;
        int cmc_threshold = 0;
        if (has_cmc_le && !ab.dynamic_amount_expr.empty())
            cmc_threshold = static_cast<int>(evaluate_dynamic_amount(ab.dynamic_amount_expr, dig_owner, orderer, ab.target));

        // Filter matching cards. A filter is dot-separated: "Card" is a type wildcard, a "cmc.."
        // token is a mana-value bound (handled above), and any other token is the required type
        // (so both "Card.Creature" and "Creature.cmcLEX" name the Creature type).
        std::vector<Entity> matching;
        for (auto e : rt.lib) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool card_matches = filters.empty();
            for (auto &f : filters) {
                std::string want_type;
                size_t start = 0;
                while (start <= f.size()) {
                    size_t dot = f.find('.', start);
                    std::string part = f.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
                    if (!part.empty() && part != "Card" && part.rfind("cmc", 0) != 0)
                        want_type = part;
                    if (dot == std::string::npos) break;
                    start = dot + 1;
                }
                bool type_ok = want_type.empty();
                // "Permanent" is not a printed type name — it's the permanent-card-type class
                // (CR 110.4a). Malevolent Rumble's ChangeValid$ Permanent matches any permanent
                // card (artifact/creature/enchantment/land/planeswalker/battle).
                if (!type_ok && want_type == "Permanent")
                    type_ok = is_permanent_card(cd);
                else if (!type_ok)
                    for (auto &t : cd.types)
                        if (t.name == want_type) { type_ok = true; break; }
                if (type_ok) { card_matches = true; break; }
            }
            if (!card_matches) continue;
            // Apply the mana-value bound if present (cmc <= threshold).
            if (has_cmc_le && card_mana_value(cd) > cmc_threshold) continue;
            matching.push_back(e);
        }

        game_log("%s looks at the top %zu card(s) of %s library.\n", actor_name.c_str(), rt.lib.size(),
                 owner_poss.c_str());

        // Reveal$ True (Goblin Guide): the looked-at cards are shown to ALL players. Log the
        // reveal publicly (visible to both seats, not redacted) and record it in the belief-state
        // multi-hot so the non-owner's observation carries the revealed identity, regardless of
        // where each card subsequently goes (hand if it matches, else back on top).
        if (ab.dig_reveal) {
            for (auto e : rt.lib) {
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                game_log("%s reveals %s from the top of %s library.\n", actor_name.c_str(), cd.name.c_str(),
                         owner_poss.c_str());
                mark_card_revealed(e, dig_owner);
            }
        }

        // How many of the looked-at cards may be taken (default 1). A conditional
        // ChangeNum$ (Flow State) raises this to its true-value when the summed
        // graveyard counts satisfy the compare; a plain numeric ChangeNum$ uses amount.
        // ChangeNum$ Any (Fateseal) means the player may take any number (0..pool) of the
        // looked-at cards; treat it as optional with a take limit of the whole pool.
        bool any_count = ab.change_num_any;
        rt.optional = ab.optional_choice || any_count;
        rt.take_count = 1;
        if (ab.change_num >= 0) {
            // Explicit ChangeNum$ N (incl. 0 = "look but take nothing", Birthing Ritual DBDigBis).
            rt.take_count = static_cast<size_t>(ab.change_num);
        } else if (ab.cond_amount_active) {
            int sum = 0;
            for (auto &expr : ab.cond_amount_exprs)
                sum += static_cast<int>(evaluate_dynamic_amount(expr, dig_owner, orderer, 0));
            rt.take_count = compare_svar(sum, ab.cond_amount_compare) ? ab.cond_amount_if_true : ab.amount;
        } else if (any_count) {
            rt.take_count = matching.size();
        } else if (ab.change_num_all) {
            // ChangeNum$ All (Goblin Guide): take every matching card, automatically.
            rt.take_count = matching.size();
        } else if (ab.amount > 0) {
            rt.take_count = ab.amount;
        }
        rt.pool = matching;
        rt.init = true;
    }

    // Present choices, one card at a time until take_count are taken (or the player
    // declines / the pool runs dry). The pool/picks live in the rt so a resume
    // re-enters the suspended pick with the identical (frozen) menu.
    for (; rt.pick < rt.take_count; ++rt.pick) {
        std::vector<LegalAction> dig_actions;
        if (rt.optional) {
            // The decline entry names where the picks and the unchosen rest go (Fateseal: "Put
            // nothing on the bottom of library (rest stay on top of library)").
            LegalAction la(PASS_PRIORITY, dig_decline_label(dig_chosen_destination(ab), dig_chosen_on_bottom(ab),
                                                            dig_rest_destination(ab), dig_rest_on_bottom(ab),
                                                            !rt.chosen.empty()));
            la.category = ActionCategory::DIG_CHOICE;
            dig_actions.push_back(la);
        }
        for (auto e : rt.pool) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DIG_CHOICE;
            dig_actions.push_back(la);
        }
        // If no matching and not optional, fall through (all go to bottom)
        if (dig_actions.empty()) break;
        // ChangeNum$ All (Goblin Guide): the take is mandatory and automatic — no player choice
        // / DIG_CHOICE prompt. Take the next pooled card (rt.optional is false here, so index 0
        // is the first matching card).
        if (ab.change_num_all) {
            Entity sel = rt.pool.front();
            rt.chosen.push_back(sel);
            rt.pool.erase(rt.pool.begin());
            continue;
        }
        // The looker (ab.controller) is the resolving seat, so this is a no-op
        // swap — the exact seat today's inline get_input read from.
        int choice = ctx.ask(dig_actions, looker, ab.source);
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        Entity sel = dig_actions[static_cast<size_t>(choice)].source_entity;
        if (sel == 0) break;  // chose "Take nothing"
        rt.chosen.push_back(sel);
        rt.pool.erase(std::remove(rt.pool.begin(), rt.pool.end(), sel), rt.pool.end());
    }

    Zone::ZoneValue chosen_dest = dig_chosen_destination(ab);
    bool on_bottom = dig_chosen_on_bottom(ab);
    for (Entity chosen : rt.chosen) {
        orderer->add_to_zone(on_bottom, chosen, chosen_dest, owner_sees);
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        if (chosen_dest == Zone::LIBRARY) {
            game_log_private(looker, "%s puts %s on the %s of %s library.\n", actor_name.c_str(),
                cd.name.c_str(), on_bottom ? "bottom" : "top", owner_poss.c_str());
            game_log_redacted(looker, "%s puts a card on the %s of %s library.\n",
                actor_name.c_str(), on_bottom ? "bottom" : "top", owner_poss.c_str());
        } else if (chosen_dest == Zone::BATTLEFIELD) {
            // Public information once it hits the battlefield.
            game_log("%s puts %s onto the battlefield.\n", actor_name.c_str(), cd.name.c_str());
        } else {
            const std::string where = chosen_dest == Zone::EXILE       ? std::string("exile")
                                      : chosen_dest == Zone::GRAVEYARD ? owner_poss + " graveyard"
                                      : actor == dig_owner             ? std::string("hand")
                                                                       : owner_poss + " hand";
            game_log_private(looker, "%s puts %s into %s.\n", actor_name.c_str(), cd.name.c_str(),
                where.c_str());
            game_log_redacted(looker, "%s puts a card into %s.\n", actor_name.c_str(), where.c_str());
        }
    }

    // RememberChanged$ True (Light Up the Stage): stash the moved (chosen) cards in
    // cur_game.remembered_entities so a paired DB$ Effect sub-ability can grant a play
    // permission on exactly those cards (mirrors ChangeZone's RememberChanged behaviour).
    if (ab.remember_changed)
        for (Entity chosen : rt.chosen) cur_game.remembered_entities.push_back(chosen);

    // Remaining cards go to bottom of library
    std::vector<Entity> remaining;
    for (auto e : rt.lib) {
        if (std::find(rt.chosen.begin(), rt.chosen.end(), e) == rt.chosen.end()) remaining.push_back(e);
    }
    if (ab.rest_random_order) {
        // Shuffle remaining with game RNG (platform-stable — see stable_rng.h)
        stable_shuffle(remaining, cur_game.gen);
    }
    // DestinationZone2$ routes the unchosen remainder somewhere other than the library
    // (Malevolent Rumble: "Put the rest into your graveyard"). Default (-1) stays the library.
    if (dig_rest_destination(ab) != Zone::LIBRARY) {
        Zone::ZoneValue rest_dest = dig_rest_destination(ab);
        for (auto e : remaining) {
            orderer->add_to_zone(false, e, rest_dest, owner_sees);
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            game_log("%s puts %s into %s %s.\n", actor_name.c_str(), cd.name.c_str(), owner_poss.c_str(),
                     rest_dest == Zone::GRAVEYARD ? "graveyard"
                     : rest_dest == Zone::EXILE   ? "exile"
                                                  : "hand");
        }
        return HandlerResult::DONE_RUN_SUBS;
    }
    // Unchosen cards normally go to the bottom; LibraryPosition2$ 0 (Fateseal) keeps
    // them on top instead (i.e. you may bottom the looked-at card, else it stays put).
    bool rest_on_bottom = dig_rest_on_bottom(ab);
    for (auto e : remaining) {
        orderer->add_to_zone(rest_on_bottom, e, Zone::LIBRARY, owner_sees);
    }
    game_log("%s puts %zu card(s) on the %s of %s library.\n", actor_name.c_str(), remaining.size(),
             rest_on_bottom ? "bottom" : "top", owner_poss.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_dig(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "DigNum") {
        // Value may be a literal int or an SVar reference (e.g. "X")
        if (!value.empty() && (std::isdigit(value[0]) || value[0] == '-')) {
            ab.dig_num = static_cast<size_t>(std::stoi(value));
        } else {
            ab.dig_num = 0;
            ab.dig_num_expr = value;
        }
        return true;
    } else if (key == "DestinationZone") {
        if (value == "Library") ab.dig_destination = Zone::LIBRARY;
        else if (value == "Hand") ab.dig_destination = Zone::HAND;
        else if (value == "Graveyard") ab.dig_destination = Zone::GRAVEYARD;
        else if (value == "Battlefield") ab.dig_destination = Zone::BATTLEFIELD;
        else if (value == "Exile") ab.dig_destination = Zone::EXILE;
        return true;
    } else if (key == "DestinationZone2") {
        // Where the looked-at-but-unchosen remainder goes (default: back to the library).
        // Malevolent Rumble: Graveyard ("Put the rest into your graveyard").
        if (value == "Library") ab.dig_rest_destination = Zone::LIBRARY;
        else if (value == "Hand") ab.dig_rest_destination = Zone::HAND;
        else if (value == "Graveyard") ab.dig_rest_destination = Zone::GRAVEYARD;
        else if (value == "Battlefield") ab.dig_rest_destination = Zone::BATTLEFIELD;
        else if (value == "Exile") ab.dig_rest_destination = Zone::EXILE;
        return true;
    } else if (key == "LibraryPosition") {
        ab.dig_library_position = std::stoi(value);
        return true;
    } else if (key == "LibraryPosition2") {
        // Where the looked-at-but-unchosen cards go: 0 = top, otherwise bottom.
        ab.dig_rest_library_position = std::stoi(value);
        return true;
    } else if (key == "ChangeValid") {
        ab.change_valid = value;
        return true;
    } else if (key == "RestRandomOrder" || key == "RandomOrder") {
        ab.rest_random_order = (value == "True");
        return true;
    } else if (key == "Reveal") {
        // Reveal$ True — the looked-at cards are shown to all players (Goblin Guide).
        ab.dig_reveal = (value == "True");
        return true;
    }
    return false;
}

}  // namespace effects
