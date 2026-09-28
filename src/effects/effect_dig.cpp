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
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../queries/filters.h"
#include "../queries/players.h"
#include "../stable_rng.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static Zone::ZoneValue dig_chosen_destination(const Ability &ab);
static bool dig_chosen_on_bottom(const Ability &ab);
static Zone::ZoneValue dig_rest_destination(const Ability &ab);
static bool dig_rest_on_bottom(const Ability &ab);
static bool dig_exiles_face_down(const Ability &ab, Zone::ZoneValue dest);
static bool dig_is_blind(const Ability &ab);
static bool dig_chosen_revealed(const Ability &ab, Zone::ZoneValue dest);

// Where the chosen cards go: DestinationZone$, else the hand.
static Zone::ZoneValue dig_chosen_destination(const Ability &ab) {
    return ab.def->dig_destination >= 0 ? static_cast<Zone::ZoneValue>(ab.def->dig_destination) : Zone::HAND;
}

// Whether chosen cards put into a library go on the bottom (LibraryPosition$ 0 = top). Only
// meaningful with DestinationZone$ set.
static bool dig_chosen_on_bottom(const Ability &ab) {
    return ab.def->dig_destination >= 0 && ab.def->dig_library_position != 0;
}

// Where the unchosen rest go: DestinationZone2$, else the library.
static Zone::ZoneValue dig_rest_destination(const Ability &ab) {
    return ab.def->dig_rest_destination >= 0 ? static_cast<Zone::ZoneValue>(ab.def->dig_rest_destination)
                                        : Zone::LIBRARY;
}

// Whether the unchosen rest go on the bottom of the library (LibraryPosition2$ 0 keeps them on
// top — Fateseal).
static bool dig_rest_on_bottom(const Ability &ab) { return ab.def->dig_rest_library_position != 0; }

// ExileFaceDown$ True (Triumph of Saint Katherine): the cards this dig moves into exile are
// exiled face down (CR 406.3), so their identities stay hidden and are not publicly revealed.
static bool dig_exiles_face_down(const Ability &ab, Zone::ZoneValue dest) {
    return ab.def->exile_face_down && dest == Zone::EXILE;
}

// Whether a chosen card is shown to all players. Every looked-at card is with Reveal$ True
// (Goblin Guide). Otherwise a card chosen for a ChangeValid$ quality is revealed to show it has
// that quality (Once Upon a Time's "you may reveal a creature or land card from among them and
// put it into your hand"), unless the ability says NoReveal$ True or exiles it face down.
static bool dig_chosen_revealed(const Ability &ab, Zone::ZoneValue dest) {
    if (ab.def->dig_reveal) return true;
    const PeekParams *pp = std::get_if<PeekParams>(&ab.def->params);
    if (pp && pp->no_reveal) return false;
    return !ab.def->change_valid.empty() && !dig_exiles_face_down(ab, dest);
}

// A face-down exile of every card in the slice, with no filter and no reveal, is a blind move:
// no player is instructed to look at the cards (Triumph's "exile ... the top six cards of your
// library in a face-down pile"), so none of them becomes known to anyone (CR 406.3).
static bool dig_is_blind(const Ability &ab) {
    return dig_exiles_face_down(ab, dig_chosen_destination(ab)) && ab.def->change_num_all &&
           ab.def->change_valid.empty() && !ab.def->dig_reveal;
}

namespace effects {

HandlerResult dig(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source.lki_entity());
    // Look at top N cards, player picks one matching filter, rest go to bottom.
    // When the ability targets a player (Fateseal, e.g. Jace +2), the dug library is
    // the TARGET player's, not the controller's.
    Zone::Ownership dig_owner = ab.controller;
    if (ab.target.get() != 0 && global_coordinator.entity_has_component<Player>(ab.target.get()))
        dig_owner = seat_of_player(ab.target.get());
    // The LOOKER (who sees the cards and makes the choices) is always the ability's controller.
    // For a fateseal on an opponent's library (Jace +2) that differs from dig_owner: the owner
    // must NOT learn any card placed back on top, and private card-name logs are pinned to the
    // looker and redacted from the owner.
    Zone::Ownership looker = ab.controller;
    LibraryTopView top_view = (dig_owner == looker) ? LibraryTopView::OWNER
                                                    : LibraryTopView::OPPONENT;
    // Narrative actor: the ability's controller when the dug player is a chosen target (Jace
    // +2 looks at the target's library), else the dug player acting on their own library
    // (Goblin Guide: the defending player reveals and takes the card). Zones are the dig owner's
    // ("their" when the actor owns them, else "Player B's").
    Zone::Ownership actor = (ab.def->valid_tgts != "N_A") ? looker : dig_owner;
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
        size_t effective_dig_num = ab.def->dig_num;
        if (!ab.def->dig_num_expr.empty()) {
            effective_dig_num = evaluate_dynamic_amount(ab.def->dig_num_expr, dig_owner, orderer, 0);
        }
        rt.lib = orderer->get_library_top(dig_owner, effective_dig_num);

        // ChangeValid$ (Once Upon a Time's Card.Creature,Card.Land, Malevolent Rumble's Permanent,
        // Birthing Ritual's Creature.cmcLEX) is matched by the shared filter matcher against each
        // card. Its X (cmcLEX) is resolved from dynamic_amount_expr when the script supplies one
        // (Birthing Ritual: 1 + the sacrificed creature's mana value), else it is the X paid.
        MatchCtx mctx;
        mctx.controller = dig_owner;
        mctx.source = ab.source.lki_entity();
        if (!ab.def->change_valid.empty() && ab.def->change_valid.find("cmcLE") != std::string::npos &&
            !ab.def->dynamic_amount_expr.empty()) {
            mctx.cmc_bound = static_cast<int>(
                evaluate_dynamic_amount(ab.def->dynamic_amount_expr, dig_owner, orderer, ab.target.get()));
            mctx.cmc_op = "LE";
        }
        std::vector<Entity> matching;
        for (auto e : rt.lib)
            if (ab.def->change_valid.empty() || card_matches_filter(e, ab.def->change_valid, mctx))
                matching.push_back(e);

        if (dig_is_blind(ab))
            game_log("%s exiles the top %zu card(s) of %s library face down.\n", actor_name.c_str(),
                     rt.lib.size(), owner_poss.c_str());
        else
            game_log("%s looks at the top %zu card(s) of %s library.\n", actor_name.c_str(),
                     rt.lib.size(), owner_poss.c_str());

        // Reveal$ True (Goblin Guide): the looked-at cards are shown to ALL players. Log the
        // reveal publicly (visible to both seats, not redacted) and record it in the belief-state
        // multi-hot so the non-owner's observation carries the revealed identity, regardless of
        // where each card subsequently goes (hand if it matches, else back on top).
        if (ab.def->dig_reveal) {
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
        bool any_count = ab.def->change_num_any;
        rt.optional = ab.def->optional_choice || any_count;
        rt.take_count = 1;
        if (ab.def->change_num >= 0) {
            // Explicit ChangeNum$ N (incl. 0 = "look but take nothing", Birthing Ritual DBDigBis).
            rt.take_count = static_cast<size_t>(ab.def->change_num);
        } else if (ab.def->cond_amount_active) {
            int sum = 0;
            for (auto &expr : ab.def->cond_amount_exprs)
                sum += static_cast<int>(evaluate_dynamic_amount(expr, dig_owner, orderer, 0));
            rt.take_count = compare_svar(sum, ab.def->cond_amount_compare) ? ab.def->cond_amount_if_true : ab.def->amount;
        } else if (any_count) {
            rt.take_count = matching.size();
        } else if (ab.def->change_num_all) {
            // ChangeNum$ All (Goblin Guide): take every matching card, automatically.
            rt.take_count = matching.size();
        } else if (ab.def->amount > 0) {
            rt.take_count = ab.def->amount;
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
        if (ab.def->change_num_all) {
            Entity sel = rt.pool.front();
            rt.chosen.push_back(sel);
            rt.pool.erase(rt.pool.begin());
            continue;
        }
        // The looker (ab.controller) is the resolving seat, so this is a no-op
        // swap — the exact seat today's inline get_input read from.
        int choice = ctx.ask(dig_actions, looker, ab.source.lki_entity());
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        Entity sel = dig_actions[static_cast<size_t>(choice)].source_entity;
        if (sel == 0) break;  // chose "Take nothing"
        rt.chosen.push_back(sel);
        rt.pool.erase(std::remove(rt.pool.begin(), rt.pool.end(), sel), rt.pool.end());
    }

    Zone::ZoneValue chosen_dest = dig_chosen_destination(ab);
    bool on_bottom = dig_chosen_on_bottom(ab);
    const bool chosen_face_down = dig_exiles_face_down(ab, chosen_dest);
    const bool blind = dig_is_blind(ab);
    const bool chosen_revealed = dig_chosen_revealed(ab, chosen_dest);
    for (Entity chosen : rt.chosen) {
        // A revealed card is recorded in the opponent's belief state before it moves, so it
        // stays known if it lands in a hand (CR 701.20a).
        if (chosen_revealed) mark_card_revealed(chosen, dig_owner);
        if (chosen_dest == Zone::BATTLEFIELD) {
            // An Aura chooses what it enchants as it enters, or stays put without a legal object
            // (CR 303.4f/g) — the shared uncast battlefield entry.
            if (put_onto_battlefield(orderer, FrameCtx::blocking(), chosen) != Zone::BATTLEFIELD) continue;
        } else {
            orderer->add_to_zone(on_bottom, chosen, chosen_dest, top_view, chosen_face_down);
        }
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        if (blind) {
            // Nobody saw the card; the summary line above narrates the move.
        } else if (chosen_dest == Zone::LIBRARY) {
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
            const char *face = chosen_face_down ? " face down" : "";
            if (chosen_revealed) {
                game_log("%s reveals %s and puts it into %s.\n", actor_name.c_str(), cd.name.c_str(),
                         where.c_str());
                continue;
            }
            game_log_private(looker, "%s puts %s into %s%s.\n", actor_name.c_str(), cd.name.c_str(),
                where.c_str(), face);
            game_log_redacted(looker, "%s puts a card into %s%s.\n", actor_name.c_str(), where.c_str(),
                              face);
        }
    }

    // RememberChanged$ True (Light Up the Stage): stash the moved (chosen) cards in
    // cur_game.remembered_entities so a paired DB$ Effect sub-ability can grant a play
    // permission on exactly those cards (mirrors ChangeZone's RememberChanged behaviour).
    if (ab.def->remember_changed)
        for (Entity chosen : rt.chosen) cur_game.remembered_entities.push_back(ObjectRef::of(chosen));

    // Remaining cards go to bottom of library
    std::vector<Entity> remaining;
    for (auto e : rt.lib) {
        if (std::find(rt.chosen.begin(), rt.chosen.end(), e) == rt.chosen.end()) remaining.push_back(e);
    }
    if (ab.def->rest_random_order) {
        // Shuffle remaining with game RNG (platform-stable — see stable_rng.h)
        stable_shuffle(remaining, cur_game.rng.engine);
    }
    // DestinationZone2$ routes the unchosen remainder somewhere other than the library
    // (Malevolent Rumble: "Put the rest into your graveyard"). Default (-1) stays the library.
    if (dig_rest_destination(ab) != Zone::LIBRARY) {
        Zone::ZoneValue rest_dest = dig_rest_destination(ab);
        const bool rest_face_down = dig_exiles_face_down(ab, rest_dest);
        for (auto e : remaining) {
            orderer->add_to_zone(false, e, rest_dest, top_view, rest_face_down);
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            if (rest_face_down) {
                game_log_private(looker, "%s puts %s into exile face down.\n", actor_name.c_str(),
                                 cd.name.c_str());
                game_log_redacted(looker, "%s puts a card into exile face down.\n",
                                  actor_name.c_str());
                continue;
            }
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
        orderer->add_to_zone(rest_on_bottom, e, Zone::LIBRARY, top_view);
    }
    game_log("%s puts %zu card(s) on the %s of %s library.\n", actor_name.c_str(), remaining.size(),
             rest_on_bottom ? "bottom" : "top", owner_poss.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_dig(AbilityDef &ab, const std::string &key, const std::string &value) {
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
