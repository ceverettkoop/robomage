#include "machine_io.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>

#include "card_vocab.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// ── Static helpers ────────────────────────────────────────────────────────────

static int get_card_vocab_idx(Entity e);
static int get_stack_card_vocab_idx(Entity e);
static bool entity_is_land(Entity e);
static void push_player_block(std::vector<float>& out, const PlayerState& ps);
static void push_perm_slot(std::vector<float>& out, const PermanentState& p);

static int get_card_vocab_idx(Entity e) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return -1;
    return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
}

static int get_stack_card_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    if (global_coordinator.entity_has_component<Ability>(e)) {
        Entity src = global_coordinator.GetComponent<Ability>(e).source;
        if (global_coordinator.entity_has_component<CardData>(src))
            return card_name_to_index(global_coordinator.GetComponent<CardData>(src).name);
    }
    return -1;
}

static bool entity_is_land(Entity e) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    for (auto& t : global_coordinator.GetComponent<CardData>(e).types)
        if (t.name == "Land") return true;
    return false;
}

static void push_player_block(std::vector<float>& out, const PlayerState& ps) {
    out.push_back(static_cast<float>(ps.life) / 20.0f);
    out.push_back(static_cast<float>(ps.hand_ct) / 10.0f);
    out.push_back(static_cast<float>(ps.poison_counters) / 10.0f);
    for (int i = 0; i < 6; i++) out.push_back(static_cast<float>(ps.mana[i]) / 10.0f);
}

// Pushes PERM_SLOT_SIZE floats. Empty slot (card_vocab_idx == -1) = all zeros.
static void push_perm_slot(std::vector<float>& out, const PermanentState& p) {
    if (p.card_vocab_idx == -1) {
        for (int i = 0; i < PERM_SLOT_SIZE; i++) out.push_back(0.0f);
        return;
    }
    out.push_back(static_cast<float>(p.power) / 10.0f);
    out.push_back(static_cast<float>(p.toughness) / 10.0f);
    out.push_back(p.is_tapped ? 1.0f : 0.0f);
    out.push_back(p.is_attacking ? 1.0f : 0.0f);
    out.push_back(p.is_blocking ? 1.0f : 0.0f);
    out.push_back(p.has_summoning_sickness ? 1.0f : 0.0f);
    out.push_back(static_cast<float>(p.damage) / 10.0f);
    out.push_back(p.controller_is_self ? 1.0f : 0.0f);
    out.push_back(p.is_creature ? 1.0f : 0.0f);
    out.push_back(p.is_land ? 1.0f : 0.0f);
    for (int i = 0; i < N_CARD_TYPES; i++)
        out.push_back(i == p.card_vocab_idx ? 1.0f : 0.0f);
}

// ── populate_gamestate ────────────────────────────────────────────────────────

void populate_gamestate(GameState* gs, Zone::Ownership viewer) {
    memset(gs, 0, sizeof(*gs));

    // Mark all card_vocab_idx slots as empty (-1)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        gs->self_permanents[i].card_vocab_idx = -1;
        gs->opp_permanents[i].card_vocab_idx  = -1;
    }
    for (int i = 0; i < MAX_HAND_SLOTS; i++) gs->self_hand[i] = -1;
    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        gs->self_graveyard[i] = -1;
        gs->opp_graveyard[i]  = -1;
        gs->self_exile[i]     = -1;
        gs->opp_exile[i]      = -1;
    }

    Zone::Ownership priority_owner = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    if (viewer == Zone::UNKNOWN) viewer = priority_owner;

    Zone::Ownership active_owner = cur_game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity viewer_entity = (viewer == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    Entity opp_entity    = (viewer == Zone::PLAYER_A) ? cur_game.player_b_entity : cur_game.player_a_entity;

    gs->cur_step            = cur_game.cur_step;
    gs->turn                = static_cast<int>(cur_game.turn);
    gs->is_active_player    = (viewer == active_owner);
    gs->viewer_has_priority = (viewer == priority_owner);
    gs->self_is_player_a    = (viewer == Zone::PLAYER_A);

    // Fill player stat fields (hand_ct filled in the entity pass below)
    auto fill_player_stats = [&](PlayerState& ps, Entity ent) {
        auto& p = global_coordinator.GetComponent<Player>(ent);
        ps.life = p.life_total;
        ps.poison_counters = static_cast<int>(p.poison_counters);
        ps.lands_played_this_turn = static_cast<int>(p.lands_played_this_turn);
        int mana_counts[6] = {};
        for (Colors c : p.mana) {
            int idx = static_cast<int>(c);
            if (idx >= 0 && idx < 6) mana_counts[idx]++;
        }
        for (int i = 0; i < 6; i++) ps.mana[i] = mana_counts[i];
    };
    fill_player_stats(gs->self, viewer_entity);
    fill_player_stats(gs->opponent, opp_entity);

    // Stack items collected during the pass, sorted descending by entity ID afterward
    struct StackItem { Entity e; StackEntry entry; };
    StackItem stack_items[MAX_STACK_DISPLAY + 8];
    int stack_item_count = 0;

    // Slot fill counters
    int self_bf = 0, opp_bf = 0;
    int self_gy = 0, opp_gy = 0;
    int self_ex = 0, opp_ex = 0;
    int self_hand_idx = 0;

    // Single pass over all entities (naturally ascending entity ID order)
    for (Entity e = 0; e < MAX_ENTITIES; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(e);
        bool is_self = (zone.owner == viewer);

        switch (zone.location) {
            case Zone::HAND:
                if (is_self) {
                    gs->self.hand_ct++;
                    if (self_hand_idx < MAX_HAND_SLOTS)
                        gs->self_hand[self_hand_idx++] = get_card_vocab_idx(e);
                } else {
                    gs->opponent.hand_ct++;
                }
                break;

            case Zone::LIBRARY:
                if (is_self) gs->self_library_ct++;
                else         gs->opp_library_ct++;
                break;

            case Zone::GRAVEYARD:
                if (is_self && self_gy < MAX_GY_SLOTS)
                    gs->self_graveyard[self_gy++] = get_card_vocab_idx(e);
                else if (!is_self && opp_gy < MAX_GY_SLOTS)
                    gs->opp_graveyard[opp_gy++] = get_card_vocab_idx(e);
                break;

            case Zone::EXILE:
                if (is_self && self_ex < MAX_GY_SLOTS)
                    gs->self_exile[self_ex++] = get_card_vocab_idx(e);
                else if (!is_self && opp_ex < MAX_GY_SLOTS)
                    gs->opp_exile[opp_ex++] = get_card_vocab_idx(e);
                break;

            case Zone::STACK: {
                gs->stack_size++;
                if (stack_item_count < MAX_STACK_DISPLAY + 8) {
                    StackEntry se;
                    se.card_vocab_idx    = get_stack_card_vocab_idx(e);
                    se.controller_is_self = (zone.owner == viewer);
                    se.is_spell           = global_coordinator.entity_has_component<Spell>(e);
                    stack_items[stack_item_count++] = {e, se};
                }
                break;
            }

            case Zone::BATTLEFIELD: {
                if (!global_coordinator.entity_has_component<Permanent>(e)) break;
                auto& perm = global_coordinator.GetComponent<Permanent>(e);

                PermanentState ps;
                ps.card_vocab_idx        = get_card_vocab_idx(e);
                ps.controller_is_self    = (perm.controller == viewer);
                ps.is_tapped             = perm.is_tapped;
                ps.has_summoning_sickness = perm.has_summoning_sickness;
                ps.is_creature           = global_coordinator.entity_has_component<Creature>(e);
                ps.is_land               = entity_is_land(e);

                if (ps.is_creature) {
                    auto& cr     = global_coordinator.GetComponent<Creature>(e);
                    ps.power     = static_cast<int>(cr.power);
                    ps.toughness = static_cast<int>(cr.toughness);
                    ps.is_attacking = cr.is_attacking;
                    ps.is_blocking  = cr.is_blocking;
                } else {
                    ps.power = ps.toughness = 0;
                    ps.is_attacking = ps.is_blocking = false;
                }

                ps.damage = 0;
                if (global_coordinator.entity_has_component<Damage>(e))
                    ps.damage = static_cast<int>(global_coordinator.GetComponent<Damage>(e).damage_counters);

                if (ps.controller_is_self && self_bf < MAX_BATTLEFIELD_SLOTS)
                    gs->self_permanents[self_bf++] = ps;
                else if (!ps.controller_is_self && opp_bf < MAX_BATTLEFIELD_SLOTS)
                    gs->opp_permanents[opp_bf++] = ps;
                break;
            }

            default:
                break;
        }
    }

    // Sort stack entries by descending entity ID (top of stack = highest ID)
    std::sort(stack_items, stack_items + stack_item_count,
              [](const StackItem& a, const StackItem& b) { return a.e > b.e; });
    int copy_count = std::min(stack_item_count, MAX_STACK_DISPLAY);
    for (int i = 0; i < copy_count; i++)
        gs->stack[i] = stack_items[i].entry;

    gs->opp_hand_ct = gs->opponent.hand_ct;
}

// ── populate_query ────────────────────────────────────────────────────────────

void populate_query(Query* q, const std::vector<LegalAction>& actions) {
    memset(q, 0, sizeof(*q));
    int n = std::min(static_cast<int>(actions.size()), MAX_ACTIONS);
    q->num_choices = n;

    Zone::Ownership priority_owner = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity priority_ent = cur_game.player_a_has_priority ? cur_game.player_a_entity : cur_game.player_b_entity;
    Entity opp_ent      = cur_game.player_a_has_priority ? cur_game.player_b_entity : cur_game.player_a_entity;

    for (int i = 0; i < n; i++) {
        const LegalAction& la = actions[static_cast<size_t>(i)];
        ActionChoice& ac = q->choices[i];

        ac.category = static_cast<int>(la.category);
        ac.slot_idx = -1;

        // Card vocab index from source entity (or ability source)
        Entity src = la.source_entity;
        int vocab_idx = -1;
        if (src != 0) {
            if (global_coordinator.entity_has_component<CardData>(src)) {
                vocab_idx = card_name_to_index(global_coordinator.GetComponent<CardData>(src).name);
            } else if (global_coordinator.entity_has_component<Ability>(src)) {
                Entity ab_src = global_coordinator.GetComponent<Ability>(src).source;
                if (global_coordinator.entity_has_component<CardData>(ab_src))
                    vocab_idx = card_name_to_index(global_coordinator.GetComponent<CardData>(ab_src).name);
            }
        }
        ac.card_vocab_idx = vocab_idx;

        // Controller is self
        ac.controller_is_self = false;
        if (src != 0) {
            if (global_coordinator.entity_has_component<Permanent>(src))
                ac.controller_is_self = (global_coordinator.GetComponent<Permanent>(src).controller == priority_owner);
            else if (src == priority_ent)
                ac.controller_is_self = true;
            else if (global_coordinator.entity_has_component<Zone>(src))
                ac.controller_is_self = (global_coordinator.GetComponent<Zone>(src).owner == priority_owner);
        }

        // Zone reference
        ac.zone_ref = REF_NONE;
        if (src != 0 && global_coordinator.entity_has_component<Zone>(src)) {
            auto& z = global_coordinator.GetComponent<Zone>(src);
            bool is_self_owned = (z.owner == priority_owner);
            switch (z.location) {
                case Zone::BATTLEFIELD:
                    ac.zone_ref = is_self_owned ? REF_SELF_BATTLEFIELD : REF_OPP_BATTLEFIELD;
                    break;
                case Zone::HAND:
                    ac.zone_ref = is_self_owned ? REF_SELF_HAND : REF_NONE;
                    break;
                case Zone::STACK:
                    ac.zone_ref = REF_STACK;
                    break;
                case Zone::GRAVEYARD:
                    ac.zone_ref = is_self_owned ? REF_SELF_GY : REF_OPP_GY;
                    break;
                case Zone::EXILE:
                    ac.zone_ref = is_self_owned ? REF_SELF_EXILE : REF_OPP_EXILE;
                    break;
                default:
                    break;
            }
        } else if (src == priority_ent) {
            ac.zone_ref = REF_PLAYER_SELF;
        } else if (src == opp_ent) {
            ac.zone_ref = REF_PLAYER_OPP;
        }

        snprintf(ac.description, MAX_CHOICE_DESC, "%s", la.description.c_str());
    }
}

// ── serialize_state ───────────────────────────────────────────────────────────

std::vector<float> serialize_state(const GameState* gs) {
    std::vector<float> state;
    state.reserve(static_cast<size_t>(STATE_SIZE));

    // Header: self (9) + opp (9) + step one-hot (12) + flags (3) = 33
    push_player_block(state, gs->self);
    push_player_block(state, gs->opponent);
    for (int i = 0; i < 12; i++)
        state.push_back((gs->cur_step == static_cast<Step>(i)) ? 1.0f : 0.0f);
    state.push_back(gs->is_active_player ? 1.0f : 0.0f);
    state.push_back(gs->self_is_player_a ? 1.0f : 0.0f);
    state.push_back(static_cast<float>(gs->stack_size) / 10.0f);

    // Self permanents (48 x 42 = 2016)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++)
        push_perm_slot(state, gs->self_permanents[i]);

    // Opp permanents (48 x 42 = 2016)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++)
        push_perm_slot(state, gs->opp_permanents[i]);

    // Stack (12 x 34 = 408): controller_is_self(1) + card_id one-hot(32) + is_spell(1)
    int stored_stack = std::min(gs->stack_size, MAX_STACK_DISPLAY);
    for (int i = 0; i < MAX_STACK_DISPLAY; i++) {
        if (i < stored_stack) {
            state.push_back(gs->stack[i].controller_is_self ? 1.0f : 0.0f);
            int idx = gs->stack[i].card_vocab_idx;
            for (int j = 0; j < N_CARD_TYPES; j++)
                state.push_back(j == idx ? 1.0f : 0.0f);
            state.push_back(gs->stack[i].is_spell ? 1.0f : 0.0f);
        } else {
            for (int j = 0; j < STACK_SLOT_SIZE; j++) state.push_back(0.0f);
        }
    }

    // Self graveyard (64 x 32 = 2048)
    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        int idx = gs->self_graveyard[i];
        for (int j = 0; j < GY_SLOT_SIZE; j++)
            state.push_back(j == idx ? 1.0f : 0.0f);
    }

    // Opp graveyard (64 x 32 = 2048)
    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        int idx = gs->opp_graveyard[i];
        for (int j = 0; j < GY_SLOT_SIZE; j++)
            state.push_back(j == idx ? 1.0f : 0.0f);
    }

    // NOTE: exile zones are populated in GameState but not serialized here.
    // Add them back once cards that use exile are implemented.

    // Self hand (10 x 32 = 320)
    for (int i = 0; i < MAX_HAND_SLOTS; i++) {
        int idx = gs->self_hand[i];
        for (int j = 0; j < N_CARD_TYPES; j++)
            state.push_back(j == idx ? 1.0f : 0.0f);
    }

    assert(static_cast<int>(state.size()) == STATE_SIZE);
    return state;
}
