#include "machine_io.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "card_vocab.h"
#include "classes/colors.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/counter_map.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/token.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "queries/lki.h"
#include "queries/player_resources.h"
#include "queries/players.h"
#include "queries/zones.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static int token_vocab_idx(Entity e);
static int last_known_vocab_idx(Entity e);
static int get_card_vocab_idx(Entity e);
static void fill_player_state(PlayerState &ps, Entity player_entity);
static void fill_permanent_state(PermanentState &ps, Entity e);
static void format_counter_summary(const CounterMap &counters, char *buf, size_t buf_len);

// Vocab index for a token entity: its registered token-band index (keyed by the
// token SCRIPT stem — display names collide, stems don't), else TOKEN_SENTINEL.
static int token_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Token>(e)) {
        int idx = token_script_to_index(global_coordinator.GetComponent<Token>(e).script_name);
        if (idx >= 0) return idx;
    }
    return TOKEN_SENTINEL;
}

// Vocab index of an object that left the battlefield and ceased to exist (a token, CR 111.7),
// from its last-known information: a token's token-band index (else TOKEN_SENTINEL), a card's
// name index. -1 when no last-known information was captured.
static int last_known_vocab_idx(Entity e) {
    const LastKnownInfo *lki = lki_for(e);
    if (!lki || lki->name.empty()) return -1;
    if (!lki->is_token) return card_name_to_index(lki->name);
    int idx = token_script_to_index(lki->token_script);
    return idx >= 0 ? idx : TOKEN_SENTINEL;
}

// Vocab index of a zoned card or permanent: a permanent by its current name (a token by its token
// index), any other card by its CardData name.
static int get_card_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token) return token_vocab_idx(e);
        return card_name_to_index(perm.name);
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return token_vocab_idx(e);
    return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
}

int action_card_vocab_idx(Entity e) {
    if (e == 0) return -1;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? token_vocab_idx(e) : card_name_to_index(perm.name);
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    if (global_coordinator.entity_has_component<Ability>(e)) {
        Entity src = global_coordinator.GetComponent<Ability>(e).source.lki_entity();
        if (global_coordinator.entity_has_component<Permanent>(src)) {
            auto &sp = global_coordinator.GetComponent<Permanent>(src);
            return sp.is_token ? token_vocab_idx(src) : card_name_to_index(sp.name);
        }
        if (global_coordinator.entity_has_component<CardData>(src))
            return card_name_to_index(global_coordinator.GetComponent<CardData>(src).name);
        // An ability whose token source has already left play keeps no Permanent/CardData;
        // a lingering Token component still identifies it as a token.
        if (global_coordinator.entity_has_component<Token>(src)) return token_vocab_idx(src);
        // A source token that has ceased to exist entirely: its last-known identity.
        return last_known_vocab_idx(src);
    }
    // An object with no live components left (a token that ceased to exist, CR 111.7) is
    // identified by its last-known information.
    return last_known_vocab_idx(e);
}

// Compact display summary of a permanent's typed counter store for the board
// printout ("charge:2, +1/+1:3, loyalty:4"). Engine counter types are uppercase
// tokens (CR 122.1 kinds: "P1P1", "M1M1", "CHARGE", "LOYALTY", ...); render P1P1/M1M1
// with their conventional +1/+1 / -1/-1 names and everything else lowercased.
// Writes the empty string when the permanent has no counters.
static void format_counter_summary(const CounterMap &counters, char *buf, size_t buf_len) {
    buf[0] = '\0';
    size_t len = 0;
    for (const auto &c : counters) {
        std::string label;
        if (c.first == "P1P1")
            label = "+1/+1";
        else if (c.first == "M1M1")
            label = "-1/-1";
        else {
            label = c.first;
            for (auto &ch : label) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        int n = snprintf(buf + len, buf_len - len, "%s%s:%d", len > 0 ? ", " : "", label.c_str(), c.second);
        if (n < 0 || static_cast<size_t>(n) >= buf_len - len) break;  // truncated: keep what fit
        len += static_cast<size_t>(n);
    }
}

// A player's life, poison and energy counters and mana pool.
static void fill_player_state(PlayerState &ps, Entity player_entity) {
    auto &p = global_coordinator.GetComponent<Player>(player_entity);
    ps.life = p.life_total;
    ps.poison_counters = p.counter_count("POISON");
    ps.energy = player_energy(p);
    for (Colors c : p.mana) {
        int idx = static_cast<int>(c);
        if (idx >= 0 && idx < 6) ps.mana[idx]++;
    }
}

static void fill_permanent_state(PermanentState &ps, Entity e) {
    auto &perm = global_coordinator.GetComponent<Permanent>(e);
    ps.card_vocab_idx = get_card_vocab_idx(e);
    ps.is_tapped = perm.is_tapped;
    ps.has_summoning_sickness = perm.has_summoning_sickness;
    ps.is_creature = global_coordinator.entity_has_component<Creature>(e);
    if (ps.is_creature) {
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        ps.power = static_cast<int>(cr.power);
        ps.toughness = static_cast<int>(cr.toughness);
    }
    if (global_coordinator.entity_has_component<Damage>(e))
        ps.damage = static_cast<int>(global_coordinator.GetComponent<Damage>(e).damage_counters);
    format_counter_summary(perm.counters, ps.counters, sizeof(ps.counters));
}

void populate_gamestate(GameState *gs, Zone::Ownership viewer) {
    memset(gs, 0, sizeof(*gs));
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        gs->self_permanents[i].card_vocab_idx = -1;
        gs->opp_permanents[i].card_vocab_idx = -1;
    }
    for (int i = 0; i < MAX_HAND_SLOTS; i++) gs->self_hand[i] = -1;

    if (viewer == Zone::UNKNOWN) viewer = priority_seat();
    gs->cur_step = cur_game.turn_state.step;
    gs->turn = static_cast<int>(cur_game.turn_state.turn);
    gs->is_active_player = (viewer == active_seat());
    gs->self_is_player_a = (viewer == Zone::PLAYER_A);
    fill_player_state(gs->self, get_player_entity(viewer));
    fill_player_state(gs->opponent, get_player_entity(opponent_of(viewer)));

    // Every stack object, sorted by distance_from_top (0 = the top) before the first
    // MAX_STACK_DISPLAY are kept, so the slots hold the top of the stack.
    std::vector<std::pair<size_t, Entity>> stack_items;
    int self_bf = 0, opp_bf = 0, self_hand = 0;
    for (Entity e : zoned_entities()) {
        const Zone &zone = global_coordinator.GetComponent<Zone>(e);
        switch (zone.location) {
            case Zone::HAND:
                if (zone.owner == viewer && self_hand < MAX_HAND_SLOTS)
                    gs->self_hand[self_hand++] = get_card_vocab_idx(e);
                break;
            case Zone::STACK:
                stack_items.push_back({zone.distance_from_top, e});
                break;
            case Zone::BATTLEFIELD:
                // Phased-out permanents are shown too (the board lists every permanent).
                if (!global_coordinator.entity_has_component<Permanent>(e)) break;
                if (global_coordinator.GetComponent<Permanent>(e).controller == viewer) {
                    if (self_bf < MAX_BATTLEFIELD_SLOTS) fill_permanent_state(gs->self_permanents[self_bf++], e);
                } else {
                    if (opp_bf < MAX_BATTLEFIELD_SLOTS) fill_permanent_state(gs->opp_permanents[opp_bf++], e);
                }
                break;
            default:
                break;
        }
    }
    std::sort(stack_items.begin(), stack_items.end(),
        [](const std::pair<size_t, Entity> &a, const std::pair<size_t, Entity> &b) { return a.first < b.first; });
    gs->stack_size = static_cast<int>(stack_items.size());
    const int shown = std::min(gs->stack_size, MAX_STACK_DISPLAY);
    for (int i = 0; i < shown; i++)
        gs->stack[i].card_vocab_idx = action_card_vocab_idx(stack_items[static_cast<size_t>(i)].second);
}
