#include "machine_io.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <vector>

#include "card_vocab.h"
#include "classes/game.h"
#include "classes/match_state.h"
#include "components/ability.h"
#include "components/carddata.h"
#include "components/token.h"
#include "components/creature.h"
#include "components/damage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "game_queries.h"
#include "systems/state_manager_internal.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// ── Static helpers ────────────────────────────────────────────────────────────

static int get_card_vocab_idx(Entity e);
static void push_player_block(std::vector<float>& out, const PlayerState& ps);
static void push_perm_slot(std::vector<float>& out, const PermanentState& p);
static void format_counter_summary(const CounterMap& counters, char* buf, size_t buf_len);
static void add_stack_target(StackEntry& se, int& n, Entity tgt, Zone::Ownership viewer);
static void fill_stack_choices(const Ability& ab, StackEntry& se, Zone::Ownership viewer);

static int get_card_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token) return TOKEN_SENTINEL;
        return card_name_to_index(perm.name);
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return TOKEN_SENTINEL;
    return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
}

// Vocab index for an action's source entity or a stack entity. The single chain
// shared by populate_query (BQUERY) and record_chosen_action (action log) so the
// two never disagree, and by the stack feature extractor (a stack entry is a spell
// with CardData or a standalone ability whose source resolves the same way).
int action_card_vocab_idx(Entity e) {
    if (e == 0) return -1;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? TOKEN_SENTINEL : card_name_to_index(perm.name);
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    if (global_coordinator.entity_has_component<Ability>(e)) {
        Entity src = global_coordinator.GetComponent<Ability>(e).source;
        if (global_coordinator.entity_has_component<Permanent>(src)) {
            auto& sp = global_coordinator.GetComponent<Permanent>(src);
            return sp.is_token ? TOKEN_SENTINEL : card_name_to_index(sp.name);
        }
        if (global_coordinator.entity_has_component<CardData>(src))
            return card_name_to_index(global_coordinator.GetComponent<CardData>(src).name);
        // An ability whose token source has already left play keeps no Permanent/CardData;
        // a lingering Token component still identifies it as a token (stack extractor case).
        if (global_coordinator.entity_has_component<Token>(src))
            return TOKEN_SENTINEL;
    }
    return -1;
}

// Record one announced target of a stack object into the entry's next free target
// sub-slot (bounded by MAX_STACK_TGTS; excess targets are silently truncated).
static void add_stack_target(StackEntry& se, int& n, Entity tgt, Zone::Ownership viewer) {
    if (tgt == 0 || n >= MAX_STACK_TGTS) return;
    StackTarget& st = se.targets[n];
    st.present = true;
    st.is_player = global_coordinator.entity_has_component<Player>(tgt);
    Zone::Ownership ctrl = Zone::UNKNOWN;
    if (st.is_player) {
        ctrl = (tgt == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    } else if (global_coordinator.entity_has_component<Permanent>(tgt)) {
        ctrl = global_coordinator.GetComponent<Permanent>(tgt).controller;
    } else if (global_coordinator.entity_has_component<Zone>(tgt)) {
        // Non-permanent target (a spell on the stack, a graveyard card): its owner.
        ctrl = global_coordinator.GetComponent<Zone>(tgt).owner;
    }
    st.controller_is_self = (ctrl == viewer);
    st.card_vocab_idx = st.is_player ? -1 : action_card_vocab_idx(tgt);
    n++;
}

// Fill a stack entry's announced choices — targets and chosen modal modes — from the
// object's Ability (all public info, announced as it was cast / put on the stack,
// CR 601.2b/c). Targets are recorded in announcement order: the primary ability's,
// then targeting sub-abilities', then each cast-chosen mode's.
static void fill_stack_choices(const Ability& ab, StackEntry& se, Zone::Ownership viewer) {
    int n = 0;
    auto add_ability_targets = [&](const Ability& a) {
        if (!a.targets.empty())
            for (Entity t : a.targets) add_stack_target(se, n, t, viewer);
        else
            add_stack_target(se, n, a.target, viewer);
    };
    add_ability_targets(ab);
    for (const Ability& sub : ab.subabilities) add_ability_targets(sub);
    for (int idx : ab.charm_chosen) {
        if (idx < 0 || static_cast<size_t>(idx) >= ab.charm_choices.size()) continue;
        if (idx < MAX_STACK_MODES) se.chosen_modes[idx] = true;
        add_ability_targets(ab.charm_choices[static_cast<size_t>(idx)]);
    }
}

// Compact display summary of a permanent's typed counter store for the board
// printout ("charge:2, +1/+1:3, loyalty:4"). Engine counter types are uppercase
// tokens (CR 122.1 kinds: "P1P1", "M1M1", "CHARGE", "LOYALTY", ...); render P1P1/M1M1
// with their conventional +1/+1 / -1/-1 names and everything else lowercased.
// Writes the empty string when the permanent has no counters. Display only — this
// never feeds the ML state vector.
static void format_counter_summary(const CounterMap& counters, char* buf, size_t buf_len) {
    buf[0] = '\0';
    size_t len = 0;
    for (const auto& c : counters) {
        std::string label;
        if (c.first == "P1P1")      label = "+1/+1";
        else if (c.first == "M1M1") label = "-1/-1";
        else {
            label = c.first;
            for (auto& ch : label) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        int n = snprintf(buf + len, buf_len - len, "%s%s:%d", len > 0 ? ", " : "", label.c_str(), c.second);
        if (n < 0 || static_cast<size_t>(n) >= buf_len - len) break;  // truncated: keep what fit
        len += static_cast<size_t>(n);
    }
}

int action_card_vocab_idx(const LegalAction& la) {
    // A modal-DFC back-face play/cast (e.g. Witch-Blessed Meadow land, or a nonland back
    // like Tergrid's Lantern) uses the combined card as its source, whose CardData is the
    // FRONT face — resolve the back face's name so the emitted/logged id matches the face
    // actually being played or cast.
    if ((la.play_back_face || la.cast_back_face) && la.source_entity != 0 &&
        global_coordinator.entity_has_component<CardData>(la.source_entity)) {
        const auto& cd = global_coordinator.GetComponent<CardData>(la.source_entity);
        if (cd.backside) return card_name_to_index(cd.backside->name);
    }
    return action_card_vocab_idx(la.source_entity);
}


static void push_player_block(std::vector<float>& out, const PlayerState& ps) {
    out.push_back(static_cast<float>(ps.life) / 20.0f);
    out.push_back(static_cast<float>(ps.hand_ct) / 10.0f);
    out.push_back(static_cast<float>(ps.poison_counters) / 10.0f);
    for (int i = 0; i < 6; i++) out.push_back(static_cast<float>(ps.mana[i]) / 10.0f);
}

// Pushes PERM_SLOT_SIZE floats (11 status + 1 card-id). Empty slot
// (card_vocab_idx == -1) = 11 zeros + the empty/unknown id sentinel.
static void push_perm_slot(std::vector<float>& out, const PermanentState& p) {
    if (p.card_vocab_idx == -1) {
        out.insert(out.end(), PERM_SLOT_SIZE - 1, 0.0f);
        out.push_back(norm_card_id(-1));
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
    out.push_back(static_cast<float>(p.loyalty) / 10.0f);
    out.push_back(norm_card_id(p.card_vocab_idx));
}

// ── populate_gamestate ────────────────────────────────────────────────────────

void populate_gamestate(GameState* gs, Zone::Ownership viewer) {
    memset(gs, 0, sizeof(*gs));

    // Mark all card_vocab_idx slots as empty (-1)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        gs->self_permanents[i].card_vocab_idx = -1;
        gs->opp_permanents[i].card_vocab_idx  = -1;
    }
    for (int i = 0; i < MAX_HAND_SLOTS; i++) { gs->self_hand[i] = -1; gs->opp_known_hand[i] = -1; }
    for (int i = 0; i < MAX_GY_SLOTS; i++) {
        gs->self_graveyard[i] = -1;
        gs->opp_graveyard[i]  = -1;
    }
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++) gs->known_top_library_self[i] = -1;

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

    // Pending decision context: the spell/ability currently making a mid-resolution choice
    // (set via PendingDecisionScope). Controller derived from the source entity the same way
    // populate_query derives per-action controller_is_self.
    gs->pending_decision_card = -1;
    gs->pending_decision_ctrl_is_self = false;
    if (cur_game.pending_decision_source != 0) {
        extern bool sideboard_phase;
        extern Zone::Ownership sideboard_phase_player;
        Entity pd = cur_game.pending_decision_source;
        gs->pending_decision_card = action_card_vocab_idx(pd);
        if (global_coordinator.entity_has_component<Permanent>(pd))
            gs->pending_decision_ctrl_is_self =
                (global_coordinator.GetComponent<Permanent>(pd).controller == viewer);
        else if (global_coordinator.entity_has_component<Zone>(pd))
            gs->pending_decision_ctrl_is_self =
                (global_coordinator.GetComponent<Zone>(pd).owner == viewer);
        else if (sideboard_phase)
            // A sideboard IN/OUT source is a bare load_card template entity with
            // neither Permanent nor Zone; the sideboarding player owns it.
            gs->pending_decision_ctrl_is_self = (sideboard_phase_player == viewer);
    }

    // Match context (extern globals from main.cpp)
    extern int match_game_number;
    extern int match_wins_a;
    extern int match_wins_b;
    extern bool sideboard_phase;
    gs->match_game_number = match_game_number;
    if (viewer == Zone::PLAYER_A) {
        gs->match_wins_self = match_wins_a;
        gs->match_wins_opp  = match_wins_b;
    } else {
        gs->match_wins_self = match_wins_b;
        gs->match_wins_opp  = match_wins_a;
    }
    gs->is_sideboard_phase = sideboard_phase;

    // Viewer's known top-of-library cache
    const int* viewer_known = (viewer == Zone::PLAYER_A)
        ? cur_game.known_top_library_a : cur_game.known_top_library_b;
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++)
        gs->known_top_library_self[i] = viewer_known[i];

    // Opponent-of-viewer's match-scoped revealed-cards multi-hot.
    const unsigned char* opp_revealed = (viewer == Zone::PLAYER_A)
        ? g_revealed_by_b : g_revealed_by_a;
    for (int i = 0; i < REVEALED_CARD_TYPES; i++)
        gs->opp_revealed[i] = opp_revealed[i];

    // Fill player stat fields (hand_ct filled in the entity pass below)
    auto fill_player_stats = [&](PlayerState& ps, Entity ent) {
        auto& p = global_coordinator.GetComponent<Player>(ent);
        ps.life = p.life_total;
        ps.poison_counters = p.counter_count("POISON");
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

    // Stack items collected during the pass, sorted by distance_from_top afterward
    // (the engine's true stack order; 0 = top of stack).
    struct StackItem { size_t dist; StackEntry entry; };
    StackItem stack_items[MAX_STACK_DISPLAY + 8];
    int stack_item_count = 0;

    // Slot fill counters
    int self_bf = 0, opp_bf = 0;
    int self_gy = 0, opp_gy = 0;
    int self_hand_idx = 0;
    int opp_known_hand_idx = 0;

    // Single pass over all entities (naturally ascending entity ID order).
    // Use high-water-mark instead of MAX_ENTITIES to skip unallocated slots.
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e) {
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
                    // Opponent-hand cards the viewer has had revealed are carried by
                    // their specific identity (not just the match-scoped multi-hot).
                    if (zone.identity_known && opp_known_hand_idx < MAX_HAND_SLOTS)
                        gs->opp_known_hand[opp_known_hand_idx++] = get_card_vocab_idx(e);
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

            // Exile is intentionally not collected: it is never serialized to the
            // ML state (see gamestate.h). Revisit when exile-using cards matter.

            case Zone::STACK: {
                gs->stack_size++;
                if (stack_item_count < MAX_STACK_DISPLAY + 8) {
                    StackEntry se = {};
                    se.card_vocab_idx    = action_card_vocab_idx(e);
                    se.controller_is_self = (zone.owner == viewer);
                    se.is_spell           = global_coordinator.entity_has_component<Spell>(e);
                    for (int t = 0; t < MAX_STACK_TGTS; t++) se.targets[t].card_vocab_idx = -1;
                    if (global_coordinator.entity_has_component<Ability>(e)) {
                        const auto& ab = global_coordinator.GetComponent<Ability>(e);
                        fill_stack_choices(ab, se, viewer);
                        if (ab.target != 0) {
                            std::string tname = target_display_name(cur_game, ab.target);
                            strncpy(se.target_name, tname.c_str(), sizeof(se.target_name) - 1);
                            se.target_name[sizeof(se.target_name) - 1] = '\0';
                        }
                    }
                    stack_items[stack_item_count++] = {zone.distance_from_top, se};
                }
                break;
            }

            case Zone::BATTLEFIELD: {
                // Phased-out permanents are treated as nonexistent (702.26e): don't serialize.
                if (!is_battlefield_permanent(e)) break;
                auto& perm = global_coordinator.GetComponent<Permanent>(e);

                PermanentState ps;
                ps.card_vocab_idx        = get_card_vocab_idx(e);
                ps.controller_is_self    = (perm.controller == viewer);
                ps.is_tapped             = perm.is_tapped;
                ps.has_summoning_sickness = perm.has_summoning_sickness;
                ps.is_creature           = global_coordinator.entity_has_component<Creature>(e);
                // Inline land check using already-retrieved perm.types (avoids redundant GetComponent)
                ps.is_land = false;
                for (auto& t : perm.types) {
                    if (t.name == "Land") { ps.is_land = true; break; }
                }

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

                ps.loyalty = get_counters(e, "LOYALTY");  // nonzero only for planeswalkers

                format_counter_summary(perm.counters, ps.counters, sizeof(ps.counters));

                ps.token_name[0] = '\0';
                if (perm.is_token && global_coordinator.entity_has_component<Token>(e)) {
                    const auto& tok = global_coordinator.GetComponent<Token>(e);
                    strncpy(ps.token_name, tok.name.c_str(), sizeof(ps.token_name) - 1);
                    ps.token_name[sizeof(ps.token_name) - 1] = '\0';
                }

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

    // Sort stack entries by distance_from_top ascending (index 0 = top of stack)
    std::sort(stack_items, stack_items + stack_item_count,
              [](const StackItem& a, const StackItem& b) { return a.dist < b.dist; });
    int copy_count = std::min(stack_item_count, MAX_STACK_DISPLAY);
    for (int i = 0; i < copy_count; i++)
        gs->stack[i] = stack_items[i].entry;

    // Action history: copy from ring buffer, newest first, with perspective normalization
    gs->action_history_len = cur_game.action_history_count;
    bool viewer_is_a = (viewer == Zone::PLAYER_A);
    float cat_max = static_cast<float>(ACTION_CATEGORY_MAX);
    float card_types = static_cast<float>(N_CARD_TYPES);
    float id_null = -1.0f / card_types;
    for (int i = 0; i < ACTION_HISTORY_SIZE; i++) {
        int base = i * 4;
        if (i < gs->action_history_len) {
            // Read newest first: walk backwards from write position
            int ring_idx = (cur_game.action_history_write - 1 - i + ACTION_HISTORY_SIZE) % ACTION_HISTORY_SIZE;
            const auto& entry = cur_game.action_history[ring_idx];
            gs->action_history[base + 0] = static_cast<float>(entry.category) / cat_max;
            gs->action_history[base + 1] = entry.card_vocab_idx >= 0
                ? static_cast<float>(entry.card_vocab_idx) / card_types
                : id_null;
            gs->action_history[base + 2] = (entry.player_a == viewer_is_a) ? 1.0f : 0.0f;
            gs->action_history[base + 3] = static_cast<float>(entry.turn) / TURN_NORMALIZER;
        } else {
            gs->action_history[base + 0] = 0.0f;
            gs->action_history[base + 1] = 0.0f;
            gs->action_history[base + 2] = 0.0f;
            gs->action_history[base + 3] = 0.0f;
        }
    }
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

        Entity src = la.source_entity;

        // Card vocab index from source entity (or ability source). Shared with the
        // action log (input_logger) so the logged and emitted ids cannot diverge.
        // The LegalAction overload also resolves a modal-DFC back-face play to the
        // back face's id (front-face source entity would otherwise mis-report it).
        ac.card_vocab_idx = action_card_vocab_idx(la);

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

        ac.card_is_public = la.card_is_public;

        snprintf(ac.description, MAX_CHOICE_DESC, "%s", la.description.c_str());
    }
}

// ── serialize_state ───────────────────────────────────────────────────────────

const std::vector<float>& serialize_state(const GameState* gs) {
    // Reused across calls: the game loop is single-threaded and the caller consumes the
    // result (fwrite) before the next call, so a thread_local scratch buffer is safe.
    // clear() keeps the capacity from the first call, so subsequent calls don't realloc
    // the ~135 KB vector that was previously heap-allocated every decision.
    static thread_local std::vector<float> state;
    state.clear();
    state.reserve(static_cast<size_t>(STATE_SIZE));

    // Header: self (9) + opp (9) + step one-hot (13) + flags (3) = 34
    push_player_block(state, gs->self);
    push_player_block(state, gs->opponent);
    for (int i = 0; i < 13; i++)
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

    // Stack (12 x 25 = 300): controller_is_self(1) + card_id(1) + is_spell(1) +
    // chosen-mode multi-hot(6) + 4 announced-target sub-slots x
    // [present, is_player, controller_is_self, card_id](16). See machine_io.h.
    int stored_stack = std::min(gs->stack_size, MAX_STACK_DISPLAY);
    for (int i = 0; i < MAX_STACK_DISPLAY; i++) {
        if (i < stored_stack) {
            const StackEntry& se = gs->stack[i];
            state.push_back(se.controller_is_self ? 1.0f : 0.0f);
            state.push_back(norm_card_id(se.card_vocab_idx));
            state.push_back(se.is_spell ? 1.0f : 0.0f);
            for (int m = 0; m < MAX_STACK_MODES; m++)
                state.push_back(se.chosen_modes[m] ? 1.0f : 0.0f);
            for (int t = 0; t < MAX_STACK_TGTS; t++) {
                const StackTarget& st = se.targets[t];
                state.push_back(st.present ? 1.0f : 0.0f);
                state.push_back(st.is_player ? 1.0f : 0.0f);
                state.push_back(st.controller_is_self ? 1.0f : 0.0f);
                state.push_back(norm_card_id(st.card_vocab_idx));
            }
        } else {
            state.push_back(0.0f);
            state.push_back(norm_card_id(-1));
            state.push_back(0.0f);
            for (int m = 0; m < MAX_STACK_MODES; m++) state.push_back(0.0f);
            for (int t = 0; t < MAX_STACK_TGTS; t++) {
                state.push_back(0.0f);
                state.push_back(0.0f);
                state.push_back(0.0f);
                state.push_back(norm_card_id(-1));
            }
        }
    }

    // Self graveyard (64 x 1 = 64)
    for (int i = 0; i < MAX_GY_SLOTS; i++)
        state.push_back(norm_card_id(gs->self_graveyard[i]));

    // Opp graveyard (64 x 1 = 64)
    for (int i = 0; i < MAX_GY_SLOTS; i++)
        state.push_back(norm_card_id(gs->opp_graveyard[i]));

    // NOTE: exile zones are populated in GameState but not serialized here.
    // Add them back once cards that use exile are implemented.

    // Self hand (10 x 1 = 10)
    for (int i = 0; i < MAX_HAND_SLOTS; i++)
        state.push_back(norm_card_id(gs->self_hand[i]));

    // Action history (128 x 4 = 512, newest first)
    for (int i = 0; i < ACTION_HISTORY_SIZE * 4; i++)
        state.push_back(gs->action_history[i]);

    // Match context (4 floats, all 0.0 in single-game mode)
    state.push_back(gs->match_game_number >= 0 ? static_cast<float>(gs->match_game_number) / 3.0f : 0.0f);
    state.push_back(static_cast<float>(gs->match_wins_self) / 2.0f);
    state.push_back(static_cast<float>(gs->match_wins_opp) / 2.0f);
    state.push_back(gs->is_sideboard_phase ? 1.0f : 0.0f);

    // Library counts & post-board flag (3 floats)
    state.push_back(static_cast<float>(gs->self_library_ct) / 60.0f);
    state.push_back(static_cast<float>(gs->opp_library_ct) / 60.0f);
    state.push_back(gs->match_game_number > 0 ? 1.0f : 0.0f);

    // Current turn (1 float)
    state.push_back(static_cast<float>(gs->turn) / TURN_NORMALIZER);

    // Known top-of-library cards for the viewer (5 slots x 1 float = 5)
    // Sentinel id = unknown.
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++)
        state.push_back(norm_card_id(gs->known_top_library_self[i]));

    // Opponent revealed-cards multi-hot (N_CARD_TYPES floats; all zeros = none seen yet).
    // Accumulated across the match, perspective-relative to the viewer.
    for (int i = 0; i < REVEALED_CARD_TYPES; i++)
        state.push_back(gs->opp_revealed[i] ? 1.0f : 0.0f);

    // Known opponent-hand cards (10 x 1 = 10): specific card identities the viewer
    // has had revealed from the opponent's hand and that are still in hand. Sentinel
    // id = empty/unknown slot. Distinct from the multi-hot above: this tracks the
    // exact card and clears when that card leaves the hand.
    for (int i = 0; i < MAX_HAND_SLOTS; i++)
        state.push_back(norm_card_id(gs->opp_known_hand[i]));

    // Pending decision context (2 floats): card id of the spell/ability making the
    // current mid-resolution choice (sentinel = none) + its controller-is-viewer flag.
    state.push_back(norm_card_id(gs->pending_decision_card));
    state.push_back(gs->pending_decision_ctrl_is_self ? 1.0f : 0.0f);

    assert(static_cast<int>(state.size()) == STATE_SIZE);
    return state;
}
