#include "machine_io.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

#include "card_vocab.h"
#include "error.h"
#include "classes/deck_state.h"
#include "classes/game.h"
#include "classes/match_context.h"
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

static int token_vocab_idx(Entity e);
static int get_card_vocab_idx(Entity e);
static int slot_ref_of(Entity e);
static void push_player_block(std::vector<float>& out, const PlayerState& ps);
static void push_perm_slot(std::vector<float>& out, const PermanentState& p);
static void format_counter_summary(const CounterMap& counters, char* buf, size_t buf_len);
static void add_stack_target(StackEntry& se, int& n, Entity tgt, Zone::Ownership viewer);
static void fill_stack_choices(const Ability& ab, StackEntry& se, Zone::Ownership viewer);
static void fill_permanent_state(PermanentState& ps, Entity e, Zone::Ownership viewer);
static void fill_stack_entry(StackEntry& se, Entity e, Zone::Ownership viewer);
static void fill_decklist_block(int* ids, int* counts, int n_slots,
                                const std::vector<DecklistEntry>& entries,
                                const char* block_name);

// ── Entity → reference-slot map ───────────────────────────────────────────────
// Maps every serialized entity to its slot in the unified viewer-relative
// reference space (0-47 self perms in pack order, 48-95 opp perms, 96-107 stack
// top-first; see machine_io.h). Rebuilt from scratch — cleared, then filled — by
// every populate_gamestate call (pass A collects, the map is built, pass B fills
// the ref fields through it), and consumed again by populate_query for the
// per-action slot_refs.
//
// Staleness invariant: the machine-mode emit path is always
//   populate_gamestate(); populate_query(); cli_emit_machine_query();
// back-to-back for one decision (input_logger.cpp), with no game mutation in
// between. Calling populate_query WITHOUT a preceding populate_gamestate for the
// same decision is a programming error — it would resolve slot_refs against the
// previous decision's stale map.
static std::unordered_map<Entity, int> g_entity_slot_map;

// Reference-slot of `e` in the unified space, or -1 when the entity is not
// serialized there (players, hand/GY/library cards, truncated overflow, e == 0).
static int slot_ref_of(Entity e) {
    if (e == 0) return -1;
    auto it = g_entity_slot_map.find(e);
    return it != g_entity_slot_map.end() ? it->second : -1;
}

// Vocab index for a token entity: its registered token-band index (keyed by the
// token SCRIPT stem — display names collide, stems don't), else TOKEN_SENTINEL.
static int token_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Token>(e)) {
        int idx = token_script_to_index(global_coordinator.GetComponent<Token>(e).script_name);
        if (idx >= 0) return idx;
    }
    return TOKEN_SENTINEL;
}

static int get_card_vocab_idx(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto& perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token) return token_vocab_idx(e);
        return card_name_to_index(perm.name);
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return token_vocab_idx(e);
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
        return perm.is_token ? token_vocab_idx(e) : card_name_to_index(perm.name);
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    if (global_coordinator.entity_has_component<Ability>(e)) {
        Entity src = global_coordinator.GetComponent<Ability>(e).source;
        if (global_coordinator.entity_has_component<Permanent>(src)) {
            auto& sp = global_coordinator.GetComponent<Permanent>(src);
            return sp.is_token ? token_vocab_idx(src) : card_name_to_index(sp.name);
        }
        if (global_coordinator.entity_has_component<CardData>(src))
            return card_name_to_index(global_coordinator.GetComponent<CardData>(src).name);
        // An ability whose token source has already left play keeps no Permanent/CardData;
        // a lingering Token component still identifies it as a token (stack extractor case).
        if (global_coordinator.entity_has_component<Token>(src))
            return token_vocab_idx(src);
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
    // Instance-level join: which serialized slot the target occupies (-1 for players
    // and entities outside the reference space). Requires the entity->slot map to be
    // built first, so stack entries are filled in populate_gamestate's pass B.
    st.slot_ref = slot_ref_of(tgt);
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
    out.push_back(static_cast<float>(ps.energy) / 10.0f);
}

// Pushes PERM_SLOT_SIZE floats (35 status + chosen-name id + returnable-exile id + card-id;
// per-slot offsets documented in machine_io.h). Empty slot (card_vocab_idx == -1) = 35 zeros
// + THREE id-family empty sentinels (chosen-name, returnable-exile, card-id; a 0.0 pad would
// alias vocab index 0 and defeat empty-slot masking).
static void push_perm_slot(std::vector<float>& out, const PermanentState& p) {
    if (p.card_vocab_idx == -1) {
        out.insert(out.end(), PERM_SLOT_SIZE - 3, 0.0f);
        out.push_back(norm_card_id(-1));  // chosen_name_idx sentinel
        out.push_back(norm_card_id(-1));  // returnable_exile_idx sentinel
        out.push_back(norm_card_id(-1));  // card_id sentinel
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
    out.push_back(static_cast<float>(p.p1p1_net) / 10.0f);  // signed
    out.push_back(static_cast<float>(p.other_counters) / 10.0f);
    out.push_back(norm_ref(p.attached_to_ref));
    out.push_back(norm_ref(p.attached_by_ref));
    out.push_back(norm_ref(p.attack_target_ref));
    out.push_back(norm_ref(p.blocking_target_ref));
    out.push_back(p.is_blocked ? 1.0f : 0.0f);
    out.push_back(p.is_phased_out ? 1.0f : 0.0f);
    for (int k = 0; k < N_OBS_KEYWORDS; k++)
        out.push_back(p.keywords[k] ? 1.0f : 0.0f);
    out.push_back(norm_card_id(p.chosen_name_idx));      // [35] chosen-name id
    out.push_back(norm_card_id(p.returnable_exile_idx)); // [36] returnable-exile id
    out.push_back(norm_card_id(p.card_vocab_idx));       // [37] card id (LAST)
}

// Pass-B fill of one battlefield permanent's PermanentState. Runs after the
// entity->slot map is built so the attachment/combat reference fields resolve.
static void fill_permanent_state(PermanentState& ps, Entity e, Zone::Ownership viewer) {
    auto& perm = global_coordinator.GetComponent<Permanent>(e);

    ps.card_vocab_idx        = get_card_vocab_idx(e);
    // Named card chosen on this permanent (Pithing Needle / Disruptor Flute named
    // card, Petrified Hamlet named land). card_name_to_index returns -1 for an
    // empty or out-of-vocab name, which norm_card_id maps to the empty sentinel.
    ps.chosen_name_idx       = perm.chosen_name.empty()
                                   ? -1 : card_name_to_index(perm.chosen_name);
    // Most recently exiled card linked to this permanent that still has a live return path (a
    // Static Prison holding a real card, a Flickerwisp/Phelia EOT blink) — 0/none => sentinel.
    // Use the guarded vocab-idx helper (an exiled card is not a Permanent; an exiled token can't
    // persist, but the helper handles the token case regardless).
    Entity returnable = returnable_exiled_card(e);
    ps.returnable_exile_idx  = returnable == 0 ? -1 : get_card_vocab_idx(returnable);
    ps.controller_is_self    = (perm.controller == viewer);
    ps.is_tapped             = perm.is_tapped;
    ps.has_summoning_sickness = perm.has_summoning_sickness;
    ps.is_creature           = global_coordinator.entity_has_component<Creature>(e);
    // Inline land check using already-retrieved perm.types (avoids redundant GetComponent)
    ps.is_land = false;
    for (auto& t : perm.types) {
        if (t.name == "Land") { ps.is_land = true; break; }
    }

    ps.attack_target_ref   = -1;
    ps.blocking_target_ref = -1;
    if (ps.is_creature) {
        auto& cr     = global_coordinator.GetComponent<Creature>(e);
        ps.power     = static_cast<int>(cr.power);
        ps.toughness = static_cast<int>(cr.toughness);
        ps.is_attacking = cr.is_attacking;
        ps.is_blocking  = cr.is_blocking;
        ps.is_blocked   = cr.is_blocked;
        // attack_target is a player or planeswalker entity; a player is not in the
        // reference space, so "attacking the player" serializes as -1 (+ is_attacking).
        ps.attack_target_ref   = slot_ref_of(cr.attack_target);
        ps.blocking_target_ref = slot_ref_of(cr.blocking_target);
    } else {
        ps.power = ps.toughness = 0;
        ps.is_attacking = ps.is_blocking = ps.is_blocked = false;
    }

    ps.damage = 0;
    if (global_coordinator.entity_has_component<Damage>(e))
        ps.damage = static_cast<int>(global_coordinator.GetComponent<Damage>(e).damage_counters);

    ps.loyalty = get_counters(e, "LOYALTY");  // nonzero only for planeswalkers
    ps.p1p1_net = get_counters(e, "P1P1") - get_counters(e, "M1M1");
    ps.other_counters = 0;
    for (const auto& c : perm.counters)
        if (c.first != "P1P1" && c.first != "M1M1" && c.first != "LOYALTY")
            ps.other_counters += c.second;

    ps.attached_to_ref = slot_ref_of(perm.equipped_to);
    ps.attached_by_ref = slot_ref_of(perm.equipped_by);
    ps.is_phased_out   = perm.is_phased_out;

    for (int k = 0; k < N_OBS_KEYWORDS; k++)
        ps.keywords[k] = permanent_has_keyword(e, OBS_KEYWORDS[k]);

    format_counter_summary(perm.counters, ps.counters, sizeof(ps.counters));

    ps.token_name[0] = '\0';
    if (perm.is_token && global_coordinator.entity_has_component<Token>(e)) {
        const auto& tok = global_coordinator.GetComponent<Token>(e);
        strncpy(ps.token_name, tok.name.c_str(), sizeof(ps.token_name) - 1);
        ps.token_name[sizeof(ps.token_name) - 1] = '\0';
    }
}

// Pass-B fill of one stack object's StackEntry. Runs after the entity->slot map is
// built so the announced targets' slot_refs (add_stack_target) resolve.
static void fill_stack_entry(StackEntry& se, Entity e, Zone::Ownership viewer) {
    se = StackEntry{};
    se.card_vocab_idx     = action_card_vocab_idx(e);
    se.controller_is_self = (global_coordinator.GetComponent<Zone>(e).owner == viewer);
    se.is_spell           = global_coordinator.entity_has_component<Spell>(e);
    for (int t = 0; t < MAX_STACK_TGTS; t++) {
        se.targets[t].card_vocab_idx = -1;
        se.targets[t].slot_ref       = -1;
    }

    // X / amount + cast qualifiers: public info announced at cast (CR 601.2b/f) for a
    // spell; a triggered/activated ability carries its Ability::amount and zero qualifiers.
    if (se.is_spell) {
        const auto& sp = global_coordinator.GetComponent<Spell>(e);
        se.x_or_amount         = sp.x_paid;
        se.is_copy             = sp.is_copy;
        for (bool k : sp.kicked)
            if (k) { se.kicked_any = true; break; }
        se.cast_with_flashback = sp.cast_with_flashback;
        se.cast_with_evoke     = sp.cast_with_evoke;
        se.cast_with_escape    = sp.cast_with_escape;
        se.cast_with_offspring = sp.cast_with_offspring;
        se.cast_with_impending = sp.cast_with_impending;
    }

    if (global_coordinator.entity_has_component<Ability>(e)) {
        const auto& ab = global_coordinator.GetComponent<Ability>(e);
        if (!se.is_spell) se.x_or_amount = static_cast<int>(ab.amount);
        fill_stack_choices(ab, se, viewer);
        if (ab.target != 0) {
            std::string tname = target_display_name(cur_game, ab.target);
            strncpy(se.target_name, tname.c_str(), sizeof(se.target_name) - 1);
            se.target_name[sizeof(se.target_name) - 1] = '\0';
        }
    }
}

// Fill a deck-identity block (id + count slot arrays) from a sorted (ascending by
// vocab id) DecklistEntry list. Packs into the leading slots with no holes; the
// caller has pre-marked every slot empty (id = -1, count = 0). Fatal on overflow
// (loud, never a silent truncation) — deck_state_set already guards the static
// lists, but the live-library caller passes a freshly-built list, so re-guard here.
static void fill_decklist_block(int* ids, int* counts, int n_slots,
                                const std::vector<DecklistEntry>& entries,
                                const char* block_name) {
    if (static_cast<int>(entries.size()) > n_slots)
        fatal_error("serialize_state: " + std::string(block_name) + " has " +
                    std::to_string(entries.size()) + " distinct card names, exceeds " +
                    std::to_string(n_slots) + " serialized slots");
    for (size_t i = 0; i < entries.size(); i++) {
        ids[i]    = entries[i].vocab_idx;
        counts[i] = entries[i].count;
    }
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
        gs->self_exile[i]     = -1;
        gs->opp_exile[i]      = -1;
    }
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++) gs->known_top_library_self[i] = -1;
    // Deck-identity tail blocks: id = -1 (empty sentinel), count 0.
    for (int i = 0; i < DECKLIST_MAIN_SLOTS; i++) {
        gs->self_live_library_id[i] = -1;
        gs->self_live_library_ct[i] = 0;
        gs->self_deck_main_id[i]    = -1;
        gs->self_deck_main_ct[i]    = 0;
        gs->opp_deck_main_id[i]     = -1;
        gs->opp_deck_main_ct[i]     = 0;
    }
    for (int i = 0; i < DECKLIST_SIDE_SLOTS; i++) {
        gs->self_deck_side_id[i] = -1;
        gs->self_deck_side_ct[i] = 0;
        gs->opp_deck_side_id[i]  = -1;
        gs->opp_deck_side_ct[i]  = 0;
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
    extern const SideboardPhaseState *sideboard_phase_state;
    extern MatchContext g_match_ctx;
    // During the between-games phase the observation is ABOUT the upcoming game, so
    // report that game's index rather than the one that just ended (which left
    // match_game_number behind). Without this the game-1->2 sideboard root reports
    // game_number 0 and is_post_board 0, describing a game already over.
    gs->match_game_number = sideboard_phase ? match_game_number + 1 : match_game_number;
    if (viewer == Zone::PLAYER_A) {
        gs->match_wins_self = match_wins_a;
        gs->match_wins_opp  = match_wins_b;
    } else {
        gs->match_wins_self = match_wins_b;
        gs->match_wins_opp  = match_wins_a;
    }
    gs->is_sideboard_phase = sideboard_phase;

    // Starting player of the game this observation pertains to. During the phase
    // that is the UPCOMING game, whose starting player play_bo3_match already fixed
    // (the loser of the game that just ended) before either sideboard stage ran.
    const bool a_first = sideboard_phase ? g_match_ctx.a_goes_first
                                         : cur_game.pregame.a_goes_first;
    gs->self_plays_first = (a_first == (viewer == Zone::PLAYER_A));

    // Sideboard-phase progress, read straight off the running phase state so it
    // cannot drift from the phase's own bookkeeping (null outside the phase).
    gs->sideboard_swaps_made = sideboard_phase_state ? sideboard_phase_state->sb_swaps : 0;
    gs->sideboard_delta      = sideboard_phase_state ? sideboard_phase_state->delta : 0;

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
        ps.energy = player_energy(p);
        ps.lands_played_this_turn = static_cast<int>(p.lands_played_this_turn);
        ps.city_blessing = p.has_city_blessing;
        int mana_counts[6] = {};
        for (Colors c : p.mana) {
            int idx = static_cast<int>(c);
            if (idx >= 0 && idx < 6) mana_counts[idx]++;
        }
        for (int i = 0; i < 6; i++) ps.mana[i] = mana_counts[i];
    };
    fill_player_stats(gs->self, viewer_entity);
    fill_player_stats(gs->opponent, opp_entity);

    // Global extras (serialized at the end of the state vector; see machine_io.h)
    gs->self.is_monarch     = (cur_game.monarch_entity == viewer_entity);
    gs->opponent.is_monarch = (cur_game.monarch_entity == opp_entity);
    bool viewer_is_player_a = (viewer == Zone::PLAYER_A);
    gs->self.revolt     = viewer_is_player_a ? cur_game.revolt_player_a : cur_game.revolt_player_b;
    gs->opponent.revolt = viewer_is_player_a ? cur_game.revolt_player_b : cur_game.revolt_player_a;
    for (Zone::Ownership et : cur_game.extra_turns) {
        if (et == viewer) gs->self.extra_turns_pending++;
        else              gs->opponent.extra_turns_pending++;
    }
    gs->is_day   = (cur_game.day_night == Game::DN_DAY);
    gs->is_night = (cur_game.day_night == Game::DN_NIGHT);
    gs->pending_choice_kind = static_cast<int>(cur_game.pending_choice);

    // ── Pass A (collect) ─────────────────────────────────────────────────────
    // One ascending-entity-ID scan collects the entities of every serialized zone;
    // battlefield/stack fills happen in pass B once the entity->slot map exists, so
    // reference fields (attachments, combat pairing, stack-target slots) can resolve
    // forward as well as backward.

    // Stack items sorted by distance_from_top after the scan (the engine's true
    // stack order; 0 = top of stack).
    struct StackItem { size_t dist; Entity ent; };
    StackItem stack_items[MAX_STACK_DISPLAY + 8];
    int stack_item_count = 0;

    // Graveyard/exile cards as (distance_from_top, vocab id), sorted for recency order.
    struct GyItem { size_t dist; int vocab_idx; };
    std::vector<GyItem> self_gy_items, opp_gy_items;
    std::vector<GyItem> self_exile_items, opp_exile_items;
    self_gy_items.reserve(MAX_GY_SLOTS);
    opp_gy_items.reserve(MAX_GY_SLOTS);
    self_exile_items.reserve(MAX_GY_SLOTS);
    opp_exile_items.reserve(MAX_GY_SLOTS);

    // Viewer's LIVE library contents tallied by vocab id (std::map keeps it sorted
    // ascending — the required packed slot order). Viewer-only; the opponent's live
    // library is hidden. A negative/sentinel id can't occur (every library card is a
    // vocab-registered deck card, already validated by deck_state_set) but is skipped
    // defensively so a stray one never lands in a slot.
    std::map<int, int> self_live_lib;

    // Battlefield permanents in pack order (ascending entity ID)
    Entity self_ents[MAX_BATTLEFIELD_SLOTS];
    Entity opp_ents[MAX_BATTLEFIELD_SLOTS];
    int self_bf = 0, opp_bf = 0;
    int self_hand_idx = 0;
    int opp_known_hand_idx = 0;

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
                if (is_self) {
                    gs->self_library_ct++;
                    int lid = get_card_vocab_idx(e);
                    if (lid >= 0 && lid != TOKEN_SENTINEL) self_live_lib[lid]++;
                } else {
                    gs->opp_library_ct++;
                }
                break;

            case Zone::GRAVEYARD:
                if (is_self) self_gy_items.push_back({zone.distance_from_top, get_card_vocab_idx(e)});
                else         opp_gy_items.push_back({zone.distance_from_top, get_card_vocab_idx(e)});
                break;

            case Zone::EXILE:
                // Collected per-owner in recency order, exactly like the graveyard.
                // Most exile is public, so both sides are serialized. The exception is a card
                // exiled FACE DOWN (CR 708.2, The Creation of Avacyn chapter I): its identity is
                // hidden from the opponent, so an opponent-owned face-down exile emits the unknown
                // id sentinel (-1) — the viewer still sees a card is there, just not which. The
                // owner (who exiled it from their own library) still sees its true identity.
                // get_card_vocab_idx guards a missing CardData (a token that ever
                // sits here resolves via its Token band / TOKEN_SENTINEL, no crash).
                if (is_self) self_exile_items.push_back({zone.distance_from_top, get_card_vocab_idx(e)});
                else         opp_exile_items.push_back({zone.distance_from_top,
                                 zone.is_face_down ? -1 : get_card_vocab_idx(e)});
                break;

            case Zone::STACK:
                gs->stack_size++;
                if (stack_item_count < MAX_STACK_DISPLAY + 8)
                    stack_items[stack_item_count++] = {zone.distance_from_top, e};
                break;

            case Zone::BATTLEFIELD:
                // Serialization exception to the phasing rule (see game_queries.h):
                // phased-out permanents ARE collected — their slot stays visible with
                // is_phased_out set — so the explicit Permanent check replaces
                // is_battlefield_permanent (the zone is already known from the switch).
                if (!global_coordinator.entity_has_component<Permanent>(e)) break;
                if (global_coordinator.GetComponent<Permanent>(e).controller == viewer) {
                    if (self_bf < MAX_BATTLEFIELD_SLOTS) self_ents[self_bf++] = e;
                } else {
                    if (opp_bf < MAX_BATTLEFIELD_SLOTS) opp_ents[opp_bf++] = e;
                }
                break;

            default:
                break;
        }
    }

    // Sort stack entries by distance_from_top ascending (index 0 = top of stack)
    std::sort(stack_items, stack_items + stack_item_count,
              [](const StackItem& a, const StackItem& b) { return a.dist < b.dist; });
    int stored_stack = std::min(stack_item_count, MAX_STACK_DISPLAY);

    // Build the entity->slot reference map (see its declaration for the invariant):
    // 0-47 self perms, 48-95 opp perms, 96-107 stack top-first; overflow stays absent.
    g_entity_slot_map.clear();
    for (int i = 0; i < self_bf; i++)
        g_entity_slot_map[self_ents[i]] = i;
    for (int i = 0; i < opp_bf; i++)
        g_entity_slot_map[opp_ents[i]] = MAX_BATTLEFIELD_SLOTS + i;
    for (int i = 0; i < stored_stack; i++)
        g_entity_slot_map[stack_items[i].ent] = 2 * MAX_BATTLEFIELD_SLOTS + i;

    // ── Pass B (fill) ────────────────────────────────────────────────────────
    for (int i = 0; i < self_bf; i++)
        fill_permanent_state(gs->self_permanents[i], self_ents[i], viewer);
    for (int i = 0; i < opp_bf; i++)
        fill_permanent_state(gs->opp_permanents[i], opp_ents[i], viewer);
    for (int i = 0; i < stored_stack; i++)
        fill_stack_entry(gs->stack[i], stack_items[i].ent, viewer);

    // Graveyards in RECENCY order: slot 0 = most recent arrival (lowest distance_from_top)
    auto fill_graveyard = [](int* slots, std::vector<GyItem>& items) {
        std::sort(items.begin(), items.end(),
                  [](const GyItem& a, const GyItem& b) { return a.dist < b.dist; });
        int n = std::min(static_cast<int>(items.size()), MAX_GY_SLOTS);
        for (int i = 0; i < n; i++) slots[i] = items[static_cast<size_t>(i)].vocab_idx;
    };
    fill_graveyard(gs->self_graveyard, self_gy_items);
    fill_graveyard(gs->opp_graveyard, opp_gy_items);

    // Exile zones in the same RECENCY order (slot 0 = most recent arrival).
    fill_graveyard(gs->self_exile, self_exile_items);
    fill_graveyard(gs->opp_exile, opp_exile_items);

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

    // ── Deck-identity tail blocks ─────────────────────────────────────────────
    // Self LIVE library (packed ascending by vocab id from the std::map tally).
    {
        std::vector<DecklistEntry> live;
        live.reserve(self_live_lib.size());
        for (const auto& kv : self_live_lib) live.push_back({kv.first, kv.second});
        fill_decklist_block(gs->self_live_library_id, gs->self_live_library_ct,
                            DECKLIST_MAIN_SLOTS, live, "self live library");
    }
    // Viewer's OWN current 75, from deck_state's LIVE store — the deck the viewer
    // is actually piloting, tracking every sideboard swap as it lands.
    fill_decklist_block(gs->self_deck_main_id, gs->self_deck_main_ct,
                        DECKLIST_MAIN_SLOTS, deck_state_live_main(viewer),
                        "self maindeck");
    fill_decklist_block(gs->self_deck_side_id, gs->self_deck_side_ct,
                        DECKLIST_SIDE_SLOTS, deck_state_live_side(viewer),
                        "self sideboard");
    // Opponent-of-viewer REGISTERED decklist (maindeck + sideboard). Frozen at
    // the match's registered 75 — deliberately NOT the post-board split, which is
    // hidden information in game 2+ (see deck_state.h).
    Zone::Ownership opp_owner = (viewer == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
    fill_decklist_block(gs->opp_deck_main_id, gs->opp_deck_main_ct,
                        DECKLIST_MAIN_SLOTS, deck_state_registered_main(opp_owner),
                        "opp maindeck");
    fill_decklist_block(gs->opp_deck_side_id, gs->opp_deck_side_ct,
                        DECKLIST_SIDE_SLOTS, deck_state_registered_side(opp_owner),
                        "opp sideboard");
}

// ── populate_query ────────────────────────────────────────────────────────────

void populate_query(Query* q, const std::vector<LegalAction>& actions) {
    memset(q, 0, sizeof(*q));
    int n = std::min(static_cast<int>(actions.size()), MAX_ACTIONS);
    if (static_cast<int>(actions.size()) > MAX_ACTIONS)
        // Machine-mode agents can never pick a truncated action, so an
        // over-wide menu silently restricts the policy — make it observable.
        fprintf(stderr,
                "WARNING: legal-action menu has %zu entries; machine query "
                "truncated to MAX_ACTIONS=%d — choices beyond that are "
                "unreachable for machine-mode agents\n",
                actions.size(), MAX_ACTIONS);
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

        // Action <-> entity join: the source's slot in the unified reference space.
        // Resolved through the entity->slot map built by populate_gamestate — valid
        // only because the emit path always runs populate_gamestate immediately
        // before populate_query (see the map's staleness invariant above).
        ac.slot_ref = slot_ref_of(src);

        // Controller is self
        ac.controller_is_self = false;
        if (src != 0) {
            if (global_coordinator.entity_has_component<Permanent>(src))
                ac.controller_is_self = (global_coordinator.GetComponent<Permanent>(src).controller == priority_owner);
            else if (src == priority_ent)
                ac.controller_is_self = true;
            else if (global_coordinator.entity_has_component<Zone>(src))
                ac.controller_is_self = (global_coordinator.GetComponent<Zone>(src).owner == priority_owner);
            // A sideboard IN/OUT source is a bare load_card template entity with
            // neither Permanent nor Zone, so controller_is_self stays false; the
            // emitter (cli_output) then writes the ctrl-null sentinel because its
            // zone_ref is REF_NONE. That's the correct "zone-less/unknown" encoding
            // — the card id is still emitted, so the choice's identity is visible.
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
                    ac.zone_ref = is_self_owned ? REF_SELF_HAND : REF_OPP_HAND;
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
        ac.option_ordinal = la.option_ordinal;

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

    // Header: self (10) + opp (10) + step one-hot (13) + flags (3) = 36
    push_player_block(state, gs->self);
    push_player_block(state, gs->opponent);
    for (int i = 0; i < 13; i++)
        state.push_back((gs->cur_step == static_cast<Step>(i)) ? 1.0f : 0.0f);
    state.push_back(gs->is_active_player ? 1.0f : 0.0f);
    state.push_back(gs->self_is_player_a ? 1.0f : 0.0f);
    state.push_back(static_cast<float>(gs->stack_size) / 10.0f);

    // Self permanents (48 x 38 = 1824)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++)
        push_perm_slot(state, gs->self_permanents[i]);

    // Opp permanents (48 x 38 = 1824)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++)
        push_perm_slot(state, gs->opp_permanents[i]);

    // Stack (12 x 37 = 444): controller_is_self(1) + card_id(1) + is_spell(1) +
    // x_or_amount(1) + cast qualifiers(7) + chosen-mode multi-hot(6) + 4
    // announced-target sub-slots x [present, is_player, controller_is_self,
    // slot_ref, card_id](20). See machine_io.h.
    int stored_stack = std::min(gs->stack_size, MAX_STACK_DISPLAY);
    for (int i = 0; i < MAX_STACK_DISPLAY; i++) {
        if (i < stored_stack) {
            const StackEntry& se = gs->stack[i];
            state.push_back(se.controller_is_self ? 1.0f : 0.0f);
            state.push_back(norm_card_id(se.card_vocab_idx));
            state.push_back(se.is_spell ? 1.0f : 0.0f);
            state.push_back(static_cast<float>(se.x_or_amount) / 10.0f);
            state.push_back(se.is_copy ? 1.0f : 0.0f);
            state.push_back(se.kicked_any ? 1.0f : 0.0f);
            state.push_back(se.cast_with_flashback ? 1.0f : 0.0f);
            state.push_back(se.cast_with_evoke ? 1.0f : 0.0f);
            state.push_back(se.cast_with_escape ? 1.0f : 0.0f);
            state.push_back(se.cast_with_offspring ? 1.0f : 0.0f);
            state.push_back(se.cast_with_impending ? 1.0f : 0.0f);
            for (int m = 0; m < MAX_STACK_MODES; m++)
                state.push_back(se.chosen_modes[m] ? 1.0f : 0.0f);
            for (int t = 0; t < MAX_STACK_TGTS; t++) {
                const StackTarget& st = se.targets[t];
                state.push_back(st.present ? 1.0f : 0.0f);
                state.push_back(st.is_player ? 1.0f : 0.0f);
                state.push_back(st.controller_is_self ? 1.0f : 0.0f);
                state.push_back(norm_ref(st.slot_ref));
                state.push_back(norm_card_id(st.card_vocab_idx));
            }
        } else {
            state.push_back(0.0f);
            state.push_back(norm_card_id(-1));
            state.insert(state.end(), 1 + 1 + 7 + MAX_STACK_MODES, 0.0f);
            for (int t = 0; t < MAX_STACK_TGTS; t++) {
                state.insert(state.end(), STACK_TGT_FIELDS - 1, 0.0f);
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

    // Self exile (64 x 1 = 64), recency-ordered (slot 0 = most recent arrival)
    for (int i = 0; i < MAX_GY_SLOTS; i++)
        state.push_back(norm_card_id(gs->self_exile[i]));

    // Opp exile (64 x 1 = 64)
    for (int i = 0; i < MAX_GY_SLOTS; i++)
        state.push_back(norm_card_id(gs->opp_exile[i]));

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

    // Global extras (22 floats): lands played, priority, monarch, city's blessing,
    // revolt, pending extra turns, day/night, mandatory-choice one-hot, then
    // self_plays_first and the two sideboard-phase progress scalars. See the
    // [5955-5976] block in machine_io.h.
    state.push_back(static_cast<float>(gs->self.lands_played_this_turn) / 10.0f);
    state.push_back(static_cast<float>(gs->opponent.lands_played_this_turn) / 10.0f);
    state.push_back(gs->viewer_has_priority ? 1.0f : 0.0f);
    state.push_back(gs->self.is_monarch ? 1.0f : 0.0f);
    state.push_back(gs->opponent.is_monarch ? 1.0f : 0.0f);
    state.push_back(gs->self.city_blessing ? 1.0f : 0.0f);
    state.push_back(gs->opponent.city_blessing ? 1.0f : 0.0f);
    state.push_back(gs->self.revolt ? 1.0f : 0.0f);
    state.push_back(gs->opponent.revolt ? 1.0f : 0.0f);
    state.push_back(static_cast<float>(gs->self.extra_turns_pending) / 3.0f);
    state.push_back(static_cast<float>(gs->opponent.extra_turns_pending) / 3.0f);
    state.push_back(gs->is_day ? 1.0f : 0.0f);
    state.push_back(gs->is_night ? 1.0f : 0.0f);
    // MandatoryChoice one-hot, NONE at index 0 (see the enum in classes/game.h).
    // N_MANDATORY_CHOICES tracks the enum, so adding a choice kind widens this
    // one-hot and machine_io.h's offset chain shifts every later block with it.
    for (int i = 0; i < N_MANDATORY_CHOICES; i++)
        state.push_back(gs->pending_choice_kind == i ? 1.0f : 0.0f);
    state.push_back(gs->self_plays_first ? 1.0f : 0.0f);
    state.push_back(static_cast<float>(gs->sideboard_swaps_made) /
                    static_cast<float>(SIDEBOARD_SWAP_CAP));
    // Drift mapped to [0, 1] with "balanced" at the 0.5 midpoint, so the two
    // unbalanced poles sit symmetrically either side of it.
    state.push_back((static_cast<float>(gs->sideboard_delta) + 1.0f) / 2.0f);

    // ── Deck-identity tail blocks (see machine_io.h [5977-6328]) ───────────────
    // Each slot is (card_id, count): empty slot id = -1 sentinel (count 0); count
    // normalized /4.0. Slots are packed ascending by vocab id with no holes.
    auto push_decklist_block = [&](const int* ids, const int* counts, int n_slots) {
        for (int i = 0; i < n_slots; i++) {
            state.push_back(norm_card_id(ids[i]));
            state.push_back(static_cast<float>(counts[i]) / 4.0f);
        }
    };
    // Self LIVE library (48 x 2 = 96)
    push_decklist_block(gs->self_live_library_id, gs->self_live_library_ct, DECKLIST_MAIN_SLOTS);
    // Self LIVE deck configuration: maindeck (48 x 2 = 96) then sideboard (15 x 2 = 30)
    push_decklist_block(gs->self_deck_main_id, gs->self_deck_main_ct, DECKLIST_MAIN_SLOTS);
    push_decklist_block(gs->self_deck_side_id, gs->self_deck_side_ct, DECKLIST_SIDE_SLOTS);
    // Opponent REGISTERED maindeck (48 x 2 = 96)
    push_decklist_block(gs->opp_deck_main_id, gs->opp_deck_main_ct, DECKLIST_MAIN_SLOTS);
    // Opponent STATIC sideboard (15 x 2 = 30)
    push_decklist_block(gs->opp_deck_side_id, gs->opp_deck_side_ct, DECKLIST_SIDE_SLOTS);

    // Loud, NDEBUG-surviving length check: cli_output fwrites STATE_SIZE floats from this
    // buffer, so an under-fill would silently OOB-read under BUILD=RELEASE (where assert() is
    // compiled out). fatal_error exits the process rather than corrupting the BQUERY payload.
    if (static_cast<int>(state.size()) != STATE_SIZE)
        fatal_error("serialize_state: state vector size " + std::to_string(state.size()) +
                    " != STATE_SIZE " + std::to_string(STATE_SIZE));
    return state;
}
