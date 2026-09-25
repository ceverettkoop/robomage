#include "machine_io.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

#include "card_db.h"
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
#include "game_driver.h"                  // priority_window_open
#include "game_queries.h"
#include "mana_system.h"                    // mana_potential (mana-development block)
#include "parse.h"                          // name_to_uid
#include "systems/rules_modifying.h"        // rules_mod::land_drops_remaining
#include "systems/state_manager_internal.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// ── Static helpers ────────────────────────────────────────────────────────────

static int token_vocab_idx(Entity e);
static int get_card_vocab_idx(Entity e);
static int slot_ref_of(Entity e);
static void push_player_block(std::vector<float>& out, const PlayerState& ps);
static void push_mana_dev_block(std::vector<float>& out, const PlayerState& ps,
                                bool with_lands_in_hand);
static void push_log_vitals_block(std::vector<float>& out, int life, int library_ct);
static void push_per_turn_block(std::vector<float>& out, const PlayerState& ps);
static void fill_player_effects(PlayerState& ps, Zone::Ownership player);
static void push_player_effects_block(std::vector<float>& out, const PlayerState& ps);
static void push_perm_slot(std::vector<float>& out, const PermanentState& p);
static void format_counter_summary(const CounterMap& counters, char* buf, size_t buf_len);
static void add_stack_target(StackEntry& se, int& n, Entity tgt, Zone::Ownership viewer);
static void fill_stack_choices(const Ability& ab, StackEntry& se, Zone::Ownership viewer);
static void fill_permanent_state(PermanentState& ps, Entity e);
static void fill_stack_entry(StackEntry& se, Entity e, Zone::Ownership viewer);
static int battlefield_slot_ref_of(Entity e);
static int creator_slot_ref(const DelayedTriggerLink& link);
static int first_subject_battlefield_ref(const DelayedTriggerLink& link, Entity watched);
static void fill_delayed_triggers(GameState* gs, Zone::Ownership viewer,
                                  const std::vector<Entity>& stack_delayed);
static void push_delayed_slot(std::vector<float>& out, const DelayedTriggerEntry& d);
static void fill_zone_card(ZoneCardEntry& z, Entity e, Zone::Ownership viewer,
                           Zone::Ownership opp_view, bool hide_face_down);
static void push_zone_card(std::vector<float>& out, const ZoneCardEntry& z, bool with_counters);
static void fill_decklist_block(int* ids, int* counts, int n_slots,
                                const std::vector<DecklistEntry>& entries,
                                const char* block_name);
static int revealed_card_identity(int vocab_idx);
static void fill_opp_revealed_bits(GameState* gs, const unsigned char* opp_revealed);

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
// used by populate_query (BQUERY) and by the stack feature extractor (a stack entry
// is a spell with CardData or a standalone ability whose source resolves the same way).
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
    out.push_back(static_cast<float>(ps.life) / static_cast<float>(LIFE_NORMALIZER));
    out.push_back(static_cast<float>(ps.hand_ct) / 10.0f);
    out.push_back(static_cast<float>(ps.poison_counters) / 10.0f);
    for (int i = 0; i < 6; i++) out.push_back(static_cast<float>(ps.mana[i]) / 10.0f);
    out.push_back(static_cast<float>(ps.energy) / 10.0f);
}

// Pushes one half of the MANA DEVELOPMENT block: MANA_DEV_SELF_SIZE floats for the
// viewer (with_lands_in_hand), MANA_DEV_OPP_SIZE for the opponent (whose hand is hidden,
// so lands_in_hand is omitted rather than emitted as a zero — a zero would read as
// "no lands in hand", a claim about hidden information). Field order and normalizers
// are documented in machine_io.h.
static void push_mana_dev_block(std::vector<float>& out, const PlayerState& ps,
                                bool with_lands_in_hand) {
    const float count_norm = static_cast<float>(MANA_COUNT_NORMALIZER);
    for (int i = 0; i < MANA_DEV_COLORS; i++)
        out.push_back(static_cast<float>(ps.mana_potential[i]) / count_norm);
    out.push_back(static_cast<float>(ps.mana_potential_total) / count_norm);
    out.push_back(static_cast<float>(ps.lands_in_play) / count_norm);
    if (with_lands_in_hand)
        out.push_back(static_cast<float>(ps.lands_in_hand) / count_norm);
    out.push_back(static_cast<float>(ps.land_drops_remaining) /
                  static_cast<float>(LAND_DROPS_NORMALIZER));
}

// Pushes one player's half of the LOG VITALS block: LOG_VITALS_PLAYER_SIZE floats,
// log_life then log_library. Life comes from the PlayerState and the library count
// from the GameState's own library-context counter (the same integer the linear
// library float is built from), so the two encodings of a value can never disagree
// about which count they describe. Both fields are public information, hence the
// identical self and opponent halves.
static void push_log_vitals_block(std::vector<float>& out, int life, int library_ct) {
    out.push_back(norm_log_count(life, LOG_LIFE_DENOM));
    out.push_back(norm_log_count(library_ct, LOG_LIBRARY_DENOM));
}

// Pushes one player's half of the PER-TURN COUNTERS block: PER_TURN_PLAYER_SIZE floats,
// the six counts (spells, noncreature spells, instant/sorcery spells, cards drawn /
// PER_TURN_COUNT_NORMALIZER; life gained, life lost / LIFE_NORMALIZER) then the W/U/B/R/G
// spell-color multi-hot. All public, so both halves carry the same fields.
static void push_per_turn_block(std::vector<float>& out, const PlayerState& ps) {
    const float count_norm = static_cast<float>(PER_TURN_COUNT_NORMALIZER);
    const float life_norm  = static_cast<float>(LIFE_NORMALIZER);
    out.push_back(static_cast<float>(ps.spells_cast_this_turn) / count_norm);
    out.push_back(static_cast<float>(ps.noncreature_spells_cast_this_turn) / count_norm);
    out.push_back(static_cast<float>(ps.instant_sorcery_spells_cast_this_turn) / count_norm);
    out.push_back(static_cast<float>(ps.cards_drawn_this_turn) / count_norm);
    out.push_back(static_cast<float>(ps.life_gained_this_turn) / life_norm);
    out.push_back(static_cast<float>(ps.life_lost_this_turn) / life_norm);
    for (int i = 0; i < PER_TURN_COLOR_FIELDS; i++)
        out.push_back(ps.spell_colors_cast_this_turn[i] ? 1.0f : 0.0f);
}

// Copies player_effects(player) (game_queries.h) into the PlayerState's player-effects fields,
// keeping the first MAX_EMBLEM_SLOTS emblem card ids.
static void fill_player_effects(PlayerState& ps, Zone::Ownership player) {
    const PlayerEffects fx = player_effects(player);
    ps.protection_from_everything = fx.protection_from_everything;
    ps.cant_gain_life = fx.cant_gain_life;
    for (int i = 0; i < 5; i++) ps.hexproof_from[i] = fx.hexproof_from[i];
    ps.spells_cant_be_countered = fx.spells_cant_be_countered;
    ps.may_cast_sorceries_as_flash = fx.may_cast_sorceries_as_flash;
    ps.restricted_to_sorcery_speed = fx.restricted_to_sorcery_speed;
    for (int i = 0; i < MAX_EMBLEM_SLOTS; i++)
        ps.emblem_card_idx[i] = i < static_cast<int>(fx.emblem_vocab_idx.size())
                                    ? fx.emblem_vocab_idx[static_cast<size_t>(i)] : -1;
    ps.floating_trigger_source_idx = fx.floating_trigger_vocab_idx;
}

// Pushes one player's half of the PLAYER EFFECTS block: PLAYER_EFFECTS_PLAYER_SIZE floats, the
// PLAYER_EFFECTS_FLAGS flags then the emblem card ids and the floating-trigger source card id
// (per-field offsets documented in machine_io.h).
static void push_player_effects_block(std::vector<float>& out, const PlayerState& ps) {
    out.push_back(ps.protection_from_everything ? 1.0f : 0.0f);
    out.push_back(ps.cant_gain_life ? 1.0f : 0.0f);
    for (int i = 0; i < 5; i++) out.push_back(ps.hexproof_from[i] ? 1.0f : 0.0f);
    out.push_back(ps.spells_cant_be_countered ? 1.0f : 0.0f);
    out.push_back(ps.may_cast_sorceries_as_flash ? 1.0f : 0.0f);
    out.push_back(ps.restricted_to_sorcery_speed ? 1.0f : 0.0f);
    for (int i = 0; i < MAX_EMBLEM_SLOTS; i++) out.push_back(norm_card_id(ps.emblem_card_idx[i]));
    out.push_back(norm_card_id(ps.floating_trigger_source_idx));
}

// Pushes PERM_SLOT_SIZE floats (40 status + chosen-name id + returnable-exile id + card-id;
// per-slot offsets documented in machine_io.h). Empty slot (card_vocab_idx == -1) = 40 zeros
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
    const float count_norm = static_cast<float>(PER_TURN_COUNT_NORMALIZER);
    out.push_back(p.entered_this_turn ? 1.0f : 0.0f);
    out.push_back(static_cast<float>(p.ability_resolutions_this_turn) / count_norm);
    out.push_back(static_cast<float>(p.activations_this_turn) / count_norm);
    out.push_back(p.cant_be_blocked_this_turn ? 1.0f : 0.0f);
    out.push_back(p.combat_damage_prevented ? 1.0f : 0.0f);
    out.push_back(p.pending_delayed_subject ? 1.0f : 0.0f);
    for (int k = 0; k < N_OBS_KEYWORDS; k++)
        out.push_back(p.keywords[k] ? 1.0f : 0.0f);
    out.push_back(norm_card_id(p.chosen_name_idx));      // [40] chosen-name id
    out.push_back(norm_card_id(p.returnable_exile_idx)); // [41] returnable-exile id
    out.push_back(norm_card_id(p.card_vocab_idx));       // [42] card id (LAST)
}

// Pass-B fill of one battlefield permanent's PermanentState. Runs after the
// entity->slot map is built so the attachment/combat reference fields resolve.
static void fill_permanent_state(PermanentState& ps, Entity e) {
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

    ps.entered_this_turn = entered_battlefield_this_turn(static_cast<long>(perm.entered_on_turn));
    ps.ability_resolutions_this_turn = ability_resolutions_this_turn(e);
    ps.activations_this_turn = permanent_activations_this_turn(perm);
    ps.cant_be_blocked_this_turn =
        ps.is_creature && global_coordinator.GetComponent<Creature>(e).cant_be_blocked_this_turn;
    ps.combat_damage_prevented = cur_game.combat_damage_shielded(e);
    ps.pending_delayed_subject = is_waiting_delayed_trigger_subject(e);

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

// The vocab index of the physical card a revealed name belongs to: the name's
// card_db entry (DFC back faces are aliased onto the card's one entity) resolved
// through that entity's CardData name, i.e. the front face a decklist registers.
// -1 when the name has no loaded card.
static int revealed_card_identity(int vocab_idx) {
    auto it = card_db.find(name_to_uid(card_index_to_name(vocab_idx)));
    if (it == card_db.end() || !global_coordinator.entity_has_component<CardData>(it->second))
        return -1;
    return card_name_to_index(global_coordinator.GetComponent<CardData>(it->second).name);
}

// Project the opponent's match-scoped reveal set (indexed by vocab id) onto the
// filled opp registered-decklist slots. A slot is revealed when its card's own
// name was revealed or, for a double-faced card, when its back face was (a
// transformed permanent, an MDFC played back face up). A revealed name with no
// slot even then is dropped; debug builds print a stderr WARNING once per vocab
// id per process.
static void fill_opp_revealed_bits(GameState* gs, const unsigned char* opp_revealed) {
    // Slot of each registered card: main slots as i, side slots as MAIN + i.
    int slot_of[REVEALED_SIZE];
    for (int id = 0; id < REVEALED_SIZE; id++) slot_of[id] = -1;
    for (int i = 0; i < DECKLIST_MAIN_SLOTS; i++) {
        int id = gs->opp_deck_main_id[i];
        if (id >= 0 && id < REVEALED_SIZE) slot_of[id] = i;
    }
    for (int i = 0; i < DECKLIST_SIDE_SLOTS; i++) {
        int id = gs->opp_deck_side_id[i];
        if (id >= 0 && id < REVEALED_SIZE) slot_of[id] = DECKLIST_MAIN_SLOTS + i;
    }
    for (int id = 0; id < REVEALED_SIZE; id++) {
        if (!opp_revealed[id]) continue;
        int slot = slot_of[id];
        if (slot < 0) {
            int card = revealed_card_identity(id);
            if (card >= 0 && card < REVEALED_SIZE) slot = slot_of[card];
        }
        if (slot < 0) {
#ifndef NDEBUG
            static unsigned char warned[REVEALED_SIZE] = {};
            if (!warned[id]) {
                warned[id] = 1;
                fprintf(stderr,
                        "WARNING: opponent revealed card vocab id %d (%s), which is not in "
                        "their registered 75; the observation has no decklist slot for it\n",
                        id, card_index_to_name(id));
            }
#endif
            continue;
        }
        if (slot < DECKLIST_MAIN_SLOTS)
            gs->opp_deck_main_revealed[slot] = 1;
        else
            gs->opp_deck_side_revealed[slot - DECKLIST_MAIN_SLOTS] = 1;
    }
}

// Slot ref of `e` restricted to the battlefield part of the ref space (-1 otherwise).
static int battlefield_slot_ref_of(Entity e) {
    int r = slot_ref_of(e);
    return (r >= 0 && r < 2 * MAX_BATTLEFIELD_SLOTS) ? r : -1;
}

// The delayed trigger's creator slot when it is on the battlefield or the stack. The entity
// must still be the same card (its vocab idx matches the one captured at registration), so a
// creator that left play and whose entity id was reused never points at an unrelated object.
static int creator_slot_ref(const DelayedTriggerLink& link) {
    if (link.creator == 0 || action_card_vocab_idx(link.creator) != link.creator_vocab_idx) return -1;
    return slot_ref_of(link.creator);
}

// Battlefield slot of the watched object if it is there, else of the first subject still there.
static int first_subject_battlefield_ref(const DelayedTriggerLink& link, Entity watched) {
    int r = battlefield_slot_ref_of(watched);
    if (r >= 0) return r;
    for (Entity s : link.subjects) {
        r = battlefield_slot_ref_of(s);
        if (r >= 0) return r;
    }
    return -1;
}

// Pass-B fill of the delayed-trigger block: the waiting Game::delayed_triggers records plus
// the fired stack objects in `stack_delayed`, packed in ascending seq and truncated at
// MAX_DELAYED_TRIGGER_SLOTS. Needs the entity->slot map for the refs.
static void fill_delayed_triggers(GameState* gs, Zone::Ownership viewer,
                                  const std::vector<Entity>& stack_delayed) {
    Entity viewer_entity = (viewer == Zone::PLAYER_A) ? cur_game.player_a_entity
                                                      : cur_game.player_b_entity;
    std::vector<std::pair<uint32_t, DelayedTriggerEntry>> entries;
    entries.reserve(cur_game.delayed_triggers.size() + stack_delayed.size());
    for (const auto& dt : cur_game.delayed_triggers) {
        const DelayedTriggerLink& link = dt.ability.delayed_link;
        DelayedTriggerEntry d{};
        d.present            = true;
        d.controller_is_self = (dt.owner_entity == viewer_entity);
        d.on_stack           = false;
        d.stack_ref          = -1;
        d.creator_card_idx   = link.creator_vocab_idx;
        d.creator_ref        = creator_slot_ref(link);
        d.subject_ref        = first_subject_battlefield_ref(link, dt.watch_entity);
        d.subject_card_idx   = link.subject_vocab_idx;
        d.fire_kind          = link.fire_kind;
        d.fires_this_turn    = delayed_trigger_fires_this_turn(dt);
        entries.push_back({link.seq, d});
    }
    for (Entity e : stack_delayed) {
        const DelayedTriggerLink& link = global_coordinator.GetComponent<Ability>(e).delayed_link;
        DelayedTriggerEntry d{};
        d.present            = true;
        d.controller_is_self = (global_coordinator.GetComponent<Zone>(e).owner == viewer);
        d.on_stack           = true;
        d.stack_ref          = slot_ref_of(e);
        d.creator_card_idx   = link.creator_vocab_idx;
        d.creator_ref        = creator_slot_ref(link);
        d.subject_ref        = first_subject_battlefield_ref(link, 0);
        d.subject_card_idx   = link.subject_vocab_idx;
        d.fire_kind          = link.fire_kind;
        d.fires_this_turn    = false;
        entries.push_back({link.seq, d});
    }
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<uint32_t, DelayedTriggerEntry>& a,
                 const std::pair<uint32_t, DelayedTriggerEntry>& b) { return a.first < b.first; });
#ifndef NDEBUG
    if (static_cast<int>(entries.size()) > MAX_DELAYED_TRIGGER_SLOTS)
        // The observation drops the newest entries past the block width — make it observable.
        fprintf(stderr,
                "WARNING: %zu pending delayed triggers; observation truncated to "
                "MAX_DELAYED_TRIGGER_SLOTS=%d\n",
                entries.size(), MAX_DELAYED_TRIGGER_SLOTS);
#endif
    int n = std::min(static_cast<int>(entries.size()), MAX_DELAYED_TRIGGER_SLOTS);
    for (int i = 0; i < n; i++) gs->delayed[i] = entries[static_cast<size_t>(i)].second;
}

// One graveyard / exile slot. An opponent's face-down exiled card (CR 708.2, The Creation
// of Avacyn chapter I) is hidden from the viewer: the slot keeps the unknown-id sentinel
// and zeroed flags and counters, so it shows only that a card is there. The owner still
// sees its own face-down card in full. get_card_vocab_idx guards a missing CardData (a
// token resolves via its Token band / TOKEN_SENTINEL).
static void fill_zone_card(ZoneCardEntry& z, Entity e, Zone::Ownership viewer,
                           Zone::Ownership opp_view, bool hide_face_down) {
    z = ZoneCardEntry{};
    z.card_idx = -1;
    if (hide_face_down && global_coordinator.GetComponent<Zone>(e).is_face_down) return;
    z.card_idx = get_card_vocab_idx(e);
    CardPlayPermission self_perm = card_play_permission(e, viewer);
    CardPlayPermission opp_perm = card_play_permission(e, opp_view);
    z.playable_by_self = self_perm.playable();
    z.playable_by_opp = opp_perm.playable();
    // Expires only when every permission covering the card, for either player, lapses.
    z.play_expires_this_turn = (z.playable_by_self || z.playable_by_opp) &&
                               (!z.playable_by_self || self_perm.expires_this_turn) &&
                               (!z.playable_by_opp || opp_perm.expires_this_turn);
    if (global_coordinator.GetComponent<Zone>(e).location == Zone::EXILE)
        z.counters = exiled_card_counters(e);
}

// Serialize one graveyard slot (card_id, playable_by_self, playable_by_opp,
// play_expires_this_turn) or, with_counters, one exile slot (the same + counters).
static void push_zone_card(std::vector<float>& out, const ZoneCardEntry& z, bool with_counters) {
    out.push_back(norm_card_id(z.card_idx));
    out.push_back(z.playable_by_self ? 1.0f : 0.0f);
    out.push_back(z.playable_by_opp ? 1.0f : 0.0f);
    out.push_back(z.play_expires_this_turn ? 1.0f : 0.0f);
    if (with_counters)
        out.push_back(static_cast<float>(z.counters) / static_cast<float>(ZONE_COUNTER_NORMALIZER));
}

// Pushes DELAYED_SLOT_SIZE floats (per-slot offsets documented in machine_io.h). Empty slot =
// zeros with the two card-id sentinels.
static void push_delayed_slot(std::vector<float>& out, const DelayedTriggerEntry& d) {
    if (!d.present) {
        out.insert(out.end(), 4, 0.0f);
        out.push_back(norm_card_id(-1));  // creator_card_id sentinel
        out.insert(out.end(), 2, 0.0f);
        out.push_back(norm_card_id(-1));  // subject_card_id sentinel
        out.insert(out.end(), DELAYED_FIRE_KINDS + 1, 0.0f);
        return;
    }
    out.push_back(1.0f);
    out.push_back(d.controller_is_self ? 1.0f : 0.0f);
    out.push_back(d.on_stack ? 1.0f : 0.0f);
    out.push_back(d.on_stack ? norm_ref(d.stack_ref) : 0.0f);
    out.push_back(norm_card_id(d.creator_card_idx));
    out.push_back(norm_ref(d.creator_ref));
    out.push_back(norm_ref(d.subject_ref));
    out.push_back(norm_card_id(d.subject_card_idx));
    for (int k = 0; k < DELAYED_FIRE_KINDS; k++)
        out.push_back(d.fire_kind == k ? 1.0f : 0.0f);
    out.push_back(d.fires_this_turn ? 1.0f : 0.0f);
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
        gs->self_graveyard[i].card_idx = -1;
        gs->opp_graveyard[i].card_idx  = -1;
        gs->self_exile[i].card_idx     = -1;
        gs->opp_exile[i].card_idx      = -1;
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
        gs->opp_deck_main_revealed[i] = 0;
    }
    for (int i = 0; i < DECKLIST_SIDE_SLOTS; i++) {
        gs->self_deck_side_id[i] = -1;
        gs->self_deck_side_ct[i] = 0;
        gs->opp_deck_side_id[i]  = -1;
        gs->opp_deck_side_ct[i]  = 0;
        gs->opp_deck_side_revealed[i] = 0;
    }

    Zone::Ownership priority_owner = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    if (viewer == Zone::UNKNOWN) viewer = priority_owner;

    Zone::Ownership active_owner = cur_game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity viewer_entity = (viewer == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    Entity opp_entity    = (viewer == Zone::PLAYER_A) ? cur_game.player_b_entity : cur_game.player_a_entity;

    gs->cur_step            = cur_game.cur_step;
    gs->turn                = static_cast<int>(cur_game.turn);
    gs->is_active_player    = (viewer == active_owner);
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
        ps.spells_cast_this_turn = static_cast<int>(p.spells_cast_this_turn);
        ps.noncreature_spells_cast_this_turn = static_cast<int>(p.noncreature_spells_cast_this_turn);
        ps.instant_sorcery_spells_cast_this_turn =
            static_cast<int>(p.instant_sorcery_spells_cast_this_turn);
        ps.cards_drawn_this_turn = static_cast<int>(p.cards_drawn_this_turn.size());
        ps.life_gained_this_turn = p.life_gained_this_turn;
        ps.life_lost_this_turn   = p.life_lost_this_turn;
        const Colors spell_colors[5] = {WHITE, BLUE, BLACK, RED, GREEN};
        for (int i = 0; i < 5; i++)
            ps.spell_colors_cast_this_turn[i] = p.spell_colors_cast_this_turn.count(spell_colors[i]) > 0;
    };
    fill_player_stats(gs->self, viewer_entity);
    fill_player_stats(gs->opponent, opp_entity);
    fill_player_effects(gs->self, viewer);
    fill_player_effects(gs->opponent, viewer == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A);

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

    // Priority-window context: the pass flags are meaningful only in an ordinary
    // priority window (UNTAP/CLEANUP set both as a step-advance device, and a mid-flow
    // prompt leaves whatever the interrupted round had), so outside one all three stay 0.
    if (priority_window_open()) {
        gs->is_priority_window = true;
        gs->self_has_passed = viewer_is_player_a ? cur_game.a_has_passed : cur_game.b_has_passed;
        gs->opp_has_passed  = viewer_is_player_a ? cur_game.b_has_passed : cur_game.a_has_passed;
    }

    // Mulligan state. Game::pregame is the game this observation's board belongs to,
    // which during the sideboard phase is the game that just ended, so the phase
    // leaves the fields at 0.
    if (!sideboard_phase) {
        const Game::PregameState &pg = cur_game.pregame;
        gs->self_mulligans_taken = viewer_is_player_a ? pg.mulls_a : pg.mulls_b;
        gs->opp_mulligans_taken  = viewer_is_player_a ? pg.mulls_b : pg.mulls_a;
        if (pg.stage == Game::PregameState::MULL_BOTTOM && pg.bottoming_owner == viewer)
            gs->self_bottom_remaining = pg.bottom_remaining;
    }

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

    // Graveyard/exile cards as (distance_from_top, entity), sorted for recency order.
    struct GyItem { size_t dist; Entity ent; };
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
    // Stack objects that are fired delayed triggers (any depth, not only the displayed 12).
    std::vector<Entity> stack_delayed;

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
                    // Land drops available FROM HAND (mana-development block). Counted in the
                    // same pass, through the shared card_playable_as_land predicate the
                    // PLAY_LAND enumeration uses (so a modal DFC with a land back face counts).
                    // Viewer-only: the opponent's hand is hidden information.
                    if (global_coordinator.entity_has_component<CardData>(e) &&
                        card_playable_as_land(global_coordinator.GetComponent<CardData>(e)))
                        gs->self.lands_in_hand++;
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
                (is_self ? self_gy_items : opp_gy_items).push_back({zone.distance_from_top, e});
                break;

            case Zone::EXILE:
                // Collected per-owner in recency order, exactly like the graveyard (the
                // face-down masking is applied in fill_zone_card).
                (is_self ? self_exile_items : opp_exile_items).push_back({zone.distance_from_top, e});
                break;

            case Zone::STACK:
                gs->stack_size++;
                if (stack_item_count < MAX_STACK_DISPLAY + 8)
                    stack_items[stack_item_count++] = {zone.distance_from_top, e};
                // A fired delayed trigger's stack object (delayed-trigger block).
                if (global_coordinator.entity_has_component<Ability>(e) &&
                    global_coordinator.GetComponent<Ability>(e).delayed_link.seq != 0)
                    stack_delayed.push_back(e);
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
        fill_permanent_state(gs->self_permanents[i], self_ents[i]);
    for (int i = 0; i < opp_bf; i++)
        fill_permanent_state(gs->opp_permanents[i], opp_ents[i]);
    for (int i = 0; i < stored_stack; i++)
        fill_stack_entry(gs->stack[i], stack_items[i].ent, viewer);
    fill_delayed_triggers(gs, viewer, stack_delayed);

    // ── Mana development ──────────────────────────────────────────────────────
    // Reuses the battlefield entities pass A already collected (both sides in one
    // set, since mana_potential re-guards by controller) instead of re-scanning the
    // ECS. mana_potential applies the live-permanent guard itself, so the phased-out
    // permanents deliberately kept in the serialized slots are excluded here.
    {
        std::set<Entity> bf_entities(self_ents, self_ents + self_bf);
        bf_entities.insert(opp_ents, opp_ents + opp_bf);
        auto fill_mana_dev = [&](PlayerState& ps, Zone::Ownership owner, Entity ent) {
            ManaPotential mp = mana_potential(owner, bf_entities);
            for (int i = 0; i < MANA_DEV_COLORS; i++) ps.mana_potential[i] = mp.by_color[i];
            // Untapped sources PLUS whatever is already floating in the pool.
            size_t pool = global_coordinator.entity_has_component<Player>(ent)
                              ? global_coordinator.GetComponent<Player>(ent).mana.size() : 0;
            ps.mana_potential_total  = mp.sources + static_cast<int>(pool);
            ps.lands_in_play         = mp.lands;
            ps.land_drops_remaining  = rules_mod::land_drops_remaining(owner);
        };
        Zone::Ownership opp_view = (viewer == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        fill_mana_dev(gs->self, viewer, viewer_entity);
        fill_mana_dev(gs->opponent, opp_view, opp_entity);
    }

    // Graveyards and exile in RECENCY order: slot 0 = most recent arrival (lowest
    // distance_from_top).
    {
        Zone::Ownership opp_view = (viewer == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        auto fill_zone = [&](ZoneCardEntry* slots, std::vector<GyItem>& items, bool hide_face_down) {
            std::sort(items.begin(), items.end(),
                      [](const GyItem& a, const GyItem& b) { return a.dist < b.dist; });
            int n = std::min(static_cast<int>(items.size()), MAX_GY_SLOTS);
            for (int i = 0; i < n; i++)
                fill_zone_card(slots[i], items[static_cast<size_t>(i)].ent, viewer, opp_view,
                               hide_face_down);
        };
        fill_zone(gs->self_graveyard, self_gy_items, false);
        fill_zone(gs->opp_graveyard, opp_gy_items, false);
        fill_zone(gs->self_exile, self_exile_items, false);
        fill_zone(gs->opp_exile, opp_exile_items, true);
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
    // Opponent-of-viewer's match-scoped reveal set, projected onto those slots.
    fill_opp_revealed_bits(gs, (viewer == Zone::PLAYER_A) ? g_revealed_by_b : g_revealed_by_a);
}

// ── populate_query ────────────────────────────────────────────────────────────

void populate_query(Query* q, const std::vector<LegalAction>& actions) {
    memset(q, 0, sizeof(*q));
    int n = std::min(static_cast<int>(actions.size()), MAX_ACTIONS);
#ifndef NDEBUG
    if (static_cast<int>(actions.size()) > MAX_ACTIONS)
        // Machine-mode agents can never pick a truncated action, so an
        // over-wide menu silently restricts the policy — make it observable.
        fprintf(stderr,
                "WARNING: legal-action menu has %zu entries; machine query "
                "truncated to MAX_ACTIONS=%d — choices beyond that are "
                "unreachable for machine-mode agents\n",
                actions.size(), MAX_ACTIONS);
#endif
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

    // Self permanents (48 x 43 = 2064)
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++)
        push_perm_slot(state, gs->self_permanents[i]);

    // Opp permanents (48 x 43 = 2064)
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

    // Graveyards (64 x GY_SLOT_SIZE per side), then exile (64 x EXILE_SLOT_SIZE per
    // side), self first, recency-ordered (slot 0 = most recent arrival).
    for (int i = 0; i < MAX_GY_SLOTS; i++) push_zone_card(state, gs->self_graveyard[i], false);
    for (int i = 0; i < MAX_GY_SLOTS; i++) push_zone_card(state, gs->opp_graveyard[i], false);
    for (int i = 0; i < MAX_GY_SLOTS; i++) push_zone_card(state, gs->self_exile[i], true);
    for (int i = 0; i < MAX_GY_SLOTS; i++) push_zone_card(state, gs->opp_exile[i], true);

    // Self hand (10 x 1 = 10)
    for (int i = 0; i < MAX_HAND_SLOTS; i++)
        state.push_back(norm_card_id(gs->self_hand[i]));

    // Match context (4 floats, all 0.0 in single-game mode)
    state.push_back(gs->match_game_number >= 0 ? static_cast<float>(gs->match_game_number) / 3.0f : 0.0f);
    state.push_back(static_cast<float>(gs->match_wins_self) / 2.0f);
    state.push_back(static_cast<float>(gs->match_wins_opp) / 2.0f);
    state.push_back(gs->is_sideboard_phase ? 1.0f : 0.0f);

    // Library counts (2 floats)
    state.push_back(static_cast<float>(gs->self_library_ct) / static_cast<float>(LIBRARY_NORMALIZER));
    state.push_back(static_cast<float>(gs->opp_library_ct) / static_cast<float>(LIBRARY_NORMALIZER));

    // Current turn (1 float)
    state.push_back(static_cast<float>(gs->turn) / TURN_NORMALIZER);

    // Known top-of-library cards for the viewer (5 slots x 1 float = 5)
    // Sentinel id = unknown.
    for (int i = 0; i < KNOWN_TOP_LIBRARY_SIZE; i++)
        state.push_back(norm_card_id(gs->known_top_library_self[i]));

    // Known opponent-hand cards (10 x 1 = 10): specific card identities the viewer
    // has had revealed from the opponent's hand and that are still in hand. Sentinel
    // id = empty/unknown slot. Distinct from the opp decklist revealed bits: this
    // tracks the exact card and clears when that card leaves the hand.
    for (int i = 0; i < MAX_HAND_SLOTS; i++)
        state.push_back(norm_card_id(gs->opp_known_hand[i]));

    // Pending decision context (2 floats): card id of the spell/ability making the
    // current mid-resolution choice (sentinel = none) + its controller-is-viewer flag.
    state.push_back(norm_card_id(gs->pending_decision_card));
    state.push_back(gs->pending_decision_ctrl_is_self ? 1.0f : 0.0f);

    // Global extras (27 floats): lands played, monarch, city's blessing, revolt,
    // pending extra turns, day/night, the priority-window context, the mulligan
    // state, the mandatory-choice one-hot, then self_plays_first and the two
    // sideboard-phase progress scalars. See the [4898-4924] block in machine_io.h.
    state.push_back(static_cast<float>(gs->self.lands_played_this_turn) / 10.0f);
    state.push_back(static_cast<float>(gs->opponent.lands_played_this_turn) / 10.0f);
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
    state.push_back(gs->self_has_passed ? 1.0f : 0.0f);
    state.push_back(gs->opp_has_passed ? 1.0f : 0.0f);
    state.push_back(gs->is_priority_window ? 1.0f : 0.0f);
    const float mull_norm = static_cast<float>(MULLIGAN_NORMALIZER);
    state.push_back(static_cast<float>(gs->self_mulligans_taken) / mull_norm);
    state.push_back(static_cast<float>(gs->opp_mulligans_taken) / mull_norm);
    state.push_back(static_cast<float>(gs->self_bottom_remaining) / mull_norm);
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

    // ── Deck-identity tail blocks (see machine_io.h [4925-5340]) ───────────────
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
    // Opponent slots carry a third float: the match-scoped revealed bit.
    auto push_opp_decklist_block = [&](const int* ids, const int* counts,
                                       const unsigned char* revealed, int n_slots) {
        for (int i = 0; i < n_slots; i++) {
            state.push_back(norm_card_id(ids[i]));
            state.push_back(static_cast<float>(counts[i]) / 4.0f);
            state.push_back(revealed[i] ? 1.0f : 0.0f);
        }
    };
    // Opponent REGISTERED maindeck (48 x 3 = 144)
    push_opp_decklist_block(gs->opp_deck_main_id, gs->opp_deck_main_ct,
                            gs->opp_deck_main_revealed, DECKLIST_MAIN_SLOTS);
    // Opponent REGISTERED sideboard (16 x 3 = 48)
    push_opp_decklist_block(gs->opp_deck_side_id, gs->opp_deck_side_ct,
                            gs->opp_deck_side_revealed, DECKLIST_SIDE_SLOTS);

    // ── Mana development (see machine_io.h [5341-5359]) ───────────────────────
    // Self (10 floats) then opponent (9 — no lands_in_hand, which is hidden).
    push_mana_dev_block(state, gs->self, /*with_lands_in_hand=*/true);
    push_mana_dev_block(state, gs->opponent, /*with_lands_in_hand=*/false);

    // ── Log-scaled vitals (see machine_io.h [5360-5363]) ──────────────────────
    // The same life/library counts already emitted linearly above (player blocks,
    // library-context block), re-warped through log1p so the near-zero region —
    // where the game is decided and the linear floats have their least resolution —
    // gets proportional resolution. Self (2 floats) then opponent (2). Both
    // encodings are kept deliberately; see the rationale in machine_io.h.
    push_log_vitals_block(state, gs->self.life, gs->self_library_ct);
    push_log_vitals_block(state, gs->opponent.life, gs->opp_library_ct);

    // ── Per-turn counters (see machine_io.h [5364-5385]) ──────────────────────
    // Self (11 floats) then opponent (11): the per-turn counts and the spell-color
    // multi-hot.
    push_per_turn_block(state, gs->self);
    push_per_turn_block(state, gs->opponent);

    // ── Pending delayed triggers (see machine_io.h [5386-5593]) ───────────────
    for (int i = 0; i < DELAYED_SLOTS; i++)
        push_delayed_slot(state, gs->delayed[i]);

    // ── Player effects (see machine_io.h [6490-6515]) ─────────────────────────
    // Self (13 floats) then opponent (13).
    push_player_effects_block(state, gs->self);
    push_player_effects_block(state, gs->opponent);

    // Loud, NDEBUG-surviving length check: cli_output fwrites STATE_SIZE floats from this
    // buffer, so an under-fill would silently OOB-read under BUILD=RELEASE (where assert() is
    // compiled out). fatal_error exits the process rather than corrupting the BQUERY payload.
    if (static_cast<int>(state.size()) != STATE_SIZE)
        fatal_error("serialize_state: state vector size " + std::to_string(state.size()) +
                    " != STATE_SIZE " + std::to_string(STATE_SIZE));
    return state;
}
