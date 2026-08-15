#include "obs_builder.h"

#include <array>
#include <cctype>
#include <cmath>
#include <cstring>

#include <string>

#include "classes/game.h"  // MandatoryChoice enum (mandatory-choice one-hot width)
#include "components/zone.h"
#include "error.h"
// Not consumed here — included so the torch-free `make actor-syntax` tier
// compiles the duplicate-edge merge predicate (its layout reads must track
// this file's block constants).
#include "menu_merge.h"
#include "game_driver.h"  // sideboard_phase / sideboard_phase_player + deck names
#include "gen/archetypes_gen.h"  // ARCH_DECK_TAGS, ARCH_UNKNOWN, ARCH_N
#include "gen/card_costs_gen.h"  // CARD_COST_MATRIX, CARD_ABILITY_COST_MATRIX, N_COST_FEATS

// ── State-vector offsets ────────────────────────────────────────────────────
// The actor no longer keeps its own copy of the offset chain. Every absolute
// block offset it needs (SELF_PERM_START, HAND_START, MATCH_CTX_START,
// REVEALED_START, EXTRAS_*, the deck-identity tail, ...) comes from the
// OFFSET_CHAIN in src/machine_io.h, which is derived from that header's block
// widths and pinned by its own `== STATE_SIZE` static_assert. train/env.py
// derives the same chain from the same widths (mirrored by train/gen_enums.py),
// and ci_check.py's `actorobs` tier compiles a generated TU that static_asserts
// the two chains equal boundary-by-boundary.
//
// Only offsets WITHIN a slot — which machine_io.h does not name — are defined
// here, derived from the slot widths it does export.
//
// The three id-family floats sit LAST in a permanent slot (chosen-name id, then
// returnable-exile id, then card id) — mirror env.py's _PERM_*_OFF.
static constexpr int PERM_CHOSEN_NAME_OFF = PERM_SLOT_SIZE - 3;
static constexpr int PERM_RETURNABLE_OFF = PERM_SLOT_SIZE - 2;
static constexpr int PERM_CARD_OFF = PERM_SLOT_SIZE - 1;  // card id is LAST in a perm slot
// Stack-slot layout (mirror env.py): [1] is the object card id; the announced
// target sub-slots start after ctrl+id+is_spell + x_or_amount + quals + modes.
static constexpr int STACK_TGT_START = STACK_HEAD_FIELDS + STACK_XAMT_FIELDS +
                                       STACK_QUAL_FIELDS + STACK_MODE_SLOTS;  // == 17
static constexpr int STACK_TGT_ID_OFF = STACK_TGT_FIELDS - 1;     // card id is LAST per sub-slot
static_assert(STACK_TGT_START == 17, "stack target sub-slots start at slot offset 17");

// Block ENDs the sideboard obs mask below needs; machine_io.h names each block's
// start, and one block's end is the next one's start.
static constexpr int HIST_END = MATCH_CTX_START;
static constexpr int KNOWN_TOP_LIB_END = REVEALED_START;
static constexpr int REVEALED_END = OPP_KNOWN_HAND_START;
static constexpr int OPP_KNOWN_HAND_END = PENDING_DECISION_START;
static constexpr int PENDING_DECISION_END = EXTRAS_START;
static constexpr int SELF_LIVE_LIB_END = SELF_DECK_MAIN_START;
static constexpr int SELF_DECK_MAIN_END = SELF_DECK_SIDE_START;
static constexpr int SELF_DECK_SIDE_END = OPP_DECK_MAIN_START;
static constexpr int OPP_DECK_MAIN_END = OPP_DECK_SIDE_START;

// Per-action block starts + ACTOR_REF_ZONE_MAX now live in obs_builder.h
// (shared with menu_merge.h).
static constexpr int HAND_COST_START = ACT_ORDS_START + MAX_ACTIONS;
static constexpr int BF_COST_START = HAND_COST_START + MAX_HAND_SLOTS * ACTOR_N_COST_FEATS;
// Matchup tail: raw bucket float, then one-hot(self arch), then one-hot(opp arch).
static constexpr int MATCHUP_TAIL_START = BF_COST_START + MAX_BATTLEFIELD_SLOTS * ACTOR_N_COST_FEATS;
static constexpr int ARCH_ONEHOT_START = MATCHUP_TAIL_START + 1;
static_assert(ARCH_ONEHOT_START + 2 * ARCH_N == ACTOR_OBS_SIZE,
              "matchup tail must end exactly at ACTOR_OBS_SIZE");

static_assert(N_COST_FEATS == ACTOR_N_COST_FEATS, "cost matrix width mismatch");

// ── Deck -> archetype index ─────────────────────────────────────────────────
// Mirror of train/archetypes.py::normalize_deck: a deck spec becomes its
// canonical decks/-relative stem (lowercase, '/' separators, no ".dk" suffix, no
// leading "./" or "/", nothing above bin/resources/decks/). The generated
// ARCH_DECK_TAGS keys are already in that form.
static std::string normalize_deck(const std::string& deck) {
    std::string stem = deck;
    for (char& c : stem) {
        if (c == '\\') c = '/';
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    static const std::string prefix = "resources/decks/";
    size_t idx = stem.rfind(prefix);
    if (idx != std::string::npos) stem = stem.substr(idx + prefix.size());
    while (stem.compare(0, 2, "./") == 0) stem = stem.substr(2);
    while (!stem.empty() && stem.front() == '/') stem = stem.substr(1);
    if (stem.size() > 3 && stem.compare(stem.size() - 3, 3, ".dk") == 0)
        stem = stem.substr(0, stem.size() - 3);
    while (!stem.empty() && stem.back() == '/') stem.pop_back();
    return stem;
}

// Archetype index for a deck spec: ARCH_DECK_TAGS lookup, else ARCH_UNKNOWN —
// exactly archetypes.arch_index().
static int arch_index_for_deck(const std::string& deck) {
    if (deck.empty()) return ARCH_UNKNOWN;
    std::string stem = normalize_deck(deck);
    for (int i = 0; i < ARCH_N_DECK_TAGS; i++)
        if (stem == ARCH_DECK_TAGS[i].stem) return ARCH_DECK_TAGS[i].arch;
    return ARCH_UNKNOWN;
}

// Write env.py's matchup tail for the seat the observation is written for.
// `self_is_a` comes from the state vector's own self_is_a flag, so the tail is
// perspective-relative like every other block (and stays correct during a bo3
// sideboard phase, where the viewer is sideboard_phase_player).
static void write_matchup_tail(float* o, bool self_is_a) {
    // The deck names are set once at startup; cache their archetype indices but
    // re-resolve if a caller ever swaps decks mid-run.
    static std::string cached_a, cached_b;
    static int arch_a = ARCH_UNKNOWN, arch_b = ARCH_UNKNOWN;
    if (cached_a != deck_a_name || cached_b != deck_b_name) {
        cached_a = deck_a_name;
        cached_b = deck_b_name;
        arch_a = arch_index_for_deck(cached_a);
        arch_b = arch_index_for_deck(cached_b);
    }
    int self_arch = self_is_a ? arch_a : arch_b;
    int opp_arch = self_is_a ? arch_b : arch_a;
    o[MATCHUP_TAIL_START] = static_cast<float>(self_arch * ARCH_N + opp_arch);
    for (int i = 0; i < 2 * ARCH_N; i++) o[ARCH_ONEHOT_START + i] = 0.0f;
    o[ARCH_ONEHOT_START + self_arch] = 1.0f;
    o[ARCH_ONEHOT_START + ARCH_N + opp_arch] = 1.0f;
}

// ── Sideboard observation mask ──────────────────────────────────────────────
// Bit-exact C++ twin of train/env.py::_build_sideboard_mask. During the bo3
// between-games sideboard phase the engine keeps the ended game's ECS alive, so
// the raw state vector describes the STALE terminal board — noise for a
// sideboarding decision. env.py zeroes every state block except the ones that
// inform sideboarding (graveyards+exile, action history, match/library/turn ctx,
// the opponent revealed multi-hot, the pending-decision context, and BOTH opponent
// static-decklist blocks), plus the self-is-A seat flag; card-id positions inside
// masked blocks are filled with the empty sentinel (-1/N_CARD_TYPES), NOT 0.0 (0.0
// decodes to real vocab index 0). We build the identical keep/fill pair once.
// The MANA DEVELOPMENT block is board state of the ENDED game (untapped sources, lands
// in play, land drops left in a turn that is over), so — like the player blocks and the
// global extras — it is masked: it is simply not kept below, and holds no card-id slot,
// so the default 0.0 fill is correct. env.py's mask does the same by omission.
// The LOG VITALS block is masked the same way and for the same reason: it re-states the
// player blocks' life and the library-context counts, and the PLAYER blocks are masked
// here, so keeping the log copy would leak the ended game's life totals back in. (The
// LINEAR library counts inside the kept match/library-context range do survive the mask;
// the log copy does not, so during the sideboard phase the two encodings are deliberately
// NOT redundant — obsinv asserts the zeroed block there instead of the log identity.)
struct SideboardMask {
    std::array<bool, STATE_SIZE> keep{};   // value-initialized: all false
    std::array<float, STATE_SIZE> fill{};  // value-initialized: all 0.0f
};

static const SideboardMask& sideboard_mask() {
    static const SideboardMask mask = [] {
        SideboardMask m;
        auto keep_range = [&](int lo, int hi) {
            for (int i = lo; i < hi; i++) m.keep[static_cast<size_t>(i)] = true;
        };
        keep_range(GY_START, HAND_START);                   // graveyards + exile (self + opp)
        keep_range(HIST_START, HIST_END);                   // action history ring
        keep_range(MATCH_CTX_START, KNOWN_TOP_LIB_START);   // match + library ctx + current turn
        keep_range(REVEALED_START, REVEALED_END);           // opponent revealed multi-hot
        keep_range(PENDING_DECISION_START, PENDING_DECISION_END);  // pending-decision context
        keep_range(OPP_DECK_MAIN_START, OPP_DECK_SIDE_END); // opponent registered decklist (both blocks)
        keep_range(SELF_DECK_MAIN_START, SELF_DECK_SIDE_END);  // the viewer's own live 75
        keep_range(EXTRAS_SB_CTX_START, EXTRAS_END);        // plays-first + sideboard progress
        // The "self is Player A" seat flag must survive (it is live during the
        // sideboard phase, set from sideboard_phase_player, and every obs consumer
        // routes seats by it).
        m.keep[static_cast<size_t>(ACTOR_SELF_IS_A_IDX)] = true;

        const float null_id = -1.0f / static_cast<float>(N_CARD_TYPES);
        auto card_id_slot = [&](int i) {
            if (!m.keep[static_cast<size_t>(i)]) m.fill[static_cast<size_t>(i)] = null_id;
        };
        // All three id-family floats per permanent slot (self + opp).
        for (int s = 0; s < MAX_BATTLEFIELD_SLOTS; s++) {
            for (int off : {PERM_CHOSEN_NAME_OFF, PERM_RETURNABLE_OFF, PERM_CARD_OFF}) {
                card_id_slot(SELF_PERM_START + s * PERM_SLOT_SIZE + off);
                card_id_slot(OPP_PERM_START + s * PERM_SLOT_SIZE + off);
            }
        }
        // Stack: object card id + each announced-target sub-slot's card id.
        for (int s = 0; s < MAX_STACK_DISPLAY; s++) {
            int base = STACK_START + s * STACK_SLOT_SIZE;
            card_id_slot(base + 1);
            int tgt0 = base + STACK_TGT_START;
            for (int t = 0; t < STACK_TGT_SLOTS; t++)
                card_id_slot(tgt0 + t * STACK_TGT_FIELDS + STACK_TGT_ID_OFF);
        }
        for (int i = HAND_START; i < HAND_START + MAX_HAND_SLOTS; i++) card_id_slot(i);       // self hand
        for (int i = KNOWN_TOP_LIB_START; i < KNOWN_TOP_LIB_END; i++) card_id_slot(i);        // known top-5
        for (int i = OPP_KNOWN_HAND_START; i < OPP_KNOWN_HAND_END; i++) card_id_slot(i);      // known opp hand
        // Every decklist block's card-id position (card_id_slot skips the kept ones,
        // so this stays "all card-id positions" rather than "the masked ones").
        for (int s = 0; s < DECKLIST_MAIN_SLOTS; s++) {
            card_id_slot(SELF_LIVE_LIB_START + s * DECKLIST_SLOT_SIZE);   // self live library
            card_id_slot(SELF_DECK_MAIN_START + s * DECKLIST_SLOT_SIZE);  // self maindeck
        }
        for (int s = 0; s < DECKLIST_SIDE_SLOTS; s++)                     // self sideboard
            card_id_slot(SELF_DECK_SIDE_START + s * DECKLIST_SLOT_SIZE);
        return m;
    }();
    return mask;
}

// Apply the sideboard mask in place over the state region [0, STATE_SIZE): mirror
// env.py's `np.copyto(state_arr, _SB_MASK_FILL, where=~_SB_MASK_KEEP)`.
static void apply_sideboard_mask(float* o) {
    const SideboardMask& m = sideboard_mask();
    for (int i = 0; i < STATE_SIZE; i++)
        if (!m.keep[static_cast<size_t>(i)]) o[i] = m.fill[static_cast<size_t>(i)];
}

// Decode a normalized card-id float back to a vocab index. env.py uses np.rint
// (round-half-even); ids are exact multiples of 1/N_CARD_TYPES so llrintf (which
// also rounds half-to-even under the default mode) reproduces it exactly.
static int decode_card_id(float f) {
    return static_cast<int>(std::llrintf(f * static_cast<float>(N_CARD_TYPES)));
}

// Write one cost row (N_COST_FEATS floats) from `matrix` for vocab `id`; a
// negative id (empty slot) writes a zero row — exactly _gather_costs in env.py.
static void write_cost_row(float* dst, const float matrix[][N_COST_FEATS], int id) {
    if (id < 0) {
        for (int k = 0; k < N_COST_FEATS; k++) dst[k] = 0.0f;
        return;
    }
    int safe = id;
    if (safe < 0) safe = 0;
    if (safe > N_CARD_TYPES - 1) safe = N_CARD_TYPES - 1;
    for (int k = 0; k < N_COST_FEATS; k++) dst[k] = matrix[safe][k];
}

ActorObs build_obs(const std::vector<LegalAction>& actions) {
    // Mirror the machine-branch order in input_logger.cpp: populate the game
    // state (which also rebuilds the entity->slot map that populate_query and
    // serialize_state both consult), then the query, then serialize the state.
    // bo1 never sideboards, so the viewer is always the priority player.
    Zone::Ownership viewer = sideboard_phase ? sideboard_phase_player : Zone::UNKNOWN;
    static thread_local GameState local_gs;
    populate_gamestate(&local_gs, viewer);
    Query q;
    populate_query(&q, actions);
    const std::vector<float>& state = serialize_state(&local_gs);

    ActorObs result;
    result.num_choices = q.num_choices;
    result.obs.assign(static_cast<size_t>(ACTOR_OBS_SIZE), 0.0f);
    float* o = result.obs.data();

    // [0, STATE_SIZE): state vector verbatim.
    std::memcpy(o, state.data(), sizeof(float) * static_cast<size_t>(STATE_SIZE));

    // During the bo3 between-games sideboard phase the state vector describes the
    // stale terminal board; mask it down to the sideboard-relevant blocks exactly
    // like env.py (applied to [0, STATE_SIZE) BEFORE the cost gathers below so the
    // derived hand/bf cost rows zero out from the sentinel-filled ids, matching the
    // env). is_sideboard_phase is MATCH_CTX_START + 3.
    if (o[MATCH_CTX_START + 3] > 0.5f) apply_sideboard_mask(o);

    // Per-action metadata. Reproduce EXACTLY the padded arrays cli_output.cpp
    // writes into the BQUERY payload, then apply the same normalization env.py
    // applies when it folds those arrays into the obs:
    //   cats : int32 (pad 0)  -> value / ACTION_CATEGORY_MAX   (float64 divide)
    //   ids  : float32 (pad -1/N)  -> verbatim
    //   ctrl : float32 (pad -1/N)  -> verbatim
    //   zone : int32 (pad 0)  -> value / REF_ZONE_MAX          (float64 divide)
    //   refs : int32 (pad -1) -> (value + 1) / N_ENTITY_REF_SLOTS (float64 divide)
    //   ords : int32 (pad -1) -> (value + 1) / (OPTION_ORDINAL_MAX + 1) (float64 divide)
    const float id_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    const float ctrl_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    for (int i = 0; i < MAX_ACTIONS; i++) {
        int cat = 0;
        float idf = id_null;
        float ctrlf = ctrl_null;
        int zoneval = 0;
        int refval = -1;
        int ordval = -1;
        if (i < q.num_choices) {
            const ActionChoice& ac = q.choices[i];
            cat = ac.category;
            idf = ac.card_vocab_idx >= 0
                      ? static_cast<float>(ac.card_vocab_idx) / static_cast<float>(N_CARD_TYPES)
                      : id_null;
            ctrlf = (ac.zone_ref != REF_NONE)
                        ? (ac.controller_is_self ? 1.0f : 0.0f)
                        : ctrl_null;
            zoneval = static_cast<int>(ac.zone_ref);
            refval = ac.slot_ref;
            ordval = ac.option_ordinal;
        }
        // int-array normalizers go through float64 (numpy int/int -> float64 ->
        // cast to float32) so the last-ULP rounding matches env.py exactly.
        o[ACT_CATS_START + i] =
            static_cast<float>(static_cast<double>(cat) / static_cast<double>(ACTION_CATEGORY_MAX));
        o[ACT_IDS_START + i] = idf;
        o[ACT_CTRL_START + i] = ctrlf;
        o[ACT_ZONE_START + i] =
            static_cast<float>(static_cast<double>(zoneval) / static_cast<double>(ACTOR_REF_ZONE_MAX));
        o[ACT_REFS_START + i] =
            static_cast<float>(static_cast<double>(refval + 1) / static_cast<double>(N_ENTITY_REF_SLOTS));
        o[ACT_ORDS_START + i] =
            static_cast<float>(static_cast<double>(ordval + 1) / static_cast<double>(OPTION_ORDINAL_MAX + 1));
    }

    // Hand cast-cost rows: gather by each self-hand slot's card id.
    for (int i = 0; i < MAX_HAND_SLOTS; i++) {
        int id = decode_card_id(o[HAND_START + i]);
        write_cost_row(o + HAND_COST_START + i * N_COST_FEATS, CARD_COST_MATRIX, id);
    }

    // Battlefield activated-ability cost rows: gather by each of the 48 self
    // permanent slots' card ids (the card id is LAST within each perm slot).
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        int id = decode_card_id(o[SELF_PERM_START + i * PERM_SLOT_SIZE + PERM_CARD_OFF]);
        write_cost_row(o + BF_COST_START + i * N_COST_FEATS, CARD_ABILITY_COST_MATRIX, id);
    }

    // Matchup tail (value bucket + archetype one-hots), from the state's own
    // self_is_a flag — read AFTER the sideboard mask, which deliberately keeps
    // that flag live.
    write_matchup_tail(o, o[ACTOR_SELF_IS_A_IDX] > 0.5f);

    return result;
}
