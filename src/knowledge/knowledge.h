#ifndef KNOWLEDGE_KNOWLEDGE_H
#define KNOWLEDGE_KNOWLEDGE_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../state_format/decode.h"
#include "../state_format/schema.h"
#include "frame_view.h"
#include "k_rows.h"
#include "player_model.h"
#include "solver.h"

// ── The knowledge sidecar: one seat's recall and deduction (docs/knowledge_store.md) ──────────
//
// A pure function of one viewer's frame stream: the SCHEMA, CARDS and DECKLISTS frames, then every
// STATE frame (its events in order, then its snapshot). It never sees engine state; it may include
// only the standard library and state_format/decode.h / schema.h (the rmstate gate tier checks).
//
// The model is a function of the events alone, whichever frames they arrive in (a consumer fed at
// every decision and one fed only at its own decisions agree at every frame both see). Per event:
//   1. the event updates the model by its rule (knowledge_store §5; one on_* function per kind);
//   2. a network that changed is compacted (compact.h), which restarts its bound stores (deferred
//      while an order-hidden batch waits for its library_place, which is in the same frame);
//   3. bound propagation brings each player's per-card bounds to their fixpoint (bounds.h), and the
//      copies seen are updated.
// Per STATE frame (apply_frame), after its events:
//   4. the snapshot is checked against the model: zone sizes, and every identity the snapshot shows
//      (the viewer's hand, reveal windows, public and face-down objects). A disagreement is a
//      missing event or a model bug, reported as an `ERROR: knowledge …` line (take_errors). The
//      snapshot adds nothing to the model, apart from the own_deck rows a game_start in the frame
//      roots the viewer's own deck in (constant for the whole game);
//   5. the exact solver (solver.h) computes every reported pool's per-card least and most copies
//      under a per-player step budget.
// rows() then gives the k_* tables for the current state.
//
// The model per player (PlayerModel) is a network of nodes (sets of cards the viewer cannot tell
// apart) rooted in the player's list: game 1 the registered main deck and sideboard, games 2-3 the
// registered 75 for the opponent (split between deck and sideboard in a way the viewer does not
// know) and the viewer's current own_deck for themselves; under --closed-decklist the opponent's
// root is free apart from the copies seen earlier in the match. Networks reset at game_start;
// match memory (copies seen per match) persists.
//
// Deterministic: ordered containers only, so the same frames give the same bytes. Errors are
// returned, never thrown (-fno-exceptions).

namespace knowledge {

class Knowledge {
    public:
        explicit Knowledge(rmstate::KnowledgeLevel level = rmstate::KnowledgeLevel::EXPERT);

        // One record payload of the viewer's stream: SCHEMA, CARDS, DECKLISTS or STATE. A frame
        // for another viewer, or one that does not decode, is an error.
        void feed(const std::string &payload);

        // The decoded entry points feed() uses; unit tests drive the model through them directly.
        void begin_match(int viewer, const Codes &codes, const Decklists &lists);
        void apply_frame(const Snapshot &snap);

        // The k_* tables for the current state (k_meta only at level off).
        KRows rows() const;
        // Appends rows() to a frame being built.
        void write_tables(rmstate::FrameWriter &w) const;
        // rows() as the table blocks of this viewer's STATE frame (k_meta … k_known_hand, laid out
        // as rmstate::split_frame_blocks gives them; `n` = how many): what the engine appends to
        // the frame it fed (rmstate::append_frame_blocks) and rmknow --isolation recomputes.
        // False before the SCHEMA frame.
        bool table_blocks(std::string *blocks, uint32_t *n) const;
        // A human-readable summary of the current knowledge (rmknow --text).
        std::string text() const;
        // ERROR: lines produced since the last call.
        std::vector<std::string> take_errors();

        // The whole state (match constants included; card names, which only text() reads, are
        // not), so a restored instance continues exactly as this one would.
        void serialize(std::string &out) const;
        bool deserialize(const std::string &in);

        int viewer() const { return viewer_; }
        bool started() const { return viewer_ != NO_SEAT; }
        uint64_t schema_hash() const { return schema_.hash; }
        const rmstate::Schema &schema() const { return schema_; }
        const Codes &codes() const { return codes_; }
        // Card names from the CARDS frame (for text()); empty before it.
        const std::map<int, std::string> &names() const { return names_; }
        void set_names(const std::map<int, std::string> &names) { names_ = names; }
        // The model of `player` (0 = A, 1 = B), read-only, for tools and tests.
        const PlayerModel &player(int p) const { return players_[p]; }
        bool game_active() const { return game_active_; }
        // The last STATE frame's game and decision.
        uint8_t frame_game() const { return frame_game_; }
        uint32_t frame_decision() const { return frame_decision_; }
        // Steps the last frame's exact solver used (k_meta.steps): a function of the model, so of
        // the events; 0 for a frame with no live game.
        uint64_t steps() const { return solve_steps_; }
        // For tools and tests: the node sets of the pools the tables may report for player `p`
        // (the solver's groups, in its order), and the solver's answers for them as of the last
        // frame.
        std::vector<std::vector<NodeId>> pool_groups(int p) const;
        const Solution &solution(int p) const { return solved_[p]; }
        // For tests: compaction on (the default) or off. Not part of the serialised state.
        void set_compaction(bool on) { compaction_ = on; }

    private:
        // ── frame flow ──
        void apply_events(const std::vector<Event> &events);
        void apply_event(const Event &ev);
        void settle();
        void compact();
        void propagate();
        void solve();
        void update_seen();
        void check(const Snapshot &snap);
        void error(const std::string &what);

        // ── update rules (knowledge_store §5), one per event kind ──
        void on_game_start(const Event &ev);
        void on_game_end(const Event &ev);
        void on_draw(const Event &ev);
        void on_move(const Event &ev);
        void on_library_place(const Event &ev);
        void on_reveal(const Event &ev);
        void on_reveal_end(const Event &ev);
        void on_shuffle(const Event &ev);
        void on_reorder(const Event &ev);
        void on_face_turn(const Event &ev);

        // ── shared steps ──
        void build_roots(int p, uint16_t library, uint16_t sideboard);
        std::vector<NodeId> add_root_parts(
            int p, RootKind kind, const std::vector<std::pair<int, int>> &counts, const std::vector<uint16_t> &sizes);
        NodeId take_unit(int p, Zn zone, const EventCard &c, const Event &ev);
        void put_unit(int p, Zn zone, NodeId n, const EventCard &c, const Event &ev);
        NodeId take_library_at(int p, const LibPos &from);
        NodeId take_from_set(int p, std::vector<NodeId> &zone, int card, bool drawn_this_turn, uint16_t turn);
        size_t merge_candidates(int p, std::vector<NodeId> &zone, const std::vector<size_t> &at, uint16_t turn);
        void assign_revealed(
            int p, std::vector<NodeId> &zone, const std::vector<const EventCard *> &cards, uint16_t turn);
        NodeId fresh_unit(int p, Loc loc);
        bool drew_this_turn(int p, uint16_t turn) const;
        bool can_hold(int p, NodeId n, int card) const;
        void pin_unit(int p, NodeId n, int card);
        bool remove_live(int p, NodeId n);
        NodeId handle_node(uint32_t h, int *player) const;
        void bind_handle(int p, NodeId n, uint32_t h);
        void drop_handle(uint32_t h);
        bool library_index(int p, const LibPos &pos, size_t size, size_t *out);
        bool library_range(int p, const LibPos &pos, size_t *a, size_t *b);

        // ── output ──
        // A pool the tables may report: the opponent's hand, one library run or a block of runs
        // (from run `first` to run `last`), the opponent's sideboard, a face-down card, the
        // set-aside units.
        struct PoolSpec {
                rmstate::PoolKind kind = rmstate::PoolKind::HAND;
                std::vector<NodeId> nodes;
                LibPos range;
                uint32_t obj = 0;
                int first = -1, last = -1;  // a library block's runs; first == last for one run
        };
        std::vector<PoolSpec> pool_specs(int p) const;
        GroupBounds pool_bounds(int p, size_t spec, const std::vector<PoolSpec> &specs) const;
        void pool_rows(int p, KRows &out, const PoolSpec &spec, const GroupBounds &b) const;
        void player_rows(int p, KRows &out) const;

        // match constants
        rmstate::KnowledgeLevel level_;
        int viewer_ = NO_SEAT;
        bool have_schema_ = false;
        std::string schema_payload_;
        rmstate::Schema schema_;
        Codes codes_;
        std::map<int, std::string> names_;
        Decklists lists_;
        // match memory
        std::map<int, uint8_t> seen_match_[2];
        // game state
        uint32_t last_seq_ = 0;
        bool game_active_ = false;
        uint8_t game_ = 0;
        bool closed_[2] = {false, false};
        std::vector<DeckEntry> own_deck_;  // the viewer's own_deck rows of the frame that started the game
        PlayerModel players_[2];
        // A draw a later reveal may name (a draw link): a player's first non-opening draw of a turn,
        // kept for that turn.
        struct DrawRef {
                int player = NO_SEAT;
                NodeId node = NO_NODE;
                uint16_t turn = 0;
        };
        std::map<uint32_t, std::pair<int, NodeId>> handles_;  // the viewer's handle → (player, node)
        std::map<uint32_t, DrawRef> draws_;                   // a draw event's seq → its unit
        std::map<int, uint8_t> seen_game_[2];
        bool reported_contradiction_[2][2] = {{false, false}, {false, false}};
        uint64_t steps_ = 0;
        uint16_t turn_ = 0;  // the game's latest turn (of any event since game_start)
        // The solver's answers for pool_specs(p), and the steps they took.
        Solution solved_[2];
        uint64_t solve_steps_ = 0;
        Solver solver_;
        bool compaction_ = true;
        // frame context
        uint8_t frame_game_ = 0;
        uint32_t frame_decision_ = 0;
        uint32_t frame_seq_ = 0;
        const std::vector<DeckEntry> *frame_own_deck_ = nullptr;  // the frame's own_deck rows, while it applies
        std::map<uint16_t, LibPos> batch_place_;  // this frame's batches: library_place range
        std::map<uint16_t, std::pair<int, std::vector<NodeId>>> batch_nodes_;  // batch → (player, units placed)
        std::vector<std::string> errors_;
};

}  // namespace knowledge

#endif /* KNOWLEDGE_KNOWLEDGE_H */
