#ifndef ROBOMAGE_CHECK_FRAMES_H
#define ROBOMAGE_CHECK_FRAMES_H

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "../../src/state_format/decode.h"
#include "../../src/state_format/schema.h"
#include "proc.h"

// ── robomage_check: RMSTATE records, decoded with the std-only decoder ────────────────────────
//
// The engine's frames reach the gate as record files (an --rmstate-dump, a saved stream) or as a
// live --machine-format rmstate session, which run_stream answers. Frames are decoded with
// src/state_format/decode.h (standard library only), each viewer's with its own SCHEMA frame.

namespace rmcheck {

// Frame kinds and viewers as the frame header numbers them (state_format/schema.h).
constexpr int KIND_SCHEMA = 0, KIND_CARDS = 1, KIND_DECKLISTS = 2, KIND_STATE = 3, KIND_METAGAME = 4;
constexpr int VIEWER_A = 0, VIEWER_B = 1, VIEWER_OMNISCIENT = 2;

// The frame kind (or viewer) byte of a payload; -1 when it is too short to have one.
int frame_kind(const std::string &payload);
int frame_viewer(const std::string &payload);
const char *kind_name(int kind);
// "A", "B", "OMNISCIENT".
const char *viewer_name(int viewer);

bool read_records(const std::string &path, std::vector<std::string> *payloads, std::string *err);
bool write_records(const std::string &path, const std::vector<std::string> &payloads);

// Decodes a stream's frames, each viewer's against the schema its own SCHEMA frame gave.
class FrameDecoder {
    public:
        // Decodes `payload` (a SCHEMA frame also becomes its viewer's schema); false with `err`.
        bool feed(const std::string &payload, rmstate::DecodedFrame *out, std::string *err);
        // The schema `viewer`'s frames decode with (the first one seen when it sent none), or nullptr.
        const rmstate::Schema *schema(int viewer) const;

    private:
        rmstate::Schema schemas_[3];
        bool have_[3] = {false, false, false};
};

// One table of a decoded frame with its columns by name. Cells follow the JSON rendering's rules
// (state_format/json.h): a card, index, row: or nullable enum column at all ones is null.
class Rows {
    public:
        Rows(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *table);
        size_t size() const { return table_ ? table_->rows.size() : 0; }
        // The column's index, or -1.
        int col(const char *name) const;
        bool null(size_t row, int col) const;
        uint64_t u(size_t row, int col) const;
        int64_t i(size_t row, int col) const;
        const std::string &text(size_t row, int col) const;
        // An enum column's value name (the number when its table has none; "" when null).
        std::string name(size_t row, int col) const;
        // The column's raw cells of a row, for equality.
        const std::vector<rmstate::Value> &row(size_t r) const { return table_->rows[r]; }

    private:
        const rmstate::Schema &schema_;
        const rmstate::DecodedTable *table_ = nullptr;
        const rmstate::TableDef *def_ = nullptr;
};

// The table definition named `name`, or nullptr.
const rmstate::TableDef *table_named(const rmstate::Schema &schema, const char *name);
// A STATE frame's header row values ("decision", "game", …; 0 when absent).
uint64_t header_value(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *col);

// One decision of a live --machine-format rmstate session.
struct StreamDecision {
        int viewer = 0;
        uint64_t decision = 0, game = 0;
        bool sideboarding = false;
        std::vector<std::string> descriptions;  // the menu, row by row
        const rmstate::DecodedFrame *frame = nullptr;
        const rmstate::Schema *schema = nullptr;
};

struct StreamRun {
        std::vector<std::string> payloads;  // every record, in order
        std::vector<int> replies;           // every reply written
        int terminals = 0;                  // terminal STATE frames
        std::string error;                  // a protocol error ("" when none)
        ProcResult proc;                    // exit status and stderr
};

// Plays one --machine-format rmstate session (`spec` is its engine command): every non-terminal
// STATE frame is answered with reply(decision). `run.error` is set on a framing or decode error or
// a decision frame without a menu.
StreamRun run_stream(const ProcSpec &spec, const std::function<int(const StreamDecision &)> &reply);

// Random replies from Python's random.Random(seed) (py_random.h): the game is conceded (-2) past
// decision `cap`; else -1 (the confirm row) with probability 0.05, else a uniform menu row.
std::function<int(const StreamDecision &)> random_replies(uint64_t seed, uint64_t cap);

// Every stream frame against the dump written alongside it
// (`dump`): the match frames equal, each STATE frame equal to the dump's for its (viewer, game,
// sideboarding, decision, terminal) apart from the event tables, each seat's events equal in
// order, no frame number twice, the match frames before a seat's first STATE frame, the terminal
// frames at the end. Failure messages, labelled `label`.
std::vector<std::string> check_stream(
    const std::vector<std::string> &stream, const std::vector<std::string> &dump, const std::string &label);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_FRAMES_H */
