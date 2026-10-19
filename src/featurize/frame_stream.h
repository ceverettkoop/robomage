#ifndef FEATURIZE_FRAME_STREAM_H
#define FEATURIZE_FRAME_STREAM_H

#include <functional>
#include <string>

#include "../state_format/decode.h"

// ── A recorded RMSTATE stream, frame by frame (docs/state_format.md §9), standard library and the decoder only
//
// An --rmstate-dump file (every viewer's frames) or a saved --machine stream (one seat's): each
// record is decoded with its viewer's own SCHEMA frame, the viewer's latest CARDS, DECKLISTS and
// METAGAME frames are kept, and every STATE frame is handed over as it is read, so a whole match is
// never held in memory.

namespace featurize {

// One viewer's once-per-match frames, decoded (the frames' tables point into `schema`).
struct ViewerFrames {
        bool have_schema = false;
        rmstate::Schema schema;
        rmstate::DecodedFrame cards, decklists, metagame;
};

// Calls on_state(viewer's frames, frame) for every STATE frame of the stream at `path` (viewer 0 A,
// 1 B, 2 omniscient, as the frame header numbers them). False with `err` on an unreadable file, a
// framing or decode error, a frame before its viewer's SCHEMA frame, or when on_state returns false
// (its `err`).
bool read_stream(const std::string &path,
    const std::function<bool(const ViewerFrames &, const rmstate::DecodedFrame &, std::string *)> &on_state,
    std::string *err);

// Reads a file of one viewer's once-per-match frames and no STATE frame (robomage --cards-frame's
// SCHEMA, CARDS and METAGAME) into `out`. False with `err` on an unreadable file, a decode error, a
// STATE frame, or frames of more than one viewer or none.
bool read_match_frames(const std::string &path, ViewerFrames *out, std::string *err);

}  // namespace featurize

#endif /* FEATURIZE_FRAME_STREAM_H */
