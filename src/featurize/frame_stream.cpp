#include "frame_stream.h"

#include <cstdio>
#include <utility>

static bool feed_record(const std::string &payload, featurize::ViewerFrames viewers[3], rmstate::DecodedFrame *state,
    const rmstate::DecodedFrame **decoded, bool *is_state, std::string *err);
static bool read_stream_records(FILE *f, const std::string &label, featurize::ViewerFrames viewers[3],
    rmstate::DecodedFrame *state,
    const std::function<bool(const rmstate::DecodedFrame &, bool, std::string *)> &on_frame, std::string *err);
static bool read_match_file(FILE *f, const std::string &label, featurize::ViewerFrames *out, std::string *err);

// Decodes one record into its viewer's frames, or into `state` (is_state) for a STATE frame;
// `decoded` is the frame it decoded into (null for a SCHEMA frame).
static bool feed_record(const std::string &payload, featurize::ViewerFrames viewers[3], rmstate::DecodedFrame *state,
    const rmstate::DecodedFrame **decoded, bool *is_state, std::string *err) {
    *is_state = false;
    *decoded = nullptr;
    if (payload.size() < rmstate::FRAME_HEADER_SIZE) {
        *err = "a record shorter than a frame header";
        return false;
    }
    const uint8_t kind = static_cast<uint8_t>(payload[6]), viewer = static_cast<uint8_t>(payload[7]);
    if (viewer > 2) {
        *err = "a frame of viewer " + std::to_string(viewer);
        return false;
    }
    featurize::ViewerFrames &v = viewers[viewer];
    rmstate::DecodeResult res;
    if (kind == static_cast<uint8_t>(rmstate::FrameKind::SCHEMA)) {
        v = featurize::ViewerFrames();
        res = rmstate::decode_schema_frame(payload, &v.schema);
        v.have_schema = res.ok();
    } else if (!v.have_schema) {
        *err = "a frame before its viewer's SCHEMA frame";
        return false;
    } else if (kind == static_cast<uint8_t>(rmstate::FrameKind::STATE)) {
        res = rmstate::decode_frame(payload, v.schema, state);
        *is_state = true;
        *decoded = state;
    } else {
        rmstate::DecodedFrame *dst = kind == static_cast<uint8_t>(rmstate::FrameKind::CARDS)       ? &v.cards
                                     : kind == static_cast<uint8_t>(rmstate::FrameKind::DECKLISTS) ? &v.decklists
                                                                                                   : nullptr;
        if (!dst) {
            *err = "a frame of kind " + std::to_string(kind);
            return false;
        }
        res = rmstate::decode_frame(payload, v.schema, dst);
        *decoded = dst;
    }
    if (!res.ok()) *err = "a frame does not decode: " + res.message;
    return res.ok();
}

// Every record of `f` (closed here; null: `label` cannot be read), decoded into `viewers` /
// `state`; on_frame(frame, is_state) for each frame but a SCHEMA frame. Errors name `label`.
static bool read_stream_records(FILE *f, const std::string &label, featurize::ViewerFrames viewers[3],
    rmstate::DecodedFrame *state,
    const std::function<bool(const rmstate::DecodedFrame &, bool, std::string *)> &on_frame, std::string *err) {
    if (!f) {
        *err = "cannot read " + label;
        return false;
    }
    rmstate::RecordReader reader(f);
    std::string payload;
    rmstate::DecodeResult res;
    bool ok = true;
    while (ok && reader.next(&payload, &res)) {
        bool is_state = false;
        const rmstate::DecodedFrame *decoded = nullptr;
        ok = feed_record(payload, viewers, state, &decoded, &is_state, err) &&
             (!decoded || on_frame(*decoded, is_state, err));
        if (!ok) *err = label + ": record " + std::to_string(reader.count()) + ": " + *err;
    }
    std::fclose(f);
    if (ok && !res.ok()) {
        *err = label + ": " + res.message;
        ok = false;
    }
    return ok;
}

// Reads one viewer's once-per-match frames from `f` (read_stream_records' `f` and `label`).
static bool read_match_file(FILE *f, const std::string &label, featurize::ViewerFrames *out, std::string *err) {
    featurize::ViewerFrames viewers[3];
    rmstate::DecodedFrame state;
    int seen = -1;
    const bool ok = read_stream_records(
        f, label, viewers, &state,
        [&](const rmstate::DecodedFrame &frame, bool is_state, std::string *e) {
            if (is_state) {
                *e = "a STATE frame in a file of match frames";
                return false;
            }
            if (seen >= 0 && seen != frame.header.viewer) {
                *e = "match frames of two viewers";
                return false;
            }
            seen = frame.header.viewer;
            return true;
        },
        err);
    if (!ok) return false;
    if (seen < 0 || !viewers[seen].have_schema) {
        *err = label + ": no frame after a SCHEMA frame";
        return false;
    }
    // A move keeps the schema's table buffer, which the decoded frames' tables point into.
    *out = std::move(viewers[seen]);
    return true;
}

namespace featurize {

bool read_stream(const std::string &path,
    const std::function<bool(const ViewerFrames &, const rmstate::DecodedFrame &, std::string *)> &on_state,
    std::string *err) {
    ViewerFrames viewers[3];
    rmstate::DecodedFrame state;
    return read_stream_records(
        std::fopen(path.c_str(), "rb"), path, viewers, &state,
        [&](const rmstate::DecodedFrame &frame, bool is_state, std::string *e) {
            return !is_state || on_state(viewers[frame.header.viewer], frame, e);
        },
        err);
}

bool read_match_frames(const std::string &path, ViewerFrames *out, std::string *err) {
    return read_match_file(std::fopen(path.c_str(), "rb"), path, out, err);
}

bool read_match_records(const std::string &records, ViewerFrames *out, std::string *err) {
    if (records.empty()) {
        *err = "the match frames: no record";
        return false;
    }
    // A read-only stream over the bytes (fmemopen does not write to a "rb" buffer).
    return read_match_file(
        fmemopen(const_cast<char *>(records.data()), records.size(), "rb"), "the match frames", out, err);
}

}  // namespace featurize
