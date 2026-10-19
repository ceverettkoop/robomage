#include "selfcheck_featurize.h"

#include <cstdio>
#include <utility>

#include "../card_vocab.h"
#include "../featurize/featurizer.h"
#include "decode.h"
#include "schema.h"
#include "selfcheck.h"

namespace {

// One seat's featurizer inputs, from its own match frames.
struct SeatInputs {
        bool ready = false;
        rmstate::Schema schema;
        featurize::CardTables tables;
        featurize::FeatureSpec spec;
};

struct Counts {
        size_t decisions = 0, compared = 0, refused = 0, errors = 0;
};

}  // namespace

bool selfcheck_featurize = false;

static SeatInputs seats[2];
static Counts counts;

static int seat_index(Zone::Ownership viewer);
static void fail(const char *check, Zone::Ownership viewer, const std::string &detail);
static bool seat_tensors(const SeatInputs &in, const std::string &payload, featurize::Tensors *out, std::string *err);
static std::string first_difference(
    const featurize::FeatureSpec &spec, const featurize::Tensors &a, const featurize::Tensors &b);

static int seat_index(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A ? 0 : viewer == Zone::PLAYER_B ? 1 : -1;
}

static void fail(const char *check, Zone::Ownership viewer, const std::string &detail) {
    ++counts.errors;
    selfcheck_state_mismatch("featurize", viewer, std::string(check) + ": " + detail);
}

// A seat frame's tensors from the actor entry point.
static bool seat_tensors(const SeatInputs &in, const std::string &payload, featurize::Tensors *out, std::string *err) {
    rmstate::DecodedFrame frame;
    const rmstate::DecodeResult res = rmstate::decode_frame(payload, in.schema, &frame);
    if (!res.ok()) {
        *err = "the frame does not decode: " + res.message;
        return false;
    }
    return featurize::featurize_seat(in.spec, in.tables, frame, out, err);
}

// The first tensor whose bytes differ.
static std::string first_difference(
    const featurize::FeatureSpec &spec, const featurize::Tensors &a, const featurize::Tensors &b) {
    const featurize::RmtbFile fa = featurize::tensors_file(spec, {a}), fb = featurize::tensors_file(spec, {b});
    for (size_t i = 0; i < fa.arrays.size(); ++i)
        if (fa.arrays[i].data != fb.arrays[i].data || fa.arrays[i].shape != fb.arrays[i].shape)
            return fa.arrays[i].name + " differs";
    return "";
}

void selfcheck_featurize_match_frames(Zone::Ownership viewer, const std::vector<std::string> &payloads) {
    const int s = seat_index(viewer);
    if (!selfcheck_featurize || s < 0 || payloads.empty()) return;
    SeatInputs &in = seats[s];
    in = SeatInputs();
    rmstate::DecodedFrame cards, metagame;
    rmstate::DecodeResult res = rmstate::decode_schema_frame(payloads[0], &in.schema);
    for (size_t i = 1; i < payloads.size() && res.ok(); ++i) {
        rmstate::DecodedFrame f;
        res = rmstate::decode_frame(payloads[i], in.schema, &f);
        if (f.header.kind == static_cast<uint8_t>(rmstate::FrameKind::CARDS))
            cards = std::move(f);
        else if (f.header.kind == static_cast<uint8_t>(rmstate::FrameKind::METAGAME))
            metagame = std::move(f);
    }
    std::string err;
    if (!res.ok()) {
        fail("inputs", viewer, "a match frame does not decode: " + res.message);
        return;
    }
    if (!featurize::build_card_tables(in.schema, cards, static_cast<uint64_t>(N_CARD_TYPES) + 1, 0, &in.tables, &err) ||
        !featurize::build_spec(in.schema, metagame, in.tables, &in.spec, &err)) {
        fail("inputs", viewer, err);
        return;
    }
    in.ready = true;
}

void selfcheck_featurize_compare(
    const char *check, Zone::Ownership viewer, const std::string &frame, const std::string &other) {
    const int s = seat_index(viewer);
    if (!selfcheck_featurize || s < 0 || !seats[s].ready) return;
    const SeatInputs &in = seats[s];
    std::string head, tail, with_tail = other, err;
    uint32_t n_tail = 0;
    rmstate::DecodeResult res =
        rmstate::split_frame_blocks(frame, static_cast<uint16_t>(rmstate::TableId::K_META), &head, &tail, &n_tail);
    if (res.ok()) res = rmstate::append_frame_blocks(&with_tail, tail, n_tail);
    if (!res.ok()) {
        fail(check, viewer, "the k_* tables do not carry over: " + res.message);
        return;
    }
    featurize::Tensors a, b;
    if (!seat_tensors(in, frame, &a, &err) || !seat_tensors(in, with_tail, &b, &err)) {
        fail(check, viewer, err);
        return;
    }
    ++counts.compared;
    const std::string diff = first_difference(in.spec, a, b);
    if (!diff.empty()) fail(check, viewer, "the tensors differ: " + diff);
}

void selfcheck_featurize_omniscient(const std::string &frame) {
    if (!selfcheck_featurize || !seats[0].ready) return;
    ++counts.decisions;
    rmstate::DecodedFrame f;
    const rmstate::DecodeResult res = rmstate::decode_frame(frame, seats[0].schema, &f);
    featurize::Tensors t;
    std::string err;
    if (!res.ok()) {
        fail("isolation", Zone::UNKNOWN, "the omniscient frame does not decode: " + res.message);
        return;
    }
    if (featurize::featurize_seat(seats[0].spec, seats[0].tables, f, &t, &err)) {
        fail("isolation", Zone::UNKNOWN, "the actor entry point featurized the omniscient frame");
        return;
    }
    if (!featurize::featurize_viewer(seats[0].spec, seats[0].tables, f, &t, &err)) {
        fail("isolation", Zone::UNKNOWN, "the learner entry point refused the omniscient frame: " + err);
        return;
    }
    ++counts.refused;
}

void selfcheck_featurize_match_over() {
    if (!selfcheck_featurize) return;
    std::fflush(stdout);
    std::fprintf(stderr, "SELFCHECK-FEATURIZE: decisions=%zu compared=%zu refused=%zu errors=%zu\n", counts.decisions,
        counts.compared, counts.refused, counts.errors);
    std::fflush(stderr);
    counts = Counts();
    seats[0] = SeatInputs();
    seats[1] = SeatInputs();
}
