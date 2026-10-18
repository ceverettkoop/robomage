#include "knowledge_feed.h"

#include <cstdint>
#include <cstdio>
#include <memory>

#include "../cli_output.h"
#include "../error.h"
#include "../game_driver.h"
#include "../knowledge/knowledge.h"
#include "../snapshot.h"
#include "decode.h"
#include "schema.h"

namespace {

constexpr int N_CONSUMERS = 4;  // ObsConsumer's values
// Error lines printed per instance before the rest are only counted.
constexpr size_t MAX_ERROR_LINES = 40;

// One writer's instance for one seat.
struct Channel {
        std::unique_ptr<knowledge::Knowledge> know;
        size_t errors = 0;  // error lines it produced
};

}  // namespace

static Channel channels[N_CONSUMERS][2];
static bool hooks_registered = false;

static int seat_index(Zone::Ownership viewer);
static const char *consumer_word(ObsConsumer consumer);
static Channel &channel(ObsConsumer consumer, int seat);
static void pass_errors(ObsConsumer consumer, int seat, std::vector<std::string> *errors);
static void put_u32(std::string &out, uint32_t v);
static uint32_t get_u32(const std::string &in, size_t at);
static void save_state(std::string &blob);
static void restore_state(const std::string &blob);

static int seat_index(Zone::Ownership viewer) {
    return viewer == Zone::PLAYER_A ? 0 : viewer == Zone::PLAYER_B ? 1 : -1;
}

static const char *consumer_word(ObsConsumer consumer) {
    switch (consumer) {
        case ObsConsumer::DUMP:
            return "dump";
        case ObsConsumer::STREAM:
            return "stream";
        case ObsConsumer::SELFCHECK:
            return "selfcheck";
        case ObsConsumer::AGENT:
            return "agent";
    }
    return "?";
}

rmstate::KnowledgeLevel knowledge_level_of(Zone::Ownership viewer) {
    const int level = viewer == Zone::PLAYER_A ? knowledge_level_a : viewer == Zone::PLAYER_B ? knowledge_level_b : 0;
    return static_cast<rmstate::KnowledgeLevel>(level);
}

// The writer's instance for the seat, made at its first frame (registering the snapshot hooks
// with the first instance of the process).
static Channel &channel(ObsConsumer consumer, int seat) {
    Channel &ch = channels[static_cast<int>(consumer)][seat];
    if (!ch.know)
        ch.know.reset(new knowledge::Knowledge(knowledge_level_of(seat == 0 ? Zone::PLAYER_A : Zone::PLAYER_B)));
    if (!hooks_registered) {
        snapshot_set_sidecar_hooks(save_state, restore_state);
        hooks_registered = true;
    }
    return ch;
}

static void pass_errors(ObsConsumer consumer, int seat, std::vector<std::string> *errors) {
    Channel &ch = channels[static_cast<int>(consumer)][seat];
    for (std::string &line : ch.know->take_errors()) {
        ++ch.errors;
        line += std::string(" (") + consumer_word(consumer) + " frames)";
        if (errors) {
            errors->push_back(line);
            continue;
        }
        if (ch.errors > MAX_ERROR_LINES) continue;
        // Under --machine-format rmstate stdout is stderr's descriptor with a buffer of its own.
        std::fflush(stdout);
        std::fprintf(stderr, "%s\n", line.c_str());
        if (ch.errors == MAX_ERROR_LINES)
            cli_error(
                std::string("knowledge (") + consumer_word(consumer) + " frames): further errors are not printed");
    }
}

void knowledge_feed_match_frame(ObsConsumer consumer, Zone::Ownership viewer, const std::string &payload) {
    const int seat = seat_index(viewer);
    if (seat < 0) return;
    channel(consumer, seat).know->feed(payload);
    pass_errors(consumer, seat, nullptr);
}

void knowledge_append_tables(
    ObsConsumer consumer, Zone::Ownership viewer, std::string *payload, std::vector<std::string> *errors) {
    const int seat = seat_index(viewer);
    if (seat < 0) return;
    knowledge::Knowledge &know = *channel(consumer, seat).know;
    know.feed(*payload);
    pass_errors(consumer, seat, errors);
    std::string blocks;
    uint32_t n = 0;
    if (!know.table_blocks(&blocks, &n))
        fatal_error("knowledge: no k_* tables for a STATE frame (no SCHEMA frame fed)");
    const rmstate::DecodeResult res = rmstate::append_frame_blocks(payload, blocks, n);
    if (!res.ok()) fatal_error("knowledge: the k_* tables do not append to the STATE frame: " + res.message);
}

const knowledge::Knowledge *knowledge_instance(ObsConsumer consumer, Zone::Ownership viewer) {
    const int seat = seat_index(viewer);
    if (seat < 0) return nullptr;
    return channels[static_cast<int>(consumer)][seat].know.get();
}

std::string knowledge_state_blob() {
    std::string blob;
    save_state(blob);
    return blob;
}

// ── Snapshot hooks ────────────────────────────────────────────────────────────────────────────

static void put_u32(std::string &out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

static uint32_t get_u32(const std::string &in, size_t at) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(static_cast<unsigned char>(in[at + i])) << (8 * i);
    return v;
}

// Every instance, in writer and seat order: a presence byte, then its serialised state with its
// length.
static void save_state(std::string &blob) {
    blob.clear();
    std::string state;
    for (Channel(&row)[2] : channels) {
        for (Channel &ch : row) {
            blob.push_back(ch.know ? 1 : 0);
            if (!ch.know) continue;
            state.clear();
            ch.know->serialize(state);
            put_u32(blob, static_cast<uint32_t>(state.size()));
            blob += state;
        }
    }
}

// Puts back what save_state wrote. An instance the blob does not hold (an empty blob: the slot was
// saved before the first instance was made) is dropped, as it was then.
static void restore_state(const std::string &blob) {
    size_t at = 0;
    std::string state;
    for (int c = 0; c < N_CONSUMERS; ++c) {
        for (int seat = 0; seat < 2; ++seat) {
            Channel &ch = channels[c][seat];
            const bool present = at < blob.size() && blob[at] != 0;
            if (at < blob.size()) ++at;
            if (!present) {
                ch.know.reset();
                continue;
            }
            if (blob.size() - at < 4) fatal_error("knowledge: a snapshot's sidecar state is truncated");
            const uint32_t len = get_u32(blob, at);
            at += 4;
            if (blob.size() - at < len) fatal_error("knowledge: a snapshot's sidecar state is truncated");
            state.assign(blob, at, len);
            at += len;
            if (!ch.know)
                ch.know.reset(
                    new knowledge::Knowledge(knowledge_level_of(seat == 0 ? Zone::PLAYER_A : Zone::PLAYER_B)));
            if (!ch.know->deserialize(state)) fatal_error("knowledge: a snapshot's sidecar state does not restore");
        }
    }
}
