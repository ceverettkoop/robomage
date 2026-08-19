#include "oracle_client.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "error.h"
#include "obs_builder.h"  // ACTOR_OBS_SIZE

namespace {
constexpr int kHello = 1;
constexpr int kNewGame = 2;
constexpr int kQuery = 3;

// Minimal JSON string field. Deck names are .dk stems (letters, digits, '/',
// '_', '-') and the spec is a fixed literal, so only the quote/backslash
// escapes are needed to stay well-formed on any future input.
std::string json_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}
}  // namespace

ScriptedOracle::~ScriptedOracle() {
    if (fd_ >= 0) close(fd_);
}

void ScriptedOracle::send_all(const void* buf, long n) {
    const char* p = static_cast<const char*>(buf);
    while (n > 0) {
        long w = write(fd_, p, static_cast<size_t>(n));
        if (w <= 0) fatal_error("scripted oracle: socket write failed");
        p += w;
        n -= w;
    }
}

void ScriptedOracle::recv_all(void* buf, long n) {
    char* p = static_cast<char*>(buf);
    while (n > 0) {
        long r = read(fd_, p, static_cast<size_t>(n));
        if (r <= 0) fatal_error("scripted oracle: socket read failed/EOF");
        p += r;
        n -= r;
    }
}

void ScriptedOracle::send_frame(int kind, const void* payload, int len) {
    int32_t hdr[2] = {static_cast<int32_t>(kind), static_cast<int32_t>(len)};
    send_all(hdr, sizeof(hdr));
    if (len > 0) send_all(payload, len);
}

void ScriptedOracle::recv_frame(int expect_kind, void* payload, int len) {
    int32_t hdr[2];
    recv_all(hdr, sizeof(hdr));
    if (hdr[0] != expect_kind || hdr[1] != len)
        fatal_error("scripted oracle: unexpected reply frame (kind " +
                    std::to_string(hdr[0]) + " len " + std::to_string(hdr[1]) +
                    ", expected kind " + std::to_string(expect_kind) + " len " +
                    std::to_string(len) + ")");
    if (len > 0) recv_all(payload, len);
}

void ScriptedOracle::connect(const std::string& socket_path,
                             const std::string& deck_a,
                             const std::string& deck_b,
                             const std::string& spec) {
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) fatal_error("scripted oracle: socket() failed");
    sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (socket_path.size() >= sizeof(addr.sun_path))
        fatal_error("scripted oracle: socket path too long: " + socket_path);
    std::memcpy(addr.sun_path, socket_path.c_str(), socket_path.size() + 1);
    if (::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0)
        fatal_error("scripted oracle: cannot connect to " + socket_path +
                    " — is train/scripted_oracle.py running?");
    std::string hello = "{\"deck_a\": \"" + json_escape(deck_a) +
                        "\", \"deck_b\": \"" + json_escape(deck_b) +
                        "\", \"spec\": \"" + json_escape(spec) + "\"}";
    send_frame(kHello, hello.data(), static_cast<int>(hello.size()));
    int32_t obs_size = 0;
    recv_frame(kHello, &obs_size, sizeof(obs_size));
    if (obs_size != ACTOR_OBS_SIZE)
        fatal_error("scripted oracle: OBS_SIZE mismatch — oracle " +
                    std::to_string(obs_size) + ", actor " +
                    std::to_string(ACTOR_OBS_SIZE) +
                    " (rebuild/regenerate one side)");
}

void ScriptedOracle::new_game() {
    if (fd_ < 0) fatal_error("scripted oracle: new_game before connect");
    send_frame(kNewGame, nullptr, 0);
}

int ScriptedOracle::act(const float* obs, int num_choices) {
    if (fd_ < 0) fatal_error("scripted oracle: act before connect");
    const int payload_len =
        static_cast<int>(sizeof(int32_t)) +
        ACTOR_OBS_SIZE * static_cast<int>(sizeof(float));
    std::string payload(static_cast<size_t>(payload_len), '\0');
    int32_t nc = static_cast<int32_t>(num_choices);
    std::memcpy(&payload[0], &nc, sizeof(nc));
    std::memcpy(&payload[sizeof(nc)], obs,
                static_cast<size_t>(ACTOR_OBS_SIZE) * sizeof(float));
    send_frame(kQuery, payload.data(), payload_len);
    int32_t action = -1;
    recv_frame(kQuery, &action, sizeof(action));
    if (action < 0 || action >= num_choices)
        fatal_error("scripted oracle: returned out-of-menu action " +
                    std::to_string(action) + " (menu " +
                    std::to_string(num_choices) + ")");
    return static_cast<int>(action);
}
