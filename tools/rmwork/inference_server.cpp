#include "inference_server.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>

#include "files.h"
#include "model_zip.h"

namespace {

// How long a stopped server has to exit after SIGTERM before it is killed.
const double STOP_SECONDS = 3.0;
// How much of rmserve's stderr a failure shows.
const size_t STDERR_TAIL = 4000;

}  // namespace

static std::string stderr_tail(const std::string &path);
static bool numeric_id(const std::string &id);

static std::string stderr_tail(const std::string &path) {
    std::string text;
    if (!rmwork::read_file(path, &text)) return "";
    return text.size() > STDERR_TAIL ? text.substr(text.size() - STDERR_TAIL) : text;
}

// rmserve's policy ids are unsigned 32-bit integers.
static bool numeric_id(const std::string &id) {
    return !id.empty() && id.size() <= 9 && id.find_first_not_of("0123456789") == std::string::npos;
}

namespace rmwork {

std::string server_config_key(const ServerConfig &c) {
    std::string k = c.binary + '\n' + c.socket + '\n' + c.cwd;
    for (const auto &p : c.policies) k += '\n' + p.first + '=' + p.second;
    return k;
}

bool socket_listening(const std::string &path) {
    sockaddr_un addr;
    if (path.empty() || path.size() >= sizeof addr.sun_path) return false;
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return false;
    std::memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    const bool ok = connect(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof addr) == 0;
    close(fd);
    return ok;
}

void InferenceServer::start(const ServerConfig &c) {
    stop();
    key_ = server_config_key(c);
    message_.clear();
    detail_.clear();
    argv_.clear();
    socket_ = c.socket;
    sockaddr_un addr;
    if (c.socket.empty()) return fail("No server socket: set Settings \xE2\x86\x92 Models \xE2\x86\x92 Server socket.", "");
    if (c.socket.size() >= sizeof addr.sun_path)
        return fail("The server socket's path is too long for a Unix socket (" + std::to_string(c.socket.size()) +
                        " characters): " + c.socket,
            "");
    if (socket_listening(c.socket)) {
        state_ = State::EXTERNAL;
        return;
    }
    if (!is_executable(c.binary))
        return fail("No rmserve at " + c.binary + ". Build it with `make actor` (it needs libtorch).", "");
    argv_ = {c.binary, "--socket", c.socket};
    for (const auto &p : c.policies) {
        if (!numeric_id(p.first)) continue;
        std::string why;
        const std::string model = model_file(p.second, &why);
        if (model.empty()) return fail("Policy " + p.first + "'s model cannot be read: " + why, "");
        argv_.push_back("--policy");
        argv_.push_back(p.first + "=" + model);
    }
    if (argv_.size() == 3)
        return fail("No policy for rmserve to serve. Add one in Settings \xE2\x86\x92 Models: on the Policies' Add "
                    "row, give a number as its id and choose a model export, then press Add.",
            "");
    if (!make_dirs(c.log_dir) || !make_dirs(parent_dir(c.socket)))
        return fail("Cannot create " + c.log_dir + " or the socket's directory.", "");
    err_path_ = c.log_dir + "/rmserve.err";
    std::string err;
    if (!child_.start(argv_, c.cwd, c.log_dir + "/rmserve.out", err_path_, &err))
        return fail("rmserve did not start: " + err, "");
    owned_ = true;
    state_ = State::STARTING;
}

void InferenceServer::stop() {
    if (owned_) child_.terminate_and_reap(STOP_SECONDS);
    owned_ = false;
    state_ = State::OFF;
}

bool InferenceServer::poll() {
    if (!owned_ || (state_ != State::STARTING && state_ != State::RUNNING)) return false;
    if (!child_.poll()) {
        owned_ = false;
        fail("rmserve stopped (" + child_.status_text() + ").", stderr_tail(err_path_));
        return true;
    }
    if (state_ == State::STARTING && socket_listening(socket_)) state_ = State::RUNNING;
    return false;
}

void InferenceServer::fail(const std::string &message, const std::string &detail) {
    state_ = State::FAILED;
    message_ = message;
    detail_ = detail;
}

const char *server_state_text(InferenceServer::State s) {
    switch (s) {
        case InferenceServer::State::OFF:
            return "Inference: server off";
        case InferenceServer::State::EXTERNAL:
            return "Inference: socket present";
        case InferenceServer::State::STARTING:
            return "Inference: starting server";
        case InferenceServer::State::RUNNING:
            return "Inference: server running";
        case InferenceServer::State::FAILED:
            return "Inference: server failed";
    }
    return "";
}

}  // namespace rmwork
