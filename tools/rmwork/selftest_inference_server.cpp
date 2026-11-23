#include "selftest_inference_server.h"

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>

#include "files.h"
#include "inference_server.h"
#include "paths.h"
#include "settings.h"

static bool write_script(const std::string &path, const std::string &body);
static int listen_at(const std::string &path);
static bool wait_for(rmwork::InferenceServer *s, rmwork::InferenceServer::State want, double seconds);

// An executable shell script.
static bool write_script(const std::string &path, const std::string &body) {
    std::string err;
    return rmwork::write_file_atomic(path, "#!/bin/sh\n" + body + "\n", &err) && chmod(path.c_str(), 0755) == 0;
}

// A listening Unix socket at `path` (the descriptor; -1 when it cannot be made).
static int listen_at(const std::string &path) {
    sockaddr_un addr;
    if (path.size() >= sizeof addr.sun_path) return -1;
    unlink(path.c_str());
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    std::memset(&addr, 0, sizeof addr);
    addr.sun_family = AF_UNIX;
    std::memcpy(addr.sun_path, path.c_str(), path.size() + 1);
    if (bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof addr) != 0 || listen(fd, 4) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

// Polls `s` until it reaches `want` or `seconds` pass.
static bool wait_for(rmwork::InferenceServer *s, rmwork::InferenceServer::State want, double seconds) {
    for (double t = 0; t < seconds; t += 0.02) {
        s->poll();
        if (s->state() == want) return true;
        usleep(20000);
    }
    return s->state() == want;
}

namespace rmwork {

void check_inference_server_units(const CheckFn &check) {
    using State = InferenceServer::State;
    // A short directory: a Unix socket's path is limited to about 100 characters.
    const std::string dir = make_temp_dir("rmws_");
    const std::string args = dir + "/args.txt", model = dir + "/m.ts.pt";
    std::string err;
    write_file_atomic(model, "model", &err);
    const std::string sleeper = dir + "/sleeper", recorder = dir + "/recorder", failer = dir + "/failer";
    check(write_script(sleeper, "exec sleep 30") && write_script(recorder, "echo \"$@\" > " + args + "\nexec sleep 30") &&
              write_script(failer, "echo 'rmserve: model refused: tables_digest differs' >&2\nexit 2"),
        "Server: the stand-in servers are not written");

    ServerConfig c;
    c.binary = recorder;
    c.socket = dir + "/s.sock";
    c.cwd = dir;
    c.log_dir = dir + "/log";
    c.policies = {{"0", model}, {"name", model}};
    InferenceServer s;
    s.start(c);
    check(s.state() == State::STARTING, "Server: a started server is not STARTING");
    for (int i = 0; i < 100 && !file_exists(args); ++i) usleep(20000);
    std::string got;
    read_file(args, &got);
    check(got == "--socket " + c.socket + " --policy 0=" + model + "\n",
        "Server: rmserve's arguments are not --socket and the numeric policy: " + got);
    const int fd = listen_at(c.socket);
    check(fd >= 0 && wait_for(&s, State::RUNNING, 2.0), "Server: a listening socket does not make it RUNNING");
    s.stop();
    check(s.state() == State::OFF, "Server: a stopped server is not OFF");
    if (fd >= 0) close(fd);
    unlink(c.socket.c_str());

    c.binary = failer;
    s.start(c);
    check(wait_for(&s, State::FAILED, 2.0) && s.detail().find("tables_digest differs") != std::string::npos,
        "Server: a server that exits is not FAILED with its stderr");

    c.binary = sleeper;
    const int ext = listen_at(c.socket);
    s.start(c);
    check(ext >= 0 && s.state() == State::EXTERNAL && s.argv().empty(),
        "Server: a socket already listening is not taken as EXTERNAL");
    s.stop();
    check(socket_listening(c.socket), "Server: stopping an EXTERNAL server touched it");
    if (ext >= 0) close(ext);
    unlink(c.socket.c_str());

    c.binary = dir + "/missing";
    s.start(c);
    check(s.state() == State::FAILED && s.message().find("make actor") != std::string::npos,
        "Server: a missing rmserve is not refused naming make actor");
    c.binary = sleeper;
    c.policies = {{"name", model}};
    s.start(c);
    check(s.state() == State::FAILED, "Server: no numeric policy is not refused");
    c.policies = {{"0", dir + "/none.zip"}};
    s.start(c);
    check(s.state() == State::FAILED && s.message().find("Policy 0") != std::string::npos,
        "Server: an unreadable zipped policy is not refused");
    c.socket = dir + "/" + std::string(120, 's');
    c.policies = {{"0", model}};
    s.start(c);
    check(s.state() == State::FAILED, "Server: a socket path too long is not refused");
    s.stop();

    Settings set = default_settings();
    check(!set.models.start_server && socket_path(set).empty(), "Server: off by default, with no socket");
    set.models.start_server = true;
    check(socket_path(set) == default_server_socket(), "Server: the default socket is not used when it starts");
    Settings back;
    check(settings_from_json(settings_to_json(set), &back, &err) && back.models.start_server,
        "Server: start_server does not round-trip through settings.json");
    check(socket_listening("") == false, "Server: an empty socket path listens");
    remove_tree(dir);
}

}  // namespace rmwork
