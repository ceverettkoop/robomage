#include "subprocess.h"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

extern char **environ;

static void close_pair(int fds[2]);
static int pipe_cloexec(int fds[2]);

static void close_pair(int fds[2]) {
    for (int i = 0; i < 2; ++i)
        if (fds[i] >= 0) close(fds[i]);
}

// A pipe whose ends are close-on-exec (atomically where pipe2 exists).
static int pipe_cloexec(int fds[2]) {
#ifdef __APPLE__
    if (pipe(fds) != 0) return -1;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC);
    fcntl(fds[1], F_SETFD, FD_CLOEXEC);
    return 0;
#else
    return pipe2(fds, O_CLOEXEC);
#endif
}

namespace rmwork {

ChildProcess::~ChildProcess() {
    if (pid_ > 0 && (!reaped_ || group_)) kill_and_reap();
}

bool ChildProcess::start(const std::vector<std::string> &argv, const std::string &cwd, const std::string &stdout_path,
    const std::string &stderr_path, std::string *err, bool own_group) {
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 1, stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (stderr_path == stdout_path)
        posix_spawn_file_actions_adddup2(&actions, 1, 2);
    else
        posix_spawn_file_actions_addopen(&actions, 2, stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    if (own_group) {
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETPGROUP);
        posix_spawnattr_setpgroup(&attr, 0);
    }
    group_ = own_group;
    const bool ok = spawn(argv, &actions, &attr, err);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    return ok;
}

bool ChildProcess::start_piped(const std::vector<std::string> &argv, const std::string &cwd,
    const std::string &stderr_path, int *to_child, int *from_child, std::string *err) {
    int in[2] = {-1, -1}, out[2] = {-1, -1};
    if (pipe_cloexec(in) != 0 || pipe_cloexec(out) != 0) {
        *err = std::string("cannot make a pipe: ") + std::strerror(errno);
        close_pair(in);
        close_pair(out);
        return false;
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, in[0], 0);
    posix_spawn_file_actions_adddup2(&actions, out[1], 1);
    posix_spawn_file_actions_addopen(&actions, 2, stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
    posix_spawnattr_t attr;
    posix_spawnattr_init(&attr);
    short flags = POSIX_SPAWN_SETPGROUP;
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    // Without pipe2 a pipe is close-on-exec only after it is made; the child gets only 0, 1 and 2.
    flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#endif
    posix_spawnattr_setflags(&attr, flags);
    posix_spawnattr_setpgroup(&attr, 0);
    group_ = true;
    const bool ok = spawn(argv, &actions, &attr, err);
    posix_spawnattr_destroy(&attr);
    posix_spawn_file_actions_destroy(&actions);
    close(in[0]);
    close(out[1]);
    if (!ok) {
        close(in[1]);
        close(out[0]);
        return false;
    }
    *to_child = in[1];
    *from_child = out[0];
    return true;
}

bool ChildProcess::spawn(const std::vector<std::string> &argv, posix_spawn_file_actions_t *actions,
    const posix_spawnattr_t *attr, std::string *err) {
    if (argv.empty()) {
        *err = "no program";
        return false;
    }
    std::vector<std::string> store = argv;
    std::vector<char *> args;
    for (std::string &s : store) args.push_back(&s[0]);
    args.push_back(nullptr);
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, args[0], actions, attr, args.data(), environ);
    if (rc != 0) {
        *err = "cannot start " + argv[0] + ": " + std::strerror(rc);
        return false;
    }
    pid_ = pid;
    reaped_ = false;
    exit_code_ = -1;
    signal_ = 0;
    return true;
}

bool ChildProcess::poll() {
    if (pid_ <= 0 || reaped_) return false;
    int status = 0;
    pid_t r = 0;
    do {
        r = waitpid(pid_, &status, WNOHANG);
    } while (r < 0 && errno == EINTR);
    if (r == 0) return true;
    record(r == pid_ ? status : 0);
    return false;
}

void ChildProcess::kill_and_reap() {
    if (pid_ <= 0) return;
    // The group too once the child is gone: what it started may still hold its pipes.
    if (group_) kill(-pid_, SIGKILL);
    group_ = false;
    if (reaped_) return;
    kill(pid_, SIGKILL);
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    record(status);
}

void ChildProcess::terminate_and_reap(double seconds) {
    if (pid_ <= 0 || reaped_) {
        kill_and_reap();
        return;
    }
    kill(group_ ? -pid_ : pid_, SIGTERM);
    const useconds_t step_us = 20000;
    for (double waited = 0; waited < seconds; waited += step_us / 1e6) {
        if (!poll()) break;
        usleep(step_us);
    }
    kill_and_reap();
}

std::string ChildProcess::status_text() const {
    if (!reaped_) return "running";
    if (signal_) return "killed by signal " + std::to_string(signal_);
    return "exit " + std::to_string(exit_code_);
}

void ChildProcess::record(int status) {
    reaped_ = true;
    if (WIFEXITED(status)) {
        exit_code_ = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        signal_ = WTERMSIG(status);
    }
}

}  // namespace rmwork
