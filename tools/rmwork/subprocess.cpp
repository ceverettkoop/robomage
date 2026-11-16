#include "subprocess.h"

#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>

extern char **environ;

namespace rmwork {

ChildProcess::~ChildProcess() {
    if (pid_ > 0 && !reaped_) kill_and_reap();
}

bool ChildProcess::start(const std::vector<std::string> &argv, const std::string &cwd, const std::string &stdout_path,
    const std::string &stderr_path, std::string *err) {
    if (argv.empty()) {
        *err = "no program";
        return false;
    }
    std::vector<std::string> store = argv;
    std::vector<char *> args;
    for (std::string &s : store) args.push_back(&s[0]);
    args.push_back(nullptr);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 1, stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    posix_spawn_file_actions_addopen(&actions, 2, stderr_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (!cwd.empty()) posix_spawn_file_actions_addchdir_np(&actions, cwd.c_str());
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, args[0], &actions, nullptr, args.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
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
    if (pid_ <= 0 || reaped_) return;
    kill(pid_, SIGKILL);
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    record(status);
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
