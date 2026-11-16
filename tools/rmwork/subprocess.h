#ifndef RMWORK_SUBPROCESS_H
#define RMWORK_SUBPROCESS_H

#include <sys/types.h>

#include <string>
#include <vector>

// ── A child process with its output in files (posix_spawn) ──────────────────────────────────────
//
// The workbench runs the engine as a subprocess, never linking it: posix_spawn (no fork, so the
// UI's memory is not copied), stdin from /dev/null, stdout and stderr each into a file, so nothing
// has to drain a pipe while the caller waits. The caller polls (poll()) and may kill it; the
// destructor kills and reaps a child still running.

namespace rmwork {

class ChildProcess {
    public:
        ChildProcess() = default;
        ChildProcess(const ChildProcess &) = delete;
        ChildProcess &operator=(const ChildProcess &) = delete;
        ~ChildProcess();

        // Starts argv[0] (a path) with `argv` in `cwd` ("" = this process's), stdout into
        // `stdout_path` and stderr into `stderr_path` (each created or truncated). False with `err`.
        bool start(const std::vector<std::string> &argv, const std::string &cwd, const std::string &stdout_path,
            const std::string &stderr_path, std::string *err);
        // True while the child runs; once it has exited, false with exited() set.
        bool poll();
        // SIGKILL, then reaps it.
        void kill_and_reap();
        bool exited() const { return reaped_; }
        // Exited normally with status 0.
        bool ok() const { return reaped_ && exit_code_ == 0 && signal_ == 0; }
        // "exit 1", "killed by signal 9".
        std::string status_text() const;

    private:
        void record(int status);
        pid_t pid_ = -1;
        bool reaped_ = false;
        int exit_code_ = -1;
        int signal_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_SUBPROCESS_H */
