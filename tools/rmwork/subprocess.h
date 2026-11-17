#ifndef RMWORK_SUBPROCESS_H
#define RMWORK_SUBPROCESS_H

#include <spawn.h>
#include <sys/types.h>

#include <string>
#include <vector>

// ── A child process with its output in files (posix_spawn) ──────────────────────────────────────
//
// The workbench runs the engine as a subprocess, never linking it: posix_spawn (no fork, so the
// UI's memory is not copied). start(): stdin from /dev/null, stdout and stderr each into a file, so
// nothing has to drain a pipe while the caller waits. start_piped() (Play's session): stdin and
// stdout are pipes the caller writes and reads, stderr a file, and the child leads a process group
// of its own, so kill_and_reap() also kills what it started (a teacher subprocess holding the
// pipe). The caller polls (poll()) and may kill it; the destructor kills and reaps a child still
// running.

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
        // Starts argv[0] in `cwd` in a new process group with stdin and stdout on pipes: `to_child`
        // gets the write end of its stdin, `from_child` the read end of its stdout (both close on
        // exec, owned by the caller); stderr into `stderr_path`. False with `err`.
        bool start_piped(const std::vector<std::string> &argv, const std::string &cwd, const std::string &stderr_path,
            int *to_child, int *from_child, std::string *err);
        // True while the child runs; once it has exited, false with exited() set.
        bool poll();
        // SIGKILL (to its process group when start_piped started it), then reaps it.
        void kill_and_reap();
        bool exited() const { return reaped_; }
        // Exited normally with status 0.
        bool ok() const { return reaped_ && exit_code_ == 0 && signal_ == 0; }
        // "exit 1", "killed by signal 9".
        std::string status_text() const;

    private:
        void record(int status);
        bool spawn(const std::vector<std::string> &argv, posix_spawn_file_actions_t *actions,
            const posix_spawnattr_t *attr, std::string *err);
        pid_t pid_ = -1;
        bool group_ = false;  // the child leads its own process group
        bool reaped_ = false;
        int exit_code_ = -1;
        int signal_ = 0;
};

}  // namespace rmwork

#endif /* RMWORK_SUBPROCESS_H */
