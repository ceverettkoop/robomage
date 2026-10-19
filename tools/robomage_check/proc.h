#ifndef ROBOMAGE_CHECK_PROC_H
#define ROBOMAGE_CHECK_PROC_H

#include <sys/types.h>

#include <functional>
#include <string>
#include <vector>

// ── robomage_check: engine subprocesses (fork/exec, pipes, poll) and the job pool ─────────────

namespace rmcheck {

// One process to run to completion.
struct ProcSpec {
        std::vector<std::string> argv;  // argv[0] is the program path
        std::string cwd;                // empty = this process's
        std::string stdin_data;         // written to its stdin, which is then closed
        std::vector<std::string> env;   // KEY=VALUE entries added to (or replacing in) this environment
        double timeout_s = 600;         // killed past this many seconds
        bool merge_stderr = false;      // stderr into `out` (in write order) instead of `err`
};

struct ProcResult {
        bool started = false;    // false: the fork or exec failed (`err` says why)
        int exit_code = -1;      // the exit status, or -1 when killed by a signal
        int signal = 0;          // the signal that killed it, or 0
        bool timed_out = false;  // killed by the timeout
        std::string out, err;
        double seconds = 0;
        // Exited normally with status 0.
        bool ok() const { return started && !timed_out && signal == 0 && exit_code == 0; }
        // "exit N" / "killed by signal N" / "timed out after T s" / "could not start: …".
        std::string status_text() const;
};

ProcResult run_process(const ProcSpec &spec);

// One process driven interactively: its stdout and stderr are read as they come (both drained,
// so neither pipe can fill up), and lines are written to its stdin.
class Session {
    public:
        // Starts `spec` (its stdin_data and merge_stderr are not used). False when it cannot start.
        bool start(const ProcSpec &spec, std::string *err);
        ~Session();
        // Reads until stdout holds `n` more bytes than have been taken (true) or ends (false).
        bool fill(size_t n);
        // Reads until stdout holds a '\n' at or after the taken bytes (true) or ends (false).
        bool fill_line();
        // The stdout bytes read and not yet taken.
        const std::string &pending() const { return out_; }
        // Drops the first `n` pending stdout bytes.
        void take(size_t n);
        // The next RMSTATE stream record's payload ("RMSTATE <n>\n" + n bytes): true with the
        // payload, false at the end of stdout (`err` empty) or on a framing error (`err` says what).
        bool next_record(std::string *payload, std::string *err);
        // Writes `line` plus '\n' to stdin; false when stdin is closed.
        bool write_line(const std::string &line);
        void close_stdin();
        // Reads both pipes to their end and waits for the exit; returns the result with all stderr
        // (and any untaken stdout in `out`).
        ProcResult finish();
        // Everything read from stderr so far.
        const std::string &stderr_text() const { return err_; }
        // Reads until stderr holds `needle` (true) or both pipes end (false).
        bool wait_stderr(const std::string &needle);
        // Sends `sig` to the process.
        void send_signal(int sig);
        // Moves the deadline (the process is killed past it) to `seconds` from now.
        void set_timeout(double seconds);

    private:
        // One poll round over the open pipes; false when both are closed or the deadline passed.
        bool pump();
        void kill_child();
        pid_t pid_ = -1;
        int in_fd_ = -1, out_fd_ = -1, err_fd_ = -1;
        std::string out_, err_;
        double deadline_ = 0, started_at_ = 0;
        bool timed_out_ = false;
};

// Runs fn(0) … fn(n - 1) on `jobs` threads (at least one); returns when all are done.
void parallel_for(size_t n, int jobs, const std::function<void(size_t)> &fn);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_PROC_H */
