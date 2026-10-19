#include "proc.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <thread>

#include "util.h"

extern char **environ;

// The argv and environment of a child, built before fork (the child may only call
// async-signal-safe functions until it execs).
struct ChildImage {
        std::vector<std::string> argv_store, env_store;
        std::vector<char *> argv, envp;
};

static void build_image(const rmcheck::ProcSpec &spec, ChildImage *img);
static pid_t spawn(const rmcheck::ProcSpec &spec, int *in_fd, int *out_fd, int *err_fd, bool merge, std::string *err);
static void set_nonblocking(int fd);
static bool read_some(int fd, std::string *into);
static void close_fd(int *fd);
static void reap(pid_t pid, rmcheck::ProcResult *r);

static void build_image(const rmcheck::ProcSpec &spec, ChildImage *img) {
    img->argv_store = spec.argv;
    for (char **e = environ; e && *e; ++e) {
        const std::string entry = *e;
        const std::string key = entry.substr(0, entry.find('='));
        bool replaced = false;
        for (const std::string &add : spec.env)
            if (add.substr(0, add.find('=')) == key) replaced = true;
        if (!replaced) img->env_store.push_back(entry);
    }
    for (const std::string &add : spec.env) img->env_store.push_back(add);
    for (std::string &s : img->argv_store) img->argv.push_back(&s[0]);
    img->argv.push_back(nullptr);
    for (std::string &s : img->env_store) img->envp.push_back(&s[0]);
    img->envp.push_back(nullptr);
}

// Forks and execs `spec`; the parent's ends of the pipes come back non-blocking and close-on-exec
// (so a child another thread starts never holds them open).
static pid_t spawn(const rmcheck::ProcSpec &spec, int *in_fd, int *out_fd, int *err_fd, bool merge, std::string *err) {
    if (spec.argv.empty()) {
        *err = "no program";
        return -1;
    }
    ChildImage img;
    build_image(spec, &img);
    int in_p[2], out_p[2], err_p[2] = {-1, -1};
    if (pipe2(in_p, O_CLOEXEC) != 0 || pipe2(out_p, O_CLOEXEC) != 0 || (!merge && pipe2(err_p, O_CLOEXEC) != 0)) {
        *err = std::string("pipe: ") + std::strerror(errno);
        return -1;
    }
    const char *cwd = spec.cwd.empty() ? nullptr : spec.cwd.c_str();
    const pid_t pid = fork();
    if (pid < 0) {
        *err = std::string("fork: ") + std::strerror(errno);
        return -1;
    }
    if (pid == 0) {
        dup2(in_p[0], 0);
        dup2(out_p[1], 1);
        dup2(merge ? out_p[1] : err_p[1], 2);
        if (cwd && chdir(cwd) != 0) _exit(127);
        execve(img.argv[0], img.argv.data(), img.envp.data());
        const char msg[] = "robomage_check: exec failed\n";
        ssize_t ignored = write(2, msg, sizeof msg - 1);
        (void)ignored;
        _exit(127);
    }
    close(in_p[0]);
    close(out_p[1]);
    if (!merge) close(err_p[1]);
    *in_fd = in_p[1];
    *out_fd = out_p[0];
    *err_fd = merge ? -1 : err_p[0];
    set_nonblocking(*in_fd);
    set_nonblocking(*out_fd);
    if (*err_fd >= 0) set_nonblocking(*err_fd);
    return pid;
}

static void set_nonblocking(int fd) {
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);
}

// Reads what `fd` has; false at its end (or on an error).
static bool read_some(int fd, std::string *into) {
    char buf[65536];
    while (true) {
        const ssize_t n = read(fd, buf, sizeof buf);
        if (n > 0) {
            into->append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n == 0) return false;
        if (errno == EINTR) continue;
        return errno == EAGAIN || errno == EWOULDBLOCK;
    }
}

static void close_fd(int *fd) {
    if (*fd >= 0) close(*fd);
    *fd = -1;
}

static void reap(pid_t pid, rmcheck::ProcResult *r) {
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (WIFEXITED(status)) {
        r->exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        r->signal = WTERMSIG(status);
    }
}

namespace rmcheck {

std::string ProcResult::status_text() const {
    if (!started) return "could not start: " + err;
    if (timed_out) return "timed out after " + fmt1(seconds) + " s";
    if (signal) return "killed by signal " + std::to_string(signal);
    return "exit " + std::to_string(exit_code);
}

ProcResult run_process(const ProcSpec &spec) {
    ProcResult r;
    const double t0 = now_seconds();
    int in_fd = -1, out_fd = -1, err_fd = -1;
    const pid_t pid = spawn(spec, &in_fd, &out_fd, &err_fd, spec.merge_stderr, &r.err);
    if (pid < 0) return r;
    r.started = true;
    size_t written = 0;
    if (spec.stdin_data.empty()) close_fd(&in_fd);
    const double deadline = t0 + spec.timeout_s;
    while (out_fd >= 0 || err_fd >= 0) {
        pollfd fds[3];
        int n = 0;
        if (out_fd >= 0) fds[n++] = {out_fd, POLLIN, 0};
        if (err_fd >= 0) fds[n++] = {err_fd, POLLIN, 0};
        if (in_fd >= 0) fds[n++] = {in_fd, POLLOUT, 0};
        const double left = deadline - now_seconds();
        if (left <= 0) {
            kill(pid, SIGKILL);
            r.timed_out = true;
            break;
        }
        const int ms = left > 1e6 ? 1000000000 : static_cast<int>(left * 1000) + 1;
        if (poll(fds, static_cast<nfds_t>(n), ms) < 0 && errno != EINTR) break;
        for (int i = 0; i < n; ++i) {
            if (!fds[i].revents) continue;
            if (fds[i].fd == out_fd) {
                if (!read_some(out_fd, &r.out)) close_fd(&out_fd);
            } else if (fds[i].fd == err_fd) {
                if (!read_some(err_fd, &r.err)) close_fd(&err_fd);
            } else if (fds[i].fd == in_fd) {
                if (fds[i].revents & (POLLERR | POLLHUP)) {
                    close_fd(&in_fd);
                    continue;
                }
                const ssize_t w = write(in_fd, spec.stdin_data.data() + written, spec.stdin_data.size() - written);
                if (w > 0) written += static_cast<size_t>(w);
                if ((w < 0 && errno != EAGAIN && errno != EINTR) || written == spec.stdin_data.size()) close_fd(&in_fd);
            }
        }
    }
    close_fd(&in_fd);
    close_fd(&out_fd);
    close_fd(&err_fd);
    reap(pid, &r);
    r.seconds = now_seconds() - t0;
    return r;
}

bool Session::start(const ProcSpec &spec, std::string *err) {
    started_at_ = now_seconds();
    deadline_ = started_at_ + spec.timeout_s;
    pid_ = spawn(spec, &in_fd_, &out_fd_, &err_fd_, false, err);
    return pid_ >= 0;
}

Session::~Session() {
    if (pid_ >= 0) {
        kill_child();
        ProcResult ignored;
        reap(pid_, &ignored);
    }
    close_fd(&in_fd_);
    close_fd(&out_fd_);
    close_fd(&err_fd_);
}

void Session::kill_child() {
    if (pid_ >= 0) kill(pid_, SIGKILL);
}

bool Session::pump() {
    if (out_fd_ < 0 && err_fd_ < 0) return false;
    const double left = deadline_ - now_seconds();
    if (left <= 0) {
        timed_out_ = true;
        kill_child();
        close_fd(&out_fd_);
        close_fd(&err_fd_);
        return false;
    }
    pollfd fds[2];
    int n = 0;
    if (out_fd_ >= 0) fds[n++] = {out_fd_, POLLIN, 0};
    if (err_fd_ >= 0) fds[n++] = {err_fd_, POLLIN, 0};
    const int ms = left > 1e6 ? 1000000000 : static_cast<int>(left * 1000) + 1;
    if (poll(fds, static_cast<nfds_t>(n), ms) < 0 && errno != EINTR) return false;
    for (int i = 0; i < n; ++i) {
        if (!fds[i].revents) continue;
        if (fds[i].fd == out_fd_) {
            if (!read_some(out_fd_, &out_)) close_fd(&out_fd_);
        } else if (!read_some(err_fd_, &err_)) {
            close_fd(&err_fd_);
        }
    }
    return true;
}

bool Session::fill(size_t n) {
    while (out_.size() < n) {
        if (out_fd_ < 0) return false;
        if (!pump()) return out_.size() >= n;
    }
    return true;
}

bool Session::fill_line() {
    while (out_.find('\n') == std::string::npos) {
        if (out_fd_ < 0) return false;
        if (!pump()) return out_.find('\n') != std::string::npos;
    }
    return true;
}

bool Session::wait_stderr(const std::string &needle) {
    while (err_.find(needle) == std::string::npos)
        if (!pump()) return err_.find(needle) != std::string::npos;
    return true;
}

void Session::send_signal(int sig) {
    if (pid_ >= 0) kill(pid_, sig);
}

void Session::set_timeout(double seconds) {
    deadline_ = now_seconds() + seconds;
}

void Session::take(size_t n) {
    out_.erase(0, n);
}

bool Session::next_record(std::string *payload, std::string *err) {
    err->clear();
    if (!fill_line()) {
        if (!out_.empty()) *err = "the stream ends inside a record header";
        if (timed_out_) *err = "timed out";
        return false;
    }
    const size_t nl = out_.find('\n');
    const std::string head = out_.substr(0, nl);
    if (!starts_with(head, "RMSTATE ") || head.size() == 8 ||
        head.find_first_not_of("0123456789", 8) != std::string::npos) {
        *err = "bad record header " + quoted(head.substr(0, 40));
        return false;
    }
    const size_t n = std::strtoull(head.c_str() + 8, nullptr, 10);
    if (!fill(nl + 1 + n)) {
        *err = timed_out_ ? "timed out" : "the stream ends inside a record";
        return false;
    }
    *payload = out_.substr(nl + 1, n);
    take(nl + 1 + n);
    return true;
}

bool Session::write_line(const std::string &line) {
    if (in_fd_ < 0) return false;
    const std::string data = line + "\n";
    size_t done = 0;
    while (done < data.size()) {
        const ssize_t w = write(in_fd_, data.data() + done, data.size() - done);
        if (w > 0) {
            done += static_cast<size_t>(w);
            continue;
        }
        if (w < 0 && (errno == EAGAIN || errno == EINTR)) {
            // The child is not reading: keep its output pipes drained while waiting.
            pollfd p = {in_fd_, POLLOUT, 0};
            if (out_fd_ >= 0 || err_fd_ >= 0) pump();
            poll(&p, 1, 100);
            if (now_seconds() > deadline_) return false;
            continue;
        }
        close_fd(&in_fd_);
        return false;
    }
    return true;
}

void Session::close_stdin() {
    close_fd(&in_fd_);
}

ProcResult Session::finish() {
    ProcResult r;
    r.started = pid_ >= 0;
    close_fd(&in_fd_);
    while (pump()) {
    }
    r.timed_out = timed_out_;
    if (pid_ >= 0) {
        reap(pid_, &r);
        pid_ = -1;
    }
    r.out = out_;
    r.err = err_;
    r.seconds = now_seconds() - started_at_;
    return r;
}

void parallel_for(size_t n, int jobs, const std::function<void(size_t)> &fn) {
    std::atomic<size_t> next(0);
    const size_t workers = std::max<size_t>(1, std::min<size_t>(n, static_cast<size_t>(jobs > 0 ? jobs : 1)));
    std::vector<std::thread> pool;
    for (size_t w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            for (size_t i = next++; i < n; i = next++) fn(i);
        });
    for (std::thread &t : pool) t.join();
}

}  // namespace rmcheck
