#include "teacher.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "../error.h"
#include "../game_driver.h"
#include "../state_format/agent_feed.h"
#include "../state_format/schema.h"
#include "py_random.h"

std::string teacher_command[2];
std::string teacher_mix_option;
std::string teacher_timeout_option;

namespace {

// One teacher process and the socket its stdin and stdout are.
struct TeacherProc {
        std::string command;
        pid_t pid = -1;
        int fd = -1;
        std::string in;  // bytes read and not yet taken
};

// Everything the channel keeps across the run.
struct TeacherRun {
        TeacherProc proc[2];
        double mix = 0;
        double timeout_s = 60;
        agents::PyRandom coin;
        // The current decision: the taught seat whose reply is pending or read (-1 none).
        int seat = -1;
        size_t rows = 0;
        size_t number = 0;
        bool read = false;
        int label = -1;
        agents::Behaviour mixed;
};

using Clock = std::chrono::steady_clock;

}  // namespace

static TeacherRun *teacher = nullptr;

static const char *seat_name(int s);
static std::string who(int s);
static double parse_number(const char *flag, const std::string &text, double lo, double hi);
static void spawn(int s);
static void close_fds_from(int lowest, int bound);
static void cleanup_at_exit();
[[noreturn]] static void teacher_failed(int s, const std::string &what);
static std::string reap(int s, double wait_s);
static Clock::time_point deadline_from_now();
static void wait_ready(int s, short events, Clock::time_point deadline, const std::string &doing);
static void write_all(int s, const std::string &data);
static void write_record(int s, const std::string &payload);
static std::string read_line(int s);
static int parse_label(int s, const std::string &line);
static void finish_proc(int s, const char *when);

static const char *seat_name(int s) {
    return s == 0 ? "A" : "B";
}

// "teacher A (`<command>`)", for messages.
static std::string who(int s) {
    return std::string("teacher ") + seat_name(s) + " (`" + teacher->proc[s].command + "`)";
}

// A decimal number in [lo, hi]; fatal otherwise.
static double parse_number(const char *flag, const std::string &text, double lo, double hi) {
    char *end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (text.empty() || end == text.c_str() || *end != '\0' || !(v >= lo && v <= hi))
        fatal_error(std::string(flag) + ": expected a number in [" + std::to_string(lo) + ", " + std::to_string(hi) +
                    "], got '" + text + "'");
    return v;
}

bool teacher_enabled() {
    return !teacher_command[0].empty() || !teacher_command[1].empty();
}

bool teacher_on(int s) {
    return teacher && (s == 0 || s == 1) && teacher->proc[s].pid > 0;
}

bool teacher_mixing() {
    return teacher && teacher->mix > 0;
}

void teacher_install(uint64_t coin_seed, const bool in_process[2]) {
    const bool running = teacher && (teacher->proc[0].pid > 0 || teacher->proc[1].pid > 0);
    if (!teacher_enabled() && !running) {
        if (!teacher_mix_option.empty() || !teacher_timeout_option.empty())
            fatal_error("--teacher-mix and --teacher-timeout need --teacher-a or --teacher-b");
        return;
    }
    if (!teacher) {
        teacher = new TeacherRun();
        std::atexit(cleanup_at_exit);
    }
    teacher->mix = teacher_mix_option.empty() ? 0 : parse_number("--teacher-mix", teacher_mix_option, 0, 1);
    teacher->timeout_s =
        teacher_timeout_option.empty() ? 60 : parse_number("--teacher-timeout", teacher_timeout_option, 0.001, 1e9);
    if (teacher->mix > 0 && replay_mode)
        fatal_error("--teacher-mix: under --replay the decision log decides every reply; a teacher only labels");
    for (int s = 0; s < 2; ++s)
        if (!teacher_command[s].empty() && teacher->mix > 0 && !(in_process && in_process[s]))
            fatal_error(std::string("--teacher-mix: seat ") + seat_name(s) +
                        " is taught but not decided by an in-process agent or script (--agent-" + (s == 0 ? "a" : "b") +
                        ", --play, --actions); its replies cannot be mixed");
    teacher->coin.seed_int(coin_seed + (uint64_t{1} << 32));
    teacher->seat = -1;
    for (int s = 0; s < 2; ++s) {
        if (teacher->proc[s].pid > 0 && teacher->proc[s].command != teacher_command[s])
            finish_proc(s, "when the match's teacher of its seat changed");
        if (!teacher_command[s].empty() && teacher->proc[s].pid <= 0) spawn(s);
    }
}

// Starts seat s's teacher: /bin/sh -c <command>, its stdin and stdout one end of a stream socket
// pair, every other descriptor but stderr closed.
static void spawn(int s) {
    TeacherProc &p = teacher->proc[s];
    p.command = teacher_command[s];
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0)
        fatal_error(std::string("--teacher-") + (s == 0 ? "a" : "b") + ": socketpair: " + std::strerror(errno));
    const long open_max = sysconf(_SC_OPEN_MAX);
    const int bound = open_max > 0 && open_max < 65536 ? static_cast<int>(open_max) : 65536;
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) fatal_error(std::string("--teacher-") + (s == 0 ? "a" : "b") + ": fork: " + std::strerror(errno));
    if (pid == 0) {
        // The engine ends its teacher (end of input); a terminal's interrupt is the engine's to handle.
        signal(SIGINT, SIG_IGN);
        if (dup2(sv[1], STDIN_FILENO) < 0 || dup2(sv[1], STDOUT_FILENO) < 0) _exit(127);
        close_fds_from(STDERR_FILENO + 1, bound);
        execl("/bin/sh", "sh", "-c", p.command.c_str(), static_cast<char *>(nullptr));
        _exit(127);
    }
    close(sv[1]);
    p.pid = pid;
    p.fd = sv[0];
    fcntl(p.fd, F_SETFL, fcntl(p.fd, F_GETFL) | O_NONBLOCK);
}

// In the forked child: closes every descriptor from `lowest` (close_range where the kernel has it).
static void close_fds_from(int lowest, int bound) {
#ifdef SYS_close_range
    if (syscall(SYS_close_range, static_cast<unsigned>(lowest), ~0U, 0U) == 0) return;
#endif
    for (int fd = lowest; fd < bound; ++fd) close(fd);
}

// At exit: every teacher still running gets end of input, a moment to exit, then SIGKILL.
static void cleanup_at_exit() {
    if (!teacher) return;
    for (int s = 0; s < 2; ++s)
        if (teacher->proc[s].pid > 0) reap(s, 2.0);
}

// Closes seat s's socket and waits up to `wait_s` for its teacher to exit (then kills it); returns
// how it ended ("exited with status N", "was killed by signal N").
static std::string reap(int s, double wait_s) {
    TeacherProc &p = teacher->proc[s];
    if (p.fd >= 0) close(p.fd);
    p.fd = -1;
    if (p.pid <= 0) return "had ended";
    const Clock::time_point deadline =
        Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(wait_s));
    int status = 0;
    pid_t r;
    while ((r = waitpid(p.pid, &status, WNOHANG)) == 0 && Clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    if (r == 0) {
        kill(p.pid, SIGKILL);
        r = waitpid(p.pid, &status, 0);
    }
    p.pid = -1;
    if (r < 0) return "could not be waited for";
    if (WIFEXITED(status)) return "exited with status " + std::to_string(WEXITSTATUS(status));
    if (WIFSIGNALED(status)) return "was killed by signal " + std::to_string(WTERMSIG(status));
    return "ended";
}

// A teacher failure: the run ends with a fatal error naming the seat and how the teacher ended.
[[noreturn]] static void teacher_failed(int s, const std::string &what) {
    const std::string name = who(s);
    const std::string ended = reap(s, 2.0);
    fatal_error(name + ": " + what + " (it " + ended + ")");
}

// Waits until seat s's socket is ready for `events`, or fails the run at the deadline.
static void wait_ready(int s, short events, Clock::time_point deadline, const std::string &doing) {
    for (;;) {
        const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        if (left <= 0) {
            char limit[32];
            std::snprintf(limit, sizeof limit, "%g", teacher->timeout_s);
            teacher_failed(s, "no progress " + doing + " within " + limit + " s (--teacher-timeout)");
        }
        pollfd pfd{teacher->proc[s].fd, events, 0};
        const int r = poll(&pfd, 1, static_cast<int>(left < 1000000 ? left : 1000000));
        if (r > 0) return;
        if (r < 0 && errno != EINTR) teacher_failed(s, std::string("poll: ") + std::strerror(errno));
    }
}

// The deadline of one wait: --teacher-timeout from now.
static Clock::time_point deadline_from_now() {
    return Clock::now() +
           std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(teacher->timeout_s));
}

static void write_all(int s, const std::string &data) {
    const Clock::time_point deadline = deadline_from_now();
    size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = send(teacher->proc[s].fd, data.data() + off, data.size() - off, MSG_NOSIGNAL);
        if (n > 0) {
            off += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            wait_ready(s, POLLOUT, deadline, "reading its frames");
            continue;
        }
        teacher_failed(s, "stopped reading its frames at decision " + std::to_string(teacher->number) + " (" +
                              std::strerror(n < 0 ? errno : EPIPE) + ")");
    }
}

static void write_record(int s, const std::string &payload) {
    write_all(s, rmstate::frame_record(payload));
}

// The next reply line (without its '\n') from seat s's teacher.
static std::string read_line(int s) {
    TeacherProc &p = teacher->proc[s];
    const Clock::time_point deadline = deadline_from_now();
    size_t nl;
    while ((nl = p.in.find('\n')) == std::string::npos) {
        if (p.in.size() > 4096) teacher_failed(s, "a reply line longer than 4096 bytes");
        wait_ready(s, POLLIN, deadline, "replying to decision " + std::to_string(teacher->number));
        char buf[4096];
        const ssize_t n = recv(p.fd, buf, sizeof buf, 0);
        if (n > 0) {
            p.in.append(buf, static_cast<size_t>(n));
            continue;
        }
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        teacher_failed(s, "closed its output before replying to decision " + std::to_string(teacher->number));
    }
    std::string line = p.in.substr(0, nl);
    p.in.erase(0, nl + 1);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
}

// A reply as a label: a menu row (-1 the last row), or -1 for a concession sentinel; fatal otherwise.
static int parse_label(int s, const std::string &line) {
    const char *text = line.c_str();
    while (*text == ' ' || *text == '\t') ++text;
    char *end = nullptr;
    errno = 0;
    const long v = std::strtol(text, &end, 10);
    while (end && (*end == ' ' || *end == '\t')) ++end;
    const long rows = static_cast<long>(teacher->rows);
    const bool parsed = end != text && *end == '\0' && errno == 0;
    if (parsed && v >= 0 && v < rows) return static_cast<int>(v);
    if (parsed && v == -1 && rows > 0) return static_cast<int>(rows - 1);
    if (parsed && (v == -2 || v == -3)) return -1;
    teacher_failed(s, "reply '" + line + "' at decision " + std::to_string(teacher->number) + " is no menu row of " +
                          std::to_string(rows) + ", -1, -2 or -3");
}

void teacher_send(
    int s, const std::vector<std::string> &match_frames, const std::string &state, size_t menu_rows, size_t number) {
    if (!teacher_on(s)) return;
    if (teacher->seat >= 0) fatal_error("teacher channel: a decision was sent before the last one was closed");
    teacher->seat = s;
    teacher->rows = menu_rows;
    teacher->number = number;
    teacher->read = false;
    teacher->label = -1;
    for (const std::string &payload : match_frames) write_record(s, payload);
    write_record(s, state);
}

int teacher_label() {
    if (!teacher || teacher->seat < 0) return -1;
    if (!teacher->read) {
        teacher->label = parse_label(teacher->seat, read_line(teacher->seat));
        teacher->read = true;
    }
    return teacher->label;
}

int teacher_take_label() {
    const int label = teacher_label();
    if (teacher) teacher->seat = -1;
    return label;
}

int teacher_mix(
    int agent_reply, const agents::Behaviour *agent_behaviour, size_t menu_rows, const agents::Behaviour **recorded) {
    *recorded = agent_behaviour;
    if (!teacher_mixing() || teacher->seat < 0) return agent_reply;
    const int label = teacher_label();
    const bool play_label = teacher->coin.random() < teacher->mix;
    if (label < 0) return agent_reply;
    const double p = teacher->mix;
    agents::Behaviour &m = teacher->mixed;
    const bool agent_mu = agent_behaviour && agent_behaviour->mu.size() == menu_rows;
    if (agent_mu) {
        m = *agent_behaviour;
        for (size_t k = 0; k < menu_rows; ++k)
            m.mu[k] = p * (static_cast<int>(k) == label ? 1.0 : 0.0) + (1 - p) * agent_behaviour->mu[k];
        *recorded = &m;
    } else if (p == 1) {
        m = agents::Behaviour();
        m.mu.assign(menu_rows, 0.0);
        m.mu[static_cast<size_t>(label)] = 1.0;
        *recorded = &m;
    } else {
        *recorded = nullptr;
    }
    return play_label ? label : agent_reply;
}

void teacher_match_over() {
    if (!teacher) return;
    for (int s = 0; s < 2; ++s) {
        if (!teacher_on(s)) continue;
        std::vector<std::string> match_frames;
        std::string state;
        agent_feed_terminal(s == 0 ? Zone::PLAYER_A : Zone::PLAYER_B, &match_frames, &state);
        teacher->number = 0;
        for (const std::string &payload : match_frames) write_record(s, payload);
        write_record(s, state);
    }
}

// Ends seat s's teacher: end of input, then it must exit with status 0 having written nothing more
// (`when` names the moment in a failure's message).
static void finish_proc(int s, const char *when) {
    TeacherProc &p = teacher->proc[s];
    shutdown(p.fd, SHUT_WR);
    const Clock::time_point deadline = deadline_from_now();
    for (;;) {
        wait_ready(s, POLLIN, deadline, "exiting at the end of its input");
        char buf[4096];
        const ssize_t n = recv(p.fd, buf, sizeof buf, 0);
        if (n == 0) break;
        if (n < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (n < 0) break;
        p.in.append(buf, static_cast<size_t>(n));
    }
    if (!p.in.empty()) teacher_failed(s, "wrote '" + p.in.substr(0, 40) + "' after its last decision");
    const std::string name = who(s);
    const std::string ended = reap(s, teacher->timeout_s);
    if (ended != "exited with status 0") fatal_error(name + ": " + ended + " " + when);
    p.in.clear();
}

void teacher_finish() {
    if (!teacher) return;
    for (int s = 0; s < 2; ++s)
        if (teacher->proc[s].pid > 0) finish_proc(s, "at the end of the run");
}
