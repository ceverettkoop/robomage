#include "server.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <utility>

#include "../../src/featurize/rmtb.h"

namespace {

// The self-pipe the signal handler writes and the loop polls.
int signal_pipe[2] = {-1, -1};

}  // namespace

static void on_signal(int);
static void install_signals();
static bool set_nonblocking(int fd);
static double now_seconds();

static void on_signal(int) {
    const char b = 1;
    const ssize_t ignored = write(signal_pipe[1], &b, 1);
    (void)ignored;
}

static void install_signals() {
    if (pipe(signal_pipe) == 0) {
        set_nonblocking(signal_pipe[0]);
        set_nonblocking(signal_pipe[1]);
    }
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);
}

static bool set_nonblocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

namespace rmserve {

Server::Server() = default;

Server::~Server() {
    for (auto &c : clients_)
        if (c->fd >= 0) close(c->fd);
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        unlink(opt_.socket.c_str());
    }
}

void Server::log(const std::string &line) const {
    std::fprintf(stderr, "rmserve: %s\n", line.c_str());
    std::fflush(stderr);
}

bool Server::start(const Options &opt, std::string *err) {
    opt_ = opt;
    started_ = now_seconds();
    install_signals();
    for (const PolicySpec &p : opt_.policies) {
        size_t ignored = 0;
        if (find_policy(p.id, &ignored)) {
            *err = "policy " + std::to_string(p.id) + " is given twice";
            return false;
        }
        Policy pol;
        pol.id = p.id;
        pol.evaluator = actor::load_model_evaluator(p.model, spec_, err);
        if (!pol.evaluator) return false;
        const actor::ModelInfo &m = pol.evaluator->info();
        if (policies_.empty()) {
            tables_ = m.manifest;
        } else {
            const featurize::Refusal r = featurize::check_compatible(m.manifest, tables_, true);
            if (r.refused()) {
                *err = p.model + ": not an export of policy " + std::to_string(policies_[0].id) +
                       "'s tables (every policy serves one featurization): " + r.message;
                return false;
            }
        }
        pol.version = p.has_version ? p.version : m.policy_version;
        log("policy " + std::to_string(pol.id) + " = " + p.model + " (" + m.format + ", version " +
            std::to_string(pol.version) + ")");
        policies_.push_back(std::move(pol));
    }
    listen_fd_ = inference::listen_unix(opt_.socket, 64, err);
    if (listen_fd_ < 0) return false;
    set_nonblocking(listen_fd_);
    log("listening on " + opt_.socket + " (spec " + tables_.spec_hash + ", max batch " +
        std::to_string(opt_.max_batch) + ", deadline " + std::to_string(opt_.deadline_us) + " us, " +
        std::to_string(opt_.threads) + " threads)");
    return true;
}

const Server::Policy *Server::find_policy(uint32_t id, size_t *index) const {
    for (size_t i = 0; i < policies_.size(); ++i)
        if (policies_[i].id == id) {
            *index = i;
            return &policies_[i];
        }
    return nullptr;
}

void Server::accept_clients() {
    while (true) {
        const int fd = accept(listen_fd_, nullptr, nullptr);
        if (fd < 0) {
            if (errno == EINTR) continue;
            return;
        }
        set_nonblocking(fd);
        std::unique_ptr<Client> c(new Client());
        c->fd = fd;
        c->number = ++next_client_;
        c->name = "client " + std::to_string(c->number);
        ++clients_seen_;
        clients_.push_back(std::move(c));
    }
}

bool Server::read_client(Client *c) {
    char buf[65536];
    while (true) {
        const ssize_t r = recv(c->fd, buf, sizeof buf, 0);
        if (r > 0) {
            c->inbuf.append(buf, static_cast<size_t>(r));
            continue;
        }
        if (r < 0 && errno == EINTR) continue;
        if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
        if (opt_.verbose || c->pending) log(c->name + " closed the connection");
        drop(c);
        return false;
    }
    inference::Message m;
    bool malformed = false;
    while (inference::take_message(&c->inbuf, &m, &malformed))
        if (!handle(c, m)) return false;
    if (malformed) return refuse(c, "protocol", "not an RMIP message");
    return true;
}

bool Server::handle(Client *c, const inference::Message &m) {
    if (m.type == inference::MSG_HELLO) return hello(c, m.body);
    if (m.type == inference::MSG_EVAL) return eval(c, m.body);
    if (m.type == inference::MSG_BYE) {
        if (opt_.verbose) log(c->name + " said BYE after " + std::to_string(c->requests) + " requests");
        drop(c);
        return false;
    }
    return refuse(c, "protocol", "unexpected message type " + std::to_string(m.type));
}

bool Server::hello(Client *c, const std::string &body) {
    if (c->welcomed) return refuse(c, "protocol", "a second HELLO");
    inference::Hello h;
    std::string err;
    if (!inference::parse_hello(body, &h, &err)) return refuse(c, "protocol", err);
    c->name = "client " + std::to_string(c->number) + " (" + h.client + ")";
    if (h.protocol != inference::PROTOCOL_VERSION)
        return refuse(c, "protocol",
            "protocol " + std::to_string(h.protocol) + " (this server speaks " +
                std::to_string(inference::PROTOCOL_VERSION) + ")");
    if (h.policies.empty()) return refuse(c, "policy", "no policies requested");
    for (uint32_t id : h.policies) {
        size_t index = 0;
        if (!find_policy(id, &index)) return refuse(c, "policy", "policy " + std::to_string(id) + " is not served");
        if (std::count(h.policies.begin(), h.policies.end(), id) > 1)
            return refuse(c, "policy", "policy " + std::to_string(id) + " is requested twice");
        c->policy_index.push_back(index);
    }
    const featurize::Refusal r = featurize::check_compatible(tables_, h.tables, true);
    if (r.refused()) return refuse(c, r.code, "the served models against the client's tables: " + r.message);
    const inference::Dims mine = {static_cast<uint32_t>(spec_.F), static_cast<uint32_t>(spec_.C),
        static_cast<uint32_t>(spec_.K), static_cast<uint32_t>(spec_.Q)};
    if (have_dims_ && !(h.dims == mine)) return refuse(c, "protocol", "the client's dims are not another client's");
    if (h.request_bytes == 0 || h.response_bytes == 0 || h.request_bytes % 8 || h.response_bytes % 8)
        return refuse(c, "shm", "request_bytes and response_bytes must be positive multiples of 8");
    if (!c->region.open(h.shm, h.request_bytes + h.response_bytes, &err)) return refuse(c, "shm", err);
    featurize::parse_hex16(h.tables.spec_hash, &c->spec_hash);
    if (!have_dims_) {
        spec_.F = h.dims.F;
        spec_.C = h.dims.C;
        spec_.K = h.dims.K;
        spec_.Q = h.dims.Q;
        have_dims_ = true;
    }
    c->request_bytes = h.request_bytes;
    c->response_bytes = h.response_bytes;
    c->policies = h.policies;
    c->welcomed = true;
    inference::Welcome w;
    w.server = "rmserve";
    w.max_batch = opt_.max_batch;
    for (size_t index : c->policy_index) w.policies.push_back({policies_[index].id, policies_[index].version});
    if (!inference::send_message(c->fd, inference::MSG_WELCOME, inference::welcome_json(w), &err)) {
        log(c->name + ": " + err);
        drop(c);
        return false;
    }
    log(c->name + " connected (" + std::to_string(h.policies.size()) + " policies, " +
        std::to_string(h.request_bytes >> 20) + " MiB request area)");
    return true;
}

bool Server::eval(Client *c, const std::string &body) {
    if (!c->welcomed) return refuse(c, "protocol", "EVAL before the handshake");
    if (c->pending) return refuse(c, "protocol", "a second request in flight");
    uint64_t seq = 0, length = 0;
    if (!inference::parse_seq_body(body, &seq, &length)) return refuse(c, "protocol", "a malformed EVAL");
    Queued q;
    q.client = c;
    std::string err;
    if (!inference::decode_request(c->region.data(), c->request_bytes, &q.request, &err))
        return refuse(c, "request", err);
    const inference::Request &r = q.request;
    if (r.spec_hash != c->spec_hash)
        return refuse(c, "spec_hash",
            "the request's spec_hash " + featurize::hex16(r.spec_hash) + " is not the handshake's " +
                featurize::hex16(c->spec_hash));
    const inference::Dims mine = {static_cast<uint32_t>(spec_.F), static_cast<uint32_t>(spec_.C),
        static_cast<uint32_t>(spec_.K), static_cast<uint32_t>(spec_.Q)};
    if (!(r.dims == mine)) return refuse(c, "request", "the request's dims are not the handshake's");
    if (r.policies != c->policies) return refuse(c, "request", "the request's policies are not the handshake's");
    if (r.seq != seq) return refuse(c, "request", "the request's seq is not EVAL's");
    q.arrived = now_seconds();
    c->pending = true;
    queue_.push_back(std::move(q));
    return true;
}

bool Server::refuse(Client *c, const std::string &code, const std::string &message) {
    std::string ignored;
    inference::send_message(c->fd, inference::MSG_ERROR, inference::error_json(code, message), &ignored);
    log("refused " + c->name + ": " + code + ": " + message);
    ++refused_;
    drop(c);
    return false;
}

void Server::drop(Client *c) {
    queue_.erase(
        std::remove_if(queue_.begin(), queue_.end(), [c](const Queued &q) { return q.client == c; }), queue_.end());
    close(c->fd);
    c->fd = -1;
    c->region.close();
}

bool Server::ready(double now) const {
    if (queue_.empty()) return false;
    size_t queued = 0, welcomed = 0;
    for (const Queued &q : queue_) queued += q.request.decisions.size();
    for (const auto &c : clients_) welcomed += c->fd >= 0 && c->welcomed ? 1 : 0;
    return queued >= opt_.max_batch || queue_.size() >= welcomed ||
           now >= queue_.front().arrived + 1e-6 * static_cast<double>(opt_.deadline_us);
}

double Server::next_timeout_ms(double now) const {
    if (queue_.empty()) return -1;
    const double left = queue_.front().arrived + 1e-6 * static_cast<double>(opt_.deadline_us) - now;
    return left <= 0 ? 0 : left * 1000.0;
}

void Server::answer(Queued &q, std::vector<std::vector<actor::Evaluation>> &by_policy, std::vector<size_t> &cursor,
    uint32_t batch, uint32_t compute_us) {
    Client *c = q.client;
    inference::Response r;
    r.spec_hash = q.request.spec_hash;
    r.seq = q.request.seq;
    r.batch = batch;
    r.compute_us = compute_us;
    std::vector<size_t> rows;
    for (const featurize::Tensors &t : q.request.decisions) rows.push_back(t.a());
    for (size_t index : c->policy_index) {
        inference::PolicyResult pr;
        pr.id = policies_[index].id;
        pr.version = policies_[index].version;
        for (size_t d = 0; d < q.request.decisions.size(); ++d) {
            const actor::Evaluation &e = by_policy[index][cursor[index]++];
            pr.value.push_back(e.value);
            pr.logits.insert(pr.logits.end(), e.policies[0].logits.begin(), e.policies[0].logits.end());
            pr.log_pi.insert(pr.log_pi.end(), e.policies[0].log_pi.begin(), e.policies[0].log_pi.end());
        }
        r.policies.push_back(std::move(pr));
    }
    std::string err;
    size_t length = 0;
    c->pending = false;
    if (!inference::encode_response(c->region.data() + c->request_bytes, c->response_bytes, r, rows, &length, &err)) {
        refuse(c, "request", err);
        return;
    }
    if (!inference::send_message(c->fd, inference::MSG_DONE, inference::seq_body(r.seq, length), &err)) {
        log(c->name + ": " + err);
        drop(c);
        return;
    }
    ++c->requests;
    ++requests_;
    decisions_ += q.request.decisions.size();
}

void Server::flush() {
    size_t take = 0, decisions = 0;
    while (take < queue_.size() && (take == 0 || decisions + queue_[take].request.decisions.size() <= opt_.max_batch)) {
        decisions += queue_[take].request.decisions.size();
        ++take;
    }
    std::vector<Queued> batch(std::make_move_iterator(queue_.begin()),
        std::make_move_iterator(queue_.begin() + static_cast<std::ptrdiff_t>(take)));
    queue_.erase(queue_.begin(), queue_.begin() + static_cast<std::ptrdiff_t>(take));
    std::vector<std::vector<actor::Evaluation>> by_policy(policies_.size());
    std::vector<size_t> cursor(policies_.size(), 0);
    const double t0 = now_seconds();
    for (size_t p = 0; p < policies_.size(); ++p) {
        std::vector<const featurize::Tensors *> in;
        for (const Queued &q : batch)
            if (std::find(q.client->policy_index.begin(), q.client->policy_index.end(), p) !=
                q.client->policy_index.end())
                for (const featurize::Tensors &t : q.request.decisions) in.push_back(&t);
        if (in.empty()) continue;
        std::string err;
        if (!policies_[p].evaluator->evaluate(in, &by_policy[p], &err)) {
            for (Queued &q : batch)
                if (q.client->fd >= 0)
                    refuse(q.client, "eval", "policy " + std::to_string(policies_[p].id) + ": " + err);
            return;
        }
    }
    const double dt = now_seconds() - t0;
    compute_s_ += dt;
    ++batches_;
    max_seen_ = std::max(max_seen_, decisions);
    const uint32_t us = static_cast<uint32_t>(std::min(dt * 1e6, 4294967295.0));
    if (opt_.verbose)
        log("batch of " + std::to_string(decisions) + " decisions (" + std::to_string(take) + " requests) in " +
            std::to_string(us) + " us");
    for (Queued &q : batch)
        if (q.client->fd >= 0) answer(q, by_policy, cursor, static_cast<uint32_t>(decisions), us);
    if (opt_.pause_after >= 0 && requests_ >= static_cast<size_t>(opt_.pause_after)) {
        paused_ = true;
        log("paused after " + std::to_string(requests_) + " requests (--pause-after)");
    }
}

void Server::print_stats() const {
    char buf[320];
    std::snprintf(buf, sizeof buf,
        "RMSERVE-STATS: clients=%zu requests=%zu decisions=%zu batches=%zu mean_batch=%.2f max_batch=%zu "
        "compute_s=%.3f wall_s=%.3f refused=%zu",
        clients_seen_, requests_, decisions_, batches_,
        batches_ ? static_cast<double>(decisions_) / static_cast<double>(batches_) : 0.0, max_seen_, compute_s_,
        now_seconds() - started_, refused_);
    std::fprintf(stderr, "%s\n", buf);
    std::fflush(stderr);
}

int Server::run() {
    while (true) {
        std::vector<pollfd> fds;
        fds.push_back({signal_pipe[0], POLLIN, 0});
        if (!paused_) {
            fds.push_back({listen_fd_, POLLIN, 0});
            for (const auto &c : clients_)
                if (c->fd >= 0) fds.push_back({c->fd, POLLIN, 0});
        }
        const double ms = paused_ ? -1 : next_timeout_ms(now_seconds());
        const int timeout = ms < 0 ? -1 : static_cast<int>(ms) + (ms > static_cast<int>(ms) ? 1 : 0);
        const int pr = poll(fds.data(), static_cast<nfds_t>(fds.size()), timeout);
        if (pr < 0 && errno != EINTR) {
            log(std::string("poll: ") + std::strerror(errno));
            return 1;
        }
        if (pr > 0 && (fds[0].revents & POLLIN)) break;
        if (paused_) continue;
        if (pr > 0 && (fds[1].revents & POLLIN)) accept_clients();
        for (size_t i = 2; i < fds.size(); ++i) {
            if (!fds[i].revents) continue;
            for (const auto &c : clients_)
                if (c->fd == fds[i].fd) {
                    read_client(c.get());
                    break;
                }
        }
        clients_.erase(std::remove_if(clients_.begin(), clients_.end(),
                           [](const std::unique_ptr<Client> &c) { return c->fd < 0; }),
            clients_.end());
        while (!paused_ && ready(now_seconds())) flush();
        clients_.erase(std::remove_if(clients_.begin(), clients_.end(),
                           [](const std::unique_ptr<Client> &c) { return c->fd < 0; }),
            clients_.end());
    }
    for (const auto &c : clients_)
        if (c->fd >= 0 && c->pending) {
            std::string ignored;
            inference::send_message(
                c->fd, inference::MSG_ERROR, inference::error_json("shutdown", "the server is stopping"), &ignored);
        }
    log("stopping (signal)");
    print_stats();
    return 0;
}

}  // namespace rmserve
