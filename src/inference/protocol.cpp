#include "protocol.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>

#include "../featurize/json.h"

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the protocol's arrays are copied as host bytes");

namespace {

const char MSG_MAGIC[4] = {'R', 'M', 'I', 'P'};
const char REQUEST_MAGIC[4] = {'R', 'M', 'R', 'Q'};
const char RESPONSE_MAGIC[4] = {'R', 'M', 'R', 'S'};
const size_t MSG_HEADER = 12;
const size_t REQUEST_HEADER = 64;
const size_t RESPONSE_HEADER = 56;
// The largest width a request may declare (a bound for the size arithmetic, far above any spec's).
const uint32_t MAX_DIM = 1u << 16;

// Where each of a request's arrays starts (bytes from the request's start) and where it ends.
struct RequestLayout {
        size_t policies = 0, n = 0, r = 0, a = 0;
        size_t tok_type = 0, tok_num = 0, tok_cat = 0, tok_card = 0, tok_ability = 0, tok_mask = 0, rel = 0,
               act_tok = 0, act_mask = 0, end = 0;
};

// Where each policy's arrays start in a response, and its end.
struct ResponseLayout {
        std::vector<size_t> value, logits, log_pi;
        size_t end = 0;
};

}  // namespace

static size_t align8(size_t x);
static double now_seconds();
static RequestLayout request_layout(
    const inference::Dims &dims, size_t policies, size_t d, size_t n, size_t r, size_t a);
static ResponseLayout response_layout(size_t decisions, size_t policies, size_t menu_rows);
static void put_u32(uint8_t *at, uint32_t v);
static void put_u64(uint8_t *at, uint64_t v);
static uint32_t get_u32(const uint8_t *at);
static uint64_t get_u64(const uint8_t *at);
template <typename T>
static void put_array(uint8_t *dst, size_t at, const std::vector<T> &v);
template <typename T>
static void get_array(const uint8_t *src, size_t at, size_t count, std::vector<T> *out);
static bool json_u64(const featurize::Json &j, const char *key, uint64_t *out);
static bool json_string(const featurize::Json &j, const char *key, std::string *out);
static bool json_ids(const featurize::Json &j, const char *key, std::vector<uint32_t> *out);
static inference::RecvStatus read_exact(int fd, uint8_t *buf, size_t n, double deadline, std::string *err);
static bool fill_unix_address(const std::string &path, sockaddr_un *addr, std::string *err);
static bool check_index(const std::vector<int64_t> &v, size_t stride, size_t first, size_t count, size_t n);

static size_t align8(size_t x) {
    return (x + 7) & ~static_cast<size_t>(7);
}

static double now_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// docs/inference_protocol.md §4: the header, the policy ids, n, r, a, then the tensors in RMTB
// order, each array at the next multiple of 8.
static RequestLayout request_layout(
    const inference::Dims &dims, size_t policies, size_t d, size_t n, size_t r, size_t a) {
    RequestLayout l;
    size_t at = REQUEST_HEADER;
    const auto next = [&at](size_t bytes) {
        const size_t start = align8(at);
        at = start + bytes;
        return start;
    };
    l.policies = next(4 * policies);
    l.n = next(8 * d);
    l.r = next(8 * d);
    l.a = next(8 * d);
    l.tok_type = next(8 * n);
    l.tok_num = next(4 * n * dims.F);
    l.tok_cat = next(8 * n * dims.C);
    l.tok_card = next(8 * n * dims.K);
    l.tok_ability = next(8 * n * dims.Q);
    l.tok_mask = next(n);
    l.rel = next(8 * 3 * r);
    l.act_tok = next(8 * a);
    l.act_mask = next(a);
    l.end = at;
    return l;
}

static ResponseLayout response_layout(size_t decisions, size_t policies, size_t menu_rows) {
    ResponseLayout l;
    size_t at = RESPONSE_HEADER + 16 * policies;
    for (size_t p = 0; p < policies; ++p) {
        l.value.push_back(align8(at));
        at = l.value.back() + 4 * decisions;
        l.logits.push_back(align8(at));
        at = l.logits.back() + 4 * menu_rows;
        l.log_pi.push_back(align8(at));
        at = l.log_pi.back() + 4 * menu_rows;
    }
    l.end = at;
    return l;
}

static void put_u32(uint8_t *at, uint32_t v) {
    std::memcpy(at, &v, 4);
}

static void put_u64(uint8_t *at, uint64_t v) {
    std::memcpy(at, &v, 8);
}

static uint32_t get_u32(const uint8_t *at) {
    uint32_t v;
    std::memcpy(&v, at, 4);
    return v;
}

static uint64_t get_u64(const uint8_t *at) {
    uint64_t v;
    std::memcpy(&v, at, 8);
    return v;
}

template <typename T>
static void put_array(uint8_t *dst, size_t at, const std::vector<T> &v) {
    if (!v.empty()) std::memcpy(dst + at, v.data(), v.size() * sizeof(T));
}

template <typename T>
static void get_array(const uint8_t *src, size_t at, size_t count, std::vector<T> *out) {
    out->resize(count);
    if (count) std::memcpy(out->data(), src + at, count * sizeof(T));
}

static bool json_u64(const featurize::Json &j, const char *key, uint64_t *out) {
    const featurize::Json *v = j.get(key);
    if (!v || !v->is_number() || !v->is_integer() || v->as_number() < 0) return false;
    *out = static_cast<uint64_t>(v->as_int());
    return true;
}

static bool json_string(const featurize::Json &j, const char *key, std::string *out) {
    const featurize::Json *v = j.get(key);
    if (!v || !v->is_string()) return false;
    *out = v->as_string();
    return true;
}

static bool json_ids(const featurize::Json &j, const char *key, std::vector<uint32_t> *out) {
    const featurize::Json *v = j.get(key);
    if (!v || !v->is_array()) return false;
    out->clear();
    for (const featurize::Json &e : v->items()) {
        if (!e.is_number() || !e.is_integer() || e.as_number() < 0 || e.as_number() > 4294967295.0) return false;
        out->push_back(static_cast<uint32_t>(e.as_int()));
    }
    return true;
}

static inference::RecvStatus read_exact(int fd, uint8_t *buf, size_t n, double deadline, std::string *err) {
    size_t got = 0;
    while (got < n) {
        int ms = -1;
        if (deadline > 0) {
            const double left = deadline - now_seconds();
            if (left <= 0) return inference::RecvStatus::TIMEOUT;
            ms = static_cast<int>(std::min(left * 1000.0, 2.0e9)) + 1;
        }
        pollfd p = {fd, POLLIN, 0};
        const int pr = poll(&p, 1, ms);
        if (pr < 0 && errno == EINTR) continue;
        if (pr < 0) {
            *err = std::string("poll: ") + std::strerror(errno);
            return inference::RecvStatus::FAILED;
        }
        if (pr == 0) continue;
        const ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r > 0) {
            got += static_cast<size_t>(r);
            continue;
        }
        if (r == 0) return inference::RecvStatus::CLOSED;
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
        if (errno == ECONNRESET || errno == EPIPE) return inference::RecvStatus::CLOSED;
        *err = std::string("recv: ") + std::strerror(errno);
        return inference::RecvStatus::FAILED;
    }
    return inference::RecvStatus::OK;
}

static bool fill_unix_address(const std::string &path, sockaddr_un *addr, std::string *err) {
    std::memset(addr, 0, sizeof *addr);
    addr->sun_family = AF_UNIX;
    if (path.empty() || path.size() >= sizeof addr->sun_path) {
        *err = "the socket path '" + path + "' is empty or longer than " + std::to_string(sizeof addr->sun_path - 1) +
               " bytes";
        return false;
    }
    std::memcpy(addr->sun_path, path.c_str(), path.size() + 1);
    return true;
}

// Every `stride`th value of v from v[first], `count` of them, lies in [0, n).
static bool check_index(const std::vector<int64_t> &v, size_t stride, size_t first, size_t count, size_t n) {
    for (size_t k = 0; k < count; ++k) {
        const int64_t x = v[first + k * stride];
        if (x < 0 || static_cast<uint64_t>(x) >= n) return false;
    }
    return true;
}

namespace inference {

bool send_message(int fd, uint16_t type, const std::string &body, std::string *err) {
    std::string bytes(MSG_HEADER, '\0');
    std::memcpy(&bytes[0], MSG_MAGIC, 4);
    const uint16_t reserved = 0;
    std::memcpy(&bytes[4], &type, 2);
    std::memcpy(&bytes[6], &reserved, 2);
    const uint32_t len = static_cast<uint32_t>(body.size());
    std::memcpy(&bytes[8], &len, 4);
    bytes += body;
    size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t w = send(fd, bytes.data() + done, bytes.size() - done, MSG_NOSIGNAL);
        if (w > 0) {
            done += static_cast<size_t>(w);
            continue;
        }
        if (w < 0 && errno == EINTR) continue;
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd p = {fd, POLLOUT, 0};
            if (poll(&p, 1, 5000) <= 0) {
                *err = "the peer is not reading";
                return false;
            }
            continue;
        }
        *err = std::string("send: ") + (w < 0 ? std::strerror(errno) : "nothing written");
        return false;
    }
    return true;
}

RecvStatus recv_message(int fd, double timeout_s, Message *out, std::string *err) {
    const double deadline = timeout_s > 0 ? now_seconds() + timeout_s : 0;
    uint8_t head[MSG_HEADER];
    RecvStatus s = read_exact(fd, head, MSG_HEADER, deadline, err);
    if (s != RecvStatus::OK) return s;
    const uint32_t len = get_u32(head + 8);
    if (std::memcmp(head, MSG_MAGIC, 4) != 0 || len > MAX_BODY) {
        *err = "not an RMIP message header";
        return RecvStatus::MALFORMED;
    }
    std::memcpy(&out->type, head + 4, 2);
    out->body.assign(len, '\0');
    if (len == 0) return RecvStatus::OK;
    return read_exact(fd, reinterpret_cast<uint8_t *>(&out->body[0]), len, deadline, err);
}

bool take_message(std::string *buffer, Message *out, bool *malformed) {
    *malformed = false;
    if (buffer->size() < MSG_HEADER) return false;
    const uint8_t *head = reinterpret_cast<const uint8_t *>(buffer->data());
    const uint32_t len = get_u32(head + 8);
    if (std::memcmp(head, MSG_MAGIC, 4) != 0 || len > MAX_BODY) {
        *malformed = true;
        return false;
    }
    if (buffer->size() < MSG_HEADER + len) return false;
    std::memcpy(&out->type, head + 4, 2);
    out->body = buffer->substr(MSG_HEADER, len);
    buffer->erase(0, MSG_HEADER + len);
    return true;
}

std::string hello_json(const Hello &h) {
    std::string ids;
    for (uint32_t id : h.policies) ids += (ids.empty() ? "" : ",") + std::to_string(id);
    return "{\"protocol\":" + std::to_string(h.protocol) + ",\"client\":" + featurize::json_quote(h.client) +
           ",\"spec_hash\":" + featurize::json_quote(h.tables.spec_hash) +
           ",\"vocab_hash\":" + featurize::json_quote(h.tables.vocab_hash) +
           ",\"text_model\":" + featurize::json_quote(h.tables.text_model) +
           ",\"layout_hash\":" + featurize::json_quote(h.tables.layout_hash) +
           ",\"tables_digest\":" + featurize::json_quote(h.tables.tables_digest) +
           ",\"vocab\":" + featurize::vocab_json(h.tables.vocab) + ",\"dims\":{\"F\":" + std::to_string(h.dims.F) +
           ",\"C\":" + std::to_string(h.dims.C) + ",\"K\":" + std::to_string(h.dims.K) +
           ",\"Q\":" + std::to_string(h.dims.Q) + "},\"shm\":" + featurize::json_quote(h.shm) +
           ",\"request_bytes\":" + std::to_string(h.request_bytes) +
           ",\"response_bytes\":" + std::to_string(h.response_bytes) + ",\"policies\":[" + ids + "]}";
}

bool parse_hello(const std::string &body, Hello *out, std::string *err) {
    featurize::Json j;
    if (!featurize::Json::parse(body, &j, err) || !j.is_object()) {
        if (err->empty()) *err = "HELLO is not a JSON object";
        return false;
    }
    uint64_t protocol = 0, f = 0, c = 0, k = 0, q = 0;
    const featurize::Json *dims = j.get("dims");
    if (!json_u64(j, "protocol", &protocol)) {
        *err = "HELLO has no protocol";
        return false;
    }
    out->protocol = static_cast<uint32_t>(std::min<uint64_t>(protocol, 0xFFFFFFFFu));
    const featurize::Json *vocab = j.get("vocab");
    std::string why;
    if (!json_string(j, "client", &out->client) || !json_string(j, "spec_hash", &out->tables.spec_hash) ||
        !json_string(j, "vocab_hash", &out->tables.vocab_hash) ||
        !json_string(j, "text_model", &out->tables.text_model) ||
        !json_string(j, "layout_hash", &out->tables.layout_hash) ||
        !json_string(j, "tables_digest", &out->tables.tables_digest) || !vocab ||
        !featurize::vocab_from_json(*vocab, &out->tables.vocab, &why) || !json_string(j, "shm", &out->shm) ||
        !json_u64(j, "request_bytes", &out->request_bytes) || !json_u64(j, "response_bytes", &out->response_bytes) ||
        !json_ids(j, "policies", &out->policies) || !dims || !dims->is_object() || !json_u64(*dims, "F", &f) ||
        !json_u64(*dims, "C", &c) || !json_u64(*dims, "K", &k) || !json_u64(*dims, "Q", &q)) {
        *err = "HELLO lacks a key or has one of the wrong type (docs/inference_protocol.md §3)";
        return false;
    }
    if (f > MAX_DIM || c > MAX_DIM || k > MAX_DIM || q > MAX_DIM) {
        *err = "HELLO's dims are out of range";
        return false;
    }
    out->dims = {
        static_cast<uint32_t>(f), static_cast<uint32_t>(c), static_cast<uint32_t>(k), static_cast<uint32_t>(q)};
    return true;
}

std::string welcome_json(const Welcome &w) {
    std::string pol;
    for (const WelcomePolicy &p : w.policies)
        pol += std::string(pol.empty() ? "" : ",") + "{\"id\":" + std::to_string(p.id) +
               ",\"version\":" + std::to_string(p.version) + "}";
    return "{\"protocol\":" + std::to_string(w.protocol) + ",\"server\":" + featurize::json_quote(w.server) +
           ",\"max_batch\":" + std::to_string(w.max_batch) + ",\"policies\":[" + pol + "]}";
}

bool parse_welcome(const std::string &body, Welcome *out, std::string *err) {
    featurize::Json j;
    if (!featurize::Json::parse(body, &j, err) || !j.is_object()) {
        if (err->empty()) *err = "WELCOME is not a JSON object";
        return false;
    }
    uint64_t protocol = 0;
    const featurize::Json *pol = j.get("policies");
    if (!json_u64(j, "protocol", &protocol) || !json_string(j, "server", &out->server) ||
        !json_u64(j, "max_batch", &out->max_batch) || !pol || !pol->is_array()) {
        *err = "WELCOME lacks a key or has one of the wrong type";
        return false;
    }
    out->protocol = static_cast<uint32_t>(std::min<uint64_t>(protocol, 0xFFFFFFFFu));
    out->policies.clear();
    for (const featurize::Json &p : pol->items()) {
        uint64_t id = 0;
        WelcomePolicy w;
        if (!p.is_object() || !json_u64(p, "id", &id) || !json_u64(p, "version", &w.version) || id > 0xFFFFFFFFu) {
            *err = "WELCOME's policies are not {id, version} objects";
            return false;
        }
        w.id = static_cast<uint32_t>(id);
        out->policies.push_back(w);
    }
    return true;
}

std::string error_json(const std::string &code, const std::string &message) {
    return "{\"code\":" + featurize::json_quote(code) + ",\"message\":" + featurize::json_quote(message) + "}";
}

void parse_error(const std::string &body, std::string *code, std::string *message) {
    featurize::Json j;
    std::string err;
    code->clear();
    *message = body;
    if (!featurize::Json::parse(body, &j, &err) || !j.is_object()) return;
    json_string(j, "code", code);
    json_string(j, "message", message);
}

std::string seq_body(uint64_t seq, uint64_t length) {
    std::string b(16, '\0');
    std::memcpy(&b[0], &seq, 8);
    std::memcpy(&b[8], &length, 8);
    return b;
}

bool parse_seq_body(const std::string &body, uint64_t *seq, uint64_t *length) {
    if (body.size() != 16) return false;
    std::memcpy(seq, body.data(), 8);
    std::memcpy(length, body.data() + 8, 8);
    return true;
}

int connect_unix(const std::string &path, std::string *err) {
    sockaddr_un addr;
    if (!fill_unix_address(path, &addr, err)) return -1;
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        *err = std::string("socket: ") + std::strerror(errno);
        return -1;
    }
    while (connect(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof addr) != 0) {
        if (errno == EINTR) continue;
        *err = "cannot connect: " + std::string(std::strerror(errno));
        ::close(fd);
        return -1;
    }
    return fd;
}

int listen_unix(const std::string &path, int backlog, std::string *err) {
    struct stat st;
    if (lstat(path.c_str(), &st) == 0) {
        if (!S_ISSOCK(st.st_mode)) {
            *err = path + " exists and is not a socket";
            return -1;
        }
        ::unlink(path.c_str());
    }
    const std::string tmp = path + ".tmp" + std::to_string(getpid());
    sockaddr_un addr;
    if (!fill_unix_address(tmp, &addr, err)) return -1;
    ::unlink(tmp.c_str());
    const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        *err = std::string("socket: ") + std::strerror(errno);
        return -1;
    }
    if (bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof addr) != 0 || listen(fd, backlog) != 0 ||
        rename(tmp.c_str(), path.c_str()) != 0) {
        *err = path + ": " + std::strerror(errno);
        ::close(fd);
        ::unlink(tmp.c_str());
        return -1;
    }
    return fd;
}

SharedRegion::~SharedRegion() {
    close();
}

bool SharedRegion::create(const std::string &prefix, size_t size, std::string *err) {
    close();
    static std::atomic<unsigned> counter(0);
    const unsigned long long t =
        static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count() & 0xFFFFFFFFFFull);
    name_ = prefix + "-" + std::to_string(getpid()) + "-" + std::to_string(counter++) + "-" + std::to_string(t);
    const int fd = shm_open(name_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) {
        *err = "shm_open " + name_ + ": " + std::strerror(errno);
        return false;
    }
    linked_ = true;
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        *err = "ftruncate " + name_ + ": " + std::strerror(errno);
        ::close(fd);
        close();
        return false;
    }
    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        *err = "mmap " + name_ + ": " + std::strerror(errno);
        close();
        return false;
    }
    data_ = static_cast<uint8_t *>(p);
    size_ = size;
    return true;
}

bool SharedRegion::open(const std::string &name, size_t size, std::string *err) {
    close();
    const int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) {
        *err = "shm_open " + name + ": " + std::strerror(errno);
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || static_cast<size_t>(st.st_size) < size || size == 0) {
        *err = "the region " + name + " is smaller than its areas (" + std::to_string(size) + " bytes)";
        ::close(fd);
        return false;
    }
    void *p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        *err = "mmap " + name + ": " + std::strerror(errno);
        return false;
    }
    name_ = name;
    data_ = static_cast<uint8_t *>(p);
    size_ = size;
    return true;
}

void SharedRegion::unlink() {
    if (linked_) shm_unlink(name_.c_str());
    linked_ = false;
}

void SharedRegion::close() {
    unlink();
    if (data_) munmap(data_, size_);
    data_ = nullptr;
    size_ = 0;
}

size_t request_size(const Dims &dims, size_t policies, const std::vector<const featurize::Tensors *> &batch) {
    size_t n = 0, r = 0, a = 0;
    for (const featurize::Tensors *t : batch) {
        n += t->n();
        r += t->r();
        a += t->a();
    }
    return request_layout(dims, policies, batch.size(), n, r, a).end;
}

bool encode_request(uint8_t *dst, size_t cap, uint64_t spec_hash, uint64_t seq, const Dims &dims,
    const std::vector<uint32_t> &policies, const std::vector<const featurize::Tensors *> &batch, size_t *length,
    std::string *err) {
    size_t n = 0, r = 0, a = 0;
    for (const featurize::Tensors *t : batch) {
        if (t->tok_num.size() != t->n() * dims.F || t->tok_cat.size() != t->n() * dims.C ||
            t->tok_card.size() != t->n() * dims.K || t->tok_ability.size() != t->n() * dims.Q ||
            t->tok_mask.size() != t->n() || t->act_mask.size() != t->a() || t->rel.size() != 3 * t->r()) {
            *err = "a decision's tensors are not of the spec's widths";
            return false;
        }
        n += t->n();
        r += t->r();
        a += t->a();
    }
    const RequestLayout l = request_layout(dims, policies.size(), batch.size(), n, r, a);
    if (l.end > cap) {
        *err = "the request takes " + std::to_string(l.end) + " bytes, more than the request area's " +
               std::to_string(cap);
        return false;
    }
    std::memset(dst, 0, l.end);
    std::memcpy(dst, REQUEST_MAGIC, 4);
    put_u32(dst + 4, PROTOCOL_VERSION);
    put_u64(dst + 8, spec_hash);
    put_u64(dst + 16, seq);
    put_u32(dst + 24, static_cast<uint32_t>(batch.size()));
    put_u32(dst + 28, static_cast<uint32_t>(policies.size()));
    put_u32(dst + 32, dims.F);
    put_u32(dst + 36, dims.C);
    put_u32(dst + 40, dims.K);
    put_u32(dst + 44, dims.Q);
    put_u64(dst + 48, l.end);
    put_array(dst, l.policies, policies);
    RequestLayout at = l;
    for (size_t d = 0; d < batch.size(); ++d) {
        const featurize::Tensors &t = *batch[d];
        put_u64(dst + l.n + 8 * d, t.n());
        put_u64(dst + l.r + 8 * d, t.r());
        put_u64(dst + l.a + 8 * d, t.a());
        put_array(dst, at.tok_type, t.tok_type);
        at.tok_type += 8 * t.n();
        put_array(dst, at.tok_num, t.tok_num);
        at.tok_num += 4 * t.tok_num.size();
        put_array(dst, at.tok_cat, t.tok_cat);
        at.tok_cat += 8 * t.tok_cat.size();
        put_array(dst, at.tok_card, t.tok_card);
        at.tok_card += 8 * t.tok_card.size();
        put_array(dst, at.tok_ability, t.tok_ability);
        at.tok_ability += 8 * t.tok_ability.size();
        put_array(dst, at.tok_mask, t.tok_mask);
        at.tok_mask += t.tok_mask.size();
        put_array(dst, at.rel, t.rel);
        at.rel += 8 * t.rel.size();
        put_array(dst, at.act_tok, t.act_tok);
        at.act_tok += 8 * t.act_tok.size();
        put_array(dst, at.act_mask, t.act_mask);
        at.act_mask += t.act_mask.size();
    }
    *length = l.end;
    return true;
}

bool decode_request(const uint8_t *src, size_t cap, Request *out, std::string *err) {
    if (cap < REQUEST_HEADER || std::memcmp(src, REQUEST_MAGIC, 4) != 0) {
        *err = "the request area holds no request (bad magic)";
        return false;
    }
    if (get_u32(src + 4) != PROTOCOL_VERSION) {
        *err = "the request's protocol is " + std::to_string(get_u32(src + 4));
        return false;
    }
    out->spec_hash = get_u64(src + 8);
    out->seq = get_u64(src + 16);
    const size_t d = get_u32(src + 24), p = get_u32(src + 28);
    out->dims = {get_u32(src + 32), get_u32(src + 36), get_u32(src + 40), get_u32(src + 44)};
    const uint64_t length = get_u64(src + 48);
    const Dims &dims = out->dims;
    if (d == 0 || p == 0 || length > cap || 24 * d + 4 * p > cap || dims.F > MAX_DIM || dims.C > MAX_DIM ||
        dims.K > MAX_DIM || dims.Q > MAX_DIM) {
        *err = "the request's header is out of bounds (D " + std::to_string(d) + ", P " + std::to_string(p) +
               ", length " + std::to_string(length) + ", area " + std::to_string(cap) + ")";
        return false;
    }
    const RequestLayout head = request_layout(dims, p, d, 0, 0, 0);
    if (head.tok_type > length) {
        *err = "the request ends inside its sizes";
        return false;
    }
    get_array(src, head.policies, p, &out->policies);
    std::vector<int64_t> ns, rs, as;
    get_array(src, head.n, d, &ns);
    get_array(src, head.r, d, &rs);
    get_array(src, head.a, d, &as);
    size_t n = 0, r = 0, a = 0;
    for (size_t i = 0; i < d; ++i) {
        if (ns[i] < 1 || rs[i] < 0 || as[i] < 1 || static_cast<uint64_t>(ns[i]) > cap ||
            static_cast<uint64_t>(rs[i]) > cap || static_cast<uint64_t>(as[i]) > cap) {
            *err = "decision " + std::to_string(i) + "'s sizes are out of range";
            return false;
        }
        n += static_cast<size_t>(ns[i]);
        r += static_cast<size_t>(rs[i]);
        a += static_cast<size_t>(as[i]);
        if (n > cap || r > cap || a > cap) {
            *err = "the request's sizes exceed the request area";
            return false;
        }
    }
    const RequestLayout l = request_layout(dims, p, d, n, r, a);
    if (l.end != length) {
        *err = "the request's length " + std::to_string(length) + " is not its layout's " + std::to_string(l.end);
        return false;
    }
    out->decisions.assign(d, featurize::Tensors());
    RequestLayout at = l;
    for (size_t i = 0; i < d; ++i) {
        featurize::Tensors &t = out->decisions[i];
        const size_t tn = static_cast<size_t>(ns[i]), tr = static_cast<size_t>(rs[i]), ta = static_cast<size_t>(as[i]);
        get_array(src, at.tok_type, tn, &t.tok_type);
        at.tok_type += 8 * tn;
        get_array(src, at.tok_num, tn * dims.F, &t.tok_num);
        at.tok_num += 4 * tn * dims.F;
        get_array(src, at.tok_cat, tn * dims.C, &t.tok_cat);
        at.tok_cat += 8 * tn * dims.C;
        get_array(src, at.tok_card, tn * dims.K, &t.tok_card);
        at.tok_card += 8 * tn * dims.K;
        get_array(src, at.tok_ability, tn * dims.Q, &t.tok_ability);
        at.tok_ability += 8 * tn * dims.Q;
        get_array(src, at.tok_mask, tn, &t.tok_mask);
        at.tok_mask += tn;
        get_array(src, at.rel, 3 * tr, &t.rel);
        at.rel += 8 * 3 * tr;
        get_array(src, at.act_tok, ta, &t.act_tok);
        at.act_tok += 8 * ta;
        get_array(src, at.act_mask, ta, &t.act_mask);
        at.act_mask += ta;
        if (!check_index(t.act_tok, 1, 0, ta, tn) || !check_index(t.rel, 3, 0, tr, tn) ||
            !check_index(t.rel, 3, 1, tr, tn)) {
            *err = "decision " + std::to_string(i) + " has an act_tok or rel index outside its tokens";
            return false;
        }
        for (size_t k = 0; k < tr; ++k)
            if (t.rel[3 * k + 2] < 0) {
                *err = "decision " + std::to_string(i) + " has a negative relation type";
                return false;
            }
    }
    return true;
}

size_t response_size(size_t decisions, size_t policies, size_t menu_rows) {
    return response_layout(decisions, policies, menu_rows).end;
}

bool encode_response(uint8_t *dst, size_t cap, const Response &r, const std::vector<size_t> &menu_rows, size_t *length,
    std::string *err) {
    size_t a = 0;
    for (size_t m : menu_rows) a += m;
    const size_t d = menu_rows.size(), p = r.policies.size();
    const ResponseLayout l = response_layout(d, p, a);
    if (l.end > cap) {
        *err = "the response takes " + std::to_string(l.end) + " bytes, more than the response area's " +
               std::to_string(cap);
        return false;
    }
    for (const PolicyResult &pr : r.policies)
        if (pr.value.size() != d || pr.logits.size() != a || pr.log_pi.size() != a) {
            *err = "a policy's results are not the request's decisions";
            return false;
        }
    std::memset(dst, 0, l.end);
    std::memcpy(dst, RESPONSE_MAGIC, 4);
    put_u32(dst + 4, PROTOCOL_VERSION);
    put_u64(dst + 8, r.spec_hash);
    put_u64(dst + 16, r.seq);
    put_u32(dst + 24, static_cast<uint32_t>(d));
    put_u32(dst + 28, static_cast<uint32_t>(p));
    put_u32(dst + 32, r.batch);
    put_u32(dst + 36, r.compute_us);
    put_u64(dst + 40, l.end);
    for (size_t i = 0; i < p; ++i) {
        put_u32(dst + RESPONSE_HEADER + 16 * i, r.policies[i].id);
        put_u64(dst + RESPONSE_HEADER + 16 * i + 8, r.policies[i].version);
        put_array(dst, l.value[i], r.policies[i].value);
        put_array(dst, l.logits[i], r.policies[i].logits);
        put_array(dst, l.log_pi[i], r.policies[i].log_pi);
    }
    *length = l.end;
    return true;
}

bool decode_response(
    const uint8_t *src, size_t cap, const std::vector<size_t> &menu_rows, Response *out, std::string *err) {
    if (cap < RESPONSE_HEADER || std::memcmp(src, RESPONSE_MAGIC, 4) != 0 || get_u32(src + 4) != PROTOCOL_VERSION) {
        *err = "the response area holds no response of protocol " + std::to_string(PROTOCOL_VERSION);
        return false;
    }
    size_t a = 0;
    for (size_t m : menu_rows) a += m;
    const size_t d = get_u32(src + 24), p = get_u32(src + 28);
    if (d != menu_rows.size() || p == 0 || 16 * p + RESPONSE_HEADER > cap) {
        *err = "the response is for " + std::to_string(d) + " decisions and " + std::to_string(p) +
               " policies, the request had " + std::to_string(menu_rows.size()) + " decisions";
        return false;
    }
    const ResponseLayout l = response_layout(d, p, a);
    if (get_u64(src + 40) != l.end || l.end > cap) {
        *err = "the response's length is not its layout's";
        return false;
    }
    out->spec_hash = get_u64(src + 8);
    out->seq = get_u64(src + 16);
    out->batch = get_u32(src + 32);
    out->compute_us = get_u32(src + 36);
    out->policies.assign(p, PolicyResult());
    for (size_t i = 0; i < p; ++i) {
        PolicyResult &pr = out->policies[i];
        pr.id = get_u32(src + RESPONSE_HEADER + 16 * i);
        pr.version = get_u64(src + RESPONSE_HEADER + 16 * i + 8);
        get_array(src, l.value[i], d, &pr.value);
        get_array(src, l.logits[i], a, &pr.logits);
        get_array(src, l.log_pi[i], a, &pr.log_pi);
    }
    return true;
}

void log_softmax(const float *logits, size_t n, float *out) {
    if (n == 0) return;
    double mx = logits[0];
    for (size_t i = 0; i < n; ++i) mx = std::max(mx, static_cast<double>(logits[i]));
    double sum = 0;
    for (size_t i = 0; i < n; ++i) sum += std::exp(static_cast<double>(logits[i]) - mx);
    const double lse = mx + std::log(sum);
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(static_cast<double>(logits[i]) - lse);
}

}  // namespace inference
