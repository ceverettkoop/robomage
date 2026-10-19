// rmserve: the reference inference server (plan step 5d, docs/inference_protocol.md).
//
//   rmserve --socket PATH --policy ID=MODEL [--policy ID=MODEL …] [--version ID=N …]
//           [--max-batch N] [--deadline-us N] [--threads N] [--verbose] [--pause-after N]
//
// Serves the policies (TorchScript .ts.pt or AOTInductor .pt2 exports of the network contract,
// loaded by the actor's in-process evaluator, src/actor/evaluator.h; paths relative to the working
// directory) to any number of clients at the Unix socket PATH, batching their requests across
// clients up to --max-batch decisions (default 256) or --deadline-us after the oldest (default
// 2000), and returns each requested policy's logits, log π and value with its version: --version
// ID=N, else the model's policy_version. --threads sets libtorch's intra-op threads (default 1).
// --verbose logs every batch. --pause-after N (for tests) stops serving after N responses, leaving
// the clients waiting, until the process is killed. SIGTERM or SIGINT stops it cleanly (the socket
// removed, an RMSERVE-STATS line); everything is logged to stderr. Exit 0 after a signal, 2 on bad
// arguments or a model or socket that cannot be opened.

#include <ATen/Parallel.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "server.h"

static int usage();
static bool parse_id_value(const char *arg, uint32_t *id, std::string *value);
static bool parse_count(const char *arg, long min, long *out);
static bool parse_args(int argc, char **argv, rmserve::Options *opt);

static int usage() {
    std::fprintf(stderr,
        "usage: rmserve --socket PATH --policy ID=MODEL [--policy ID=MODEL ...] [--version ID=N ...]\n"
        "               [--max-batch N] [--deadline-us N] [--threads N] [--verbose] [--pause-after N]\n");
    return 2;
}

// "ID=VALUE" with ID an unsigned 32-bit integer and VALUE non-empty.
static bool parse_id_value(const char *arg, uint32_t *id, std::string *value) {
    const char *eq = std::strchr(arg, '=');
    if (!eq || eq == arg || !eq[1]) return false;
    char *end = nullptr;
    const unsigned long long v = std::strtoull(arg, &end, 10);
    if (end != eq || arg[0] == '-' || v > 0xFFFFFFFFull) return false;
    *id = static_cast<uint32_t>(v);
    *value = eq + 1;
    return true;
}

static bool parse_count(const char *arg, long min, long *out) {
    char *end = nullptr;
    *out = std::strtol(arg, &end, 10);
    return *arg && !*end && *out >= min;
}

static bool parse_args(int argc, char **argv, rmserve::Options *opt) {
    std::map<uint32_t, uint64_t> versions;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        long n = 0;
        uint32_t id = 0;
        std::string value;
        if (a == "--verbose") {
            opt->verbose = true;
        } else if (!has_value) {
            std::fprintf(stderr, "rmserve: %s needs a value (or is unknown)\n", a.c_str());
            return false;
        } else if (a == "--socket") {
            opt->socket = argv[++i];
        } else if (a == "--policy" && parse_id_value(argv[i + 1], &id, &value)) {
            ++i;
            rmserve::PolicySpec p;
            p.id = id;
            p.model = value;
            opt->policies.push_back(p);
        } else if (a == "--version" && parse_id_value(argv[i + 1], &id, &value)) {
            char *end = nullptr;
            const unsigned long long v = std::strtoull(value.c_str(), &end, 10);
            if (*end || value[0] == '-') {
                std::fprintf(stderr, "rmserve: --version %s: not an unsigned integer\n", argv[i + 1]);
                return false;
            }
            versions[id] = v;
            ++i;
        } else if (a == "--max-batch" && parse_count(argv[i + 1], 1, &n)) {
            opt->max_batch = static_cast<size_t>(n);
            ++i;
        } else if (a == "--deadline-us" && parse_count(argv[i + 1], 0, &n)) {
            opt->deadline_us = n;
            ++i;
        } else if (a == "--threads" && parse_count(argv[i + 1], 1, &n)) {
            opt->threads = static_cast<int>(n);
            ++i;
        } else if (a == "--pause-after" && parse_count(argv[i + 1], 0, &n)) {
            opt->pause_after = n;
            ++i;
        } else {
            std::fprintf(stderr, "rmserve: bad argument %s %s\n", a.c_str(), argv[i + 1]);
            return false;
        }
    }
    if (opt->socket.empty() || opt->policies.empty()) {
        std::fprintf(stderr, "rmserve: --socket and at least one --policy are required\n");
        return false;
    }
    for (const auto &v : versions) {
        bool found = false;
        for (rmserve::PolicySpec &p : opt->policies)
            if (p.id == v.first) {
                p.has_version = found = true;
                p.version = v.second;
            }
        if (!found) {
            std::fprintf(stderr, "rmserve: --version names policy %u, which has no --policy\n", v.first);
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    rmserve::Options opt;
    if (!parse_args(argc, argv, &opt)) return usage();
    rmserve::Server server;
    std::string err;
    if (!server.start(opt, &err)) {
        std::fprintf(stderr, "rmserve: %s\n", err.c_str());
        return 2;
    }
    at::set_num_threads(opt.threads);
    return server.run();
}
