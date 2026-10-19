#include "nn_agent.h"

#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "../agents/policy_projection.h"
#include "../state_format/cards_file.h"
#include "../str_util.h"
#include "remote_evaluator.h"

namespace {

// argmax's tie: rows whose logits are within this (absolute, and relative to the largest) of the
// largest are tied, and the first of them is taken, so the float drift between two exports of one
// model cannot reorder rows that are equal in exact arithmetic (two identical cards in hand).
const double ARGMAX_TIE = 1e-5;

}  // namespace

static std::vector<actor::NnAgent *> &finishing();
static void finish_all();
static double seconds_since(std::chrono::steady_clock::time_point t0);
static bool parse_policies(const std::string &value, std::vector<uint32_t> *out);
static std::unique_ptr<actor::Evaluator> make_evaluator(
    const actor::NnOptions &opt, const featurize::Featurization &f, std::string *err);
static std::unique_ptr<agents::Agent> make_nn_agent(const std::string &args, agents::PyRandom *rng, std::string *err);

// The agents whose dump file and stats line are written at process exit (the in-process agents
// live until then).
static std::vector<actor::NnAgent *> &finishing() {
    static std::vector<actor::NnAgent *> agents;
    return agents;
}

static void finish_all() {
    for (actor::NnAgent *a : finishing()) a->finish();
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// A comma-separated list of distinct policy ids; false on a bad or repeated entry.
static bool parse_policies(const std::string &value, std::vector<uint32_t> *out) {
    out->clear();
    for (const std::string &id : split(value, ',')) {
        char *end = nullptr;
        const unsigned long long v = std::strtoull(id.c_str(), &end, 10);
        if (id.empty() || *end != '\0' || id[0] == '-' || v > 0xFFFFFFFFull) return false;
        const uint32_t p = static_cast<uint32_t>(v);
        if (std::find(out->begin(), out->end(), p) != out->end()) return false;
        out->push_back(p);
    }
    return !out->empty();
}

// The in-process model, checked against the featurizer, or the server's connection.
static std::unique_ptr<actor::Evaluator> make_evaluator(
    const actor::NnOptions &opt, const featurize::Featurization &f, std::string *err) {
    if (!opt.remote.empty()) {
        actor::RemoteOptions r;
        r.endpoint = opt.remote;
        r.policies = opt.policies;
        r.timeout_s = opt.timeout;
        r.request_bytes = opt.shm_mib << 20;
        return actor::connect_remote_evaluator(r, f, err);
    }
    std::unique_ptr<actor::Evaluator> e = actor::load_model_evaluator(opt.model, f.spec, err);
    if (!e || !actor::check_model(e->info(), f, err)) return nullptr;
    return e;
}

// The nn: factory: the options, the featurizer's inputs from the engine, the evaluator.
static std::unique_ptr<agents::Agent> make_nn_agent(const std::string &args, agents::PyRandom *rng, std::string *err) {
    actor::NnOptions opt;
    if (!actor::NnAgent::parse_options(args, &opt, err)) return nullptr;
    for (const actor::NnAgent *a : finishing())
        for (const std::string &file : {opt.dump, opt.evals})
            if (!file.empty() && (a->options().dump == file || a->options().evals == file)) {
                *err = "nn: both seats write " + file;
                return nullptr;
            }
    std::unique_ptr<featurize::Featurization> f(new featurize::Featurization());
    if (!load_engine_featurization(f.get(), err)) {
        *err = "nn: the featurizer's inputs: " + *err;
        return nullptr;
    }
    const actor::ModelStamp stamp = opt.remote.empty() ? actor::model_stamp(opt.model) : actor::ModelStamp();
    std::unique_ptr<actor::Evaluator> e = make_evaluator(opt, *f, err);
    if (!e) {
        *err = "nn: " + *err;
        return nullptr;
    }
    return std::unique_ptr<agents::Agent>(new actor::NnAgent(std::move(opt), rng, std::move(f), std::move(e), stamp));
}

namespace actor {

ModelStamp model_stamp(const std::string &path) {
    ModelStamp m;
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return m;
    m.ok = true;
    m.mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000 + static_cast<int64_t>(st.st_mtim.tv_nsec);
    m.size = static_cast<int64_t>(st.st_size);
    m.ino = static_cast<uint64_t>(st.st_ino);
    m.dev = static_cast<uint64_t>(st.st_dev);
    return m;
}

bool NnAgent::begin_match(std::string *err) {
    if (!opt_.remote.empty()) return true;
    const ModelStamp now = model_stamp(opt_.model);
    if (now == stamp_) return true;
    std::unique_ptr<Evaluator> e = make_evaluator(opt_, *featurization_, err);
    if (!e) {
        *err = "nn: reloading " + opt_.model + ": " + *err;
        return false;
    }
    evaluator_ = std::move(e);
    stamp_ = now;
    ++reloads_;
    return true;
}

bool NnAgent::parse_options(const std::string &args, NnOptions *out, std::string *err) {
    const std::vector<std::string> parts = split(args, ':');
    if (parts.empty() || parts[0].empty()) {
        *err = "nn: expects nn:<model>[:<key>=<value>...] or nn:remote:<endpoint>[:...], got 'nn:" + args + "'";
        return false;
    }
    if (ascii_lower(parts[0]) != "remote") {
        out->model = parts[0];
        return parse_keys(parts, 1, out, err);
    }
    if (parts.size() < 2 || parts[1].empty()) {
        *err = "nn:remote: expects nn:remote:<socket path>[:<key>=<value>...], got 'nn:" + args + "'";
        return false;
    }
    out->remote = parts[1];
    return parse_keys(parts, 2, out, err);
}

bool NnAgent::parse_keys(const std::vector<std::string> &parts, size_t first, NnOptions *out, std::string *err) {
    const bool remote = !out->remote.empty();
    for (size_t i = first; i < parts.size(); ++i) {
        const std::string &p = parts[i];
        const size_t eq = p.find('=');
        const std::string key = ascii_lower(p.substr(0, eq)), value = eq == std::string::npos ? "" : p.substr(eq + 1);
        const bool has_value = eq != std::string::npos;
        char *end = nullptr;
        if (key == "argmax" && !has_value) {
            out->argmax = true;
        } else if (key == "stats" && !has_value) {
            out->stats = true;
        } else if (key == "temp" && has_value) {
            out->temp = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(out->temp > 0) || !std::isfinite(out->temp)) {
                *err = "nn: temp= expects a positive number, got '" + value + "'";
                return false;
            }
        } else if (key == "thresh" && has_value) {
            out->thresh = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(out->thresh >= 0 && out->thresh < 1)) {
                *err = "nn: thresh= expects a probability, 0 <= p < 1, got '" + value + "'";
                return false;
            }
        } else if (key == "disc" && has_value) {
            const unsigned long long n = std::strtoull(value.c_str(), &end, 10);
            if (value.empty() || *end != '\0' || value[0] == '-' || n > 1000000) {
                *err = "nn: disc= expects a whole number of steps (0 to 1000000), got '" + value + "'";
                return false;
            }
            out->disc = static_cast<int>(n);
        } else if (key == "dump" && has_value && !value.empty()) {
            out->dump = value;
        } else if (key == "evals" && has_value && !value.empty()) {
            out->evals = value;
        } else if (remote && key == "policies" && has_value) {
            if (!parse_policies(value, &out->policies)) {
                *err = "nn: policies= expects distinct policy ids separated by commas, got '" + value + "'";
                return false;
            }
        } else if (remote && key == "timeout" && has_value) {
            out->timeout = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(out->timeout > 0) || !std::isfinite(out->timeout)) {
                *err = "nn: timeout= expects a positive number of seconds, got '" + value + "'";
                return false;
            }
        } else if (remote && key == "shm" && has_value) {
            const unsigned long long mib = std::strtoull(value.c_str(), &end, 10);
            if (value.empty() || *end != '\0' || value[0] == '-' || mib == 0 || mib > 65536) {
                *err = "nn: shm= expects a size in MiB (1 to 65536), got '" + value + "'";
                return false;
            }
            out->shm_mib = static_cast<size_t>(mib);
        } else if (!remote && (key == "policies" || key == "timeout" || key == "shm")) {
            *err = "nn: '" + p + "' is an option of nn:remote: only";
            return false;
        } else {
            *err = "nn: unknown option '" + p +
                   "' (keys: temp=<t>, thresh=<p>, disc=<n>, argmax, dump=<file>, evals=<file>, stats" +
                   (remote ? ", policies=<ids>, timeout=<s>, shm=<MiB>)" : ")");
            return false;
        }
    }
    return true;
}

NnAgent::NnAgent(NnOptions opt, agents::PyRandom *rng, std::unique_ptr<featurize::Featurization> f,
    std::unique_ptr<Evaluator> evaluator, ModelStamp stamp)
    : opt_(std::move(opt)), rng_(rng), featurization_(std::move(f)), evaluator_(std::move(evaluator)), stamp_(stamp) {
    if (opt_.dump.empty() && opt_.evals.empty() && !opt_.stats) return;
    if (finishing().empty()) std::atexit(finish_all);
    finishing().push_back(this);
}

void NnAgent::distribution(const std::vector<float> &logits) {
    std::vector<double> &mu = behaviour_.mu;
    const size_t n = logits.size();
    double mx = logits[0];
    for (float l : logits) mx = std::max(mx, static_cast<double>(l));
    mu.assign(n, 0.0);
    if (opt_.argmax) {
        const double floor = mx - ARGMAX_TIE * (1.0 + std::fabs(mx));
        size_t best = 0;
        while (static_cast<double>(logits[best]) < floor) ++best;
        mu[best] = 1.0;
        return;
    }
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        mu[i] = std::exp((static_cast<double>(logits[i]) - mx) / opt_.temp);
        sum += mu[i];
    }
    for (double &p : mu) p /= sum;
    agents::threshold_policy(&mu, opt_.thresh);
    agents::discretize_policy(&mu, opt_.disc);
}

int NnAgent::choose() {
    const std::vector<double> &mu = behaviour_.mu;
    if (opt_.argmax || mu.size() == 1) {
        size_t best = 0;
        for (size_t i = 1; i < mu.size(); ++i)
            if (mu[i] > mu[best]) best = i;
        return static_cast<int>(best);
    }
    return static_cast<int>(rng_->weighted_index(mu.data(), mu.size()));
}

int NnAgent::decide(const agents::Decision &d) {
    seat_ = d.view.seat();
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    if (!featurize::featurize_seat(featurization_->spec, featurization_->tables, d.view.frame(), &tensors_, &err)) {
        failure_ = "nn: featurize: " + err;
        return agents::AGENT_FAILED;
    }
    const auto t1 = std::chrono::steady_clock::now();
    featurize_s_ += std::chrono::duration<double>(t1 - t0).count();
    if (tensors_.a() != d.view.n() || tensors_.a() == 0) {
        failure_ = "nn: the featurized menu has " + std::to_string(tensors_.a()) + " rows, the decision " +
                   std::to_string(d.view.n());
        return agents::AGENT_FAILED;
    }
    if (!evaluator_->evaluate({&tensors_}, &evals_, &err)) {
        failure_ = "nn: " + err;
        return agents::AGENT_FAILED;
    }
    evaluate_s_ += seconds_since(t1);
    ++decisions_;
    const PolicyEval &policy = evals_.at(0).policies.at(0);
    distribution(policy.logits);
    behaviour_.policy_version = policy.version;
    behaviour_.policies.resize(evals_.at(0).policies.size());
    for (size_t i = 0; i < behaviour_.policies.size(); ++i) {
        const PolicyEval &pe = evals_.at(0).policies[i];
        agents::PolicyEvaluation &out = behaviour_.policies[i];
        out.id = opt_.remote.empty() ? 0 : static_cast<int64_t>(opt_.policies.at(i));
        out.version = pe.version;
        out.value = pe.value;
        out.log_pi = pe.log_pi;
    }
    behaviour_.tensors = &tensors_;
    behaviour_.spec_hash = featurization_->spec.hash;
    have_behaviour_ = true;
    if (!opt_.dump.empty()) dumped_.push_back(tensors_);
    const int row = choose();
    if (!opt_.evals.empty()) {
        evaluated_.push_back(evals_.at(0));
        evaluated_mu_.insert(evaluated_mu_.end(), behaviour_.mu.begin(), behaviour_.mu.end());
        evaluated_rows_.push_back(row);
    }
    return row;
}

featurize::RmtbFile NnAgent::evaluations_file() const {
    featurize::RmtbFile f;
    f.kind = "evaluations";
    f.has_vocab_hash = f.has_spec_hash = true;
    f.vocab_hash = featurization_->spec.vocab_hash;
    f.spec_hash = featurization_->spec.hash;
    const size_t p = evaluator_->policies();
    std::vector<int64_t> ids, versions, offsets{0};
    std::vector<float> values, logits, log_pi;
    for (size_t i = 0; i < p; ++i) ids.push_back(opt_.remote.empty() ? 0 : opt_.policies[i]);
    for (const Evaluation &e : evaluated_) {
        const size_t rows = e.policies.at(0).logits.size();
        for (const PolicyEval &pe : e.policies) {
            versions.push_back(static_cast<int64_t>(pe.version));
            values.push_back(pe.value);
        }
        for (size_t r = 0; r < rows; ++r)
            for (const PolicyEval &pe : e.policies) {
                logits.push_back(pe.logits.at(r));
                log_pi.push_back(pe.log_pi.at(r));
            }
        offsets.push_back(offsets.back() + static_cast<int64_t>(rows));
    }
    const uint64_t d = evaluated_.size(), a = static_cast<uint64_t>(offsets.back());
    f.arrays.push_back(featurize::rmtb_ints("policy", featurize::Dtype::I64, {p}, ids));
    f.arrays.push_back(featurize::rmtb_ints("version", featurize::Dtype::I64, {d, p}, versions));
    f.arrays.push_back(featurize::rmtb_floats("value", featurize::Dtype::F32, {d, p}, values));
    f.arrays.push_back(featurize::rmtb_floats("logits", featurize::Dtype::F32, {a, p}, logits));
    f.arrays.push_back(featurize::rmtb_floats("log_pi", featurize::Dtype::F32, {a, p}, log_pi));
    f.arrays.push_back(featurize::rmtb_ints("menu_offsets", featurize::Dtype::I64, {d + 1}, offsets));
    const std::vector<float> mu(evaluated_mu_.begin(), evaluated_mu_.end());
    f.arrays.push_back(featurize::rmtb_floats("mu", featurize::Dtype::F32, {a}, mu));
    f.arrays.push_back(featurize::rmtb_ints("chosen", featurize::Dtype::I64, {d}, evaluated_rows_));
    return f;
}

void NnAgent::finish() {
    std::fflush(stdout);
    std::string err;
    if (!opt_.dump.empty() &&
        !featurize::rmtb_write_file(opt_.dump, featurize::tensors_file(featurization_->spec, dumped_), &err))
        std::printf("ERROR: nn: dump: %s\n", err.c_str());
    if (!opt_.evals.empty() && !featurize::rmtb_write_file(opt_.evals, evaluations_file(), &err))
        std::printf("ERROR: nn: evals: %s\n", err.c_str());
    if (opt_.stats)
        std::printf(
            "NN-STATS: seat %s decisions=%zu featurize_s=%.3f evaluate_s=%.3f model=%s format=%s reloads=%zu%s\n",
            seat_ == 0   ? "A"
            : seat_ == 1 ? "B"
                         : "-",
            decisions_, featurize_s_, evaluate_s_, evaluator_->info().path.c_str(), evaluator_->info().format.c_str(),
            reloads_, evaluator_->stats().c_str());
    std::fflush(stdout);
}

void register_nn_agent() {
    agents::register_agent_kind("nn:", make_nn_agent);
}

}  // namespace actor
