#include "nn_agent.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "../state_format/cards_file.h"
#include "../str_util.h"

namespace {

// argmax's tie: rows whose logits are within this (absolute, and relative to the largest) of the
// largest are tied, and the first of them is taken, so the float drift between two exports of one
// model cannot reorder rows that are equal in exact arithmetic (two identical cards in hand).
const double ARGMAX_TIE = 1e-5;

}  // namespace

static std::vector<actor::NnAgent *> &finishing();
static void finish_all();
static double seconds_since(std::chrono::steady_clock::time_point t0);
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

// The nn: factory: the options, the featurizer's inputs from the engine, the model, checked.
static std::unique_ptr<agents::Agent> make_nn_agent(const std::string &args, agents::PyRandom *rng, std::string *err) {
    actor::NnOptions opt;
    if (!actor::NnAgent::parse_options(args, &opt, err)) return nullptr;
    std::unique_ptr<featurize::Featurization> f(new featurize::Featurization());
    if (!load_engine_featurization(f.get(), err)) {
        *err = "nn: the featurizer's inputs: " + *err;
        return nullptr;
    }
    std::unique_ptr<actor::Evaluator> e = actor::load_model_evaluator(opt.model, f->spec, err);
    if (!e || !actor::check_model(e->info(), *f, err)) {
        *err = "nn: " + *err;
        return nullptr;
    }
    for (const actor::NnAgent *a : finishing())
        if (!opt.dump.empty() && a->options().dump == opt.dump) {
            *err = "nn: both seats dump to " + opt.dump;
            return nullptr;
        }
    return std::unique_ptr<agents::Agent>(new actor::NnAgent(std::move(opt), rng, std::move(f), std::move(e)));
}

namespace actor {

bool NnAgent::parse_options(const std::string &args, NnOptions *out, std::string *err) {
    const std::vector<std::string> parts = split(args, ':');
    if (parts.empty() || parts[0].empty()) {
        *err = "nn: expects nn:<model>[:<key>=<value>...], got 'nn:" + args + "'";
        return false;
    }
    if (ascii_lower(parts[0]) == "remote") {
        *err =
            "nn:remote: the remote evaluator (docs/inference_protocol.md) is not implemented yet (plan step 5d part 2)";
        return false;
    }
    out->model = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        const std::string &p = parts[i];
        const size_t eq = p.find('=');
        const std::string key = ascii_lower(p.substr(0, eq)), value = eq == std::string::npos ? "" : p.substr(eq + 1);
        if (key == "argmax" && eq == std::string::npos) {
            out->argmax = true;
        } else if (key == "stats" && eq == std::string::npos) {
            out->stats = true;
        } else if (key == "temp" && eq != std::string::npos) {
            char *end = nullptr;
            out->temp = std::strtod(value.c_str(), &end);
            if (value.empty() || *end != '\0' || !(out->temp > 0) || !std::isfinite(out->temp)) {
                *err = "nn: temp= expects a positive number, got '" + value + "'";
                return false;
            }
        } else if (key == "dump" && eq != std::string::npos && !value.empty()) {
            out->dump = value;
        } else {
            *err = "nn: unknown option '" + p + "' (keys: temp=<t>, argmax, dump=<file>, stats)";
            return false;
        }
    }
    return true;
}

NnAgent::NnAgent(NnOptions opt, agents::PyRandom *rng, std::unique_ptr<featurize::Featurization> f,
    std::unique_ptr<Evaluator> evaluator)
    : opt_(std::move(opt)), rng_(rng), featurization_(std::move(f)), evaluator_(std::move(evaluator)) {
    if (opt_.dump.empty() && !opt_.stats) return;
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
    have_behaviour_ = true;
    if (!opt_.dump.empty()) dumped_.push_back(tensors_);
    return choose();
}

void NnAgent::finish() {
    std::fflush(stdout);
    if (!opt_.dump.empty()) {
        std::string err;
        if (!featurize::rmtb_write_file(opt_.dump, featurize::tensors_file(featurization_->spec, dumped_), &err))
            std::printf("ERROR: nn: dump: %s\n", err.c_str());
    }
    if (opt_.stats)
        std::printf("NN-STATS: seat %s decisions=%zu featurize_s=%.3f evaluate_s=%.3f model=%s format=%s\n",
            seat_ == 0   ? "A"
            : seat_ == 1 ? "B"
                         : "-",
            decisions_, featurize_s_, evaluate_s_, evaluator_->info().path.c_str(), evaluator_->info().format.c_str());
    std::fflush(stdout);
}

void register_nn_agent() {
    agents::register_agent_kind("nn:", make_nn_agent);
}

}  // namespace actor
