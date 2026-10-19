#include "traj.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "util.h"

static bool behaviour_array(const std::string &name);
static bool agent_attr(const std::string &key);
static bool tensors_of_file(const featurize::RmtbFile &f, std::vector<featurize::Tensors> *out, std::string *err);
static float f16_round(float v);
template <typename T>
static bool slice_of(const featurize::RmtbFile &f, const std::string &name, size_t d, size_t width, std::vector<T> *out,
    std::string *err);

// The arrays only the live run's behaviour policy fills.
static bool behaviour_array(const std::string &name) {
    return name == "has_mu" || name == "mu" || name == "policy_version" || rmcheck::starts_with(name, "eval_");
}

// The attrs that name the deciders (and what only the live run knows of them).
static bool agent_attr(const std::string &key) {
    return key == "agent_a" || key == "agent_b" || key == "game_seed" || key == "policy_versions_a" ||
           key == "policy_versions_b";
}

static float f16_round(float v) {
    return featurize::f16_to_f32(featurize::f32_to_f16(v));
}

// Decision `d`'s rows of the ragged tensor `name` of a tensors file (`width` values per row).
template <typename T>
static bool slice_of(const featurize::RmtbFile &f, const std::string &name, size_t d, size_t width, std::vector<T> *out,
    std::string *err) {
    const featurize::RmtbArray *a = f.find(name);
    std::vector<int64_t> offsets;
    const featurize::RmtbArray *o = f.find(name + "_offsets");
    if (!a || !o || !featurize::rmtb_values(*o, &offsets) || d + 1 >= offsets.size()) {
        *err = "the tensors file has no " + name;
        return false;
    }
    std::vector<T> all;
    if (!featurize::rmtb_values(*a, &all)) {
        *err = "the tensors file's " + name + " does not read";
        return false;
    }
    out->assign(all.begin() + offsets[d] * static_cast<int64_t>(width),
        all.begin() + offsets[d + 1] * static_cast<int64_t>(width));
    return true;
}

// Every decision of a tensors file (featurize::tensors_file's layout).
static bool tensors_of_file(const featurize::RmtbFile &f, std::vector<featurize::Tensors> *out, std::string *err) {
    out->clear();
    std::vector<int64_t> viewer, game, decision, sideboarding;
    const featurize::RmtbArray *v = f.find("viewer"), *num = f.find("tok_num"), *cat = f.find("tok_cat"),
                               *card = f.find("tok_card"), *ab = f.find("tok_ability");
    if (!v || !num || !cat || !card || !ab || !featurize::rmtb_values(*v, &viewer) ||
        !featurize::rmtb_values(*f.find("game"), &game) || !featurize::rmtb_values(*f.find("decision"), &decision) ||
        !featurize::rmtb_values(*f.find("sideboarding"), &sideboarding) || num->shape.size() != 2 ||
        cat->shape.size() != 2 || card->shape.size() != 2 || ab->shape.size() != 2) {
        *err = "not a tensors file";
        return false;
    }
    for (size_t d = 0; d < viewer.size(); ++d) {
        featurize::Tensors t;
        t.viewer = static_cast<uint8_t>(viewer[d]);
        t.game = game[d];
        t.decision = decision[d];
        t.sideboarding = sideboarding[d] != 0;
        std::vector<int64_t> tok_mask, act_mask;
        if (!slice_of(f, "tok_type", d, 1, &t.tok_type, err) ||
            !slice_of(f, "tok_num", d, static_cast<size_t>(num->shape[1]), &t.tok_num, err) ||
            !slice_of(f, "tok_cat", d, static_cast<size_t>(cat->shape[1]), &t.tok_cat, err) ||
            !slice_of(f, "tok_card", d, static_cast<size_t>(card->shape[1]), &t.tok_card, err) ||
            !slice_of(f, "tok_ability", d, static_cast<size_t>(ab->shape[1]), &t.tok_ability, err) ||
            !slice_of(f, "tok_mask", d, 1, &tok_mask, err) || !slice_of(f, "rel", d, 3, &t.rel, err) ||
            !slice_of(f, "act_tok", d, 1, &t.act_tok, err) || !slice_of(f, "act_mask", d, 1, &act_mask, err))
            return false;
        t.tok_mask.assign(tok_mask.begin(), tok_mask.end());
        t.act_mask.assign(act_mask.begin(), act_mask.end());
        out->push_back(std::move(t));
    }
    return true;
}

namespace rmcheck {

bool read_episodes(const std::string &dir, std::vector<EpisodeFile> *out, std::string *err) {
    out->clear();
    if (!list_dir(dir, ".tmp").empty()) {
        *err = dir + " holds a temporary file (an episode not renamed into place)";
        return false;
    }
    for (const std::string &name : list_dir(dir, ".rmtraj")) {
        EpisodeFile e;
        e.name = name;
        e.path = path_join(dir, name);
        const size_t g = name.rfind(".g");
        e.game_number = g == std::string::npos ? 0 : std::atoi(name.c_str() + g + 2);
        featurize::RmtbFile f;
        if (!featurize::rmtb_read_file(e.path, &f, err)) return false;
        if (!featurize::read_episode(f, &e.ep, err)) {
            *err = e.path + ": " + *err;
            return false;
        }
        out->push_back(std::move(e));
    }
    std::sort(out->begin(), out->end(),
        [](const EpisodeFile &a, const EpisodeFile &b) { return a.game_number < b.game_number; });
    return true;
}

std::string compare_tensors(const featurize::Tensors &episode, const featurize::Tensors &featurized) {
    if (episode.viewer != featurized.viewer || episode.game != featurized.game ||
        episode.decision != featurized.decision || episode.sideboarding != featurized.sideboarding)
        return "the decision key or viewer";
    if (episode.tok_type != featurized.tok_type) return "tok_type";
    if (episode.tok_num.size() != featurized.tok_num.size()) return "tok_num's size";
    for (size_t i = 0; i < episode.tok_num.size(); ++i) {
        const float a = episode.tok_num[i], b = f16_round(featurized.tok_num[i]);
        if (std::memcmp(&a, &b, sizeof a) != 0) return "tok_num[" + std::to_string(i) + "]";
    }
    if (episode.tok_cat != featurized.tok_cat) return "tok_cat";
    if (episode.tok_card != featurized.tok_card) return "tok_card";
    if (episode.tok_ability != featurized.tok_ability) return "tok_ability";
    if (episode.tok_mask != featurized.tok_mask) return "tok_mask";
    if (episode.rel != featurized.rel) return "rel";
    if (episode.act_tok != featurized.act_tok) return "act_tok";
    if (episode.act_mask != featurized.act_mask) return "act_mask";
    return "";
}

std::vector<std::string> compare_replayed(
    const std::vector<EpisodeFile> &live, const std::vector<EpisodeFile> &replayed) {
    std::vector<std::string> fails;
    if (live.size() != replayed.size()) {
        fails.push_back("the replay wrote " + std::to_string(replayed.size()) + " episodes, the live run " +
                        std::to_string(live.size()));
        return fails;
    }
    for (size_t i = 0; i < live.size(); ++i) {
        const featurize::RmtbFile &a = live[i].ep.file, &b = replayed[i].ep.file;
        const std::string where = live[i].name + " vs its replay: ";
        if (live[i].name != replayed[i].name) fails.push_back(where + "named " + replayed[i].name);
        for (const featurize::RmtbArray &x : a.arrays) {
            if (behaviour_array(x.name)) continue;
            const featurize::RmtbArray *y = b.find(x.name);
            if (!y || y->dtype != x.dtype || y->shape != x.shape || y->data != x.data)
                fails.push_back(where + "array " + x.name + " differs");
        }
        for (const featurize::RmtbAttr &x : a.attrs) {
            if (agent_attr(x.key)) continue;
            const featurize::RmtbAttr *y = b.attr(x.key);
            if (!y || y->is_int != x.is_int || y->i != x.i || y->s != x.s) fails.push_back(where + "attr " + x.key);
        }
        if (fails.size() > 10) break;
    }
    return fails;
}

std::vector<std::string> check_against_evals(
    const std::vector<EpisodeFile> &eps, int seat, const std::string &evals, size_t *checked) {
    std::vector<std::string> fails;
    *checked = 0;
    featurize::RmtbFile f;
    std::string err;
    std::vector<int64_t> policy, version, offsets, chosen;
    std::vector<float> value, log_pi, mu;
    if (!featurize::rmtb_read_file(evals, &f, &err) || !f.find("policy") ||
        !featurize::rmtb_values(*f.find("policy"), &policy) || !featurize::rmtb_values(*f.find("version"), &version) ||
        !featurize::rmtb_values(*f.find("menu_offsets"), &offsets) ||
        !featurize::rmtb_values(*f.find("chosen"), &chosen) || !featurize::rmtb_values(*f.find("value"), &value) ||
        !featurize::rmtb_values(*f.find("log_pi"), &log_pi) || !featurize::rmtb_values(*f.find("mu"), &mu)) {
        fails.push_back(evals + ": not an evaluations file (" + err + ")");
        return fails;
    }
    const size_t p = policy.size();
    size_t d = 0;
    for (const EpisodeFile &e : eps)
        for (size_t i = 0; i < e.ep.steps.size(); ++i) {
            const featurize::EpisodeStep &s = e.ep.steps[i];
            if (s.seat != seat) continue;
            const std::string where = e.name + " step " + std::to_string(i) + ": ";
            if (d + 1 >= offsets.size()) {
                fails.push_back(where + "more steps than the evals file's " + std::to_string(chosen.size()));
                return fails;
            }
            const size_t mb = static_cast<size_t>(offsets[d]), me = static_cast<size_t>(offsets[d + 1]);
            bool same = s.has_mu && s.mu.size() == me - mb && s.chosen == chosen[d] &&
                        s.policy_version == version[d * p] && s.policies.size() == p;
            for (size_t r = 0; same && r < s.mu.size(); ++r) same = s.mu[r] == mu[mb + r];
            for (size_t k = 0; same && k < p; ++k) {
                const featurize::PolicyEvalRecord &pr = s.policies[k];
                same = pr.id == policy[k] && static_cast<int64_t>(pr.version) == version[d * p + k] &&
                       pr.value == value[d * p + k] && pr.log_pi.size() == me - mb;
                for (size_t r = 0; same && r < pr.log_pi.size(); ++r) same = pr.log_pi[r] == log_pi[(mb + r) * p + k];
            }
            if (!same) fails.push_back(where + "mu, the reply, the version or an evaluation differs from " + evals);
            ++d;
            ++*checked;
            if (fails.size() > 10) return fails;
        }
    if (d != chosen.size())
        fails.push_back("the episodes hold " + std::to_string(d) + " steps of the seat, " + evals + " " +
                        std::to_string(chosen.size()));
    return fails;
}

std::vector<std::string> check_against_dump(
    const std::vector<EpisodeFile> &eps, int seat, const std::string &dump, size_t *checked) {
    std::vector<std::string> fails;
    *checked = 0;
    featurize::RmtbFile f;
    std::string err;
    std::vector<featurize::Tensors> list;
    if (!featurize::rmtb_read_file(dump, &f, &err) || !tensors_of_file(f, &list, &err)) {
        fails.push_back(dump + ": " + err);
        return fails;
    }
    size_t d = 0;
    for (const EpisodeFile &e : eps)
        for (size_t i = 0; i < e.ep.steps.size(); ++i) {
            if (e.ep.steps[i].seat != seat) continue;
            if (d >= list.size()) {
                fails.push_back(e.name + ": more steps of the seat than " + dump + "'s " + std::to_string(list.size()));
                return fails;
            }
            const std::string diff = compare_tensors(e.ep.steps[i].tensors, list[d]);
            if (!diff.empty())
                fails.push_back(e.name + " step " + std::to_string(i) + ": " + diff + " differs from " + dump);
            ++d;
            ++*checked;
            if (fails.size() > 10) return fails;
        }
    if (d != list.size())
        fails.push_back("the episodes hold " + std::to_string(d) + " steps of the seat, " + dump + " " +
                        std::to_string(list.size()));
    return fails;
}

}  // namespace rmcheck
