#include "episodes.h"

#include <dirent.h>

#include <algorithm>

#include "../../src/featurize/rmtb.h"
#include "files.h"

static std::string base_name(const std::string &path);
static std::string log_stem(const std::string &log_path);
static int game_number(const std::string &name, const std::string &stem);
static bool ends_with(const std::string &s, const std::string &suffix);

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// "a0-m12.rmlog" -> "a0-m12".
static std::string log_stem(const std::string &log_path) {
    std::string name = base_name(log_path);
    if (ends_with(name, ".rmlog")) name.resize(name.size() - 6);
    return name;
}

// n for "<stem>.g<n>.rmtraj", else -1.
static int game_number(const std::string &name, const std::string &stem) {
    const std::string head = stem + ".g";
    if (name.compare(0, head.size(), head) != 0 || !ends_with(name, ".rmtraj")) return -1;
    const std::string digits = name.substr(head.size(), name.size() - head.size() - 7);
    if (digits.empty() || digits.find_first_not_of("0123456789") != std::string::npos || digits.size() > 3) return -1;
    return std::atoi(digits.c_str());
}

static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

namespace rmwork {

std::vector<std::string> episode_dirs_for_log(const std::string &log_path) {
    std::vector<std::string> dirs;
    const std::string dir = parent_dir(log_path);
    dirs.push_back(dir.empty() ? "." : dir);
    // <run>/rmlog/<purpose>/<stem>.rmlog -> <run>/episodes/<purpose>
    const std::string purpose = base_name(dir);
    const std::string rmlog_dir = parent_dir(dir);
    if (base_name(rmlog_dir) == "rmlog") dirs.push_back(parent_dir(rmlog_dir) + "/episodes/" + purpose);
    // <out>/<purpose>/rmlog/<stem>.rmlog -> <out>/<purpose>/traj
    if (purpose == "rmlog") dirs.push_back(parent_dir(dir) + "/traj");
    return dirs;
}

std::vector<std::string> episode_files_for_log(const std::string &log_path) {
    const std::string stem = log_stem(log_path);
    std::vector<std::pair<int, std::string>> found;
    for (const std::string &dir : episode_dirs_for_log(log_path)) {
        // Only the exact names are looked for: a run's episode directory holds tens of thousands.
        for (int g = 1; g <= 9; ++g) {
            const std::string path = dir + "/" + stem + ".g" + std::to_string(g) + ".rmtraj";
            if (path_kind(path) == PathKind::FILE) found.emplace_back(g, path);
        }
        if (!found.empty()) break;
    }
    std::sort(found.begin(), found.end());
    std::vector<std::string> out;
    for (const auto &f : found) out.push_back(f.second);
    return out;
}

std::string log_for_episode(const std::string &episode_path, const std::string &engine_workdir, std::string *err) {
    featurize::RmtbFile file;
    if (!featurize::rmtb_read_file(episode_path, &file, err)) return "";
    if (const featurize::RmtbAttr *a = file.attr("rmlog")) {
        if (!a->is_int && !a->s.empty()) {
            const std::string p = a->s[0] == '/' ? a->s : engine_workdir + "/" + a->s;
            if (path_kind(p) == PathKind::FILE) return p;
        }
    }
    const std::string name = base_name(episode_path);
    const size_t g = name.rfind(".g");
    if (g == std::string::npos || game_number(name, name.substr(0, g)) < 0) {
        *err = name + " is not named <log stem>.g<n>.rmtraj, and its rmlog attribute names no file";
        return "";
    }
    const std::string stem = name.substr(0, g);
    const std::string dir = parent_dir(episode_path);
    std::vector<std::string> candidates = {(dir.empty() ? "." : dir) + "/" + stem + ".rmlog"};
    // <run>/episodes/<purpose> -> <run>/rmlog/<purpose>; <out>/<purpose>/traj -> <out>/<purpose>/rmlog
    const std::string up = parent_dir(dir);
    if (base_name(up) == "episodes")
        candidates.push_back(parent_dir(up) + "/rmlog/" + base_name(dir) + "/" + stem + ".rmlog");
    if (base_name(dir) == "traj") candidates.push_back(up + "/rmlog/" + stem + ".rmlog");
    for (const std::string &c : candidates)
        if (path_kind(c) == PathKind::FILE) return c;
    *err = "no decision log " + stem + ".rmlog beside the episode or in its run directory";
    return "";
}

bool join_episode(Match *m, const std::string &path, JoinReport *report, std::string *err) {
    *report = JoinReport();
    featurize::RmtbFile file;
    if (!featurize::rmtb_read_file(path, &file, err)) return false;
    if (const featurize::RmtbAttr *seed = file.attr("engine_seed")) {
        if (seed->is_int && static_cast<uint64_t>(seed->i) != m->log.seed) {
            *err = base_name(path) + " is of another match: its engine seed " + std::to_string(seed->i) +
                   ", the log's " + std::to_string(m->log.seed);
            return false;
        }
    }
    featurize::Episode ep;
    if (!featurize::read_episode(file, &ep, err)) return false;
    const char *agent_keys[2] = {"agent_a", "agent_b"};
    for (size_t s = 0; s < 2; ++s)
        if (const featurize::RmtbAttr *a = file.attr(agent_keys[s]))
            if (!a->is_int && !a->s.empty()) m->agents[s] = a->s;
    report->steps = ep.steps.size();
    for (featurize::EpisodeStep &s : ep.steps) {
        const int at = m->find_decision(static_cast<int>(s.game), s.sideboarding, s.decision);
        const std::string where = "game " + std::to_string(s.game + 1) + (s.sideboarding ? " sideboarding" : "") +
                                  " decision " + std::to_string(s.decision);
        if (at < 0) {
            report->problems.push_back(where + ": no such decision in the replay");
            continue;
        }
        DecisionInfo &d = m->decisions[static_cast<size_t>(at)];
        if (static_cast<int>(s.seat) != d.decider) {
            report->problems.push_back(where + ": the episode's seat is not the deciding seat");
            continue;
        }
        if (s.chosen != d.reply) {
            report->problems.push_back(where + ": the episode's reply " + std::to_string(s.chosen) +
                                       " is not the log's " + std::to_string(d.reply));
            continue;
        }
        EpisodeStepInfo info;
        info.file = path;
        info.seat = s.seat;
        info.chosen = s.chosen;
        info.has_mu = s.has_mu;
        info.mu = std::move(s.mu);
        info.policy_version = s.policy_version;
        info.teacher = s.teacher;
        info.policies = std::move(s.policies);
        s.tensors = featurize::Tensors();
        if (d.episode_step >= 0) {
            m->episode_steps[static_cast<size_t>(d.episode_step)] = std::move(info);
        } else {
            d.episode_step = static_cast<int>(m->episode_steps.size());
            m->episode_steps.push_back(std::move(info));
        }
        ++report->joined;
    }
    if (std::find(m->episode_files.begin(), m->episode_files.end(), path) == m->episode_files.end())
        m->episode_files.push_back(path);
    return true;
}

}  // namespace rmwork
