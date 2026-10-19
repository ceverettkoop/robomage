#ifndef ROBOMAGE_CHECK_TRAJ_H
#define ROBOMAGE_CHECK_TRAJ_H

#include <string>
#include <vector>

#include "../../src/featurize/episode.h"
#include "gate.h"

// ── robomage_check: trajectory episodes (--trajectory-out, docs/trajectory_format.md) ─────────
//
// Shared by the traj tier (tier_traj.cpp) and the actor tier's nn: checks: the episodes of a
// directory read with the C++ reader (src/featurize/episode.h), and their comparisons.

namespace rmcheck {

struct EpisodeFile {
        std::string path, name;
        int game_number = 0;  // n of <stem>.g<n>.rmtraj
        featurize::Episode ep;
};

// The episodes of `dir` (*.rmtraj) in game order, read; false with `err` (a file that does not
// read, or a temporary file left behind).
bool read_episodes(const std::string &dir, std::vector<EpisodeFile> *out, std::string *err);

// "" when an episode's step tensors equal the featurizer's (`featurized`), its tok_num rounded to
// f16 as the episode stores it; else the first difference.
std::string compare_tensors(const featurize::Tensors &episode, const featurize::Tensors &featurized);

// Failures where the episodes of a --replay of the run's decision log differ from the live ones:
// the same files, every array byte for byte but the behaviour policy's (has_mu, mu, policy_version,
// eval_*), the same attrs but the agents', the game seed and the policy versions.
std::vector<std::string> compare_replayed(
    const std::vector<EpisodeFile> &live, const std::vector<EpisodeFile> &replayed);

// Failures where the recorded steps of `seat` (0 A, 1 B) in `eps` differ from an nn: agent's
// evals= file: one step per evaluation in order, μ, the chosen row, the policy version and every
// policy's value and log π equal. `checked` gets the steps compared.
std::vector<std::string> check_against_evals(
    const std::vector<EpisodeFile> &eps, int seat, const std::string &evals, size_t *checked);

// Failures where the recorded steps of `seat` in `eps` differ from an nn: agent's dump= tensors
// file (compare_tensors, step by step).
std::vector<std::string> check_against_dump(
    const std::vector<EpisodeFile> &eps, int seat, const std::string &dump, size_t *checked);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_TRAJ_H */
