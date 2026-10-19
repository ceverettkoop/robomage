#ifndef ROBOMAGE_CHECK_GATE_H
#define ROBOMAGE_CHECK_GATE_H

#include <mutex>
#include <string>
#include <utility>
#include <vector>

// ── robomage_check: the gate's options, its report and the tiers ──────────────────────────────

namespace rmcheck {

struct Options {
        std::string repo;     // the repository root
        std::string bin_dir;  // <repo>/bin: the engine's working directory (resources/)
        std::string binary;   // the engine
        std::string rmknow;   // the knowledge tool beside it
        std::string rmfeat;   // the featurizer tool beside it
        std::string out_dir;  // transcripts, engine stderr, draw logs
        std::string python;   // the interpreter for the smoke tier ("" = none found)
        int jobs = 4;
        long seed = 1;
        int fuzz_games = 8;
        int manual_mana_games = 2;  // fuzz: --no-auto-mana games per ring matchup
        int smoke_games = 2;
        std::string matchups;  // fuzz / smoke matchup preset ("" = the tier's default)
        std::string mode = "explore";
        bool record = false;                         // replay, cli: write the baselines instead of checking them
        std::vector<std::string> scenario_patterns;  // scenarios: name globs ("" = all)
        std::string show;  // scenarios: print each run's output | narrative | dump (knowledge lines)

        std::string corpus_dir() const { return repo + "/tests/corpus"; }
        std::string scenario_dir() const { return repo + "/tests/scenarios"; }
        std::string cli_dir() const { return repo + "/tests/cli"; }
        std::string decks_dir() const { return bin_dir + "/resources/decks"; }
        // <out_dir>/<sub>, created.
        std::string out(const std::string &sub) const;
};

// Errors (they fail the gate) and warnings (flagged for review), each printed as it is reported.
// Thread-safe: a tier's jobs report from the pool's threads.
class Report {
    public:
        void error(const std::string &tier, const std::string &msg);
        void warn(const std::string &tier, const std::string &msg);
        // A progress line ("  <msg>").
        void info(const std::string &msg);
        size_t error_count() const;
        const std::vector<std::pair<std::string, std::string>> &errors() const { return errors_; }
        const std::vector<std::pair<std::string, std::string>> &warnings() const { return warnings_; }

    private:
        mutable std::mutex mu_;
        std::vector<std::pair<std::string, std::string>> errors_, warnings_;
};

// The league roster: the stems of decks/league/*.dk, sorted.
std::vector<std::string> league_decks(const Options &opt);
// A matchup preset (mirrors | ring | mirrors+ring | all | "a:b,c:d") as (deck_a, deck_b) specs
// relative to decks/ ("league/<stem>").
std::vector<std::pair<std::string, std::string>> matchups(const Options &opt, const std::string &preset);
// The bare deck stem of a spec ("league/bug" → "bug").
std::string deck_stem(const std::string &spec);

void tier_vocab(const Options &opt, Report &rep);
void tier_concede(const Options &opt, Report &rep);
void tier_scenarios(const Options &opt, Report &rep);
void tier_cli(const Options &opt, Report &rep);
void tier_replay(const Options &opt, Report &rep);
void tier_state(const Options &opt, Report &rep);
void tier_featurize(const Options &opt, Report &rep);
void tier_traj(const Options &opt, Report &rep);
void tier_actor(const Options &opt, Report &rep);
void tier_knowledge(const Options &opt, Report &rep);
void tier_fuzz(const Options &opt, Report &rep);
void tier_bench(const Options &opt, Report &rep);
void tier_smoke(const Options &opt, Report &rep);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_GATE_H */
