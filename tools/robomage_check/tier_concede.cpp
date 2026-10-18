#include <cstdlib>
#include <memory>
#include <regex>

#include "frames.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "util.h"

// The concede tier's checks, reported as they are made (train/test_concede.py's check()).
struct ConcedeChecks {
        rmcheck::Report *rep;
        int passed = 0;
        void check(bool ok, const std::string &what) {
            if (ok)
                ++passed;
            else
                rep->error("concede", what);
        }
};

// One parked-query line (test_concede.py's _PARKED_CASES).
struct ParkedCase {
        const char *name, *deck_a, *pick, *prompt, *announce, *continued;
};

// The replies of a parked-concession line (test_concede.py's _ParkedScript): row 0 until A's pick is
// on the menu, the pick, row 0 until the parked prompt, then the sentinel; in a bo3 after a game
// concession, row 0 through sideboarding and a second game concession at game 2's first decision.
struct ParkedScript {
        std::string pick, prompt;
        int sentinel = -2;
        bool bo3 = false;
        int stage = 0;  // 0 pick, 1 prompt, 2 game 2, 3 done
        int conceded = 0;
        int reply(const rmcheck::StreamDecision &d) {
            if (stage == 0) {
                for (size_t i = 0; i < d.descriptions.size(); ++i)
                    if (rmcheck::contains(d.descriptions[i], pick.c_str())) {
                        stage = 1;
                        return static_cast<int>(i);
                    }
                return 0;
            }
            if (stage == 1) {
                bool at = prompt.empty();
                for (const std::string &desc : d.descriptions) at = at || rmcheck::contains(desc, prompt.c_str());
                if (!at) return 0;
                stage = bo3 && sentinel == -2 ? 2 : 3;
                ++conceded;
                return sentinel;
            }
            if (stage == 2 && !d.sideboarding) {
                stage = 3;
                ++conceded;
                return -2;
            }
            return 0;
        }
};

static rmcheck::ProcSpec stream_spec(const rmcheck::Options &opt, const std::vector<std::string> &flags);
static std::vector<std::string> result_lines(const std::string &text);
static std::string bquery_text(const std::string &out);
static std::string other(const std::string &seat);
static std::string seat_of(int viewer);
static void natural_lines(const rmcheck::Options &opt, ConcedeChecks &c);
static void determinism(const rmcheck::Options &opt, ConcedeChecks &c);
static void parked(const rmcheck::Options &opt, ConcedeChecks &c);
static void check_parked_replay(const rmcheck::Options &opt, ConcedeChecks &c, const std::string &label,
    const std::string &log, const std::vector<std::string> &live, const std::string &announce, int sentinel);

static rmcheck::ProcSpec stream_spec(const rmcheck::Options &opt, const std::vector<std::string> &flags) {
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--machine-format", "rmstate", "--narrative"};
    spec.argv.insert(spec.argv.end(), flags.begin(), flags.end());
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 300;
    return spec;
}

// The GAME_RESULT / MATCH_RESULT lines of `text`.
static std::vector<std::string> result_lines(const std::string &text) {
    std::vector<std::string> out;
    for (const std::string &l : rmcheck::split_lines(text))
        if (rmcheck::starts_with(l, "GAME_RESULT: ") || rmcheck::starts_with(l, "MATCH_RESULT: ")) out.push_back(l);
    return out;
}

// A BQUERY --machine stdout without its frames ("BQUERY: <n> <state size> <max actions>" + the
// binary payload of 4 × (state size + 7 × max actions) bytes, without --narrative): its text lines.
static std::string bquery_text(const std::string &out) {
    std::string text;
    size_t at = 0;
    while (at < out.size()) {
        if (out.compare(at, 8, "BQUERY: ") == 0) {
            const size_t nl = out.find('\n', at);
            long n = 0, state = 0, max_actions = 0;
            if (nl == std::string::npos ||
                std::sscanf(out.substr(at, nl - at).c_str(), "BQUERY: %ld %ld %ld", &n, &state, &max_actions) != 3)
                break;
            at = nl + 1 + static_cast<size_t>(4 * (state + 7 * max_actions));
            continue;
        }
        size_t nl = out.find('\n', at);
        if (nl == std::string::npos) nl = out.size();
        text += out.substr(at, nl - at) + "\n";
        at = nl + 1;
    }
    return text;
}

static std::string other(const std::string &seat) {
    return seat == "Player A" ? "Player B" : "Player A";
}

static std::string seat_of(int viewer) {
    return viewer == rmcheck::VIEWER_A ? "Player A" : "Player B";
}

// test_concede.py's checks 1-3b: delver vs mav, row 0 for twelve decisions, then a concession.
static void natural_lines(const rmcheck::Options &opt, ConcedeChecks &c) {
    const std::vector<std::string> decks = {"--deck-a", "delver", "--deck-b", "mav"};
    // [1] bo1: the conceding seat loses.
    {
        std::string conceder;
        std::vector<std::string> flags = {"--seed", "4242"};
        flags.insert(flags.end(), decks.begin(), decks.end());
        rmcheck::StreamRun run = rmcheck::run_stream(stream_spec(opt, flags), [&](const rmcheck::StreamDecision &d) {
            if (d.decision < 13) return 0;
            if (conceder.empty()) conceder = seat_of(d.viewer);
            return -2;
        });
        const std::string &err = run.proc.err;
        c.check(run.error.empty() && run.proc.ok(),
            "[1] bo1: the session ended cleanly (" + run.error + run.proc.status_text() + ")");
        c.check(!conceder.empty(), "[1] bo1: reached a mid-game decision without the game ending");
        c.check(rmcheck::contains(err, (conceder + " concedes").c_str()),
            "[1] bo1: the narrative announced the concession");
        c.check(result_lines(err) == std::vector<std::string>{"GAME_RESULT: 1 " + other(conceder) + " wins"},
            "[1] bo1: GAME_RESULT names the conceder's opponent");
        c.check(run.terminals == 2, "[1] bo1: one terminal frame per seat");
    }
    // [2] bo3: a game concession is a game loss; the match goes on into sideboarding and game 2.
    {
        int stage = 0;  // 0 before the concession, 1 sideboarding expected, 2 in sideboarding, 3 game 2
        bool sideboard_next = false, game2 = false;
        std::vector<std::string> flags = {"--seed", "909", "--bo3"};
        flags.insert(flags.end(), decks.begin(), decks.end());
        rmcheck::StreamRun run = rmcheck::run_stream(stream_spec(opt, flags), [&](const rmcheck::StreamDecision &d) {
            if (stage == 0) {
                if (d.decision < 13) return 0;
                stage = 1;
                return -2;
            }
            if (stage == 1) {
                sideboard_next = d.sideboarding;
                stage = 2;
            }
            if (stage == 2 && d.sideboarding) return 0;
            game2 = !d.sideboarding && d.game >= 1;
            stage = 3;
            return -3;
        });
        const std::vector<std::string> results = result_lines(run.proc.err);
        c.check(run.error.empty() && run.proc.ok(),
            "[2] bo3: the session ended cleanly (" + run.error + run.proc.status_text() + ")");
        c.check(sideboard_next, "[2] bo3: the decision after the game concession is a sideboarding one");
        c.check(game2, "[2] bo3: game 2 of the match started after sideboarding");
        c.check(results.size() == 3 && rmcheck::starts_with(results[0], "GAME_RESULT: 1 ") &&
                    rmcheck::starts_with(results[1], "GAME_RESULT: 2 ") &&
                    rmcheck::starts_with(results[2], "MATCH_RESULT: "),
            "[2] bo3: game 1 and game 2 results, then the match result");
    }
    // [3] bo3: a match concession ends the match, the opponent winning it.
    {
        std::string conceder;
        std::vector<std::string> flags = {"--seed", "909", "--bo3"};
        flags.insert(flags.end(), decks.begin(), decks.end());
        rmcheck::StreamRun run = rmcheck::run_stream(stream_spec(opt, flags), [&](const rmcheck::StreamDecision &d) {
            if (d.decision < 13) return 0;
            conceder = seat_of(d.viewer);
            return -3;
        });
        const std::string &err = run.proc.err;
        c.check(run.error.empty() && run.proc.ok(), "[3] bo3 -3: the session ended cleanly");
        c.check(rmcheck::contains(err, ("MATCH_RESULT: " + other(conceder) + " wins").c_str()),
            "[3] bo3 -3: MATCH_RESULT named the opponent the match winner");
        c.check(
            rmcheck::contains(err, "concedes the match"), "[3] bo3 -3: the narrative announced the match concession");
        c.check(!rmcheck::contains(err, "MATCH GAME 2"), "[3] bo3 -3: no second game was played");
        c.check(run.terminals == 2, "[3] bo3 -3: one terminal frame per seat");
    }
    // [3b] the sentinels between games: -2 has no live game (ignored), -3 still ends the match.
    {
        int stage = 0;
        std::string sideboarder;
        bool continued = false;
        std::vector<std::string> flags = {"--seed", "909", "--bo3"};
        flags.insert(flags.end(), decks.begin(), decks.end());
        rmcheck::StreamRun run = rmcheck::run_stream(stream_spec(opt, flags), [&](const rmcheck::StreamDecision &d) {
            if (stage == 0) {
                if (d.decision < 13) return 0;
                stage = 1;
                return -2;
            }
            if (stage == 1) {
                stage = d.sideboarding ? 2 : 9;
                return -2;
            }
            if (stage == 2) {
                continued = true;
                sideboarder = seat_of(d.viewer);
                stage = 3;
                return -3;
            }
            return 0;
        });
        const std::string &err = run.proc.err;
        c.check(run.error.empty() && run.proc.ok(), "[3b] between games: the session ended cleanly");
        c.check(continued, "[3b] between games: -2 with no live game was ignored safely (the match went on)");
        c.check(rmcheck::contains(err, "no live game to concede"), "[3b] between games: the engine said so");
        c.check(
            !sideboarder.empty() && rmcheck::contains(err, ("MATCH_RESULT: " + other(sideboarder) + " wins").c_str()),
            "[3b] between games: -3 ended the match, crediting the opponent");
    }
}

// test_concede.py's check 4: the same seed and replies give the same game, the sentinel is logged
// and --replay reproduces the concession.
static void determinism(const rmcheck::Options &opt, ConcedeChecks &c) {
    const std::string dir = rmcheck::make_temp_dir(opt.out("concede"), "determinism_");
    const std::string log = rmcheck::path_join(dir, "game.rmlog");
    std::string conceder;
    const auto reply = [&](const rmcheck::StreamDecision &d) {
        if (d.decision < 13) return 0;
        conceder = seat_of(d.viewer);
        return -2;
    };
    const std::vector<std::string> base = {"--seed", "31337", "--deck-a", "delver", "--deck-b", "mav"};
    std::vector<std::string> logged = base;
    logged.push_back("--decision-log");
    logged.push_back(log);
    const rmcheck::StreamRun a = rmcheck::run_stream(stream_spec(opt, base), reply);
    const rmcheck::StreamRun b = rmcheck::run_stream(stream_spec(opt, logged), reply);
    c.check(a.replies == b.replies, "[4] the same seed produced the same replies");
    c.check(a.payloads == b.payloads, "[4] every frame is byte-identical across the two runs");
    c.check(result_lines(a.proc.err) == result_lines(b.proc.err), "[4] the same result both runs");
    std::string text;
    c.check(rmcheck::read_file(log, &text), "[4] a decision log was written");
    const size_t at = text.find("DECISIONS\n");
    std::vector<std::string> decisions;
    if (at != std::string::npos)
        for (const std::string &l : rmcheck::split_lines(text.substr(at + 10)))
            if (!rmcheck::trim(l).empty()) decisions.push_back(rmcheck::trim(l));
    c.check(!decisions.empty() && decisions.back() == "-2", "[4] the log's last decision is the sentinel (-2)");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--replay", log};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 120;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    c.check(r.ok(), "[4] --replay exited cleanly (" + r.status_text() + ")");
    c.check(
        !conceder.empty() && rmcheck::contains(r.out, (conceder + " concedes - " + other(conceder) + " wins!").c_str()),
        "[4] the replayed game ended on the same concession");
    c.check(!rmcheck::contains(r.out + r.err, "FATAL"), "[4] the replay produced no fatal error");
    rmcheck::remove_tree(dir);
}

static void check_parked_replay(const rmcheck::Options &opt, ConcedeChecks &c, const std::string &label,
    const std::string &log, const std::vector<std::string> &live, const std::string &announce, int sentinel) {
    std::string text;
    if (!rmcheck::read_file(log, &text)) {
        c.check(false, label + ": a decision log was written");
        return;
    }
    const size_t at = text.find("DECISIONS\n");
    bool logged = false;
    if (at != std::string::npos)
        for (const std::string &l : rmcheck::split_lines(text.substr(at + 10)))
            logged = logged || rmcheck::trim(l) == std::to_string(sentinel);
    c.check(logged, label + ": the log holds the sentinel");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--replay", log, "--narrative"};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 120;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    const std::string out = r.out + r.err;
    c.check(r.ok() && !rmcheck::contains(out, "FATAL"), label + ": --replay exited cleanly (" + r.status_text() + ")");
    c.check(result_lines(out) == live, label + ": --replay reproduced the results");
    const size_t conceded = out.find("Input: " + std::to_string(sentinel) + " (concede)");
    c.check(conceded != std::string::npos && out.substr(0, conceded).find(announce) != std::string::npos,
        label + ": --replay conceded at the parked prompt");
}

// test_concede.py's check 7: a concession at a parked query (Brainstorm's put-back, mid-resolution;
// Cavern of Souls' ETB creature type, an SBE-pass prompt) ends the game at once, over the RMSTATE
// stream (driven by its frames) and BQUERY (the same replies piped), and the logs replay to it.
static void parked(const rmcheck::Options &opt, ConcedeChecks &c) {
    const std::string temp = opt.decks_dir() + "/temp";
    rmcheck::make_dirs(temp);
    const std::vector<std::pair<std::string, std::string>> decks = {
        {"rmcheck_concede_brainstorm", "1 Brainstorm\n19 Island\n"},
        {"rmcheck_concede_cavern", "1 Cavern of Souls\n1 Delver of Secrets\n1 Murktide Regent\n17 Island\n"},
        {"rmcheck_concede_lands", "20 Forest\n"}};
    for (const auto &d : decks) rmcheck::write_file(temp + "/" + d.first + ".dk", d.second);
    const ParkedCase cases[] = {
        {"Brainstorm put-back (RESOLUTION)", "rmcheck_concede_brainstorm", "Cast Brainstorm", "",
            "chooses a card from Player A hand", "puts .* to top of library"},
        {"Cavern of Souls creature type (SBE_LATCHED)", "rmcheck_concede_cavern", "Play Cavern of Souls",
            "Choose creature type", "Choose a creature type for Cavern of Souls", "chooses .*creature type"},
    };
    const std::string dir = rmcheck::make_temp_dir(opt.out("concede"), "parked_");
    long seed = 104300;
    for (const ParkedCase &pc : cases)
        for (int line = 0; line < 3; ++line) {
            const int sentinel = line == 2 ? -3 : -2;
            const bool bo3 = line > 0;
            ++seed;
            const std::string label = std::string(pc.name) + ", " + (bo3 ? "bo3 " : "bo1 ") +
                                      (sentinel == -2 ? "CONCEDE_GAME" : "CONCEDE_MATCH");
            std::vector<std::string> flags = {"--seed", std::to_string(seed), "--deck-a",
                std::string("temp/") + pc.deck_a, "--deck-b", "temp/rmcheck_concede_lands", "--no-shuffle",
                "--battlefield-a", "Island"};
            if (bo3) flags.push_back("--bo3");
            // RMSTATE: the replies follow the menus.
            const std::string log = rmcheck::path_join(dir, "rmstate_" + std::to_string(seed) + ".rmlog");
            std::vector<std::string> logged = flags;
            logged.push_back("--decision-log");
            logged.push_back(log);
            std::shared_ptr<ParkedScript> script = std::make_shared<ParkedScript>();
            script->pick = pc.pick;
            script->prompt = pc.prompt;
            script->sentinel = sentinel;
            script->bo3 = bo3;
            const rmcheck::StreamRun run = rmcheck::run_stream(
                stream_spec(opt, logged), [script](const rmcheck::StreamDecision &d) { return script->reply(d); });
            const std::string &text = run.proc.err;
            const std::string tag = label + ", rmstate";
            c.check(run.error.empty() && run.proc.ok(),
                tag + ": engine exit 0 (" + run.error + run.proc.status_text() + ")");
            c.check(run.terminals == 2, tag + ": one terminal frame per seat");
            c.check(!rmcheck::contains(text, "FATAL"), tag + ": no FATAL");
            c.check(script->conceded == (bo3 && sentinel == -2 ? 2 : 1),
                tag + ": the line reached the parked prompt and conceded");
            c.check(rmcheck::contains(text, pc.announce), tag + ": the parked prompt was offered");
            const size_t cut = text.find("concedes");
            std::string after = cut == std::string::npos ? text : text.substr(cut + 8);
            after = after.substr(0, after.find("GAME_RESULT"));
            c.check(!std::regex_search(after, std::regex(pc.continued)),
                tag + ": nothing of the conceded game resolved after the concession");
            std::vector<std::string> kinds;
            for (const std::string &l : result_lines(text)) kinds.push_back(l.substr(0, l.find(':')));
            const std::vector<std::string> results = result_lines(text);
            if (!bo3) {
                c.check(kinds == std::vector<std::string>{"GAME_RESULT"}, tag + ": one GAME_RESULT");
                c.check(rmcheck::contains(text, "GAME_RESULT: 1 Player B wins"), tag + ": the conceder (A) lost");
            } else if (sentinel == -2) {
                c.check(kinds == std::vector<std::string>{"GAME_RESULT", "GAME_RESULT", "MATCH_RESULT"},
                    tag + ": game 1 conceded, game 2 played and conceded, then MATCH_RESULT");
                c.check(
                    rmcheck::contains(text, "GAME_RESULT: 1 Player B wins"), tag + ": game 1 lost by the conceder (A)");
            } else {
                c.check(kinds == std::vector<std::string>{"GAME_RESULT", "MATCH_RESULT"},
                    tag + ": GAME_RESULT then MATCH_RESULT");
                c.check(rmcheck::contains(text, "MATCH_RESULT: Player B wins 0-1"),
                    tag + ": the match concession credited Player B");
            }
            check_parked_replay(opt, c, tag, log, results, pc.announce, sentinel);
            // BQUERY: the same replies piped into --machine.
            const std::string blog = rmcheck::path_join(dir, "bquery_" + std::to_string(seed) + ".rmlog");
            rmcheck::ProcSpec spec;
            spec.argv = {opt.binary, "--machine"};
            spec.argv.insert(spec.argv.end(), flags.begin(), flags.end());
            spec.argv.push_back("--decision-log");
            spec.argv.push_back(blog);
            spec.cwd = opt.bin_dir;
            spec.timeout_s = 300;
            for (int r : run.replies) spec.stdin_data += std::to_string(r) + "\n";
            const rmcheck::ProcResult b = rmcheck::run_process(spec);
            const std::string btext = bquery_text(b.out) + b.err;
            const std::string btag = label + ", bquery";
            c.check(b.ok() && !rmcheck::contains(btext, "FATAL"),
                btag + ": engine exit 0, no FATAL (" + b.status_text() + ")");
            c.check(result_lines(btext) == results, btag + ": the same results as the RMSTATE session");
            check_parked_replay(opt, c, btag, blog, results, pc.announce, sentinel);
        }
    for (const auto &d : decks) std::remove((temp + "/" + d.first + ".dk").c_str());
    rmcheck::remove_tree(dir);
}

namespace rmcheck {

// The concede tier: train/test_concede.py's engine checks (1-4 and 7; its checks 5 and 6 test the
// Python front end's GameDriver) on the engine alone.
void tier_concede(const Options &opt, Report &rep) {
    ConcedeChecks c{&rep};
    natural_lines(opt, c);
    determinism(opt, c);
    parked(opt, c);
    rep.info("concede: " + std::to_string(c.passed) + " check(s) passed");
}

}  // namespace rmcheck
