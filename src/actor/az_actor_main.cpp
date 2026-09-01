// bin/az_actor — in-process AlphaZero actor skeleton (Phase D, milestone M5).
//
// Drives full games through the engine (via game_driver's play_single_game) with
// no stdio BQUERY round-trip: an InputLogger input-provider hook builds the
// bit-exact observation in-process (obs_builder), runs a TorchScript-exported
// AZNet (az_evaluator), and returns the greedy action. `--dump-obs` writes each
// decision's observation to a binary file so train/test_actor_parity.py can prove
// bit-parity with the Python env pipeline.
//
// This binary links the engine objects MINUS obj/main.o (it provides its own
// main) plus the src/actor/* TUs and libtorch. No search yet (M5).

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "az_evaluator.h"
#include "az_mcts.h"
#include "npz_writer.h"
#include "oracle_client.h"
#include "td_targets.h"
#include "classes/deck.h"
#include "classes/match_state.h"
#include "components/zone.h"
#include "error.h"
#include "game_driver.h"
#include "input_logger.h"
#include "obs_builder.h"
#include "search_server.h"

namespace {

struct ActorConfig {
    std::string deck = "delver";
    std::string deck_b;  // empty -> mirror (= deck)
    std::string model;
    // Two-model gate/eval matches (the az_eval promotion gate on the actor):
    // Player B's OWN evaluator source — a local TorchScript (--model-b) or a
    // second central-server socket (--eval-server-b). Empty (default) = both
    // seats share the one seat-A evaluator (pure self-play / parity, exactly
    // as before). Requires a seat-A net source (--model or --eval-server, not
    // --uniform) and is incompatible with --selfplay: gate games measure
    // strength between two DIFFERENT nets, so their decisions are not
    // self-play training data.
    std::string model_b;
    std::string eval_server_b;
    // Eval device for the TorchScript forward: "cpu" (default) or "cuda" (the
    // Radeon under the ROCm build — HIP registers as the cuda backend). Stage A
    // of docs/gpu_selfplay_inference_plan.md; search math is device-independent
    // (priors/value come back to CPU inside AZEvaluator).
    std::string device = "cpu";
    // Stage C: Unix-socket path of a central inference server
    // (train/az_eval_server.py). Mutually exclusive with --model/--uniform;
    // --device is inert with it (the server owns the device).
    std::string eval_server;
    std::string dump_obs;
    std::string dump_visits;
    std::string resources;  // empty -> <cwd>/resources
    unsigned int seed = 1;
    int games = 1;
    bool bo3 = false;  // --bo3: each of `games` units is a best-of-three MATCH
    bool uniform = false;
    // MCTS (--search) config.
    bool search = false;
    int sims = 128;
    int worlds = 4;
    double c_puct = 1.5;
    int batch = 1;
    // Cross-world batched leaf evaluation (see az_mcts.h). Mutually exclusive
    // with --batch K>1.
    bool cross_world = false;
    uint32_t world_seeds = 42;  // --world-seeds base (see az_mcts.h seed formula)
    // bo3 sideboard plan-search budget (mirrors az_selfplay.py / az_mcts.h).
    // -1 = the compiled defaults (kDefaultSb* in az_mcts.cpp).
    int sb_branches = -1;
    int sb_worlds = -1;
    int sb_rollout_turns = -1;
    // Prior-rollout sideboard self-play (see az_mcts.h). Default OFF ('plan')
    // so gate/eval/parity callers keep the plan search unless az_selfplay
    // explicitly passes --sb-selfplay-mode prior for generation.
    bool sb_prior_mode = false;
    double sb_explore_temp = 1.0;
    double sb_explore_eps = 0.10;
    // In-game leaf-rollout horizon in player turns (see az_mcts.h). 0 = off.
    int rollout_turns = 0;
    // Duplicate-edge merging (see az_mcts.h / menu_merge.h; must match the
    // Python side's merge_dupes or visit parity breaks).
    int merge_dupes = 1;
    // Self-play (--selfplay, implies --search) config.
    bool selfplay = false;
    // Playout-cap randomization (self-play only; see az_mcts.h). Compiled
    // defaults mirror cli_spec's DEFAULT_AZ_FULL_SEARCH_FRAC /
    // DEFAULT_AZ_FAST_SIMS; az_selfplay always passes both flags explicitly.
    double full_search_frac = 0.25;
    int fast_sims = 128;
    // --record: write trainer-schema shards of the searched decisions WITHOUT
    // the self-play exploration knobs (no root noise, argmax picks) — the
    // eval/gate recorder. Allowed with a seat-B evaluator (a two-model gate's
    // samples are priced per game like self-play: z from the mover's seat).
    bool record = false;
    double noise_eps = 0.25;
    double noise_alpha = 1.0;
    int temp_moves = 20;
    // n-step TD horizon for the recorded value target (mirrors
    // cli_spec.DEFAULT_AZ_TD_N; az_selfplay always passes --td-n explicitly).
    int td_n = 10;
    std::string out_dir;               // empty -> ../train/az_data/<deck>
    bool rng_seed_set = false;
    uint32_t rng_seed = 0;             // default derived from --seed
    // vs-scripted seat (--scripted-seat A|B + --scripted-oracle <socket>):
    // that seat's real decisions come from the scripted-agent oracle
    // (train/scripted_oracle.py); the OTHER seat searches as usual. 0 = none.
    int scripted_seat = 0;             // 0 none, 1 = Player A, 2 = Player B
    std::string scripted_oracle;       // Unix socket path
    // Opponent-pool self-play (--net-seat A|B + --opp-model <path.ts.pt>):
    // the primary evaluator (--model/--eval-server) pilots and RECORDS the
    // net seat while --opp-model (an OLDER checkpoint's TorchScript, loaded
    // locally) pilots the other noise-free/argmax — the C++ twin of
    // az_selfplay._play_match's opp_evaluator/net_is_a mode. Requires
    // --selfplay; mutually exclusive with --scripted-seat and --model-b/
    // --eval-server-b. 0 = none.
    int net_seat = 0;                  // 0 none, 1 = Player A, 2 = Player B
    std::string opp_model;             // opponent checkpoint TorchScript path
};

// Shard flush threshold — mirrors az_selfplay.py::FLUSH_SAMPLES.
constexpr size_t FLUSH_SAMPLES = 4096;

void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --deck <name> [--deck-b <name>] [--seed N] [--games N] "
                 "[--bo3] [--model <path.ts.pt> | --uniform | --eval-server <socket>] "
                 "[--device cpu|cuda] [--dump-obs <file>]\n"
                 "       [--model-b <path.ts.pt> | --eval-server-b <socket>] "
                 "(seat-B evaluator for two-model gate/eval matches; "
                 "incompatible with --selfplay/--uniform)\n"
                 "       [--search [--sims N] [--worlds N] [--c F] [--batch K | "
                 "--cross-world] [--world-seeds BASE] [--dump-visits <file>]] "
                 "[--resources <dir>]\n"
                 "       [--sb-branches N] [--sb-worlds N] [--sb-rollout-turns N] "
                 "(bo3 sideboard plan-search budget; -1=compiled defaults)\n"
                 "       [--sb-selfplay-mode prior|plan] [--sb-explore-temp F] "
                 "[--sb-explore-eps F] (prior-rollout sideboard self-play; "
                 "default plan)\n"
                 "       [--rollout-turns N] "
                 "(in-game leaf-rollout horizon in player turns; 0=off)\n"
                 "       [--merge-dupes 0|1] (merge interchangeable duplicate menu "
                 "actions into one search edge; default 1)\n"
                 "       [--selfplay [--noise-eps F] [--noise-alpha F] "
                 "[--temp-moves N] [--td-n N] [--out-dir <dir>] [--rng-seed N] "
                 "[--full-search-frac F] [--fast-sims N]]\n"
                 "       (playout cap, self-play only: P(full --sims search, "
                 "pi recorded) per in-game root; the rest run --fast-sims "
                 "with no pi. 1.0 = every root full)\n"
                 "       [--record [--out-dir <dir>] [--td-n N]] (eval-mode shard "
                 "recording: searched roots stored, no noise, argmax picks; "
                 "allowed with --model-b)\n"
                 "       [--scripted-seat A|B --scripted-oracle <socket>] "
                 "(vs-scripted: that seat plays via train/scripted_oracle.py)\n"
                 "       [--net-seat A|B --opp-model <path.ts.pt>] "
                 "(opponent-pool self-play: --model/--eval-server pilots and "
                 "records the net seat, --opp-model pilots the other "
                 "noise-free/argmax; requires --selfplay)\n",
                 prog);
}

const char* need_arg(int argc, char const* argv[], int& i, const char* flag) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "error: %s requires an argument\n", flag);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char const* argv[]) {
#ifndef NDEBUG
    // Debug builds carry the az_mcts world-consistency menu instrumentation
    // (per-node menu capture + full-content compare on every descent step),
    // which costs search throughput. Say so up front, so a slow production run
    // is traceable to the build type. Build with `make actor BUILD=RELEASE`.
    std::fprintf(stderr,
                 "az_actor: DEBUG build (world-consistency menu instrumentation "
                 "active; searches run slower) — use `make actor BUILD=RELEASE` "
                 "for production self-play\n");
#endif
    ActorConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--deck") {
            cfg.deck = need_arg(argc, argv, i, "--deck");
        } else if (a == "--deck-b") {
            cfg.deck_b = need_arg(argc, argv, i, "--deck-b");
        } else if (a == "--seed") {
            cfg.seed = static_cast<unsigned int>(std::stoul(need_arg(argc, argv, i, "--seed")));
        } else if (a == "--games") {
            cfg.games = std::stoi(need_arg(argc, argv, i, "--games"));
        } else if (a == "--bo3") {
            cfg.bo3 = true;
        } else if (a == "--model") {
            cfg.model = need_arg(argc, argv, i, "--model");
        } else if (a == "--model-b") {
            cfg.model_b = need_arg(argc, argv, i, "--model-b");
        } else if (a == "--eval-server-b") {
            cfg.eval_server_b = need_arg(argc, argv, i, "--eval-server-b");
        } else if (a == "--device") {
            cfg.device = need_arg(argc, argv, i, "--device");
        } else if (a == "--eval-server") {
            cfg.eval_server = need_arg(argc, argv, i, "--eval-server");
        } else if (a == "--uniform") {
            cfg.uniform = true;
        } else if (a == "--dump-obs") {
            cfg.dump_obs = need_arg(argc, argv, i, "--dump-obs");
        } else if (a == "--dump-visits") {
            cfg.dump_visits = need_arg(argc, argv, i, "--dump-visits");
        } else if (a == "--search") {
            cfg.search = true;
        } else if (a == "--sims") {
            cfg.sims = std::stoi(need_arg(argc, argv, i, "--sims"));
        } else if (a == "--worlds") {
            cfg.worlds = std::stoi(need_arg(argc, argv, i, "--worlds"));
        } else if (a == "--c") {
            cfg.c_puct = std::stod(need_arg(argc, argv, i, "--c"));
        } else if (a == "--batch") {
            cfg.batch = std::stoi(need_arg(argc, argv, i, "--batch"));
        } else if (a == "--cross-world") {
            cfg.cross_world = true;
        } else if (a == "--world-seeds") {
            cfg.world_seeds = static_cast<uint32_t>(
                std::stoul(need_arg(argc, argv, i, "--world-seeds")));
        } else if (a == "--sb-branches") {
            cfg.sb_branches = std::stoi(need_arg(argc, argv, i, "--sb-branches"));
        } else if (a == "--sb-worlds") {
            cfg.sb_worlds = std::stoi(need_arg(argc, argv, i, "--sb-worlds"));
        } else if (a == "--rollout-turns") {
            cfg.rollout_turns = std::stoi(need_arg(argc, argv, i, "--rollout-turns"));
        } else if (a == "--sb-rollout-turns") {
            cfg.sb_rollout_turns = std::stoi(need_arg(argc, argv, i, "--sb-rollout-turns"));
        } else if (a == "--sb-selfplay-mode") {
            const std::string m = need_arg(argc, argv, i, "--sb-selfplay-mode");
            if (m != "prior" && m != "plan") {
                std::fprintf(stderr,
                             "error: --sb-selfplay-mode takes prior or plan\n");
                std::exit(2);
            }
            cfg.sb_prior_mode = (m == "prior");
        } else if (a == "--sb-explore-temp") {
            cfg.sb_explore_temp =
                std::stod(need_arg(argc, argv, i, "--sb-explore-temp"));
        } else if (a == "--sb-explore-eps") {
            cfg.sb_explore_eps =
                std::stod(need_arg(argc, argv, i, "--sb-explore-eps"));
        } else if (a == "--merge-dupes") {
            cfg.merge_dupes = std::stoi(need_arg(argc, argv, i, "--merge-dupes"));
        } else if (a == "--selfplay") {
            cfg.selfplay = true;
            cfg.search = true;  // self-play implies MCTS search
        } else if (a == "--record") {
            cfg.record = true;
            cfg.search = true;  // only searched roots carry a sample
        } else if (a == "--full-search-frac") {
            cfg.full_search_frac =
                std::stod(need_arg(argc, argv, i, "--full-search-frac"));
        } else if (a == "--fast-sims") {
            cfg.fast_sims = std::stoi(need_arg(argc, argv, i, "--fast-sims"));
        } else if (a == "--noise-eps") {
            cfg.noise_eps = std::stod(need_arg(argc, argv, i, "--noise-eps"));
        } else if (a == "--noise-alpha") {
            cfg.noise_alpha = std::stod(need_arg(argc, argv, i, "--noise-alpha"));
        } else if (a == "--temp-moves") {
            cfg.temp_moves = std::stoi(need_arg(argc, argv, i, "--temp-moves"));
        } else if (a == "--td-n") {
            cfg.td_n = std::stoi(need_arg(argc, argv, i, "--td-n"));
        } else if (a == "--out-dir") {
            cfg.out_dir = need_arg(argc, argv, i, "--out-dir");
        } else if (a == "--rng-seed") {
            cfg.rng_seed = static_cast<uint32_t>(
                std::stoul(need_arg(argc, argv, i, "--rng-seed")));
            cfg.rng_seed_set = true;
        } else if (a == "--net-seat") {
            std::string s = need_arg(argc, argv, i, "--net-seat");
            if (s == "A" || s == "a") {
                cfg.net_seat = 1;
            } else if (s == "B" || s == "b") {
                cfg.net_seat = 2;
            } else {
                std::fprintf(stderr, "error: --net-seat takes A or B\n");
                return 2;
            }
        } else if (a == "--opp-model") {
            cfg.opp_model = need_arg(argc, argv, i, "--opp-model");
        } else if (a == "--scripted-seat") {
            std::string s = need_arg(argc, argv, i, "--scripted-seat");
            if (s == "A" || s == "a") {
                cfg.scripted_seat = 1;
            } else if (s == "B" || s == "b") {
                cfg.scripted_seat = 2;
            } else {
                std::fprintf(stderr, "error: --scripted-seat takes A or B\n");
                return 2;
            }
        } else if (a == "--scripted-oracle") {
            cfg.scripted_oracle = need_arg(argc, argv, i, "--scripted-oracle");
        } else if (a == "--resources") {
            cfg.resources = need_arg(argc, argv, i, "--resources");
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if ((!cfg.uniform && cfg.model.empty() && cfg.eval_server.empty()) ||
        (cfg.uniform && !cfg.model.empty()) ||
        (!cfg.eval_server.empty() && (cfg.uniform || !cfg.model.empty()))) {
        std::fprintf(stderr, "error: exactly one of --model <path.ts.pt>, --uniform, "
                             "or --eval-server <socket> is required\n");
        print_usage(argv[0]);
        return 2;
    }
    const bool have_b = !cfg.model_b.empty() || !cfg.eval_server_b.empty();
    if (have_b) {
        if (!cfg.model_b.empty() && !cfg.eval_server_b.empty()) {
            std::fprintf(stderr, "error: --model-b and --eval-server-b are mutually "
                                 "exclusive (one seat-B evaluator source)\n");
            return 2;
        }
        if (cfg.uniform) {
            std::fprintf(stderr, "error: a seat-B evaluator (--model-b/"
                                 "--eval-server-b) requires a seat-A net source "
                                 "(--model or --eval-server), not --uniform\n");
            return 2;
        }
        if (cfg.selfplay) {
            std::fprintf(stderr, "error: --selfplay is incompatible with a seat-B "
                                 "evaluator: two-model games are gate/eval matches, "
                                 "not self-play training data\n");
            return 2;
        }
    }

    if (cfg.cross_world && cfg.batch > 1) {
        std::fprintf(stderr, "error: --cross-world and --batch K>1 are mutually "
                             "exclusive (two different leaf-deferral disciplines)\n");
        return 2;
    }

    if ((cfg.scripted_seat != 0) != !cfg.scripted_oracle.empty()) {
        std::fprintf(stderr, "error: --scripted-seat and --scripted-oracle must "
                             "be given together\n");
        return 2;
    }
    if (cfg.scripted_seat != 0 && !cfg.search) {
        std::fprintf(stderr, "error: --scripted-seat requires --search (the "
                             "other seat is the searching net seat)\n");
        return 2;
    }

    if ((cfg.net_seat != 0) != !cfg.opp_model.empty()) {
        std::fprintf(stderr, "error: --net-seat and --opp-model must be given "
                             "together\n");
        return 2;
    }
    if (cfg.net_seat != 0) {
        if (!cfg.selfplay) {
            std::fprintf(stderr, "error: --net-seat/--opp-model is the "
                                 "opponent-pool SELF-PLAY mode (requires "
                                 "--selfplay)\n");
            return 2;
        }
        if (cfg.scripted_seat != 0) {
            std::fprintf(stderr, "error: a match has one opponent — "
                                 "--net-seat/--opp-model and --scripted-seat "
                                 "are mutually exclusive\n");
            return 2;
        }
        if (have_b) {
            std::fprintf(stderr, "error: --opp-model and --model-b/"
                                 "--eval-server-b are mutually exclusive (the "
                                 "opponent-pool net is wired by --net-seat)\n");
            return 2;
        }
        if (cfg.uniform) {
            std::fprintf(stderr, "error: --net-seat requires a real net source "
                                 "(--model or --eval-server), not --uniform\n");
            return 2;
        }
    }

    // RESOURCE_DIR mirrors robomage's convention: getcwd()/resources (the binary
    // is run from bin/), overridable with --resources.
    if (!cfg.resources.empty()) {
        RESOURCE_DIR = cfg.resources;
    } else {
        char buf[FILENAME_MAX];
        RESOURCE_DIR = getcwd(buf, FILENAME_MAX);
        RESOURCE_DIR += "/resources";
    }

    // Machine mode routes every decision through the input-provider hook; keep
    // narrative + search off. The global flag gates play_single_game's populate
    // skips; init_machine flips the InputLogger's own machine flag so get_input
    // takes the machine (provider) branch rather than the interactive CLI loop.
    // enable_log=false: no decision log (millions of decisions).
    machine_mode = true;
    narrative_mode = false;
    // Opt-in debugging: ROBOMAGE_ACTOR_NARRATIVE=1 prints the full game_log
    // narrative (normally suppressed under the machine schedule) for every
    // line INCLUDING search simulations — the transition story right before a
    // world-consistency FATAL names the exact divergent event. Output is huge;
    // use only on a pinned single-matchup repro.
    if (std::getenv("ROBOMAGE_ACTOR_NARRATIVE")) narrative_mode = true;
    // deck_b defaults to deck (mirror); a distinct --deck-b drives a cross-deck
    // self-play matchup (the generalist net values every state, so cross-deck
    // negamax backup in MCTS is sound).
    const std::string deck_b_name_eff = cfg.deck_b.empty() ? cfg.deck : cfg.deck_b;
    deck_a_name = cfg.deck;
    deck_b_name = deck_b_name_eff;
    InputLogger::instance().init_machine(cfg.seed, RESOURCE_DIR, false, DecisionLogHeader{});

    AZEvaluator evaluator;
    if (!cfg.eval_server.empty())
        evaluator.connect_server(cfg.eval_server);
    else if (!cfg.uniform)
        evaluator.load(cfg.model, cfg.device);
    // Seat-B evaluator (two-model gate/eval matches): its own local module or
    // its own server connection; null pointer = single-model (both seats on
    // `evaluator`).
    AZEvaluator evaluator_b;
    AZEvaluator* eval_a_ptr = cfg.uniform ? nullptr : &evaluator;
    AZEvaluator* eval_b_ptr = nullptr;
    if (!cfg.eval_server_b.empty()) {
        evaluator_b.connect_server(cfg.eval_server_b);
        eval_b_ptr = &evaluator_b;
    } else if (!cfg.model_b.empty()) {
        evaluator_b.load(cfg.model_b, cfg.device);
        eval_b_ptr = &evaluator_b;
    } else if (!cfg.opp_model.empty()) {
        // Opponent-pool self-play: the older checkpoint always loads locally
        // (the central eval server, when any, serves the learner). Wire the
        // two evaluators BY SEAT so az_mcts's per-decision seat selection
        // (eval_a = Player A, eval_b = Player B) needs no special case: the
        // learner takes its --net-seat seat, the opponent the other.
        evaluator_b.load(cfg.opp_model, "cpu");
        if (cfg.net_seat == 1) {
            eval_b_ptr = &evaluator_b;
        } else {
            eval_b_ptr = eval_a_ptr;
            eval_a_ptr = &evaluator_b;
        }
    }

    // Optional binary obs dump: per decision, int32 num_choices then
    // ACTOR_OBS_SIZE float32s (little-endian, raw append).
    FILE* dump = nullptr;
    if (!cfg.dump_obs.empty()) {
        dump = std::fopen(cfg.dump_obs.c_str(), "wb");
        if (!dump) fatal_error("az_actor: cannot open --dump-obs file: " + cfg.dump_obs);
    }

    // MCTS (--search): the in-process PUCT search drives every decision through
    // its state machine (real moves, simulation steps, restore-unwinds). The
    // game-end hook backs up terminal simulation lines. Without --search the
    // legacy greedy argmax path (M5) is used.
    std::unique_ptr<AZMcts> mcts;
    if (cfg.search) {
        MCTSConfig mc;
        mc.sims = cfg.sims;
        mc.worlds = cfg.worlds;
        mc.c_puct = cfg.c_puct;
        mc.batch = cfg.batch;
        mc.cross_world = cfg.cross_world;
        mc.world_seed_base = cfg.world_seeds;
        mc.sb_branches = cfg.sb_branches;      // -1 = compiled default
        mc.sb_worlds = cfg.sb_worlds;          // -1 = compiled default
        mc.rollout_turns = cfg.rollout_turns;
        mc.sb_rollout_turns = cfg.sb_rollout_turns;  // -1 = compiled default
        mc.sb_prior_mode = cfg.sb_prior_mode;
        mc.sb_explore_temp = cfg.sb_explore_temp;
        mc.sb_explore_eps = cfg.sb_explore_eps;
        mc.merge_dupes = cfg.merge_dupes != 0;
        mc.selfplay = cfg.selfplay;
        mc.record = cfg.record;
        mc.noise_eps = cfg.selfplay ? cfg.noise_eps : 0.0;  // never leak into parity
        // Like noise_eps: the playout cap never leaks into parity/record runs.
        mc.full_search_frac = cfg.selfplay ? cfg.full_search_frac : 1.0;
        mc.fast_sims = cfg.fast_sims;
        mc.noise_alpha = cfg.noise_alpha;
        mc.temp_moves = cfg.temp_moves;
        mc.selfplay_rng_seed = cfg.rng_seed_set ? cfg.rng_seed : cfg.seed;
        mc.scripted_seat = cfg.scripted_seat;
        mc.net_seat = cfg.net_seat;
        mcts = std::make_unique<AZMcts>(mc, eval_a_ptr, eval_b_ptr);
        search_set_game_end_hook([&](int winner) { return mcts->on_game_end(winner); });
    }

    // vs-scripted: connect to the scripted-agent oracle and route the scripted
    // seat's real decisions to it (see az_mcts.h's scripted_seat contract).
    std::unique_ptr<ScriptedOracle> oracle;
    if (cfg.scripted_seat != 0) {
        oracle = std::make_unique<ScriptedOracle>();
        oracle->connect(cfg.scripted_oracle, cfg.deck, deck_b_name_eff,
                        "scripted:hard");
        mcts->set_scripted_provider(
            [&oracle](const float* o, int nc) { return oracle->act(o, nc); });
    }

    // Self-play shard accumulator: default out-dir ../train/az_data/<deck> (the
    // binary runs from bin/, so this resolves to the repo's train/az_data/<deck>,
    // matching az_selfplay.py's layout and the trainer's shard_*.npz glob).
    const bool recording = cfg.selfplay || cfg.record;
    std::unique_ptr<ShardAccumulator> shards;
    if (recording) {
        std::string dir = cfg.out_dir.empty()
                              ? ("../train/az_data/" + cfg.deck)
                              : cfg.out_dir;
        shards = std::make_unique<ShardAccumulator>(
            dir, FLUSH_SAMPLES, static_cast<size_t>(ACTOR_OBS_SIZE),
            static_cast<size_t>(MAX_ACTIONS));
        std::fprintf(stderr,
                     "az_actor: %s out_dir=%s sims=%d worlds=%d "
                     "noise_eps=%.3f noise_alpha=%.3f temp_moves=%d td_n=%d "
                     "full_search_frac=%.3f fast_sims=%d\n",
                     cfg.selfplay ? "self-play" : "record (eval-mode)",
                     dir.c_str(), cfg.sims, cfg.worlds,
                     cfg.selfplay ? cfg.noise_eps : 0.0, cfg.noise_alpha,
                     cfg.selfplay ? cfg.temp_moves : 0, cfg.td_n,
                     cfg.selfplay ? cfg.full_search_frac : 1.0,
                     cfg.fast_sims);
    }

    InputLogger::instance().set_input_provider(
        [&](const std::vector<LegalAction>& actions) -> int {
            if (dump) {
                ActorObs ob = build_obs(actions);
                int32_t nc = static_cast<int32_t>(ob.num_choices);
                std::fwrite(&nc, sizeof(int32_t), 1, dump);
                std::fwrite(ob.obs.data(), sizeof(float),
                            static_cast<size_t>(ACTOR_OBS_SIZE), dump);
            }
            if (mcts) return mcts->on_decision(actions);
            ActorObs ob = build_obs(actions);
            if (cfg.uniform) return 0;
            // Greedy path: same per-seat selection as the search path — seat B's
            // decisions use its own net when one was given.
            AZEvaluator& e =
                (eval_b_ptr != nullptr && ob.obs[ACTOR_SELF_IS_A_IDX] <= 0.5f)
                    ? evaluator_b
                    : evaluator;
            return e.argmax_action(ob.obs.data(), ob.num_choices);
        });

    // Player A plays --deck; Player B plays --deck-b (defaults to --deck: mirror).
    Deck deck_a(RESOURCE_DIR + "/decks/" + cfg.deck + ".dk");
    Deck deck_b(RESOURCE_DIR + "/decks/" + deck_b_name_eff + ".dk");

    // Per-game self-play sample backfill: price each stored sample from its
    // mover's perspective vs the GAME's winner (draw -> 0), accumulate, then flush
    // at the GAME boundary so a game is never split across shards (mirrors
    // az_selfplay.py's per-game pricing — in bo3 each sample carries the result of
    // the particular game it belonged to, not the match). `game_log_idx` numbers
    // the SELFPLAY lines across all games of the run. Returns the sample count.
    int game_log_idx = 0;
    auto backfill_selfplay = [&](int winner) {
        if (!recording) return;
        bool draw = winner != static_cast<int>(Zone::PLAYER_A) &&
                    winner != static_cast<int>(Zone::PLAYER_B);
        bool winner_is_a = winner == static_cast<int>(Zone::PLAYER_A);
        const std::vector<SelfPlaySample>& gs = mcts->game_samples();
        // Price each sample, then derive the n-step TD targets over the WHOLE
        // game's sample stream at once (the chain never leaves this buffer:
        // it holds exactly one game plus any sideboard roots that gated it,
        // mirroring az_selfplay.py's per-(game, sideboard) chains).
        std::vector<TdSample> td_in(gs.size());
        for (size_t i = 0; i < gs.size(); i++) {
            const SelfPlaySample& s = gs[i];
            td_in[i].z = draw ? 0.0f : (s.mover_is_a == winner_is_a ? 1.0f : -1.0f);
            td_in[i].q = s.q;
            td_in[i].explored = s.explored;
            td_in[i].is_sideboard = s.is_sideboard;
            td_in[i].mover_is_a = s.mover_is_a;
        }
        std::vector<float> td_q;
        compute_td_targets(td_in, cfg.td_n, td_q);
        for (size_t i = 0; i < gs.size(); i++) {
            const SelfPlaySample& s = gs[i];
            shards->add_sample(s.obs.data(), s.pi.data(), td_in[i].z,
                               s.mask.data(), s.q,
                               static_cast<uint8_t>(s.explored ? 1 : 0), td_q[i]);
        }
        shards->maybe_flush();
        const char* wstr = draw ? "DRAW" : (winner_is_a ? "A" : "B");
        std::printf("SELFPLAY: game %d samples=%zu winner=%s\n", ++game_log_idx,
                    gs.size(), wstr);
        std::fflush(stdout);
        // End-of-game reset AFTER pricing+flushing: clears the buffer and resets
        // the tau counter at the game boundary. Any bo3 sideboard samples recorded
        // between now and the next game's start accumulate into the now-empty
        // buffer and are priced by the NEXT game's z (mirrors az_selfplay.py, where
        // a sideboard sample carries game_idx == k+1 and game_move restarts at 0).
        mcts->end_game();
    };

    if (cfg.bo3) {
        // Each of `--games` units is a best-of-three MATCH. Space match seeds by 3
        // (a match uses seeds match_seed + {0,1,2}) so games never share a seed
        // across matches. play_bo3_match owns the exact sequencing main.cpp uses
        // (game boundaries, loser-on-the-play, revealed accumulator, sideboarding),
        // with per-game hooks. The sample buffer + tau counter reset ONCE per match
        // here (begin_match) and again at each game boundary inside the after-game
        // hook (backfill_selfplay -> end_game, AFTER pricing that game). before_game
        // no longer clears anything, so sideboard samples recorded between a game's
        // backfill and the next game's start stay buffered and are priced by the
        // NEXT game (matching Python's game_idx == k+1 sideboard samples + the
        // game_move reset at the boundary).
        for (int m = 0; m < cfg.games; m++) {
            unsigned int match_seed = cfg.seed + static_cast<unsigned int>(m) * 3u;
            std::srand(match_seed);
            if (recording) mcts->begin_match(match_seed);
            // agent.new_game() at match start + after every completed game —
            // the exact call sites az_selfplay._play_match uses (reset + each
            // GAME_RESULT boundary, before the next game's sideboard prompts).
            if (oracle) oracle->new_game();
            play_bo3_match(
                deck_a, deck_b, match_seed,
                [&](int, bool) {},
                [&](int, int winner) {
                    backfill_selfplay(winner);
                    if (oracle) oracle->new_game();
                });
        }
    } else {
        for (int g = 0; g < cfg.games; g++) {
            unsigned int seed_g = cfg.seed + static_cast<unsigned int>(g);
            // Mirror main.cpp's single-game setup: srand(seed), reset the
            // match-scoped revealed accumulator, fresh ECS, then Player A on the play.
            std::srand(seed_g);
            match_reset_revealed();
            EcsSystems sys = init_ecs();
            // Reset per-game move counter + samples; seed_g keys the cap coin
            // (a bo1 match is one game, matching the Python per-match seed).
            if (recording) mcts->begin_match(seed_g);
            if (oracle) oracle->new_game();
            int winner = play_single_game(sys, deck_a, deck_b, true, seed_g);
            bool draw = winner != static_cast<int>(Zone::PLAYER_A) &&
                        winner != static_cast<int>(Zone::PLAYER_B);
            if (draw)
                std::printf("GAME_RESULT: %d draw\n", g + 1);
            else
                std::printf("GAME_RESULT: %d Player %s wins\n", g + 1,
                            winner == Zone::PLAYER_A ? "A" : "B");
            std::fflush(stdout);
            backfill_selfplay(winner);
        }
    }

    if (recording) {
        shards->flush_final();
        std::printf("SELFPLAY: total_samples=%zu shards=%zu\n",
                    shards->total_samples(), shards->shards_written());
        std::fflush(stdout);
    }

    InputLogger::instance().clear_input_provider();
    if (dump) std::fclose(dump);

    // --dump-visits: per searched root, int32 root_index, int32 num_choices,
    // then the root's record. A TREE root writes int64[num_choices] summed
    // visit counts; a PLAN (sideboard) root writes num_choices as its NEGATIVE
    // (the record-type tag) followed by float64 q[num_choices] and
    // pi[num_choices]. Little-endian throughout; the parity test reads this
    // and asserts an exact match against the Python run_search /
    // run_plan_search results.
    if (cfg.search && !cfg.dump_visits.empty()) {
        search_clear_game_end_hook();
        FILE* vf = std::fopen(cfg.dump_visits.c_str(), "wb");
        if (!vf) fatal_error("az_actor: cannot open --dump-visits file: " + cfg.dump_visits);
        for (const SearchRootResult& r : mcts->results()) {
            int32_t ri = static_cast<int32_t>(r.root_index);
            int32_t nc = static_cast<int32_t>(r.num_choices);
            std::fwrite(&ri, sizeof(int32_t), 1, vf);
            if (r.plan_root) {
                int32_t tag = -nc;
                std::fwrite(&tag, sizeof(int32_t), 1, vf);
                std::fwrite(r.q.data(), sizeof(double),
                            static_cast<size_t>(r.num_choices), vf);
                std::fwrite(r.pi.data(), sizeof(double),
                            static_cast<size_t>(r.num_choices), vf);
            } else {
                std::fwrite(&nc, sizeof(int32_t), 1, vf);
                std::fwrite(r.visits.data(), sizeof(int64_t),
                            static_cast<size_t>(r.num_choices), vf);
            }
        }
        std::fclose(vf);
        std::fprintf(stderr, "az_actor: dumped %zu searched roots to %s\n",
                     mcts->results().size(), cfg.dump_visits.c_str());
    } else if (cfg.search) {
        search_clear_game_end_hook();
    }
    return 0;
}
