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
    std::string model;
    std::string dump_obs;
    std::string dump_visits;
    std::string resources;  // empty -> <cwd>/resources
    unsigned int seed = 1;
    int games = 1;
    bool uniform = false;
    // MCTS (--search) config.
    bool search = false;
    int sims = 128;
    int worlds = 4;
    double c_puct = 1.5;
    int batch = 1;
    uint32_t world_seeds = 42;  // --world-seeds base (see az_mcts.h seed formula)
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s --deck <name> [--seed N] [--games N] "
                 "[--model <path.ts.pt> | --uniform] [--dump-obs <file>]\n"
                 "       [--search [--sims N] [--worlds N] [--c F] [--batch K] "
                 "[--world-seeds BASE] [--dump-visits <file>]] [--resources <dir>]\n",
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
    ActorConfig cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--deck") {
            cfg.deck = need_arg(argc, argv, i, "--deck");
        } else if (a == "--seed") {
            cfg.seed = static_cast<unsigned int>(std::stoul(need_arg(argc, argv, i, "--seed")));
        } else if (a == "--games") {
            cfg.games = std::stoi(need_arg(argc, argv, i, "--games"));
        } else if (a == "--model") {
            cfg.model = need_arg(argc, argv, i, "--model");
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
        } else if (a == "--world-seeds") {
            cfg.world_seeds = static_cast<uint32_t>(
                std::stoul(need_arg(argc, argv, i, "--world-seeds")));
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

    if (!cfg.uniform && cfg.model.empty()) {
        std::fprintf(stderr, "error: one of --model <path.ts.pt> or --uniform is required\n");
        print_usage(argv[0]);
        return 2;
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
    deck_a_name = cfg.deck;
    deck_b_name = cfg.deck;
    InputLogger::instance().init_machine(cfg.seed, RESOURCE_DIR, false, DecisionLogHeader{});

    AZEvaluator evaluator;
    if (!cfg.uniform) evaluator.load(cfg.model);

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
        mc.world_seed_base = cfg.world_seeds;
        mcts = std::make_unique<AZMcts>(mc, cfg.uniform ? nullptr : &evaluator);
        search_set_game_end_hook([&](int winner) { return mcts->on_game_end(winner); });
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
            return evaluator.argmax_action(ob.obs.data(), ob.num_choices);
        });

    // Mirror match: both seats play the same deck.
    Deck deck(RESOURCE_DIR + "/decks/" + cfg.deck + ".dk");

    for (int g = 0; g < cfg.games; g++) {
        unsigned int seed_g = cfg.seed + static_cast<unsigned int>(g);
        // Mirror main.cpp's single-game setup: srand(seed), reset the match-scoped
        // revealed accumulator, fresh ECS, then play Player A on the play.
        std::srand(seed_g);
        match_reset_revealed();
        EcsSystems sys = init_ecs();
        int winner = play_single_game(sys, deck, deck, true, seed_g);
        std::printf("GAME_RESULT: %d Player %s wins\n", g + 1,
                    winner == Zone::PLAYER_A ? "A" : "B");
        std::fflush(stdout);
    }

    InputLogger::instance().clear_input_provider();
    if (dump) std::fclose(dump);

    // --dump-visits: per searched root, int32 root_index, int32 num_choices,
    // int64[num_choices] summed visit counts (little-endian). The parity test
    // reads this and asserts an exact match against the Python run_search visits.
    if (cfg.search && !cfg.dump_visits.empty()) {
        search_clear_game_end_hook();
        FILE* vf = std::fopen(cfg.dump_visits.c_str(), "wb");
        if (!vf) fatal_error("az_actor: cannot open --dump-visits file: " + cfg.dump_visits);
        for (const SearchRootResult& r : mcts->results()) {
            int32_t ri = static_cast<int32_t>(r.root_index);
            int32_t nc = static_cast<int32_t>(r.num_choices);
            std::fwrite(&ri, sizeof(int32_t), 1, vf);
            std::fwrite(&nc, sizeof(int32_t), 1, vf);
            std::fwrite(r.visits.data(), sizeof(int64_t),
                        static_cast<size_t>(r.num_choices), vf);
        }
        std::fclose(vf);
        std::fprintf(stderr, "az_actor: dumped %zu searched roots to %s\n",
                     mcts->results().size(), cfg.dump_visits.c_str());
    } else if (cfg.search) {
        search_clear_game_end_hook();
    }
    return 0;
}
