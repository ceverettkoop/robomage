#include "driver.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "../cli/human.h"
#include "../error.h"
#include "../featurize/featurizer.h"
#include "../game_driver.h"
#include "../input_logger.h"
#include "../state_format/agent_feed.h"
#include "../state_format/cards_file.h"
#include "../state_format/stream.h"
#include "agent.h"
#include "play_spec.h"
#include "trace.h"

namespace {

// Everything the hook keeps across the run.
struct AgentRun {
        agents::PyRandom rng;
        std::unique_ptr<agents::Agent> seat_agent[2];
        std::unique_ptr<agents::Agent> human[2];  // a human seat's CLI (cli/human.h)
        std::unique_ptr<agents::Agent> first[2];  // a seat's player under a script when it has no agent
        std::unique_ptr<agents::Agent> script;    // --play / --actions
        agents::PlayScript *play = nullptr;       // `script` when it is a --play script
        bool decides[2] = {false, false};
        agents::SeatStream streams[2];
        // The current decision, as built by agents_begin_decision.
        int seat = -1;
        size_t number = 0;
        rmstate::DecodedFrame frame;
        std::optional<agents::DecisionView> view;
        bool traced = false;
        // --featurize-seats: the inputs, the last tensors and the totals.
        std::unique_ptr<featurize::Featurization> featurization;
        featurize::Tensors tensors;
        size_t featurized = 0;
        double featurize_s = 0;
};

}  // namespace

static AgentRun *run = nullptr;

static int seat_index(Zone::Ownership seat);
static std::unique_ptr<agents::Agent> agent_for(const std::string &spec, const char *flag);
static int decide(Zone::Ownership seat, const std::vector<LegalAction> &actions);
static void report_script();
static void report_featurized();
static void featurize_decision();
static void print_line(const std::string &line);

static int seat_index(Zone::Ownership seat) {
    return seat == Zone::PLAYER_A ? 0 : seat == Zone::PLAYER_B ? 1 : -1;
}

static std::unique_ptr<agents::Agent> agent_for(const std::string &spec, const char *flag) {
    std::string err;
    std::unique_ptr<agents::Agent> agent = agents::make_agent(spec, &run->rng, &err);
    if (!agent) fatal_error(std::string(flag) + ": " + err);
    return agent;
}

// The game log channel: stdout (stderr's descriptor while the RMSTATE stream is on).
static void print_line(const std::string &line) {
    std::printf("%s\n", line.c_str());
}

void agents_install(unsigned int engine_seed) {
    const bool scripted = play_script_given || actions_script_given;
    const bool any_agent = !agent_a_spec.empty() || !agent_b_spec.empty() || scripted;
    const bool any_human = human_seat[0] || human_seat[1];
    if (!any_agent && !any_human && !trace_decisions) return;
    if (any_agent && replay_mode)
        fatal_error("--agent-a/--agent-b, --play and --actions cannot be combined with --replay (the log decides)");
    if (play_script_given && actions_script_given) fatal_error("--play and --actions are exclusive");
    run = new AgentRun();
    run->rng.seed_int(game_seed_given ? game_seed_value : engine_seed);
    const std::string *specs[2] = {&agent_a_spec, &agent_b_spec};
    const char *flags[2] = {"--agent-a", "--agent-b"};
    for (int s = 0; s < 2; ++s) {
        if (!specs[s]->empty()) run->seat_agent[s] = agent_for(*specs[s], flags[s]);
        if (human_seat[s]) run->human[s].reset(new cli::HumanAgent(human_seat[0] && human_seat[1]));
        run->decides[s] = scripted || run->seat_agent[s] || run->human[s];
    }
    if (scripted) {
        agents::Agent *players[2];
        for (int s = 0; s < 2; ++s) {
            if (!run->seat_agent[s] && !run->human[s]) run->first[s] = agent_for("first", flags[s]);
            players[s] = run->seat_agent[s] ? run->seat_agent[s].get()
                         : run->human[s]    ? run->human[s].get()
                                            : run->first[s].get();
        }
        if (play_script_given) {
            run->play = new agents::PlayScript(agents::parse_spec_list(play_script_specs), players[0], players[1]);
            run->script.reset(run->play);
        } else {
            std::vector<int> replies;
            if (!agents::parse_int_list(actions_script, &replies))
                fatal_error("--actions: expected comma-separated integers, got '" + actions_script + "'");
            run->script.reset(new agents::ActionsAgent(std::move(replies), players[0], players[1]));
        }
        std::atexit(report_script);
    }
    if (featurize_seats) {
        if (!run->decides[0] && !run->decides[1])
            fatal_error(
                "--featurize-seats: no seat is decided by an in-process agent (--agent-a/-b, --play, --actions)");
        std::string err;
        run->featurization.reset(new featurize::Featurization());
        if (!load_engine_featurization(run->featurization.get(), &err)) fatal_error("--featurize-seats: " + err);
        std::atexit(report_featurized);
    }
    for (int s = 0; s < 2; ++s)
        if (run->decides[s] || trace_decisions) agent_feed_serve(s == 0 ? Zone::PLAYER_A : Zone::PLAYER_B);
    if (run->decides[0] || run->decides[1])
        InputLogger::instance().set_input_provider(decide, run->decides[0], run->decides[1], true);
    if (any_human && !InputLogger::instance().log_file_name().empty()) {
        print_line("Decision log: " + InputLogger::instance().log_file_name());
        std::fflush(stdout);
    }
}

bool agents_begin_decision(const std::vector<LegalAction> &actions, size_t number) {
    if (!run) return false;
    run->view.reset();
    run->traced = false;
    const Zone::Ownership seat = rmstate_deciding_seat();
    const int s = seat_index(seat);
    if (s < 0 || !agent_feed_serves(seat)) return false;
    std::vector<std::string> match_frames;
    std::string state;
    agent_feed_decision(seat, actions, &match_frames, &state);
    agents::SeatStream &stream = run->streams[s];
    std::string err;
    for (const std::string &payload : match_frames)
        if (!stream.feed_match_frame(payload, &err)) fatal_error("in-process agent frames: " + err);
    if (!stream.decode_state(state, &run->frame, &err)) fatal_error("in-process agent frames: " + err);
    run->seat = s;
    run->number = number;
    run->view.emplace(stream, run->frame, s);
    if (run->featurization && run->decides[s]) featurize_decision();
    if (!trace_decisions) return false;
    std::fflush(stdout);
    for (const std::string &line : agents::trace_decision_lines(*run->view, number)) print_line(line);
    run->traced = true;
    return true;
}

void agents_trace_choice(int choice) {
    if (!run || !run->traced || !run->view) return;
    print_line(agents::trace_choice_line(*run->view, choice));
    run->traced = false;
}

// The decision hook: the deciding seat's agent (or the script) on the view agents_begin_decision
// built for this decision.
static int decide(Zone::Ownership seat, const std::vector<LegalAction> &actions) {
    (void)actions;
    const int s = seat_index(seat);
    if (s < 0 || !run->view || run->seat != s) fatal_error("in-process agent: no frame was built for this decision");
    agents::Agent *agent = run->script          ? run->script.get()
                           : run->seat_agent[s] ? run->seat_agent[s].get()
                                                : run->human[s].get();
    const agents::Decision d{*run->view, run->number > 0 ? run->number - 1 : 0};
    const int choice = agent->decide(d);
    if (choice == agents::AGENT_FAILED) fatal_error(agent->failure());
    if (choice == agents::AGENT_QUIT) InputLogger::instance().end_run_by_quit();
    return choice;
}

// --featurize-seats: the decision's frame through the actor entry point, timed.
static void featurize_decision() {
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    if (!featurize::featurize_seat(
            run->featurization->spec, run->featurization->tables, run->frame, &run->tensors, &err))
        fatal_error("--featurize-seats: " + err);
    run->featurize_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++run->featurized;
}

static void report_featurized() {
    if (!run || !run->featurization) return;
    std::fflush(stdout);
    char buf[96];
    std::snprintf(buf, sizeof buf, "FEATURIZE-SEATS: %zu decisions in %.3f s", run->featurized, run->featurize_s);
    print_line(buf);
    std::fflush(stdout);
}

static void report_script() {
    if (!run || !run->play) return;
    std::fflush(stdout);
    const std::vector<int> &resolved = run->play->resolved();
    if (!resolved.empty()) {
        std::string list;
        for (int r : resolved) list += (list.empty() ? "" : ",") + std::to_string(r);
        print_line("\nresolved --actions: " + list);
    }
    const std::vector<std::string> unused = run->play->unused_specs();
    if (!unused.empty()) {
        std::string list;
        for (const std::string &u : unused) list += (list.empty() ? "" : ", ") + u;
        print_line("\nSCRIPT INCOMPLETE: " + std::to_string(unused.size()) + " --play spec(s) never applied: " + list);
    }
    std::fflush(stdout);
}
