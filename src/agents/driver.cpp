#include "driver.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

#include "../card_vocab.h"
#include "../cli/human.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../featurize/featurizer.h"
#include "../game_driver.h"
#include "../input_logger.h"
#include "../queries/characteristics.h"
#include "../queries/players.h"
#include "../state_format/agent_feed.h"
#include "../state_format/cards_file.h"
#include "../state_format/encode.h"
#include "../state_format/enum_names.h"
#include "../state_format/knowledge_feed.h"
#include "../state_format/stream.h"
#include "actor.h"
#include "agent.h"
#include "play_spec.h"
#include "teacher.h"
#include "trace.h"
#include "trajectory.h"

namespace {

// Everything the hook keeps across the run.
struct AgentRun {
        agents::PyRandom rng;
        std::unique_ptr<agents::Agent> seat_agent[2];
        std::string seat_spec[2];  // seat_agent's spec
        // Agents of earlier matches no seat plays now, with their specs (seat_agent_for_match reuses them).
        std::vector<std::pair<std::string, std::unique_ptr<agents::Agent>>> idle;
        std::unique_ptr<agents::Agent> human[2];  // a human seat's CLI (cli/human.h)
        std::unique_ptr<agents::Agent> first[2];  // a seat's player under a script when it has no agent
        std::unique_ptr<agents::Agent> script;    // --play / --actions
        agents::PlayScript *play = nullptr;       // `script` when it is a --play script
        bool decides[2] = {false, false};
        std::unique_ptr<agents::SeatStream> streams[2];  // made anew for each match
        // The current decision, as built by agents_begin_decision.
        int seat = -1;
        size_t number = 0;
        rmstate::DecodedFrame frame;
        std::optional<agents::DecisionView> view;
        bool traced = false;
        // The behaviour distribution the decision's reply was drawn from (decide(): the deciding
        // agent's, or the teacher channel's mixture), for the trajectory writer.
        const agents::Behaviour *behaviour = nullptr;
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
static void seat_agent_for_match(int s, const std::string &spec, const char *flag);
static int decide(Zone::Ownership seat, const std::vector<LegalAction> &actions);
static void report_script();
static void report_featurized();
static void featurize_decision();
static void print_line(const std::string &line);
static void seat_specs(std::string out[2]);
static agents::MenuAction auto_menu_row(const LegalAction &la, Zone::Ownership seat, const std::string &description);

static int seat_index(Zone::Ownership seat) {
    return seat == Zone::PLAYER_A ? 0 : seat == Zone::PLAYER_B ? 1 : -1;
}

static std::unique_ptr<agents::Agent> agent_for(const std::string &spec, const char *flag) {
    std::string err;
    std::unique_ptr<agents::Agent> agent = agents::make_agent(spec, &run->rng, &err);
    if (!agent) fatal_error(std::string(flag) + ": " + err);
    return agent;
}

// Seat s's agent for the match about to start: the one an earlier match left idle with the same
// spec (the seat's own first), else a new one; the seat's agent of the last match goes idle. Its
// begin_match runs; fatal when it fails. No agent for an empty spec.
static void seat_agent_for_match(int s, const std::string &spec, const char *flag) {
    if (run->seat_agent[s]) run->idle.emplace_back(run->seat_spec[s], std::move(run->seat_agent[s]));
    run->seat_spec[s] = spec;
    if (spec.empty()) return;
    for (size_t i = run->idle.size(); i-- > 0;)
        if (run->idle[i].first == spec) {
            run->seat_agent[s] = std::move(run->idle[i].second);
            run->idle.erase(run->idle.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
    if (!run->seat_agent[s]) run->seat_agent[s] = agent_for(spec, flag);
    std::string err;
    if (!run->seat_agent[s]->begin_match(&err)) fatal_error(std::string(flag) + ": " + err);
}

// The game log channel: stdout (stderr's descriptor while the RMSTATE stream is on).
static void print_line(const std::string &line) {
    std::printf("%s\n", line.c_str());
}

void agents_install(unsigned int engine_seed) {
    const bool scripted = play_script_given || actions_script_given;
    const bool any_agent = !agent_a_spec.empty() || !agent_b_spec.empty() || scripted;
    const bool any_human = human_seat[0] || human_seat[1];
    const uint64_t coin_seed = game_seed_given ? game_seed_value : engine_seed;
    if (!any_agent && !any_human && !trace_decisions && !trajectory_enabled() && !teacher_enabled()) {
        trajectory_install(engine_seed, nullptr);
        teacher_install(coin_seed, nullptr);
        return;
    }
    if (any_agent && replay_mode)
        fatal_error("--agent-a/--agent-b, --play and --actions cannot be combined with --replay (the log decides)");
    if (play_script_given && actions_script_given) fatal_error("--play and --actions are exclusive");
    const bool first_match = !run;
    if (first_match) run = new AgentRun();
    run->rng.seed_int(game_seed_given ? game_seed_value : engine_seed);
    const std::string *specs[2] = {&agent_a_spec, &agent_b_spec};
    const char *flags[2] = {"--agent-a", "--agent-b"};
    for (int s = 0; s < 2; ++s) {
        seat_agent_for_match(s, *specs[s], flags[s]);
        if (human_seat[s] && !run->human[s]) run->human[s].reset(new cli::HumanAgent(human_seat[0] && human_seat[1]));
        run->decides[s] = scripted || run->seat_agent[s] || run->human[s];
        run->streams[s].reset(new agents::SeatStream());
    }
    if (scripted && first_match) {
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
    if (featurize_seats && first_match) {
        if (!run->decides[0] && !run->decides[1])
            fatal_error(
                "--featurize-seats: no seat is decided by an in-process agent (--agent-a/-b, --play, --actions)");
        std::string err;
        run->featurization.reset(new featurize::Featurization());
        if (!load_engine_featurization(run->featurization.get(), &err)) fatal_error("--featurize-seats: " + err);
        std::atexit(report_featurized);
    }
    std::string specs_for_trajectory[2];
    seat_specs(specs_for_trajectory);
    trajectory_install(engine_seed, specs_for_trajectory);
    const bool in_process[2] = {run->decides[0] && !run->human[0], run->decides[1] && !run->human[1]};
    teacher_install(coin_seed, in_process);
    agent_feed_reset();
    knowledge_feed_reset();
    for (int s = 0; s < 2; ++s)
        if (run->decides[s] || trace_decisions || trajectory_records_seat(s) || teacher_on(s))
            agent_feed_serve(s == 0 ? Zone::PLAYER_A : Zone::PLAYER_B);
    if (run->decides[0] || run->decides[1])
        InputLogger::instance().set_input_provider(decide, run->decides[0], run->decides[1], true);
    else
        InputLogger::instance().clear_input_provider();
    if (any_human && first_match && !InputLogger::instance().log_file_name().empty()) {
        print_line("Decision log: " + InputLogger::instance().log_file_name());
        std::fflush(stdout);
    }
}

void agents_begin_decision(const std::vector<LegalAction> &actions, size_t number) {
    if (!run) return;
    run->view.reset();
    run->traced = false;
    run->behaviour = nullptr;
    const Zone::Ownership seat = rmstate_deciding_seat();
    const int s = seat_index(seat);
    run->seat = s;
    if (s < 0 || !agent_feed_serves(seat)) {
        trajectory_begin_decision(s, nullptr, actions);
        return;
    }
    std::vector<std::string> match_frames;
    std::string state;
    agent_feed_decision(seat, actions, &match_frames, &state);
    teacher_send(s, match_frames, state, actions.size(), number);
    agents::SeatStream &stream = *run->streams[s];
    std::string err;
    for (const std::string &payload : match_frames)
        if (!stream.feed_match_frame(payload, &err)) fatal_error("in-process agent frames: " + err);
    if (!stream.decode_state(state, &run->frame, &err)) fatal_error("in-process agent frames: " + err);
    run->number = number;
    run->view.emplace(stream, run->frame, s);
    if (run->featurization && run->decides[s]) featurize_decision();
    trajectory_begin_decision(s, &run->frame, actions);
    if (!trace_decisions) return;
    std::fflush(stdout);
    for (const std::string &line : agents::trace_decision_lines(*run->view, number)) print_line(line);
    run->traced = true;
}

void agents_auto_decision(const LegalAction &taken) {
    if (!run) return;
    const Zone::Ownership seat = rmstate_deciding_seat();
    const int s = seat_index(seat);
    if (s < 0) return;
    const std::string description = menu_row_description(taken, seat);
    if (run->play) run->play->auto_taken(s == 0 ? 'A' : 'B', auto_menu_row(taken, seat, description));
    if (!trace_decisions) return;
    std::fflush(stdout);
    print_line(std::string("| auto: Player ") + (s == 0 ? "A" : "B") + ": " + description);
}

// A row of a menu taken without a frame, as the --play resolver would read it in `seat`'s frame
// (resolver_menu): its source's card name (the face shown on the battlefield) and controller.
static agents::MenuAction auto_menu_row(const LegalAction &la, Zone::Ownership seat, const std::string &description) {
    agents::MenuAction a;
    a.category = action_category_enum_name(la.category);
    a.description = description;
    const Entity src = la.source_entity;
    const int cid = menu_row_card(la, seat);
    const bool zoned = src != 0 && global_coordinator.entity_has_component<Zone>(src);
    const Zone *z = zoned ? &global_coordinator.GetComponent<Zone>(src) : nullptr;
    if (cid >= 0)
        a.card = ascii_fold_card_name(
            z && z->location == Zone::BATTLEFIELD ? entity_name(src) : std::string(card_index_to_name(cid)));
    if (src != 0 && global_coordinator.entity_has_component<Player>(src)) {
        a.controller = seat_of_player(src) == seat ? 1 : 0;
    } else if (z && (z->location == Zone::BATTLEFIELD || z->location == Zone::HAND || z->location == Zone::STACK ||
                        z->location == Zone::GRAVEYARD || z->location == Zone::EXILE)) {
        const Zone::Ownership controller =
            z->location == Zone::BATTLEFIELD || z->location == Zone::STACK ? source_controller(src) : z->owner;
        a.controller = controller == seat ? 1 : 0;
    }
    return a;
}

void agents_end_decision(int choice, int reply) {
    if (!run) return;
    if (run->traced && run->view) {
        print_line(agents::trace_choice_line(*run->view, choice));
        run->traced = false;
    }
    const int label = teacher_take_label();
    trajectory_end_decision(reply, run->behaviour, label);
    actor_note_decision(run->seat, reply, run->behaviour);
}

// The decision hook: the deciding seat's agent (or the script) on the view agents_begin_decision
// built for this decision, its reply mixed with a teacher's label under --teacher-mix.
static int decide(Zone::Ownership seat, const std::vector<LegalAction> &actions) {
    const int s = seat_index(seat);
    if (s < 0 || !run->view || run->seat != s) fatal_error("in-process agent: no frame was built for this decision");
    agents::Agent *agent = run->script          ? run->script.get()
                           : run->seat_agent[s] ? run->seat_agent[s].get()
                                                : run->human[s].get();
    const agents::Decision d{*run->view, run->number > 0 ? run->number - 1 : 0};
    const int choice = agent->decide(d);
    if (choice == agents::AGENT_FAILED) fatal_error(agent->failure());
    if (choice == agents::AGENT_QUIT) InputLogger::instance().end_run_by_quit();
    return teacher_mix(choice, agent->last_behaviour(), actions.size(), &run->behaviour);
}

// Each seat's decider as the trajectory writer's header names it: the --replay log, the --play /
// --actions script (then the seat's agent), the seat's agent spec, a human seat, or the RMSTATE
// stream.
static void seat_specs(std::string out[2]) {
    const std::string *specs[2] = {&agent_a_spec, &agent_b_spec};
    for (int s = 0; s < 2; ++s) {
        const std::string own = !specs[s]->empty() ? *specs[s] : human_seat[s] ? "human" : "stream";
        if (replay_mode)
            out[s] = "replay";
        else if (play_script_given || actions_script_given)
            out[s] = std::string(play_script_given ? "play" : "actions") + "+" + (specs[s]->empty() ? "first" : own);
        else
            out[s] = own;
    }
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
