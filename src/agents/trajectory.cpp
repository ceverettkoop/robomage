#include "trajectory.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <memory>
#include <set>

#include "../error.h"
#include "../featurize/setup.h"
#include "../game_driver.h"
#include "../input_logger.h"
#include "../state_format/cards_file.h"
#include "../state_format/encode.h"
#include "../state_format/knowledge_feed.h"
#include "../state_format/stream.h"
#include "../str_util.h"

std::string trajectory_out_dir;
std::string trajectory_seats_option = "AB";
bool trajectory_omniscient = false;

namespace {

// The omniscient viewer's frames the learner-side featurization reads: its schema, and the
// registered lists of its DECKLISTS and METAGAME frames (the targets' opp_list / opp_archetype).
struct OmniscientInputs {
        bool ready = false;
        rmstate::Schema schema;
        featurize::RegisteredLists lists;
};

// The episode being built: the game's header facts and its decisions so far.
struct OpenEpisode {
        std::unique_ptr<featurize::EpisodeWriter> writer;
        bool game_started = false;
        int64_t game = -1;  // the state format's header.game (0-based); -1 until known
        int64_t wins_a = 0, wins_b = 0;
        bool a_goes_first = true;
        int64_t game_engine_seed = -1;
        bool conceded = false;
        int conceder = -1;  // seat index
        size_t game_decisions = 0;
        std::set<uint64_t> versions[2];
};

// Everything the writer keeps across the run.
struct TrajectoryRun {
        featurize::Featurization featurization;
        unsigned int engine_seed = 0;
        std::string stem;
        std::string agent_spec[2];
        bool record[2] = {true, true};
        OmniscientInputs omni;
        std::unique_ptr<OpenEpisode> episode;
        // The decision being read (trajectory_begin_decision).
        int seat = -1;
        const rmstate::DecodedFrame *frame = nullptr;
        featurize::Tensors omni_tensors;
        featurize::Targets targets;
};

}  // namespace

static TrajectoryRun *traj = nullptr;

static std::string default_log_name(unsigned int engine_seed);
static std::string stem_of(const std::string &log_path);
static bool make_dirs(const std::string &path);
static void parse_seats();
static OpenEpisode &open_episode();
static void omniscient_decision(const rmstate::DecodedFrame &seat_frame, const std::vector<LegalAction> &actions);
static void load_omniscient_inputs();
static const char *knowledge_name(Zone::Ownership seat);
static std::string versions_text(const std::set<uint64_t> &versions);
static void write_episode(featurize::EndReason reason, int result_a);

// The decision log's name when --decision-log names none.
static std::string default_log_name(unsigned int engine_seed) {
    return "s" + std::to_string(engine_seed) + ".rmlog";
}

// The episodes' stem: the decision log's file name without its extension.
static std::string stem_of(const std::string &log_path) {
    const size_t slash = log_path.rfind('/');
    std::string name = slash == std::string::npos ? log_path : log_path.substr(slash + 1);
    const size_t dot = name.rfind('.');
    if (dot != std::string::npos && dot > 0) name.erase(dot);
    return name;
}

// mkdir -p.
static bool make_dirs(const std::string &path) {
    for (size_t at = path.find('/', 1);; at = path.find('/', at + 1)) {
        const std::string part = at == std::string::npos ? path : path.substr(0, at);
        if (!part.empty() && mkdir(part.c_str(), 0777) != 0 && errno != EEXIST) return false;
        if (at == std::string::npos) break;
    }
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static void parse_seats() {
    const std::string s = ascii_lower(trajectory_seats_option);
    if (s != "a" && s != "b" && s != "ab")
        fatal_error("--trajectory-seats: expected A, B or AB, got '" + trajectory_seats_option + "'");
    traj->record[0] = s.find('a') != std::string::npos;
    traj->record[1] = s.find('b') != std::string::npos;
}

bool trajectory_enabled() {
    return !trajectory_out_dir.empty();
}

bool trajectory_records_seat(int s) {
    return traj && (s == 0 || s == 1) && traj->record[s];
}

void trajectory_configure_decision_log(unsigned int engine_seed) {
    if (!trajectory_enabled() || replay_mode) return;
    log_decisions_flag = true;
    if (decision_log_path.empty()) {
        if (!make_dirs(trajectory_out_dir)) fatal_error("--trajectory-out: cannot create " + trajectory_out_dir);
        decision_log_path = trajectory_out_dir + "/" + default_log_name(engine_seed);
    }
}

void trajectory_install(unsigned int engine_seed, const std::string agent_specs[2]) {
    if (!trajectory_enabled()) {
        if (trajectory_omniscient || trajectory_seats_option != "AB")
            fatal_error("--trajectory-seats and --trajectory-omniscient need --trajectory-out DIR");
        return;
    }
    traj = new TrajectoryRun();
    parse_seats();
    if (!make_dirs(trajectory_out_dir)) fatal_error("--trajectory-out: cannot create " + trajectory_out_dir);
    std::string err;
    if (!load_engine_featurization(&traj->featurization, &err)) fatal_error("--trajectory-out: " + err);
    traj->engine_seed = engine_seed;
    traj->stem = stem_of(replay_mode ? replay_file_path : decision_log_path);
    for (int s = 0; s < 2; ++s) traj->agent_spec[s] = agent_specs[s];
}

// The episode decisions go to, opened at the first decision or game start after the last one was
// written.
static OpenEpisode &open_episode() {
    if (!traj->episode) {
        traj->episode.reset(new OpenEpisode());
        traj->episode->writer.reset(new featurize::EpisodeWriter(traj->featurization.spec, trajectory_omniscient));
        traj->episode->wins_a = match_wins_a;
        traj->episode->wins_b = match_wins_b;
    }
    return *traj->episode;
}

static void load_omniscient_inputs() {
    OmniscientInputs &o = traj->omni;
    rmstate::DecodeResult res = rmstate::decode_schema_frame(encode_schema_frame(Zone::UNKNOWN), &o.schema);
    if (!res.ok()) fatal_error("--trajectory-omniscient: the SCHEMA frame does not decode: " + res.message);
    rmstate::DecodedFrame decklists, metagame;
    res = rmstate::decode_frame(encode_decklists_frame(Zone::UNKNOWN), o.schema, &decklists);
    if (res.ok()) res = rmstate::decode_frame(encode_metagame_frame(Zone::UNKNOWN), o.schema, &metagame);
    if (!res.ok()) fatal_error("--trajectory-omniscient: a match frame does not decode: " + res.message);
    std::string err;
    if (!featurize::registered_lists(o.schema, decklists, metagame, &o.lists, &err))
        fatal_error("--trajectory-omniscient: " + err);
    o.ready = true;
}

// The omniscient frame of this decision, featurized, and the seat's learner targets.
static void omniscient_decision(const rmstate::DecodedFrame &seat_frame, const std::vector<LegalAction> &actions) {
    if (!traj->omni.ready) load_omniscient_inputs();
    rmstate::DecodedFrame omni;
    const rmstate::DecodeResult res = rmstate::decode_frame(
        encode_state_frame(Zone::UNKNOWN, actions, rmstate_deciding_seat(), {}, false), traj->omni.schema, &omni);
    if (!res.ok()) fatal_error("--trajectory-omniscient: the omniscient frame does not decode: " + res.message);
    std::string err;
    const featurize::Featurization &f = traj->featurization;
    if (!featurize::featurize_viewer(f.spec, f.tables, omni, &traj->omni_tensors, &err) ||
        !featurize::decision_targets(
            f.spec, traj->omni.schema, seat_frame, omni, traj->omni.lists, &traj->targets, &err))
        fatal_error("--trajectory-omniscient: " + err);
}

void trajectory_begin_decision(int seat, const rmstate::DecodedFrame *frame, const std::vector<LegalAction> &actions) {
    if (!traj) return;
    ++open_episode().game_decisions;
    traj->seat = seat;
    traj->frame = trajectory_records_seat(seat) ? frame : nullptr;
    if (traj->frame && trajectory_omniscient) omniscient_decision(*traj->frame, actions);
}

void trajectory_end_decision(int reply, const agents::Behaviour *behaviour, int teacher) {
    if (!traj) return;
    OpenEpisode &ep = open_episode();
    if ((reply == CONCEDE_GAME && !sideboard_phase) || reply == CONCEDE_MATCH) {
        ep.conceded = true;
        ep.conceder = traj->seat;
    }
    if (!traj->frame) return;
    const featurize::Featurization &f = traj->featurization;
    const agents::Behaviour *b = behaviour;
    featurize::EpisodeStep step;
    std::string err;
    if (b && b->tensors && b->spec_hash == f.spec.hash) {
        step.tensors = *b->tensors;
    } else if (!featurize::featurize_seat(f.spec, f.tables, *traj->frame, &step.tensors, &err)) {
        fatal_error("--trajectory-out: " + err);
    }
    step.seat = static_cast<uint8_t>(traj->seat);
    step.game = step.tensors.game;
    step.decision = step.tensors.decision;
    step.sideboarding = step.tensors.sideboarding;
    step.chosen = reply;
    step.teacher = teacher;
    if (b && b->mu.size() == step.tensors.a()) {
        step.has_mu = true;
        step.mu.assign(b->mu.begin(), b->mu.end());
        step.policy_version = static_cast<int64_t>(b->policy_version);
        ep.versions[traj->seat].insert(b->policy_version);
        for (const agents::PolicyEvaluation &p : b->policies)
            step.policies.push_back(featurize::PolicyEvalRecord{p.id, p.version, p.value, p.log_pi});
    }
    if (trajectory_omniscient) {
        step.omniscient = traj->omni_tensors;
        step.targets = traj->targets;
    }
    if (ep.game < 0) ep.game = step.game;
    if (!ep.writer->add(step, &err)) fatal_error("--trajectory-out: " + err);
    traj->frame = nullptr;
}

void trajectory_game_start(bool a_goes_first, unsigned int game_engine_seed) {
    if (!traj) return;
    OpenEpisode &ep = open_episode();
    ep.game_started = true;
    ep.game = match_game_number < 0 ? 0 : match_game_number;
    ep.a_goes_first = a_goes_first;
    ep.game_engine_seed = game_engine_seed;
    ep.wins_a = match_wins_a;
    ep.wins_b = match_wins_b;
}

void trajectory_game_end(int winner) {
    if (!traj || !traj->episode) return;
    const int result_a = winner == Zone::PLAYER_A ? 1 : winner == Zone::PLAYER_B ? -1 : 0;
    const featurize::EndReason reason = traj->episode->conceded ? featurize::EndReason::CONCESSION
                                        : result_a == 0         ? featurize::EndReason::DRAW
                                                                : featurize::EndReason::WIN;
    write_episode(reason, result_a);
}

void trajectory_match_over() {
    if (!traj || !traj->episode) return;
    const OpenEpisode &ep = *traj->episode;
    if (!ep.conceded) {
        traj->episode.reset();
        return;
    }
    write_episode(featurize::EndReason::CONCESSION, ep.conceder == 0 ? -1 : ep.conceder == 1 ? 1 : 0);
}

void trajectory_end_run(featurize::EndReason reason) {
    if (!traj || !traj->episode) return;
    write_episode(reason, 0);
}

static const char *knowledge_name(Zone::Ownership seat) {
    switch (knowledge_level_of(seat)) {
        case rmstate::KnowledgeLevel::OFF:
            return "off";
        case rmstate::KnowledgeLevel::EXPERT:
            return "expert";
        case rmstate::KnowledgeLevel::EXPERT_BELIEF:
            return "expert+belief";
    }
    return "";
}

// The distinct versions, ascending, comma-separated.
static std::string versions_text(const std::set<uint64_t> &versions) {
    std::string out;
    for (const uint64_t v : versions) out += (out.empty() ? "" : ",") + std::to_string(v);
    return out;
}

// Writes the open episode as DIR/<stem>.g<n>.rmtraj and closes it.
static void write_episode(featurize::EndReason reason, int result_a) {
    std::unique_ptr<OpenEpisode> ep = std::move(traj->episode);
    const int64_t game = ep->game < 0 ? 0 : ep->game;
    std::vector<featurize::RmtbAttr> attrs;
    auto add_s = [&attrs](const char *k, const std::string &v) {
        featurize::RmtbAttr a;
        a.key = k;
        a.s = v;
        attrs.push_back(a);
    };
    auto add_i = [&attrs](const char *k, int64_t v) {
        featurize::RmtbAttr a;
        a.key = k;
        a.is_int = true;
        a.i = v;
        attrs.push_back(a);
    };
    add_s("format", bo3_mode ? "bo3" : "bo1");
    add_i("game", game);
    add_i("wins_a", ep->wins_a);
    add_i("wins_b", ep->wins_b);
    add_s("first", !ep->game_started ? "" : ep->a_goes_first ? "A" : "B");
    add_s("deck_a", deck_a_name);
    add_s("deck_b", deck_b_name);
    add_i("engine_seed", traj->engine_seed);
    add_i("game_engine_seed", ep->game_engine_seed);
    add_i("game_seed", game_seed_given ? static_cast<int64_t>(game_seed_value) : -1);
    add_s("agent_a", traj->agent_spec[0]);
    add_s("agent_b", traj->agent_spec[1]);
    add_s("knowledge_a", knowledge_name(Zone::PLAYER_A));
    add_s("knowledge_b", knowledge_name(Zone::PLAYER_B));
    add_s("seats", std::string(traj->record[0] ? "A" : "") + (traj->record[1] ? "B" : ""));
    add_s("policy_versions_a", versions_text(ep->versions[0]));
    add_s("policy_versions_b", versions_text(ep->versions[1]));
    add_s("rmlog", replay_mode ? replay_file_path : InputLogger::instance().log_file_name());
    add_i("result_a", result_a);
    add_i("result_b", -result_a);
    add_s("end_reason", featurize::end_reason_name(reason));
    add_i("decisions", static_cast<int64_t>(ep->writer->steps()));
    add_i("game_decisions", static_cast<int64_t>(ep->game_decisions));
    const std::string path = trajectory_out_dir + "/" + traj->stem + ".g" + std::to_string(game + 1) + ".rmtraj";
    std::string err;
    if (!featurize::rmtb_write_file(path, ep->writer->file(traj->featurization.text_model, attrs), &err))
        fatal_error("--trajectory-out: " + err);
}
