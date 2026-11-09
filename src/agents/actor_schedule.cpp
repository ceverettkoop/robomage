#include "actor_schedule.h"

#include <sys/stat.h>

#include <cmath>
#include <set>

#include "../featurize/json.h"
#include "../featurize/rmtb.h"
#include "../state_format/schema.h"

namespace agents {

namespace {

using featurize::Json;

// The keys an entry may carry (the defaults object the same, but id).
const char *const ENTRY_KEYS[] = {"id", "weight", "purpose", "deck_a", "deck_b", "format", "swap", "agent_a", "agent_b",
    "teacher_a", "teacher_b", "teacher_mix", "knowledge_a", "knowledge_b", "max_game_decisions", "trajectory_seats",
    "trajectory_omniscient", "rmlog_dir", "trajectory_dir", "sb_explore"};
const char *const PURPOSES[] = {"train", "eval"};

}  // namespace

static bool known_key(const std::string &key);
static bool is_purpose(const std::string &p);
static bool get_string(const Json &v, const std::string &key, std::string *out, std::string *err);
static bool get_optional_string(const Json &v, const std::string &key, std::string *out, std::string *err);
static bool knowledge_value(const std::string &name, int *out);
static bool apply_outputs(const Json &obj, const std::string &where, PurposeOutputs *out, std::string *err);
static bool apply_entry_keys(const Json &obj, const std::string &where, ScheduleEntry *e, std::string *err);
static bool positive_number(const Json &v, const std::string &name, double *out, std::string *err);
static bool apply_card_weights(
    const Json &obj, const std::string &where, std::map<std::string, double> *out, std::string *err);
static bool apply_deck_weights(const Json &obj, const std::string &where, SbCardWeights *out, std::string *err);
static bool apply_sb_explore(const Json &obj, const std::string &where, SbExploreSpec *out, std::string *err);
static bool check_entry(const ScheduleEntry &e, const std::string &decks_dir, std::string *err);
static bool file_exists(const std::string &path);

static bool known_key(const std::string &key) {
    for (const char *k : ENTRY_KEYS)
        if (key == k) return true;
    return false;
}

static bool is_purpose(const std::string &p) {
    for (const char *k : PURPOSES)
        if (p == k) return true;
    return false;
}

static bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// A string member; false with `err` when it is another type.
static bool get_string(const Json &v, const std::string &key, std::string *out, std::string *err) {
    if (!v.is_string()) {
        *err = key + " must be a string";
        return false;
    }
    *out = v.as_string();
    return true;
}

// A string member, or null for "".
static bool get_optional_string(const Json &v, const std::string &key, std::string *out, std::string *err) {
    if (v.is_null()) {
        out->clear();
        return true;
    }
    return get_string(v, key, out, err);
}

static bool knowledge_value(const std::string &name, int *out) {
    if (name == "off")
        *out = 0;
    else if (name == "expert")
        *out = 1;
    else
        return false;
    return true;
}

// {"rmlog_dir": DIR, "trajectory_dir": DIR or null} into `out` (only the keys present).
static bool apply_outputs(const Json &obj, const std::string &where, PurposeOutputs *out, std::string *err) {
    if (!obj.is_object()) {
        *err = where + " must be an object";
        return false;
    }
    for (const auto &m : obj.members()) {
        if (m.first == "rmlog_dir") {
            if (!get_string(m.second, where + ".rmlog_dir", &out->rmlog_dir, err)) return false;
            if (out->rmlog_dir.empty()) {
                *err = where + ".rmlog_dir must not be empty";
                return false;
            }
            out->rmlog_set = true;
        } else if (m.first == "trajectory_dir") {
            if (!get_optional_string(m.second, where + ".trajectory_dir", &out->trajectory_dir, err)) return false;
            out->trajectory_set = true;
        } else {
            *err = where + ": unknown key '" + m.first + "' (rmlog_dir, trajectory_dir)";
            return false;
        }
    }
    return true;
}

// The keys of an entry (or the defaults) applied onto `e`.
static bool apply_entry_keys(const Json &obj, const std::string &where, ScheduleEntry *e, std::string *err) {
    if (!obj.is_object()) {
        *err = where + " must be an object";
        return false;
    }
    for (const auto &m : obj.members()) {
        const std::string &k = m.first;
        const Json &v = m.second;
        const std::string name = where + "." + k;
        const int side = k.size() > 2 && k.compare(k.size() - 2, 2, "_b") == 0 ? 1 : 0;
        std::string text;
        if (!known_key(k)) {
            *err = where + ": unknown key '" + k + "'";
            return false;
        }
        if (k == "id") {
            if (!get_string(v, name, &e->id, err)) return false;
        } else if (k == "weight") {
            if (!v.is_number() || !(v.as_number() >= 0) || !std::isfinite(v.as_number())) {
                *err = name + " must be a number >= 0";
                return false;
            }
            e->weight = v.as_number();
        } else if (k == "purpose") {
            if (!get_string(v, name, &e->purpose, err)) return false;
            if (!is_purpose(e->purpose)) {
                *err = name + " must be \"train\" or \"eval\"";
                return false;
            }
        } else if (k == "deck_a" || k == "deck_b") {
            if (!get_string(v, name, &e->deck[side], err)) return false;
        } else if (k == "format") {
            if (!get_string(v, name, &text, err)) return false;
            if (text != "bo3" && text != "bo1") {
                *err = name + " must be \"bo3\" or \"bo1\"";
                return false;
            }
            e->bo3 = text == "bo3";
        } else if (k == "swap") {
            if (!get_string(v, name, &text, err)) return false;
            if (text == "none")
                e->swap = SeatSwap::NONE;
            else if (text == "decks")
                e->swap = SeatSwap::DECKS;
            else if (text == "sides")
                e->swap = SeatSwap::SIDES;
            else {
                *err = name + " must be \"none\", \"decks\" or \"sides\"";
                return false;
            }
        } else if (k == "agent_a" || k == "agent_b") {
            if (!get_string(v, name, &e->agent[side], err)) return false;
        } else if (k == "teacher_a" || k == "teacher_b") {
            if (!get_optional_string(v, name, &e->teacher[side], err)) return false;
        } else if (k == "teacher_mix") {
            if (!v.is_number() || !(v.as_number() >= 0 && v.as_number() <= 1)) {
                *err = name + " must be a number in [0, 1]";
                return false;
            }
            e->teacher_mix = v.as_number();
        } else if (k == "knowledge_a" || k == "knowledge_b") {
            if (!get_string(v, name, &text, err)) return false;
            if (!knowledge_value(text, &e->knowledge[side])) {
                *err = name + " must be off or expert";
                return false;
            }
        } else if (k == "max_game_decisions") {
            if (!v.is_number() || !v.is_integer() || v.as_number() < 0) {
                *err = name + " must be an integer >= 0";
                return false;
            }
            e->max_game_decisions = v.as_int();
        } else if (k == "trajectory_seats") {
            if (!get_string(v, name, &e->trajectory_seats, err)) return false;
            if (e->trajectory_seats != "A" && e->trajectory_seats != "B" && e->trajectory_seats != "AB") {
                *err = name + " must be A, B or AB";
                return false;
            }
        } else if (k == "trajectory_omniscient") {
            if (!v.is_bool()) {
                *err = name + " must be true or false";
                return false;
            }
            e->trajectory_omniscient = v.as_bool();
        } else if (k == "rmlog_dir") {
            if (!get_string(v, name, &e->outputs.rmlog_dir, err)) return false;
            if (e->outputs.rmlog_dir.empty()) {
                *err = name + " must not be empty";
                return false;
            }
            e->outputs.rmlog_set = true;
        } else if (k == "trajectory_dir") {
            if (!get_optional_string(v, name, &e->outputs.trajectory_dir, err)) return false;
            e->outputs.trajectory_set = true;
        } else if (k == "sb_explore") {
            if (!apply_sb_explore(v, name, &e->sb_explore, err)) return false;
        }
    }
    return true;
}

// A finite number > 0.
static bool positive_number(const Json &v, const std::string &name, double *out, std::string *err) {
    if (!v.is_number() || !(v.as_number() > 0) || !std::isfinite(v.as_number())) {
        *err = name + " must be a number > 0";
        return false;
    }
    *out = v.as_number();
    return true;
}

// {card: weight > 0, ...}, keyed by sb_card_key.
static bool apply_card_weights(
    const Json &obj, const std::string &where, std::map<std::string, double> *out, std::string *err) {
    if (!obj.is_object()) {
        *err = where + " must be an object of card names and weights";
        return false;
    }
    for (const auto &m : obj.members()) {
        const std::string key = sb_card_key(m.first);
        if (key.empty()) {
            *err = where + ": '" + m.first + "' is not a card name";
            return false;
        }
        if (out->count(key)) {
            *err = where + ": '" + m.first + "' names a card twice";
            return false;
        }
        if (!positive_number(m.second, where + "." + m.first, &(*out)[key], err)) return false;
    }
    return true;
}

// One deck's {"in": {...}, "out": {...}}.
static bool apply_deck_weights(const Json &obj, const std::string &where, SbCardWeights *out, std::string *err) {
    if (!obj.is_object()) {
        *err = where + " must be an object";
        return false;
    }
    for (const auto &m : obj.members()) {
        if (m.first != "in" && m.first != "out") {
            *err = where + ": unknown key '" + m.first + "' (in, out)";
            return false;
        }
        if (!apply_card_weights(m.second, where + "." + m.first, m.first == "in" ? &out->in : &out->out, err))
            return false;
    }
    return true;
}

// An sb_explore object; it replaces whatever `out` held.
static bool apply_sb_explore(const Json &obj, const std::string &where, SbExploreSpec *out, std::string *err) {
    if (!obj.is_object()) {
        *err = where + " must be an object";
        return false;
    }
    SbExploreSpec spec;
    bool have_p = false;
    for (const auto &m : obj.members()) {
        const std::string &k = m.first;
        const Json &v = m.second;
        const std::string name = where + "." + k;
        if (k == "p") {
            if (!v.is_number() || !(v.as_number() >= 0 && v.as_number() <= 1)) {
                *err = name + " must be a number in [0, 1]";
                return false;
            }
            spec.p = v.as_number();
            have_p = true;
        } else if (k == "count_weights") {
            const size_t n = SB_MAX_COUNT + 1;
            if (!v.is_array() || v.items().size() != n) {
                *err = name + " must be an array of " + std::to_string(n) + " numbers (k = 0 .. " +
                       std::to_string(SB_MAX_COUNT) + ")";
                return false;
            }
            double total = 0;
            for (size_t i = 0; i < n; ++i) {
                const Json &x = v.items()[i];
                if (!x.is_number() || !(x.as_number() >= 0) || !std::isfinite(x.as_number())) {
                    *err = name + "[" + std::to_string(i) + "] must be a number >= 0";
                    return false;
                }
                spec.count_weights[i] = x.as_number();
                total += x.as_number();
            }
            if (!(total > 0)) {
                *err = name + " must not be all 0";
                return false;
            }
        } else if (k == "weights") {
            if (!v.is_object()) {
                *err = name + " must be an object keyed by deck stem";
                return false;
            }
            for (const auto &d : v.members())
                if (!apply_deck_weights(d.second, name + "." + d.first, &spec.weights[d.first], err)) return false;
        } else if (k == "default_weight") {
            if (!positive_number(v, name, &spec.default_weight, err)) return false;
        } else if (k == "agents") {
            std::string text;
            if (!get_string(v, name, &text, err)) return false;
            if (text != "nn" && text != "all") {
                *err = name + " must be \"nn\" or \"all\"";
                return false;
            }
            spec.every_agent = text == "all";
        } else {
            *err = where + ": unknown key '" + k + "' (p, count_weights, weights, default_weight, agents)";
            return false;
        }
    }
    if (!have_p) {
        *err = where + ": no p";
        return false;
    }
    *out = std::move(spec);
    return true;
}

// A resolved entry is complete: both decks (whose files exist), both agents, a teacher for a mix.
static bool check_entry(const ScheduleEntry &e, const std::string &decks_dir, std::string *err) {
    const std::string where = "entry '" + e.id + "'";
    for (int s = 0; s < 2; ++s) {
        const char *side = s == 0 ? "a" : "b";
        if (e.deck[s].empty()) {
            *err = where + ": no deck_" + side + " (the entry, the defaults or --deck-" + side + ")";
            return false;
        }
        if (!file_exists(decks_dir + "/" + e.deck[s] + ".dk")) {
            *err = where + ": deck_" + side + " '" + e.deck[s] + "' is not a deck (" + decks_dir + "/" + e.deck[s] +
                   ".dk)";
            return false;
        }
        if (e.agent[s].empty()) {
            *err = where + ": no agent_" + side + " (the entry, the defaults or --agent-" + side + ")";
            return false;
        }
    }
    if (e.teacher_mix > 0 && e.teacher[0].empty() && e.teacher[1].empty()) {
        *err = where + ": teacher_mix needs teacher_a or teacher_b";
        return false;
    }
    if (e.purpose == "eval" && e.sb_explore.p > 0) {
        *err = where + ": an eval entry does not explore (sb_explore p > 0; evaluations play the policy)";
        return false;
    }
    return true;
}

bool parse_schedule(
    const std::string &text, const ScheduleEntry &base, const std::string &decks_dir, Schedule *out, std::string *err) {
    Json root;
    if (!Json::parse(text, &root, err)) return false;
    if (!root.is_object()) {
        *err = "the schedule must be a JSON object";
        return false;
    }
    Schedule s;
    ScheduleEntry defaults = base;
    const Json *entries = nullptr;
    for (const auto &m : root.members()) {
        if (m.first == "version") {
            s.version = m.second.text();
        } else if (m.first == "outputs") {
            if (!m.second.is_object()) {
                *err = "outputs must be an object";
                return false;
            }
            for (const auto &p : m.second.members()) {
                if (!is_purpose(p.first)) {
                    *err = "outputs: unknown purpose '" + p.first + "' (train, eval)";
                    return false;
                }
                if (!apply_outputs(p.second, "outputs." + p.first, &s.outputs[p.first], err)) return false;
            }
        } else if (m.first == "defaults") {
            if (m.second.has("id")) {
                *err = "defaults: id belongs to an entry";
                return false;
            }
            if (!apply_entry_keys(m.second, "defaults", &defaults, err)) return false;
        } else if (m.first == "entries") {
            entries = &m.second;
        } else {
            *err = "unknown key '" + m.first + "' (version, outputs, defaults, entries)";
            return false;
        }
    }
    if (!entries || !entries->is_array() || entries->items().empty()) {
        *err = "entries must be a non-empty array";
        return false;
    }
    std::set<std::string> ids;
    for (size_t i = 0; i < entries->items().size(); ++i) {
        ScheduleEntry e = defaults;
        e.id.clear();
        if (!apply_entry_keys(entries->items()[i], "entries[" + std::to_string(i) + "]", &e, err)) return false;
        if (e.id.empty()) {
            *err = "entries[" + std::to_string(i) + "]: no id";
            return false;
        }
        if (!ids.insert(e.id).second) {
            *err = "entries[" + std::to_string(i) + "]: id '" + e.id + "' is not unique";
            return false;
        }
        if (!check_entry(e, decks_dir, err)) return false;
        s.total_weight += e.weight;
        s.entries.push_back(std::move(e));
    }
    if (!(s.total_weight > 0)) {
        *err = "no entry has a positive weight";
        return false;
    }
    if (s.version.empty()) s.version = featurize::hex16(rmstate::fnv1a64(text.data(), text.size()));
    *out = std::move(s);
    return true;
}

SbExplore sb_explore_for_deck(const SbExploreSpec &spec, const std::string &deck) {
    SbExplore out;
    out.p = spec.p;
    out.count_weights = spec.count_weights;
    out.default_weight = spec.default_weight;
    const auto it = spec.weights.find(deck);
    if (it != spec.weights.end()) out.weights = it->second;
    return out;
}

uint64_t splitmix64(uint64_t x) {
    uint64_t z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

MatchPlan plan_match(const Schedule &schedule, uint64_t game_seed) {
    MatchPlan plan;
    const double u = static_cast<double>(splitmix64(game_seed ^ 0x6A09E667F3BCC909ull) >> 11) * 0x1.0p-53;
    const double target = u * schedule.total_weight;
    double acc = 0;
    plan.entry = schedule.entries.size();
    for (size_t i = 0; i < schedule.entries.size(); ++i) {
        if (!(schedule.entries[i].weight > 0)) continue;
        acc += schedule.entries[i].weight;
        plan.entry = i;
        if (target < acc) break;
    }
    const bool coin = (splitmix64(game_seed ^ 0xBB67AE8584CAA73Bull) >> 63) == 1;
    const ScheduleEntry &e = schedule.entries[plan.entry];
    plan.swapped = coin && e.swap != SeatSwap::NONE;
    if (plan.swapped) {
        plan.deck_of_seat[0] = 1;
        plan.deck_of_seat[1] = 0;
        if (e.swap == SeatSwap::SIDES) {
            plan.side_of_seat[0] = 1;
            plan.side_of_seat[1] = 0;
        }
    }
    return plan;
}

}  // namespace agents
