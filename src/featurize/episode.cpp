#include "episode.h"

#include <algorithm>

namespace {

// The widths a token type's slots take in each compact array, per the spec.
struct Widths {
        size_t num = 0, cat = 0, card = 0, ability = 0;  // num: the type's numeric slots (informational)
};

// One featurization's arrays read back.
struct TokenArrays {
        std::vector<int64_t> tok_offsets, rel_offsets, num_offsets, tok_type, num_index, cat, card, ability, rel;
        std::vector<float> num_value;
};

}  // namespace

static featurize::Dtype index_dtype(const std::vector<int64_t> &values);
static bool int_array(
    const featurize::RmtbFile &f, const std::string &name, std::vector<int64_t> *out, std::string *err);
static bool float_array(
    const featurize::RmtbFile &f, const std::string &name, std::vector<float> *out, std::string *err);
static bool int_attr(const featurize::RmtbFile &f, const char *key, int64_t *out, std::string *err);
static bool offsets_ok(const std::vector<int64_t> &offsets, size_t steps, size_t total);
static bool read_tokens(
    const featurize::RmtbFile &f, const std::string &prefix, size_t steps, TokenArrays *out, std::string *err);
static bool expand_tokens(const TokenArrays &t, size_t d, const std::vector<Widths> &widths, size_t F, size_t C,
    size_t K, size_t Q, int64_t menu_type, featurize::Tensors *out, std::string *err, Widths *pos);
static bool read_list(const featurize::RmtbFile &f, const std::string &name, size_t steps, std::vector<int64_t> *values,
    std::vector<int64_t> *offsets, std::string *err);
static void slice(
    const std::vector<int64_t> &values, const std::vector<int64_t> &offsets, size_t d, std::vector<int64_t> *out);

// u16 when every value fits, else i32.
static featurize::Dtype index_dtype(const std::vector<int64_t> &values) {
    for (const int64_t v : values)
        if (v < 0 || v > 0xFFFF) return featurize::Dtype::I32;
    return featurize::Dtype::U16;
}

static bool int_array(
    const featurize::RmtbFile &f, const std::string &name, std::vector<int64_t> *out, std::string *err) {
    const featurize::RmtbArray *a = f.find(name);
    if (!a || !featurize::rmtb_values(*a, out)) {
        *err = "no integer array " + name;
        return false;
    }
    return true;
}

static bool float_array(
    const featurize::RmtbFile &f, const std::string &name, std::vector<float> *out, std::string *err) {
    const featurize::RmtbArray *a = f.find(name);
    if (!a || !featurize::rmtb_values(*a, out)) {
        *err = "no float array " + name;
        return false;
    }
    return true;
}

static bool int_attr(const featurize::RmtbFile &f, const char *key, int64_t *out, std::string *err) {
    const featurize::RmtbAttr *a = f.attr(key);
    if (!a || !a->is_int) {
        *err = std::string("no integer attr ") + key;
        return false;
    }
    *out = a->i;
    return true;
}

// `steps` + 1 non-decreasing offsets from 0 to `total`.
static bool offsets_ok(const std::vector<int64_t> &offsets, size_t steps, size_t total) {
    if (offsets.size() != steps + 1 || offsets[0] != 0 || offsets.back() != static_cast<int64_t>(total)) return false;
    for (size_t i = 0; i < steps; ++i)
        if (offsets[i + 1] < offsets[i]) return false;
    return true;
}

static bool read_tokens(
    const featurize::RmtbFile &f, const std::string &prefix, size_t steps, TokenArrays *out, std::string *err) {
    if (!int_array(f, prefix + "tok_offsets", &out->tok_offsets, err) ||
        !int_array(f, prefix + "rel_offsets", &out->rel_offsets, err) ||
        !int_array(f, prefix + "tok_type", &out->tok_type, err) ||
        !int_array(f, prefix + "tok_num_offsets", &out->num_offsets, err) ||
        !int_array(f, prefix + "tok_num_index", &out->num_index, err) ||
        !float_array(f, prefix + "tok_num_value", &out->num_value, err) ||
        !int_array(f, prefix + "tok_cat", &out->cat, err) || !int_array(f, prefix + "tok_card", &out->card, err) ||
        !int_array(f, prefix + "tok_ability", &out->ability, err) || !int_array(f, prefix + "rel", &out->rel, err))
        return false;
    if (!offsets_ok(out->tok_offsets, steps, out->tok_type.size()) || out->rel.size() % 3 ||
        !offsets_ok(out->rel_offsets, steps, out->rel.size() / 3) || out->num_index.size() != out->num_value.size() ||
        !offsets_ok(out->num_offsets, steps, out->num_index.size())) {
        *err =
            prefix + "tok_offsets, " + prefix + "rel_offsets or " + prefix + "tok_num_offsets do not tile their arrays";
        return false;
    }
    return true;
}

// Decision `d`'s tensors from the compact arrays. `pos` holds the per-type arrays' positions where
// the decision's values start (the sums of the widths of every token before it); the caller walks
// the decisions in order and it advances.
static bool expand_tokens(const TokenArrays &t, size_t d, const std::vector<Widths> &widths, size_t F, size_t C,
    size_t K, size_t Q, int64_t menu_type, featurize::Tensors *out, std::string *err, Widths *pos) {
    const size_t b = static_cast<size_t>(t.tok_offsets[d]), e = static_cast<size_t>(t.tok_offsets[d + 1]);
    const size_t n = e - b;
    out->tok_type.assign(
        t.tok_type.begin() + static_cast<std::ptrdiff_t>(b), t.tok_type.begin() + static_cast<std::ptrdiff_t>(e));
    out->tok_num.assign(n * F, 0.0f);
    out->tok_cat.assign(n * C, 0);
    out->tok_card.assign(n * K, -1);
    out->tok_ability.assign(n * Q, -1);
    out->tok_mask.assign(n, 1);
    out->act_tok.clear();
    for (size_t i = 0; i < n; ++i) {
        const int64_t ty = out->tok_type[i];
        if (ty < 0 || static_cast<size_t>(ty) >= widths.size()) {
            *err = "a token type out of range";
            return false;
        }
        const Widths &w = widths[static_cast<size_t>(ty)];
        if (w.cat > C || w.card > K || w.ability > Q || pos->cat + w.cat > t.cat.size() ||
            pos->card + w.card > t.card.size() || pos->ability + w.ability > t.ability.size()) {
            *err = "the per-type arrays end before their tokens' widths";
            return false;
        }
        std::copy_n(t.cat.begin() + static_cast<std::ptrdiff_t>(pos->cat), w.cat, out->tok_cat.begin() + i * C);
        std::copy_n(t.card.begin() + static_cast<std::ptrdiff_t>(pos->card), w.card, out->tok_card.begin() + i * K);
        std::copy_n(
            t.ability.begin() + static_cast<std::ptrdiff_t>(pos->ability), w.ability, out->tok_ability.begin() + i * Q);
        pos->cat += w.cat;
        pos->card += w.card;
        pos->ability += w.ability;
        if (ty == menu_type) out->act_tok.push_back(static_cast<int64_t>(i));
    }
    for (size_t k = static_cast<size_t>(t.num_offsets[d]); k < static_cast<size_t>(t.num_offsets[d + 1]); ++k) {
        if (t.num_index[k] < 0 || static_cast<size_t>(t.num_index[k]) >= n * F) {
            *err = "a numeric feature's index is past its decision's tokens";
            return false;
        }
        out->tok_num[static_cast<size_t>(t.num_index[k])] = t.num_value[k];
    }
    const size_t rb = static_cast<size_t>(t.rel_offsets[d]) * 3, re = static_cast<size_t>(t.rel_offsets[d + 1]) * 3;
    out->rel.assign(t.rel.begin() + static_cast<std::ptrdiff_t>(rb), t.rel.begin() + static_cast<std::ptrdiff_t>(re));
    for (size_t i = 0; i < out->rel.size(); i += 3)
        if (out->rel[i] < 0 || out->rel[i + 1] < 0 || static_cast<size_t>(out->rel[i]) >= n ||
            static_cast<size_t>(out->rel[i + 1]) >= n) {
            *err = "a relation names no token of its decision";
            return false;
        }
    out->act_mask.assign(out->act_tok.size(), 1);
    return true;
}

// The ragged list `name`: its values and `<name>_offsets`.
static bool read_list(const featurize::RmtbFile &f, const std::string &name, size_t steps, std::vector<int64_t> *values,
    std::vector<int64_t> *offsets, std::string *err) {
    if (!int_array(f, name, values, err) || !int_array(f, name + "_offsets", offsets, err)) return false;
    if (!offsets_ok(*offsets, steps, values->size())) {
        *err = name + "_offsets do not tile " + name;
        return false;
    }
    return true;
}

// Decision `d`'s entries of a ragged list.
static void slice(
    const std::vector<int64_t> &values, const std::vector<int64_t> &offsets, size_t d, std::vector<int64_t> *out) {
    out->assign(values.begin() + offsets[d], values.begin() + offsets[d + 1]);
}

namespace featurize {

const char *const EPISODE_KIND = "episode";

const char *end_reason_name(EndReason r) {
    switch (r) {
        case EndReason::WIN:
            return "win";
        case EndReason::CONCESSION:
            return "concession";
        case EndReason::DRAW:
            return "draw";
        case EndReason::CAP:
            return "cap";
        case EndReason::QUIT:
            return "quit";
    }
    return "";
}

EpisodeWriter::EpisodeWriter(const FeatureSpec &spec, bool omniscient) : spec_(spec), omniscient_(omniscient) {}

bool EpisodeWriter::compact(const Tensors &t, Tokens *out, std::string *err) const {
    const size_t n = t.n();
    if (t.tok_num.size() != n * spec_.F || t.tok_cat.size() != n * spec_.C || t.tok_card.size() != n * spec_.K ||
        t.tok_ability.size() != n * spec_.Q || t.rel.size() % 3 || t.act_mask.size() != t.a()) {
        *err = "episode: a decision's tensors are not the spec's shapes";
        return false;
    }
    size_t menu = 0;
    bool menu_in_order = true;
    for (size_t i = 0; i < n; ++i) {
        const int64_t ty = t.tok_type[i];
        if (ty < 0 || static_cast<size_t>(ty) >= spec_.types.size() || ty > 0xFF) {
            *err = "episode: a token type out of the spec's range";
            return false;
        }
        if (ty != spec_.menu_type) continue;
        menu_in_order = menu_in_order && menu < t.a() && t.act_tok[menu] == static_cast<int64_t>(i);
        ++menu;
    }
    if (!menu_in_order || menu != t.a()) {
        *err = "episode: act_tok is not the menu tokens in order";
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        const TokenType &type = spec_.types[static_cast<size_t>(t.tok_type[i])];
        const std::ptrdiff_t at = static_cast<std::ptrdiff_t>(i);
        out->tok_type.push_back(t.tok_type[i]);
        for (size_t k = i * spec_.F; k < i * spec_.F + type.numeric.size(); ++k) {
            const uint16_t h = f32_to_f16(t.tok_num[k]);
            if (h == 0) continue;
            out->num_index.push_back(static_cast<int64_t>(k));
            out->num_value.push_back(f16_to_f32(h));
        }
        out->cat.insert(out->cat.end(), t.tok_cat.begin() + at * static_cast<std::ptrdiff_t>(spec_.C),
            t.tok_cat.begin() + at * static_cast<std::ptrdiff_t>(spec_.C) +
                static_cast<std::ptrdiff_t>(type.categorical.size()));
        out->card.insert(out->card.end(), t.tok_card.begin() + at * static_cast<std::ptrdiff_t>(spec_.K),
            t.tok_card.begin() + at * static_cast<std::ptrdiff_t>(spec_.K) +
                static_cast<std::ptrdiff_t>(type.cards.size()));
        out->ability.insert(out->ability.end(), t.tok_ability.begin() + at * static_cast<std::ptrdiff_t>(spec_.Q),
            t.tok_ability.begin() + at * static_cast<std::ptrdiff_t>(spec_.Q) +
                static_cast<std::ptrdiff_t>(type.abilities.size()));
    }
    out->rel = t.rel;
    out->num_offsets.push_back(static_cast<int64_t>(out->num_index.size()));
    out->tok_offsets.push_back(static_cast<int64_t>(n));
    out->rel_offsets.push_back(static_cast<int64_t>(t.r()));
    return true;
}

void EpisodeWriter::append(const Tokens &one, Tokens *to) {
    to->tok_type.insert(to->tok_type.end(), one.tok_type.begin(), one.tok_type.end());
    to->num_index.insert(to->num_index.end(), one.num_index.begin(), one.num_index.end());
    to->num_value.insert(to->num_value.end(), one.num_value.begin(), one.num_value.end());
    to->num_offsets.push_back(to->num_offsets.back() + one.num_offsets.back());
    to->cat.insert(to->cat.end(), one.cat.begin(), one.cat.end());
    to->card.insert(to->card.end(), one.card.begin(), one.card.end());
    to->ability.insert(to->ability.end(), one.ability.begin(), one.ability.end());
    to->rel.insert(to->rel.end(), one.rel.begin(), one.rel.end());
    to->tok_offsets.push_back(to->tok_offsets.back() + one.tok_offsets.back());
    to->rel_offsets.push_back(to->rel_offsets.back() + one.rel_offsets.back());
}

bool EpisodeWriter::add(const EpisodeStep &s, std::string *err) {
    const size_t a = s.tensors.a();
    if (s.has_mu && s.mu.size() != a) {
        *err = "episode: mu is not over the menu rows";
        return false;
    }
    for (const PolicyEvalRecord &p : s.policies)
        if (p.log_pi.size() != a) {
            *err = "episode: a policy's log pi is not over the menu rows";
            return false;
        }
    if (omniscient_ && s.omniscient.a() != a) {
        *err = "episode: the omniscient frame's menu is not the seat's";
        return false;
    }
    // Both featurizations are compacted before either is appended, so a refused step leaves no trace.
    Tokens seat_one, omni_one;
    if (!compact(s.tensors, &seat_one, err)) return false;
    if (omniscient_ && !compact(s.omniscient, &omni_one, err)) return false;
    append(seat_one, &seat_tokens_);
    if (omniscient_) append(omni_one, &omni_tokens_);
    seat_.push_back(s.seat);
    game_.push_back(s.game);
    decision_.push_back(s.decision);
    sideboarding_.push_back(s.sideboarding ? 1 : 0);
    chosen_.push_back(s.chosen);
    has_mu_.push_back(s.has_mu ? 1 : 0);
    policy_version_.push_back(s.policy_version);
    teacher_.push_back(s.teacher);
    menu_offsets_.push_back(menu_offsets_.back() + static_cast<int64_t>(a));
    if (s.has_mu)
        mu_.insert(mu_.end(), s.mu.begin(), s.mu.end());
    else
        mu_.insert(mu_.end(), a, 0.0f);
    policies_.push_back(s.policies);
    max_policies_ = std::max(max_policies_, s.policies.size());
    if (!omniscient_) return true;
    append_list(s.targets.opp_hand, &opp_hand_, &opp_hand_offsets_);
    append_list(s.targets.own_library, &own_library_, &own_library_offsets_);
    append_list(s.targets.opp_library, &opp_library_, &opp_library_offsets_);
    return true;
}

void EpisodeWriter::append_list(
    const std::vector<int64_t> &one, std::vector<int64_t> *values, std::vector<int64_t> *offsets) {
    values->insert(values->end(), one.begin(), one.end());
    offsets->push_back(offsets->back() + static_cast<int64_t>(one.size()));
}

void EpisodeWriter::tokens_arrays(const Tokens &t, const std::string &prefix, RmtbFile *f) const {
    const uint64_t d = t.tok_offsets.size() - 1;
    const uint64_t n = t.tok_type.size();
    f->arrays.push_back(rmtb_ints(prefix + "tok_offsets", Dtype::I64, {d + 1}, t.tok_offsets));
    f->arrays.push_back(rmtb_ints(prefix + "rel_offsets", Dtype::I64, {d + 1}, t.rel_offsets));
    f->arrays.push_back(rmtb_ints(prefix + "tok_type", Dtype::U8, {n}, t.tok_type));
    f->arrays.push_back(rmtb_ints(prefix + "tok_num_offsets", Dtype::I64, {d + 1}, t.num_offsets));
    f->arrays.push_back(
        rmtb_ints(prefix + "tok_num_index", index_dtype(t.num_index), {t.num_index.size()}, t.num_index));
    f->arrays.push_back(rmtb_floats(prefix + "tok_num_value", Dtype::F16, {t.num_value.size()}, t.num_value));
    f->arrays.push_back(rmtb_ints(
        prefix + "tok_cat", spec_.categorical_size <= 0x10000 ? Dtype::U16 : Dtype::I32, {t.cat.size()}, t.cat));
    f->arrays.push_back(
        rmtb_ints(prefix + "tok_card", spec_.card_rows <= 0x8000 ? Dtype::I16 : Dtype::I32, {t.card.size()}, t.card));
    f->arrays.push_back(rmtb_ints(
        prefix + "tok_ability", spec_.ability_rows <= 0x8000 ? Dtype::I16 : Dtype::I32, {t.ability.size()}, t.ability));
    f->arrays.push_back(rmtb_ints(prefix + "rel", index_dtype(t.rel), {t.rel.size() / 3, 3}, t.rel));
}

RmtbFile EpisodeWriter::file(const std::string &text_model, const std::vector<RmtbAttr> &attrs) const {
    RmtbFile f;
    f.kind = EPISODE_KIND;
    f.has_vocab_hash = f.has_spec_hash = true;
    f.vocab_hash = spec_.vocab_hash;
    f.spec_hash = spec_.hash;
    f.model = text_model;
    f.set_attr("episode_version", EPISODE_VERSION);
    f.set_attr("F", static_cast<int64_t>(spec_.F));
    f.set_attr("C", static_cast<int64_t>(spec_.C));
    f.set_attr("K", static_cast<int64_t>(spec_.K));
    f.set_attr("Q", static_cast<int64_t>(spec_.Q));
    f.set_attr("menu_type", spec_.menu_type);
    f.set_attr("omniscient", omniscient_ ? 1 : 0);
    f.attrs.insert(f.attrs.end(), attrs.begin(), attrs.end());

    const uint64_t d = seat_.size(), a = static_cast<uint64_t>(menu_offsets_.back());
    std::vector<int64_t> widths;
    for (const TokenType &t : spec_.types) {
        widths.push_back(static_cast<int64_t>(t.numeric.size()));
        widths.push_back(static_cast<int64_t>(t.categorical.size()));
        widths.push_back(static_cast<int64_t>(t.cards.size()));
        widths.push_back(static_cast<int64_t>(t.abilities.size()));
    }
    RmtbArray w = rmtb_ints("type_widths", Dtype::I32, {spec_.types.size(), 4}, widths);
    w.columns = {"num", "cat", "card", "ability"};
    f.arrays.push_back(w);
    f.arrays.push_back(rmtb_ints("seat", Dtype::U8, {d}, seat_));
    f.arrays.push_back(rmtb_ints("game", Dtype::I32, {d}, game_));
    f.arrays.push_back(rmtb_ints("decision", Dtype::I32, {d}, decision_));
    f.arrays.push_back(rmtb_ints("sideboarding", Dtype::U8, {d}, sideboarding_));
    f.arrays.push_back(rmtb_ints("chosen", Dtype::I32, {d}, chosen_));
    f.arrays.push_back(rmtb_ints("has_mu", Dtype::U8, {d}, has_mu_));
    f.arrays.push_back(rmtb_ints("policy_version", Dtype::I64, {d}, policy_version_));
    f.arrays.push_back(rmtb_ints("teacher", Dtype::I32, {d}, teacher_));
    f.arrays.push_back(rmtb_ints("menu_offsets", Dtype::I64, {d + 1}, menu_offsets_));
    tokens_arrays(seat_tokens_, "", &f);
    f.arrays.push_back(rmtb_floats("mu", Dtype::F32, {a}, mu_));
    if (max_policies_ > 0) {
        const size_t p = max_policies_;
        std::vector<int64_t> ids, versions;
        std::vector<float> values, log_pi;
        for (size_t i = 0; i < policies_.size(); ++i) {
            const std::vector<PolicyEvalRecord> &ps = policies_[i];
            for (size_t k = 0; k < p; ++k) {
                ids.push_back(k < ps.size() ? ps[k].id : -1);
                versions.push_back(k < ps.size() ? static_cast<int64_t>(ps[k].version) : 0);
                values.push_back(k < ps.size() ? ps[k].value : 0.0f);
            }
            const size_t rows = static_cast<size_t>(menu_offsets_[i + 1] - menu_offsets_[i]);
            for (size_t r = 0; r < rows; ++r)
                for (size_t k = 0; k < p; ++k) log_pi.push_back(k < ps.size() ? ps[k].log_pi[r] : 0.0f);
        }
        f.arrays.push_back(rmtb_ints("eval_policy", Dtype::I64, {d, p}, ids));
        f.arrays.push_back(rmtb_ints("eval_version", Dtype::I64, {d, p}, versions));
        f.arrays.push_back(rmtb_floats("eval_value", Dtype::F32, {d, p}, values));
        f.arrays.push_back(rmtb_floats("eval_log_pi", Dtype::F32, {a, p}, log_pi));
    }
    if (omniscient_) {
        tokens_arrays(omni_tokens_, "omni_", &f);
        const Dtype card = spec_.card_rows <= 0x8000 ? Dtype::I16 : Dtype::I32;
        f.arrays.push_back(rmtb_ints("opp_hand", card, {opp_hand_.size()}, opp_hand_));
        f.arrays.push_back(rmtb_ints("opp_hand_offsets", Dtype::I64, {d + 1}, opp_hand_offsets_));
        f.arrays.push_back(rmtb_ints("own_library", card, {own_library_.size()}, own_library_));
        f.arrays.push_back(rmtb_ints("own_library_offsets", Dtype::I64, {d + 1}, own_library_offsets_));
        f.arrays.push_back(rmtb_ints("opp_library", card, {opp_library_.size()}, opp_library_));
        f.arrays.push_back(rmtb_ints("opp_library_offsets", Dtype::I64, {d + 1}, opp_library_offsets_));
    }
    return f;
}

bool read_episode(const RmtbFile &file, Episode *out, std::string *err) {
    *out = Episode();
    out->file = file;
    int64_t version = 0, F = 0, C = 0, K = 0, Q = 0, menu_type = 0, omniscient = 0;
    if (file.kind != EPISODE_KIND) {
        *err = "not an episode file (kind " + file.kind + ")";
        return false;
    }
    if (!int_attr(file, "episode_version", &version, err)) return false;
    if (version != EPISODE_VERSION) {
        *err = "episode version " + std::to_string(version) + ", expected " + std::to_string(EPISODE_VERSION);
        return false;
    }
    if (!int_attr(file, "F", &F, err) || !int_attr(file, "C", &C, err) || !int_attr(file, "K", &K, err) ||
        !int_attr(file, "Q", &Q, err) || !int_attr(file, "menu_type", &menu_type, err) ||
        !int_attr(file, "omniscient", &omniscient, err))
        return false;
    out->omniscient = omniscient != 0;
    std::vector<int64_t> widths_flat, seat, game, decision, sideboarding, chosen, has_mu, policy_version, teacher,
        menu_offsets;
    std::vector<float> mu;
    if (!int_array(file, "type_widths", &widths_flat, err)) return false;
    if (widths_flat.size() % 4) {
        *err = "type_widths is not [T, 4]";
        return false;
    }
    if (!int_array(file, "seat", &seat, err) || !int_array(file, "game", &game, err) ||
        !int_array(file, "decision", &decision, err) || !int_array(file, "sideboarding", &sideboarding, err) ||
        !int_array(file, "chosen", &chosen, err) || !int_array(file, "has_mu", &has_mu, err) ||
        !int_array(file, "policy_version", &policy_version, err) || !int_array(file, "teacher", &teacher, err) ||
        !int_array(file, "menu_offsets", &menu_offsets, err) || !float_array(file, "mu", &mu, err))
        return false;
    std::vector<Widths> widths;
    for (size_t i = 0; i < widths_flat.size(); i += 4)
        widths.push_back(Widths{static_cast<size_t>(widths_flat[i]), static_cast<size_t>(widths_flat[i + 1]),
            static_cast<size_t>(widths_flat[i + 2]), static_cast<size_t>(widths_flat[i + 3])});
    const size_t d = seat.size();
    for (const std::vector<int64_t> *v : {&game, &decision, &sideboarding, &chosen, &has_mu, &policy_version, &teacher})
        if (v->size() != d) {
            *err = "the per-decision arrays differ in length";
            return false;
        }
    if (!offsets_ok(menu_offsets, d, mu.size())) {
        *err = "menu_offsets do not tile mu";
        return false;
    }
    TokenArrays seat_tokens, omni_tokens;
    if (!read_tokens(file, "", d, &seat_tokens, err)) return false;
    if (out->omniscient && !read_tokens(file, "omni_", d, &omni_tokens, err)) return false;
    std::vector<int64_t> eval_policy, eval_version;
    std::vector<float> eval_value, eval_log_pi;
    if (const RmtbArray *ep = file.find("eval_policy")) {
        if (ep->shape.size() != 2 || ep->shape[0] != d) {
            *err = "eval_policy is not [D, P]";
            return false;
        }
        out->policies = static_cast<size_t>(ep->shape[1]);
        if (!int_array(file, "eval_policy", &eval_policy, err) ||
            !int_array(file, "eval_version", &eval_version, err) ||
            !float_array(file, "eval_value", &eval_value, err) || !float_array(file, "eval_log_pi", &eval_log_pi, err))
            return false;
        if (eval_version.size() != d * out->policies || eval_value.size() != d * out->policies ||
            eval_log_pi.size() != mu.size() * out->policies) {
            *err = "the eval_* arrays are not [D, P] / [sum A, P]";
            return false;
        }
    }
    std::vector<int64_t> opp_hand, opp_hand_off, own_library, own_library_off, opp_library, opp_library_off;
    if (out->omniscient && (!read_list(file, "opp_hand", d, &opp_hand, &opp_hand_off, err) ||
                               !read_list(file, "own_library", d, &own_library, &own_library_off, err) ||
                               !read_list(file, "opp_library", d, &opp_library, &opp_library_off, err)))
        return false;
    Widths seat_pos, omni_pos;
    const size_t p = out->policies;
    for (size_t i = 0; i < d; ++i) {
        EpisodeStep s;
        s.seat = static_cast<uint8_t>(seat[i]);
        s.game = game[i];
        s.decision = decision[i];
        s.sideboarding = sideboarding[i] != 0;
        s.chosen = chosen[i];
        s.has_mu = has_mu[i] != 0;
        s.policy_version = policy_version[i];
        s.teacher = teacher[i];
        const size_t mb = static_cast<size_t>(menu_offsets[i]), me = static_cast<size_t>(menu_offsets[i + 1]);
        if (s.has_mu)
            s.mu.assign(mu.begin() + static_cast<std::ptrdiff_t>(mb), mu.begin() + static_cast<std::ptrdiff_t>(me));
        if (!expand_tokens(seat_tokens, i, widths, static_cast<size_t>(F), static_cast<size_t>(C),
                static_cast<size_t>(K), static_cast<size_t>(Q), menu_type, &s.tensors, err, &seat_pos))
            return false;
        if (s.tensors.a() != me - mb) {
            *err = "decision " + std::to_string(i) + ": its menu tokens are not its menu_offsets rows";
            return false;
        }
        s.tensors.viewer = s.seat;
        s.tensors.game = s.game;
        s.tensors.decision = s.decision;
        s.tensors.sideboarding = s.sideboarding;
        for (size_t k = 0; k < p; ++k) {
            if (eval_policy[i * p + k] < 0) continue;
            PolicyEvalRecord r;
            r.id = eval_policy[i * p + k];
            r.version = static_cast<uint64_t>(eval_version[i * p + k]);
            r.value = eval_value[i * p + k];
            for (size_t row = mb; row < me; ++row) r.log_pi.push_back(eval_log_pi[row * p + k]);
            s.policies.push_back(r);
        }
        if (out->omniscient) {
            if (!expand_tokens(omni_tokens, i, widths, static_cast<size_t>(F), static_cast<size_t>(C),
                    static_cast<size_t>(K), static_cast<size_t>(Q), menu_type, &s.omniscient, err, &omni_pos))
                return false;
            s.omniscient.viewer = static_cast<uint8_t>(rmstate::Viewer::OMNISCIENT);
            s.omniscient.game = s.game;
            s.omniscient.decision = s.decision;
            s.omniscient.sideboarding = s.sideboarding;
            s.targets.seat = s.seat;
            s.targets.game = s.game;
            s.targets.decision = s.decision;
            s.targets.sideboarding = s.sideboarding;
            slice(opp_hand, opp_hand_off, i, &s.targets.opp_hand);
            slice(own_library, own_library_off, i, &s.targets.own_library);
            slice(opp_library, opp_library_off, i, &s.targets.opp_library);
        }
        out->steps.push_back(std::move(s));
    }
    if (seat_pos.cat != seat_tokens.cat.size() || seat_pos.card != seat_tokens.card.size() ||
        seat_pos.ability != seat_tokens.ability.size() ||
        (out->omniscient && (omni_pos.cat != omni_tokens.cat.size() || omni_pos.card != omni_tokens.card.size() ||
                                omni_pos.ability != omni_tokens.ability.size()))) {
        *err = "the per-type arrays hold more values than their tokens' widths";
        return false;
    }
    return true;
}

}  // namespace featurize
