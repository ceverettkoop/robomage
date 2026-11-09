#include "compat.h"

#include <map>
#include <set>

#include "json.h"
#include "registry.h"

namespace {

// The most names a refusal lists.
const size_t SHOWN = 4;
const char *const REBUILD =
    "; an export is built for one vocab and tables: rebuild it from its checkpoint on the current tables "
    "(docs/featurizer.md §6.2)";

}  // namespace

static std::string listed(const std::vector<std::string> &items);
static std::string card_text(uint64_t id, const std::string &name);
static featurize::Refusal refuse(const std::string &code, const std::string &message, bool exact);
static featurize::Refusal legacy_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact);
static featurize::Refusal vocab_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact);
static featurize::Refusal registry_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact);
static bool string_member(const featurize::Json &o, const char *key, std::string *out, std::string *err);

// The first few of `items`, comma-separated, and how many more.
static std::string listed(const std::vector<std::string> &items) {
    std::string out;
    for (size_t i = 0; i < items.size() && i < SHOWN; ++i) out += (i ? ", " : "") + items[i];
    if (items.size() > SHOWN) out += " and " + std::to_string(items.size() - SHOWN) + " more";
    return out;
}

static std::string card_text(uint64_t id, const std::string &name) {
    return "card " + std::to_string(id) + " '" + name + "'";
}

static featurize::Refusal refuse(const std::string &code, const std::string &message, bool exact) {
    featurize::Refusal r;
    r.code = code;
    r.message = message + (exact ? REBUILD : "");
    return r;
}

// A model made before manifests: its tables' own hashes only.
static featurize::Refusal legacy_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact) {
    const std::pair<const char *, std::pair<const std::string *, const std::string *>> fields[] = {
        {"spec_hash", {&model.spec_hash, &now.spec_hash}}, {"vocab_hash", {&model.vocab_hash, &now.vocab_hash}},
        {"text_model", {&model.text_model, &now.text_model}}};
    for (const auto &f : fields)
        if (*f.second.first != *f.second.second)
            return refuse(f.first,
                "made before model manifests (no layout_hash), its " + std::string(f.first) + " " + *f.second.first +
                    " is not the tables' " + *f.second.second,
                exact);
    return {};
}

// Every (id, name) of the model's vocab in the current one; exact: no current id the model lacks.
static featurize::Refusal vocab_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact) {
    std::map<uint64_t, const std::string *> cur;
    for (const auto &e : now.vocab) cur[e.first] = &e.second;
    std::vector<std::string> gone, renamed, added;
    std::set<uint64_t> known;
    for (const auto &e : model.vocab) {
        known.insert(e.first);
        auto it = cur.find(e.first);
        if (it == cur.end())
            gone.push_back(card_text(e.first, e.second));
        else if (*it->second != e.second)
            renamed.push_back("card " + std::to_string(e.first) + " is '" + e.second + "' in the model's vocab, '" +
                              *it->second + "' now");
    }
    if (!gone.empty())
        return refuse("vocab_hash", "the current vocab lacks " + listed(gone) + " (an id removed)", exact);
    if (!renamed.empty()) return refuse("vocab_hash", listed(renamed) + " (an id renamed or reused)", exact);
    if (!exact) return {};
    for (const auto &e : now.vocab)
        if (!known.count(e.first)) added.push_back(card_text(e.first, e.second));
    if (!added.empty())
        return refuse("vocab_hash", "the current vocab has " + listed(added) + ", which the model's lacks", exact);
    return {};
}

// No entry of the model's registry moved in the current one (both carried).
static featurize::Refusal registry_check(const featurize::Manifest &model, const featurize::Manifest &now, bool exact) {
    if (model.registry.empty() || now.registry.empty()) return {};
    featurize::Registry base, cur;
    std::string err;
    if (!featurize::Registry::parse(model.registry, &base, &err))
        return refuse("layout_hash", "the model's layout registry does not parse: " + err, exact);
    if (!featurize::Registry::parse(now.registry, &cur, &err))
        return refuse("layout_hash", "the current layout registry does not parse: " + err, exact);
    const std::vector<std::string> moves = featurize::registry_moves(base, cur);
    if (!moves.empty())
        return refuse(
            "layout_hash", "the layout registry moved what the model was trained on: " + listed(moves), exact);
    return {};
}

static bool string_member(const featurize::Json &o, const char *key, std::string *out, std::string *err) {
    const featurize::Json *v = o.get(key);
    if (!v) return true;
    if (!v->is_string()) {
        *err = std::string("manifest: ") + key + " is not a string";
        return false;
    }
    *out = v->as_string();
    return true;
}

namespace featurize {

Refusal check_compatible(const Manifest &model, const Manifest &now, bool exact) {
    if (model.layout_hash.empty()) return legacy_check(model, now, exact);
    if (model.layout_hash != now.layout_hash)
        return refuse("layout_hash",
            "its layout_hash " + model.layout_hash + " is not the tables' " + now.layout_hash +
                " (a layout bump: migrate the checkpoint)",
            exact);
    if (model.text_model != now.text_model)
        return refuse("text_model",
            "its text_model " + model.text_model + " is not the text table's " + now.text_model +
                " (a re-pinned text model)",
            exact);
    Refusal r = vocab_check(model, now, exact);
    if (!r.refused()) r = registry_check(model, now, exact);
    if (r.refused() || !exact) return r;
    if (model.spec_hash != now.spec_hash)
        return refuse("spec_hash", "its spec_hash " + model.spec_hash + " is not the tables' " + now.spec_hash, exact);
    if (model.vocab_hash != now.vocab_hash)
        return refuse(
            "vocab_hash", "its vocab_hash " + model.vocab_hash + " is not the vocab's " + now.vocab_hash, exact);
    if (!model.tables_digest.empty() && !now.tables_digest.empty() && model.tables_digest != now.tables_digest)
        return refuse("tables_digest",
            "its tables' digest " + model.tables_digest + " is not the current tables' " + now.tables_digest +
                " (a known card's rows changed: the table drift, rmfeat --table-drift)",
            exact);
    return {};
}

bool vocab_from_json(const Json &j, VocabManifest *out, std::string *err) {
    out->clear();
    if (!j.is_array()) {
        *err = "the vocab manifest is not a JSON array";
        return false;
    }
    for (const Json &e : j.items()) {
        if (!e.is_array() || e.items().size() != 2 || !e.items()[0].is_integer() || e.items()[0].as_int() < 0 ||
            !e.items()[1].is_string()) {
            *err = "a vocab manifest entry is not [id, name]";
            return false;
        }
        out->emplace_back(static_cast<uint64_t>(e.items()[0].as_int()), e.items()[1].as_string());
    }
    return true;
}

std::string vocab_json(const VocabManifest &vocab) {
    std::string out = "[";
    for (size_t i = 0; i < vocab.size(); ++i)
        out +=
            std::string(i ? "," : "") + "[" + std::to_string(vocab[i].first) + "," + json_quote(vocab[i].second) + "]";
    return out + "]";
}

bool parse_vocab_json(const std::string &text, VocabManifest *out, std::string *err) {
    Json j;
    return Json::parse(text, &j, err) && vocab_from_json(j, out, err);
}

std::string manifest_json(const Manifest &m, bool with_registry) {
    std::string out = "{\"spec_hash\":" + json_quote(m.spec_hash) + ",\"vocab_hash\":" + json_quote(m.vocab_hash) +
                      ",\"text_model\":" + json_quote(m.text_model) + ",\"layout_hash\":" + json_quote(m.layout_hash) +
                      ",\"tables_digest\":" + json_quote(m.tables_digest) + ",\"vocab\":" + vocab_json(m.vocab);
    if (with_registry) out += ",\"registry\":" + json_quote(m.registry);
    return out + "}";
}

bool parse_manifest_json(const std::string &text, Manifest *out, std::string *err) {
    Json j;
    if (!Json::parse(text, &j, err)) return false;
    if (!j.is_object()) {
        *err = "the manifest is not a JSON object";
        return false;
    }
    *out = Manifest();
    if (!string_member(j, "spec_hash", &out->spec_hash, err) ||
        !string_member(j, "vocab_hash", &out->vocab_hash, err) ||
        !string_member(j, "text_model", &out->text_model, err) ||
        !string_member(j, "layout_hash", &out->layout_hash, err) ||
        !string_member(j, "tables_digest", &out->tables_digest, err) ||
        !string_member(j, "registry", &out->registry, err))
        return false;
    const Json *v = j.get("vocab");
    return !v || vocab_from_json(*v, &out->vocab, err);
}

}  // namespace featurize
