#include "card_images.h"

#include <algorithm>
#include <cctype>

#include "../../src/card_vocab.h"
#include "../../src/featurize/json.h"
#include "files.h"
#include "paths.h"

namespace {

const char *const FETCH_SCRIPT = "tools/scryfall/fetch_images.py";

}  // namespace

static std::string str_of(const featurize::Json &obj, const char *key);
static std::string name_key(const std::string &name);
static bool read_printing(const featurize::Json &v, rmwork::ImagePrinting *out);
static bool read_card(const featurize::Json &v, rmwork::ImageCard *out);
static std::string shell_word(const std::string &s);

static std::string str_of(const featurize::Json &obj, const char *key) {
    const featurize::Json *v = obj.get(key);
    return v && v->is_string() ? v->as_string() : std::string();
}

// A name compared case- and accent-insensitively, without a trailing " Token".
static std::string name_key(const std::string &name) {
    std::string s = ascii_fold_card_name(name);
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const std::string suffix = " token";
    if (s.size() > suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0)
        s.resize(s.size() - suffix.size());
    return s;
}

static bool read_printing(const featurize::Json &v, rmwork::ImagePrinting *out) {
    if (!v.is_object()) return false;
    out->set = str_of(v, "set");
    out->set_name = str_of(v, "set_name");
    out->number = str_of(v, "number");
    out->released = str_of(v, "released");
    out->set_type = str_of(v, "set_type");
    const featurize::Json *promo = v.get("promo");
    out->promo = promo && promo->truthy();
    const featurize::Json *ok = v.get("default_ok");
    out->default_ok = ok && ok->truthy();
    const featurize::Json *faces = v.get("faces");
    if (!faces || !faces->is_array()) return false;
    for (const featurize::Json &f : faces->items()) {
        if (!f.is_object()) return false;
        rmwork::ImageFace face;
        face.name = str_of(f, "name");
        face.url = str_of(f, "url");
        face.file = str_of(f, "file");
        if (face.file.empty() || face.file.find("..") != std::string::npos || face.file[0] == '/') return false;
        out->faces.push_back(std::move(face));
    }
    return !out->set.empty() && !out->number.empty();
}

static bool read_card(const featurize::Json &v, rmwork::ImageCard *out) {
    if (!v.is_object()) return false;
    out->key = str_of(v, "key");
    out->stem = str_of(v, "stem");
    out->scryfall_name = str_of(v, "scryfall_name");
    const featurize::Json *index = v.get("index");
    out->index = index && index->is_number() ? static_cast<int>(index->as_int()) : -1;
    const featurize::Json *token = v.get("token");
    out->token = token && token->truthy();
    if (const featurize::Json *names = v.get("names"))
        for (const featurize::Json &n : names->items())
            if (n.is_string()) out->names.push_back(n.as_string());
    const featurize::Json *printings = v.get("printings");
    if (out->key.empty() || !printings || !printings->is_array()) return false;
    for (const featurize::Json &p : printings->items()) {
        rmwork::ImagePrinting pr;
        if (!read_printing(p, &pr)) return false;
        out->printings.push_back(std::move(pr));
    }
    const featurize::Json *chosen = v.get("chosen");
    out->chosen = chosen && chosen->is_number() ? static_cast<int>(chosen->as_int()) : 0;
    if (out->chosen < 0 || out->chosen >= static_cast<int>(out->printings.size())) out->chosen = 0;
    return true;
}

// `s` quoted for a POSIX shell when it needs it.
static std::string shell_word(const std::string &s) {
    bool plain = !s.empty();
    for (char c : s)
        if (!std::isalnum(static_cast<unsigned char>(c)) && std::string("-_./=:@%+,").find(c) == std::string::npos)
            plain = false;
    if (plain) return s;
    std::string out = "'";
    for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    return out + "'";
}

namespace rmwork {

std::string ImageCard::display() const {
    if (!token) return key;
    return (names.empty() ? stem : names.front()) + " token";
}

bool ImageManifest::parse(const std::string &text, std::string *err) {
    cards_.clear();
    missing_.clear();
    by_index_.clear();
    by_key_.clear();
    token_names_.clear();
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) return false;
    const featurize::Json *cards = doc.is_object() ? doc.get("cards") : nullptr;
    if (!cards || !cards->is_array()) {
        *err = "no \"cards\" list";
        return false;
    }
    for (const featurize::Json &c : cards->items()) {
        ImageCard card;
        if (!read_card(c, &card)) {
            *err = "a malformed card entry" + (card.key.empty() ? std::string() : " (" + card.key + ")");
            cards_.clear();
            return false;
        }
        cards_.push_back(std::move(card));
    }
    if (const featurize::Json *missing = doc.get("missing"))
        for (const featurize::Json &m : missing->items())
            if (m.is_object()) missing_.emplace_back(str_of(m, "key"), str_of(m, "why"));
    for (size_t i = 0; i < cards_.size(); ++i) {
        const ImageCard &c = cards_[i];
        if (c.index >= 0) by_index_.emplace(c.index, i);
        by_key_.emplace(c.key, i);
        if (c.token)
            for (const std::string &n : c.names) token_names_.emplace(name_key(n), i);
    }
    return true;
}

bool ImageManifest::load(const std::string &path, std::string *err) {
    std::string text;
    if (!read_file(path, &text)) {
        cards_.clear();
        missing_.clear();
        *err = file_exists(path) ? path + " cannot be read" : "";
        return false;
    }
    if (!parse(text, err)) {
        *err = path + ": " + *err;
        return false;
    }
    return true;
}

const ImageCard *ImageManifest::by_index(int index) const {
    auto it = by_index_.find(index);
    return it == by_index_.end() ? nullptr : &cards_[it->second];
}

const ImageCard *ImageManifest::by_key(const std::string &key) const {
    auto it = by_key_.find(key);
    return it == by_key_.end() ? nullptr : &cards_[it->second];
}

const ImageCard *ImageManifest::token_by_name(const std::string &name) const {
    auto it = token_names_.find(name_key(name));
    return it == token_names_.end() ? nullptr : &cards_[it->second];
}

const ImageCard *ImageManifest::card_for(const CardFace &face) const {
    if (const ImageCard *c = by_index(face.card)) return c;
    return face.token ? token_by_name(face.name) : nullptr;
}

std::string printings_to_json(const PrintingOverrides &o) {
    std::string out = "{";
    bool first = true;
    for (const auto &e : o) {
        out += std::string(first ? "\n" : ",\n") + "  " + featurize::json_quote(e.first) +
               ": {\"set\": " + featurize::json_quote(e.second.set) +
               ", \"number\": " + featurize::json_quote(e.second.number) + "}";
        first = false;
    }
    return out + (first ? "}\n" : "\n}\n");
}

bool printings_from_json(const std::string &text, PrintingOverrides *out, std::string *err) {
    out->clear();
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) return false;
    if (!doc.is_object()) {
        *err = "not an object";
        return false;
    }
    for (const auto &m : doc.members()) {
        PrintingChoice c;
        if (m.second.is_object()) {
            c.set = str_of(m.second, "set");
            c.number = str_of(m.second, "number");
        }
        if (c.set.empty() || c.number.empty()) {
            *err = "\"" + m.first + "\" has no set and number";
            out->clear();
            return false;
        }
        (*out)[m.first] = c;
    }
    return true;
}

bool load_printings(const std::string &path, PrintingOverrides *out, std::string *err) {
    out->clear();
    if (!file_exists(path)) return true;
    std::string text;
    if (!read_file(path, &text)) {
        *err = path + " cannot be read";
        return false;
    }
    if (!printings_from_json(text, out, err)) {
        *err = path + ": " + *err;
        return false;
    }
    return true;
}

bool save_printings(const std::string &path, const PrintingOverrides &o, std::string *err) {
    if (!make_dirs(parent_dir(path))) {
        *err = "cannot create " + parent_dir(path);
        return false;
    }
    return write_file_atomic(path, printings_to_json(o), err);
}

std::string printings_path(const std::string &config_dir) {
    return config_dir + "/printings.json";
}

int find_printing(const ImageCard &card, const std::string &set, const std::string &number) {
    for (size_t i = 0; i < card.printings.size(); ++i)
        if (card.printings[i].set == set && card.printings[i].number == number) return static_cast<int>(i);
    return -1;
}

const ImagePrinting *shown_printing(const ImageCard &card, const PrintingOverrides &overrides, bool *overridden) {
    if (overridden) *overridden = false;
    auto o = overrides.find(card.key);
    if (o != overrides.end()) {
        const int i = find_printing(card, o->second.set, o->second.number);
        if (i >= 0) {
            if (overridden) *overridden = true;
            return &card.printings[static_cast<size_t>(i)];
        }
    }
    if (card.printings.empty()) return nullptr;
    return &card.printings[static_cast<size_t>(card.chosen)];
}

int image_face_index(const ImagePrinting &p, const CardFace &face) {
    if (p.faces.empty()) return -1;
    if (p.faces.size() == 1) return 0;
    const std::string want = name_key(face.name);
    for (size_t i = 0; i < p.faces.size(); ++i)
        if (name_key(p.faces[i].name) == want) return static_cast<int>(i);
    return face.back ? 1 : 0;
}

std::string fetch_printing_command(const std::string &cache, const ImageCard &card, const ImagePrinting &p) {
    std::string cmd = std::string("python3 ") + FETCH_SCRIPT;
    if (cache != default_image_cache()) cmd += " --cache " + shell_word(cache);
    return cmd + " --printing " + shell_word(card.key + "=" + p.set + ":" + p.number);
}

std::vector<std::string> fetch_printing_argv(
    const std::string &repo, const std::string &cache, const ImageCard &card, const ImagePrinting &p) {
    return {"/usr/bin/env", "python3", repo + "/" + FETCH_SCRIPT, "--quiet", "--cache", cache, "--printing",
        card.key + "=" + p.set + ":" + p.number};
}

std::string fetch_all_command(const std::string &cache) {
    std::string cmd = std::string("python3 ") + FETCH_SCRIPT;
    if (cache != default_image_cache()) cmd += " --cache " + shell_word(cache);
    return cmd;
}

void CardImages::configure(const std::string &cache_dir, const std::string &printings_file) {
    if (configured_ && cache_dir == cache_dir_ && printings_file == printings_file_) return;
    cache_dir_ = cache_dir;
    printings_file_ = printings_file;
    configured_ = true;
    reload();
}

void CardImages::reload() {
    std::string err;
    has_manifest_ = manifest_.load(cache_dir_ + "/manifest.json", &err);
    manifest_error_ = has_manifest_ ? std::string() : err;
    overrides_error_.clear();
    if (!load_printings(printings_file_, &overrides_, &err)) overrides_error_ = err;
    exists_.clear();
    textures_.clear();
}

bool CardImages::set_override(const std::string &key, const PrintingChoice &choice, std::string *err) {
    PrintingOverrides next = overrides_;
    if (choice.set.empty()) {
        next.erase(key);
    } else {
        next[key] = choice;
    }
    if (!save_printings(printings_file_, next, err)) return false;
    overrides_ = next;
    return true;
}

bool CardImages::clear_overrides(std::string *err) {
    if (!save_printings(printings_file_, PrintingOverrides(), err)) return false;
    overrides_.clear();
    return true;
}

std::string CardImages::path_of(const ImageFace &f) const {
    return cache_dir_ + "/" + f.file;
}

bool CardImages::cached(const ImageFace &f) const {
    const std::string path = path_of(f);
    auto it = exists_.find(path);
    if (it != exists_.end()) return it->second;
    const bool e = path_kind(path) == PathKind::FILE;
    exists_[path] = e;
    return e;
}

bool CardImages::printing_cached(const ImagePrinting &p) const {
    if (p.faces.empty()) return false;
    for (const ImageFace &f : p.faces)
        if (!cached(f)) return false;
    return true;
}

std::string CardImages::file_for(const CardFace &face) const {
    if (!has_manifest_) return "";
    const ImageCard *card = manifest_.card_for(face);
    if (!card) return "";
    const ImagePrinting *p = shown_printing(*card, overrides_);
    if (!p) return "";
    const int i = image_face_index(*p, face);
    if (i < 0) return "";
    const ImageFace &f = p->faces[static_cast<size_t>(i)];
    return cached(f) ? path_of(f) : std::string();
}

ImageCacheStatus CardImages::status() const {
    ImageCacheStatus s;
    s.manifest = has_manifest_;
    s.error = manifest_error_;
    std::map<int, bool> vocab;
    for (int i = 0; i < CARD_VOCAB_SIZE; ++i) vocab[card_vocab_entries[i].index] = false;
    s.vocab = static_cast<int>(vocab.size());
    s.missing = static_cast<int>(manifest_.missing().size());
    if (!has_manifest_) return s;
    for (const ImageCard &c : manifest_.cards()) {
        const ImagePrinting *p = shown_printing(c, overrides_);
        const bool have = p && printing_cached(*p);
        if (c.token) {
            ++s.tokens;
            s.token_images += have ? 1 : 0;
        } else if (vocab.count(c.index) && have && !vocab[c.index]) {
            vocab[c.index] = true;
            ++s.with_images;
        }
    }
    return s;
}

}  // namespace rmwork
