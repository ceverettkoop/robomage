#include "setup.h"

#include <fstream>

#include "../state_format/schema.h"
#include "drift.h"
#include "oracle_text.h"

namespace featurize {

const char *const Featurization::CARD_FILE = "card.rmtb";
const char *const Featurization::ABILITY_FILE = "ability.rmtb";
const char *const Featurization::PART_FILE = "part.rmtb";
const char *const Featurization::SPEC_FILE = "spec.json";
const char *const Featurization::MANIFEST_FILE = "manifest.json";
const char *const Featurization::REGISTRY_FILE = "registry.json";

static bool write_text_file(const std::string &path, const std::string &text, std::string *err);

// `text` into `path`.
static bool write_text_file(const std::string &path, const std::string &text, std::string *err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!f.good()) {
        *err = "cannot write " + path;
        return false;
    }
    return true;
}

bool Featurization::read_tables_dir(const std::string &dir, CardTables *out, uint64_t *spec_hash, std::string *err) {
    CardTableFiles files;
    if (!rmtb_read_file(dir + "/" + CARD_FILE, &files.card, err) ||
        !rmtb_read_file(dir + "/" + ABILITY_FILE, &files.ability, err) ||
        !rmtb_read_file(dir + "/" + PART_FILE, &files.part, err) || !read_card_tables(files, out, err))
        return false;
    if (!files.card.has_spec_hash || files.ability.spec_hash != files.card.spec_hash ||
        files.part.spec_hash != files.card.spec_hash) {
        *err = dir + ": the card tables' files carry different spec hashes";
        return false;
    }
    *spec_hash = files.card.spec_hash;
    return true;
}

bool Featurization::load(const std::string &cards_path, const std::string &text_table_path,
    const std::string &tables_dir, std::string *err) {
    return read_match_frames(cards_path, &source, err) && build(text_table_path, tables_dir, err);
}

bool Featurization::load_records(
    const std::string &records, const std::string &text_table_path, const std::string &tables_dir, std::string *err) {
    return read_match_records(records, &source, err) && build(text_table_path, tables_dir, err);
}

bool Featurization::build(const std::string &text_table_path, const std::string &tables_dir, std::string *err) {
    RmtbFile text;
    if (!rmtb_read_file(text_table_path, &text, err)) return false;
    const RmtbArray *vectors = text_vectors(text);
    if (!vectors || !text.has_vocab_hash) {
        *err = text_table_path + ": not a text table with a vocab hash and an f32 vectors array";
        return false;
    }
    uint64_t want_spec = 0;
    if (tables_dir.empty()) {
        if (!build_card_tables(
                source.schema, source.cards, vectors->shape[0], text.vocab_hash, &registry, &tables, err))
            return false;
    } else if (!read_tables_dir(tables_dir, &tables, &want_spec, err)) {
        return false;
    } else {
        build_printed_rows(source.schema, source.cards, &tables);
    }
    if (tables.vocab_hash != text.vocab_hash || tables.card_rows != vectors->shape[0]) {
        *err = "the card tables are not of the text table's vocab";
        return false;
    }
    tables.text_rows = vectors->shape[0];
    tables.text_width = vectors->shape[1];
    text_model = text.model;
    if (!build_spec(source.schema, tables, options, &registry, &spec, err)) return false;
    if (!tables_dir.empty() && want_spec != spec.hash) {
        *err =
            tables_dir + ": the card tables carry spec hash " + hex16(want_spec) + ", the spec is " + hex16(spec.hash);
        return false;
    }
    return true;
}

bool Featurization::export_tables(const std::string &dir, std::string *err) const {
    const CardTableFiles files = card_table_files(tables, spec.hash);
    return rmtb_write_file(dir + "/" + CARD_FILE, files.card, err) &&
           rmtb_write_file(dir + "/" + ABILITY_FILE, files.ability, err) &&
           rmtb_write_file(dir + "/" + PART_FILE, files.part, err) &&
           write_text_file(dir + "/" + SPEC_FILE, spec.json + "\n", err) &&
           write_text_file(dir + "/" + MANIFEST_FILE, manifest_json(manifest(), false) + "\n", err) &&
           write_text_file(dir + "/" + REGISTRY_FILE, registry.json(), err);
}

Manifest Featurization::manifest() const {
    Manifest m;
    m.spec_hash = hex16(spec.hash);
    m.vocab_hash = hex16(spec.vocab_hash);
    m.text_model = text_model;
    m.layout_hash = hex16(spec.layout_hash);
    const std::string digests = digest_json(card_digests(tables, tables.names));
    m.tables_digest = hex16(rmstate::fnv1a64(digests.data(), digests.size()));
    for (size_t id = 0; id < tables.names.size(); ++id)
        if (id < tables.present.size() && tables.present[id]) m.vocab.emplace_back(id, tables.names[id]);
    m.registry = registry.json();
    return m;
}

}  // namespace featurize
