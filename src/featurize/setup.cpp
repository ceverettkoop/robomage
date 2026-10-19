#include "setup.h"

#include <fstream>

#include "oracle_text.h"

namespace featurize {

const char *const Featurization::CARD_FILE = "card.rmtb";
const char *const Featurization::ABILITY_FILE = "ability.rmtb";
const char *const Featurization::PART_FILE = "part.rmtb";
const char *const Featurization::SPEC_FILE = "spec.json";

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
        if (!build_card_tables(source.schema, source.cards, vectors->shape[0], text.vocab_hash, &tables, err))
            return false;
    } else if (!read_tables_dir(tables_dir, &tables, &want_spec, err)) {
        return false;
    }
    if (tables.vocab_hash != text.vocab_hash || tables.card_rows != vectors->shape[0]) {
        *err = "the card tables are not of the text table's vocab";
        return false;
    }
    tables.text_rows = vectors->shape[0];
    tables.text_width = vectors->shape[1];
    text_model = text.model;
    if (!build_spec(source.schema, source.metagame, tables, &spec, err)) return false;
    if (!tables_dir.empty() && want_spec != spec.hash) {
        *err =
            tables_dir + ": the card tables carry spec hash " + hex16(want_spec) + ", the spec is " + hex16(spec.hash);
        return false;
    }
    return true;
}

bool Featurization::export_tables(const std::string &dir, std::string *err) const {
    const CardTableFiles files = card_table_files(tables, spec.hash);
    if (!rmtb_write_file(dir + "/" + CARD_FILE, files.card, err) ||
        !rmtb_write_file(dir + "/" + ABILITY_FILE, files.ability, err) ||
        !rmtb_write_file(dir + "/" + PART_FILE, files.part, err))
        return false;
    std::ofstream f(dir + "/" + SPEC_FILE, std::ios::binary | std::ios::trunc);
    f.write(spec.json.data(), static_cast<std::streamsize>(spec.json.size()));
    f.put('\n');
    if (!f.good()) {
        *err = "cannot write " + dir + "/" + SPEC_FILE;
        return false;
    }
    return true;
}

}  // namespace featurize
