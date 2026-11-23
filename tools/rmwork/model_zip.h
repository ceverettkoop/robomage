#ifndef RMWORK_MODEL_ZIP_H
#define RMWORK_MODEL_ZIP_H

#include <cstdint>
#include <string>
#include <vector>

// ── Models packaged in zip archives ──────────────────────────────────────────────────────────────
//
// A model export (`*.ts.pt`, `*.pt2`) may be named inside a zip archive as `<archive>.zip!<member>`
// ("…/models/league26-24h-model.zip!runs/league26-24h/exports/final.ts.pt"); a bare `<archive>.zip`
// names the archive's one export. The workbench reads the archive itself (stored and deflated members,
// zlib) and hands the engine the member extracted into the workbench cache
// (`<cache>/models/<archive stem>-<member crc32>/<member>`), extracting it once and reusing it while
// its size matches. The archives the New Game dialog lists live in the models directory
// (`default_models_dir()`, <repo>/models).

namespace rmwork {

// One file in an archive's central directory.
struct ZipEntry {
        std::string name;  // its path inside the archive
        uint16_t method = 0;  // 0 stored, 8 deflated
        uint16_t flags = 0;
        uint32_t crc32 = 0;
        uint64_t compressed_size = 0;
        uint64_t size = 0;
        uint64_t local_offset = 0;
};

// The archive's files (directories left out); false with `why` when it is not a readable zip.
bool zip_entries(const std::string &zip, std::vector<ZipEntry> *out, std::string *why);
// `entry`'s bytes, checked against its CRC-32; false with `why` (encrypted, an unknown method, damaged).
bool zip_read(const std::string &zip, const ZipEntry &entry, std::string *out, std::string *why);

// A model export's name (`*.ts.pt` or `*.pt2`).
bool is_model_export(const std::string &name);
// Splits `ref` into its archive and member: true for `<x>.zip!<member>` and a bare `<x>.zip` (member "").
bool split_zip_model(const std::string &ref, std::string *zip, std::string *member);
// The archive's model exports.
std::vector<ZipEntry> zip_model_entries(const std::string &zip, std::string *why);
// The menu label of an archive's export: "<archive>: <run>/<file>" for runs/<run>/exports/<file>,
// else "<archive>: <member>".
std::string zip_model_label(const std::string &zip, const std::string &member);
// Whether `ref` names a model that is there: a file, or an export inside an archive ("" when it
// is; else why not).
std::string model_problem(const std::string &ref);
// The file the engine loads for `ref`: `ref` itself unless it names an archive's export, which is
// extracted into the cache; "" with `why` when that cannot be done.
std::string model_file(const std::string &ref, std::string *why);

}  // namespace rmwork

#endif /* RMWORK_MODEL_ZIP_H */
