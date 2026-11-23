#include "selftest_model_zip.h"

#include <zlib.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "files.h"
#include "model_zip.h"

namespace {

// One file of a synthetic archive.
struct Member {
        std::string name;
        std::string data;
        uint16_t method = 0;  // 0 stored, 8 deflated
        uint16_t flags = 0;
        bool bad_crc = false;
};

}  // namespace

static void put16(std::string *b, uint32_t v);
static void put32(std::string *b, uint32_t v);
static std::string deflate_raw(const std::string &in);
static std::string build_zip(const std::vector<Member> &members);
static bool write_zip(const std::string &path, const std::vector<Member> &members);

static void put16(std::string *b, uint32_t v) {
    b->push_back(static_cast<char>(v & 0xFF));
    b->push_back(static_cast<char>((v >> 8) & 0xFF));
}

static void put32(std::string *b, uint32_t v) {
    put16(b, v & 0xFFFF);
    put16(b, v >> 16);
}

// `in` as a raw deflate stream (no zlib header), as a zip's deflated member holds it.
static std::string deflate_raw(const std::string &in) {
    z_stream z;
    std::memset(&z, 0, sizeof z);
    deflateInit2(&z, Z_BEST_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
    std::string out(deflateBound(&z, static_cast<uLong>(in.size())), '\0');
    z.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(in.data()));
    z.avail_in = static_cast<uInt>(in.size());
    z.next_out = reinterpret_cast<Bytef *>(&out[0]);
    z.avail_out = static_cast<uInt>(out.size());
    deflate(&z, Z_FINISH);
    out.resize(z.total_out);
    deflateEnd(&z);
    return out;
}

// The bytes of a zip archive holding `members`, in order.
static std::string build_zip(const std::vector<Member> &members) {
    std::string body, central;
    for (const Member &m : members) {
        const std::string data = m.method == 8 ? deflate_raw(m.data) : m.data;
        uint32_t crc = static_cast<uint32_t>(
            crc32(crc32(0L, Z_NULL, 0), reinterpret_cast<const Bytef *>(m.data.data()), static_cast<uInt>(m.data.size())));
        if (m.bad_crc) crc ^= 1u;
        const uint32_t offset = static_cast<uint32_t>(body.size());
        put32(&body, 0x04034b50u);
        put16(&body, 20);
        put16(&body, m.flags);
        put16(&body, m.method);
        put32(&body, 0);
        put32(&body, crc);
        put32(&body, static_cast<uint32_t>(data.size()));
        put32(&body, static_cast<uint32_t>(m.data.size()));
        put16(&body, static_cast<uint32_t>(m.name.size()));
        put16(&body, 0);
        body += m.name + data;
        put32(&central, 0x02014b50u);
        put16(&central, 20);
        put16(&central, 20);
        put16(&central, m.flags);
        put16(&central, m.method);
        put32(&central, 0);
        put32(&central, crc);
        put32(&central, static_cast<uint32_t>(data.size()));
        put32(&central, static_cast<uint32_t>(m.data.size()));
        put16(&central, static_cast<uint32_t>(m.name.size()));
        put16(&central, 0);
        put16(&central, 0);
        put16(&central, 0);
        put16(&central, 0);
        put32(&central, 0);
        put32(&central, offset);
        central += m.name;
    }
    std::string out = body + central;
    put32(&out, 0x06054b50u);
    put16(&out, 0);
    put16(&out, 0);
    put16(&out, static_cast<uint32_t>(members.size()));
    put16(&out, static_cast<uint32_t>(members.size()));
    put32(&out, static_cast<uint32_t>(central.size()));
    put32(&out, static_cast<uint32_t>(body.size()));
    put16(&out, 0);
    return out;
}

static bool write_zip(const std::string &path, const std::vector<Member> &members) {
    std::string err;
    return rmwork::write_file_atomic(path, build_zip(members), &err);
}

namespace rmwork {

void check_model_zip_units(const CheckFn &check, const std::string &tmp) {
    const char *saved = std::getenv("XDG_CACHE_HOME");
    const std::string saved_cache = saved ? saved : "";
    const std::string dir = tmp + "/model_zip";
    make_dirs(dir);
    setenv("XDG_CACHE_HOME", (dir + "/cache").c_str(), 1);

    std::string policy;
    for (int i = 0; i < 5000; ++i) policy += "weights " + std::to_string(i * 7919 % 1000) + "\n";
    const std::string zip = dir + "/pack.zip";
    check(write_zip(zip, {{"runs/", "", 0, 0, false},
                          {"runs/r1/README.txt", "read me", 0, 0, false},
                          {"runs/r1/exports/final.ts.pt", policy, 8, 0, false},
                          {"runs/r1/exports/m9.pt2", "stored policy", 0, 0, false}}),
          "Zip: the synthetic archive is not written");

    std::vector<ZipEntry> all;
    std::string why;
    check(zip_entries(zip, &all, &why) && all.size() == 3, "Zip: the archive's entries are not its three files");
    const std::vector<ZipEntry> models = zip_model_entries(zip, &why);
    check(models.size() == 2 && models[0].name == "runs/r1/exports/final.ts.pt" && models[0].method == 8 &&
              models[1].name == "runs/r1/exports/m9.pt2",
          "Zip: the exports are not the deflated .ts.pt and the stored .pt2");
    check(zip_model_label("pack.zip", "runs/r1/exports/final.ts.pt") == "pack.zip: r1/final.ts.pt" &&
              zip_model_label("pack.zip", "a/b.pt2") == "pack.zip: a/b.pt2",
          "Zip: the labels are not \"<archive>: <run>/<file>\" or the member");

    std::string z, member;
    check(split_zip_model(zip + "!a/b.pt2", &z, &member) && z == zip && member == "a/b.pt2" &&
              split_zip_model(zip, &z, &member) && member.empty() && !split_zip_model(dir + "/m.ts.pt", &z, &member),
          "Zip: references do not split into archive and member");

    const std::string ref = zip + "!runs/r1/exports/final.ts.pt";
    check(model_problem(ref).empty(), "Zip: a present member is reported missing: " + model_problem(ref));
    const std::string path = model_file(ref, &why);
    std::string got;
    check(!path.empty() && path.find(dir + "/cache/robomage/workbench/models/pack-") == 0 &&
              read_file(path, &got) && got == policy,
          "Zip: the deflated export is not extracted into the cache intact (" + why + ")");
    const FileStamp first = file_stamp(path);
    check(model_file(ref, &why) == path && file_stamp(path).mtime_ns == first.mtime_ns,
          "Zip: a cached extraction is not reused");
    const std::string stored = model_file(zip + "!runs/r1/exports/m9.pt2", &why);
    check(!stored.empty() && read_file(stored, &got) && got == "stored policy",
          "Zip: the stored export is not extracted intact");
    check(model_file(dir + "/plain.ts.pt", &why) == dir + "/plain.ts.pt", "Zip: a plain path is not passed through");

    check(!model_problem(zip + "!runs/r1/exports/nope.pt2").empty() && model_file(zip + "!nope.pt2", &why).empty(),
          "Zip: a missing member is not refused");
    check(model_file(zip, &why).empty() && why.find("2 model exports") != std::string::npos,
          "Zip: a bare archive of two exports is not refused");
    const std::string one = dir + "/one.zip";
    write_zip(one, {{"x/only.ts.pt", policy, 8, 0, false}});
    const std::string only = model_file(one, &why);
    check(!only.empty() && read_file(only, &got) && got == policy, "Zip: a bare archive's one export is not used");

    const std::string bad = dir + "/bad.zip";
    write_zip(bad, {{"bad.ts.pt", policy, 8, 0, true}});
    check(model_file(bad, &why).empty() && why.find("CRC-32") != std::string::npos,
          "Zip: a damaged CRC-32 is not refused");
    const std::string escape = dir + "/escape.zip";
    write_zip(escape, {{"../escape.ts.pt", "x", 0, 0, false}});
    check(model_file(escape, &why).empty() && !file_exists(dir + "/cache/robomage/workbench/escape.ts.pt"),
          "Zip: a member path leaving its directory is not refused");
    const std::string locked = dir + "/locked.zip";
    write_zip(locked, {{"locked.ts.pt", "x", 0, 1, false}});
    check(model_file(locked, &why).empty() && why.find("encrypted") != std::string::npos,
          "Zip: an encrypted member is not refused");
    const std::string notzip = dir + "/not.zip";
    write_file_atomic(notzip, "not an archive at all, but long enough to search", &why);
    check(!zip_entries(notzip, &all, &why) && !model_problem(notzip).empty(), "Zip: a file that is not a zip is read");
    check(!model_problem(dir + "/missing.zip").empty(), "Zip: a missing archive is not reported");

    if (saved) setenv("XDG_CACHE_HOME", saved_cache.c_str(), 1);
    else unsetenv("XDG_CACHE_HOME");
}

}  // namespace rmwork
