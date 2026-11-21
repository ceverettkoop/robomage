#include "model_zip.h"

#include <zlib.h>

#include <cstdio>
#include <cstring>

#include "files.h"
#include "paths.h"

namespace {

const uint32_t SIG_EOCD = 0x06054b50u;
const uint32_t SIG_CENTRAL = 0x02014b50u;
const uint32_t SIG_LOCAL = 0x04034b50u;
const size_t EOCD_SIZE = 22;
const size_t CENTRAL_SIZE = 46;
const size_t LOCAL_SIZE = 30;
// The end-of-central-directory record and the longest archive comment after it.
const size_t EOCD_SEARCH = EOCD_SIZE + 0xFFFF;
const uint32_t ZIP64_MARK = 0xFFFFFFFFu;
const uint16_t METHOD_STORED = 0;
const uint16_t METHOD_DEFLATED = 8;
const uint16_t FLAG_ENCRYPTED = 1;
const char *const RUNS_PREFIX = "runs/";
const char *const EXPORTS_DIR = "/exports/";

}  // namespace

static uint16_t le16(const std::string &b, size_t at);
static uint32_t le32(const std::string &b, size_t at);
static bool read_range(FILE *f, uint64_t offset, size_t n, std::string *out);
static bool ends_with(const std::string &s, const char *suffix);
static std::string base_name(const std::string &path);
static bool safe_member(const std::string &name);
static bool inflate_raw(const std::string &in, uint64_t size, std::string *out);
static bool find_entry(const std::string &zip, const std::string &member, rmwork::ZipEntry *found, std::string *why);

// ── Little-endian reads and file ranges ──────────────────────────────────────────────────────────

static uint16_t le16(const std::string &b, size_t at) {
    return static_cast<uint16_t>(static_cast<unsigned char>(b[at]) | static_cast<unsigned char>(b[at + 1]) << 8);
}

static uint32_t le32(const std::string &b, size_t at) {
    return static_cast<uint32_t>(le16(b, at)) | static_cast<uint32_t>(le16(b, at + 2)) << 16;
}

// `n` bytes of `f` from `offset`; false when the file is shorter.
static bool read_range(FILE *f, uint64_t offset, size_t n, std::string *out) {
    out->assign(n, '\0');
    if (std::fseek(f, static_cast<long>(offset), SEEK_SET) != 0) return false;
    return n == 0 || std::fread(&(*out)[0], 1, n, f) == n;
}

static bool ends_with(const std::string &s, const char *suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() > n && s.compare(s.size() - n, n, suffix) == 0;
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// A member path that stays inside the directory it is extracted into.
static bool safe_member(const std::string &name) {
    if (name.empty() || name[0] == '/' || name.find('\\') != std::string::npos) return false;
    size_t start = 0;
    while (start <= name.size()) {
        const size_t slash = name.find('/', start);
        const std::string part = name.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (part == "..") return false;
        if (slash == std::string::npos) break;
        start = slash + 1;
    }
    return true;
}

// A raw deflate stream (no zlib header) inflated to exactly `size` bytes.
static bool inflate_raw(const std::string &in, uint64_t size, std::string *out) {
    out->assign(static_cast<size_t>(size), '\0');
    z_stream z;
    std::memset(&z, 0, sizeof z);
    if (inflateInit2(&z, -MAX_WBITS) != Z_OK) return false;
    z.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(in.data()));
    z.avail_in = static_cast<uInt>(in.size());
    z.next_out = reinterpret_cast<Bytef *>(out->empty() ? nullptr : &(*out)[0]);
    z.avail_out = static_cast<uInt>(out->size());
    const int rc = inflate(&z, Z_FINISH);
    const bool ok = rc == Z_STREAM_END && z.total_out == size;
    inflateEnd(&z);
    return ok;
}

// The export `member` names in `zip`, or its one export when `member` is "".
static bool find_entry(const std::string &zip, const std::string &member, rmwork::ZipEntry *found, std::string *why) {
    std::string err;
    const std::vector<rmwork::ZipEntry> models = rmwork::zip_model_entries(zip, &err);
    if (!err.empty()) {
        *why = err;
        return false;
    }
    const rmwork::ZipEntry *pick = nullptr;
    if (member.empty()) {
        if (models.size() != 1) {
            *why = models.empty() ? "No model export (*.ts.pt, *.pt2) in " + zip + "."
                                  : zip + " holds " + std::to_string(models.size()) +
                                        " model exports; name one as " + zip + "!<member>.";
            return false;
        }
        pick = &models[0];
    }
    for (const rmwork::ZipEntry &e : models)
        if (!pick && e.name == member) pick = &e;
    if (!pick) {
        *why = "No model export " + member + " in " + zip + ".";
        return false;
    }
    if (!safe_member(pick->name)) {
        *why = "The member path " + pick->name + " in " + zip + " leaves its directory.";
        return false;
    }
    *found = *pick;
    return true;
}

namespace rmwork {

bool zip_entries(const std::string &zip, std::vector<ZipEntry> *out, std::string *why) {
    out->clear();
    FILE *f = std::fopen(zip.c_str(), "rb");
    if (!f) {
        *why = "cannot open " + zip;
        return false;
    }
    const FileStamp st = file_stamp(zip);
    const uint64_t tail = st.size < EOCD_SEARCH ? st.size : EOCD_SEARCH;
    std::string b;
    if (st.size < EOCD_SIZE || !read_range(f, st.size - tail, static_cast<size_t>(tail), &b)) {
        std::fclose(f);
        *why = zip + " is not a zip archive";
        return false;
    }
    size_t eocd = std::string::npos;
    for (size_t i = b.size() - EOCD_SIZE + 1; i-- > 0;)
        if (le32(b, i) == SIG_EOCD) {
            eocd = i;
            break;
        }
    if (eocd == std::string::npos) {
        std::fclose(f);
        *why = zip + " is not a zip archive (no end of central directory)";
        return false;
    }
    const uint16_t count = le16(b, eocd + 10);
    const uint32_t cd_size = le32(b, eocd + 12), cd_offset = le32(b, eocd + 16);
    if (count == 0xFFFF || cd_size == ZIP64_MARK || cd_offset == ZIP64_MARK) {
        std::fclose(f);
        *why = zip + " is a zip64 archive, which the workbench does not read";
        return false;
    }
    std::string cd;
    const bool read = read_range(f, cd_offset, cd_size, &cd);
    std::fclose(f);
    if (!read) {
        *why = zip + " is damaged (its central directory is past its end)";
        return false;
    }
    size_t at = 0;
    for (uint16_t i = 0; i < count; ++i) {
        if (at + CENTRAL_SIZE > cd.size() || le32(cd, at) != SIG_CENTRAL) {
            *why = zip + " is damaged (central directory entry " + std::to_string(i) + ")";
            return false;
        }
        const size_t name_len = le16(cd, at + 28), extra_len = le16(cd, at + 30), comment_len = le16(cd, at + 32);
        if (at + CENTRAL_SIZE + name_len > cd.size()) {
            *why = zip + " is damaged (central directory entry " + std::to_string(i) + ")";
            return false;
        }
        ZipEntry e;
        e.flags = le16(cd, at + 8);
        e.method = le16(cd, at + 10);
        e.crc32 = le32(cd, at + 16);
        e.compressed_size = le32(cd, at + 20);
        e.size = le32(cd, at + 24);
        e.local_offset = le32(cd, at + 42);
        e.name = cd.substr(at + CENTRAL_SIZE, name_len);
        if (e.compressed_size == ZIP64_MARK || e.size == ZIP64_MARK || e.local_offset == ZIP64_MARK) {
            *why = zip + " holds a zip64 member (" + e.name + "), which the workbench does not read";
            return false;
        }
        if (!e.name.empty() && e.name.back() != '/') out->push_back(e);
        at += CENTRAL_SIZE + name_len + extra_len + comment_len;
    }
    return true;
}

bool zip_read(const std::string &zip, const ZipEntry &entry, std::string *out, std::string *why) {
    if (entry.flags & FLAG_ENCRYPTED) {
        *why = entry.name + " in " + zip + " is encrypted";
        return false;
    }
    if (entry.method != METHOD_STORED && entry.method != METHOD_DEFLATED) {
        *why = entry.name + " in " + zip + " uses compression method " + std::to_string(entry.method) +
               " (the workbench reads stored and deflated members)";
        return false;
    }
    FILE *f = std::fopen(zip.c_str(), "rb");
    if (!f) {
        *why = "cannot open " + zip;
        return false;
    }
    std::string local, data;
    bool ok = read_range(f, entry.local_offset, LOCAL_SIZE, &local) && le32(local, 0) == SIG_LOCAL;
    if (ok) {
        const uint64_t start = entry.local_offset + LOCAL_SIZE + le16(local, 26) + le16(local, 28);
        ok = read_range(f, start, static_cast<size_t>(entry.compressed_size), &data);
    }
    std::fclose(f);
    if (!ok) {
        *why = entry.name + " in " + zip + " is damaged (its data is past the archive's end)";
        return false;
    }
    if (entry.method == METHOD_STORED) {
        *out = std::move(data);
    } else if (!inflate_raw(data, entry.size, out)) {
        *why = entry.name + " in " + zip + " is damaged (it does not inflate)";
        return false;
    }
    const uLong crc = crc32(crc32(0L, Z_NULL, 0), reinterpret_cast<const Bytef *>(out->data()),
                            static_cast<uInt>(out->size()));
    if (out->size() != entry.size || static_cast<uint32_t>(crc) != entry.crc32) {
        *why = entry.name + " in " + zip + " is damaged (its CRC-32 does not match)";
        return false;
    }
    return true;
}

bool is_model_export(const std::string &name) {
    return ends_with(name, ".ts.pt") || ends_with(name, ".pt2");
}

bool split_zip_model(const std::string &ref, std::string *zip, std::string *member) {
    const size_t bang = ref.find(".zip!");
    if (bang != std::string::npos) {
        *zip = ref.substr(0, bang + 4);
        *member = ref.substr(bang + 5);
        return true;
    }
    if (!ends_with(ref, ".zip")) return false;
    *zip = ref;
    member->clear();
    return true;
}

std::vector<ZipEntry> zip_model_entries(const std::string &zip, std::string *why) {
    std::vector<ZipEntry> all, out;
    if (!zip_entries(zip, &all, why)) return out;
    for (const ZipEntry &e : all)
        if (is_model_export(e.name)) out.push_back(e);
    return out;
}

std::string zip_model_label(const std::string &zip, const std::string &member) {
    std::string shown = member;
    const size_t exports = member.find(EXPORTS_DIR);
    if (member.compare(0, std::strlen(RUNS_PREFIX), RUNS_PREFIX) == 0 && exports != std::string::npos &&
        member.find('/', exports + std::strlen(EXPORTS_DIR)) == std::string::npos) {
        const size_t run = std::strlen(RUNS_PREFIX);
        shown = member.substr(run, exports - run) + "/" + member.substr(exports + std::strlen(EXPORTS_DIR));
    }
    return base_name(zip) + ": " + shown;
}

std::string model_problem(const std::string &ref) {
    std::string zip, member, why;
    if (!split_zip_model(ref, &zip, &member))
        return path_kind(ref) == PathKind::FILE ? "" : "No model at " + ref + ".";
    if (path_kind(zip) != PathKind::FILE) return "No archive at " + zip + ".";
    ZipEntry found;
    if (!find_entry(zip, member, &found, &why)) return why;
    return "";
}

std::string model_file(const std::string &ref, std::string *why) {
    std::string zip, member;
    if (!split_zip_model(ref, &zip, &member)) return ref;
    ZipEntry e;
    if (!find_entry(zip, member, &e, why)) return "";
    char crc[9];
    std::snprintf(crc, sizeof crc, "%08x", e.crc32);
    std::string stem = base_name(zip);
    stem = stem.substr(0, stem.size() - 4);
    const std::string path = workbench_cache_dir() + "/models/" + stem + "-" + crc + "/" + e.name;
    const FileStamp have = file_stamp(path);
    if (have.found && have.regular && have.size == e.size) return path;
    std::string data;
    if (!zip_read(zip, e, &data, why)) return "";
    if (!make_dirs(parent_dir(path))) {
        *why = "cannot create " + parent_dir(path);
        return "";
    }
    if (!write_file_atomic(path, data, why)) return "";
    return path;
}

}  // namespace rmwork
