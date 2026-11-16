#include "files.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace rmwork {

bool make_dirs(const std::string &path) {
    if (path.empty()) return false;
    std::string cur;
    for (size_t i = 0; i <= path.size(); ++i) {
        if (i == path.size() || path[i] == '/') {
            if (!cur.empty() && cur != "/" && mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
        }
        if (i < path.size()) cur += path[i];
    }
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string parent_dir(const std::string &path) {
    const size_t slash = path.rfind('/');
    if (slash == std::string::npos) return "";
    return slash == 0 ? "/" : path.substr(0, slash);
}

bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

PathKind path_kind(const std::string &path) {
    struct stat st;
    if (path.empty() || stat(path.c_str(), &st) != 0) return PathKind::MISSING;
    if (S_ISREG(st.st_mode)) return PathKind::FILE;
    if (S_ISDIR(st.st_mode)) return PathKind::DIRECTORY;
    if (S_ISSOCK(st.st_mode)) return PathKind::SOCKET;
    return PathKind::OTHER;
}

bool is_executable(const std::string &path) {
    return path_kind(path) == PathKind::FILE && access(path.c_str(), X_OK) == 0;
}

std::string executable_path() {
    std::vector<char> buf(4096);
    const ssize_t n = readlink("/proc/self/exe", buf.data(), buf.size() - 1);
    if (n <= 0) return "";
    return std::string(buf.data(), static_cast<size_t>(n));
}

std::string current_dir() {
    std::vector<char> buf(4096);
    return getcwd(buf.data(), buf.size()) ? std::string(buf.data()) : std::string();
}

std::string absolute_path(const std::string &path) {
    if (path.empty() || path[0] == '/') return path;
    return current_dir() + "/" + path;
}

bool read_file(const std::string &path, std::string *out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[8192];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

bool write_file_atomic(const std::string &path, const std::string &data, std::string *err) {
    const std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        *err = "cannot write " + tmp + ": " + std::strerror(errno);
        return false;
    }
    const bool wrote = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    const bool closed = std::fclose(f) == 0;
    if (!wrote || !closed) {
        *err = "cannot write " + tmp + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        *err = "cannot rename " + tmp + " to " + path + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

std::string make_temp_dir(const char *prefix) {
    const char *base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base && *base ? base : "/tmp") + "/" + prefix + "XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) return "";
    return std::string(buf.data());
}

void remove_tree(const std::string &path) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        if (DIR *d = opendir(path.c_str())) {
            while (const dirent *e = readdir(d)) {
                const std::string name = e->d_name;
                if (name != "." && name != "..") remove_tree(path + "/" + name);
            }
            closedir(d);
        }
        rmdir(path.c_str());
    } else {
        unlink(path.c_str());
    }
}

}  // namespace rmwork
