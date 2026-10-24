#include "util.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>

static bool is_space(char c);
static bool compiles_in_child(const std::vector<std::string> &rxs, size_t from, size_t to);

static bool is_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

// Whether rxs[from, to) all compile, in a forked child (a malformed pattern aborts it; its stderr goes to /dev/null).
static bool compiles_in_child(const std::vector<std::string> &rxs, size_t from, size_t to) {
    std::fflush(stdout);
    const pid_t pid = fork();
    if (pid < 0) return true;
    if (pid == 0) {
        const int null = open("/dev/null", O_WRONLY);
        if (null >= 0) dup2(null, 2);
        for (size_t k = from; k < to; ++k) {
            const std::regex re(rxs[k]);
            (void)re;
        }
        _exit(0);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
    }
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

namespace rmcheck {

bool starts_with(const std::string &s, const char *prefix) {
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

bool ends_with(const std::string &s, const char *suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool contains(const std::string &s, const char *needle) {
    return s.find(needle) != std::string::npos;
}

std::string trim(const std::string &s) {
    size_t b = 0, e = s.size();
    while (b < e && is_space(s[b])) ++b;
    while (e > b && is_space(s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::vector<std::string> split_lines(const std::string &text) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < text.size()) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) nl = text.size();
        size_t end = nl;
        if (end > start && text[end - 1] == '\r') --end;
        out.push_back(text.substr(start, end - start));
        start = nl + 1;
    }
    return out;
}

std::vector<std::string> split(const std::string &s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t at = s.find(sep, start);
        if (at == std::string::npos) {
            out.push_back(s.substr(start));
            return out;
        }
        out.push_back(s.substr(start, at - start));
        start = at + 1;
    }
}

std::string join(const std::vector<std::string> &parts, const char *sep) {
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i) out += sep;
        out += parts[i];
    }
    return out;
}

std::string quoted(const std::string &s) {
    std::string out = "'";
    for (unsigned char c : s) {
        if (c == '\'' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\x%02x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "'";
}

bool read_file(const std::string &path, std::string *out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    std::fclose(f);
    return true;
}

bool write_file(const std::string &path, const std::string &data) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    return std::fclose(f) == 0 && ok;
}

std::string find_program(const char *name) {
    std::vector<std::string> dirs;
    if (const char *path = std::getenv("PATH"))
        for (const std::string &d : split(path, ':'))
            if (!d.empty()) dirs.push_back(d);
    for (const char *d : {"/usr/bin", "/usr/local/bin", "/bin"}) dirs.push_back(d);
    for (const std::string &dir : dirs) {
        const std::string p = dir + "/" + name;
        if (access(p.c_str(), X_OK) == 0) return p;
    }
    return "";
}

bool file_exists(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

bool is_dir(const std::string &path) {
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool make_dirs(const std::string &path) {
    if (path.empty() || is_dir(path)) return true;
    const std::string parent = dirname_of(path);
    if (parent != path && !make_dirs(parent)) return false;
    return mkdir(path.c_str(), 0755) == 0 || is_dir(path);
}

void remove_tree(const std::string &path) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) return;
    if (S_ISDIR(st.st_mode)) {
        DIR *d = opendir(path.c_str());
        if (d) {
            while (dirent *e = readdir(d)) {
                if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, "..")) continue;
                remove_tree(path_join(path, e->d_name));
            }
            closedir(d);
        }
        rmdir(path.c_str());
    } else {
        unlink(path.c_str());
    }
}

std::vector<std::string> list_dir(const std::string &dir, const char *suffix) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        if (ends_with(name, suffix)) out.push_back(name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

std::string path_join(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (!b.empty() && b[0] == '/') return b;
    return a.back() == '/' ? a + b : a + "/" + b;
}

std::string dirname_of(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return ".";
    if (slash == 0) return "/";
    return path.substr(0, slash);
}

std::string basename_of(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

std::string strip_ext(const std::string &path) {
    const size_t dot = path.find_last_of('.');
    const size_t slash = path.find_last_of('/');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path;
    return path.substr(0, dot);
}

std::string make_temp_dir(const std::string &parent, const std::string &prefix) {
    make_dirs(parent);
    std::string tmpl = path_join(parent, prefix + "XXXXXX");
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    if (!mkdtemp(buf.data())) return "";
    return std::string(buf.data());
}

size_t first_bad_regex(const std::vector<std::string> &rxs) {
    if (rxs.empty() || compiles_in_child(rxs, 0, rxs.size())) return std::string::npos;
    for (size_t k = 0; k < rxs.size(); ++k)
        if (!compiles_in_child(rxs, k, k + 1)) return k;
    return std::string::npos;
}

double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::string fmt1(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", v);
    return buf;
}

}  // namespace rmcheck
