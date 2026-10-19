#include "file_util.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstring>

bool make_dirs(const std::string &path) {
    for (size_t at = path.find('/', 1);; at = path.find('/', at + 1)) {
        const std::string part = at == std::string::npos ? path : path.substr(0, at);
        if (!part.empty() && mkdir(part.c_str(), 0777) != 0 && errno != EEXIST) return false;
        if (at == std::string::npos) break;
    }
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool write_file_atomic(const std::string &path, const std::string &data, std::string *err) {
    const std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        *err = tmp + ": " + std::strerror(errno);
        return false;
    }
    const bool written = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    const bool closed = std::fclose(f) == 0;
    if (!written || !closed) {
        *err = tmp + ": write failed";
        std::remove(tmp.c_str());
        return false;
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        *err = path + ": " + std::strerror(errno);
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}
