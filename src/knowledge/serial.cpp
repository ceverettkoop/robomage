#include "serial.h"

namespace knowledge {

static void put_le(std::string &out, uint64_t v, int width);

static void put_le(std::string &out, uint64_t v, int width) {
    for (int i = 0; i < width; ++i) out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

void put_u8(std::string &out, uint64_t v) {
    put_le(out, v, 1);
}

void put_u16(std::string &out, uint64_t v) {
    put_le(out, v, 2);
}

void put_u32(std::string &out, uint64_t v) {
    put_le(out, v, 4);
}

void put_u64(std::string &out, uint64_t v) {
    put_le(out, v, 8);
}

void put_i32(std::string &out, int64_t v) {
    put_le(out, static_cast<uint64_t>(static_cast<uint32_t>(static_cast<int32_t>(v))), 4);
}

void put_str(std::string &out, const std::string &s) {
    put_u32(out, s.size());
    out += s;
}

void put_u32s(std::string &out, const std::vector<uint32_t> &v) {
    put_u32(out, v.size());
    for (uint32_t x : v) put_u32(out, x);
}

uint64_t ByteReader::get(int width) {
    if (!ok || at + static_cast<size_t>(width) > in.size()) {
        ok = false;
        return 0;
    }
    uint64_t v = 0;
    for (int i = 0; i < width; ++i) v |= static_cast<uint64_t>(static_cast<unsigned char>(in[at + i])) << (8 * i);
    at += static_cast<size_t>(width);
    return v;
}

std::string ByteReader::str() {
    const uint64_t n = u32();
    if (!ok || at + n > in.size()) {
        ok = false;
        return std::string();
    }
    std::string s = in.substr(at, n);
    at += n;
    return s;
}

std::vector<uint32_t> ByteReader::u32s() {
    std::vector<uint32_t> v;
    const uint64_t n = u32();
    if (!count(n, (in.size() - at) / 4)) return v;
    v.reserve(n);
    for (uint64_t i = 0; i < n; ++i) v.push_back(static_cast<uint32_t>(u32()));
    return v;
}

bool ByteReader::count(uint64_t n, uint64_t limit) {
    if (!ok || n > limit) {
        ok = false;
        return false;
    }
    return true;
}

}  // namespace knowledge
