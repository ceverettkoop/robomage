#ifndef KNOWLEDGE_SERIAL_H
#define KNOWLEDGE_SERIAL_H

#include <cstdint>
#include <string>
#include <vector>

// ── Byte serialisation of the knowledge state (little-endian, no padding) ─────────────────────
// Readers fail softly (the engine builds with -fno-exceptions): a read past the end sets the
// reader's `ok` to false and returns 0, and the caller refuses the whole blob.

namespace knowledge {

void put_u8(std::string &out, uint64_t v);
void put_u16(std::string &out, uint64_t v);
void put_u32(std::string &out, uint64_t v);
void put_u64(std::string &out, uint64_t v);
void put_i32(std::string &out, int64_t v);
void put_str(std::string &out, const std::string &s);
void put_u32s(std::string &out, const std::vector<uint32_t> &v);

struct ByteReader {
        const std::string &in;
        size_t at;
        bool ok = true;
        ByteReader(const std::string &s, size_t start) : in(s), at(start) {}
        uint64_t u8() { return get(1); }
        uint64_t u16() { return get(2); }
        uint64_t u32() { return get(4); }
        uint64_t u64() { return get(8); }
        int32_t i32() { return static_cast<int32_t>(static_cast<uint32_t>(get(4))); }
        std::string str();
        std::vector<uint32_t> u32s();
        // A count read before a list: false (and ok = false) when it exceeds `limit`.
        bool count(uint64_t n, uint64_t limit);

    private:
        uint64_t get(int width);
};

}  // namespace knowledge

#endif /* KNOWLEDGE_SERIAL_H */
