#include "game_seed.h"

#include <vector>

namespace agents {

namespace {

using u128 = unsigned __int128;

// numpy SeedSequence(seed).generate_state(4, uint64), no spawn key.
class SeedSequence {
    public:
        explicit SeedSequence(uint64_t seed) {
            std::vector<uint32_t> entropy;
            do {
                entropy.push_back(static_cast<uint32_t>(seed));
                seed >>= 32;
            } while (seed);
            uint32_t pool[4];
            for (size_t i = 0; i < 4; i++) pool[i] = hashmix(i < entropy.size() ? entropy[i] : 0);
            for (size_t i = 0; i < 4; i++)
                for (size_t j = 0; j < 4; j++)
                    if (i != j) pool[j] = mix(pool[j], hashmix(pool[i]));
            for (size_t i = 4; i < entropy.size(); i++)
                for (size_t j = 0; j < 4; j++) pool[j] = mix(pool[j], hashmix(entropy[i]));
            uint32_t out[8];
            uint32_t hash_b = 0x8B51F9DDu;
            for (size_t i = 0; i < 8; i++) {
                uint32_t v = pool[i % 4] ^ hash_b;
                hash_b *= 0x58F38DEDu;
                v *= hash_b;
                out[i] = v ^ (v >> 16);
            }
            for (size_t k = 0; k < 4; k++) words[k] = out[2 * k] | (static_cast<uint64_t>(out[2 * k + 1]) << 32);
        }
        uint64_t words[4] = {};

    private:
        uint32_t hashmix(uint32_t value) {
            value ^= hash_const_;
            hash_const_ *= 0x931E8875u;
            value *= hash_const_;
            return value ^ (value >> 16);
        }
        static uint32_t mix(uint32_t x, uint32_t y) {
            const uint32_t r = 0xCA01F9DDu * x - 0x4973F715u * y;
            return r ^ (r >> 16);
        }
        uint32_t hash_const_ = 0x43B0D7E5u;
};

// numpy's PCG64 with its 32-bit output buffer.
class Pcg64 {
    public:
        explicit Pcg64(const uint64_t s[4]) {
            inc_ = ((((static_cast<u128>(s[2]) << 64) | s[3]) << 1) | 1);
            state_ = inc_ + ((static_cast<u128>(s[0]) << 64) | s[1]);
            state_ = state_ * MULT + inc_;
        }
        uint32_t next32() {
            if (has_spare_) {
                has_spare_ = false;
                return spare_;
            }
            state_ = state_ * MULT + inc_;
            uint64_t v = static_cast<uint64_t>(state_ >> 64) ^ static_cast<uint64_t>(state_);
            const unsigned rot = static_cast<unsigned>(state_ >> 122);
            v = (v >> rot) | (v << ((64 - rot) & 63));
            spare_ = static_cast<uint32_t>(v >> 32);
            has_spare_ = true;
            return static_cast<uint32_t>(v);
        }

    private:
        static constexpr u128 MULT = (static_cast<u128>(0x2360ED051FC65DA4ull) << 64) | 0x4385DF649FCCF645ull;
        u128 state_ = 0;
        u128 inc_ = 0;
        uint32_t spare_ = 0;
        bool has_spare_ = false;
};

}  // namespace

uint32_t engine_seed_for_game(uint64_t game_seed) {
    const SeedSequence seq(game_seed);
    Pcg64 rng(seq.words);
    const uint64_t span = (1ull << 31) - 1;  // values 0 .. 2**31 - 2
    uint64_t m = static_cast<uint64_t>(rng.next32()) * span;
    if (static_cast<uint32_t>(m) < span) {
        const uint64_t threshold = (0xFFFFFFFFull - (span - 1)) % span;
        while (static_cast<uint32_t>(m) < threshold) m = static_cast<uint64_t>(rng.next32()) * span;
    }
    return static_cast<uint32_t>(m >> 32);
}

}  // namespace agents
