#ifndef STABLE_RNG_H
#define STABLE_RNG_H

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

// Platform-stable randomness on top of the game's seeded std::mt19937.
//
// The raw mt19937 output stream is exactly specified by the C++ standard, but
// std::shuffle and std::uniform_int_distribution are implementation-defined:
// libstdc++ (Linux / CI) and libc++ (macOS) map the same stream to DIFFERENT
// permutations and draws. Every gameplay use of cur_game.gen must go through
// these helpers instead, so a seed reproduces the identical game on every
// platform — replay logs, the regression corpus (train/regression/), and
// bug-repro seeds stay portable between Mac and Linux.

// Unbiased draw in [0, n) consuming raw 32-bit mt19937 outputs (rejection
// sampling: reject the top sliver of the 2^32 range that would bias v % n).
// n is a zone/hand/library size in practice, far below 2^32.
inline uint32_t stable_rand_below(std::mt19937 &gen, uint64_t n) {
    if (n <= 1) return 0;
    const uint64_t span = uint64_t(1) << 32;
    const uint64_t limit = span - span % n;  // largest multiple of n <= 2^32
    uint64_t v;
    do {
        v = gen();
    } while (v >= limit);
    return uint32_t(v % n);
}

// Back-to-front Fisher–Yates over a vector using stable_rand_below.
template <typename T>
inline void stable_shuffle(std::vector<T> &v, std::mt19937 &gen) {
    for (size_t i = v.size(); i > 1; --i) {
        size_t j = stable_rand_below(gen, i);
        std::swap(v[i - 1], v[j]);
    }
}

#endif
