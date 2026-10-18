#include "py_random.h"

#include <vector>

namespace agents {

void PyRandom::init_genrand(uint32_t s) {
    mt_[0] = s;
    for (int i = 1; i < N; i++) mt_[i] = 1812433253u * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
    index_ = N;
}

void PyRandom::seed_int(uint64_t seed) {
    // The integer's 32-bit words, least significant first; zero is the one word 0.
    uint32_t key[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    const size_t key_length = (seed >> 32) != 0 ? 2 : 1;
    init_genrand(19650218u);
    int i = 1;
    size_t j = 0;
    for (size_t k = N > key_length ? static_cast<size_t>(N) : key_length; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525u)) + key[j] + static_cast<uint32_t>(j);
        i++;
        j++;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
        if (j >= key_length) j = 0;
    }
    for (int k = N - 1; k; k--) {
        mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941u)) - static_cast<uint32_t>(i);
        i++;
        if (i >= N) {
            mt_[0] = mt_[N - 1];
            i = 1;
        }
    }
    mt_[0] = 0x80000000u;
    index_ = N;
}

uint32_t PyRandom::genrand_uint32() {
    static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
    constexpr int M = 397;
    constexpr uint32_t UPPER = 0x80000000u, LOWER = 0x7fffffffu;
    if (index_ >= N) {
        int kk = 0;
        uint32_t y;
        for (; kk < N - M; kk++) {
            y = (mt_[kk] & UPPER) | (mt_[kk + 1] & LOWER);
            mt_[kk] = mt_[kk + M] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        for (; kk < N - 1; kk++) {
            y = (mt_[kk] & UPPER) | (mt_[kk + 1] & LOWER);
            mt_[kk] = mt_[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1u];
        }
        y = (mt_[N - 1] & UPPER) | (mt_[0] & LOWER);
        mt_[N - 1] = mt_[M - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
        index_ = 0;
    }
    uint32_t y = mt_[index_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
}

double PyRandom::random() {
    const uint32_t a = genrand_uint32() >> 5, b = genrand_uint32() >> 6;
    return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
}

uint64_t PyRandom::getrandbits(int k) {
    if (k <= 32) return genrand_uint32() >> (32 - k);
    // Words least significant first, the last one cut to the remaining bits.
    const uint64_t low = genrand_uint32();
    const uint64_t high = genrand_uint32() >> (64 - k);
    return low | (high << 32);
}

uint64_t PyRandom::randrange(uint64_t n) {
    int k = 0;
    for (uint64_t v = n; v; v >>= 1) k++;
    uint64_t r = getrandbits(k);
    while (r >= n) r = getrandbits(k);
    return r;
}

size_t PyRandom::weighted_index(const double *weights, size_t n) {
    std::vector<double> cum(n);
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) cum[i] = acc = acc + weights[i];
    const double x = random() * (cum[n - 1] + 0.0);
    // bisect.bisect_right(cum, x, 0, n - 1)
    size_t lo = 0, hi = n - 1;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (x < cum[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

}  // namespace agents
