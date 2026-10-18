#ifndef AGENTS_PY_RANDOM_H
#define AGENTS_PY_RANDOM_H

#include <cstddef>
#include <cstdint>

// ── Python's random.Random, bit for bit (CPython Modules/_randommodule.c, Lib/random.py) ──────
//
// The in-process agents draw from this generator so that a C++ agent ported from a Python one
// makes the same draws as the original for the same seed: MT19937 seeded as random.seed(int)
// seeds it (init_by_array over the integer's 32-bit words, least significant first), random() as
// genrand_res53, getrandbits(k) and randrange(n) (the rejection sampler _randbelow) as CPython
// computes them. The weighted choice is random.choices(population, weights, k=1): one random() u
// and the first index whose cumulative weight (summed left to right) exceeds u * total.

namespace agents {

class PyRandom {
    public:
        explicit PyRandom(uint64_t seed = 0) { seed_int(seed); }
        // random.seed(seed) for a non-negative integer.
        void seed_int(uint64_t seed);
        // random.random(): a double in [0, 1) with 53 random bits.
        double random();
        // random.getrandbits(k), 1 <= k <= 64.
        uint64_t getrandbits(int k);
        // random.randrange(n), n >= 1.
        uint64_t randrange(uint64_t n);
        // random.choices(range(n), weights=weights, k=1)[0]; `weights` has n > 0 entries with a
        // positive finite total.
        size_t weighted_index(const double *weights, size_t n);

    private:
        static constexpr int N = 624;
        uint32_t genrand_uint32();
        void init_genrand(uint32_t s);
        uint32_t mt_[N] = {};
        int index_ = N + 1;
};

}  // namespace agents

#endif /* AGENTS_PY_RANDOM_H */
