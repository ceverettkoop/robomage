#include "td_targets.h"

#include <algorithm>
#include <cstddef>

namespace {

// Apply the n-step rule inside one chain [lo, hi]. `out` is pre-seeded with z,
// so every branch that falls back to the outcome simply leaves its entry alone.
void fill_chain(const std::vector<TdSample>& s, int lo, int hi, int n,
                std::vector<float>& out);

void fill_chain(const std::vector<TdSample>& s, int lo, int hi, int n,
                std::vector<float>& out) {
    // next_explored[i - lo] = smallest j > i in [lo, hi] with s[j].explored,
    // else hi + 1 (an out-of-chain sentinel, which yields j_max == hi == L).
    std::vector<int> next_explored(static_cast<size_t>(hi - lo + 1), hi + 1);
    int nxt = hi + 1;
    for (int i = hi; i >= lo; i--) {
        next_explored[static_cast<size_t>(i - lo)] = nxt;
        if (s[static_cast<size_t>(i)].explored) nxt = i;
    }
    for (int t = lo; t <= hi; t++) {
        int j_max = std::min(next_explored[static_cast<size_t>(t - lo)] - 1, hi);
        int j;
        if (t + n <= j_max) {
            j = t + n;
        } else if (j_max >= hi) {
            continue;  // terminal inside the horizon -> keep z[t]
        } else if (j_max > t) {
            j = j_max;  // shortened horizon at the exploratory move
        } else {
            continue;  // immediate block -> keep z[t]
        }
        float qj = s[static_cast<size_t>(j)].q;
        out[static_cast<size_t>(t)] =
            s[static_cast<size_t>(j)].mover_is_a == s[static_cast<size_t>(t)].mover_is_a
                ? qj
                : -qj;
    }
}

}  // namespace

void compute_td_targets(const std::vector<TdSample>& samples, int td_n,
                        std::vector<float>& out) {
    const int m = static_cast<int>(samples.size());
    out.assign(static_cast<size_t>(m), 0.0f);
    for (int i = 0; i < m; i++) out[static_cast<size_t>(i)] = samples[static_cast<size_t>(i)].z;
    const int n = td_n > 1 ? td_n : 1;
    int lo = 0;
    while (lo < m) {
        // A chain is a contiguous run with the same is_sideboard flag. The
        // caller passes ONE game, so the game boundary is already the array's
        // edge; the sideboard-root samples of that game are a leading run and
        // become their own chain (never entering an in-game sample's window).
        int hi = lo;
        while (hi + 1 < m && samples[static_cast<size_t>(hi + 1)].is_sideboard ==
                                 samples[static_cast<size_t>(lo)].is_sideboard)
            hi++;
        if (!samples[static_cast<size_t>(lo)].is_sideboard)
            fill_chain(samples, lo, hi, n, out);
        lo = hi + 1;
    }
}
