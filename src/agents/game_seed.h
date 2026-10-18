#ifndef AGENTS_GAME_SEED_H
#define AGENTS_GAME_SEED_H

#include <cstdint>

namespace agents {

// The engine --seed of game seed `game_seed`, as every Python driver derives it (train/env.py,
// train/rm_driver.py's engine_seed): numpy's
// Generator(PCG64(SeedSequence(game_seed))).integers(0, 2**31 - 1) — SeedSequence's pool of four
// words, PCG64 (XSL-RR) and numpy's buffered 32-bit Lemire bounded draw. A game seed therefore
// names the same game here as it does on the drivers and in the scenario files.
uint32_t engine_seed_for_game(uint64_t game_seed);

}  // namespace agents

#endif /* AGENTS_GAME_SEED_H */
