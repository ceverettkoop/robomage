#ifndef RMKNOW_SELFTEST_H
#define RMKNOW_SELFTEST_H

// Unit tests of the knowledge sidecar's update rules on small synthetic event streams; prints one
// line per test and a summary, returns the process exit status (0 = all passed).
int run_selftest();
// The solver and compaction against the independent enumerator (exactness.cpp): `n` synthetic
// histories and `n` event streams, plus the sculpted cases.
int run_exactness(int n);
// The belief layer (src/knowledge/belief.h) against the weighted enumeration of the hypothesis
// networks and the model played out directly (exactness.cpp, generative.h): `n` simulated games.
int run_belief_exactness(int n);

#endif /* RMKNOW_SELFTEST_H */
