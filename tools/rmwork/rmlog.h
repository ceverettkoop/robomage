#ifndef RMWORK_RMLOG_H
#define RMWORK_RMLOG_H

#include <cstdint>
#include <string>
#include <vector>

// ── A decision log's header and replies (RMLOG v3, src/input_logger.cpp) ────────────────────────
//
// "RMLOG v3", SEED, FLAGS, the two embedded decklists (DECK_A / DECK_B … END_DECK), DECISIONS, then
// one reply per line: a menu row, -1, or a concession sentinel (-2 / -3); a QUIT line ends a run a
// human seat quit. The workbench reads it to check a file before replaying it, to count the
// decisions the replay must produce and to name each decision's reply.

namespace rmwork {

struct RmlogInfo {
        uint64_t seed = 0;
        std::string flags;  // the FLAGS tokens as written ("bo3 machine")
        std::string deck_a, deck_b;
        std::vector<int> replies;  // one per decision, in order
        bool quit = false;         // the log ends with QUIT
        // FLAGS names `token` ("bo3", "no-auto-mana").
        bool has_flag(const std::string &token) const;
};

// False with `err` when `text` is not an RMLOG v3 log: another first line, a missing header field,
// a decklist without END_DECK, a reply that is not an integer.
bool parse_rmlog(const std::string &text, RmlogInfo *out, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_RMLOG_H */
