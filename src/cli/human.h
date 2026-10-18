#ifndef CLI_HUMAN_H
#define CLI_HUMAN_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../agents/agent.h"

// ── A human seat: the C++ CLI (plan step 4b) ─────────────────────────────────────────────────
//
// An in-process agent (agents/agent.h) whose decisions a person makes at the terminal. Like every
// agent it reads only its seat's decoded frames, the bytes the RMSTATE stream would send the seat
// (knowledge tables included), handed over in memory by state_format/agent_feed.h; it renders them
// with cli/board.h, reads commands from stdin and writes to stdout.
//
// At each decision it prints what the seat saw happen since its previous decision (the frame's
// events, and the life totals that changed between its frames), the board when it differs from
// the one the seat last saw, the pending decision and the numbered menu, then prompts:
//
//   <n>                     menu row n; -1 = the last row (an attacker or blocker declaration's
//                           confirm)
//   concede / concede match concede the game (-2) / the match (-3)
//   board                   the board and menu again
//   zone <name> [opp]       one zone of your side (of the opponent's with opp): hand, library,
//                           graveyard, exile, battlefield, stack, sideboard
//   card <handle|name>      a card's details: an object of the board by handle (12 or #12), or a
//                           card of the match by name
//   know                    the knowledge panel in full
//   log [n|all]             the last n (default 30) lines of what this seat saw happen this match
//   help                    the commands
//   quit                    end the run
//
// Anything else re-prompts. End of input ends the run (exit status 0), as quit does.

namespace cli {

class HumanAgent : public agents::Agent {
    public:
        // `banner`: start each decision with a line naming the deciding seat (hotseat play).
        explicit HumanAgent(bool banner);
        int decide(const agents::Decision &d) override;
        std::string label() const override { return "human"; }

    private:
        // The lines the frame adds to the seat's log (its events, then life changes since the
        // seat's previous frame); appends them to log_.
        std::vector<std::string> take_news(const agents::DecisionView &v);
        // The board header, the board (when it differs from the last one shown, or `force`), the
        // pending decision and the menu.
        void show(const agents::DecisionView &v, bool force);
        // Runs one command line. Returns true with `reply` set when the line answers the decision.
        bool run_command(const std::string &line, const agents::DecisionView &v, int *reply);
        void show_log(const std::string &arg) const;
        static void print(const std::vector<std::string> &lines, const char *indent);
        static void print_help();
        [[noreturn]] static void quit(const char *why);

        bool banner_;
        std::vector<std::string> log_;      // every news line of the match, in order
        std::map<uint64_t, int64_t> life_;  // seat value -> life in the seat's previous frame
        std::vector<std::string> shown_;    // the board last printed
};

}  // namespace cli

#endif /* CLI_HUMAN_H */
