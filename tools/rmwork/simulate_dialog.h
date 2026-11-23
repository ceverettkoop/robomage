#ifndef RMWORK_SIMULATE_DIALOG_H
#define RMWORK_SIMULATE_DIALOG_H

#include <string>
#include <vector>

#include "new_game_dialog.h"
#include "settings.h"

// ── File → Simulate Match…: the choices of a simulation (simulation.h, plan §2.9) ───────────────
//
// For each seat (the first entrant is seat A of the first match) an agent (a model export from the
// policy catalog, zipped ones included, as New Game lists them; a served policy, `nn:remote:` at
// Settings → Models' socket by default; the scripted agent; explore; random), a model's options (argmax,
// or a sampling temperature) and a deck; then best of one or three, auto-mana, the first seed (match i
// plays seed + i), the number of matches, whether the seats alternate (odd matches swap the entrants'
// agents and decks) and how many engines run at once (default a quarter of the hardware threads). It
// opens with the last choices (Settings' simulate) and hands back the new ones when Run is pressed; the
// App starts the simulation and keeps them. The engine a choice needs is checked as it is made (a model
// or a served policy needs robomage_nn).

namespace rmwork {

class SimulateDialog {
    public:
        // Opens with `from`'s choices; lists the decks and models `s` names.
        void open(const SimulateSettings &from, const Settings &s, const std::string &engine_configured);
        // Draws it while open; true in the frame Run was pressed (`*out` the choices).
        bool draw(SimulateSettings *out);
        bool is_open() const { return open_ || want_open_; }
        // Why Run is disabled ("" when it is not).
        const std::string &problem() const { return problem_; }
        const SimulateSettings &choices() const { return g_; }

        // What keeps `g` from running ("" when nothing does): an unknown agent, an empty deck, a model
        // agent with no model, a served one with no server, no engine for them, no match.
        static std::string check(const SimulateSettings &g, const std::string &engine_configured);

    private:
        void draw_seat(int seat);
        bool want_open_ = false;
        bool open_ = false;
        SimulateSettings g_;
        std::vector<std::string> decks_;
        std::vector<ModelChoice> models_;
        std::string engine_;
        std::string socket_;  // Settings → Models' socket (a served policy's default)
        std::string problem_;
};

}  // namespace rmwork

#endif /* RMWORK_SIMULATE_DIALOG_H */
