#ifndef RMWORK_NEW_GAME_DIALOG_H
#define RMWORK_NEW_GAME_DIALOG_H

#include <string>
#include <vector>

#include "settings.h"

// ── Game → New Game vs Model: the choices of a Play game ─────────────────────────────────────────
//
// Your deck and seat, the opponent (a model export, an inference server, explore, random or the
// scripted agent) and its deck, best of one or three, the seed (with a random one), auto-mana and
// Model's Read. The decks are the engine's top-level, league/ and heldout/ decks (under its working
// directory's resources/decks); the models Settings → Models' policies and every run's exports under
// its run directory (`<run>/exports/*.pt2`, `*.ts.pt`), and the exports inside the zip archives in
// the models directory (<repo>/models, model_zip.h). It opens with the last choices (Settings'
// new_game) and hands back the new ones when Start is pressed; the App starts the game and keeps
// them. The engine a choice needs is checked as it is made (an nn: opponent needs robomage_nn).

namespace rmwork {

struct ModelChoice {
        std::string label;  // "league26-24h/m9.pt2", "policy 0 · …"
        std::string path;   // absolute
};

class NewGameDialog {
    public:
        // Opens with `from`'s choices; lists the decks and models `s` names.
        void open(const NewGameSettings &from, const Settings &s, const std::string &engine_configured);
        // Draws it while open; true in the frame Start was pressed (`*out` the choices).
        bool draw(NewGameSettings *out);
        bool is_open() const { return open_ || want_open_; }
        // Why Start is disabled ("" when it is not).
        const std::string &problem() const { return problem_; }
        const NewGameSettings &choices() const { return g_; }

        // The decks under `workdir`/resources/decks: top-level, league/ and heldout/ ("delver",
        // "league/burn"), in natural order.
        static std::vector<std::string> list_decks(const std::string &workdir);
        // Settings → Models' policies, the exports of every run under the run directory, then those in
        // the models directory's zip archives (paths `<archive>.zip!<member>`).
        static std::vector<ModelChoice> list_models(const Settings &s);
        // The exports alone: every run's under the run directory, then the models directory's zipped ones
        // (Settings → Models' Add row suggests them).
        static std::vector<ModelChoice> list_exports(const Settings &s);
        // What keeps `g` from starting ("" when nothing does): an empty deck, an nn: opponent with no
        // model or socket, or no engine for it.
        static std::string check(const NewGameSettings &g, const std::string &engine_configured);

    private:
        void draw_deck_combo(const char *id, std::string *deck);
        void draw_opponent();
        bool want_open_ = false;
        bool open_ = false;
        NewGameSettings g_;
        std::vector<std::string> decks_;
        std::vector<ModelChoice> models_;
        std::string engine_;
        std::string socket_;  // Settings → Models' socket (Model's Read, the server's default)
        std::string problem_;
};

}  // namespace rmwork

#endif /* RMWORK_NEW_GAME_DIALOG_H */
