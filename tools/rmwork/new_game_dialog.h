#ifndef RMWORK_NEW_GAME_DIALOG_H
#define RMWORK_NEW_GAME_DIALOG_H

#include <cstdint>
#include <string>
#include <vector>

#include "settings.h"

// ── Game → New Game vs Model: the choices of a Play game ─────────────────────────────────────────
//
// Your deck and seat, the opponent (a model export, an inference server, explore, random or the
// scripted agent) and its deck, best of one or three, the seed (with a random one), auto-mana and
// Model's Read. The decks are the engine's top-level, league/ and heldout/ decks (under its working
// directory's resources/decks); the models the runs directory's `latest` first, then Settings →
// Models' policies, then the runs' policies newest first and the zipped exports of the models
// directory (<repo>/models, model_zip.h): policy_catalog.h's rows, its hidden ones left out. It opens
// with the last choices (Settings' new_game), the model `latest` when none was chosen before, and
// hands back the new ones when Start is pressed; the App starts the game and keeps them. The engine a
// choice needs is checked as it is made (an nn: opponent needs robomage_nn).

namespace rmwork {

struct ModelChoice {
        std::string label;    // "2026-11-21 · league26-24h/m9", "latest · league26-24h m9", "policy 0 · …"
        std::string path;     // absolute
        bool latest = false;  // the runs directory's latest
};

// A seed from 1 to 999 999 (the dialogs' Random button).
uint64_t random_seed();
// The widgets the New Game and Simulate dialogs share (each in its own ID scope `id`).
// An agent kind's combo box (play_session.h's kinds, `*key` the chosen one's key).
void agent_kind_combo(const char *id, std::string *key);
// A deck's combo box over `decks` (NewGameDialog::list_decks), grouped by directory.
void deck_combo(const char *id, const std::vector<std::string> &decks, std::string *deck);
// A model's combo box over `models` and, on the next line (its label column `field_x` wide), a field for
// the export's path.
void model_combo(const char *id, const std::vector<ModelChoice> &models, std::string *model, float field_x);
// What keeps an agent of kind `key` from playing ("" nothing): a model kind with no model or one that is
// not there, a server kind with no socket or none there.
std::string agent_problem(const std::string &key, const std::string &model, const std::string &server);

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
        // `latest`, Settings → Models' policies, then the rest of list_exports.
        static std::vector<ModelChoice> list_models(const Settings &s);
        // The exports alone, the policy catalog's rows that are shown and play an export: `latest`, every
        // run's newest first, then the models directory's zipped ones (paths `<archive>.zip!<member>`;
        // Settings → Models' Add row suggests them).
        static std::vector<ModelChoice> list_exports(const Settings &s);
        // What keeps `g` from starting ("" when nothing does): an empty deck, an nn: opponent with no
        // model or socket, or no engine for it.
        static std::string check(const NewGameSettings &g, const std::string &engine_configured);

    private:
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
