#ifndef RMWORK_SETTINGS_DIALOG_H
#define RMWORK_SETTINGS_DIALOG_H

#include <string>

#include "files.h"
#include "settings.h"

// ── Edit → Settings: a modal of tabs over the settings, each change applied as it is made ───────
//
// Opened by a request (open()) and drawn every frame at the top level of the frame (draw()). The
// Keys tab rebinds an action by capturing the next chord pressed (Esc cancels); a chord another
// action already answers to where both work is refused with a message.

namespace rmwork {

class SettingsDialog {
    public:
        enum Tab {
            TABLE,
            PRINTINGS,
            KEYS,
            APPEARANCE,
            ENGINE,
            MODELS,
            STOPS
        };
        static constexpr int TAB_COUNT = 7;
        // The tab's key ("printings", "engine", …), for --settings-tab; -1 if none.
        static int find_tab(const std::string &key);
        void open(Tab tab = TABLE);
        // Edits `s` in place; `settings_path` is shown as where the settings are kept.
        void draw(Settings *s, const std::string &settings_path);
        bool is_open() const { return open_; }
        // Opens the Keys tab waiting for `a`'s new chord.
        void begin_key_capture(Action a);
        bool capturing() const { return capture_ >= 0; }
        // The Keys tab's last message (a refused chord, a reset), "" for none, and whether it is a
        // refusal.
        const std::string &key_message() const { return key_message_; }
        bool key_message_is_error() const { return key_message_error_; }
        // A row's Reset: `a` back to its default key unless another action has taken it where both
        // work; false (with the message) then.
        bool reset_key(Settings *s, Action a);
        // Reset All: every key back to its default.
        void reset_all_keys(Settings *s);

    private:
        // What a path setting must name to count as found.
        enum class Want {
            EXECUTABLE,
            DIRECTORY,
            SOCKET,
            FILE
        };
        void draw_table_tab(Settings *s, const std::string &settings_path);
        void draw_printings_tab(Settings *s);
        void draw_keys_tab(Settings *s);
        void capture_key(Settings *s);
        void draw_appearance_tab(Settings *s, const std::string &settings_path);
        static void draw_theme_choice(std::string *theme, float field_x, float field_w);
        void draw_engine_tab(Settings *s);
        void draw_models_tab(Settings *s);
        void draw_policies(ModelSettings *m, float field_x);
        static void draw_stops_tab(Settings *s);
        static bool stop_row(const char *label, std::array<bool, STOP_STEP_COUNT> *row);
        // A labelled path field ("" = `fallback`, shown as the hint) and, under it, the resolved
        // path with a found / missing mark.
        static void path_field(const char *label, std::string *value, const std::string &fallback, Want want,
            float field_x, const char *missing_note);
        static void path_mark(const std::string &resolved, Want want, const char *missing_note);
        static bool path_found(const std::string &resolved, Want want);
        void sync_from(const Settings &s);
        bool want_open_ = false;
        bool open_ = false;
        Tab tab_ = TABLE;
        bool select_tab_ = false;
        // The values each background kind had this session, so switching kind restores them.
        std::string color_ = "#6e4526";
        std::string texture_ = "wood";
        char image_[1024] = {0};
        int capture_ = -1;  // the action waiting for its new chord
        std::string key_message_;
        bool key_message_error_ = false;
        char policy_id_[64] = {0};
        char policy_model_[1024] = {0};
        std::string policy_error_;
};

}  // namespace rmwork

#endif /* RMWORK_SETTINGS_DIALOG_H */
