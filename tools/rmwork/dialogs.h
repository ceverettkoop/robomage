#ifndef RMWORK_DIALOGS_H
#define RMWORK_DIALOGS_H

#include <string>

#include "settings.h"

// ── Modal dialogs: Settings and the error dialog ─────────────────────────────────────────────────
//
// Each is opened by a request (open()) and drawn every frame at the top level of the frame
// (draw()), so a request made from inside a menu still opens its popup in the right ID scope.

namespace rmwork {

class SettingsDialog {
    public:
        enum Tab {
            TABLE,
            PRINTINGS,
            KEYS,
            APPEARANCE
        };
        static constexpr int TAB_COUNT = 4;
        // The tab's name in lower case without spaces ("printings"), for --settings-tab; -1 if none.
        static int find_tab(const std::string &key);
        void open(Tab tab = TABLE);
        // Edits `s` in place; `settings_path` is shown as where the settings are kept.
        void draw(Settings *s, const std::string &settings_path);
        bool is_open() const { return open_; }

    private:
        void draw_table_tab(Settings *s, const std::string &settings_path);
        void draw_printings_tab();
        void draw_keys_tab(const Settings &s);
        void draw_appearance_tab(Settings *s, const std::string &settings_path);
        static void draw_theme_choice(std::string *theme, float field_x, float field_w);
        void sync_from(const Settings &s);
        bool want_open_ = false;
        bool open_ = false;
        Tab tab_ = TABLE;
        bool select_tab_ = false;
        // The values each background kind had this session, so switching kind restores them.
        std::string color_ = "#6e4526";
        std::string texture_ = "wood";
        char image_[1024] = {0};
};

// An error: a title, a message, a monospace detail text (an engine's stderr tail, a decoder's
// report) and "Copy diagnostics", which puts all three on the clipboard.
class ErrorDialog {
    public:
        void open(const std::string &title, const std::string &message, const std::string &detail);
        // True in the frame "Copy diagnostics" was pressed.
        bool draw();
        bool is_open() const { return open_; }
        // The clipboard text: title, message and detail.
        std::string diagnostics() const;

    private:
        bool want_open_ = false;
        bool open_ = false;
        std::string title_, message_, detail_;
};

}  // namespace rmwork

#endif /* RMWORK_DIALOGS_H */
