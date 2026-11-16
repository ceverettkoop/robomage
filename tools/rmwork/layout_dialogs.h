#ifndef RMWORK_LAYOUT_DIALOGS_H
#define RMWORK_LAYOUT_DIALOGS_H

#include <string>

#include "settings.h"

// ── View → Save Layout As… and Manage Layouts… ───────────────────────────────────────────────────
//
// Save Layout As names the current mode's dockspace as it is now (capture_layout) and keeps it in
// the settings' layouts for that mode, asking before it replaces a layout of the same name. Manage
// Layouts lists each mode's saved layouts to show, rename or delete. Opened by a request and drawn
// every frame at the top level of the frame, like every dialog.

namespace rmwork {

class LayoutDialogs {
    public:
        void open_save(Mode m);
        void open_manage(Mode m);
        // Draws whichever is open, editing `s->layouts`. True when the user asked to show a saved
        // layout: `*mode` and `*name` say which.
        bool draw(Settings *s, Mode *mode, std::string *name);
        bool save_open() const { return save_open_; }
        bool manage_open() const { return manage_open_; }
        // The Save dialog's name field (for the self-test).
        void set_save_name(const std::string &name);

    private:
        void draw_save(Settings *s);
        bool draw_manage(Settings *s, std::string *show);
        bool draw_manage_row(NamedLayouts *all, int i, std::string *show);
        void commit_save(Settings *s, bool overwrite);
        Mode mode_ = Mode::REPLAY;
        bool want_save_ = false;
        bool want_manage_ = false;
        bool save_open_ = false;
        bool manage_open_ = false;
        char name_[64] = {0};
        bool confirm_overwrite_ = false;
        bool focus_name_ = false;
        std::string error_;
        int renaming_ = -1;
        char rename_[64] = {0};
        int confirm_delete_ = -1;
        std::string manage_error_;
};

}  // namespace rmwork

#endif /* RMWORK_LAYOUT_DIALOGS_H */
