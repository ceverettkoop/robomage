#ifndef RMWORK_DIALOGS_H
#define RMWORK_DIALOGS_H

#include <string>

// ── The error dialog ─────────────────────────────────────────────────────────────────────────────
//
// Opened by a request (open()) and drawn every frame at the top level of the frame (draw()), so a
// request made from inside a menu still opens its popup in the right ID scope, like every dialog.

namespace rmwork {

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
