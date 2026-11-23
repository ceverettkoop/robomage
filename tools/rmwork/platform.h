#ifndef RMWORK_PLATFORM_H
#define RMWORK_PLATFORM_H

#include <string>

#include "app.h"

// ── The window: GLFW, OpenGL 3 and the ImGui backends around App::frame() ───────────────────────

namespace rmwork {

struct WindowOptions {
        int width = 1600;
        int height = 1000;
        // When set: render `screenshot_frames` frames once no game is being opened and no run
        // directory read (at most `screenshot_wait_s` seconds of waiting), write the last one to
        // this PNG and return.
        std::string screenshot;
        int screenshot_frames = 30;
        double screenshot_wait_s = 300;
};

// Runs the UI until the window closes or File → Exit (or, with a screenshot, for its frames); the
// ImGui context must exist. Returns the process exit status.
int run_window(App &app, const WindowOptions &opt);

}  // namespace rmwork

#endif /* RMWORK_PLATFORM_H */
