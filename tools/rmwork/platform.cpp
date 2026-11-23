#include "platform.h"

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#include <GL/gl.h>
#endif
#include <GLFW/glfw3.h>

#include <cstdio>
#include <vector>

#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "theme.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

static void glfw_error(int code, const char *what);
static bool write_framebuffer_png(const std::string &path, int width, int height);

static void glfw_error(int code, const char *what) {
    std::fprintf(stderr, "rmwork: GLFW error %d: %s\n", code, what);
}

// The back buffer as an RGB PNG, top row first.
static bool write_framebuffer_png(const std::string &path, int width, int height) {
    std::vector<unsigned char> pixels(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    stbi_flip_vertically_on_write(1);
    return stbi_write_png(path.c_str(), width, height, 3, pixels.data(), width * 3) != 0;
}

namespace rmwork {

int run_window(App &app, const WindowOptions &opt) {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fprintf(stderr, "rmwork: cannot initialise GLFW (is there a display?)\n");
        return 1;
    }
#ifdef __APPLE__
    // macOS offers OpenGL 3 only as a forward-compatible 3.2+ core profile.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    const char *glsl_version = "#version 150";
#else
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    const char *glsl_version = "#version 130";
#endif
    GLFWwindow *window = glfwCreateWindow(opt.width, opt.height, "Robomage Workbench", nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "rmwork: cannot open an OpenGL 3 window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(opt.screenshot.empty() ? 1 : 0);
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    int status = 0;
    int frames = 0;
    const double started = glfwGetTime();
    while (!glfwWindowShouldClose(window) && !app.quit_requested()) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) != 0) {
            ImGui_ImplGlfw_Sleep(10);
            continue;
        }
        app.begin_frame();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
        int fb_w = 0, fb_h = 0;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);
        const ImVec4 bg = ImGui::ColorConvertU32ToFloat4(tok().BG);
        glClearColor(bg.x, bg.y, bg.z, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        const bool waiting = app.working() && glfwGetTime() - started < opt.screenshot_wait_s;
        if (!opt.screenshot.empty() && !waiting && ++frames >= opt.screenshot_frames) {
            if (write_framebuffer_png(opt.screenshot, fb_w, fb_h)) {
                std::printf("rmwork: wrote %s (%dx%d)\n", opt.screenshot.c_str(), fb_w, fb_h);
            } else {
                std::fprintf(stderr, "rmwork: cannot write %s\n", opt.screenshot.c_str());
                status = 1;
            }
            break;
        }
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    glfwDestroyWindow(window);
    glfwTerminate();
    return status;
}

}  // namespace rmwork
