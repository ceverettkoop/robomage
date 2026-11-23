#ifndef RMWORK_INFERENCE_SERVER_H
#define RMWORK_INFERENCE_SERVER_H

#include <string>
#include <utility>
#include <vector>

#include "subprocess.h"

// ── The inference server the workbench starts (Settings → Models → Start the server) ─────────────
//
// rmserve (beside the engine binary, `make actor`) run for the workbench session: started with the
// configured socket and policies (`--policy ID=MODEL`, a zipped export extracted first, model_zip.h),
// stopped with SIGTERM when the setting is turned off, the configuration changes or the workbench
// quits. A server already listening at the socket is used as it is and never stopped. Its output
// goes to files in the workbench cache; a server that exits is reported with the end of its stderr.

namespace rmwork {

struct ServerConfig {
        std::string binary;  // rmserve
        std::string socket;
        std::string cwd;                                         // relative model paths' directory
        std::string log_dir;                                     // rmserve.out, rmserve.err
        std::vector<std::pair<std::string, std::string>> policies;  // id, model (a file or a zip reference)
};

// The configuration as a string: a change of it restarts the server.
std::string server_config_key(const ServerConfig &c);
// Whether a server accepts connections at the Unix socket `path`.
bool socket_listening(const std::string &path);

class InferenceServer {
    public:
        enum class State {
            OFF,       // not started (or stopped)
            EXTERNAL,  // another process serves the socket
            STARTING,  // started; its socket is not listening yet
            RUNNING,   // started and listening
            FAILED     // could not start, or exited
        };
        InferenceServer() = default;
        InferenceServer(const InferenceServer &) = delete;
        InferenceServer &operator=(const InferenceServer &) = delete;
        ~InferenceServer() { stop(); }

        // Starts rmserve for `c` (after stopping one this object started), or takes the server
        // already listening at its socket (EXTERNAL). FAILED with message() when it cannot start.
        void start(const ServerConfig &c);
        // Stops the server this object started (SIGTERM, then SIGKILL after 3 s); OFF.
        void stop();
        // STARTING becomes RUNNING once the socket listens; a started server that exited is FAILED.
        // True when this call made it FAILED.
        bool poll();
        State state() const { return state_; }
        // Why it failed ("" otherwise); the end of rmserve's stderr in detail().
        const std::string &message() const { return message_; }
        const std::string &detail() const { return detail_; }
        // The configuration it was started (or taken, or refused) with.
        const std::string &key() const { return key_; }
        // The command line it ran.
        const std::vector<std::string> &argv() const { return argv_; }

    private:
        void fail(const std::string &message, const std::string &detail);
        ChildProcess child_;
        bool owned_ = false;
        State state_ = State::OFF;
        std::string key_, message_, detail_, socket_, err_path_;
        std::vector<std::string> argv_;
};

// "Inference: server running", "Inference: starting server", … (the status bar).
const char *server_state_text(InferenceServer::State s);

}  // namespace rmwork

#endif /* RMWORK_INFERENCE_SERVER_H */
