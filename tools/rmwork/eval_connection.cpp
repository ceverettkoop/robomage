#include "eval_connection.h"

#include <chrono>
#include <thread>

#include "../../src/actor/remote_evaluator.h"
#include "files.h"
#include "subprocess.h"

namespace rmwork {

std::string oracle_text_table(const std::string &workdir) {
    return workdir + "/resources/embeddings/oracle_text.rmtb";
}

bool load_engine_features(const std::string &engine, const std::string &workdir, const std::string &cards_path,
    const std::atomic<bool> *stop, featurize::Featurization *out, std::string *why) {
    make_dirs(parent_dir(cards_path));
    ChildProcess cards;
    std::string err;
    const std::string cards_err = cards_path + ".err";
    // The engine runs in `workdir`: a relative path is the caller's, so it is made absolute here.
    if (!cards.start({absolute_path(engine), "--cards-frame", cards_path}, workdir, "/dev/null", cards_err, &err)) {
        *why = "the cards file: " + err;
        return false;
    }
    while (cards.poll()) {
        if (stop && stop->load()) {
            cards.kill_and_reap();
            *why = "";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!cards.ok()) {
        *why = "the cards file: " + engine + " --cards-frame " + cards.status_text();
        return false;
    }
    if (!out->load(cards_path, oracle_text_table(workdir), "", &err)) {
        *why = "the featurizer's inputs: " + err;
        return false;
    }
    return true;
}

std::string server_absent(const std::string &socket) {
    if (socket.empty()) return "no server socket is configured (Settings \xE2\x86\x92 Models)";
    if (path_kind(socket) != PathKind::SOCKET) return "no server at " + socket;
    return "";
}

std::unique_ptr<actor::Evaluator> connect_server(const std::string &socket, const std::vector<uint32_t> &policies,
    double timeout_s, const std::string &client, const featurize::Featurization &features, std::string *why) {
    actor::RemoteOptions opt;
    opt.endpoint = socket;
    opt.policies = policies;
    opt.timeout_s = timeout_s;
    opt.client = client;
    return actor::connect_remote_evaluator(opt, features, why);
}

}  // namespace rmwork
