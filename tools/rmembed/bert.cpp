#include "bert.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>

#include "../../src/featurize/json.h"
#include "safetensors.h"

bool BertEncoder::read_config(const std::string &path, std::string *err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        *err = "cannot read " + path;
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    featurize::Json cfg;
    std::string perr;
    if (!featurize::Json::parse(ss.str(), &cfg, &perr) || !cfg.is_object()) {
        *err = path + ": not a JSON object " + perr;
        return false;
    }
    const std::pair<const char *, int64_t *> ints[] = {{"hidden_size", &hidden_}, {"num_hidden_layers", &layers_},
        {"num_attention_heads", &heads_}, {"intermediate_size", &intermediate_},
        {"max_position_embeddings", &max_positions_}, {"type_vocab_size", &type_vocab_}, {"vocab_size", &vocab_size_}};
    for (const auto &k : ints) {
        const featurize::Json *v = cfg.get(k.first);
        if (!v || !v->is_integer() || v->as_int() <= 0) {
            *err = path + ": " + k.first + " is not a positive integer";
            return false;
        }
        *k.second = v->as_int();
    }
    if (hidden_ % heads_ != 0) {
        *err = path + ": hidden_size is not a multiple of num_attention_heads";
        return false;
    }
    const featurize::Json *eps = cfg.get("layer_norm_eps"), *act = cfg.get("hidden_act"),
                          *pos = cfg.get("position_embedding_type");
    if (eps) {
        if (!eps->is_number()) {
            *err = path + ": layer_norm_eps is not a number";
            return false;
        }
        eps_ = eps->as_number();
    }
    if (!act || !act->is_string() || act->as_string() != "gelu") {
        *err = path + ": hidden_act must be \"gelu\" (exact); others are not implemented";
        return false;
    }
    if (pos && !(pos->is_string() && pos->as_string() == "absolute")) {
        *err = path + ": only absolute position embeddings are implemented";
        return false;
    }
    return true;
}

// The tensor `name` (under the checkpoint's name prefix) of shape `shape`, moved out of tensors_.
bool BertEncoder::take(
    const std::string &name, const std::vector<int64_t> &shape, torch::Tensor *out, std::string *err) {
    auto it = tensors_.find(name_prefix_ + name);
    if (it == tensors_.end()) {
        *err = "model.safetensors has no " + name_prefix_ + name;
        return false;
    }
    if (it->second.sizes().vec() != shape) {
        std::ostringstream m;
        m << name_prefix_ << name << " has shape " << it->second.sizes() << ", expected " << c10::IntArrayRef(shape);
        *err = m.str();
        return false;
    }
    if (it->second.scalar_type() != torch::kFloat32) {
        *err = name_prefix_ + name + " is not a floating tensor";
        return false;
    }
    *out = it->second;
    tensors_.erase(it);
    return true;
}

// A LayerNorm's parameters: <prefix>.weight / .bias, or the older <prefix>.gamma / .beta.
bool BertEncoder::take_layer_norm(const std::string &prefix, torch::Tensor *w, torch::Tensor *b, std::string *err) {
    const bool old = tensors_.count(name_prefix_ + prefix + ".gamma") > 0;
    return take(prefix + (old ? ".gamma" : ".weight"), {hidden_}, w, err) &&
           take(prefix + (old ? ".beta" : ".bias"), {hidden_}, b, err);
}

bool BertEncoder::load(const std::string &model_dir, std::string *err) {
    if (!read_config(model_dir + "/config.json", err)) return false;
    if (!load_safetensors(model_dir + "/model.safetensors", &tensors_, err)) return false;
    name_prefix_ = tensors_.count("embeddings.word_embeddings.weight") ? "" : "bert.";
    const int64_t h = hidden_, i = intermediate_;
    if (!take("embeddings.word_embeddings.weight", {vocab_size_, h}, &word_, err) ||
        !take("embeddings.position_embeddings.weight", {max_positions_, h}, &position_, err) ||
        !take("embeddings.token_type_embeddings.weight", {type_vocab_, h}, &token_type_, err) ||
        !take_layer_norm("embeddings.LayerNorm", &emb_ln_w_, &emb_ln_b_, err))
        return false;
    stack_.assign(static_cast<size_t>(layers_), Layer());
    for (int64_t n = 0; n < layers_; ++n) {
        Layer &l = stack_[static_cast<size_t>(n)];
        const std::string p = "encoder.layer." + std::to_string(n) + ".";
        if (!take(p + "attention.self.query.weight", {h, h}, &l.q_w, err) ||
            !take(p + "attention.self.query.bias", {h}, &l.q_b, err) ||
            !take(p + "attention.self.key.weight", {h, h}, &l.k_w, err) ||
            !take(p + "attention.self.key.bias", {h}, &l.k_b, err) ||
            !take(p + "attention.self.value.weight", {h, h}, &l.v_w, err) ||
            !take(p + "attention.self.value.bias", {h}, &l.v_b, err) ||
            !take(p + "attention.output.dense.weight", {h, h}, &l.attn_out_w, err) ||
            !take(p + "attention.output.dense.bias", {h}, &l.attn_out_b, err) ||
            !take_layer_norm(p + "attention.output.LayerNorm", &l.attn_ln_w, &l.attn_ln_b, err) ||
            !take(p + "intermediate.dense.weight", {i, h}, &l.inter_w, err) ||
            !take(p + "intermediate.dense.bias", {i}, &l.inter_b, err) ||
            !take(p + "output.dense.weight", {h, i}, &l.out_w, err) ||
            !take(p + "output.dense.bias", {h}, &l.out_b, err) ||
            !take_layer_norm(p + "output.LayerNorm", &l.out_ln_w, &l.out_ln_b, err))
            return false;
    }
    tensors_.clear();  // the pooler and buffers such as position_ids are not used
    return true;
}

torch::Tensor BertEncoder::layer_forward(const Layer &l, const torch::Tensor &x, const torch::Tensor &mask_add) const {
    const int64_t b = x.size(0), n = x.size(1), hd = hidden_ / heads_;
    auto heads = [&](const torch::Tensor &t) { return t.view({b, n, heads_, hd}).permute({0, 2, 1, 3}); };
    const torch::Tensor q = heads(torch::linear(x, l.q_w, l.q_b));
    const torch::Tensor k = heads(torch::linear(x, l.k_w, l.k_b));
    const torch::Tensor v = heads(torch::linear(x, l.v_w, l.v_b));
    const double scale = 1.0 / std::sqrt(static_cast<double>(hd));
    const torch::Tensor scores = torch::matmul(q, k.transpose(-1, -2)) * scale + mask_add;
    const torch::Tensor ctx =
        torch::matmul(torch::softmax(scores, -1), v).permute({0, 2, 1, 3}).contiguous().view({b, n, hidden_});
    const torch::Tensor attn = torch::layer_norm(
        torch::linear(ctx, l.attn_out_w, l.attn_out_b) + x, {hidden_}, l.attn_ln_w, l.attn_ln_b, eps_);
    const torch::Tensor inter = torch::gelu(torch::linear(attn, l.inter_w, l.inter_b));
    return torch::layer_norm(torch::linear(inter, l.out_w, l.out_b) + attn, {hidden_}, l.out_ln_w, l.out_ln_b, eps_);
}

torch::Tensor BertEncoder::embed(const std::vector<std::vector<int64_t>> &batch, Pooling pooling) const {
    torch::NoGradGuard no_grad;
    const int64_t b = static_cast<int64_t>(batch.size());
    int64_t n = 0;
    for (const auto &ids : batch) n = std::max(n, static_cast<int64_t>(ids.size()));
    torch::Tensor ids = torch::zeros({b, n}, torch::kInt64);
    torch::Tensor mask = torch::zeros({b, n}, torch::kFloat32);
    for (int64_t r = 0; r < b; ++r) {
        const std::vector<int64_t> &row = batch[static_cast<size_t>(r)];
        for (int64_t c = 0; c < static_cast<int64_t>(row.size()); ++c) {
            ids[r][c] = row[static_cast<size_t>(c)];
            mask[r][c] = 1.0f;
        }
    }
    torch::Tensor x = torch::embedding(word_, ids) + position_.slice(0, 0, n).unsqueeze(0) + token_type_[0];
    x = torch::layer_norm(x, {hidden_}, emb_ln_w_, emb_ln_b_, eps_);
    const torch::Tensor mask_add = ((1.0f - mask) * std::numeric_limits<float>::lowest()).view({b, 1, 1, n});
    for (const Layer &l : stack_) x = layer_forward(l, x, mask_add);
    torch::Tensor pooled;
    if (pooling == Pooling::CLS) {
        pooled = x.select(1, 0);
    } else {
        pooled = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, true).clamp_min(1e-9);
    }
    return pooled / pooled.norm(2, 1, true).clamp_min(1e-12);
}
