#ifndef RMEMBED_BERT_H
#define RMEMBED_BERT_H

#include <torch/torch.h>

#include <map>
#include <string>
#include <vector>

// ── A BERT encoder in the libtorch C++ API, with sentence pooling (docs/featurizer.md §5) ─────
//
// Reads a Hugging Face BERT model directory: config.json (hidden_size, num_hidden_layers,
// num_attention_heads, intermediate_size, layer_norm_eps, hidden_act "gelu",
// max_position_embeddings, type_vocab_size, vocab_size; absolute position embeddings) and
// model.safetensors (the standard tensor names, with or without a "bert." prefix; LayerNorm
// weight/bias or gamma/beta). Forward, in float32 on the CPU, as transformers' BertModel in eval
// mode: word + position + token type 0 embeddings, LayerNorm; per layer self-attention under the
// padding mask, dense + residual + LayerNorm, intermediate dense with exact (erf) GELU, output
// dense + residual + LayerNorm. Then pooling (cls: the first token's vector; mean: the mean over
// real tokens) and L2 normalisation.

class BertEncoder {
    public:
        enum class Pooling {
            CLS,
            MEAN
        };

        // False with `err` on an unreadable or unsupported config, or a missing or misshapen tensor.
        bool load(const std::string &model_dir, std::string *err);

        // The pooled, normalised vectors [batch, hidden] of token id sequences (each [CLS] … [SEP],
        // none longer than max_positions()); shorter ones are padded and masked.
        torch::Tensor embed(const std::vector<std::vector<int64_t>> &batch, Pooling pooling) const;

        int64_t hidden() const { return hidden_; }
        int64_t max_positions() const { return max_positions_; }
        int64_t vocab_size() const { return vocab_size_; }

    private:
        struct Layer {
                torch::Tensor q_w, q_b, k_w, k_b, v_w, v_b;
                torch::Tensor attn_out_w, attn_out_b, attn_ln_w, attn_ln_b;
                torch::Tensor inter_w, inter_b, out_w, out_b, out_ln_w, out_ln_b;
        };

        bool read_config(const std::string &path, std::string *err);
        bool take(const std::string &name, const std::vector<int64_t> &shape, torch::Tensor *out, std::string *err);
        bool take_layer_norm(const std::string &prefix, torch::Tensor *w, torch::Tensor *b, std::string *err);
        torch::Tensor layer_forward(const Layer &l, const torch::Tensor &x, const torch::Tensor &mask_add) const;

        std::map<std::string, torch::Tensor> tensors_;  // loaded weights, emptied as they are taken
        std::string name_prefix_;                       // "" or "bert."
        int64_t hidden_ = 0, layers_ = 0, heads_ = 0, intermediate_ = 0;
        int64_t max_positions_ = 0, type_vocab_ = 0, vocab_size_ = 0;
        double eps_ = 1e-12;
        torch::Tensor word_, position_, token_type_, emb_ln_w_, emb_ln_b_;
        std::vector<Layer> stack_;
};

#endif /* RMEMBED_BERT_H */
