#ifndef RMEMBED_SAFETENSORS_H
#define RMEMBED_SAFETENSORS_H

#include <torch/torch.h>

#include <map>
#include <string>

// ── A safetensors file read directly (https://github.com/huggingface/safetensors) ─────────────
//
//   u64 LE header length | JSON header | raw tensor data
//
// The header maps each tensor name to {"dtype", "shape", "data_offsets": [begin, end]} (offsets
// into the data that follows the header; "__metadata__" is skipped). Floating tensors (F32, F16,
// BF16, F64) come back as float32 CPU tensors, integer ones (I64, I32) as int64; any other dtype,
// a shape that disagrees with its byte range, or a range outside the file is an error.
bool load_safetensors(const std::string &path, std::map<std::string, torch::Tensor> *out, std::string *err);

#endif /* RMEMBED_SAFETENSORS_H */
