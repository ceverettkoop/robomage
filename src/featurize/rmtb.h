#ifndef FEATURIZE_RMTB_H
#define FEATURIZE_RMTB_H

#include <cstdint>
#include <string>
#include <vector>

// ── RMTB: a file of one or more named arrays (docs/featurizer.md §4.4), standard library only ─
//
//   magic "RMTB" | u32 LE version (1) | u32 LE meta length | meta | data
//
// The meta is compact UTF-8 JSON, padded with spaces to a multiple of 8 bytes (the meta length
// counts the padding), with its keys in this fixed order, each optional key left out when unset:
//
//   kind, vocab_hash, spec_hash, model, pooling, norm_version, rows_meta, arrays
//
// so the same file always encodes to the same bytes. `arrays` lists every array in file order as
// {"name","dtype","shape":[…],"offset"[,"columns":[…]]}: its values are row-major little-endian
// values of dtype (f32, f16 = IEEE binary16, i64, i32, u16, u8) at `offset` bytes from the start
// of the data section (12 + meta length), each offset the previous array's end rounded up to a
// multiple of 8 (zero padding between), the file ending with the last array. `columns` (optional)
// names the last dimension's entries. Ragged data is a values array plus an i64 offsets array.
// python/rmtb.py reads the same layout with the standard library.

namespace featurize {

enum class Dtype {
    F32,
    F16,
    I64,
    I32,
    U16,
    U8
};

// A rows_meta entry of the text table: a present row's vocab id and the FNV-1a 64 of its
// normalised text (oracle_text.h).
struct RmtbRowMeta {
        int64_t id = 0;
        uint64_t hash = 0;
};

// One array: its values as little-endian bytes of `dtype` (rmtb_f32 / rmtb_i64 … build them,
// rmtb_values reads them).
struct RmtbArray {
        std::string name;
        Dtype dtype = Dtype::F32;
        std::vector<uint64_t> shape;
        std::vector<std::string> columns;  // names of the last dimension's entries, or empty
        std::string data;
        // The number of values (the product of the shape; 1 for a scalar).
        uint64_t count() const;
};

struct RmtbFile {
        std::string kind;  // card, text, ability, part, tensors, …
        bool has_vocab_hash = false;
        uint64_t vocab_hash = 0;
        bool has_spec_hash = false;
        uint64_t spec_hash = 0;
        std::string model;          // the text table's encoder, "<repository>@<commit>"; "" = absent
        std::string pooling;        // the text table's pooling (cls, mean); "" = absent
        int64_t norm_version = -1;  // the text table's normalisation version; -1 = absent
        bool has_rows_meta = false;
        std::vector<RmtbRowMeta> rows_meta;  // ascending id
        std::vector<RmtbArray> arrays;       // file order
        // The array `name`, or nullptr.
        const RmtbArray *find(const std::string &name) const;
};

// A value's width in bytes, and the dtype's name in the meta.
size_t dtype_width(Dtype d);
const char *dtype_name(Dtype d);

// IEEE binary16 from a float (round to nearest, ties to even; overflow to infinity, NaN kept),
// and back (exact).
uint16_t f32_to_f16(float v);
float f16_to_f32(uint16_t h);

// An array of `values` stored as f32 or f16 (`dtype`), or as i64, i32, u16 or u8 (`dtype`, each
// value cast to that width). Empty data (and so a refused encode) when the values are not shape's
// count many.
RmtbArray rmtb_floats(
    const std::string &name, Dtype dtype, const std::vector<uint64_t> &shape, const std::vector<float> &values);
RmtbArray rmtb_ints(
    const std::string &name, Dtype dtype, const std::vector<uint64_t> &shape, const std::vector<int64_t> &values);

// An array's values: as floats (f32, f16) or as integers (the integer dtypes); false for the
// other kind.
bool rmtb_values(const RmtbArray &a, std::vector<float> *out);
bool rmtb_values(const RmtbArray &a, std::vector<int64_t> *out);

// The meta's JSON (unpadded), with each array's offset as the file lays it out.
std::string rmtb_meta_json(const RmtbFile &file);

// The file bytes; "" when an array's data is not its shape's count of values, a `columns` list is
// not the last dimension's size, or two arrays share a name.
std::string rmtb_encode(const RmtbFile &file);

// Decodes file bytes; false with `err` (what is wrong) on a bad magic or version, a meta that is
// not the layout above, or arrays whose offsets and sizes do not tile the data section.
bool rmtb_decode(const std::string &bytes, RmtbFile *out, std::string *err);

// rmtb_decode of the file at `path` (an unreadable file is an error too).
bool rmtb_read_file(const std::string &path, RmtbFile *out, std::string *err);

// Writes `file` to `path` through a temporary file renamed over it, so a reader never sees half
// a file.
bool rmtb_write_file(const std::string &path, const RmtbFile &file, std::string *err);

// 16 lowercase hex digits; the inverse (false unless exactly 16 hex digits).
std::string hex16(uint64_t v);
bool parse_hex16(const std::string &s, uint64_t *out);

}  // namespace featurize

#endif /* FEATURIZE_RMTB_H */
