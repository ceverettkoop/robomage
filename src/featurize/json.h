#ifndef FEATURIZE_JSON_H
#define FEATURIZE_JSON_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// ── A small JSON reader, standard library only ─────────────────────────────────────────────────
//
// RFC 8259 values; objects keep their key order. Errors are returned, never thrown. Shared by the
// gate (the scenario files and the divergence registries), the RMTB tables' meta (rmtb.h), the
// model pin and the text-table inputs (oracle_text.h) and rmembed.

namespace featurize {

class Json {
    public:
        enum class Type {
            NUL,
            BOOL,
            NUMBER,
            STRING,
            ARRAY,
            OBJECT
        };

        // Parses `text` into `out`; false with `err` (what and where) on malformed input.
        static bool parse(const std::string &text, Json *out, std::string *err);

        Type type() const { return type_; }
        bool is_null() const { return type_ == Type::NUL; }
        bool is_bool() const { return type_ == Type::BOOL; }
        bool is_number() const { return type_ == Type::NUMBER; }
        bool is_string() const { return type_ == Type::STRING; }
        bool is_array() const { return type_ == Type::ARRAY; }
        bool is_object() const { return type_ == Type::OBJECT; }

        bool as_bool() const { return bool_; }
        double as_number() const { return number_; }
        // The number as an integer (truncated).
        int64_t as_int() const { return static_cast<int64_t>(number_); }
        // True for a number with no fraction or exponent in its text.
        bool is_integer() const { return integer_; }
        const std::string &as_string() const { return string_; }
        const std::vector<Json> &items() const { return items_; }
        const std::vector<std::pair<std::string, Json>> &members() const { return members_; }

        // The member `key` of an object, or nullptr.
        const Json *get(const std::string &key) const;
        bool has(const std::string &key) const { return get(key) != nullptr; }
        // Python truthiness: false, null, 0, "", [] and {} are false.
        bool truthy() const;
        // A string's text, an integer's digits, a number's shortest text (str() of the value).
        std::string text() const;

        // An object member set to `v` (replacing a member of that key) or removed.
        void set(const std::string &key, const Json &v);
        void erase(const std::string &key);

    private:
        class Parser;
        Type type_ = Type::NUL;
        bool bool_ = false;
        double number_ = 0;
        bool integer_ = false;
        std::string string_;
        std::vector<Json> items_;
        std::vector<std::pair<std::string, Json>> members_;
};

// `s` as a JSON string literal: quotes, backslashes and control characters escaped, every other
// byte (UTF-8 included) as it is.
std::string json_quote(const std::string &s);

}  // namespace featurize

#endif /* FEATURIZE_JSON_H */
