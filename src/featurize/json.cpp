#include "json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace featurize {

// A recursive-descent reader over one text.
class Json::Parser {
    public:
        explicit Parser(const std::string &text) : s_(text) {}

        bool document(Json *out, std::string *err) {
            ws();
            if (!value(out, 0)) {
                *err = error_;
                return false;
            }
            ws();
            if (pos_ != s_.size()) {
                *err = where("trailing characters");
                return false;
            }
            return true;
        }

    private:
        static constexpr int MAX_DEPTH = 256;

        std::string where(const std::string &what) const {
            size_t line = 1, col = 1;
            for (size_t i = 0; i < pos_ && i < s_.size(); ++i) {
                if (s_[i] == '\n') {
                    ++line;
                    col = 1;
                } else {
                    ++col;
                }
            }
            return what + " at line " + std::to_string(line) + " column " + std::to_string(col);
        }

        bool fail(const std::string &what) {
            if (error_.empty()) error_ = where(what);
            return false;
        }

        void ws() {
            while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r'))
                ++pos_;
        }

        bool literal(const char *word) {
            size_t i = 0;
            for (; word[i]; ++i)
                if (pos_ + i >= s_.size() || s_[pos_ + i] != word[i]) return false;
            pos_ += i;
            return true;
        }

        bool value(Json *out, int depth) {
            if (depth > MAX_DEPTH) return fail("nesting too deep");
            if (pos_ >= s_.size()) return fail("unexpected end");
            const char c = s_[pos_];
            if (c == '{') return object(out, depth);
            if (c == '[') return array(out, depth);
            if (c == '"') {
                out->type_ = Type::STRING;
                return string(&out->string_);
            }
            if (literal("true")) {
                out->type_ = Type::BOOL;
                out->bool_ = true;
                return true;
            }
            if (literal("false")) {
                out->type_ = Type::BOOL;
                out->bool_ = false;
                return true;
            }
            if (literal("null")) {
                out->type_ = Type::NUL;
                return true;
            }
            if (c == '-' || (c >= '0' && c <= '9')) return number(out);
            return fail("unexpected character");
        }

        bool object(Json *out, int depth) {
            out->type_ = Type::OBJECT;
            ++pos_;
            ws();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                return true;
            }
            while (true) {
                ws();
                if (pos_ >= s_.size() || s_[pos_] != '"') return fail("expected a key");
                std::string key;
                if (!string(&key)) return false;
                ws();
                if (pos_ >= s_.size() || s_[pos_] != ':') return fail("expected ':'");
                ++pos_;
                ws();
                out->members_.emplace_back(key, Json());
                if (!value(&out->members_.back().second, depth + 1)) return false;
                ws();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == '}') {
                    ++pos_;
                    return true;
                }
                return fail("expected ',' or '}'");
            }
        }

        bool array(Json *out, int depth) {
            out->type_ = Type::ARRAY;
            ++pos_;
            ws();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                return true;
            }
            while (true) {
                ws();
                out->items_.emplace_back();
                if (!value(&out->items_.back(), depth + 1)) return false;
                ws();
                if (pos_ < s_.size() && s_[pos_] == ',') {
                    ++pos_;
                    continue;
                }
                if (pos_ < s_.size() && s_[pos_] == ']') {
                    ++pos_;
                    return true;
                }
                return fail("expected ',' or ']'");
            }
        }

        bool hex4(uint32_t *v) {
            if (pos_ + 4 > s_.size()) return fail("truncated \\u escape");
            *v = 0;
            for (int i = 0; i < 4; ++i) {
                const char c = s_[pos_++];
                *v <<= 4;
                if (c >= '0' && c <= '9')
                    *v |= static_cast<uint32_t>(c - '0');
                else if (c >= 'a' && c <= 'f')
                    *v |= static_cast<uint32_t>(c - 'a' + 10);
                else if (c >= 'A' && c <= 'F')
                    *v |= static_cast<uint32_t>(c - 'A' + 10);
                else
                    return fail("bad \\u escape");
            }
            return true;
        }

        static void utf8(uint32_t cp, std::string *out) {
            if (cp < 0x80) {
                *out += static_cast<char>(cp);
            } else if (cp < 0x800) {
                *out += static_cast<char>(0xC0 | (cp >> 6));
                *out += static_cast<char>(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                *out += static_cast<char>(0xE0 | (cp >> 12));
                *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                *out += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
                *out += static_cast<char>(0xF0 | (cp >> 18));
                *out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
                *out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                *out += static_cast<char>(0x80 | (cp & 0x3F));
            }
        }

        bool string(std::string *out) {
            ++pos_;  // the opening quote
            while (true) {
                if (pos_ >= s_.size()) return fail("unterminated string");
                const char c = s_[pos_++];
                if (c == '"') return true;
                if (static_cast<unsigned char>(c) < 0x20) return fail("control character in a string");
                if (c != '\\') {
                    *out += c;
                    continue;
                }
                if (pos_ >= s_.size()) return fail("unterminated escape");
                const char e = s_[pos_++];
                switch (e) {
                    case '"':
                        *out += '"';
                        break;
                    case '\\':
                        *out += '\\';
                        break;
                    case '/':
                        *out += '/';
                        break;
                    case 'b':
                        *out += '\b';
                        break;
                    case 'f':
                        *out += '\f';
                        break;
                    case 'n':
                        *out += '\n';
                        break;
                    case 'r':
                        *out += '\r';
                        break;
                    case 't':
                        *out += '\t';
                        break;
                    case 'u': {
                        uint32_t cp;
                        if (!hex4(&cp)) return false;
                        if (cp >= 0xD800 && cp < 0xDC00 && pos_ + 1 < s_.size() && s_[pos_] == '\\' &&
                            s_[pos_ + 1] == 'u') {
                            pos_ += 2;
                            uint32_t lo;
                            if (!hex4(&lo)) return false;
                            if (lo >= 0xDC00 && lo < 0xE000) {
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            } else {
                                utf8(cp, out);
                                cp = lo;
                            }
                        }
                        utf8(cp, out);
                        break;
                    }
                    default:
                        return fail("bad escape");
                }
            }
        }

        bool number(Json *out) {
            const size_t start = pos_;
            bool integer = true;
            if (s_[pos_] == '-') ++pos_;
            while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
            if (pos_ < s_.size() && s_[pos_] == '.') {
                integer = false;
                ++pos_;
                while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
            }
            if (pos_ < s_.size() && (s_[pos_] == 'e' || s_[pos_] == 'E')) {
                integer = false;
                ++pos_;
                if (pos_ < s_.size() && (s_[pos_] == '+' || s_[pos_] == '-')) ++pos_;
                while (pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '9') ++pos_;
            }
            const std::string text = s_.substr(start, pos_ - start);
            if (text == "-" || text.empty()) return fail("bad number");
            out->type_ = Type::NUMBER;
            out->integer_ = integer;
            out->number_ = std::strtod(text.c_str(), nullptr);
            return true;
        }

        const std::string &s_;
        size_t pos_ = 0;
        std::string error_;
};

bool Json::parse(const std::string &text, Json *out, std::string *err) {
    *out = Json();
    Parser p(text);
    return p.document(out, err);
}

const Json *Json::get(const std::string &key) const {
    for (const auto &m : members_)
        if (m.first == key) return &m.second;
    return nullptr;
}

void Json::set(const std::string &key, const Json &v) {
    type_ = Type::OBJECT;
    for (auto &m : members_)
        if (m.first == key) {
            m.second = v;
            return;
        }
    members_.emplace_back(key, v);
}

void Json::erase(const std::string &key) {
    for (size_t i = 0; i < members_.size(); ++i)
        if (members_[i].first == key) {
            members_.erase(members_.begin() + static_cast<long>(i));
            return;
        }
}

bool Json::truthy() const {
    switch (type_) {
        case Type::NUL:
            return false;
        case Type::BOOL:
            return bool_;
        case Type::NUMBER:
            return number_ != 0;
        case Type::STRING:
            return !string_.empty();
        case Type::ARRAY:
            return !items_.empty();
        case Type::OBJECT:
            return !members_.empty();
    }
    return false;
}

std::string Json::text() const {
    if (type_ == Type::STRING) return string_;
    if (type_ == Type::NUMBER) {
        if (integer_) return std::to_string(static_cast<long long>(number_));
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.17g", number_);
        return buf;
    }
    if (type_ == Type::BOOL) return bool_ ? "True" : "False";
    if (type_ == Type::NUL) return "None";
    return "";
}

std::string json_quote(const std::string &s) {
    std::string out = "\"";
    for (const char c : s) {
        const unsigned char u = static_cast<unsigned char>(c);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\r') {
            out += "\\r";
        } else if (u < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", u);
            out += buf;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

}  // namespace featurize
