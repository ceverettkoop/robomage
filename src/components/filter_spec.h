#ifndef FILTER_SPEC_H
#define FILTER_SPEC_H

#include <string>

struct CompiledFilter;

// A Forge filter spec ("Creature.YouCtrl+nonToken,Planeswalker") as a definition holds it: a handle
// to the spec compiled once by the qualifier table (queries/filters.cpp). Compiled specs are
// interned by text in a process-wide append-only store and name objects only by what they mean, so
// a handle stays valid, like a const AbilityDef *, across snapshot restores and per-game ECS resets,
// and two handles are equal exactly when their texts are. The empty spec has no compiled form.
class FilterSpec {
    public:
        FilterSpec() = default;
        explicit FilterSpec(const std::string &text);
        explicit FilterSpec(const char *text) : FilterSpec(std::string(text)) {}
        FilterSpec &operator=(const std::string &text) { return *this = FilterSpec(text); }
        FilterSpec &operator=(const char *text) { return *this = FilterSpec(text); }

        const std::string &text() const;
        operator const std::string &() const { return text(); }
        bool empty() const { return compiled_ == nullptr; }
        const CompiledFilter *compiled() const { return compiled_; }

        bool operator==(const FilterSpec &o) const { return compiled_ == o.compiled_; }
        bool operator!=(const FilterSpec &o) const { return compiled_ != o.compiled_; }
        bool operator==(const std::string &text_) const { return text() == text_; }
        bool operator!=(const std::string &text_) const { return text() != text_; }
        bool operator==(const char *text_) const { return text() == text_; }
        bool operator!=(const char *text_) const { return text() != text_; }

    private:
        const CompiledFilter *compiled_ = nullptr;
};

#endif /* FILTER_SPEC_H */
