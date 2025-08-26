#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>

enum TypeCategory{
    TYPE,
    SUBTYPE,
    SUPERTYPE
};

struct Type{
    TypeCategory kind;
    std::string name;
    
    bool operator<(const Type& other) const {
        if (kind != other.kind) {
            return kind < other.kind;
        }
        return name < other.name;
    }
};

#endif /* TYPES_H */
