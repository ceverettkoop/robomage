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
};

#endif /* TYPES_H */
