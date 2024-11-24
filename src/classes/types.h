#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>

enum TypeCategory{
    TYPE,
    SUBTYPE,
    SUPERTYPE
};

//uhhh maybe should be entitites idk
struct Type{
    TypeCategory kind;
    std::string name;
};

#endif /* TYPES_H */
