#ifndef TYPES_H
#define TYPES_H

#include <cstdint>
#include <string>

struct Type{
    uint32_t uid;
    bool is_supertype;
    std::string name;
};

#endif /* TYPES_H */
