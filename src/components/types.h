#ifndef TYPES_H
#define TYPES_H

#include <set>

typedef enum SuperType{

};

typedef enum SubType{

};

struct Types{
    std::set<SuperType> super_types;
    std::set<SubType> sub_types;
}

#endif /* TYPES_H */
