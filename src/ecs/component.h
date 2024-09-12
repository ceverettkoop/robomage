#ifndef COMPONENT_H
#define COMPONENT_H

#include <cstdint>
#include <bitset>

// credit https://austinmorlan.com/posts/entity_component_system/

typedef uint8_t ComponentType;
const uint8_t MAX_COMPONENTS = UINT8_MAX;

typedef std::bitset<MAX_COMPONENTS> Signature;

#endif /* COMPONENT_H */
