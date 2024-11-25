// credit https://austinmorlan.com/posts/entity_component_system/
#ifndef COMPONENT_H
#define COMPONENT_H

#include <cstdint>
#include <bitset>

using ComponentType = uint8_t;
const ComponentType MAX_COMPONENTS = 32;

typedef std::bitset<MAX_COMPONENTS> Signature;

#endif /* COMPONENT_H */
