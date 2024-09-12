#ifndef EVENT_H
#define EVENT_H

#include <any>
#include <cstdint>
#include <unordered_map>

// credit https://austinmorlan.com/posts/entity_component_system/

typedef uint32_t EventId;
typedef uint32_t ParamId;

class Event {
    public:
        Event() = delete;

        explicit Event(EventId type) : mType(type) {}

        template <typename T>
        void SetParam(EventId id, T value) {
            mData[id] = value;
        }

        template <typename T>
        T GetParam(EventId id) {
            return std::any_cast<T>(mData[id]);
        }

        EventId GetType() const { return mType; }

    private:
        EventId mType{};
        std::unordered_map<EventId, std::any> mData{};
};

#endif /* EVENT_H */
