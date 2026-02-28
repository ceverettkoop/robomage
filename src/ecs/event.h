#ifndef EVENT_H
#define EVENT_H

#include <cstdint>
#include <unordered_map>
#include <any>

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
        T GetParam(EventId id) const {
            return std::any_cast<T>(mData.at(id));
        }
        bool HasParam(EventId id) const { return mData.count(id) > 0; }
        EventId GetType() const { return mType; }

    private:
        EventId mType{};
        std::unordered_map<EventId, std::any> mData{};
};

#endif /* EVENT_H */