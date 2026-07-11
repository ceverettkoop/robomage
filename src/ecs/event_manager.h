#ifndef EVENT_MANAGER_H
#define EVENT_MANAGER_H
#include <functional>
#include <list>
#include <unordered_map>
#include <vector>
#include "event.h"

// credit https://austinmorlan.com/posts/entity_component_system/
class EventManager {
    public:
        void AddListener(EventId eventId, std::function<void(Event &)> const &listener) {
            listeners[eventId].push_back(listener);
        }
        void SendEvent(Event &event) {
            mPendingEvents.push_back(event);
            uint32_t type = event.GetType();
            for (auto const &listener : listeners[type]) {
                listener(event);
            }
        }
        void SendEvent(EventId eventId) {
            Event event(eventId);
            mPendingEvents.push_back(event);
            for (auto const &listener : listeners[eventId]) {
                listener(event);
            }
        }
        // Returns all events since the last drain and clears the buffer.
        std::vector<Event> drain_pending_events() {
            std::vector<Event> result;
            result.swap(mPendingEvents);
            return result;
        }
        // Snapshot/restore only the in-flight pending events (see snapshot.h);
        // listeners are per-game constants wired at init and must NOT be touched.
        std::vector<Event> snapshot_pending() const { return mPendingEvents; }
        void restore_pending(const std::vector<Event> &events) { mPendingEvents = events; }
    private:
        std::unordered_map<EventId, std::list<std::function<void(Event &)>>> listeners;
        std::vector<Event> mPendingEvents;
};
#endif /* EVENT_MANAGER_H */