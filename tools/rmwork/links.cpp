#include "links.h"

#include <algorithm>

namespace rmwork {

void LinkState::set_context(unsigned generation) {
    if (have_generation_ && generation == generation_) return;
    have_generation_ = true;
    generation_ = generation;
    hovered_ = next_hover_ = selected_ = LinkItem();
    objects_.clear();
    players_.clear();
    rows_.clear();
}

void LinkState::new_frame() {
    hovered_ = next_hover_;
    next_hover_ = LinkItem();
    objects_.clear();
    players_.clear();
    rows_.clear();
    link(hovered_.kind != LinkItem::Kind::NONE ? hovered_ : selected_);
}

void LinkState::hover_object(uint64_t handle) {
    next_hover_ = LinkItem();
    next_hover_.kind = LinkItem::Kind::OBJECT;
    next_hover_.handle = handle;
}

void LinkState::hover_player(int seat) {
    next_hover_ = LinkItem();
    next_hover_.kind = LinkItem::Kind::PLAYER;
    next_hover_.seat = seat;
}

void LinkState::hover_menu_row(int row) {
    next_hover_ = LinkItem();
    next_hover_.kind = LinkItem::Kind::MENU_ROW;
    next_hover_.row = row;
}

void LinkState::select(const LinkItem &item) {
    selected_ = selected_ == item ? LinkItem() : item;
}

bool LinkState::object_linked(uint64_t handle) const {
    return handle != 0 && std::find(objects_.begin(), objects_.end(), handle) != objects_.end();
}

bool LinkState::player_linked(int seat) const {
    return seat >= 0 && std::find(players_.begin(), players_.end(), seat) != players_.end();
}

bool LinkState::row_linked(int row) const {
    return std::find(rows_.begin(), rows_.end(), row) != rows_.end();
}

bool LinkState::is_source(uint64_t handle) const {
    if (handle == 0) return false;
    for (const MenuLink &m : menu_)
        if (m.source == handle) return true;
    return false;
}

void LinkState::link(const LinkItem &item) {
    switch (item.kind) {
        case LinkItem::Kind::NONE:
            return;
        case LinkItem::Kind::OBJECT: {
            objects_.push_back(item.handle);
            for (size_t r = 0; r < menu_.size(); ++r)
                if (menu_[r].source == item.handle || menu_[r].target == item.handle)
                    rows_.push_back(static_cast<int>(r));
            auto it = stack_targets_.find(item.handle);
            if (it == stack_targets_.end()) return;
            for (const StackTarget &t : it->second) {
                if (t.player) {
                    players_.push_back(t.seat);
                } else if (t.handle) {
                    objects_.push_back(t.handle);
                }
            }
            return;
        }
        case LinkItem::Kind::PLAYER:
            players_.push_back(item.seat);
            for (size_t r = 0; r < menu_.size(); ++r)
                if (menu_[r].source_seat == item.seat || menu_[r].target_seat == item.seat)
                    rows_.push_back(static_cast<int>(r));
            return;
        case LinkItem::Kind::MENU_ROW: {
            rows_.push_back(item.row);
            if (item.row < 0 || static_cast<size_t>(item.row) >= menu_.size()) return;
            const MenuLink &m = menu_[static_cast<size_t>(item.row)];
            if (m.source) objects_.push_back(m.source);
            if (m.target) objects_.push_back(m.target);
            if (m.source_seat >= 0) players_.push_back(m.source_seat);
            if (m.target_seat >= 0) players_.push_back(m.target_seat);
            return;
        }
    }
}

}  // namespace rmwork
