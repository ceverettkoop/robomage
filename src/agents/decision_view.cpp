#include "decision_view.h"

#include <algorithm>

namespace agents {

DecisionView::DecisionView(const SeatStream &stream, const rmstate::DecodedFrame &frame, int seat)
    : stream_(stream),
      frame_(frame),
      seat_(seat),
      menu_(frame, stream.schema(), "menu"),
      object_(frame, stream.schema(), "object"),
      stack_(frame, stream.schema(), "stack") {
    m_category_ = menu_.col("category");
    m_source_ = menu_.col("source");
    m_source_card_ = menu_.col("source_card");
    m_source_player_rel_ = menu_.col("source_player_rel");
    m_variant_ = menu_.col("variant");
    m_description_ = menu_.col("description");
    o_handle_ = object_.col("handle");
    o_zone_ = object_.col("zone");
    o_owner_rel_ = object_.col("owner_rel");
    o_controller_rel_ = object_.col("controller_rel");
    o_card_ = object_.col("card");
    o_token_ = object_.col("token");
    o_transformed_ = object_.col("transformed");
    o_copy_of_ = object_.col("copy_of");
    s_handle_ = stack_.col("handle");
    s_pos_ = stack_.col("pos");
    s_kind_ = stack_.col("kind");
    s_source_ = stack_.col("source");
    s_source_card_ = stack_.col("source_card");
    s_ab_card_ = stack_.col("ab_card");
    s_ab_key_ = stack_.col("ab_key");
    s_variant_ = stack_.col("variant");
    z_battlefield_ = stream.enum_value("zone", "BATTLEFIELD");
    z_stack_ = stream.enum_value("zone", "STACK");
    z_hand_ = stream.enum_value("zone", "HAND");
    z_graveyard_ = stream.enum_value("zone", "GRAVEYARD");
    z_exile_ = stream.enum_value("zone", "EXILE");
    k_triggered_ = stream.enum_value("stack_kind", "TRIGGERED");
    const int back = stream.enum_value("cast_variant", "BACK_FACE");
    back_face_bit_ = back >= 0 ? (1ull << back) : 0;
    for (size_t r = 0; r < object_.rows(); ++r) objects_.emplace(object_.u(r, o_handle_), r);
    stack_by_pos_.resize(stack_.rows());
    for (size_t r = 0; r < stack_.rows(); ++r) stack_by_pos_[r] = r;
    std::stable_sort(stack_by_pos_.begin(), stack_by_pos_.end(),
        [this](size_t a, size_t b) { return stack_.u(a, s_pos_) < stack_.u(b, s_pos_); });
}

const std::string &DecisionView::category(size_t i) const {
    return stream_.enum_name("action_category", menu_.u(i, m_category_));
}

const std::string &DecisionView::description(size_t i) const {
    return menu_.text(i, m_description_);
}

int DecisionView::object_row(uint64_t h) const {
    if (h == 0) return -1;
    auto it = objects_.find(h);
    return it != objects_.end() ? static_cast<int>(it->second) : -1;
}

int DecisionView::stack_row(uint64_t h) const {
    for (size_t r : stack_by_pos_)
        if (stack_.u(r, s_handle_) == h) return static_cast<int>(r);
    return -1;
}

std::optional<std::string> DecisionView::object_name(size_t row) const {
    int cid = object_.card(row, o_card_);
    if (cid < 0) return std::nullopt;
    if (object_.u(row, o_token_)) return cards().name(cid);
    const int copy = object_.card(row, o_copy_of_);
    if (copy >= 0) cid = copy;
    return cards().name(cid, object_.u(row, o_transformed_) != 0);
}

std::optional<std::string> DecisionView::stack_card(size_t row) const {
    if (static_cast<int>(stack_.u(row, s_kind_)) == k_triggered_ && stack_.u(row, s_source_) == 0 &&
        stack_.card(row, s_ab_card_) < 0 && stack_.text(row, s_ab_key_).empty())
        return std::nullopt;
    const int self = object_row(stack_.u(row, s_handle_));
    if (self >= 0 && object_.card(static_cast<size_t>(self), o_card_) >= 0)
        return object_name(static_cast<size_t>(self));
    const int src = object_row(stack_.u(row, s_source_));
    if (src >= 0 && static_cast<int>(object_.u(static_cast<size_t>(src), o_zone_)) == z_battlefield_)
        return object_name(static_cast<size_t>(src));
    const int cid = stack_.card(row, s_source_card_);
    if (cid < 0) return std::nullopt;
    return cards().name(cid);
}

std::optional<std::string> DecisionView::card(size_t i) const {
    const int o = object_row(menu_.u(i, m_source_));
    if (o >= 0) {
        const size_t row = static_cast<size_t>(o);
        const int zone = static_cast<int>(object_.u(row, o_zone_));
        if (zone == z_stack_) {
            const int s = stack_row(object_.u(row, o_handle_));
            if (s >= 0) return stack_card(static_cast<size_t>(s));
        }
        if (zone == z_battlefield_ || object_.card(row, o_card_) < 0) return object_name(row);
    } else if (menu_.u(i, m_source_player_rel_) != REL_NONE) {
        return std::nullopt;
    }
    const int cid = menu_.card(i, m_source_card_);
    if (cid < 0) return std::nullopt;
    return cards().name(cid, (menu_.u(i, m_variant_) & back_face_bit_) != 0);
}

int DecisionView::object_face_id(size_t row) const {
    int cid = object_.card(row, o_card_);
    if (cid < 0 || object_.u(row, o_token_)) return cid;
    const int copy = object_.card(row, o_copy_of_);
    if (copy >= 0) cid = copy;
    return object_.u(row, o_transformed_) ? cards().back_face(cid) : cid;
}

int DecisionView::stack_face_id(size_t row) const {
    const int cid = stack_.card(row, s_source_card_);
    if (static_cast<int>(stack_.u(row, s_kind_)) == k_triggered_ && stack_.u(row, s_source_) == 0 &&
        stack_.card(row, s_ab_card_) < 0 && stack_.text(row, s_ab_key_).empty())
        return -1;
    const int src = object_row(stack_.u(row, s_source_));
    if (src >= 0 && static_cast<int>(object_.u(static_cast<size_t>(src), o_zone_)) == z_battlefield_)
        return object_face_id(static_cast<size_t>(src));
    if ((stack_.u(row, s_variant_) & back_face_bit_) && cid >= 0) return cards().back_face(cid);
    return cid;
}

int DecisionView::bquery_card_id(size_t i) const {
    const int o = object_row(menu_.u(i, m_source_));
    int cid = menu_.card(i, m_source_card_);
    int zone = -1;
    if (o >= 0) {
        const size_t row = static_cast<size_t>(o);
        zone = static_cast<int>(object_.u(row, o_zone_));
        if (zone == z_battlefield_) {
            cid = object_face_id(row);
        } else if (zone == z_stack_) {
            const int s = stack_row(object_.u(row, o_handle_));
            cid = s >= 0 ? stack_face_id(static_cast<size_t>(s)) : object_.card(row, o_card_);
        }
    } else if (menu_.u(i, m_source_player_rel_) != REL_NONE) {
        cid = -1;
    }
    if ((menu_.u(i, m_variant_) & back_face_bit_) && cid >= 0 && zone != z_battlefield_) cid = cards().back_face(cid);
    return cid;
}

int DecisionView::bquery_controller(size_t i) const {
    const int o = object_row(menu_.u(i, m_source_));
    if (o >= 0) {
        const size_t row = static_cast<size_t>(o);
        const int zone = static_cast<int>(object_.u(row, o_zone_));
        if (zone != z_battlefield_ && zone != z_hand_ && zone != z_stack_ && zone != z_graveyard_ && zone != z_exile_)
            return -1;
        return object_.u(row, o_controller_rel_) == REL_SELF ? 1 : 0;
    }
    const uint64_t rel = menu_.u(i, m_source_player_rel_);
    if (rel == REL_NONE) return -1;
    return rel == REL_SELF ? 1 : 0;
}

std::vector<std::string> DecisionView::hand() const {
    std::vector<std::string> out;
    for (size_t r = 0; r < object_.rows(); ++r) {
        if (static_cast<int>(object_.u(r, o_zone_)) != z_hand_ || object_.u(r, o_owner_rel_) != REL_SELF) continue;
        std::optional<std::string> name = object_name(r);
        if (name) out.push_back(std::move(*name));
    }
    return out;
}

}  // namespace agents
