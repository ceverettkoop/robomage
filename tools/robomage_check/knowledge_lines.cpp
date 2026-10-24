#include "knowledge_lines.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <tuple>

#include "frames.h"

using rmcheck::Rows;

// One seat's card names as its CARDS frames gave them so far.
struct SeatNames {
        std::map<int64_t, std::string> cards;
};

static std::string card_name(const SeatNames &n, const Rows &t, size_t r, int c);
static std::string card_of(const SeatNames &n, int64_t cid);
static std::string rel_text(uint64_t rel);
static std::string num_or_none(const Rows &t, size_t r, int c);
static std::string libpos(const Rows &t, size_t r, int c_anchor, int c_pos, int c_span, bool *present);
static void facts(
    const rmstate::DecodedFrame &f, const rmstate::Schema &s, const SeatNames &n, std::vector<std::string> *out);
static void stack_target_facts(
    const rmstate::DecodedFrame &f, const rmstate::Schema &s, const SeatNames &n, std::vector<std::string> *out);

static std::string card_of(const SeatNames &n, int64_t cid) {
    auto it = n.cards.find(cid);
    return it == n.cards.end() ? "#" + std::to_string(cid) : it->second;
}

// A card column's name ("#<id>" for an id CARDS does not list).
static std::string card_name(const SeatNames &n, const Rows &t, size_t r, int c) {
    return card_of(n, static_cast<int64_t>(t.u(r, c)));
}

static std::string rel_text(uint64_t rel) {
    return rel == 1 ? "self" : "opp";
}

static std::string num_or_none(const Rows &t, size_t r, int c) {
    return t.null(r, c) ? "None" : std::to_string(t.u(r, c));
}

// "top<pos>" / "bottom<pos>" with "+<span>" for a span above 1; `present` false for no anchor.
static std::string libpos(const Rows &t, size_t r, int c_anchor, int c_pos, int c_span, bool *present) {
    *present = !t.null(r, c_anchor);
    if (!*present) return "";
    const std::string end = t.name(r, c_anchor) == "TOP" ? "top" : "bottom";
    const uint64_t span = t.u(r, c_span);
    return end + num_or_none(t, r, c_pos) + (span > 1 ? "+" + std::to_string(span) : "");
}

static void facts(
    const rmstate::DecodedFrame &f, const rmstate::Schema &s, const SeatNames &n, std::vector<std::string> *out) {
    const Rows meta(f, s, "k_meta");
    for (size_t r = 0; r < meta.size(); ++r)
        out->push_back("meta level=" + meta.name(r, meta.col("level")) +
                       " exact=" + std::to_string(meta.u(r, meta.col("exact")) ? 1 : 0));
    const Rows seen(f, s, "k_seen");
    for (size_t r = 0; r < seen.size(); ++r)
        out->push_back("seen " + rel_text(seen.u(r, seen.col("player_rel"))) + " " +
                       card_name(n, seen, r, seen.col("card")) +
                       " game=" + std::to_string(seen.u(r, seen.col("game"))) +
                       " match=" + std::to_string(seen.u(r, seen.col("match"))));
    // pool row index -> (name, min, max) of its cards
    std::map<int64_t, std::vector<std::tuple<std::string, uint64_t, uint64_t>>> pool_cards;
    const Rows pc(f, s, "k_pool_card");
    for (size_t r = 0; r < pc.size(); ++r)
        pool_cards[pc.null(r, pc.col("pool")) ? -1 : static_cast<int64_t>(pc.u(r, pc.col("pool")))].emplace_back(
            card_name(n, pc, r, pc.col("card")), pc.u(r, pc.col("min")), pc.u(r, pc.col("max")));
    for (auto &p : pool_cards) std::sort(p.second.begin(), p.second.end());
    const Rows pools(f, s, "k_pool");
    for (size_t r = 0; r < pools.size(); ++r) {
        bool present = false;
        const std::string rng =
            libpos(pools, r, pools.col("range_anchor"), pools.col("range_pos"), pools.col("range_span"), &present);
        std::string kind = pools.name(r, pools.col("kind"));
        if (kind == "LIBRARY_RANGE") {
            kind = "library";
        } else {
            for (char &c : kind) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        std::string cards;
        for (const auto &c : pool_cards[static_cast<int64_t>(r)])
            cards += (cards.empty() ? "" : ", ") + std::get<0>(c) + " " + std::to_string(std::get<1>(c)) + "-" +
                     std::to_string(std::get<2>(c));
        out->push_back("pool " + rel_text(pools.u(r, pools.col("owner_rel"))) + " " + kind +
                       (present && !rng.empty() ? " " + rng : "") +
                       " size=" + std::to_string(pools.u(r, pools.col("size"))) + ": " + cards);
    }
    const Rows pos(f, s, "k_position");
    for (size_t r = 0; r < pos.size(); ++r) {
        bool present = false;
        const std::string at = libpos(pos, r, pos.col("at_anchor"), pos.col("at_pos"), pos.col("at_span"), &present);
        const std::string head = "position " + rel_text(pos.u(r, pos.col("owner_rel"))) + " " + (present ? at : "None");
        if (!pos.null(r, pos.col("card"))) {
            out->push_back(head + " = " + card_name(n, pos, r, pos.col("card")));
            continue;
        }
        std::vector<std::string> cands;
        if (!pos.null(r, pos.col("pool"))) {
            auto it = pool_cards.find(static_cast<int64_t>(pos.u(r, pos.col("pool"))));
            if (it != pool_cards.end())
                for (const auto &c : it->second)
                    if (std::get<2>(c) > 0) cands.push_back(std::get<0>(c));
        }
        std::sort(cands.begin(), cands.end());
        std::string list;
        for (const std::string &c : cands) list += (list.empty() ? "" : ", ") + c;
        out->push_back(head + " one of: " + list);
    }
    const Rows kh(f, s, "k_known_hand");
    for (size_t r = 0; r < kh.size(); ++r)
        out->push_back(
            "known_hand " + card_name(n, kh, r, kh.col("card")) + " x" + std::to_string(kh.u(r, kh.col("copies"))));
    stack_target_facts(f, s, n, out);
}

static void stack_target_facts(
    const rmstate::DecodedFrame &f, const rmstate::Schema &s, const SeatNames &n, std::vector<std::string> *out) {
    const Rows objects(f, s, "object"), lki(f, s, "lki"), stack(f, s, "stack"), st(f, s, "stack_target");
    std::map<uint64_t, size_t> object_row;
    for (size_t r = 0; r < objects.size(); ++r) object_row.emplace(objects.u(r, objects.col("handle")), r);
    std::map<uint64_t, size_t> stack_row;
    for (size_t r = 0; r < stack.size(); ++r) stack_row[stack.u(r, stack.col("handle"))] = r;
    for (size_t r = 0; r < st.size(); ++r) {
        std::string pos = "None", src = "?";
        auto sr = stack_row.find(st.u(r, st.col("stack")));
        if (sr != stack_row.end()) {
            pos = num_or_none(stack, sr->second, stack.col("pos"));
            if (!stack.null(sr->second, stack.col("source_card")))
                src = card_name(n, stack, sr->second, stack.col("source_card"));
        }
        std::string what;
        const int c_lki = st.col("lki");
        if (st.name(r, st.col("kind")) == "PLAYER") {
            what = std::string("player ") + (st.u(r, st.col("target_player_rel")) == 1 ? "self" : "opp");
        } else if (st.u(r, st.col("target"))) {
            auto o = object_row.find(st.u(r, st.col("target")));
            what = o != object_row.end() && !objects.null(o->second, objects.col("card"))
                       ? card_name(n, objects, o->second, objects.col("card"))
                       : "?";
        } else if (!st.null(r, c_lki) && st.u(r, c_lki) < lki.size()) {
            const size_t lr = static_cast<size_t>(st.u(r, c_lki));
            what = "departed " +
                   (lki.null(lr, lki.col("card")) ? std::string("?") : card_name(n, lki, lr, lki.col("card")));
        } else {
            what = "none";
        }
        out->push_back(
            "stack_target " + pos + " " + src + " " + std::to_string(st.u(r, st.col("order"))) + " = " + what);
    }
}

namespace rmcheck {

bool knowledge_frames(const std::vector<std::string> &payloads,
    std::vector<std::pair<char, std::vector<std::string>>> *frames, std::string *err) {
    FrameDecoder dec;
    SeatNames seats[2];
    for (const std::string &p : payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, err)) return false;
        const int viewer = frame_viewer(p), kind = frame_kind(p);
        if (viewer != VIEWER_A && viewer != VIEWER_B) continue;
        const rmstate::Schema &s = *dec.schema(viewer);
        SeatNames &n = seats[viewer];
        if (kind == KIND_CARDS) {
            const Rows card(f, s, "card");
            for (size_t r = 0; r < card.size(); ++r)
                n.cards.emplace(card.null(r, card.col("card")) ? -1 : static_cast<int64_t>(card.u(r, card.col("card"))),
                    card.text(r, card.col("name")));
        } else if (kind == KIND_STATE) {
            const Rows turn(f, s, "turn");
            const std::string when = header_value(f, s, "sideboarding")
                                         ? "sb"
                                         : "t" + std::to_string(turn.size() ? turn.u(0, turn.col("turn")) : 0);
            const std::string prefix = std::string(1, viewer == VIEWER_A ? 'A' : 'B') +
                                       " d=" + std::to_string(header_value(f, s, "decision")) + " g" +
                                       std::to_string(header_value(f, s, "game")) + " " + when + " ";
            std::vector<std::string> lines;
            facts(f, s, n, &lines);
            for (std::string &l : lines) l = prefix + l;
            frames->emplace_back(viewer == VIEWER_A ? 'A' : 'B', std::move(lines));
        }
    }
    return true;
}

}  // namespace rmcheck
