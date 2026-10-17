#ifndef RMKNOW_GENERATIVE_H
#define RMKNOW_GENERATIVE_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

// ── The belief layer's random model played out directly (docs/knowledge_store.md §10.6) ───────
//
// An exactness check of src/knowledge/belief.h that shares nothing with the network: one player's
// game under one reference list, as an exact probability distribution over the true states (the
// library's order, the hand cards the viewer does not know, the sideboard and the set-aside cards,
// by card) that every observation reweighs. The library starts as a uniformly random order of the
// list's main deck and the sideboard as its sideboard, each part's cards beyond the sizes the game
// starts with set aside (a uniformly random subset); a shuffle draws a new uniform order. The
// player's hidden choices follow the belief layer's rules: a card shown leaving the hand that the
// viewer knew to be there (revealed, or seen on top of the library before it was drawn) is that card,
// which says nothing more; otherwise it is a uniformly random card of the hand's unknown part; an
// unseen put-back is a uniformly random card of the hand (known cards included, which the viewer then
// no longer knows apart); a scry's bottomed card one of the top two, a wish's card a sideboard card
// and a card from outside the game a set-aside card, each uniformly. An observation keeps the states
// it allows, scaled by the probability of the choice it shows. probability() is then P(observations |
// list), the belief layer's likelihood up to a factor that is the same for every list (a cast's
// candidates leave out hand cards that cannot be the card shown, in every list alike).

namespace rmknow_test {

class GenerativeModel {
    public:
        // The game's start: `library` and `sideboard` cards (the sizes the game started with) from
        // the list's main deck and sideboard; at most `cap` states at a time.
        GenerativeModel(const std::vector<std::pair<int, int>> &main, const std::vector<std::pair<int, int>> &side,
            int library, int sideboard, uint64_t cap);
        void draw();                 // the top card into the hand
        void cast_unseen(int card);  // a hand card the viewer cannot tell apart from the others is `card`
        void cast_known(int card);   // a hand card the viewer knows (by handle) leaves
        void reveal_top(int card);   // the top card is `card`
        void reveal_hand(const std::vector<int> &cards);  // the hand is exactly these cards
        void put_back_unseen(bool bottom);                // an unseen hand card onto the top (or bottom) of the library
        void put_back_known(int card, bool bottom);
        void scry_bottom_one();   // one of the top two cards to the bottom
        void mill_top(int card);  // the top card is `card`; it leaves
        void shuffle();
        void wish(int card);            // a sideboard card is `card`; into the hand, known
        void from_set_aside(int card);  // a set-aside card is `card`; into the hand, known
        double probability() const;
        bool over() const { return over_; }

    private:
        struct State {
                std::string library;  // top first, one byte per card
                std::string hand;     // sorted: the hand cards the viewer does not know
                std::string side;     // sorted
                std::string aside;    // sorted
                bool operator<(const State &o) const;
        };
        using Dist = std::map<State, double>;
        void add(Dist &d, const State &s, double p);
        void check_cap(const Dist &d);
        static std::string sorted_insert(const std::string &s, char c);
        static bool remove_one(std::string &s, char c);
        static int count_of(const std::string &s, char c);
        static char code(int card);

        Dist dist_;
        std::string known_;            // hand cards the viewer knows by handle (the same in every state)
        std::string pinned_;           // hand cards the viewer knows without a handle (sorted; the same in every state)
        std::vector<bool> lib_known_;  // per library position (top first): the viewer knows its card
        uint64_t cap_;
        bool over_ = false;
};

}  // namespace rmknow_test

#endif /* RMKNOW_GENERATIVE_H */
