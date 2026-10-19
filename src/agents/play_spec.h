#ifndef AGENTS_PLAY_SPEC_H
#define AGENTS_PLAY_SPEC_H

#include <optional>
#include <string>
#include <vector>

#include "agent.h"

// ── The --play resolver: semantic action specs ───────────────────────────────────────────────
//
// A spec names an action by intent ("cast:Lightning Bolt", "target:Grizzly Bears@opp", "pass");
// it is resolved against the live menu of the decision it is applied to, so it survives menu
// reordering, and fails loudly (with the menu) when it matches no row or several distinct ones.
// The grammar, case-insensitive, one spec = one action:
//
//   pass                         pass priority / take the first choice    (PASS_PRIORITY)
//   cast:<card>                  cast a spell                              (CAST_SPELL)
//   play:<card> | land:<card>    play a land                               (PLAY_LAND)
//   activate:<card>[@side]       activate an ability                       (ACTIVATE_ABILITY)
//   target:<card|text>[@side]    choose a target                           (SELECT_TARGET)
//   attack:<card> | attack:done  declare an attacker / confirm             (SELECT_/CONFIRM_ATTACKERS)
//   block:<card> | block:done    declare a blocker / confirm               (SELECT_/CONFIRM_BLOCKERS)
//   mana:<w|u|b|r|g|c> | tap:<land>   tap for mana                         (MANA_*; tap: also the
//                                     payment loop's sources, PAYING_COSTS)
//   search:<card> | search:fail  library / exile / sideboard search        (SEARCH_LIBRARY, CHOOSE_CARD)
//   top:<card>  bottom:<card>  dig:<card>                                  (TOP_LIBRARY, BOTTOM_DECK_CARD, DIG_CHOICE)
//   mulligan | keep              the mulligan decision                     (MULLIGAN, KEEP_HAND)
//   pay:<text>                   pay an optional cost                      (PAYING_COSTS)
//   choice:<text>                any dedicated choice category or OTHER_CHOICE
//   shuffle:<text>  noshuffle:<text>                                       (SHUFFLE, DONT_SHUFFLE)
//   sb-in:<card>  sb-out:<card>  sb-done                                   sideboarding (bo3)
//   companion:<card>             companion to hand                         (COMPANION)
//   sacrifice: return: x: discard: mode: color: name: exile:  <text>        the matching choice category
//   desc:<text>                  any action, by description substring
//   concede | concede:game | concede:match   the sentinels -2 / -3 (not a menu row)
//   #<n> | <n>                   a literal menu index
//
// `<card>` matches an action's card name (case-, punctuation- and apostrophe-insensitively: an exact
// name first, else a unique substring); a card-less action matches by description substring.
// `@side` is own / opp (aliases me, my, self, you, mine / them, their, enemy, opponent): the
// action's controller. Matches that differ only by row (same category, card and controller) are
// interchangeable: the first is taken. A leading seat key `A:` / `B:` pins a spec to a seat
// (PlayScript); resolution itself ignores it.
//
// The menu as the resolver reads it (MenuAction): the card is the name of the action's card id
// (DecisionView::menu_card_id; accents folded), the controller DecisionView::menu_controller's,
// the description the menu row's.

namespace agents {

struct MenuAction {
        size_t index = 0;
        std::string category;
        std::optional<std::string> card;
        int controller = -1;  // 1 own, 0 opp, -1 none
        std::string description;
};

// The menu of `v` as the resolver reads it.
std::vector<MenuAction> resolver_menu(const DecisionView &v);

enum class ResolveKind {
    OK,
    PARSE_ERROR,
    BAD_INDEX,
    AMBIGUOUS,
    NO_MATCH
};

struct ResolveResult {
        int index = -1;  // a menu row, or a concession sentinel
        ResolveKind kind = ResolveKind::NO_MATCH;
        std::string reason;
        bool ok() const { return kind == ResolveKind::OK; }
};

// Resolves one spec against `actions`.
ResolveResult resolve_spec(const std::string &token, const std::vector<MenuAction> &actions);
// The seat a spec is keyed to ('A' / 'B'), or 0.
char spec_seat(const std::string &token);
// Whether the spec's verb is `pass` (false for a spec that does not parse).
bool spec_is_pass(const std::string &token);
// A comma-separated spec list, each spec trimmed, empty ones dropped.
std::vector<std::string> parse_spec_list(const std::string &csv);
// "[0] Pass priority  [1] Cast Lightning Bolt ...": the menu on one line, for error messages.
std::string format_menu(const std::vector<MenuAction> &actions);

// Plays a fixed list of specs (opponents.PlayController ported). The list is global: one spec is
// consumed per decision of whichever seat it is given; once it has run out the deciding seat's
// player decides (`player_a` / `player_b`, which must outlive it). A spec keyed to the seat that
// is not deciding makes the decider pass (the spec waits) until the keyed seat is on the clock; a
// decider with no pass on its menu faces a mandatory choice, which its player makes. A keyed
// spec that matches nothing at a decision of its own seat passes that seat forward through its
// priority windows (up to MAX_WAIT passes) until the action is offered. Any other failure ends
// the run (AGENT_FAILED; failure() gives the reason and the menu). Every reply it gives (the players' included) is
// recorded (resolved()), so a run replays as the plain integer list.
class PlayScript : public Agent {
    public:
        static constexpr int MAX_WAIT = 200;
        PlayScript(std::vector<std::string> specs, Agent *player_a, Agent *player_b);
        int decide(const Decision &d) override;
        std::string label() const override { return "play"; }
        // The behaviour of the seat's agent when it decided the last decision, else nullptr.
        const Behaviour *last_behaviour() const override { return delegate_ ? delegate_->last_behaviour() : nullptr; }
        const std::vector<int> &resolved() const { return resolved_; }
        // The specs not yet applied.
        std::vector<std::string> unused_specs() const;

    private:
        int player_choice(const Decision &d);
        std::vector<std::string> specs_;
        size_t next_ = 0;
        int wait_ = 0;
        Agent *players_[2];
        const Agent *delegate_ = nullptr;  // the seat's agent that decided the last decision
        std::vector<int> resolved_;
};

}  // namespace agents

#endif /* AGENTS_PLAY_SPEC_H */
