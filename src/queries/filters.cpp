#include "filters.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/object_spec.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../coverage.h"
#include "../ecs/coordinator.h"
#include "../object_ref.h"
#include "../str_util.h"
#include "../svar_eval.h"
#include "../type_constants.h"
#include "affected.h"
#include "battlefield.h"
#include "characteristics.h"
#include "combat.h"
#include "counters.h"
#include "keywords.h"
#include "lki.h"
#include "spells.h"
#include "types.h"
#include "zones.h"

// ── Unified filter matcher (declared in filters.h) ─────────────────────
// A flattened view of the one object under test, populated either from a card's printed
// characteristics or a battlefield permanent's live components, then handed to one shared
// qualifier evaluator. This is the single place the "live for types & P/T, copiable-from-
// card for color & mana value" rule (CR 105/112.7/205/208) is encoded.
namespace {

struct CharView {
    Entity entity = 0;                       // 0 when matching a bare CardData (no identity quals)
    const std::set<Type> *types = nullptr;   // live (permanent) or printed (card) type line
    std::set<Colors> colors;                 // effective (permanent) or printed (card)
    int cmc = 0;                             // mana value (always from the card; CR 112.7)
    bool has_pt = false;                     // object has power/toughness
    int power = 0, toughness = 0;
    bool counted_pt = false;                 // read them through view_pt
    bool on_battlefield = false;
    bool is_token = false;
    Zone::Ownership controller = Zone::UNKNOWN;
    Zone::Ownership owner = Zone::UNKNOWN;     // card's owner (YouOwn/OppOwn), from its Zone
    long entered_on_turn = -1;               // -1 when not on the battlefield
    bool is_attacking = false;               // live combat state (battlefield creatures only)
    bool is_blocking = false;
    bool is_unblocked = false;               // an unblocked attacker (CR 509.1h)
    bool is_tapped = false;
    bool has_x_cost = false;                 // printed mana cost contains {X} (Gaddock Teeg's hasXCost)
    bool creature_suppressed = false;        // on battlefield, has "Creature" in its type line but
                                             // is NOT currently a creature (no live Creature component)
    bool entered_by_cast = false;            // a permanent that entered by being cast (wasCastByYou)
};

struct Qualifier;

// The evaluator shapes. Object-local reads the object alone; set-relative reads it against the
// context; population-relative narrows `candidates` (filters.h, select_matching).
using LocalFn = bool (*)(const CharView &v, const Qualifier &q);
using RelativeFn = bool (*)(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
using PopulationFn = void (*)(const Qualifier &q, const MatchCtx &ctx, std::vector<Entity> &candidates);
// Reads a token its row's name matched into `q`. False: it is not this row's form, and the rows
// after it are tried.
using ParseFn = bool (*)(const std::string &token, Qualifier &q);

enum class NameForm : uint8_t {
    EXACT,            // the whole token
    PREFIX,           // the token starts with the name
    PREFIX_ANY_CASE,  // the same, ASCII case-insensitively
    ANY,              // every token; only the parser decides
};

// One qualifier: what a script spells, how it reads what the spelling carries, and how it evaluates.
struct QualifierRow {
    const char *name;
    NameForm form;
    ParseFn parse;          // null: the name is all there is to read
    LocalFn local;          // exactly one of local / relative runs on a single object
    RelativeFn relative;
    PopulationFn population;
    ObjectSpec::Kind set;   // what a set-relative evaluator resolves
};

// One token of a filter, compiled.
struct Qualifier {
    const QualifierRow *row = nullptr;
    std::string token;           // as the script spells it
    std::string name;            // without its '!'s
    bool negate = false;         // "!<qualifier>" negates it (Doomsday's Card.!IsRemembered)
    SvarOp op = SvarOp::UNKNOWN; // a comparator's operator
    std::string op_text;         // and its spelling, to report one the engine can't read
    int number = 0;              // a comparator's literal
    bool of_power = false;       // a P/T comparator reads power, else toughness
    std::string text;            // the row's name argument: a type, keyword or counter; a cmc
                                 // qualifier's bound
    Colors color = NO_COLOR;
    Zone::ZoneValue zone = Zone::SIDEBOARD;  // inZone<Zone>; `known_zone` says it names one
    bool known_zone = false;
    bool has_reading = true;     // false: the matcher fails it closed, negated or not (--parse-report's
                                 // finding); a type-line name no type has has none
    bool read_elsewhere = false; // its filter's reader reads it, whatever the matcher makes of it
    bool reads_type_name = false;  // `text` is a type-line name it requires or excludes
    ObjectSpec set;              // the row's set
};

// One OR alternative: its head, then its ANDed qualifiers, a head's own check first.
struct Alternative {
    std::string head;
    std::vector<Qualifier> qualifiers;
    size_t head_qualifiers = 0;  // how many of `qualifiers` are its head's own check
};

}  // namespace

// The compiled form of a filter spec, behind FilterSpec. Holds nothing bound to a game: what one
// evaluation is bound to is in its MatchCtx.
struct CompiledFilter {
    std::string text;
    std::vector<Alternative> alternatives;
    // owner_relative_filter's and target_filter's results, once asked.
    mutable FilterSpec owner_relative;
    mutable bool owner_relative_read = false;
    mutable FilterSpec target_form;
    mutable bool target_form_read = false;
};

static std::unordered_map<std::string, std::unique_ptr<CompiledFilter>> &compiled_filters();
static const CompiledFilter *compile_filter(const std::string &text);
static Alternative compile_alternative(const std::string &alt);
static Qualifier compile_qualifier(const std::string &token);
static void add_head_qualifier(Alternative &alt);
static bool row_names(const QualifierRow &row, const std::string &token);
static std::vector<std::pair<std::string, bool>> filter_tokens(const std::string &spec);
static std::string owner_relative_text(const std::string &spec);
static bool known_type_name(const std::string &name);
static bool all_digits(const std::string &s, size_t from);
static bool object_view(Entity e, CharView &v);
static bool matches_view(const CharView &v, const CompiledFilter *filter, const MatchCtx &ctx);
static bool alternative_matches(const Alternative &alt, const CharView &v, const MatchCtx &ctx);
static bool eval_qualifier(const Qualifier &q, const CharView &v, const MatchCtx &ctx);

static bool object_has_name(Entity e, const std::string &name);
static bool view_has_typeline(const CharView &v, const std::string &name);
static bool view_is_permanent(const CharView &v);
static bool color_token(const std::string &q, Colors &c);
static bool view_has_any_color(const CharView &v);
static bool compare_qualifier(int lhs, const Qualifier &q, int rhs);
static void warn_unknown_qualifier(const std::string &q);
static ResolveCtx set_context(const MatchCtx &ctx);
static bool object_in_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);

// `cd` as printed; `e`, when it names the card, gives its identity and owner.
static CharView printed_view(Entity e, const CardData &cd);
// The card as a filter reads it: printed, but for the color a layer-5 static sets in any zone.
static CharView card_view(Entity e, const CardData &cd);
static void read_characteristics(CharView &v, ObjectCharacteristics &&c);
static CharView card_identity_view(Entity e, const CardData &cd);
static void set_permanent_mana_value(CharView &v, Entity e, bool transformed);
static CharView permanent_view(Entity e, const Permanent &perm);
static CharView lki_view(Entity e, const LastKnownInfo &lki);

static bool parse_cmc_literal(const std::string &token, Qualifier &q);
static bool parse_cmc_named(const std::string &token, Qualifier &q);
static bool parse_pt_vs_x(const std::string &token, Qualifier &q);
static bool parse_pt_literal(const std::string &token, Qualifier &q);
static bool parse_counters(const std::string &token, Qualifier &q);
static bool parse_non_color(const std::string &token, Qualifier &q);
static bool parse_non_colorless(const std::string &token, Qualifier &q);
static bool parse_non_card_type(const std::string &token, Qualifier &q);
static bool parse_non_subtype(const std::string &token, Qualifier &q);
static bool parse_keyword_after(const std::string &token, size_t from, Qualifier &q);
static bool parse_with_keyword(const std::string &token, Qualifier &q);
static bool parse_without_keyword(const std::string &token, Qualifier &q);
static bool parse_in_zone(const std::string &token, Qualifier &q);
static bool parse_attached_to(const std::string &token, Qualifier &q);
static bool parse_mana_cost(const std::string &token, Qualifier &q);
static bool parse_color(const std::string &token, Qualifier &q);
static bool parse_read_elsewhere(const std::string &token, Qualifier &q);
static bool parse_type_line_name(const std::string &token, Qualifier &q);
static bool parse_unknown(const std::string &token, Qualifier &q);

static bool ev_always(const CharView &v, const Qualifier &q);
static bool ev_fails_closed(const CharView &v, const Qualifier &q);

static bool ev_named_card(const CharView &v, const Qualifier &q);
static bool ev_other(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_self(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_exiled_with_source(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_top_graveyard_creature(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_not_in_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_attached_to(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_equipped_by(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_you_ctrl(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_opp_ctrl(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_controlled_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_controlled_or_owned_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_owned_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_lowest_cmc_of_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_shares_color_with_other(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_you_own(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_opp_own(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_token(const CharView &v, const Qualifier &q);
static bool ev_non_token(const CharView &v, const Qualifier &q);
static bool ev_entered_this_turn(const CharView &v, const Qualifier &q);
static bool ev_least_power(const CharView &v, const Qualifier &q);
static bool ev_was_cast_by_you(const CharView &v, const Qualifier &q);
static bool ev_cast_from_your_hand(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_attacking(const CharView &v, const Qualifier &q);
static bool ev_blocking(const CharView &v, const Qualifier &q);
static bool ev_unblocked(const CharView &v, const Qualifier &q);
static bool ev_tapped(const CharView &v, const Qualifier &q);
static bool ev_untapped(const CharView &v, const Qualifier &q);
static bool ev_basic(const CharView &v, const Qualifier &q);
static bool ev_non_basic(const CharView &v, const Qualifier &q);
static bool ev_colorless(const CharView &v, const Qualifier &q);
static bool ev_has_x_cost(const CharView &v, const Qualifier &q);
static bool ev_has_basic_land_type(const CharView &v, const Qualifier &q);
static bool ev_cmc_le_x(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_cmc_literal(const CharView &v, const Qualifier &q);
static bool ev_cmc_bound(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_pt_vs_x(const CharView &v, const Qualifier &q, const MatchCtx &ctx);
static bool ev_pt_literal(const CharView &v, const Qualifier &q);
static bool ev_counters(const CharView &v, const Qualifier &q);
static bool ev_color(const CharView &v, const Qualifier &q);
static bool ev_non_color(const CharView &v, const Qualifier &q);
static bool ev_has_a_color(const CharView &v, const Qualifier &q);
static bool ev_not_type_line(const CharView &v, const Qualifier &q);
static bool ev_has_keyword(const CharView &v, const Qualifier &q);
static bool ev_lacks_keyword(const CharView &v, const Qualifier &q);
static bool ev_in_zone(const CharView &v, const Qualifier &q);
static bool ev_mana_cost(const CharView &v, const Qualifier &q);
static bool ev_type_line(const CharView &v, const Qualifier &q);
static bool ev_permanent(const CharView &v, const Qualifier &q);
static bool ev_unknown(const CharView &v, const Qualifier &q);

// A row evaluated on one object, by the object alone or against the context.
static constexpr QualifierRow local_row(const char *name, LocalFn fn) {
    return {name, NameForm::EXACT, nullptr, fn, nullptr, nullptr, ObjectSpec::DEFAULT};
}
static constexpr QualifierRow relative_row(const char *name, RelativeFn fn) {
    return {name, NameForm::EXACT, nullptr, nullptr, fn, nullptr, ObjectSpec::DEFAULT};
}
// A row reading its ObjectSpec's set.
static constexpr QualifierRow set_row(const char *name, RelativeFn fn, ObjectSpec::Kind set) {
    return {name, NameForm::EXACT, nullptr, nullptr, fn, nullptr, set};
}
// A row whose spelling carries an argument, which `parse` reads.
static constexpr QualifierRow parsed_row(const char *name, NameForm form, ParseFn parse, LocalFn fn) {
    return {name, form, parse, fn, nullptr, nullptr, ObjectSpec::DEFAULT};
}
static constexpr QualifierRow parsed_relative_row(const char *name, NameForm form, ParseFn parse, RelativeFn fn) {
    return {name, form, parse, nullptr, fn, nullptr, ObjectSpec::DEFAULT};
}
// A token the matcher has no reading for, which the code a filter reaches reads itself. It evaluates
// as the catch-all rows make it.
static constexpr QualifierRow elsewhere_row(const char *name) {
    return {name, NameForm::EXACT, parse_read_elsewhere, nullptr, nullptr, nullptr, ObjectSpec::DEFAULT};
}
// A row decided over a candidate set (select_matching); a single object has no reading of it.
static constexpr QualifierRow population_row(const char *name, PopulationFn fn) {
    return {name, NameForm::EXACT, nullptr, ev_fails_closed, nullptr, fn, ObjectSpec::DEFAULT};
}

// Every qualifier the matcher reads, in the order a token is tried against them: an exact name
// before a prefix that would also take it, and the catch-alls last. --parse-report reads these rows
// too (filter_unknown_qualifiers), so a new qualifier is one row here.
static const QualifierRow kQualifierRows[] = {
    // identity / state keywords ------------------------------------------------
    set_row("IsRemembered", object_in_set, ObjectSpec::REMEMBERED),
    // IsImprinted — one of the cards the resolving ability imprinted (Atraxa's revealed pile).
    set_row("IsImprinted", object_in_set, ObjectSpec::IMPRINTED),
    // NamedCard — the object has the name chosen by a preceding name-a-card effect (Cabal
    // Therapy's discard, CR 201.4); nothing matches when no name was chosen.
    local_row("NamedCard", ev_named_card),
    relative_row("Other", ev_other),
    relative_row("Self", ev_self),
    set_row("nonChosenCard", ev_not_in_set, ObjectSpec::CHOSEN_CARDS),
    // ChosenCard — one of the cards a ChooseCard of this resolution chose (Strategic Betrayal's
    // ChangeZoneAll exiles the creature its opponent chose).
    set_row("ChosenCard", object_in_set, ObjectSpec::CHOSEN_CARDS),
    // targetedBy — the object IS the card the resolving ability chain targeted (Cloak and
    // Dagger, Entwined's exile filter alternative "Card.targetedBy" matches exactly the
    // creature its DBPump sub chose). Fails closed when no chain target is in context.
    set_row("targetedBy", object_in_set, ObjectSpec::TARGETED),
    // "AttachedTo <filter>": the object is attached to a permanent the filter matches (Springheart
    // Nantuko's "attached to a creature you control"). The filter runs to the next '+'.
    parsed_relative_row("AttachedTo ", NameForm::PREFIX, parse_attached_to, ev_attached_to),
    // EquippedBy — the object the filter's source is attached to (Hawkeye's Bow's "equipped creature").
    // One that has left matches as it last was: a creature tapped and sacrificed by one cost.
    relative_row("EquippedBy", ev_equipped_by),
    relative_row("YouCtrl", ev_you_ctrl),
    relative_row("OppCtrl", ev_opp_ctrl),
    // "ControlledBy TriggeredDefendingPlayer" (Forge): the object is controlled by the defending
    // player of the attack that fired this trigger (Frenzied Trapbreaker's attack-trigger Destroy).
    // The trigger is controlled by the attacking player (ctx.controller); in the two-player engine
    // the defending player is that controller's sole opponent, so this is OppCtrl semantics.
    relative_row("ControlledBy TriggeredDefendingPlayer", ev_opp_ctrl),
    // "ControlledBy ParentTarget" (Forge): the object is controlled by the player the parent
    // ability targeted (Cloak and Dagger, Entwined: "up to one target creature they control").
    set_row("ControlledBy ParentTarget", ev_controlled_by_set, ObjectSpec::PARENT_TARGET),
    // "ControlledBy Remembered": the object's controller is a remembered player (Chaos Defiler's "a
    // nonland permanent that player controls", a RepeatEach's current player).
    set_row("ControlledBy Remembered", ev_controlled_or_owned_by_set, ObjectSpec::REMEMBERED_PLAYERS),
    // RememberedPlayerCtrl: the same, in Stronghold Gambit's "a card in their hand". Count$Valid's
    // never reaches this row: it counts as that player's YouCtrl (svar_eval.cpp).
    set_row("RememberedPlayerCtrl", ev_controlled_or_owned_by_set, ObjectSpec::REMEMBERED_PLAYERS),
    // lowestRememberedCMC: a remembered card, with no remembered card of lower mana value (Stronghold
    // Gambit: "each creature card revealed this way with the lowest mana value"). The comparison is
    // with the remembered cards, not with the candidates a filter is matched over, so it is no
    // population row: a card left in hand is not among them, whatever its mana value.
    set_row("lowestRememberedCMC", ev_lowest_cmc_of_set, ObjectSpec::REMEMBERED),
    // TargetedPlayerCtrl / TargetedPlayerOwn: controlled or owned by the player the ability targets
    // (Strategic Betrayal: "a creature they control and their graveyard"). Count$Valid's
    // TargetedPlayerCtrl never reaches these: it counts as that player's YouCtrl (svar_eval.cpp).
    set_row("TargetedPlayerCtrl", ev_controlled_by_set, ObjectSpec::TARGETED),
    set_row("TargetedPlayerOwn", ev_owned_by_set, ObjectSpec::TARGETED),
    // SharesColorWithOther Remembered: another object of the set has a color of this one (Grindstone's
    // milled pair; a colorless object has no color to share, CR 105.2c).
    set_row("SharesColorWithOther Remembered", ev_shares_color_with_other, ObjectSpec::REMEMBERED),
    // Ownership (CR 108.3) — distinct from control; meaningful for cards in any zone
    // (e.g. Karn's -2 "an artifact card you own from outside the game or in exile").
    // Lenient when either side is unknown (bare CardData / no "you" supplied), mirroring
    // the YouCtrl off-battlefield convention.
    relative_row("YouOwn", ev_you_own),
    relative_row("OppOwn", ev_opp_own),
    local_row("token", ev_token),
    local_row("nonToken", ev_non_token),
    local_row("ThisTurnEntered", ev_entered_this_turn),
    // leastPower — no creature on the battlefield has less power (Drop of Honey's "the creature with
    // the least power"). It compares with every creature there, not with what the filter's other
    // qualifiers leave: with an indestructible creature least, withoutIndestructible matches nothing
    // rather than promoting the next creature up.
    local_row("leastPower", ev_least_power),
    // wasCastByYou — the permanent entered by being cast (The One Ring's "if you cast it"); its
    // controller cast it, since a permanent spell enters under its caster's control (CR 608.3a).
    local_row("wasCastByYou", ev_was_cast_by_you),
    // wasCastFromYourHand — a spell being cast from its caster's hand (Bilbo's "from anywhere other
    // than your hand"); a cost static reads it, and only the cast knows the zone.
    relative_row("wasCastFromYourHand", ev_cast_from_your_hand),
    // live combat / tap state (e.g. Guide of Souls' ValidTgts$ Creature.attacking) — only a
    // battlefield permanent can be in these states; a card view leaves them false.
    local_row("attacking", ev_attacking),
    local_row("blocking", ev_blocking),
    local_row("unblocked", ev_unblocked),
    local_row("tapped", ev_tapped),
    local_row("untapped", ev_untapped),
    local_row("Basic", ev_basic),
    local_row("nonBasic", ev_non_basic),
    local_row("Colorless", ev_colorless),  // CR 105.2c
    local_row("hasXCost", ev_has_x_cost),  // {X} in the printed mana cost (Gaddock Teeg)
    // hasABasicLandType — the object's subtypes include one of the five basic land types
    // (CR 305.6; Boseiju, Who Endures' compensation search "a land card with a basic land type").
    local_row("hasABasicLandType", ev_has_basic_land_type),
    // mana-value family -------------------------------------------------------
    // cmcLEX reads the X paid, unless the caller resolved X itself and supplied it as the
    // dynamic bound (Birthing Ritual's X = 1 + the sacrificed creature's mana value).
    parsed_relative_row("cmcLEX", NameForm::EXACT, parse_cmc_named, ev_cmc_le_x),
    // cmc<OP><n>: the object's mana value against a literal bound (CR 202.3), in this
    // alternative only ("Artifact.cmcLE3", Abrupt Decay's "Permanent.nonLand+cmcLE3").
    parsed_row("cmc", NameForm::PREFIX, parse_cmc_literal, ev_cmc_literal),
    // cmc<OP><SVar>: against ctx.cmc_bound, which only its caller knows (Aether Vial's MV ==
    // charge count); with no bound supplied it passes.
    parsed_relative_row("cmc", NameForm::PREFIX, parse_cmc_named, ev_cmc_bound),
    // dynamic power/toughness vs SVar X (Ensnaring Bridge: Creature.powerGTX — "power greater
    // than the number of cards in your hand"). The X value is supplied by the caller in
    // ctx.x_bound (the static source's controller's hand size). A P/T-less object never matches,
    // and with no X provided the qualifier can't be evaluated, so it fails closed.
    parsed_relative_row("power", NameForm::PREFIX, parse_pt_vs_x, ev_pt_vs_x),
    parsed_relative_row("toughness", NameForm::PREFIX, parse_pt_vs_x, ev_pt_vs_x),
    // A "power"/"toughness" comparator qualifier (e.g. "toughnessLE2", "powerGE5"): static
    // characteristic compared against the object's P/T (CR 208.2 / 107.1).
    parsed_row("power", NameForm::PREFIX, parse_pt_literal, ev_pt_literal),
    parsed_row("toughness", NameForm::PREFIX, parse_pt_literal, ev_pt_literal),
    // A counter-count qualifier "counters_<OP><N>_<TYPE>" (e.g. counters_GE1_VOID — "with a void
    // counter on it", CR 122.1): the number of TYPE counters on the object compared against N.
    parsed_row("counters_", NameForm::PREFIX, parse_counters, ev_counters),
    // positive color ----------------------------------------------------------
    parsed_row("White", NameForm::EXACT, parse_color, ev_color),
    parsed_row("Blue", NameForm::EXACT, parse_color, ev_color),
    parsed_row("Black", NameForm::EXACT, parse_color, ev_color),
    parsed_row("Red", NameForm::EXACT, parse_color, ev_color),
    parsed_row("Green", NameForm::EXACT, parse_color, ev_color),
    // negations ---------------------------------------------------------------
    // The "non" prefix and the card-type word are matched ASCII case-insensitively:
    // Forge scripts vary the token's casing (Cityscape Leveler writes "nonland" where
    // Abrupt Decay writes "nonLand"), and a case-sensitive miss would read "land" as a subtype,
    // which matches nothing, turning the restriction into a pass for every permanent, lands
    // included.
    parsed_row("non", NameForm::PREFIX_ANY_CASE, parse_non_color, ev_non_color),
    // nonColorless = "one or more colors" (Ugin's exiles, All Is Dust). "Colorless" is not a
    // color, so it has its own row: as a non<subtype> it would match EVERY object (nothing has
    // "Colorless" in its type line).
    parsed_row("non", NameForm::PREFIX_ANY_CASE, parse_non_colorless, ev_has_a_color),
    parsed_row("non", NameForm::PREFIX_ANY_CASE, parse_non_card_type, ev_not_type_line),
    parsed_row("non", NameForm::PREFIX_ANY_CASE, parse_non_subtype, ev_not_type_line),
    // with<Keyword> — the object currently has the named keyword ability (Pick Your Poison's
    // SacValid$ Creature.withFlying). Forge writes multi-word keywords without spaces
    // (withFirstStrike), so re-insert a space before each interior capital to recover the stored
    // keyword string ("FirstStrike" → "First Strike"). permanent_has_keyword reads the effective
    // keyword list for a battlefield permanent and falls back to the printed CardData keywords
    // for an off-battlefield card view, so this is correct in either zone.
    parsed_row("with", NameForm::PREFIX, parse_with_keyword, ev_has_keyword),
    // without<Keyword> — the object doesn't have it (Rough's Creature.withoutFlying, Drop of Honey's
    // withoutIndestructible), in the same reading.
    parsed_row("without", NameForm::PREFIX, parse_without_keyword, ev_lacks_keyword),
    // inZone<Zone> — the object currently sits in the named zone. A multi-zone search filter
    // uses it to restrict ONE OR-alternative to a single origin (Cloak and Dagger, Entwined's
    // "Card.nonLand+inZoneHand": only the hand alternative takes any nonland card; the
    // battlefield alternative is the targetedBy creature). Read from the live Zone component;
    // a bare CardData view (entity 0) fails closed.
    parsed_row("inZone", NameForm::PREFIX, parse_in_zone, ev_in_zone),
    // "ManaCostN" — the object's mana value equals exactly N (Urza's Saga's chapter III tutor:
    // Artifact.ManaCost0 / Artifact.ManaCost1 = an artifact with mana cost {0} or {1}). Distinct
    // from the cmc family above (which carries a comparator); ManaCostN is the equality form Forge
    // emits as a bare property. A P/T-less / costless object reads mana value 0.
    parsed_row("ManaCost", NameForm::PREFIX, parse_mana_cost, ev_mana_cost),
    // Read where the filter is used, in the forms its cards use them: targeting (Emblem, YouDontCtrl,
    // targeting.cpp), the sameName / Remembered /
    // TargetedCard / NotDefinedTargeted / sharesNameWith movers (effect_change_zone*.cpp), chosen
    // types (effect_choose_card.cpp), the player-relative count RememberedPlayerCtrl (svar_eval.cpp),
    // cast history (!wasCast, wasCastFromYourHandByYou),
    // kicker-linked triggers ("kicked N"), delayed-trigger memory (IsTriggerRemembered), Animate
    // Dead's StrictlySelf, Eldrazi Temple's Activated mana restriction and Sylvan Library's
    // DrawnThisTurn (its own SylvanLibrary resolution, parse_one_trigger). The same token in another
    // param is a gap these rows hide; the parse report does not prove a card.
    elsewhere_row("Emblem"),
    elsewhere_row("YouDontCtrl"),
    // ExiledWithSource — a card the filter's source exiled and still tracks (Agatha's Soul Cauldron).
    relative_row("ExiledWithSource", ev_exiled_with_source),
    // TopGraveyardCreature — the creature card nearest the top of its owner's graveyard (Shallow
    // Grave; a card put there goes on top, CR 404.1).
    relative_row("TopGraveyardCreature", ev_top_graveyard_creature),
    elsewhere_row("sameName"),
    elsewhere_row("Remembered"),
    elsewhere_row("TargetedCard"),
    elsewhere_row("NotDefinedTargeted"),
    elsewhere_row("sharesNameWith Targeted"),
    elsewhere_row("ChosenType"),
    elsewhere_row("wasCast"),
    elsewhere_row("wasCastFromYourHandByYou"),
    elsewhere_row("IsTriggerRemembered"),
    elsewhere_row("StrictlySelf"),
    elsewhere_row("Activated"),
    elsewhere_row("DrawnThisTurn"),
    {"kicked", NameForm::PREFIX, parse_read_elsewhere, nullptr, nullptr, nullptr, ObjectSpec::DEFAULT},
    // leftover: a PascalCase token is a type-line (type/supertype/subtype) name to require;
    // anything else is a qualifier we don't implement → fail closed and warn once.
    parsed_row("", NameForm::ANY, parse_type_line_name, ev_type_line),
    parsed_row("", NameForm::ANY, parse_unknown, ev_unknown),
};

// The checks a head makes that are not qualifiers a script can spell after it.
static constexpr QualifierRow kHeadTypeLineRow = local_row("", ev_type_line);
static constexpr QualifierRow kHeadPermanentRow = local_row("", ev_permanent);
// An empty qualifier ("!" negating nothing) always passes.
static constexpr QualifierRow kEmptyRow = local_row("", ev_always);

// ── Compiling ────────────────────────────────────────────────────────────

static std::unordered_map<std::string, std::unique_ptr<CompiledFilter>> &compiled_filters() {
    static std::unordered_map<std::string, std::unique_ptr<CompiledFilter>> store;
    return store;
}

FilterSpec::FilterSpec(const std::string &text) : compiled_(text.empty() ? nullptr : compile_filter(text)) {}

const std::string &FilterSpec::text() const {
    static const std::string empty_text;
    return compiled_ ? compiled_->text : empty_text;
}

// The compiled form of `text`, made the first time it is asked for. Each ';' or ',' separates an
// OR alternative: Forge writes "Creature,Planeswalker" (SacValid$ on Archon of Cruelty) with a
// comma. No qualifier token contains a comma.
static const CompiledFilter *compile_filter(const std::string &text) {
    auto &store = compiled_filters();
    auto it = store.find(text);
    if (it != store.end()) return it->second.get();
    auto compiled = std::make_unique<CompiledFilter>();
    compiled->text = text;
    size_t pp = 0;
    while (pp <= text.size()) {
        size_t sc = text.find_first_of(";,", pp);
        if (sc == std::string::npos) sc = text.size();
        const std::string alt = text.substr(pp, sc - pp);
        pp = sc + 1;
        if (!alt.empty()) compiled->alternatives.push_back(compile_alternative(alt));
    }
    return store.emplace(text, std::move(compiled)).first->second.get();
}

// ONE ';'-free alternative: "head[.q][+q]…".
static Alternative compile_alternative(const std::string &alt) {
    Alternative out;
    const size_t sep = alt.find_first_of(".+");
    out.head = alt.substr(0, sep);
    add_head_qualifier(out);
    if (sep == std::string::npos) return out;
    const std::string rest = alt.substr(sep + 1);
    size_t p = 0;
    while (p <= rest.size()) {
        size_t nx = rest.find_first_of(rest.compare(p, 11, "AttachedTo ") == 0 ? "+" : ".+", p);
        if (nx == std::string::npos) nx = rest.size();
        const std::string q = rest.substr(p, nx - p);
        p = nx + 1;
        if (!q.empty()) out.qualifiers.push_back(compile_qualifier(q));
    }
    return out;
}

// What the alternative's head requires. "Self"/"Other" as a BARE head are identity qualifiers, not
// type-line names — Forge writes SacValid$ Self (Uro: "sacrifice it") and "Other" the same way.
// Treat them like Card.<head>: no type-line requirement, only the identity check. "Permanent" (CR
// 110.4a) requires an actual permanent card type: a no-op for battlefield objects (always
// permanents), it excludes instants/sorceries when matching a card in another zone.
static void add_head_qualifier(Alternative &alt) {
    const std::string &head = alt.head;
    if (head.empty() || head == "Card" || head == "Spell") return;
    alt.head_qualifiers = 1;
    if (head == "Self" || head == "Other") {
        alt.qualifiers.push_back(compile_qualifier(head));
        return;
    }
    Qualifier q;
    q.token = q.name = q.text = head;
    q.row = head == "Permanent" ? &kHeadPermanentRow : &kHeadTypeLineRow;
    alt.qualifiers.push_back(std::move(q));
}

static bool row_names(const QualifierRow &row, const std::string &token) {
    const size_t len = std::strlen(row.name);
    switch (row.form) {
    case NameForm::EXACT: return token == row.name;
    case NameForm::PREFIX: return token.compare(0, len, row.name) == 0;
    case NameForm::PREFIX_ANY_CASE: return ascii_lower(token.substr(0, len)) == row.name;
    case NameForm::ANY: return true;
    }
    return false;
}

// One '.'/'+'-joined token: its '!'s, then the first row that reads what is left.
static Qualifier compile_qualifier(const std::string &token) {
    Qualifier q;
    q.token = token;
    size_t bangs = 0;
    while (bangs < token.size() && token[bangs] == '!') bangs++;
    q.negate = bangs % 2 == 1;
    q.name = token.substr(bangs);
    q.row = &kEmptyRow;
    if (q.name.empty()) return q;
    for (const QualifierRow &row : kQualifierRows) {
        if (!row_names(row, q.name)) continue;
        if (row.parse && !row.parse(q.name, q)) continue;
        q.row = &row;
        q.set = ObjectSpec(row.set);
        if (q.reads_type_name && !known_type_name(q.text)) q.has_reading = false;
        return q;
    }
    return q;
}

static bool all_digits(const std::string &s, size_t from) {
    return std::all_of(s.begin() + static_cast<std::ptrdiff_t>(from), s.end(),
                       [](unsigned char ch) { return std::isdigit(ch); });
}

static bool parse_cmc_literal(const std::string &token, Qualifier &q) {
    if (token.size() <= 5 || !all_digits(token, 5)) return false;
    q.op_text = token.substr(3, 2);
    q.op = svar_op_of(q.op_text);
    q.text = token.substr(5);
    q.number = std::stoi(q.text);
    return true;
}

// cmc<OP><SVar>, whatever follows "cmc".
static bool parse_cmc_named(const std::string &token, Qualifier &q) {
    q.op_text = token.substr(3, 2);
    q.op = svar_op_of(q.op_text);
    if (token.size() > 5) q.text = token.substr(5);
    return true;
}

// <attr> + 2-letter op + "X".
static bool parse_pt_vs_x(const std::string &token, Qualifier &q) {
    q.of_power = token.rfind("power", 0) == 0;
    const std::string rest = token.substr(q.of_power ? 5 : 9);
    if (rest.size() != 3 || rest[2] != 'X') return false;
    q.op_text = rest.substr(0, 2);
    q.op = svar_op_of(q.op_text);
    return true;
}

static bool parse_pt_literal(const std::string &token, Qualifier &q) {
    q.of_power = token.rfind("power", 0) == 0;
    const std::string rest = token.substr(q.of_power ? 5 : 9);
    if (rest.size() < 3 || !all_digits(rest, 2)) return false;
    q.op_text = rest.substr(0, 2);
    q.op = svar_op_of(q.op_text);
    q.number = std::stoi(rest.substr(2));
    return true;
}

static bool parse_counters(const std::string &token, Qualifier &q) {
    const std::string rest = token.substr(std::strlen("counters_"));
    const size_t sep = rest.find('_', 2);
    if (rest.size() < 4 || sep == std::string::npos || sep == 2) return false;
    const std::string num = rest.substr(2, sep - 2);
    if (!all_digits(num, 0)) return false;
    q.op_text = rest.substr(0, 2);
    q.op = svar_op_of(q.op_text);
    q.number = std::stoi(num);
    q.text = rest.substr(sep + 1);
    return true;
}

static bool parse_color(const std::string &token, Qualifier &q) { return color_token(token, q.color); }

static bool parse_non_color(const std::string &token, Qualifier &q) {
    return token.size() > 3 && color_token(token.substr(3), q.color);
}

static bool parse_non_colorless(const std::string &token, Qualifier &) {
    return token.size() > 3 && ascii_lower(token.substr(3)) == "colorless";
}

// Main card types, for the non<CardType> negation (CR 110.4a + the spell-only types).
static bool parse_non_card_type(const std::string &token, Qualifier &q) {
    static const char *const kCardTypes[] = {"Land", "Creature", "Artifact", "Enchantment",
                                             "Planeswalker", "Battle", "Instant", "Sorcery", "Tribal"};
    if (token.size() <= 3) return false;
    const std::string rest_lc = ascii_lower(token.substr(3));
    for (const char *ct : kCardTypes) {
        if (rest_lc != ascii_lower(ct)) continue;
        q.text = ct;
        return true;
    }
    return false;
}

static bool parse_non_subtype(const std::string &token, Qualifier &q) {
    if (token.size() <= 3) return false;
    q.text = token.substr(3);
    q.reads_type_name = true;
    return true;
}

// The keyword spelled from `from` on, a capital starting each of its words.
static bool parse_keyword_after(const std::string &token, size_t from, Qualifier &q) {
    if (token.size() <= from || !std::isupper(static_cast<unsigned char>(token[from]))) return false;
    for (size_t i = from; i < token.size(); i++) {
        if (i > from && std::isupper(static_cast<unsigned char>(token[i]))) q.text += ' ';
        q.text += token[i];
    }
    return true;
}

static bool parse_with_keyword(const std::string &token, Qualifier &q) {
    return parse_keyword_after(token, std::strlen("with"), q);
}

static bool parse_without_keyword(const std::string &token, Qualifier &q) {
    return parse_keyword_after(token, std::strlen("without"), q);
}

static bool parse_in_zone(const std::string &token, Qualifier &q) {
    if (token.size() <= 6) return false;
    static const struct {
        const char *name;
        Zone::ZoneValue zone;
    } kZones[] = {{"Hand", Zone::HAND},           {"Battlefield", Zone::BATTLEFIELD}, {"Library", Zone::LIBRARY},
                  {"Graveyard", Zone::GRAVEYARD}, {"Exile", Zone::EXILE},             {"Stack", Zone::STACK}};
    const std::string zone = token.substr(6);
    for (const auto &z : kZones) {
        if (zone != z.name) continue;
        q.zone = z.zone;
        q.known_zone = true;
    }
    return true;
}

static bool parse_mana_cost(const std::string &token, Qualifier &q) {
    if (token.size() <= 8 || !all_digits(token, 8)) return false;
    q.number = std::stoi(token.substr(8));
    return true;
}

// Notes it and leaves the token to the rows after it.
static bool parse_read_elsewhere(const std::string &, Qualifier &q) {
    q.read_elsewhere = true;
    return false;
}

static bool parse_type_line_name(const std::string &token, Qualifier &q) {
    if (!std::isupper(static_cast<unsigned char>(token[0]))) return false;
    q.text = token;
    q.reads_type_name = true;
    return true;
}

static bool parse_unknown(const std::string &, Qualifier &q) {
    q.has_reading = false;
    return true;
}

// ── Matching ─────────────────────────────────────────────────────────────

// A token the matcher can't read fails closed, its negation too: a "!" of nothing known says nothing
// about the object, so it can't pass it.
static bool eval_qualifier(const Qualifier &q, const CharView &v, const MatchCtx &ctx) {
    const QualifierRow &row = *q.row;
    const bool holds = row.local ? row.local(v, q) : row.relative(v, q, ctx);
    return q.has_reading && holds != q.negate;
}

static bool alternative_matches(const Alternative &alt, const CharView &v, const MatchCtx &ctx) {
    for (const Qualifier &q : alt.qualifiers)
        if (!eval_qualifier(q, v, ctx)) return false;
    return true;
}

static bool matches_view(const CharView &v, const CompiledFilter *filter, const MatchCtx &ctx) {
    if (!filter) return false;  // an empty spec matches nothing (existing convention)
    for (const Alternative &alt : filter->alternatives)
        if (alternative_matches(alt, v, ctx)) return true;  // ';'/',' is OR
    return false;
}

// Whether an object in its current zone has `name`: a permanent's (the face it shows, CR 712.8e),
// a spell's cast face, a split card's either half (CR 709.4a; a fused spell's too, 709.4d).
static bool object_has_name(Entity e, const std::string &name) {
    if (global_coordinator.entity_has_component<Permanent>(e))
        return global_coordinator.GetComponent<Permanent>(e).name == name;
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    const CardData &cd = global_coordinator.GetComponent<CardData>(e);
    const CardData &face = card_characteristics(e, cd);
    if (&face != cd.combined.get()) return face.name == name;
    const std::vector<std::string> names = card_names(cd);
    return std::find(names.begin(), names.end(), name) != names.end();
}

static bool view_has_typeline(const CharView &v, const std::string &name) {
    if (!v.types) return false;
    // CR 702.151b / 702.175d: some battlefield permanents keep "Creature" in their printed type
    // line yet are not currently creatures — a Reconfigure equipment while attached, an Impending
    // permanent that still has time counters. The live Creature component is the source of truth
    // for creature-ness on the battlefield (it is what the static pass strips in those cases), so a
    // "Creature" type query consults it rather than the stale type line. Other type names, and
    // off-battlefield card views, read the type line unchanged.
    if (name == "Creature" && v.creature_suppressed) return false;
    for (const auto &t : *v.types)
        if (t.name == name) return true;
    return false;
}

// True when the object has a permanent card type (CR 110.4a). The "Permanent" filter head must
// require this so an off-battlefield card filter (e.g. Lion Sash's "if it was a permanent card",
// matched against a card now in exile) excludes instants/sorceries. Battlefield objects always
// carry a permanent type in their live type line, so this stays a no-op for permanent_view.
static bool view_is_permanent(const CharView &v) {
    if (v.on_battlefield) return true;  // a battlefield object is by definition a permanent (CR 110.4a)
    return view_has_typeline(v, "Artifact") || view_has_typeline(v, "Battle") ||
           view_has_typeline(v, "Creature") || view_has_typeline(v, "Enchantment") ||
           view_has_typeline(v, "Land") || view_has_typeline(v, "Planeswalker");
}

static bool color_token(const std::string &q, Colors &c) {
    if (q == "White") { c = WHITE; return true; }
    if (q == "Blue")  { c = BLUE;  return true; }
    if (q == "Black") { c = BLACK; return true; }
    if (q == "Red")   { c = RED;   return true; }
    if (q == "Green") { c = GREEN; return true; }
    return false;
}

// True when the object is one or more colors (CR 105.2a). Every color source (card_colors,
// effective_colors, last-known info) holds only the five colors, so this is non-emptiness.
// Single source for the Colorless / nonColorless qualifier pair.
static bool view_has_any_color(const CharView &v) { return !v.colors.empty(); }

// A comparator's reading of lhs against rhs; an operator the engine can't read is reported.
static bool compare_qualifier(int lhs, const Qualifier &q, int rhs) {
    return q.op == SvarOp::UNKNOWN ? apply_svar_op(lhs, q.op_text, rhs) : apply_svar_op(lhs, q.op, rhs);
}

static void warn_unknown_qualifier(const std::string &q) {
    static std::set<std::string> warned;  // once per distinct token — this is on the SBA/legality hot path
    if (warned.insert(q).second)
        printf("WARNING: Unrecognized filter qualifier '%s' (fail-closed, matched nothing)\n", q.c_str());
}

// The reference a set-relative qualifier's ObjectSpec resolves against.
static ResolveCtx set_context(const MatchCtx &ctx) {
    return ctx.resolve ? *ctx.resolve : ResolveCtx::of_static(ctx.controller, ctx.source);
}

static bool object_in_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    if (v.entity == 0) return false;
    const std::vector<Entity> set = resolve_filter_objects(q.set, set_context(ctx));
    return std::find(set.begin(), set.end(), v.entity) != set.end();
}

static bool ev_always(const CharView &, const Qualifier &) { return true; }
static bool ev_fails_closed(const CharView &, const Qualifier &) { return false; }


static bool ev_named_card(const CharView &v, const Qualifier &) {
    const std::string &named = cur_game.resolution.memory.named_card;
    return v.entity != 0 && !named.empty() && object_has_name(v.entity, named);
}

static bool ev_other(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return ctx.source == 0 || v.entity != ctx.source;
}

static bool ev_self(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return ctx.source != 0 && v.entity == ctx.source;
}

static bool ev_not_in_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    return !object_in_set(v, q, ctx);
}

static bool parse_attached_to(const std::string &token, Qualifier &q) {
    q.text = token.substr(std::strlen("AttachedTo "));
    return true;
}

static bool ev_attached_to(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    if (!v.on_battlefield || !global_coordinator.entity_has_component<Permanent>(v.entity)) return false;
    const Entity host = global_coordinator.GetComponent<Permanent>(v.entity).equipped_to.get();
    return host != 0 && permanent_matches_filter(host, FilterSpec(q.text), ctx);
}

static bool ev_equipped_by(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    if (v.entity == 0 || !global_coordinator.entity_has_component<Permanent>(ctx.source)) return false;
    coverage_row("EquippedBy");
    if (global_coordinator.GetComponent<Permanent>(ctx.source).equipped_to.get() == v.entity) return true;
    if (is_battlefield_permanent(v.entity)) return false;
    const LastKnownInfo *lki = departed_lki_for(v.entity);
    return lki && refs_contain(lki->attached, ctx.source);
}

static bool ev_you_ctrl(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return !v.on_battlefield || v.controller == ctx.controller;
}

static bool ev_opp_ctrl(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return !v.on_battlefield || v.controller != ctx.controller;
}

static bool ev_controlled_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    if (!v.on_battlefield) return true;
    const std::vector<Zone::Ownership> players = resolve_filter_players(q.set, set_context(ctx));
    return std::find(players.begin(), players.end(), v.controller) != players.end();
}

// Off the battlefield a card's controller is its owner (CR 108.4a).
static bool ev_controlled_or_owned_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    coverage_row("ControlledByRemembered");
    const std::vector<Zone::Ownership> players = resolve_filter_players(q.set, set_context(ctx));
    return std::find(players.begin(), players.end(), v.on_battlefield ? v.controller : v.owner) != players.end();
}

static bool ev_owned_by_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    coverage_row("TargetedPlayerOwn");
    const std::vector<Zone::Ownership> players = resolve_filter_players(q.set, set_context(ctx));
    return std::find(players.begin(), players.end(), v.owner) != players.end();
}

static bool ev_shares_color_with_other(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    coverage_row("SharesColorWithOther");
    for (Entity other : resolve_filter_objects(q.set, set_context(ctx))) {
        if (other == v.entity) continue;
        for (Colors color : effective_colors(other))
            if (v.colors.count(color)) return true;
    }
    return false;
}

static bool ev_lowest_cmc_of_set(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    coverage_row("lowestRememberedCMC");
    const std::vector<Entity> set = resolve_filter_objects(q.set, set_context(ctx));
    if (std::find(set.begin(), set.end(), v.entity) == set.end()) return false;
    for (Entity e : set)
        if (global_coordinator.entity_has_component<CardData>(e) &&
            object_mana_value(e, global_coordinator.GetComponent<CardData>(e)) < v.cmc)
            return false;
    return true;
}

static bool ev_you_own(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return ctx.controller == Zone::UNKNOWN || v.owner == Zone::UNKNOWN || v.owner == ctx.controller;
}

static bool ev_opp_own(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return ctx.controller == Zone::UNKNOWN || v.owner == Zone::UNKNOWN || v.owner != ctx.controller;
}

static bool ev_cast_from_your_hand(const CharView &, const Qualifier &, const MatchCtx &ctx) {
    return ctx.cast_from == Zone::HAND;
}

static bool ev_token(const CharView &v, const Qualifier &) { return v.is_token; }
static bool ev_non_token(const CharView &v, const Qualifier &) { return !v.is_token; }

static bool ev_entered_this_turn(const CharView &v, const Qualifier &) {
    return v.on_battlefield && entered_battlefield_this_turn(v.entered_on_turn);
}

static bool ev_least_power(const CharView &v, const Qualifier &) {
    coverage_row("leastPower");
    if (!v.on_battlefield || !v.has_pt) return false;
    for (Entity e : battlefield_permanents(zoned_entities()))
        if (global_coordinator.entity_has_component<Creature>(e) && effective_power(e) < v.power) return false;
    return true;
}

static bool ev_was_cast_by_you(const CharView &v, const Qualifier &) { return v.on_battlefield && v.entered_by_cast; }
static bool ev_attacking(const CharView &v, const Qualifier &) { return v.is_attacking; }
static bool ev_blocking(const CharView &v, const Qualifier &) { return v.is_blocking; }
static bool ev_unblocked(const CharView &v, const Qualifier &) { return v.is_unblocked; }
static bool ev_tapped(const CharView &v, const Qualifier &) { return v.is_tapped; }
static bool ev_untapped(const CharView &v, const Qualifier &) { return !v.is_tapped; }
static bool ev_basic(const CharView &v, const Qualifier &) { return v.types && has_basic_supertype(*v.types); }
static bool ev_non_basic(const CharView &v, const Qualifier &) { return v.types && !has_basic_supertype(*v.types); }
static bool ev_colorless(const CharView &v, const Qualifier &) { return !view_has_any_color(v); }
static bool ev_has_x_cost(const CharView &v, const Qualifier &) { return v.has_x_cost; }

static bool ev_has_basic_land_type(const CharView &v, const Qualifier &) {
    return v.types && has_a_basic_land_type(*v.types);
}

static bool ev_cmc_le_x(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    return v.cmc <= (ctx.cmc_bound >= 0 ? ctx.cmc_bound : current_x_paid());
}

static bool ev_cmc_literal(const CharView &v, const Qualifier &q) { return compare_qualifier(v.cmc, q, q.number); }

static bool ev_cmc_bound(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    return ctx.cmc_bound < 0 || compare_qualifier(v.cmc, q, ctx.cmc_bound);
}

// The power or toughness a qualifier compares (ObjectCharacteristics::counted_pt).
static int view_pt(const CharView &v, const Qualifier &q) {
    if (!v.counted_pt) return q.of_power ? v.power : v.toughness;
    const PowerToughness pt = card_power_toughness(v.entity, global_coordinator.GetComponent<CardData>(v.entity));
    return q.of_power ? pt.power : pt.toughness;
}

static bool ev_pt_vs_x(const CharView &v, const Qualifier &q, const MatchCtx &ctx) {
    if (!v.has_pt || ctx.x_bound == INT_MIN) return false;
    return compare_qualifier(view_pt(v, q), q, ctx.x_bound);
}

// A P/T filter never matches a P/T-less object.
static bool ev_pt_literal(const CharView &v, const Qualifier &q) {
    return v.has_pt && compare_qualifier(view_pt(v, q), q, q.number);
}

static bool ev_exiled_with_source(const CharView &v, const Qualifier &, const MatchCtx &ctx) {
    if (v.entity == 0) return false;
    if (global_coordinator.entity_has_component<Permanent>(ctx.source))
        return refs_contain(global_coordinator.GetComponent<Permanent>(ctx.source).exiled_with, v.entity);
    // A source sacrificed to pay for this ability (Ghost Vacuum): what it had exiled as it left.
    const LastKnownInfo *lki = departed_lki_for(ctx.source);
    return lki && refs_contain(lki->exiled_with, v.entity);
}

static bool ev_top_graveyard_creature(const CharView &v, const Qualifier &, const MatchCtx &) {
    if (v.entity == 0 || !global_coordinator.entity_has_component<Zone>(v.entity)) return false;
    const Zone &zone = global_coordinator.GetComponent<Zone>(v.entity);
    if (zone.location != Zone::GRAVEYARD) return false;
    coverage_row("TopGraveyardCreature");
    for (Entity e : zoned_entities()) {
        const Zone &other = global_coordinator.GetComponent<Zone>(e);
        if (other.location != Zone::GRAVEYARD || other.owner != zone.owner) continue;
        if (other.distance_from_top < zone.distance_from_top && type_set_has(effective_types(e), "Creature"))
            return false;
    }
    return true;
}

static bool ev_counters(const CharView &v, const Qualifier &q) {
    return v.entity != 0 && compare_qualifier(object_counters(v.entity, q.text), q, q.number);
}

static bool ev_color(const CharView &v, const Qualifier &q) { return v.colors.count(q.color) > 0; }
static bool ev_non_color(const CharView &v, const Qualifier &q) { return v.colors.count(q.color) == 0; }
static bool ev_has_a_color(const CharView &v, const Qualifier &) { return view_has_any_color(v); }
static bool ev_not_type_line(const CharView &v, const Qualifier &q) { return !view_has_typeline(v, q.text); }
static bool ev_has_keyword(const CharView &v, const Qualifier &q) {
    return v.entity != 0 && permanent_has_keyword(v.entity, q.text.c_str());
}

// A bare card has no keywords to read, so it has no reading of "without" either.
static bool ev_lacks_keyword(const CharView &v, const Qualifier &q) {
    coverage_row("without");
    return v.entity != 0 && !permanent_has_keyword(v.entity, q.text.c_str());
}

static bool ev_in_zone(const CharView &v, const Qualifier &q) {
    if (v.entity == 0 || !global_coordinator.entity_has_component<Zone>(v.entity)) return false;
    return q.known_zone && global_coordinator.GetComponent<Zone>(v.entity).location == q.zone;
}

static bool ev_mana_cost(const CharView &v, const Qualifier &q) { return v.cmc == q.number; }
static bool ev_type_line(const CharView &v, const Qualifier &q) { return view_has_typeline(v, q.text); }
static bool ev_permanent(const CharView &v, const Qualifier &) { return view_is_permanent(v); }

static bool ev_unknown(const CharView &, const Qualifier &q) {
    warn_unknown_qualifier(q.name);
    return false;
}

static void read_characteristics(CharView &v, ObjectCharacteristics &&c) {
    v.types = c.types;
    v.colors = std::move(c.colors);
    v.has_pt = c.has_pt;
    v.power = c.power;
    v.toughness = c.toughness;
    v.counted_pt = c.counted_pt;
}

// What a card view holds beside the object's characteristics.
static CharView card_identity_view(Entity e, const CardData &cd) {
    CharView v;
    v.entity = e;
    v.cmc = object_mana_value(e, cd);
    v.has_x_cost = (e != 0 ? card_characteristics(e, cd) : cd).has_x_cost;
    if (e != 0 && global_coordinator.entity_has_component<Zone>(e))
        v.owner = global_coordinator.GetComponent<Zone>(e).owner;
    return v;
}

static CharView printed_view(Entity e, const CardData &cd) {
    CharView v = card_identity_view(e, cd);
    read_characteristics(v, printed_characteristics(e, cd));
    return v;
}

static CharView card_view(Entity e, const CardData &cd) {
    CharView v = card_identity_view(e, cd);
    read_characteristics(v, card_object_characteristics(e, cd));
    return v;
}

// Mana value of a permanent (CR 112.7), read from the face mana_value_face picks. A token (no
// card) has the mana value it copied (Token::mana_value, CR 707.2), 0 for a scripted token.
static void set_permanent_mana_value(CharView &v, Entity e, bool transformed) {
    if (!global_coordinator.entity_has_component<CardData>(e)) {
        if (global_coordinator.entity_has_component<Token>(e))
            v.cmc = global_coordinator.GetComponent<Token>(e).mana_value;
        return;
    }
    const auto &cd = global_coordinator.GetComponent<CardData>(e);
    const CardData &mv_face = mana_value_face(cd, transformed);
    v.cmc = card_mana_value(mv_face);
    v.has_x_cost = mv_face.has_x_cost;
}

static CharView permanent_view(Entity e, const Permanent &perm) {
    CharView v;
    v.entity = e;
    read_characteristics(v, permanent_characteristics(e, perm));
    v.is_token = perm.is_token;
    v.controller = perm.controller;
    if (global_coordinator.entity_has_component<Zone>(e))
        v.owner = global_coordinator.GetComponent<Zone>(e).owner;
    v.entered_on_turn = static_cast<long>(perm.entered_on_turn);
    v.on_battlefield = true;
    v.is_tapped = perm.is_tapped;
    v.entered_by_cast = perm.entered_by_cast;
    if (global_coordinator.entity_has_component<Creature>(e)) {
        auto &cr = global_coordinator.GetComponent<Creature>(e);
        v.is_attacking = cr.is_attacking;
        v.is_blocking = cr.is_blocking;
        v.is_unblocked = is_unblocked_attacker(e);
    } else {
        // On the battlefield with no live Creature component: any "Creature" still in the type line
        // is suppressed (CR 702.151b reconfigure-while-attached / 702.175d impending). See
        // view_has_typeline — a "Creature" type query must read the component, not the stale line.
        v.creature_suppressed = true;
    }
    set_permanent_mana_value(v, e, perm.transformed);
    return v;
}

// The object `e` as it last existed on the battlefield (CR 603.10a / 608.2h): its last-known
// type line, controller, colors and P/T. Its owner and mana value come from the card, which
// keeps its identity (a vanished token is owned by its last controller and has mana value 0).
static CharView lki_view(Entity e, const LastKnownInfo &lki) {
    CharView v;
    v.entity = e;
    v.types = &lki.types;
    v.colors = lki.colors;
    v.is_token = lki.is_token;
    v.controller = lki.controller;
    v.owner = global_coordinator.entity_has_component<Zone>(e)
                  ? global_coordinator.GetComponent<Zone>(e).owner
                  : lki.controller;
    v.on_battlefield = true;
    v.entered_by_cast = lki.entered_by_cast;
    v.has_pt = type_set_has(lki.types, "Creature");
    v.power = lki.power;
    v.toughness = lki.toughness;
    set_permanent_mana_value(v, e, lki.transformed);
    return v;
}

// `e` as object_matches_filter reads it. False for an object with no view to match.
static bool object_view(Entity e, CharView &v) {
    if (is_battlefield_permanent(e)) {
        v = permanent_view(e, global_coordinator.GetComponent<Permanent>(e));
        return true;
    }
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    v = card_view(e, global_coordinator.GetComponent<CardData>(e));
    return true;
}

bool card_matches_filter(const CardData &cd, const FilterSpec &spec, const MatchCtx &ctx) {
    return matches_view(card_view(0, cd), spec.compiled(), ctx);
}

bool card_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    return matches_view(card_view(e, global_coordinator.GetComponent<CardData>(e)), spec.compiled(), ctx);
}

bool card_matches_printed(Entity e, const FilterSpec &spec, const MatchCtx &ctx) {
    if (!global_coordinator.entity_has_component<CardData>(e)) return false;
    return matches_view(printed_view(e, global_coordinator.GetComponent<CardData>(e)), spec.compiled(), ctx);
}

bool permanent_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx) {
    if (!is_battlefield_permanent(e)) return false;
    return matches_view(permanent_view(e, global_coordinator.GetComponent<Permanent>(e)), spec.compiled(), ctx);
}

bool object_matches_filter(Entity e, const FilterSpec &spec, const MatchCtx &ctx) {
    return is_battlefield_permanent(e) ? permanent_matches_filter(e, spec, ctx)
                                       : card_matches_filter(e, spec, ctx);
}

std::vector<Entity> select_matching(const std::vector<Entity> &candidates, const FilterSpec &spec,
                                    const MatchCtx &ctx) {
    std::vector<Entity> selected;
    if (spec.empty()) return selected;
    for (const Alternative &alt : spec.compiled()->alternatives) {
        std::vector<Entity> survivors;
        for (Entity e : candidates) {
            CharView v;
            if (!object_view(e, v)) continue;
            bool passes = true;
            for (const Qualifier &q : alt.qualifiers)
                passes = passes && (q.row->population || eval_qualifier(q, v, ctx));
            if (passes) survivors.push_back(e);
        }
        for (const Qualifier &q : alt.qualifiers)
            if (q.row->population) q.row->population(q, ctx, survivors);
        selected.insert(selected.end(), survivors.begin(), survivors.end());
    }
    std::vector<Entity> in_order;
    for (Entity e : candidates)
        if (std::find(selected.begin(), selected.end(), e) != selected.end()) in_order.push_back(e);
    return in_order;
}

bool zone_change_object_matches(Entity e, Zone::ZoneValue origin, Zone::ZoneValue destination,
                                const FilterSpec &spec, const MatchCtx &ctx) {
    bool left_battlefield = (origin == Zone::BATTLEFIELD);
    if (!left_battlefield && destination == Zone::BATTLEFIELD && is_battlefield_permanent(e))
        return permanent_matches_filter(e, spec, ctx);
    if (left_battlefield || destination == Zone::BATTLEFIELD) {
        if (const LastKnownInfo *lki = departed_lki_for(e))
            return matches_view(lki_view(e, *lki), spec.compiled(), ctx);
    }
    return card_matches_filter(e, spec, ctx);
}

// The '.'/'+'/','/';'-delimited tokens of a filter spec, each flagged whether it is the head
// (first) token of its OR alternative.
static std::vector<std::pair<std::string, bool>> filter_tokens(const std::string &spec) {
    std::vector<std::pair<std::string, bool>> out;
    size_t p = 0;
    bool head = true;
    while (p <= spec.size()) {
        size_t nx = spec.find_first_of(".+,;", p);
        if (nx == std::string::npos) nx = spec.size();
        out.emplace_back(spec.substr(p, nx - p), head);
        head = (nx < spec.size() && (spec[nx] == ',' || spec[nx] == ';'));
        p = nx + 1;
    }
    return out;
}

bool filter_names_token(const std::string &spec, const std::string &token) {
    for (const auto &t : filter_tokens(spec))
        if (t.first == token) return true;
    return false;
}

static std::string owner_relative_text(const std::string &spec) {
    std::string out;
    size_t p = 0;
    while (p <= spec.size()) {
        size_t nx = spec.find_first_of(".+,;", p);
        if (nx == std::string::npos) nx = spec.size();
        std::string tok = spec.substr(p, nx - p);
        if (tok == "YouCtrl") tok = "YouOwn";
        else if (tok == "OppCtrl") tok = "OppOwn";
        out += tok;
        if (nx < spec.size()) out += spec[nx];
        p = nx + 1;
    }
    return out;
}

FilterSpec owner_relative_filter(const FilterSpec &spec) {
    if (spec.empty()) return spec;
    const CompiledFilter &compiled = *spec.compiled();
    if (!compiled.owner_relative_read) {
        compiled.owner_relative = FilterSpec(owner_relative_text(compiled.text));
        compiled.owner_relative_read = true;
    }
    return compiled.owner_relative;
}

FilterSpec target_filter(const FilterSpec &valid_tgts) {
    if (valid_tgts.empty()) return valid_tgts;
    const CompiledFilter &compiled = *valid_tgts.compiled();
    if (!compiled.target_form_read) {
        std::string spec = compiled.text == "Any" ? std::string("Creature,Planeswalker") : compiled.text;
        for (size_t pos = spec.find("YouDontCtrl"); pos != std::string::npos; pos = spec.find("YouDontCtrl", pos))
            spec.replace(pos, std::string("YouDontCtrl").size(), "OppCtrl");
        compiled.target_form = FilterSpec(spec);
        compiled.target_form_read = true;
    }
    return compiled.target_form;
}

std::vector<FilterCmcBound> filter_cmc_bounds(const FilterSpec &spec) {
    std::vector<FilterCmcBound> out;
    if (spec.empty()) return out;
    for (const Alternative &alt : spec.compiled()->alternatives)
        for (size_t i = alt.head_qualifiers; i < alt.qualifiers.size(); i++) {
            const Qualifier &q = alt.qualifiers[i];
            const bool literal = q.row->local == ev_cmc_literal;
            if ((literal || q.row->relative == ev_cmc_bound || q.row->relative == ev_cmc_le_x) && !q.text.empty())
                out.push_back({q.token, q.op_text, q.text, !literal});
        }
    return out;
}

bool filter_has_head(const std::string &spec, const std::string &head) {
    for (const auto &t : filter_tokens(spec))
        if (t.second && t.first == head) return true;
    return false;
}

bool color_set_passes(const std::string &vt, const std::set<Colors> &colors) {
    static const struct { const char *tok; Colors col; } table[] = {
        {".White", WHITE}, {".Blue", BLUE}, {".Black", BLACK}, {".Red", RED}, {".Green", GREEN}};
    for (auto &e : table)
        if (vt.find(e.tok) != std::string::npos && !colors.count(e.col)) return false;
    return true;
}

// A card type, subtype or supertype name (Forge's lists, type_constants.h).
static bool known_type_name(const std::string &name) {
    return all_types.count(name) || all_subtypes.count(name) || all_supertypes.count(name);
}

std::vector<std::string> filter_unknown_qualifiers(const std::string &spec) {
    std::vector<std::string> unknown;
    const FilterSpec filter(spec);
    if (filter.empty()) return unknown;
    for (const Alternative &alt : filter.compiled()->alternatives) {
        const std::string &head = alt.head;
        if (!(head == "Card" || head == "Permanent" || head == "Spell" || head == "Self" || head == "Other" ||
                known_type_name(head) || compile_qualifier(head).read_elsewhere))
            unknown.push_back(head);
        for (size_t i = alt.head_qualifiers; i < alt.qualifiers.size(); i++) {
            const Qualifier &q = alt.qualifiers[i];
            if (!q.read_elsewhere && !q.has_reading) unknown.push_back(q.token);
        }
    }
    return unknown;
}
