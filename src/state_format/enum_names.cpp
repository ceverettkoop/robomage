#include "enum_names.h"

#include <cstdio>
#include <cstdlib>
#include <map>

#include "../classes/action.h"
#include "../classes/colors.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/zone.h"
#include "../effects/effect_kind.h"
#include "../parse.h"
#include "../pending_query.h"
#include "../type_constants.h"

// Every name function below switches over all of its enum's enumerators with no default, so an
// enumerator added without a name is a compile error here, not a silently missing string.
#pragma GCC diagnostic error "-Wswitch"

static const char *step_name(Step v);
static const char *action_category_name(ActionCategory v);
static const char *action_type_name(ActionType v);
static const char *zone_name(Zone::ZoneValue v);
static const char *seat_name(Zone::Ownership v);
static const char *color_name(Colors v);
static const char *mandatory_choice_name(MandatoryChoice v);
static const char *day_night_name(Game::DayNight v);
static const char *pending_tag_name(PendingQuery::Tag v);
static const char *window_kind_name(Game::RevealWindow::Kind v);
static const char *window_end_name(Game::RevealWindow::End v);
static const char *fire_kind_name(DelayedTriggerLink::FireKind v);
template <class E>
static rmstate::EnumTable enum_table(const char *name, int first, int last, const char *(*namer)(E));
static rmstate::EnumTable effect_kind_table();
static rmstate::EnumTable string_set_table(const char *name, const std::set<std::string> &values);
static rmstate::EnumTable keyword_table();
static const std::map<std::string, int> &index_of(const std::set<std::string> &values);

#define RMSTATE_NAME_CASE(scope, e) \
    case scope e:                   \
        return #e;

std::string action_category_enum_name(ActionCategory v) {
    return action_category_name(v);
}

static const char *step_name(Step v) {
#define C(e) RMSTATE_NAME_CASE(, e)
    switch (v) {
        C(UNTAP)
        C(UPKEEP)
        C(DRAW)
        C(FIRST_MAIN)
        C(BEGIN_COMBAT)
        C(DECLARE_ATTACKERS)
        C(DECLARE_BLOCKERS)
        C(FIRST_STRIKE_DAMAGE)
        C(COMBAT_DAMAGE)
        C(END_OF_COMBAT)
        C(SECOND_MAIN)
        C(END_STEP)
        C(CLEANUP)
    }
#undef C
    return nullptr;
}

static const char *action_category_name(ActionCategory v) {
#define C(e) RMSTATE_NAME_CASE(ActionCategory::, e)
    switch (v) {
        C(PASS_PRIORITY)
        C(SELECT_ATTACKER)
        C(CONFIRM_ATTACKERS)
        C(SELECT_BLOCKER)
        C(CONFIRM_BLOCKERS)
        C(ACTIVATE_ABILITY)
        C(CAST_SPELL)
        C(SELECT_TARGET)
        C(PLAY_LAND)
        C(OTHER_CHOICE)
        C(MULLIGAN)
        C(BOTTOM_DECK_CARD)
        C(MANA_W)
        C(MANA_U)
        C(MANA_B)
        C(MANA_R)
        C(MANA_G)
        C(MANA_C)
        C(SEARCH_LIBRARY)
        C(TOP_LIBRARY)
        C(SHUFFLE)
        C(PAYING_COSTS)
        C(DIG_CHOICE)
        C(SIDEBOARD_IN)
        C(SIDEBOARD_OUT)
        C(SIDEBOARD_DONE)
        C(SACRIFICE_PERMANENT)
        C(RETURN_PERMANENT)
        C(CHOOSE_X)
        C(DISCARD)
        C(CHOOSE_MODE)
        C(CHOOSE_MANA_COLOR)
        C(PAY_UNLESS)
        C(NAME_CARD)
        C(CHOOSE_TYPE)
        C(KEEP_LEGEND)
        C(ORDER_TRIGGERS)
        C(CHOOSE_REPLACEMENT)
        C(ATTACK_TARGET)
        C(BLOCK_TARGET)
        C(OPTIONAL_YESNO)
        C(SYLVAN_CHOICE)
        C(CHOOSE_CARD)
        C(ASSIGN_DAMAGE)
        C(COMPANION)
        C(DONT_SHUFFLE)
        C(KEEP_HAND)
        C(EXILE_FROM_YARD)
        C(CONFIRM_MODES)
    }
#undef C
    return nullptr;
}

static const char *action_type_name(ActionType v) {
#define C(e) RMSTATE_NAME_CASE(, e)
    switch (v) {
        C(PASS_PRIORITY)
        C(CAST_SPELL)
        C(ACTIVATE_ABILITY)
        C(SPECIAL_ACTION)
    }
#undef C
    return nullptr;
}

static const char *zone_name(Zone::ZoneValue v) {
#define C(e) RMSTATE_NAME_CASE(Zone::, e)
    switch (v) {
        C(LIBRARY)
        C(BATTLEFIELD)
        C(HAND)
        C(STACK)
        C(GRAVEYARD)
        C(EXILE)
        C(SIDEBOARD)
    }
#undef C
    return nullptr;
}

static const char *seat_name(Zone::Ownership v) {
#define C(e) RMSTATE_NAME_CASE(Zone::, e)
    switch (v) {
        C(UNKNOWN)
        C(PLAYER_A)
        C(PLAYER_B)
    }
#undef C
    return nullptr;
}

static const char *color_name(Colors v) {
#define C(e) RMSTATE_NAME_CASE(, e)
    switch (v) {
        C(WHITE)
        C(BLUE)
        C(BLACK)
        C(RED)
        C(GREEN)
        C(COLORLESS)
        C(GENERIC)
        C(NO_COLOR)
    }
#undef C
    return nullptr;
}

static const char *mandatory_choice_name(MandatoryChoice v) {
#define C(e) RMSTATE_NAME_CASE(, e)
    switch (v) {
        C(NONE)
        C(DECLARE_ATTACKERS_CHOICE)
        C(DECLARE_BLOCKERS_CHOICE)
        C(CLEANUP_DISCARD)
        C(CHOOSE_ENTITY)
        C(ASSIGN_COMBAT_DAMAGE_CHOICE)
    }
#undef C
    return nullptr;
}

static const char *day_night_name(Game::DayNight v) {
#define C(e) RMSTATE_NAME_CASE(Game::, e)
    switch (v) {
        C(DN_NEITHER)
        C(DN_DAY)
        C(DN_NIGHT)
    }
#undef C
    return nullptr;
}

static const char *pending_tag_name(PendingQuery::Tag v) {
#define C(e) RMSTATE_NAME_CASE(PendingQuery::, e)
    switch (v) {
        C(NONE)
        C(RESOLUTION)
        C(SBE_LATCHED)
        C(TURN_DRAW)
        C(CAST)
        C(ACTIVATION)
        C(ATTACK_TARGET)
        C(BLOCK_TARGET)
        C(DAMAGE_ASSIGN)
        C(TRIGGER_PLACE)
    }
#undef C
    return nullptr;
}

static const char *window_kind_name(Game::RevealWindow::Kind v) {
#define C(e) RMSTATE_NAME_CASE(Game::RevealWindow::, e)
    switch (v) {
        C(REVEAL)
        C(LOOK)
        C(FACE_DOWN_LOOK)
        C(CONTINUOUS_REVEAL)
    }
#undef C
    return nullptr;
}

static const char *window_end_name(Game::RevealWindow::End v) {
#define C(e) RMSTATE_NAME_CASE(Game::RevealWindow::, e)
    switch (v) {
        C(RESOLUTION)
        C(STACK_OBJECT)
        C(LEAVES_ZONE)
    }
#undef C
    return nullptr;
}

// FIRE_OTHER (-1) is no value of the format's enum: a fire kind the observation has no column for
// travels as none.
static const char *fire_kind_name(DelayedTriggerLink::FireKind v) {
#define C(e) RMSTATE_NAME_CASE(DelayedTriggerLink::, e)
    switch (v) {
        C(FIRE_OTHER)
        C(FIRE_UPKEEP)
        C(FIRE_END_STEP)
        C(FIRE_END_OF_COMBAT)
        C(FIRE_LEAVES_BATTLEFIELD)
    }
#undef C
    return nullptr;
}

// The enum `name` over the values first..last of E, named by `namer`; a value with no name (a gap
// in E's values) is fatal, so the table can never be shorter than the enum.
template <class E>
static rmstate::EnumTable enum_table(const char *name, int first, int last, const char *(*namer)(E)) {
    rmstate::EnumTable t;
    t.name = name;
    for (int v = first; v <= last; ++v) {
        const char *n = namer(static_cast<E>(v));
        if (!n) {
            fprintf(stderr, "ERROR: rmstate enum %s: value %d has no name\n", name, v);
            abort();
        }
        t.values.push_back(n);
    }
    return t;
}

// EffectKind: None, then the categories in effect_kinds.def order (the enum's own order).
static rmstate::EnumTable effect_kind_table() {
    rmstate::EnumTable t;
    t.name = "effect_kind";
    t.values.push_back("None");
#define EFFECT_KIND(kind, category, handler) t.values.push_back(category);
#define EFFECT_KIND_ELSEWHERE(kind, category) t.values.push_back(category);
#include "../effects/effect_kinds.def"
#undef EFFECT_KIND
#undef EFFECT_KIND_ELSEWHERE
    return t;
}

static rmstate::EnumTable string_set_table(const char *name, const std::set<std::string> &values) {
    rmstate::EnumTable t;
    t.name = name;
    t.values.assign(values.begin(), values.end());
    return t;
}

// The evergreen keywords the keyword table always names, in this order after the parser's.
static constexpr const char *EVERGREEN_KEYWORDS[] = {"Flying", "Reach", "First Strike", "Double Strike", "Deathtouch",
    "Lifelink", "Trample", "Vigilance", "Menace", "Haste", "Defender", "Indestructible", "Hexproof", "Shroud", "Ward",
    "Flash"};

// The parser's keyword table (parse.cpp's kKeywordTable), then the evergreen keywords
// (EVERGREEN_KEYWORDS) that the table has no entry for.
static rmstate::EnumTable keyword_table() {
    rmstate::EnumTable t;
    t.name = "keyword";
    t.values = parser_keyword_names();
    for (const char *kw : EVERGREEN_KEYWORDS) {
        bool present = false;
        for (const std::string &v : t.values) present = present || v == kw;
        if (!present) t.values.push_back(kw);
    }
    return t;
}

std::vector<rmstate::EnumTable> engine_enums() {
    std::vector<rmstate::EnumTable> out;
    out.push_back(enum_table("step", UNTAP, CLEANUP, step_name));
    out.push_back(enum_table("action_category", 0, ACTION_CATEGORY_MAX, action_category_name));
    out.push_back(enum_table("action_type", PASS_PRIORITY, SPECIAL_ACTION, action_type_name));
    out.push_back(enum_table("zone", Zone::LIBRARY, Zone::SIDEBOARD, zone_name));
    out.push_back(enum_table("seat", Zone::UNKNOWN, Zone::PLAYER_B, seat_name));
    out.push_back(enum_table("color", WHITE, NO_COLOR, color_name));
    out.push_back(enum_table("mandatory_choice", NONE, N_MANDATORY_CHOICES - 1, mandatory_choice_name));
    out.push_back(enum_table("day_night", Game::DN_NEITHER, Game::DN_NIGHT, day_night_name));
    out.push_back(enum_table("pending_tag", PendingQuery::NONE, PendingQuery::TRIGGER_PLACE, pending_tag_name));
    out.push_back(
        enum_table("window_kind", Game::RevealWindow::REVEAL, Game::RevealWindow::CONTINUOUS_REVEAL, window_kind_name));
    out.push_back(
        enum_table("window_end", Game::RevealWindow::RESOLUTION, Game::RevealWindow::LEAVES_ZONE, window_end_name));
    out.push_back(enum_table(
        "fire_kind", DelayedTriggerLink::FIRE_UPKEEP, DelayedTriggerLink::FIRE_LEAVES_BATTLEFIELD, fire_kind_name));
    out.push_back(effect_kind_table());
    out.push_back(string_set_table("card_type", all_types));
    out.push_back(string_set_table("subtype", all_subtypes));
    out.push_back(string_set_table("supertype", all_supertypes));
    out.push_back(keyword_table());
    return out;
}

std::vector<rmstate::EnumTable> all_enums() {
    std::vector<rmstate::EnumTable> out = rmstate::format_enums();
    for (rmstate::EnumTable &t : engine_enums()) out.push_back(std::move(t));
    return out;
}

// Name → index in one of the sorted type lists, built once.
static const std::map<std::string, int> &index_of(const std::set<std::string> &values) {
    static std::map<const std::set<std::string> *, std::map<std::string, int>> cache;
    auto it = cache.find(&values);
    if (it != cache.end()) return it->second;
    std::map<std::string, int> &m = cache[&values];
    int i = 0;
    for (const std::string &v : values) m.emplace(v, i++);
    return m;
}

int card_type_index(const std::string &name) {
    const auto &m = index_of(all_types);
    auto it = m.find(name);
    return it == m.end() ? -1 : it->second;
}

int subtype_index(const std::string &name) {
    const auto &m = index_of(all_subtypes);
    auto it = m.find(name);
    return it == m.end() ? -1 : it->second;
}

int supertype_index(const std::string &name) {
    const auto &m = index_of(all_supertypes);
    auto it = m.find(name);
    return it == m.end() ? -1 : it->second;
}

uint32_t card_type_mask(const std::set<Type> &types) {
    uint32_t mask = 0;
    for (const Type &t : types) {
        if (t.kind != TYPE) continue;
        const int i = card_type_index(t.name);
        if (i >= 0 && i < 32) mask |= 1u << i;
    }
    return mask;
}

uint8_t supertype_mask(const std::set<Type> &types) {
    uint32_t mask = 0;
    for (const Type &t : types) {
        if (t.kind != SUPERTYPE) continue;
        const int i = supertype_index(t.name);
        if (i >= 0 && i < 8) mask |= 1u << i;
    }
    return static_cast<uint8_t>(mask);
}

std::vector<int> subtype_indices(const std::set<Type> &types) {
    std::vector<int> out;
    for (const Type &t : types) {
        if (t.kind != SUBTYPE) continue;
        const int i = subtype_index(t.name);
        if (i >= 0) out.push_back(i);
    }
    return out;
}

int keyword_index(const std::string &keyword) {
    static const std::vector<std::string> names = keyword_table().values;
    int best = -1;
    size_t best_len = 0;
    for (size_t i = 0; i < names.size(); ++i) {
        const std::string &n = names[i];
        if (keyword.compare(0, n.size(), n) != 0) continue;
        const bool whole = keyword.size() == n.size() || keyword[n.size()] == ':' || keyword[n.size()] == ' ';
        if (whole && n.size() > best_len) {
            best = static_cast<int>(i);
            best_len = n.size();
        }
    }
    return best;
}
