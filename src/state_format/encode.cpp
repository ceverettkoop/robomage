#include "encode.h"

#include <algorithm>
#include <fstream>
#include <map>
#include <set>

#include "../card_db.h"
#include "../card_vocab.h"
#include "../classes/deck_state.h"
#include "../classes/game.h"
#include "../classes/match_context.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/spell.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../effects/effects.h"
#include "../game_driver.h"
#include "../mana_system.h"
#include "../parse.h"
#include "../queries/activation.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/combat.h"
#include "../queries/delayed_triggers.h"
#include "../queries/keywords.h"
#include "../queries/lki.h"
#include "../queries/pending_cost.h"
#include "../queries/player_effects.h"
#include "../queries/players.h"
#include "../queries/spells.h"
#include "../queries/types.h"
#include "../queries/visibility.h"
#include "../queries/zones.h"
#include "../systems/rules_modifying.h"
#include "../type_constants.h"
#include "enum_names.h"
#include "handles.h"
#include "knowledge_feed.h"
#include "metagame.h"
#include "schema.h"
#include "stream.h"

using rmstate::FrameKind;
using rmstate::FrameWriter;
using rmstate::NONE16;
using rmstate::NONE8;
using rmstate::TableId;
using Row = rmstate::FrameWriter::Row;

namespace {

// One frame's viewer: `viewer` as the handle tables and visibility take it, `self` the seat the
// relative columns are relative to (Player A for the omniscient viewer, register 31).
struct Ctx {
        Zone::Ownership viewer;
        Zone::Ownership self;
        FrameWriter &w;
};

// One entry of a card's ability list (card_ability rows, ability references): every ability of
// both faces in script order — spell / activated / triggered, gift, chapter and opening-hand
// definitions, then statics and replacements.
struct CardAbility {
        rmstate::Face face = rmstate::Face::FRONT;
        rmstate::AbilityKind kind = rmstate::AbilityKind::SPELL;
        const AbilityDef *def = nullptr;
        const StaticAbility *stat = nullptr;
        const Effect::Replacement *repl = nullptr;
};

// The zone counts of one seat (player table).
struct ZoneCounts {
        int hand = 0;
        int library = 0;
        int graveyard = 0;
        int exile = 0;
        int sideboard = 0;
};

}  // namespace

static rmstate::Viewer frame_viewer(Zone::Ownership viewer);
static Zone::Ownership viewer_key(Zone::Ownership viewer);
static bool is_seat(Zone::Ownership p);
static void put_seat(Row &r, Zone::Ownership seat, Zone::Ownership self);
static uint64_t card_or_none(int id);
static uint64_t u8_count(int64_t v);
static uint64_t u16_count(int64_t v);
static uint32_t handle(const Ctx &c, Entity e);
static uint8_t color_mask(const std::set<Colors> &colors);
static int count_color(const ManaValue &mana, Colors c);
static std::string printed_mana_cost(const CardData &cd);
static std::string cost_text(const std::string &script_line);
static rmstate::Layout card_layout(const CardData &cd);
static rmstate::Timing ability_timing(const AbilityDef &def, const CardData *cd);
static std::vector<CardAbility> card_ability_list(const CardData &front);
static std::vector<CardAbility> token_ability_list(const Token &tok);
static void append_face_abilities(std::vector<CardAbility> &out, const CardData &face, rmstate::Face which);
static void append_defs(std::vector<CardAbility> &out, const std::vector<const AbilityDef *> &defs, rmstate::Face face,
    rmstate::AbilityKind kind_override, bool use_override);
static rmstate::AbilityKind def_kind(const AbilityDef &def);
static void put_abref(Row &r, const AbilityDef *def, int card_id, Entity token_object);
static const CardData *object_card_data(Entity e);
static const CardData *card_definition(int card_id);
static void collect_token_scripts(const AbilityDef &def, std::set<std::string> &out);
static void collect_card_tokens(const CardData &cd, std::set<std::string> &out);
static bool token_script_exists(const std::string &stem);
static std::vector<int> cards_frame_ids(Zone::Ownership viewer);
static bool sees_decklist(Zone::Ownership viewer, Zone::Ownership owner);
static void write_card_face(
    FrameWriter &w, int card_id, const CardData &face, rmstate::Face which, int face_card, const std::string &script);
static void write_card_abilities(FrameWriter &w, int card_id, const std::vector<CardAbility> &list, const CardData *cd);
static void write_token_card(FrameWriter &w, int card_id, const Token &tok);
static Zone::Ownership plays_first_seat();
static int match_game_index();
static void write_header(const Ctx &c, bool terminal);
static void write_turn(const Ctx &c);
static void write_players(const Ctx &c, const ZoneCounts counts[3]);
static void write_object(const Ctx &c, Entity e, uint32_t h, int stack_pos);
static void write_object_children(const Ctx &c, Entity e, uint32_t h);
static void write_provenance(const Ctx &c, Entity e, uint32_t h);
static void write_combat(const Ctx &c, const std::vector<Entity> &battlefield);
static int write_lki(const Ctx &c, Entity departed);
static void write_stack(const Ctx &c, const std::vector<Entity> &stack);
static rmstate::StackKind stack_kind(Entity e, const Spell *sp, const Ability *ab);
static void write_stack_targets(const Ctx &c, Entity stack_object, uint32_t h, const Ability &ab);
static void write_delayed_triggers(const Ctx &c, const std::vector<Entity> &stack);
static void write_rules_state(const Ctx &c, const std::vector<Entity> &zone_cards);
static void write_player_effects(const Ctx &c, Zone::Ownership p);
static void write_reveals(const Ctx &c);
static void write_pending(const Ctx &c, Zone::Ownership decider);
static void write_pending_cost(const Ctx &c);
static void write_menu(const Ctx &c, const std::vector<LegalAction> &actions);
static bool description_hides_identity(const Ctx &c, const LegalAction &la);
static uint16_t cast_variant_mask(const LegalAction &la);
static void write_match(const Ctx &c);
static void write_own_deck(const Ctx &c);
static void write_events(const Ctx &c, const std::vector<ObsEvent> &events);
static void put_libpos(Row &r, const ObsLibPos &p);
static uint64_t zone_or_none(int zone);
static rmstate::Duration effect_duration(bool until_your_next_turn);

// ── Small helpers ─────────────────────────────────────────────────────────────────────────────

static bool is_seat(Zone::Ownership p) {
    return p == Zone::PLAYER_A || p == Zone::PLAYER_B;
}

static Zone::Ownership viewer_key(Zone::Ownership viewer) {
    return is_seat(viewer) ? viewer : Zone::UNKNOWN;
}

static rmstate::Viewer frame_viewer(Zone::Ownership viewer) {
    if (viewer == Zone::PLAYER_A) return rmstate::Viewer::A;
    if (viewer == Zone::PLAYER_B) return rmstate::Viewer::B;
    return rmstate::Viewer::OMNISCIENT;
}

// A seat pair (Appendix A.1): the absolute seat, then 0 none / 1 self / 2 opponent.
static void put_seat(Row &r, Zone::Ownership seat, Zone::Ownership self) {
    if (!is_seat(seat)) {
        r.u8(0).u8(0);
        return;
    }
    r.u8(static_cast<uint64_t>(seat)).u8(seat == self ? 1 : 2);
}

static uint64_t card_or_none(int id) {
    return id < 0 || id > 0xFFFE ? NONE16 : static_cast<uint64_t>(id);
}

static uint64_t u8_count(int64_t v) {
    return static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(v, 0xFF)));
}

static uint64_t u16_count(int64_t v) {
    return static_cast<uint64_t>(std::max<int64_t>(0, std::min<int64_t>(v, 0xFFFF)));
}

static uint32_t handle(const Ctx &c, Entity e) {
    return e == 0 ? 0 : handle_of(c.viewer, e);
}

static uint8_t color_mask(const std::set<Colors> &colors) {
    uint32_t m = 0;
    for (Colors col : colors)
        if (col >= WHITE && col <= NO_COLOR) m |= 1u << static_cast<int>(col);
    return static_cast<uint8_t>(m);
}

static int count_color(const ManaValue &mana, Colors c) {
    return static_cast<int>(mana.count(c));
}

// A card's mana cost as printed: its {X} pips, the plain pips (generic first), then the hybrid and
// Phyrexian pips; empty for a card with no mana cost (a land).
static std::string printed_mana_cost(const CardData &cd) {
    std::string out;
    for (int i = 0; i < cd.x_pip_count; ++i) out += "{X}";
    const bool other_pips = cd.x_pip_count > 0 || !cd.hybrid_mana.empty() || !cd.phyrexian_mana.empty();
    if (!cd.mana_cost.empty() || (!other_pips && !is_land_card(cd))) out += mana_value_text(cd.mana_cost);
    for (const HybridPip &pip : cd.hybrid_mana) {
        out += "{";
        if (pip.generic_alt > 0) out += std::to_string(pip.generic_alt);
        for (size_t i = 0; i < pip.colors.size(); ++i) {
            if (i > 0 || pip.generic_alt > 0) out += "/";
            out += mana_symbol_str(pip.colors[i]);
        }
        out += "}";
    }
    for (Colors c : cd.phyrexian_mana) out += std::string("{") + mana_symbol_str(c) + "/P}";
    return out;
}

// The cost of a script ability line: its Cost$ ("A:AB$ Draw | Cost$ 2 T | …" → "2 T"), or a
// keyword's argument ("K:Equip:1 R" → "1 R"); empty when it has none.
static std::string cost_text(const std::string &script_line) {
    if (script_line.rfind("K:", 0) == 0) {
        const size_t colon = script_line.find(':', 2);
        return colon == std::string::npos ? "" : script_line.substr(colon + 1);
    }
    const std::string key = "Cost$ ";
    const size_t at = script_line.find(key);
    if (at == std::string::npos) return "";
    const size_t start = at + key.size();
    size_t end = script_line.find('|', start);
    if (end == std::string::npos) end = script_line.size();
    while (end > start && script_line[end - 1] == ' ') --end;
    return script_line.substr(start, end - start);
}

static rmstate::Layout card_layout(const CardData &cd) {
    if (cd.is_split) return rmstate::Layout::SPLIT;
    if (cd.is_modal_dfc) return rmstate::Layout::MODAL_DFC;
    if (cd.backside) return rmstate::Layout::TRANSFORM;
    return rmstate::Layout::NORMAL;
}

// When an ability may be used: a spell at instant speed when it is an instant or has flash, else
// at sorcery speed; an activated ability at sorcery speed when its definition says so (a loyalty
// ability, SorcerySpeed$, an Equip-style keyword), else at instant speed; anything else any time.
static rmstate::Timing ability_timing(const AbilityDef &def, const CardData *cd) {
    if (def.ability_type == AbilityDef::SPELL) {
        if (cd && (card_has_type(*cd, "Instant") || card_has_keyword(*cd, "Flash"))) return rmstate::Timing::INSTANT;
        return rmstate::Timing::SORCERY;
    }
    if (def.ability_type == AbilityDef::ACTIVATED) {
        if (def.sorcery_speed_only || def.is_loyalty_ability || def.keyword_label == "Equip" ||
            def.keyword_label == "Reconfigure" || def.keyword_label == "Unattach")
            return rmstate::Timing::SORCERY;
        return rmstate::Timing::INSTANT;
    }
    return rmstate::Timing::ANY;
}

static rmstate::AbilityKind def_kind(const AbilityDef &def) {
    switch (def.ability_type) {
        case AbilityDef::TRIGGERED:
            return rmstate::AbilityKind::TRIGGERED;
        case AbilityDef::ACTIVATED:
            return rmstate::AbilityKind::ACTIVATED;
        case AbilityDef::SPELL:
            return rmstate::AbilityKind::SPELL;
    }
    return rmstate::AbilityKind::SPELL;
}

static void append_defs(std::vector<CardAbility> &out, const std::vector<const AbilityDef *> &defs, rmstate::Face face,
    rmstate::AbilityKind kind_override, bool use_override) {
    for (const AbilityDef *d : defs) {
        CardAbility a;
        a.face = face;
        a.def = d;
        a.kind = use_override ? kind_override : def_kind(*d);
        out.push_back(a);
    }
}

static void append_face_abilities(std::vector<CardAbility> &out, const CardData &face, rmstate::Face which) {
    append_defs(out, face.abilities, which, rmstate::AbilityKind::SPELL, false);
    append_defs(out, face.gift_abilities, which, rmstate::AbilityKind::SPELL, true);
    append_defs(out, face.saga_chapters, which, rmstate::AbilityKind::CHAPTER, true);
    append_defs(out, face.opening_hand_abilities, which, rmstate::AbilityKind::OPENING_HAND, true);
    for (const StaticAbility &s : face.static_abilities) {
        CardAbility a;
        a.face = which;
        a.kind = rmstate::AbilityKind::STATIC;
        a.stat = &s;
        out.push_back(a);
    }
    for (const Effect::Replacement &r : face.replacement_effects) {
        CardAbility a;
        a.face = which;
        a.kind = rmstate::AbilityKind::REPLACEMENT;
        a.repl = &r;
        out.push_back(a);
    }
}

static std::vector<CardAbility> card_ability_list(const CardData &front) {
    std::vector<CardAbility> out;
    append_face_abilities(out, front, rmstate::Face::FRONT);
    if (front.backside) append_face_abilities(out, *front.backside, rmstate::Face::BACK);
    return out;
}

static std::vector<CardAbility> token_ability_list(const Token &tok) {
    std::vector<CardAbility> out;
    append_defs(out, tok.abilities, rmstate::Face::FRONT, rmstate::AbilityKind::SPELL, false);
    for (const StaticAbility &s : tok.static_abilities) {
        CardAbility a;
        a.kind = rmstate::AbilityKind::STATIC;
        a.stat = &s;
        out.push_back(a);
    }
    for (const Effect::Replacement &r : tok.replacement_effects) {
        CardAbility a;
        a.kind = rmstate::AbilityKind::REPLACEMENT;
        a.repl = &r;
        out.push_back(a);
    }
    return out;
}

// The card definition an object's abilities come from: its CardData (a copy's copied
// characteristics), or null for a token or an object with none.
static const CardData *object_card_data(Entity e) {
    if (e == 0 || !global_coordinator.entity_has_component<CardData>(e)) return nullptr;
    return &global_coordinator.GetComponent<CardData>(e);
}

// The loaded definition of vocab card `card_id` (the card_db template every copy of it shares), or
// null for a token or a card not loaded this game.
static const CardData *card_definition(int card_id) {
    if (card_id < 0 || card_id >= TOKEN_VOCAB_BASE) return nullptr;
    auto it = card_db.find(name_to_uid(card_index_to_name(card_id)));
    if (it == card_db.end()) return nullptr;
    return object_card_data(it->second);
}

// An ability reference (Appendix A.1): `def`'s index in the ability list of card `card_id` (read
// from its definition, never from an object the viewer may not see; a token's from the live token
// `token_object`); an ability that is not one of them (a granted or engine-built one) has no card
// and index, only its key.
static void put_abref(Row &r, const AbilityDef *def, int card_id, Entity token_object) {
    if (!def || def == blank_ability_def()) {
        r.u16(NONE16).u16(NONE16).text("");
        return;
    }
    std::vector<CardAbility> list;
    if (const CardData *cd = card_definition(card_id))
        list = card_ability_list(*cd);
    else if (card_id >= TOKEN_VOCAB_BASE && token_object != 0 &&
             global_coordinator.entity_has_component<Token>(token_object))
        list = token_ability_list(global_coordinator.GetComponent<Token>(token_object));
    for (size_t i = 0; i < list.size(); ++i) {
        if (list[i].def != def) continue;
        r.u16(card_or_none(card_id)).u16(i).text(def->ref_key);
        return;
    }
    std::string key = def->ref_key;
    if (key.empty()) key = !def->keyword_label.empty() ? def->keyword_label : def->derived_from_keyword;
    if (key.empty() && def->subtype_derived) key = "LandType";
    r.u16(NONE16).u16(NONE16).text(key);
}

// ── CARDS ─────────────────────────────────────────────────────────────────────────────────────

// The token scripts `def` (and its sub-abilities, modes and granted abilities) can make.
static void collect_token_scripts(const AbilityDef &def, std::set<std::string> &out) {
    if (const TokenParams *tp = std::get_if<TokenParams>(&def.params))
        if (!tp->script.empty()) out.insert(tp->script);
    if (def.kind == EffectKind::Investigate) out.insert(effects::INVESTIGATE_TOKEN_SCRIPT);
    if (def.kind == EffectKind::Amass) out.insert(effects::amass_army_token_script(effects::amass_subtype(def)));
    if (def.kind == EffectKind::Mobilize) out.insert(effects::MOBILIZE_TOKEN_SCRIPT);
    for (const AbilityDef &sub : def.subabilities) collect_token_scripts(sub, out);
    for (const AbilityDef &mode : def.charm_choices) collect_token_scripts(mode, out);
    for (const AbilityDef &trig : def.effect_floating_triggers) collect_token_scripts(trig, out);
    for (const AbilityDef &granted : def.animate_granted_abilities) collect_token_scripts(granted, out);
}

// The token scripts a card can make: its abilities' token effects, and the Warrior a mobilize
// keyword makes (its trigger is built from the keyword during play, not parsed onto the card).
static void collect_card_tokens(const CardData &cd, std::set<std::string> &out) {
    for (const CardAbility &a : card_ability_list(cd))
        if (a.def) collect_token_scripts(*a.def, out);
    const CardData *faces[] = {&cd, cd.backside.get()};
    for (const CardData *face : faces) {
        if (!face) continue;
        for (const std::string &kw : face->keywords)
            if (kw.rfind("Mobilize", 0) == 0) out.insert(effects::MOBILIZE_TOKEN_SCRIPT);
    }
}

static bool token_script_exists(const std::string &stem) {
    std::ifstream f(RESOURCE_DIR + "/tokenscripts/" + stem + ".txt");
    return f.is_open();
}

// Whether `viewer` may see `owner`'s registered list (decision 11: both lists are open unless
// --closed-decklist, which leaves a seat only its own; the omniscient viewer sees both).
static bool sees_decklist(Zone::Ownership viewer, Zone::Ownership owner) {
    return !closed_decklist || !is_seat(viewer) || viewer == owner;
}

// The card ids of the cards in the registered lists `viewer` may see and in the reference
// metagame's lists (physical_card_of: a deck line naming a back face is its card's front-face id),
// ascending. Under a closed decklist, every vocab card (by its front face), whoever views: the
// opponent's cards need rows, and listing only the viewer's own would show which cards are the
// opponent's.
static std::vector<int> cards_frame_ids(Zone::Ownership viewer) {
    std::set<int> ids;
    if (closed_decklist) {
        for (int i = 0; i < CARD_VOCAB_SIZE; ++i) {
            const CardData *cd = card_data_by_name(card_vocab_entries[i].name);
            if (cd != nullptr && cd->name == card_vocab_entries[i].name) ids.insert(card_vocab_entries[i].index);
        }
        return std::vector<int>(ids.begin(), ids.end());
    }
    for (Zone::Ownership owner : {Zone::PLAYER_A, Zone::PLAYER_B}) {
        if (!sees_decklist(viewer, owner)) continue;
        for (const DecklistEntry &e : deck_state_registered_main(owner)) ids.insert(physical_card_of(e.vocab_idx));
        for (const DecklistEntry &e : deck_state_registered_side(owner)) ids.insert(physical_card_of(e.vocab_idx));
    }
    for (const ReferenceList &list : reference_metagame().lists)
        for (const auto &c : list.cards) ids.insert(c.first);
    return std::vector<int>(ids.begin(), ids.end());
}

// `script` is the card's raw script text (--rmstate-card-scripts, on its front row), else "".
static void write_card_face(
    FrameWriter &w, int card_id, const CardData &face, rmstate::Face which, int face_card, const std::string &script) {
    const uint64_t f = static_cast<uint64_t>(which);
    {
        Row r = w.row(TableId::CARD);
        r.u16(card_or_none(card_id)).u8(f).u16(card_or_none(face_card)).text(face.name).text(face.uid).u8(0);
        r.u8(static_cast<uint64_t>(card_layout(face)))
            .text(printed_mana_cost(face))
            .u8(u8_count(card_mana_value(face)));
        for (Colors col : {WHITE, BLUE, BLACK, RED, GREEN, COLORLESS, GENERIC})
            r.u8(u8_count(count_color(face.mana_cost, col)));
        r.u8(u8_count(face.x_pip_count));
        r.u8(color_mask(card_colors(face))).u32(card_type_mask(face.types)).u8(supertype_mask(face.types));
        r.i16(static_cast<int64_t>(face.power))
            .i16(static_cast<int64_t>(face.toughness))
            .u8(u8_count(face.starting_loyalty));
        r.text(face.oracle_text).text(script);
    }
    for (int st : subtype_indices(face.types))
        w.row(TableId::CARD_SUBTYPE).u16(card_or_none(card_id)).u8(f).u16(static_cast<uint64_t>(st));
    for (const std::string &kw : face.keywords) {
        const int k = keyword_index(kw);
        w.row(TableId::CARD_KEYWORD)
            .u16(card_or_none(card_id))
            .u8(f)
            .u8(k < 0 ? NONE8 : static_cast<uint64_t>(k))
            .text(kw);
    }
    for (const HybridPip &pip : face.hybrid_mana) {
        const rmstate::PipKind kind = pip.generic_alt > 0 ? rmstate::PipKind::TWOBRID : rmstate::PipKind::HYBRID;
        w.row(TableId::CARD_PIP)
            .u16(card_or_none(card_id))
            .u8(f)
            .u8(static_cast<uint64_t>(kind))
            .u8(color_mask(std::set<Colors>(pip.colors.begin(), pip.colors.end())))
            .u8(u8_count(pip.generic_alt));
    }
    for (Colors c : face.phyrexian_mana)
        w.row(TableId::CARD_PIP)
            .u16(card_or_none(card_id))
            .u8(f)
            .u8(static_cast<uint64_t>(rmstate::PipKind::PHYREXIAN))
            .u8(color_mask({c}))
            .u8(0);
}

static void write_card_abilities(
    FrameWriter &w, int card_id, const std::vector<CardAbility> &list, const CardData *cd) {
    for (size_t i = 0; i < list.size(); ++i) {
        const CardAbility &a = list[i];
        Row r = w.row(TableId::CARD_ABILITY);
        r.u16(card_or_none(card_id)).u16(i).u8(static_cast<uint64_t>(a.face)).u8(static_cast<uint64_t>(a.kind));
        if (a.def) {
            const CardData *face = cd && a.face == rmstate::Face::BACK && cd->backside ? cd->backside.get() : cd;
            r.u8(static_cast<uint64_t>(a.def->kind)).text(a.def->ref_key).text(cost_text(a.def->script_line));
            r.u8(a.def->tap_cost ? 1 : 0).u8(static_cast<uint64_t>(ability_timing(*a.def, face)));
            r.u8(u8_count(a.def->activation_limit)).text(a.def->script_line);
        } else {
            const uint32_t line = a.stat ? a.stat->script_line : a.repl->script_line;
            r.u8(NONE8).text("").text("").u8(0).u8(static_cast<uint64_t>(rmstate::Timing::ANY)).u8(0);
            r.text(script_line_text(line));
        }
    }
}

static void write_token_card(FrameWriter &w, int card_id, const Token &tok) {
    {
        Row r = w.row(TableId::CARD);
        r.u16(card_or_none(card_id)).u8(static_cast<uint64_t>(rmstate::Face::FRONT)).u16(card_or_none(card_id));
        r.text(tok.name).text(tok.script_name).u8(1).u8(static_cast<uint64_t>(rmstate::Layout::NORMAL)).text("");
        r.u8(u8_count(tok.mana_value));
        for (int i = 0; i < 8; ++i) r.u8(0);
        r.u8(color_mask(tok.explicit_colors)).u32(card_type_mask(tok.types)).u8(supertype_mask(tok.types));
        r.i16(static_cast<int64_t>(tok.power)).i16(static_cast<int64_t>(tok.toughness)).u8(0);
        r.text(tok.oracle_text).text(rmstate_card_scripts ? token_script_text(tok.script_name) : std::string());
    }
    for (int st : subtype_indices(tok.types))
        w.row(TableId::CARD_SUBTYPE)
            .u16(card_or_none(card_id))
            .u8(static_cast<uint64_t>(rmstate::Face::FRONT))
            .u16(static_cast<uint64_t>(st));
    for (const std::string &kw : tok.keywords) {
        const int k = keyword_index(kw);
        w.row(TableId::CARD_KEYWORD)
            .u16(card_or_none(card_id))
            .u8(static_cast<uint64_t>(rmstate::Face::FRONT))
            .u8(k < 0 ? NONE8 : static_cast<uint64_t>(k))
            .text(kw);
    }
    write_card_abilities(w, card_id, token_ability_list(tok), nullptr);
}

// ── Snapshot ──────────────────────────────────────────────────────────────────────────────────

// The game the snapshot is about: between games, the upcoming one (as the BQUERY serializer).
static int match_game_index() {
    return sideboard_phase ? match_game_number + 1 : match_game_number;
}

static Zone::Ownership plays_first_seat() {
    const bool a_first = sideboard_phase ? g_match_ctx.a_goes_first : cur_game.pregame.a_goes_first;
    return a_first ? Zone::PLAYER_A : Zone::PLAYER_B;
}

// (game, sideboarding, decision) names a decision of the match: a game's decisions count from 1,
// and the sideboarding phase before a game (game = the upcoming one) counts its own from 1.
static void write_header(const Ctx &c, bool terminal) {
    c.w.row(TableId::HEADER)
        .u8(u8_count(match_game_index()))
        .u32(rmstate_decision_number())
        .u8(static_cast<uint64_t>(knowledge_level_of(c.viewer)))
        .u8(terminal ? 1 : 0)
        .u8(sideboard_phase ? 1 : 0);
}

static void write_turn(const Ctx &c) {
    Row r = c.w.row(TableId::TURN);
    r.u16(u16_count(static_cast<int64_t>(cur_game.turn_state.turn)))
        .u8(static_cast<uint64_t>(cur_game.turn_state.step));
    put_seat(r, active_seat(), c.self);
    put_seat(r, priority_seat(), c.self);
    r.u8(priority_window_open() ? 1 : 0).u8(static_cast<uint64_t>(cur_game.pending.choice));
    r.u8(cur_game.combat.attackers_declared ? 1 : 0).u8(cur_game.combat.blockers_declared ? 1 : 0);
    r.u8(static_cast<uint64_t>(cur_game.day_night));
    const std::vector<Zone::Ownership> &extra = cur_game.turn_state.extra_turns;
    put_seat(r, extra.empty() ? Zone::UNKNOWN : extra.back(), c.self);
}

static void write_players(const Ctx &c, const ZoneCounts counts[3]) {
    const Zone::Ownership order[] = {c.self, opponent_of(c.self)};
    const bool window = priority_window_open();
    for (Zone::Ownership p : order) {
        const Entity pe = get_player_entity(p);
        const Player &pl = global_coordinator.GetComponent<Player>(pe);
        const ZoneCounts &zc = counts[p];
        const bool is_a = p == Zone::PLAYER_A;
        Row r = c.w.row(TableId::PLAYER);
        put_seat(r, p, c.self);
        r.i32(pl.life_total);
        for (Colors col : {WHITE, BLUE, BLACK, RED, GREEN, COLORLESS})
            r.u16(u16_count(static_cast<int64_t>(pl.mana.count(col))));
        r.u16(u16_count(zc.hand)).u16(u16_count(zc.library)).u16(u16_count(zc.graveyard)).u16(u16_count(zc.exile));
        r.u16(u16_count(zc.sideboard));
        const Game::PregameState &pg = cur_game.pregame;
        const int mulls = sideboard_phase ? 0 : (is_a ? pg.mulls_a : pg.mulls_b);
        const int to_bottom = !sideboard_phase && pg.stage == Game::PregameState::MULL_BOTTOM && pg.bottoming_owner == p
                                  ? pg.bottom_remaining
                                  : 0;
        const bool passed = window && (is_a ? cur_game.priority.a_has_passed : cur_game.priority.b_has_passed);
        r.u8(u8_count(mulls)).u8(u8_count(to_bottom)).u8(passed ? 1 : 0);
        r.u8(cur_game.monarch_entity == pe ? 1 : 0).u8(pl.has_city_blessing ? 1 : 0);
        const auto &extra = cur_game.turn_state.extra_turns;
        r.u8(u8_count(std::count(extra.begin(), extra.end(), p)));
        r.u8(u8_count(pl.lands_played_this_turn)).u8(u8_count(rules_mod::land_drops_remaining(p)));
        const ManaPotential mp = mana_potential(p, zoned_entities());
        for (int i = 0; i < 6; ++i) r.u8(u8_count(mp.by_color[i]));
        r.u8(u8_count(static_cast<int64_t>(mp.sources) + static_cast<int64_t>(pl.mana.size())));
        r.u16(u16_count(static_cast<int64_t>(pl.spells_cast_this_turn)));
        r.u16(u16_count(static_cast<int64_t>(pl.noncreature_spells_cast_this_turn)));
        r.u16(u16_count(static_cast<int64_t>(pl.instant_sorcery_spells_cast_this_turn)));
        r.u16(u16_count(static_cast<int64_t>(pl.spells_cast_this_game)));
        r.u16(u16_count(static_cast<int64_t>(pl.cards_drawn_this_turn.size())));
        r.u16(u16_count(pl.life_gained_this_turn)).u16(u16_count(pl.life_lost_this_turn));
        r.u8(color_mask(pl.spell_colors_cast_this_turn));
        r.u8(revolt_this_turn(p) ? 1 : 0)
            .u8(pl.attempted_draw_from_empty ? 1 : 0)
            .u8(pl.companion_brought_to_hand ? 1 : 0);
        r.u8(u8_count(is_a ? match_wins_a : match_wins_b));
    }
    for (Zone::Ownership p : order) {
        const Player &pl = global_coordinator.GetComponent<Player>(get_player_entity(p));
        for (const auto &kv : pl.counters) {
            Row r = c.w.row(TableId::PLAYER_COUNTER);
            put_seat(r, p, c.self);
            r.text(kv.first).u16(u16_count(kv.second));
        }
    }
}

// One object row. The characteristic columns are filled only when the viewer can see the
// object's identity (a face-down object shows only where it is and its public status).
// `stack_pos` is a stack object's position from the top (0 = top), else -1.
static void write_object(const Ctx &c, Entity e, uint32_t h, int stack_pos) {
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    const bool on_bf = z.location == Zone::BATTLEFIELD && global_coordinator.entity_has_component<Permanent>(e);
    const Permanent *perm = on_bf ? &global_coordinator.GetComponent<Permanent>(e) : nullptr;
    const Creature *cr = on_bf && global_coordinator.entity_has_component<Creature>(e)
                             ? &global_coordinator.GetComponent<Creature>(e)
                             : nullptr;
    const int card = visible_card_id(e, c.viewer);
    const bool visible = identity_visible(e, c.viewer);
    const bool is_token = (perm && perm->is_token) || global_coordinator.entity_has_component<Token>(e);
    const CardData *cd = visible ? object_card_data(e) : nullptr;

    Row r = c.w.row(TableId::OBJECT);
    r.u32(h).u8(static_cast<uint64_t>(z.location));
    put_seat(r, z.owner, c.self);
    const Zone::Ownership controller = on_bf || z.location == Zone::STACK ? source_controller(e) : z.owner;
    put_seat(r, controller, c.self);
    const bool pos_known = order_visible(z.location, z.owner, c.viewer) || (z.location == Zone::LIBRARY && visible);
    r.u16(pos_known
              ? u16_count(static_cast<int64_t>(stack_pos >= 0 ? static_cast<size_t>(stack_pos) : z.distance_from_top))
              : NONE16);
    r.u16(card_or_none(card)).u8(is_token ? 1 : 0).u8(z.is_face_down ? 1 : 0);
    const bool transformed =
        visible && ((perm && perm->transformed) || (global_coordinator.entity_has_component<Spell>(e) &&
                                                       global_coordinator.GetComponent<Spell>(e).cast_back_face));
    r.u8(transformed ? 1 : 0);
    r.u16(visible && perm && perm->printed_card && cd ? card_or_none(card_name_to_index(cd->name)) : NONE16);
    // The phasing subsystem's flag is read directly: phased-out permanents are included and
    // flagged (state_format.md §6), the one exception to the battlefield accessors' rule.
    r.u8(perm && perm->is_tapped ? 1 : 0)
        .u8(on_bf && is_summoning_sick(e) ? 1 : 0)
        .u8(perm && perm->is_phased_out ? 1 : 0);
    r.u8(cr && cr->is_attacking ? 1 : 0).u8(cr && cr->is_blocking ? 1 : 0).u8(cr && cr->is_blocked ? 1 : 0);
    const Damage *dmg = on_bf && global_coordinator.entity_has_component<Damage>(e)
                            ? &global_coordinator.GetComponent<Damage>(e)
                            : nullptr;
    r.u16(dmg ? u16_count(static_cast<int64_t>(dmg->damage_counters)) : 0)
        .u8(dmg && dmg->has_deathtouch_damage ? 1 : 0);
    if (visible) {
        const Token *tok =
            global_coordinator.entity_has_component<Token>(e) ? &global_coordinator.GetComponent<Token>(e) : nullptr;
        const CardData *face = cd ? &active_face(e, *cd) : nullptr;
        int power = face ? static_cast<int>(face->power) : tok ? static_cast<int>(tok->power) : 0;
        int toughness = face ? static_cast<int>(face->toughness) : tok ? static_cast<int>(tok->toughness) : 0;
        // A battlefield creature's post-layer P/T (a phased-out one's too: effective_power treats
        // it as gone, CR 702.26b, but it is included and flagged).
        if (cr) {
            power = static_cast<int>(cr->power);
            toughness = static_cast<int>(cr->toughness);
        }
        const std::set<Type> empty_types;
        const std::set<Type> &types = perm ? perm->types : face ? face->types : tok ? tok->types : empty_types;
        int mv = 0;
        if (cd)
            mv = object_mana_value(e, mana_value_face(*cd, perm && perm->transformed));
        else if (tok)
            mv = tok->mana_value;
        r.i16(std::max(-32768, std::min(power, 32767))).i16(std::max(-32768, std::min(toughness, 32767)));
        uint32_t type_mask = card_type_mask(types);
        // A permanent is a creature exactly while it has its Creature component: one that entered
        // for its impending cost (while it has time counters) or an attached reconfigure permanent
        // keeps Creature in its type set but is not one (CR 702.175d, 702.151b).
        if (perm && !cr) type_mask &= ~(1u << card_type_index("Creature"));
        r.u32(type_mask).u8(supertype_mask(types)).u8(color_mask(effective_colors(e))).u8(u8_count(mv));
    } else {
        r.i16(0).i16(0).u32(0).u8(0).u8(0).u8(0);
    }
    r.u32(perm ? handle(c, perm->equipped_to.get()) : 0);
    r.u32(perm ? std::min<uint64_t>(perm->timestamp_entered_battlefield, 0xFFFFFFFEu) : rmstate::NONE32);
    r.u8(perm && entered_battlefield_this_turn(static_cast<long>(perm->entered_on_turn)) ? 1 : 0);
    r.u8(cr && cr->cant_be_blocked_this_turn ? 1 : 0).u8(on_bf && cur_game.combat_damage_shielded(e) ? 1 : 0);
    r.u8(on_bf && is_waiting_delayed_trigger_subject(e) ? 1 : 0);
    r.u8(perm ? u8_count(permanent_activations_this_turn(*perm)) : 0)
        .u8(on_bf ? u8_count(ability_resolutions_this_turn(e)) : 0);
    const bool named = visible && perm && !perm->chosen_name.empty();
    r.u16(named ? card_or_none(card_name_to_index(perm->chosen_name)) : NONE16);
    const int chosen_type = visible && perm && !perm->chosen_type.empty() ? subtype_index(perm->chosen_type) : -1;
    r.u16(chosen_type < 0 ? NONE16 : static_cast<uint64_t>(chosen_type)).u16(perm ? u16_count(perm->entered_x) : 0);
    r.u8(perm && perm->is_monstrous ? 1 : 0)
        .u8(perm && perm->evoked ? 1 : 0)
        .u8(perm && perm->cast_with_escape ? 1 : 0);
    r.u8(perm && perm->unearthed ? 1 : 0).u8(perm && perm->cast_from_hand_by_controller ? 1 : 0);
}

// The object's child rows: subtypes, keywords, counters, abilities, provenance, links.
static void write_object_children(const Ctx &c, Entity e, uint32_t h) {
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    const bool on_bf = z.location == Zone::BATTLEFIELD && global_coordinator.entity_has_component<Permanent>(e);
    const Permanent *perm = on_bf ? &global_coordinator.GetComponent<Permanent>(e) : nullptr;
    const bool visible = identity_visible(e, c.viewer);
    // Counters are public even on a face-down object.
    const CounterMap &counters = perm ? perm->counters : z.counters;
    for (const auto &kv : counters) c.w.row(TableId::OBJECT_COUNTER).u32(h).text(kv.first).u16(u16_count(kv.second));
    if (!visible) return;

    const CardData *cd = object_card_data(e);
    const Token *tok =
        global_coordinator.entity_has_component<Token>(e) ? &global_coordinator.GetComponent<Token>(e) : nullptr;
    const CardData *face = cd ? &active_face(e, *cd) : nullptr;
    const std::set<Type> empty_types;
    const std::set<Type> &types = perm ? perm->types : face ? face->types : tok ? tok->types : empty_types;
    for (int st : subtype_indices(types)) c.w.row(TableId::OBJECT_SUBTYPE).u32(h).u16(static_cast<uint64_t>(st));

    std::vector<std::string> keywords;
    if (on_bf)
        keywords = permanent_keywords(e);
    else if (face)
        keywords = face->keywords;
    else if (tok)
        keywords = tok->keywords;
    for (const std::string &kw : keywords) {
        const int k = keyword_index(kw);
        c.w.row(TableId::OBJECT_KEYWORD).u32(h).u8(k < 0 ? NONE8 : static_cast<uint64_t>(k)).text(kw);
    }

    if (perm) {
        const int ab_card = cd ? card_name_to_index(cd->name) : card_identity(e, c.viewer);
        std::vector<CardAbility> own = cd    ? card_ability_list(*cd)
                                       : tok ? token_ability_list(*tok)
                                             : std::vector<CardAbility>();
        for (const Ability &ab : perm->abilities) {
            bool printed = false;
            for (const CardAbility &a : own) printed = printed || a.def == ab.def;
            Row r = c.w.row(TableId::OBJECT_ABILITY);
            r.u32(h);
            put_abref(r, ab.def, ab_card, e);
            const bool exhausted = ab.def->activation_limit > 0 && ab.activations_this_turn >= ab.def->activation_limit;
            r.u8(printed && ab.granted_by_static == 0 ? 0 : 1)
                .u8(u8_count(ab.activations_this_turn))
                .u8(exhausted ? 1 : 0);
        }
        write_provenance(c, e, h);
        for (const ObjectRef &linked : perm->exiled_with) {
            const uint32_t lh = handle(c, linked.get());
            if (lh != 0)
                c.w.row(TableId::OBJECT_LINK).u32(h).u32(lh).u8(static_cast<uint64_t>(rmstate::LinkKind::EXILED_WITH));
        }
        const uint32_t rh = handle(c, returnable_exiled_card(e));
        if (rh != 0)
            c.w.row(TableId::OBJECT_LINK).u32(h).u32(rh).u8(static_cast<uint64_t>(rmstate::LinkKind::RETURNABLE));
    }
}

static void put_provenance(const Ctx &c, uint32_t h, rmstate::Layer layer, rmstate::Duration duration) {
    c.w.row(TableId::OBJECT_PROVENANCE)
        .u32(h)
        .u32(0)
        .u16(NONE16)
        .u8(static_cast<uint64_t>(layer))
        .u8(static_cast<uint64_t>(duration));
}

// Coarse provenance (register 1): the kinds of continuous effect the permanent records it is
// under, with their duration. Static abilities' contributions are re-derived every pass and keep
// no source, so `src` is 0 throughout.
static void write_provenance(const Ctx &c, Entity e, uint32_t h) {
    using rmstate::Duration;
    using rmstate::Layer;
    const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
    if (perm.printed_card) put_provenance(c, h, Layer::COPY, Duration::PERMANENT);
    if (!perm.animate_added_types.empty() || perm.animate_make_creature)
        put_provenance(
            c, h, Layer::TYPE, perm.animate_until_my_turn ? Duration::UNTIL_YOUR_NEXT_TURN : Duration::PERMANENT);
    if (!perm.animate_added_types_eot.empty() || perm.animate_make_creature_eot)
        put_provenance(c, h, Layer::TYPE, Duration::END_OF_TURN);
    if (!perm.animate_added_keywords.empty() || !perm.animate_granted_abilities.empty())
        put_provenance(c, h, Layer::ABILITY, Duration::PERMANENT);
    if (!perm.removed_keywords_eot.empty()) put_provenance(c, h, Layer::ABILITY, Duration::END_OF_TURN);
    if (perm.abilities_removed) put_provenance(c, h, Layer::ABILITY, Duration::STATIC);
    if (perm.animate_set_pt)
        put_provenance(
            c, h, Layer::PT_SET, perm.animate_until_my_turn ? Duration::UNTIL_YOUR_NEXT_TURN : Duration::PERMANENT);
    if (!global_coordinator.entity_has_component<Creature>(e)) return;
    const Creature &cr = global_coordinator.GetComponent<Creature>(e);
    if (!cr.eot_keywords.empty()) put_provenance(c, h, Layer::ABILITY, Duration::END_OF_TURN);
    if (cr.has_set_pt) put_provenance(c, h, Layer::PT_SET, Duration::STATIC);
    if (cr.static_power_bonus != 0 || cr.static_toughness_bonus != 0)
        put_provenance(c, h, Layer::PT_MODIFY, Duration::STATIC);
    if (cr.eot_power_bonus != 0 || cr.eot_toughness_bonus != 0 || cr.prowess_bonus != 0)
        put_provenance(c, h, Layer::PT_MODIFY, Duration::END_OF_TURN);
    if (cr.counter_pt_bonus != 0) put_provenance(c, h, Layer::PT_COUNTERS, Duration::STATIC);
    if (cr.switch_pt) put_provenance(c, h, Layer::PT_SWITCH, Duration::STATIC);
}

// One row per blocker of each attacker (a row with blocker 0 for an unblocked attacker), attackers
// and blockers in handle order.
static void write_combat(const Ctx &c, const std::vector<Entity> &battlefield) {
    std::vector<std::pair<uint32_t, Entity>> attackers;
    std::vector<std::pair<uint32_t, Entity>> blockers;
    for (Entity e : battlefield) {
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        const Creature &cr = global_coordinator.GetComponent<Creature>(e);
        if (cr.is_attacking) attackers.push_back({handle(c, e), e});
        if (cr.is_blocking) blockers.push_back({handle(c, e), e});
    }
    std::sort(attackers.begin(), attackers.end());
    std::sort(blockers.begin(), blockers.end());
    for (const auto &att : attackers) {
        const Creature &cr = global_coordinator.GetComponent<Creature>(att.second);
        const Entity target = cr.attack_target.get();
        const bool at_player = target != 0 && global_coordinator.entity_has_component<Player>(target);
        auto put_row = [&](Entity blocker, uint32_t blocker_handle) {
            Row r = c.w.row(TableId::COMBAT);
            r.u32(att.first).u8(static_cast<uint64_t>(
                at_player ? rmstate::AttackTargetKind::PLAYER : rmstate::AttackTargetKind::PLANESWALKER));
            put_seat(r, at_player ? seat_of_player(target) : Zone::UNKNOWN, c.self);
            r.u32(at_player ? 0 : handle(c, target)).u32(blocker_handle);
            uint32_t assigned = 0;
            auto it = cur_game.combat.damage_assignment.find(att.second);
            if (it != cur_game.combat.damage_assignment.end()) {
                auto jt = it->second.find(blocker);
                if (jt != it->second.end()) assigned = jt->second;
            }
            r.u16(u16_count(assigned));
        };
        bool blocked = false;
        for (const auto &blk : blockers) {
            if (global_coordinator.GetComponent<Creature>(blk.second).blocking_target.get() != att.second) continue;
            blocked = true;
            put_row(blk.second, blk.first);
        }
        if (!blocked) put_row(0, 0);
    }
}

// An lki row for a departed object a frame refers to; its row index, or -1 with no last-known
// information. The object left the battlefield in public view, so its last-known
// characteristics are public (CR 608.2h).
static int write_lki(const Ctx &c, Entity departed) {
    const LastKnownInfo *lki = departed_lki_for(departed);
    if (!lki || lki->name.empty()) return -1;
    const int row = static_cast<int>(c.w.row_count(TableId::LKI));
    const int card =
        lki->is_token ? token_script_to_index(lki->token_script) : physical_card_of(card_name_to_index(lki->name));
    Row r = c.w.row(TableId::LKI);
    r.u16(card_or_none(card)).u8(lki->is_token ? 1 : 0);
    put_seat(r, lki->controller, c.self);
    r.i16(std::max(-32768, std::min(lki->power, 32767))).i16(std::max(-32768, std::min(lki->toughness, 32767)));
    r.u32(card_type_mask(lki->types)).u8(color_mask(lki->colors));
    return row;
}

// `stack` is the stack from the top; a row's pos is its index there.
static void write_stack(const Ctx &c, const std::vector<Entity> &stack) {
    for (size_t pos = 0; pos < stack.size(); ++pos) {
        const Entity e = stack[pos];
        const uint32_t h = handle(c, e);
        const bool is_spell = global_coordinator.entity_has_component<Spell>(e);
        const Spell *sp = is_spell ? &global_coordinator.GetComponent<Spell>(e) : nullptr;
        const bool has_ab = global_coordinator.entity_has_component<Ability>(e);
        const Ability *ab = has_ab ? &stack_object_ability(e) : nullptr;
        const rmstate::StackKind kind = stack_kind(e, sp, ab);
        const Entity source = sp || !ab ? 0 : ab->source.get();
        const int source_card = visible_card_id(e, c.viewer);
        Row r = c.w.row(TableId::STACK);
        r.u32(h).u16(u16_count(static_cast<int64_t>(pos)));
        r.u8(static_cast<uint64_t>(kind));
        put_seat(r, source_controller(e), c.self);
        r.u32(handle(c, source)).u16(card_or_none(source_card));
        put_abref(r, ab ? ab->def : nullptr, source_card, source);
        int x = -1;
        if (sp) {
            const CardData *cd = object_card_data(e);
            if (cd && active_face(e, *cd).has_x_cost) x = sp->x_paid;
        } else if (ab) {
            x = ab->x_paid;
        }
        r.i16(std::max(-1, std::min(x, 32767))).u16(ab ? u16_count(static_cast<int64_t>(ab->def->amount)) : 0);
        const int kicked = sp ? static_cast<int>(std::count(sp->kicked.begin(), sp->kicked.end(), true)) : 0;
        r.u8(u8_count(kicked)).u8(sp ? u8_count(sp->replicate_count) : 0);
        uint32_t variant = 0;
        if (sp) {
            using rmstate::CastVariant;
            auto bit = [&variant](CastVariant v) { variant |= 1u << static_cast<int>(v); };
            if (sp->cast_with_flashback) bit(CastVariant::FLASHBACK);
            if (sp->cast_with_evoke) bit(CastVariant::EVOKE);
            if (sp->cast_with_escape) bit(CastVariant::ESCAPE);
            if (sp->cast_with_offspring) bit(CastVariant::OFFSPRING);
            if (sp->cast_with_impending) bit(CastVariant::IMPENDING);
            if (sp->cast_with_warp) bit(CastVariant::WARP);
            if (sp->cast_back_face) bit(CastVariant::BACK_FACE);
        }
        r.u16(variant)
            .u8(sp && sp->is_copy ? 1 : 0)
            .u8(sp && sp->cant_be_countered ? 1 : 0)
            .u8(sp && sp->gift_promised ? 1 : 0);
        r.u8(sp ? u8_count(sp->mana_spent) : 0).u8(sp ? color_mask(sp->colors_spent) : 0);
        r.u32(ab ? ab->delayed_link.seq : 0);
        if (!ab) continue;
        for (int mode : ab->charm_chosen) c.w.row(TableId::STACK_MODE).u32(h).u8(u8_count(mode));
        write_stack_targets(c, e, h, *ab);
    }
}

// A stack object's kind. A card on the stack is a spell even without its Spell component: a
// resolving permanent spell drops it (and its Ability) as it starts to enter the battlefield, and
// stays on the stack while a choice its entry asks for (Mox Diamond's discard) is pending.
static rmstate::StackKind stack_kind(Entity e, const Spell *sp, const Ability *ab) {
    if (sp) return sp->is_copy ? rmstate::StackKind::COPY : rmstate::StackKind::SPELL;
    if (global_coordinator.entity_has_component<CardData>(e)) return rmstate::StackKind::SPELL;
    if (ab && ab->def->ability_type == AbilityDef::TRIGGERED) return rmstate::StackKind::TRIGGERED;
    return rmstate::StackKind::ACTIVATED;
}

// The announced targets of a stack object, in the order the observation lists them: the primary
// ability's, its targeting sub-abilities', then each chosen mode's (CR 601.2c).
static void write_stack_targets(const Ctx &c, Entity /*stack_object*/, uint32_t h, const Ability &ab) {
    int order = 0;
    auto put_target = [&](const ObjectRef &t) {
        if (t.lki_entity() == 0) return;
        const Entity live = t.get();
        const int lki_row = live == 0 ? write_lki(c, t.lki_entity()) : -1;
        Row r = c.w.row(TableId::STACK_TARGET);
        r.u32(h).u8(u8_count(order++));
        if (live != 0 && global_coordinator.entity_has_component<Player>(live)) {
            r.u8(static_cast<uint64_t>(rmstate::TargetKind::PLAYER)).u32(0);
            put_seat(r, seat_of_player(live), c.self);
            r.u16(NONE16);
            return;
        }
        r.u8(static_cast<uint64_t>(rmstate::TargetKind::OBJECT)).u32(handle(c, live));
        put_seat(r, Zone::UNKNOWN, c.self);
        r.u16(lki_row < 0 ? NONE16 : u16_count(lki_row));
    };
    auto targets_of = [](const Ability &a) {
        std::vector<ObjectRef> out = a.targets;
        if (out.empty() && !a.target.empty()) out.push_back(a.target);
        return out;
    };
    std::vector<ObjectRef> all = targets_of(ab);
    for (const Ability &sub : ab.subabilities)
        for (const ObjectRef &t : targets_of(sub)) all.push_back(t);
    for (int idx : ab.charm_chosen)
        if (idx >= 0 && static_cast<size_t>(idx) < ab.charm_choices.size())
            for (const ObjectRef &t : targets_of(ab.charm_choices[static_cast<size_t>(idx)])) all.push_back(t);
    for (const ObjectRef &t : all) put_target(t);
}

static void write_delayed_triggers(const Ctx &c, const std::vector<Entity> &stack) {
    struct Entry {
            uint32_t seq;
            const DelayedTrigger *dt;  // waiting
            Entity stack_object;       // fired, on the stack
    };
    std::vector<Entry> entries;
    for (const DelayedTrigger &dt : cur_game.delayed_triggers) entries.push_back({dt.ability.delayed_link.seq, &dt, 0});
    for (Entity e : stack)
        if (global_coordinator.entity_has_component<Ability>(e) &&
            global_coordinator.GetComponent<Ability>(e).delayed_link.seq != 0)
            entries.push_back({global_coordinator.GetComponent<Ability>(e).delayed_link.seq, nullptr, e});
    std::stable_sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.seq < b.seq; });
    for (const Entry &en : entries) {
        const Ability &ab = en.dt ? en.dt->ability : global_coordinator.GetComponent<Ability>(en.stack_object);
        const DelayedTriggerLink &link = ab.delayed_link;
        const uint32_t row = c.w.row_count(TableId::DELAYED_TRIGGER);
        const Entity creator = link.creator.get();
        const int creator_card =
            creator != 0 ? visible_card_id(creator, c.viewer) : physical_card_of(link.creator_vocab_idx);
        {
            Row r = c.w.row(TableId::DELAYED_TRIGGER);
            r.u32(link.seq);
            put_seat(r, en.dt ? seat_of_player(en.dt->owner_entity) : source_controller(en.stack_object), c.self);
            r.u8(static_cast<uint64_t>(en.dt ? rmstate::DelayedState::WAITING : rmstate::DelayedState::ON_STACK));
            r.u32(handle(c, en.stack_object)).u32(handle(c, creator)).u16(card_or_none(creator_card));
            r.u8(link.fire_kind == DelayedTriggerLink::FIRE_OTHER ? NONE8 : static_cast<uint64_t>(link.fire_kind));
            r.u16(en.dt ? u16_count(static_cast<int64_t>(en.dt->fire_on_turn)) : 0);
            r.u8(en.dt && delayed_trigger_fires_this_turn(*en.dt) ? 1 : 0)
                .u8(en.dt && en.dt->expires_end_of_turn ? 1 : 0);
            put_abref(r, ab.def, creator_card, creator);
        }
        std::vector<Entity> subjects;
        for (const ObjectRef &s : link.subjects) subjects.push_back(s.get());
        if (en.dt) subjects.push_back(en.dt->watched.get());
        std::vector<Entity> written;
        for (Entity s : subjects) {
            if (s == 0 || std::find(written.begin(), written.end(), s) != written.end()) continue;
            const uint32_t sh = handle(c, s);
            if (sh == 0) continue;
            written.push_back(s);
            c.w.row(TableId::DELAYED_SUBJECT)
                .u16(u16_count(row))
                .u32(sh)
                .u16(card_or_none(visible_card_id(s, c.viewer)));
        }
    }
}

static rmstate::Duration effect_duration(bool until_your_next_turn) {
    return until_your_next_turn ? rmstate::Duration::UNTIL_YOUR_NEXT_TURN : rmstate::Duration::END_OF_TURN;
}

// The effects that apply to a player (player_effects, queries/player_effects.h) with the duration
// of the resolved effect behind each; one a battlefield static gives has duration STATIC.
static void write_player_effects(const Ctx &c, Zone::Ownership p) {
    using rmstate::Duration;
    using rmstate::PlayerEffectKind;
    const Game::ResolvedEffects &re = cur_game.resolved_effects;
    const PlayerEffects fx = player_effects(p, zoned_entities());
    auto put = [&](PlayerEffectKind kind, uint8_t colors, Duration d, int source_card, const std::string &filter) {
        Row r = c.w.row(TableId::PLAYER_EFFECT);
        put_seat(r, p, c.self);
        r.u8(static_cast<uint64_t>(kind))
            .u8(colors)
            .u8(static_cast<uint64_t>(d))
            .u16(card_or_none(source_card))
            .text(filter);
    };
    bool resolved = false;
    for (const auto &g : re.player_protection_from_everything)
        if (g.player == p) {
            put(PlayerEffectKind::PROTECTION_FROM_EVERYTHING, 0, effect_duration(g.until_your_next_turn), -1, "");
            resolved = true;
        }
    if (fx.protection_from_everything && !resolved)
        put(PlayerEffectKind::PROTECTION_FROM_EVERYTHING, 0, Duration::STATIC, -1, "");
    const bool eot_no_gain = re.cant_gain_life_this_turn.count(p) > 0;
    if (eot_no_gain) put(PlayerEffectKind::CANT_GAIN_LIFE, 0, Duration::END_OF_TURN, -1, "");
    if (fx.cant_gain_life && !eot_no_gain) put(PlayerEffectKind::CANT_GAIN_LIFE, 0, Duration::STATIC, -1, "");
    uint8_t eot_hexproof = 0;
    for (const auto &g : re.hexproof_from_colors_this_turn)
        if (g.player == p) eot_hexproof = static_cast<uint8_t>(eot_hexproof | color_mask(g.colors));
    if (eot_hexproof) put(PlayerEffectKind::HEXPROOF_FROM, eot_hexproof, Duration::END_OF_TURN, -1, "");
    uint32_t static_hexproof = 0;
    for (int i = 0; i < 5; ++i)
        if (fx.hexproof_from[i] && !(eot_hexproof & (1u << i))) static_hexproof |= 1u << i;
    if (static_hexproof)
        put(PlayerEffectKind::HEXPROOF_FROM, static_cast<uint8_t>(static_hexproof), Duration::STATIC, -1, "");
    const bool eot_uncounterable = re.cant_counter_spells_of.count(p) > 0;
    if (eot_uncounterable) put(PlayerEffectKind::SPELLS_CANT_BE_COUNTERED, 0, Duration::END_OF_TURN, -1, "");
    if (fx.spells_cant_be_countered && !eot_uncounterable)
        put(PlayerEffectKind::SPELLS_CANT_BE_COUNTERED, 0, Duration::STATIC, -1, "");
    resolved = false;
    for (const auto &g : re.cast_with_flash_permissions)
        if (g.controller == p) {
            put(PlayerEffectKind::CAST_AS_FLASH, 0, effect_duration(g.until_your_next_turn), -1, g.filter);
            resolved = true;
        }
    if (fx.may_cast_sorceries_as_flash && !resolved) put(PlayerEffectKind::CAST_AS_FLASH, 0, Duration::STATIC, -1, "");
    if (fx.restricted_to_sorcery_speed) put(PlayerEffectKind::SORCERY_SPEED_ONLY, 0, Duration::STATIC, -1, "");
    for (const Ability &t : re.floating_triggers)
        if (t.controller == p)
            put(PlayerEffectKind::FLOATING_TRIGGER, 0, Duration::END_OF_TURN, t.floating_creator_vocab_idx, "");
}

// Emblems, player effects, play permissions and damage shields (register 18).
static void write_rules_state(const Ctx &c, const std::vector<Entity> &zone_cards) {
    for (const Emblem &em : cur_game.resolved_effects.emblems) {
        Row r = c.w.row(TableId::EMBLEM);
        put_seat(r, em.controller, c.self);
        r.u16(card_or_none(physical_card_of(em.source_vocab_idx)));
    }
    write_player_effects(c, c.self);
    write_player_effects(c, opponent_of(c.self));
    for (Entity e : zone_cards) {
        for (Zone::Ownership p : {c.self, opponent_of(c.self)}) {
            const CardPlayPermission perm = card_play_permission(e, p);
            if (!perm.playable()) continue;
            for (int bit = 0; bit <= static_cast<int>(rmstate::PermissionKind::EXILE_GRANT); ++bit) {
                if (!(perm.sources & (1u << bit))) continue;
                rmstate::PermissionResource resource = rmstate::PermissionResource::MANA;
                int amount = 0;
                if (bit == static_cast<int>(rmstate::PermissionKind::EXILE_GRANT)) {
                    if (const Game::ImpulseCastPermission *g =
                            cur_game.resolved_effects.impulse_cast_permission.find(e)) {
                        amount = g->amount;
                        switch (g->resource) {
                            case Game::ImpulseCastPermission::ENERGY:
                                resource = rmstate::PermissionResource::ENERGY;
                                break;
                            case Game::ImpulseCastPermission::LIFE:
                                resource = rmstate::PermissionResource::LIFE;
                                break;
                            case Game::ImpulseCastPermission::FREE:
                                resource = rmstate::PermissionResource::FREE;
                                break;
                            case Game::ImpulseCastPermission::NORMAL:
                                resource = rmstate::PermissionResource::MANA;
                                break;
                        }
                    }
                }
                Row r = c.w.row(TableId::PLAY_PERMISSION);
                r.u32(handle(c, e));
                put_seat(r, p, c.self);
                r.u8(static_cast<uint64_t>(bit)).u8(static_cast<uint64_t>(resource)).u16(u16_count(amount));
                r.u8(perm.expires_this_turn ? 1 : 0);
            }
        }
    }
    for (const auto &s : cur_game.resolved_effects.combat_damage_prevention_shields) {
        const uint32_t h = handle(c, s.creature.get());
        if (h == 0) continue;
        c.w.row(TableId::SHIELD)
            .u8(static_cast<uint64_t>(rmstate::ShieldKind::COMBAT_DAMAGE))
            .u32(h)
            .u8(s.prevent_as_source ? 1 : 0)
            .u8(s.prevent_as_target ? 1 : 0);
    }
}

static void write_reveals(const Ctx &c) {
    for (const VisibleWindow &win : reveal_windows_for(c.viewer)) {
        const Zone &z = global_coordinator.GetComponent<Zone>(win.card);
        Row r = c.w.row(TableId::REVEAL);
        r.u32(handle(c, win.card)).u8(static_cast<uint64_t>(z.location));
        put_seat(r, z.owner, c.self);
        r.u16(z.location == Zone::LIBRARY ? u16_count(static_cast<int64_t>(z.distance_from_top)) : NONE16);
        r.u16(card_or_none(visible_card_id(win.card, c.viewer)))
            .u8(static_cast<uint64_t>(win.kind))
            .u8(static_cast<uint64_t>(win.until));
        put_seat(r, win.audience, c.self);
    }
}

static void write_pending(const Ctx &c, Zone::Ownership decider) {
    const Game::PendingDecisions &pd = cur_game.pending;
    const Entity source =
        pd.decision_source != 0 ? pd.decision_source : (pd.query.active ? pd.query.decision_source : 0);
    if (!pd.query.active && source == 0) return;
    Row r = c.w.row(TableId::PENDING);
    r.u8(static_cast<uint64_t>(pd.query.active ? pd.query.tag : PendingQuery::NONE));
    put_seat(r, pd.query.active ? (pd.query.chooser_is_a ? Zone::PLAYER_A : Zone::PLAYER_B) : decider, c.self);
    r.u32(handle(c, source)).u16(card_or_none(visible_card_id(source, c.viewer)));
    put_seat(r,
        source != 0 && !global_coordinator.entity_has_component<Player>(source) ? source_controller(source)
                                                                                : Zone::UNKNOWN,
        c.self);
}

static void write_pending_cost(const Ctx &c) {
    PendingCost pc;
    if (!pending_cost(pc)) return;
    const uint32_t row = c.w.row_count(TableId::PENDING_COST);
    {
        Row r = c.w.row(TableId::PENDING_COST);
        r.u32(handle(c, pc.paid_for))
            .u16(card_or_none(visible_card_id(pc.paid_for, c.viewer)))
            .u8(static_cast<uint64_t>(pc.part));
        for (Colors col : {WHITE, BLUE, BLACK, RED, GREEN, COLORLESS, GENERIC})
            r.u8(u8_count(count_color(pc.remaining, col)));
        r.u8(u8_count(pc.x_pips))
            .u16(u16_count(pc.life))
            .u8(u8_count(pc.energy))
            .u8(u8_count(pc.discard))
            .u8(u8_count(pc.paid));
    }
    for (const HybridPip &pip : pc.hybrid)
        c.w.row(TableId::PENDING_COST_PIP)
            .u16(u16_count(row))
            .u8(static_cast<uint64_t>(pip.generic_alt > 0 ? rmstate::PipKind::TWOBRID : rmstate::PipKind::HYBRID))
            .u8(color_mask(std::set<Colors>(pip.colors.begin(), pip.colors.end())))
            .u8(u8_count(pip.generic_alt));
    for (Colors col : pc.phyrexian)
        c.w.row(TableId::PENDING_COST_PIP)
            .u16(u16_count(row))
            .u8(static_cast<uint64_t>(rmstate::PipKind::PHYREXIAN))
            .u8(color_mask({col}))
            .u8(0);
}

// Whether a menu entry's description could name a card the viewer cannot see: it refers to an
// object whose identity is hidden from them and the choice does not itself make it public. Such a
// description is replaced (register 12). Decided without reading the hidden identity.
static bool description_hides_identity(const Ctx &c, const LegalAction &la) {
    if (la.card_is_public || !is_seat(c.viewer)) return false;
    for (Entity e : {la.source_entity, la.target_entity})
        if (e != 0 && global_coordinator.entity_has_component<Zone>(e) && !identity_visible(e, c.viewer)) return true;
    return false;
}

static uint16_t cast_variant_mask(const LegalAction &la) {
    using rmstate::CastVariant;
    uint32_t m = 0;
    auto bit = [&m](bool on, CastVariant v) {
        if (on) m |= 1u << static_cast<int>(v);
    };
    bit(la.use_flashback, CastVariant::FLASHBACK);
    bit(la.use_escape, CastVariant::ESCAPE);
    bit(la.use_offspring, CastVariant::OFFSPRING);
    bit(la.play_back_face || la.cast_back_face, CastVariant::BACK_FACE);
    bit(la.use_alt_cost, CastVariant::ALT_COST);
    bit(la.impulse_cast, CastVariant::IMPULSE);
    bit(la.companion_to_hand, CastVariant::COMPANION);
    bit(la.suspend_action, CastVariant::SUSPEND);
    bit(la.cancel_proposal, CastVariant::CANCEL);
    return static_cast<uint16_t>(m);
}

static void write_menu(const Ctx &c, const std::vector<LegalAction> &actions) {
    for (const LegalAction &la : actions) {
        const Entity src = la.source_entity;
        const Entity tgt = la.target_entity;
        const bool src_player = src != 0 && global_coordinator.entity_has_component<Player>(src);
        const bool tgt_player = tgt != 0 && global_coordinator.entity_has_component<Player>(tgt);
        const bool src_zoned = src != 0 && global_coordinator.entity_has_component<Zone>(src);
        const int src_card = src_player           ? -1
                             : la.named_card >= 0 ? la.named_card
                             : la.card_is_public  ? public_card_id(src)
                                                  : visible_card_id(src, c.viewer);
        Row r = c.w.row(TableId::MENU);
        r.u8(static_cast<uint64_t>(la.category)).u8(static_cast<uint64_t>(la.type));
        r.u32(src_zoned ? handle(c, src) : 0)
            .u32(tgt != 0 && !tgt_player ? handle(c, tgt) : 0)
            .u16(card_or_none(src_card));
        put_seat(r, src_player ? seat_of_player(src) : Zone::UNKNOWN, c.self);
        put_seat(r, tgt_player ? seat_of_player(tgt) : Zone::UNKNOWN, c.self);
        const bool has_ability = la.type == ACTIVATE_ABILITY && la.ability.def != blank_ability_def();
        put_abref(r, has_ability ? la.ability.def : nullptr, src_card, src_zoned && handle(c, src) != 0 ? src : 0);
        r.i16(std::max(-32768, std::min(la.option_ordinal, 32767)))
            .u16(cast_variant_mask(la))
            .u8(la.card_is_public ? 1 : 0);
        if (description_hides_identity(c, la))
            r.text("(hidden card)");
        else
            r.text(la.description);
    }
}

static void write_match(const Ctx &c) {
    Row r = c.w.row(TableId::MATCH);
    r.u8(u8_count(match_game_index()));
    put_seat(r, plays_first_seat(), c.self);
    r.u8(sideboard_phase ? 1 : 0).u8(sideboard_phase_state ? u8_count(sideboard_phase_state->sb_swaps) : 0);
    r.i8(sideboard_phase_state ? std::max(-128, std::min(sideboard_phase_state->delta, 127)) : 0);
    r.u8(closed_decklist ? 1 : 0);
}

// The viewer's current 75 (the omniscient viewer's: both players'), one row per card.
static void write_own_deck(const Ctx &c) {
    std::vector<Zone::Ownership> seats;
    if (is_seat(c.viewer))
        seats.push_back(c.viewer);
    else
        seats = {Zone::PLAYER_A, Zone::PLAYER_B};
    for (Zone::Ownership p : seats) {
        std::map<int, std::pair<int, int>> deck;
        for (const DecklistEntry &e : deck_state_live_main(p)) deck[physical_card_of(e.vocab_idx)].first += e.count;
        for (const DecklistEntry &e : deck_state_live_side(p)) deck[physical_card_of(e.vocab_idx)].second += e.count;
        for (const auto &kv : deck) {
            Row r = c.w.row(TableId::OWN_DECK);
            put_seat(r, p, c.self);
            r.u16(card_or_none(kv.first)).u8(u8_count(kv.second.first)).u8(u8_count(kv.second.second));
        }
    }
}

// ── Frames ────────────────────────────────────────────────────────────────────────────────────

uint64_t rmstate_schema_hash() {
    static const uint64_t hash = [] {
        uint64_t h = 0;
        rmstate::build_schema_frame(rmstate::Viewer::OMNISCIENT, all_enums(), &h);
        return h;
    }();
    return hash;
}

std::string encode_schema_frame(Zone::Ownership viewer) {
    return rmstate::build_schema_frame(frame_viewer(viewer), all_enums(), nullptr);
}

std::string encode_cards_frame(Zone::Ownership viewer) {
    FrameWriter w(FrameKind::CARDS, frame_viewer(viewer), rmstate_schema_hash());
    std::set<std::string> token_scripts;
    for (int id : cards_frame_ids(viewer)) {
        // A card the match loaded is read from its card_db entry; one no deck holds (a closed
        // decklist lists the whole vocab) from the parse cache, which creates no entity.
        auto it = card_db.find(name_to_uid(card_index_to_name(id)));
        const CardData *loaded = it != card_db.end() && global_coordinator.entity_has_component<CardData>(it->second)
                                     ? &global_coordinator.GetComponent<CardData>(it->second)
                                     : nullptr;
        const CardData *found = loaded ? loaded : card_data_by_name(card_index_to_name(id));
        if (found == nullptr) continue;
        const CardData &cd = *found;
        write_card_face(w, id, cd, rmstate::Face::FRONT, id,
            rmstate_card_scripts ? card_script_text(card_index_to_name(id)) : std::string());
        if (cd.backside)
            write_card_face(
                w, id, *cd.backside, rmstate::Face::BACK, card_name_to_index(cd.backside->name), std::string());
        write_card_abilities(w, id, card_ability_list(cd), &cd);
        collect_card_tokens(cd, token_scripts);
    }
    // Tokens, and the tokens those tokens can make, until no new script turns up.
    std::set<std::string> done;
    std::map<int, Token> tokens;
    while (done.size() < token_scripts.size()) {
        std::vector<std::string> fresh;
        for (const std::string &s : token_scripts)
            if (!done.count(s)) fresh.push_back(s);
        for (const std::string &stem : fresh) {
            done.insert(stem);
            const int id = token_script_to_index(stem);
            if (id < 0 || !token_script_exists(stem)) continue;
            const Token tok = parse_token_script(stem);
            if (tok.name.empty()) continue;
            for (const AbilityDef *d : tok.abilities) collect_token_scripts(*d, token_scripts);
            tokens.emplace(id, tok);
        }
    }
    for (const auto &kv : tokens) write_token_card(w, kv.first, kv.second);
    return w.finish();
}

std::string encode_decklists_frame(Zone::Ownership viewer) {
    FrameWriter w(FrameKind::DECKLISTS, frame_viewer(viewer), rmstate_schema_hash());
    const Zone::Ownership self = is_seat(viewer) ? viewer : Zone::PLAYER_A;
    for (Zone::Ownership owner : {self, opponent_of(self)}) {
        if (!sees_decklist(viewer, owner)) continue;
        std::map<int, std::pair<int, int>> cards;
        int main = 0;
        int side = 0;
        for (const DecklistEntry &e : deck_state_registered_main(owner)) {
            cards[physical_card_of(e.vocab_idx)].first += e.count;
            main += e.count;
        }
        for (const DecklistEntry &e : deck_state_registered_side(owner)) {
            cards[physical_card_of(e.vocab_idx)].second += e.count;
            side += e.count;
        }
        {
            Row r = w.row(TableId::DECKLIST);
            put_seat(r, owner, self);
            r.text(owner == Zone::PLAYER_A ? deck_a_name : deck_b_name).u16(u16_count(main)).u16(u16_count(side));
        }
        for (const auto &kv : cards) {
            Row r = w.row(TableId::DECKLIST_CARD);
            put_seat(r, owner, self);
            r.u16(card_or_none(kv.first)).u8(u8_count(kv.second.first)).u8(u8_count(kv.second.second));
        }
    }
    return w.finish();
}

std::string encode_metagame_frame(Zone::Ownership viewer) {
    FrameWriter w(FrameKind::METAGAME, frame_viewer(viewer), rmstate_schema_hash());
    const ReferenceMetagame &m = reference_metagame();
    for (const std::string &label : m.archetypes) w.row(TableId::META_ARCHETYPE).text(label);
    for (const ReferenceList &list : m.lists)
        w.row(TableId::META_LIST)
            .text(list.name)
            .u8(static_cast<uint64_t>(list.archetype))
            .u16(u16_count(list.main))
            .u16(u16_count(list.side));
    for (size_t l = 0; l < m.lists.size(); ++l)
        for (const auto &c : m.lists[l].cards)
            w.row(TableId::META_LIST_CARD)
                .u16(l)
                .u16(card_or_none(c.first))
                .u8(u8_count(c.second.first))
                .u8(u8_count(c.second.second));
    return w.finish();
}

std::vector<std::string> encode_match_frames(Zone::Ownership viewer) {
    return {encode_schema_frame(viewer), encode_cards_frame(viewer), encode_decklists_frame(viewer),
        encode_metagame_frame(viewer)};
}

std::string encode_state_frame(Zone::Ownership viewer, const std::vector<LegalAction> &actions, Zone::Ownership decider,
    const std::vector<ObsEvent> &events, bool terminal) {
    const Zone::Ownership v = viewer_key(viewer);
    FrameWriter w(FrameKind::STATE, frame_viewer(v), rmstate_schema_hash());
    const Ctx c{v, is_seat(v) ? v : Zone::PLAYER_A, w};
    issue_handles_canonical(v);

    // One pass over the zoned objects: zone sizes, and the objects the viewer can identify.
    ZoneCounts counts[3];
    std::vector<std::pair<uint32_t, Entity>> objects;
    std::vector<Entity> battlefield;
    std::vector<std::pair<size_t, Entity>> stack;
    std::vector<Entity> zone_cards;  // graveyard and exile cards (play permissions)
    for (Entity e : zoned_entities()) {
        const Zone &z = global_coordinator.GetComponent<Zone>(e);
        ZoneCounts &zc = counts[is_seat(z.owner) ? z.owner : 0];
        switch (z.location) {
            case Zone::HAND:
                ++zc.hand;
                break;
            case Zone::LIBRARY:
                ++zc.library;
                break;
            case Zone::GRAVEYARD:
                ++zc.graveyard;
                break;
            case Zone::EXILE:
                ++zc.exile;
                break;
            case Zone::SIDEBOARD:
                ++zc.sideboard;
                break;
            default:
                break;
        }
        const uint32_t h = handle_peek(v, e);
        if (h == 0) continue;
        objects.push_back({h, e});
        if (z.location == Zone::BATTLEFIELD && global_coordinator.entity_has_component<Permanent>(e))
            battlefield.push_back(e);
        if (z.location == Zone::STACK) stack.push_back({z.distance_from_top, e});
        if (z.location == Zone::GRAVEYARD || z.location == Zone::EXILE) zone_cards.push_back(e);
    }
    std::sort(objects.begin(), objects.end());
    std::sort(stack.begin(), stack.end());
    std::vector<Entity> stack_objects;
    for (const auto &s : stack) stack_objects.push_back(s.second);
    std::vector<Entity> ordered_zone_cards;
    for (const auto &o : objects)
        if (std::find(zone_cards.begin(), zone_cards.end(), o.second) != zone_cards.end())
            ordered_zone_cards.push_back(o.second);
    std::vector<Entity> ordered_battlefield;
    for (const auto &o : objects)
        if (std::find(battlefield.begin(), battlefield.end(), o.second) != battlefield.end())
            ordered_battlefield.push_back(o.second);

    write_header(c, terminal);
    write_turn(c);
    write_players(c, counts);
    for (const auto &o : objects) {
        const auto at = std::find(stack_objects.begin(), stack_objects.end(), o.second);
        write_object(
            c, o.second, o.first, at == stack_objects.end() ? -1 : static_cast<int>(at - stack_objects.begin()));
        write_object_children(c, o.second, o.first);
    }
    write_combat(c, ordered_battlefield);
    write_stack(c, stack_objects);
    write_delayed_triggers(c, stack_objects);
    write_rules_state(c, ordered_zone_cards);
    write_reveals(c);
    write_pending(c, decider);
    write_pending_cost(c);
    if (!is_seat(v) || v == decider) write_menu(c, actions);
    write_match(c);
    write_own_deck(c);
    write_events(c, events);
    return w.finish();
}

std::string encode_events_frame(Zone::Ownership viewer, const std::vector<ObsEvent> &events) {
    const Zone::Ownership v = viewer_key(viewer);
    FrameWriter w(FrameKind::STATE, frame_viewer(v), rmstate_schema_hash());
    const Ctx c{v, is_seat(v) ? v : Zone::PLAYER_A, w};
    write_events(c, events);
    return w.finish();
}

// ── Events (A.6) ──────────────────────────────────────────────────────────────────────────────

static void put_libpos(Row &r, const ObsLibPos &p) {
    r.u8(p.anchor).u16(p.pos).u16(p.span);
}

static uint64_t zone_or_none(int zone) {
    return zone < 0 ? NONE8 : static_cast<uint64_t>(zone);
}

// The viewer's events in queue order: one event row each, then its cards and players as child rows
// naming the event's row.
static void write_events(const Ctx &c, const std::vector<ObsEvent> &events) {
    for (const ObsEvent &ev : events) {
        Row r = c.w.row(TableId::EVENT);
        r.u32(ev.seq).u8(static_cast<uint64_t>(ev.kind));
        put_seat(r, ev.player, c.self);
        r.u8(zone_or_none(ev.zone)).u8(zone_or_none(ev.to_zone));
        put_seat(r, ev.zone_owner, c.self);
        put_seat(r, ev.to_owner, c.self);
        r.u16(ev.count);
        put_libpos(r, ev.range);
        r.u8(ev.order_known ? 1 : 0).u16(ev.batch);
        put_seat(r, ev.to, c.self);
        r.u32(ev.source).u16(card_or_none(ev.source_card)).u16(ev.turn).u8(ev.opening ? 1 : 0).u8(ev.game);
    }
    for (size_t i = 0; i < events.size(); ++i) {
        for (const ObsEventCard &card : events[i].cards) {
            Row r = c.w.row(TableId::EVENT_CARD);
            r.u32(i).u32(card.handle_before).u32(card.handle_after).u16(card_or_none(card.card));
            put_libpos(r, card.from);
            put_libpos(r, card.to);
            r.u8(card.face_down ? 1 : 0).u8(card.token ? 1 : 0).u8(card.hand_subset);
            r.u32(card.draw_seq).u16(card.draw_idx);
        }
    }
    for (size_t i = 0; i < events.size(); ++i) {
        for (const ObsEventPlayer &p : events[i].players) {
            Row r = c.w.row(TableId::EVENT_PLAYER);
            r.u32(i);
            put_seat(r, p.player, c.self);
            r.u16(p.library).u16(p.sideboard).u8(p.wins);
        }
    }
}
