#include "schema.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace rmstate {

static void add_table(std::vector<TableDef> &out, TableId id, const char *name, FrameKind kind);
static void col(TableDef &t, const std::string &name, Storage s, const std::string &tag);
static void col_range(
    TableDef &t, const std::string &name, Storage s, const std::string &tag, int64_t min, int64_t max);
static void col_bool(TableDef &t, const std::string &name);
static void col_seat(TableDef &t, const std::string &name);
static void col_libpos(TableDef &t, const std::string &name);
static void col_abref(TableDef &t);
static void col_text(TableDef &t, const std::string &name);
static void col_handle(TableDef &t, const std::string &name);
static void col_card(TableDef &t, const std::string &name);
static void col_count(TableDef &t, const std::string &name, Storage s);
static void col_enum(TableDef &t, const std::string &name, Storage s, const std::string &enum_name);
static void col_ability_params(TableDef &t);
static std::vector<TableDef> build_table_defs();
static void define_schema_tables(std::vector<TableDef> &out);
static void define_cards_tables(std::vector<TableDef> &out);
static void define_snapshot_tables(std::vector<TableDef> &out);
static void define_event_tables(std::vector<TableDef> &out);
static void define_knowledge_tables(std::vector<TableDef> &out);
static void finish_table(TableDef &t);
static void append_le(std::string &out, uint64_t v, uint32_t width);
static void check_failed(const char *what, const TableDef *def, size_t col);

// ── Column helpers ────────────────────────────────────────────────────────────────────────────

// Starts table `id`; its columns are appended to out.back() by the helpers below, and
// finish_table computes their offsets and the row width once every table is defined.
static void add_table(std::vector<TableDef> &out, TableId id, const char *name, FrameKind kind) {
    TableDef def;
    def.id = id;
    def.name = name;
    def.frame_kind = kind;
    out.push_back(def);
}

static void col(TableDef &t, const std::string &name, Storage s, const std::string &tag) {
    ColumnDef c;
    c.name = name;
    c.storage = s;
    c.tag = tag;
    t.columns.push_back(c);
}

static void col_range(
    TableDef &t, const std::string &name, Storage s, const std::string &tag, int64_t min, int64_t max) {
    col(t, name, s, tag);
    t.columns.back().has_range = true;
    t.columns.back().min = min;
    t.columns.back().max = max;
}

static void col_bool(TableDef &t, const std::string &name) {
    col_range(t, name, Storage::U8, "bool", 0, 1);
}

// Seat pair (Appendix A.1, register 31): the absolute seat and the seat relative to the viewer.
static void col_seat(TableDef &t, const std::string &name) {
    col_range(t, name, Storage::U8, "enum:seat", 0, 2);
    col_range(t, name + "_rel", Storage::U8, "player-rel", 0, 2);
}

// Library position (Appendix A.1, §5): anchor, position from that end, span.
static void col_libpos(TableDef &t, const std::string &name) {
    col(t, name + "_anchor", Storage::U8, "enum:lib_anchor");
    col(t, name + "_pos", Storage::U16, "index");
    col(t, name + "_span", Storage::U16, "count");
}

// Ability reference (Appendix A.1, registers 2 and 25).
static void col_abref(TableDef &t) {
    col(t, "ab_card", Storage::U16, "card");
    col(t, "ab_index", Storage::U16, "index");
    col(t, "ab_key", Storage::TEXT, "text");
}

static void col_text(TableDef &t, const std::string &name) {
    col(t, name, Storage::TEXT, "text");
}

static void col_handle(TableDef &t, const std::string &name) {
    col(t, name, Storage::U32, "handle");
}

static void col_card(TableDef &t, const std::string &name) {
    col(t, name, Storage::U16, "card");
}

static void col_count(TableDef &t, const std::string &name, Storage s) {
    col(t, name, s, "count");
}

static void col_enum(TableDef &t, const std::string &name, Storage s, const std::string &enum_name) {
    col(t, name, s, "enum:" + enum_name);
}

// The effect parameters of one ability definition (Appendix A.4, register 41), shared by
// card_ability and card_ability_part.
static void col_ability_params(TableDef &t) {
    col_count(t, "amount", Storage::U16);
    col_enum(t, "amount_kind", Storage::U8, "amount_kind");
    col(t, "power_mod", Storage::I16, "signed");
    col(t, "toughness_mod", Storage::I16, "signed");
    col_enum(t, "pt_kind", Storage::U8, "amount_kind");
    col(t, "targets", Storage::U16, "mask:target_class");
    col_count(t, "target_min", Storage::U8);
    col_count(t, "target_max", Storage::U8);
}

// ── Table definitions (Appendix A.3-A.6) ──────────────────────────────────────────────────────

static void define_schema_tables(std::vector<TableDef> &out) {
    add_table(out, TableId::STRINGS, "strings", FrameKind::STATE);
    col(out.back(), "byte", Storage::U8, "text");

    add_table(out, TableId::SCHEMA_TABLE, "schema_table", FrameKind::SCHEMA);
    TableDef &st = out.back();
    col(st, "id", Storage::U16, "index");
    col_text(st, "name");
    col_enum(st, "frame_kind", Storage::U8, "frame_kind");
    col_count(st, "row_width", Storage::U32);

    add_table(out, TableId::SCHEMA_COLUMN, "schema_column", FrameKind::SCHEMA);
    TableDef &sc = out.back();
    col(sc, "table", Storage::U16, "index");
    col(sc, "col", Storage::U16, "index");
    col_text(sc, "name");
    col_enum(sc, "storage", Storage::U8, "storage");
    col_count(sc, "offset", Storage::U16);
    col_text(sc, "tag");
    col_bool(sc, "has_range");
    col(sc, "min", Storage::I64, "signed");
    col(sc, "max", Storage::I64, "signed");

    add_table(out, TableId::SCHEMA_ENUM, "schema_enum", FrameKind::SCHEMA);
    TableDef &se = out.back();
    col_text(se, "enum");
    col(se, "value", Storage::U16, "index");
    col_text(se, "name");
}

static void define_cards_tables(std::vector<TableDef> &out) {
    add_table(out, TableId::CARD_ABILITY_PART, "card_ability_part", FrameKind::CARDS);
    TableDef &ap = out.back();
    col_card(ap, "card");
    col(ap, "ability", Storage::U16, "index");
    col(ap, "part", Storage::U16, "index");
    col(ap, "parent", Storage::U16, "index");
    col_enum(ap, "kind", Storage::U8, "part_kind");
    col_count(ap, "depth", Storage::U8);
    col_enum(ap, "category", Storage::U8, "effect_kind");
    col_ability_params(ap);

    add_table(out, TableId::CARD, "card", FrameKind::CARDS);
    TableDef &c = out.back();
    col_card(c, "card");
    col_enum(c, "face", Storage::U8, "face");
    col_card(c, "face_card");
    col_text(c, "name");
    col_text(c, "uid");
    col_bool(c, "token");
    col_enum(c, "layout", Storage::U8, "layout");
    col_text(c, "mana_cost");
    col_count(c, "mana_value", Storage::U8);
    for (const char *p : {"cost_w", "cost_u", "cost_b", "cost_r", "cost_g", "cost_c", "cost_generic", "cost_x"})
        col_count(c, p, Storage::U8);
    col(c, "colors", Storage::U8, "mask:color");
    col(c, "types", Storage::U32, "mask:card_type");
    col(c, "supertypes", Storage::U8, "mask:supertype");
    col(c, "power", Storage::I16, "signed");
    col(c, "toughness", Storage::I16, "signed");
    col_count(c, "loyalty", Storage::U8);
    col_text(c, "oracle");
    col_text(c, "script");

    add_table(out, TableId::CARD_SUBTYPE, "card_subtype", FrameKind::CARDS);
    col_card(out.back(), "card");
    col_enum(out.back(), "face", Storage::U8, "face");
    col_enum(out.back(), "subtype", Storage::U16, "subtype");

    add_table(out, TableId::CARD_KEYWORD, "card_keyword", FrameKind::CARDS);
    col_card(out.back(), "card");
    col_enum(out.back(), "face", Storage::U8, "face");
    col_enum(out.back(), "kw", Storage::U8, "keyword");
    col_text(out.back(), "text");

    add_table(out, TableId::CARD_ABILITY, "card_ability", FrameKind::CARDS);
    TableDef &a = out.back();
    col_card(a, "card");
    col(a, "index", Storage::U16, "index");
    col_enum(a, "face", Storage::U8, "face");
    col_enum(a, "ability_kind", Storage::U8, "ability_kind");
    col_enum(a, "category", Storage::U8, "effect_kind");
    col_text(a, "key");
    col_text(a, "cost");
    col_bool(a, "tap");
    col_enum(a, "timing", Storage::U8, "timing");
    col_count(a, "limit", Storage::U8);
    col_text(a, "line");
    col_ability_params(a);
    col_count(a, "n_parts", Storage::U8);
    col_count(a, "charm_num", Storage::U8);

    add_table(out, TableId::CARD_PIP, "card_pip", FrameKind::CARDS);
    col_card(out.back(), "card");
    col_enum(out.back(), "face", Storage::U8, "face");
    col_enum(out.back(), "kind", Storage::U8, "pip_kind");
    col(out.back(), "colors", Storage::U8, "mask:color");
    col_count(out.back(), "generic_alt", Storage::U8);

    add_table(out, TableId::DECKLIST, "decklist", FrameKind::DECKLISTS);
    col_seat(out.back(), "player");
    col_text(out.back(), "name");
    col_count(out.back(), "main", Storage::U16);
    col_count(out.back(), "side", Storage::U16);

    add_table(out, TableId::DECKLIST_CARD, "decklist_card", FrameKind::DECKLISTS);
    col_seat(out.back(), "player");
    col_card(out.back(), "card");
    col_count(out.back(), "main", Storage::U8);
    col_count(out.back(), "side", Storage::U8);
}

static void define_snapshot_tables(std::vector<TableDef> &out) {
    add_table(out, TableId::HEADER, "header", FrameKind::STATE);
    col_count(out.back(), "game", Storage::U8);
    col_count(out.back(), "decision", Storage::U32);
    col_enum(out.back(), "knowledge", Storage::U8, "knowledge_level");
    col_bool(out.back(), "terminal");
    col_bool(out.back(), "sideboarding");

    add_table(out, TableId::TURN, "turn", FrameKind::STATE);
    TableDef &t = out.back();
    col_count(t, "turn", Storage::U16);
    col_enum(t, "step", Storage::U8, "step");
    col_seat(t, "active");
    col_seat(t, "priority");
    col_bool(t, "priority_window");
    col_enum(t, "mandatory", Storage::U8, "mandatory_choice");
    col_bool(t, "attackers_declared");
    col_bool(t, "blockers_declared");
    col_enum(t, "day_night", Storage::U8, "day_night");
    col_seat(t, "next_extra_turn");

    add_table(out, TableId::PLAYER, "player", FrameKind::STATE);
    TableDef &p = out.back();
    col_seat(p, "player");
    col(p, "life", Storage::I32, "signed");
    for (const char *m : {"mana_w", "mana_u", "mana_b", "mana_r", "mana_g", "mana_c"}) col_count(p, m, Storage::U16);
    for (const char *z : {"hand", "library", "graveyard", "exile", "sideboard"}) col_count(p, z, Storage::U16);
    col_count(p, "mulligans", Storage::U8);
    col_count(p, "to_bottom", Storage::U8);
    col_bool(p, "passed");
    col_bool(p, "monarch");
    col_bool(p, "city_blessing");
    col_count(p, "extra_turns", Storage::U8);
    col_count(p, "lands_played", Storage::U8);
    col_count(p, "land_drops_left", Storage::U8);
    for (const char *m :
        {"potential_w", "potential_u", "potential_b", "potential_r", "potential_g", "potential_c", "potential_total"})
        col_count(p, m, Storage::U8);
    for (const char *n : {"spells_turn", "noncreature_turn", "instant_sorcery_turn", "spells_game", "drawn_turn"})
        col_count(p, n, Storage::U16);
    col_count(p, "life_gained_turn", Storage::U16);
    col_count(p, "life_lost_turn", Storage::U16);
    col(p, "colors_cast_turn", Storage::U8, "mask:color");
    col_bool(p, "revolt");
    col_bool(p, "drew_from_empty");
    col_bool(p, "companion_used");
    col_count(p, "match_wins", Storage::U8);

    add_table(out, TableId::PLAYER_COUNTER, "player_counter", FrameKind::STATE);
    col_seat(out.back(), "player");
    col_text(out.back(), "kind");
    col_count(out.back(), "count", Storage::U16);

    add_table(out, TableId::OBJECT, "object", FrameKind::STATE);
    TableDef &o = out.back();
    col_handle(o, "handle");
    col_enum(o, "zone", Storage::U8, "zone");
    col_seat(o, "owner");
    col_seat(o, "controller");
    col(o, "pos", Storage::U16, "index");
    col_card(o, "card");
    col_bool(o, "token");
    col_bool(o, "face_down");
    col_bool(o, "transformed");
    col_card(o, "copy_of");
    col_bool(o, "tapped");
    col_bool(o, "sick");
    col_bool(o, "phased_out");
    col_bool(o, "attacking");
    col_bool(o, "blocking");
    col_bool(o, "blocked");
    col_count(o, "damage", Storage::U16);
    col_bool(o, "deathtouch_damage");
    col(o, "power", Storage::I16, "signed");
    col(o, "toughness", Storage::I16, "signed");
    col(o, "types", Storage::U32, "mask:card_type");
    col(o, "supertypes", Storage::U8, "mask:supertype");
    col(o, "colors", Storage::U8, "mask:color");
    col_count(o, "mana_value", Storage::U8);
    col_handle(o, "attached_to");
    col(o, "timestamp", Storage::U32, "index");
    col_bool(o, "entered_turn");
    col_bool(o, "cant_be_blocked");
    col_bool(o, "damage_shielded");
    col_bool(o, "delayed_subject");
    col_count(o, "activations_turn", Storage::U8);
    col_count(o, "resolutions_turn", Storage::U8);
    col_card(o, "chosen_name");
    col_enum(o, "chosen_type", Storage::U16, "subtype");
    col_count(o, "entered_x", Storage::U16);
    col_bool(o, "monstrous");
    col_bool(o, "evoked");
    col_bool(o, "escaped");
    col_bool(o, "unearthed");
    col_bool(o, "cast_from_hand");

    add_table(out, TableId::OBJECT_SUBTYPE, "object_subtype", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_enum(out.back(), "subtype", Storage::U16, "subtype");

    add_table(out, TableId::OBJECT_KEYWORD, "object_keyword", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_enum(out.back(), "kw", Storage::U8, "keyword");
    col_text(out.back(), "text");

    add_table(out, TableId::OBJECT_COUNTER, "object_counter", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_text(out.back(), "kind");
    col_count(out.back(), "count", Storage::U16);

    add_table(out, TableId::OBJECT_ABILITY, "object_ability", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_abref(out.back());
    col_bool(out.back(), "granted");
    col_count(out.back(), "activations_turn", Storage::U8);
    col_bool(out.back(), "exhausted");

    add_table(out, TableId::OBJECT_PROVENANCE, "object_provenance", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_handle(out.back(), "src");
    col_card(out.back(), "src_card");
    col_enum(out.back(), "layer", Storage::U8, "layer");
    col_enum(out.back(), "duration", Storage::U8, "duration");

    add_table(out, TableId::OBJECT_LINK, "object_link", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_handle(out.back(), "linked");
    col_enum(out.back(), "kind", Storage::U8, "link_kind");

    add_table(out, TableId::COMBAT, "combat", FrameKind::STATE);
    col_handle(out.back(), "attacker");
    col_enum(out.back(), "target_kind", Storage::U8, "attack_target_kind");
    col_seat(out.back(), "target_player");
    col_handle(out.back(), "target");
    col_handle(out.back(), "blocker");
    col_count(out.back(), "assigned", Storage::U16);

    add_table(out, TableId::STACK, "stack", FrameKind::STATE);
    TableDef &s = out.back();
    col_handle(s, "handle");
    col(s, "pos", Storage::U16, "index");
    col_enum(s, "kind", Storage::U8, "stack_kind");
    col_seat(s, "controller");
    col_handle(s, "source");
    col_card(s, "source_card");
    col_abref(s);
    col(s, "x", Storage::I16, "signed");
    col_count(s, "amount", Storage::U16);
    col_count(s, "kicked", Storage::U8);
    col_count(s, "replicate", Storage::U8);
    col(s, "variant", Storage::U16, "mask:cast_variant");
    col_bool(s, "copy");
    col_bool(s, "cant_be_countered");
    col_bool(s, "gift_promised");
    col_count(s, "mana_spent", Storage::U8);
    col(s, "colors_spent", Storage::U8, "mask:color");
    col(s, "delayed", Storage::U32, "index");

    add_table(out, TableId::STACK_MODE, "stack_mode", FrameKind::STATE);
    col_handle(out.back(), "stack");
    col(out.back(), "mode", Storage::U8, "index");

    add_table(out, TableId::STACK_TARGET, "stack_target", FrameKind::STATE);
    col_handle(out.back(), "stack");
    col(out.back(), "order", Storage::U8, "index");
    col_enum(out.back(), "kind", Storage::U8, "target_kind");
    col_handle(out.back(), "target");
    col_seat(out.back(), "target_player");
    col(out.back(), "lki", Storage::U16, "row:lki");

    add_table(out, TableId::DELAYED_TRIGGER, "delayed_trigger", FrameKind::STATE);
    TableDef &d = out.back();
    col(d, "seq", Storage::U32, "index");
    col_seat(d, "controller");
    col_enum(d, "state", Storage::U8, "delayed_state");
    col_handle(d, "stack");
    col_handle(d, "creator");
    col_card(d, "creator_card");
    col_enum(d, "fire", Storage::U8, "fire_kind");
    col_count(d, "fire_turn", Storage::U16);
    col_bool(d, "fires_this_turn");
    col_bool(d, "expires_eot");
    col_abref(d);

    add_table(out, TableId::DELAYED_SUBJECT, "delayed_subject", FrameKind::STATE);
    col(out.back(), "trigger", Storage::U16, "row:delayed_trigger");
    col_handle(out.back(), "subject");
    col_card(out.back(), "subject_card");

    add_table(out, TableId::EMBLEM, "emblem", FrameKind::STATE);
    col_seat(out.back(), "controller");
    col_card(out.back(), "source_card");

    add_table(out, TableId::PLAYER_EFFECT, "player_effect", FrameKind::STATE);
    col_seat(out.back(), "player");
    col_enum(out.back(), "kind", Storage::U8, "player_effect_kind");
    col(out.back(), "colors", Storage::U8, "mask:color");
    col_enum(out.back(), "duration", Storage::U8, "duration");
    col_card(out.back(), "source_card");
    col_text(out.back(), "filter");

    add_table(out, TableId::PLAY_PERMISSION, "play_permission", FrameKind::STATE);
    col_handle(out.back(), "obj");
    col_seat(out.back(), "player");
    col_enum(out.back(), "kind", Storage::U8, "permission_kind");
    col_enum(out.back(), "resource", Storage::U8, "permission_resource");
    col_count(out.back(), "amount", Storage::U16);
    col_bool(out.back(), "expires_turn");

    add_table(out, TableId::SHIELD, "shield", FrameKind::STATE);
    col_enum(out.back(), "kind", Storage::U8, "shield_kind");
    col_handle(out.back(), "obj");
    col_bool(out.back(), "as_source");
    col_bool(out.back(), "as_target");

    add_table(out, TableId::LKI, "lki", FrameKind::STATE);
    col_card(out.back(), "card");
    col_bool(out.back(), "token");
    col_seat(out.back(), "controller");
    col(out.back(), "power", Storage::I16, "signed");
    col(out.back(), "toughness", Storage::I16, "signed");
    col(out.back(), "types", Storage::U32, "mask:card_type");
    col(out.back(), "colors", Storage::U8, "mask:color");

    add_table(out, TableId::REVEAL, "reveal", FrameKind::STATE);
    col_handle(out.back(), "handle");
    col_enum(out.back(), "zone", Storage::U8, "zone");
    col_seat(out.back(), "owner");
    col(out.back(), "pos", Storage::U16, "index");
    col_card(out.back(), "card");
    col_enum(out.back(), "window", Storage::U8, "window_kind");
    col_enum(out.back(), "until", Storage::U8, "window_end");
    col_seat(out.back(), "to");

    add_table(out, TableId::PENDING, "pending", FrameKind::STATE);
    col_enum(out.back(), "tag", Storage::U8, "pending_tag");
    col_seat(out.back(), "chooser");
    col_handle(out.back(), "source");
    col_card(out.back(), "source_card");
    col_seat(out.back(), "source_controller");

    add_table(out, TableId::PENDING_COST, "pending_cost", FrameKind::STATE);
    TableDef &pc = out.back();
    col_handle(pc, "for");
    col_card(pc, "for_card");
    col_enum(pc, "part", Storage::U8, "cost_part");
    for (const char *r : {"rem_w", "rem_u", "rem_b", "rem_r", "rem_g", "rem_c", "rem_generic", "rem_x"})
        col_count(pc, r, Storage::U8);
    col_count(pc, "rem_life", Storage::U16);
    col_count(pc, "rem_energy", Storage::U8);
    col_count(pc, "rem_discard", Storage::U8);
    col_count(pc, "paid", Storage::U8);

    add_table(out, TableId::PENDING_COST_PIP, "pending_cost_pip", FrameKind::STATE);
    col(out.back(), "cost", Storage::U16, "row:pending_cost");
    col_enum(out.back(), "kind", Storage::U8, "pip_kind");
    col(out.back(), "colors", Storage::U8, "mask:color");
    col_count(out.back(), "generic_alt", Storage::U8);

    add_table(out, TableId::MENU, "menu", FrameKind::STATE);
    TableDef &m = out.back();
    col_enum(m, "category", Storage::U8, "action_category");
    col_enum(m, "type", Storage::U8, "action_type");
    col_handle(m, "source");
    col_handle(m, "target");
    col_card(m, "source_card");
    col_seat(m, "source_player");
    col_seat(m, "target_player");
    col_abref(m);
    col(m, "ordinal", Storage::I16, "signed");
    col(m, "variant", Storage::U16, "mask:cast_variant");
    col_bool(m, "card_public");
    col_text(m, "description");

    add_table(out, TableId::MATCH, "match", FrameKind::STATE);
    col_count(out.back(), "game", Storage::U8);
    col_seat(out.back(), "plays_first");
    col_bool(out.back(), "sideboard_phase");
    col_count(out.back(), "sb_swaps", Storage::U8);
    col(out.back(), "sb_drift", Storage::I8, "signed");
    col_bool(out.back(), "closed_decklist");

    add_table(out, TableId::OWN_DECK, "own_deck", FrameKind::STATE);
    col_seat(out.back(), "player");
    col_card(out.back(), "card");
    col_count(out.back(), "main", Storage::U8);
    col_count(out.back(), "side", Storage::U8);
}

static void define_event_tables(std::vector<TableDef> &out) {
    add_table(out, TableId::EVENT, "event", FrameKind::STATE);
    TableDef &e = out.back();
    col(e, "seq", Storage::U32, "index");
    col_enum(e, "kind", Storage::U8, "event_kind");
    col_seat(e, "player");
    col_enum(e, "zone", Storage::U8, "zone");
    col_enum(e, "to_zone", Storage::U8, "zone");
    col_seat(e, "zone_owner");
    col_seat(e, "to_owner");
    col_count(e, "count", Storage::U16);
    col_libpos(e, "range");
    col_bool(e, "order_known");
    col(e, "batch", Storage::U16, "index");
    col_seat(e, "to");
    col_handle(e, "source");
    col_card(e, "source_card");
    col_count(e, "turn", Storage::U16);
    col_bool(e, "opening");
    col_count(e, "game", Storage::U8);

    add_table(out, TableId::EVENT_CARD, "event_card", FrameKind::STATE);
    TableDef &c = out.back();
    col(c, "event", Storage::U32, "row:event");
    col_handle(c, "handle_before");
    col_handle(c, "handle_after");
    col_card(c, "card");
    col_libpos(c, "from");
    col_libpos(c, "to");
    col_bool(c, "face_down");
    col_bool(c, "token");
    col_enum(c, "hand_subset", Storage::U8, "hand_subset");
    col(c, "draw_seq", Storage::U32, "index");
    col(c, "draw_idx", Storage::U16, "index");

    add_table(out, TableId::EVENT_PLAYER, "event_player", FrameKind::STATE);
    col(out.back(), "event", Storage::U32, "row:event");
    col_seat(out.back(), "player");
    col_count(out.back(), "library", Storage::U16);
    col_count(out.back(), "sideboard", Storage::U16);
    col_count(out.back(), "wins", Storage::U8);
}

static void define_knowledge_tables(std::vector<TableDef> &out) {
    add_table(out, TableId::K_META, "k_meta", FrameKind::STATE);
    col_enum(out.back(), "level", Storage::U8, "knowledge_level");
    col_bool(out.back(), "exact");
    col_count(out.back(), "steps", Storage::U32);

    add_table(out, TableId::K_SEEN, "k_seen", FrameKind::STATE);
    col_seat(out.back(), "player");
    col_card(out.back(), "card");
    col_count(out.back(), "game", Storage::U8);
    col_count(out.back(), "match", Storage::U8);

    add_table(out, TableId::K_POOL, "k_pool", FrameKind::STATE);
    col_seat(out.back(), "owner");
    col_enum(out.back(), "kind", Storage::U8, "pool_kind");
    col_count(out.back(), "size", Storage::U16);
    col_libpos(out.back(), "range");
    col_handle(out.back(), "obj");

    add_table(out, TableId::K_POOL_CARD, "k_pool_card", FrameKind::STATE);
    col(out.back(), "pool", Storage::U16, "row:k_pool");
    col_card(out.back(), "card");
    col_count(out.back(), "min", Storage::U8);
    col_count(out.back(), "max", Storage::U8);

    add_table(out, TableId::K_POSITION, "k_position", FrameKind::STATE);
    col_seat(out.back(), "owner");
    col_libpos(out.back(), "at");
    col_card(out.back(), "card");
    col(out.back(), "pool", Storage::U16, "row:k_pool");

    add_table(out, TableId::K_KNOWN_HAND, "k_known_hand", FrameKind::STATE);
    col_card(out.back(), "card");
    col_count(out.back(), "copies", Storage::U8);
}

static void finish_table(TableDef &t) {
    uint32_t off = 0;
    for (ColumnDef &c : t.columns) {
        c.offset = off;
        off += storage_width(c.storage);
    }
    t.row_width = off;
}

static std::vector<TableDef> build_table_defs() {
    std::vector<TableDef> out;
    define_schema_tables(out);
    define_cards_tables(out);
    define_snapshot_tables(out);
    define_event_tables(out);
    define_knowledge_tables(out);
    for (TableDef &t : out) finish_table(t);
    for (size_t i = 1; i < out.size(); ++i)
        if (static_cast<uint16_t>(out[i - 1].id) >= static_cast<uint16_t>(out[i].id)) {
            fprintf(stderr, "ERROR: rmstate schema: table %s is out of id order\n", out[i].name.c_str());
            abort();
        }
    return out;
}

const std::vector<TableDef> &table_defs() {
    static const std::vector<TableDef> defs = build_table_defs();
    return defs;
}

const TableDef &table_def(TableId id) {
    const TableDef *d = find_table_def(static_cast<uint16_t>(id));
    if (!d) {
        fprintf(stderr, "ERROR: rmstate schema: no table %u\n", static_cast<unsigned>(id));
        abort();
    }
    return *d;
}

const TableDef *find_table_def(uint16_t id) {
    // Indexed by id once: a row of every frame looks its table up.
    static const std::vector<const TableDef *> by_id = [] {
        std::vector<const TableDef *> v;
        for (const TableDef &t : table_defs()) {
            const size_t i = static_cast<uint16_t>(t.id);
            if (v.size() <= i) v.resize(i + 1, nullptr);
            v[i] = &t;
        }
        return v;
    }();
    return id < by_id.size() ? by_id[id] : nullptr;
}

uint32_t storage_width(Storage storage) {
    switch (storage) {
        case Storage::U8:
        case Storage::I8:
            return 1;
        case Storage::U16:
        case Storage::I16:
            return 2;
        case Storage::U32:
        case Storage::I32:
        case Storage::F32:
        case Storage::TEXT:
            return 4;
        case Storage::U64:
        case Storage::I64:
            return 8;
    }
    return 0;
}

#define RMSTATE_ENUM_NAME(name) #name,
std::vector<EnumTable> format_enums() {
    return {
        {"frame_kind", {RMSTATE_FRAME_KIND(RMSTATE_ENUM_NAME)}},
        {"viewer", {RMSTATE_VIEWER(RMSTATE_ENUM_NAME)}},
        {"storage", {RMSTATE_STORAGE(RMSTATE_ENUM_NAME)}},
        {"hand_subset", {RMSTATE_HAND_SUBSET(RMSTATE_ENUM_NAME)}},
        {"event_kind", {RMSTATE_EVENT_KIND(RMSTATE_ENUM_NAME)}},
        {"lib_anchor", {RMSTATE_LIB_ANCHOR(RMSTATE_ENUM_NAME)}},
        {"face", {RMSTATE_FACE(RMSTATE_ENUM_NAME)}},
        {"layout", {RMSTATE_LAYOUT(RMSTATE_ENUM_NAME)}},
        {"ability_kind", {RMSTATE_ABILITY_KIND(RMSTATE_ENUM_NAME)}},
        {"timing", {RMSTATE_TIMING(RMSTATE_ENUM_NAME)}},
        {"pip_kind", {RMSTATE_PIP_KIND(RMSTATE_ENUM_NAME)}},
        {"stack_kind", {RMSTATE_STACK_KIND(RMSTATE_ENUM_NAME)}},
        {"target_kind", {RMSTATE_TARGET_KIND(RMSTATE_ENUM_NAME)}},
        {"attack_target_kind", {RMSTATE_ATTACK_TARGET_KIND(RMSTATE_ENUM_NAME)}},
        {"delayed_state", {RMSTATE_DELAYED_STATE(RMSTATE_ENUM_NAME)}},
        {"cost_part", {RMSTATE_COST_PART(RMSTATE_ENUM_NAME)}},
        {"layer", {RMSTATE_LAYER(RMSTATE_ENUM_NAME)}},
        {"duration", {RMSTATE_DURATION(RMSTATE_ENUM_NAME)}},
        {"link_kind", {RMSTATE_LINK_KIND(RMSTATE_ENUM_NAME)}},
        {"player_effect_kind", {RMSTATE_PLAYER_EFFECT_KIND(RMSTATE_ENUM_NAME)}},
        {"permission_kind", {RMSTATE_PERMISSION_KIND(RMSTATE_ENUM_NAME)}},
        {"permission_resource", {RMSTATE_PERMISSION_RESOURCE(RMSTATE_ENUM_NAME)}},
        {"shield_kind", {RMSTATE_SHIELD_KIND(RMSTATE_ENUM_NAME)}},
        {"cast_variant", {RMSTATE_CAST_VARIANT(RMSTATE_ENUM_NAME)}},
        {"knowledge_level", {RMSTATE_KNOWLEDGE_LEVEL(RMSTATE_ENUM_NAME)}},
        {"pool_kind", {RMSTATE_POOL_KIND(RMSTATE_ENUM_NAME)}},
        {"amount_kind", {RMSTATE_AMOUNT_KIND(RMSTATE_ENUM_NAME)}},
        {"target_class", {RMSTATE_TARGET_CLASS(RMSTATE_ENUM_NAME)}},
        {"part_kind", {RMSTATE_PART_KIND(RMSTATE_ENUM_NAME)}},
    };
}
#undef RMSTATE_ENUM_NAME

// ── FrameWriter ───────────────────────────────────────────────────────────────────────────────

static void append_le(std::string &out, uint64_t v, uint32_t width) {
    char b[8];
    for (uint32_t i = 0; i < width; ++i) b[i] = static_cast<char>((v >> (8 * i)) & 0xFFu);
    out.append(b, width);
}

static void check_failed(const char *what, const TableDef *def, size_t col) {
    fprintf(stderr, "ERROR: rmstate frame: %s (table %s, column %zu%s%s)\n", what, def ? def->name.c_str() : "?", col,
        def && col < def->columns.size() ? " " : "",
        def && col < def->columns.size() ? def->columns[col].name.c_str() : "");
    abort();
}

FrameWriter::FrameWriter(FrameKind kind, Viewer viewer, uint64_t schema_hash)
    : kind_(kind), viewer_(viewer), schema_hash_(schema_hash) {}

TableBuf &FrameWriter::buf(TableId id) {
    const uint16_t i = static_cast<uint16_t>(id);
    if (i < BUF_CACHE && buf_cache_[i]) return *buf_cache_[i];
    TableBuf &b = tables_[i];
    if (i < BUF_CACHE) buf_cache_[i] = &b;
    return b;
}

uint32_t FrameWriter::row_count(TableId id) const {
    auto it = tables_.find(static_cast<uint16_t>(id));
    return it == tables_.end() ? 0 : it->second.rows;
}

FrameWriter::Row::Row(FrameWriter &w, TableId id) : buf_(&w.buf(id)), def_(&table_def(id)) {
#ifndef NDEBUG
    if (def_->frame_kind != w.kind_ && id != TableId::STRINGS)
        check_failed("table written to the wrong frame kind", def_, 0);
    if (buf_->bytes.size() != static_cast<size_t>(buf_->rows) * def_->row_width)
        check_failed("row started before the previous one was complete", def_, 0);
#endif
    at_ = buf_->bytes.size();
    buf_->bytes.resize(at_ + def_->row_width);
}

FrameWriter::Row::~Row() {
#ifndef NDEBUG
    if (col_ != def_->columns.size()) check_failed("row ended with columns unwritten", def_, col_);
#endif
    buf_->rows++;
}

void FrameWriter::Row::put(Storage s, uint64_t bits) {
#ifndef NDEBUG
    if (col_ >= def_->columns.size()) check_failed("more values than columns", def_, col_);
    if (def_->columns[col_].storage != s) check_failed("value written with the wrong storage type", def_, col_);
#endif
    const uint32_t width = storage_width(s);
#ifndef NDEBUG
    // An unsigned value must fit its width (all ones = none is in range by construction).
    if (s == Storage::U8 || s == Storage::U16 || s == Storage::U32) {
        if (width < 8 && (bits >> (8 * width)) != 0) check_failed("value out of the column's range", def_, col_);
    }
#endif
    char *cell = &buf_->bytes[at_ + def_->columns[col_].offset];
    for (uint32_t i = 0; i < width; ++i) cell[i] = static_cast<char>((bits >> (8 * i)) & 0xFFu);
    ++col_;
}

FrameWriter::Row &FrameWriter::Row::u8(uint64_t v) {
    put(Storage::U8, v);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::u16(uint64_t v) {
    put(Storage::U16, v);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::u32(uint64_t v) {
    put(Storage::U32, v);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::u64(uint64_t v) {
    put(Storage::U64, v);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::i8(int64_t v) {
#ifndef NDEBUG
    if (v < -128 || v > 127) check_failed("value out of the column's range", def_, col_);
#endif
    put(Storage::I8, static_cast<uint64_t>(v) & 0xFFu);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::i16(int64_t v) {
#ifndef NDEBUG
    if (v < -32768 || v > 32767) check_failed("value out of the column's range", def_, col_);
#endif
    put(Storage::I16, static_cast<uint64_t>(v) & 0xFFFFu);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::i32(int64_t v) {
    put(Storage::I32, static_cast<uint64_t>(v) & 0xFFFFFFFFu);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::i64(int64_t v) {
    put(Storage::I64, static_cast<uint64_t>(v));
    return *this;
}
FrameWriter::Row &FrameWriter::Row::f32(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    put(Storage::F32, bits);
    return *this;
}
FrameWriter::Row &FrameWriter::Row::text(const std::string &s) {
    buf_->texts.emplace_back(at_ + def_->columns[col_].offset, s);
    put(Storage::TEXT, 0);
    return *this;
}

std::string FrameWriter::finish() {
    // The string pool: offset 0 is the empty string; each distinct string is stored once, at its
    // first use in table, row and column order (tables_ iterates in ascending id).
    std::string pool(1, '\0');
    std::map<std::string, uint32_t> offsets;
    for (auto &kv : tables_) {
        if (kv.first == static_cast<uint16_t>(TableId::STRINGS)) continue;
        for (const auto &cell : kv.second.texts) {
            uint32_t off = 0;
            if (!cell.second.empty()) {
                auto it = offsets.find(cell.second);
                if (it == offsets.end()) {
                    off = static_cast<uint32_t>(pool.size());
                    offsets.emplace(cell.second, off);
                    pool += cell.second;
                    pool.push_back('\0');
                } else {
                    off = it->second;
                }
            }
            for (uint32_t i = 0; i < 4; ++i)
                kv.second.bytes[cell.first + i] = static_cast<char>((off >> (8 * i)) & 0xFFu);
        }
    }
    TableBuf &strings = tables_[static_cast<uint16_t>(TableId::STRINGS)];
    strings.bytes = pool;
    strings.rows = static_cast<uint32_t>(pool.size());

    uint32_t n_tables = 0;
    for (const auto &kv : tables_)
        if (kv.second.rows > 0) ++n_tables;

    std::string out;
    size_t total = FRAME_HEADER_SIZE;
    for (const auto &kv : tables_) total += TABLE_HEADER_SIZE + kv.second.bytes.size() + ROW_ALIGN;
    out.reserve(total);
    append_le(out, FRAME_MAGIC, 4);
    append_le(out, FORMAT_VERSION, 2);
    append_le(out, static_cast<uint8_t>(kind_), 1);
    append_le(out, static_cast<uint8_t>(viewer_), 1);
    append_le(out, schema_hash_, 8);
    append_le(out, n_tables, 4);
    append_le(out, 0, 4);
    for (const auto &kv : tables_) {
        const TableBuf &t = kv.second;
        if (t.rows == 0) continue;
        const TableDef *def = find_table_def(kv.first);
        const uint32_t width = def ? def->row_width : 0;
#ifndef NDEBUG
        if (!def || t.bytes.size() != static_cast<size_t>(t.rows) * width)
            check_failed("table bytes disagree with its row width", def, 0);
#endif
        append_le(out, kv.first, 2);
        append_le(out, 0, 2);
        append_le(out, width, 4);
        append_le(out, t.rows, 4);
        append_le(out, 0, 4);
        out += t.bytes;
        while (out.size() % ROW_ALIGN != 0) out.push_back('\0');
    }
    return out;
}

uint64_t fnv1a64(const void *data, size_t len) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string build_schema_frame(Viewer viewer, const std::vector<EnumTable> &enums, uint64_t *hash_out) {
    FrameWriter w(FrameKind::SCHEMA, viewer, 0);
    for (const TableDef &t : table_defs()) {
        w.row(TableId::SCHEMA_TABLE)
            .u16(static_cast<uint16_t>(t.id))
            .text(t.name)
            .u8(static_cast<uint8_t>(t.frame_kind))
            .u32(t.row_width);
    }
    for (const TableDef &t : table_defs()) {
        for (size_t i = 0; i < t.columns.size(); ++i) {
            const ColumnDef &c = t.columns[i];
            w.row(TableId::SCHEMA_COLUMN)
                .u16(static_cast<uint16_t>(t.id))
                .u16(i)
                .text(c.name)
                .u8(static_cast<uint8_t>(c.storage))
                .u16(c.offset)
                .text(c.tag)
                .u8(c.has_range ? 1 : 0)
                .i64(c.min)
                .i64(c.max);
        }
    }
    for (const EnumTable &e : enums)
        for (size_t v = 0; v < e.values.size(); ++v) w.row(TableId::SCHEMA_ENUM).text(e.name).u16(v).text(e.values[v]);
    std::string frame = w.finish();
    const uint64_t hash = fnv1a64(frame.data() + FRAME_HEADER_SIZE, frame.size() - FRAME_HEADER_SIZE);
    for (uint32_t i = 0; i < 8; ++i) frame[8 + i] = static_cast<char>((hash >> (8 * i)) & 0xFFu);
    if (hash_out) *hash_out = hash;
    return frame;
}

std::string frame_record(const std::string &payload) {
    return "RMSTATE " + std::to_string(payload.size()) + "\n" + payload;
}

}  // namespace rmstate
