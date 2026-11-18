#include "table_model.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>

#include "../../src/agents/decision_view.h"
#include "settings.h"

namespace {

// The frame's tables and the enum values the model reads.
struct Frame {
        const agents::SeatStream &s;
        const rmwork::CardBook &book;
        const rmstate::DecodedFrame &frame;
        agents::Table obj, subtype, counter, player, pcounter, turn, stack, starget, smode, combat, menu, lki, emblem,
            peffect, header, cost;
        int z_battlefield, z_hand;
        int k_spell, k_activated, k_triggered, k_copy, t_player;
        uint64_t back_face_bit = 0;
        std::map<uint64_t, size_t> rows;  // handle -> object row
        Frame(const agents::SeatStream &stream, const rmwork::CardBook &b, const rmstate::DecodedFrame &f);
        agents::Table table(const char *name) const { return agents::Table(frame, s.schema(), name); }
};

}  // namespace

static int seat_of(uint64_t seat);
static std::string seat_label(int seat);
static std::string humanize(const std::string &name);
static std::string counter_label(const std::string &kind);
static void read_face(const Frame &f, size_t row, int *card, bool *back);
static rmwork::TableObject read_object(const Frame &f, size_t row);
static void read_counters(const Frame &f, std::vector<rmwork::TableObject> *objects);
static void read_subtypes(const Frame &f, std::vector<rmwork::TableObject> *objects);
static void read_tags(const Frame &f, size_t row, rmwork::TableObject *o);
static bool stackable(const rmwork::TableObject &o);
static bool same_land(const rmwork::TableObject &a, const rmwork::TableObject &b);
static void place_piles(rmwork::TableModel *m, const std::vector<int> &order);
static void read_players(const Frame &f, rmwork::TableModel *m);
static std::string effect_label(const Frame &f, size_t row);
static void read_turn(const Frame &f, rmwork::TableModel *m);
static std::string object_label(const Frame &f, uint64_t handle);
static bool x_unannounced(const Frame &f, uint64_t handle);
static void read_stack(const Frame &f, const agents::DecisionView &dv, rmwork::TableModel *m);
static void read_combat(const Frame &f, rmwork::TableModel *m);
static void read_menu(const Frame &f, rmwork::TableModel *m);

Frame::Frame(const agents::SeatStream &stream, const rmwork::CardBook &b, const rmstate::DecodedFrame &f)
    : s(stream), book(b), frame(f) {
    obj = table("object");
    subtype = table("object_subtype");
    counter = table("object_counter");
    player = table("player");
    pcounter = table("player_counter");
    turn = table("turn");
    stack = table("stack");
    starget = table("stack_target");
    smode = table("stack_mode");
    combat = table("combat");
    menu = table("menu");
    lki = table("lki");
    emblem = table("emblem");
    peffect = table("player_effect");
    header = table("header");
    cost = table("pending_cost");
    z_battlefield = s.enum_value("zone", "BATTLEFIELD");
    z_hand = s.enum_value("zone", "HAND");
    k_spell = s.enum_value("stack_kind", "SPELL");
    k_activated = s.enum_value("stack_kind", "ACTIVATED");
    k_triggered = s.enum_value("stack_kind", "TRIGGERED");
    k_copy = s.enum_value("stack_kind", "COPY");
    t_player = s.enum_value("target_kind", "PLAYER");
    const int bf = s.enum_value("cast_variant", "BACK_FACE");
    if (bf >= 0) back_face_bit = 1ull << bf;
    const int c_handle = obj.col("handle");
    for (size_t r = 0; r < obj.rows(); ++r) rows[obj.u(r, c_handle)] = r;
}

// The `seat` enum (1 A, 2 B) as 0 A, 1 B; -1 for none.
static int seat_of(uint64_t seat) {
    return seat == 1 ? 0 : seat == 2 ? 1 : -1;
}

static std::string seat_label(int seat) {
    return seat == 0 ? "Player A" : seat == 1 ? "Player B" : "no player";
}

// HEXPROOF_FROM -> "Hexproof from".
static std::string humanize(const std::string &name) {
    std::string out;
    for (char c : name) out += c == '_' ? ' ' : static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (!out.empty()) out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
    return out;
}

// P1P1 -> "+1/+1", M1M1 -> "-1/-1", LORE -> "Lore".
static std::string counter_label(const std::string &kind) {
    if (kind == "P1P1") return "+1/+1";
    if (kind == "M1M1") return "-1/-1";
    return humanize(kind);
}

// The face an object row shows: a copy's copied card, a token's card, a transformed card's back
// face; -1 when the frame does not identify it.
static void read_face(const Frame &f, size_t row, int *card, bool *back) {
    *card = f.obj.card(row, f.obj.col("card"));
    *back = false;
    if (*card < 0) return;
    const int copy = f.obj.card(row, f.obj.col("copy_of"));
    if (copy >= 0) *card = copy;
    if (f.obj.u(row, f.obj.col("token"))) return;
    *back = f.obj.u(row, f.obj.col("transformed")) != 0 && f.book.face(*card, true) != nullptr;
}

static rmwork::TableObject read_object(const Frame &f, size_t row) {
    const agents::Table &o = f.obj;
    rmwork::TableObject t;
    t.handle = o.u(row, o.col("handle"));
    t.controller = std::max(0, seat_of(o.u(row, o.col("controller"))));
    t.owner = std::max(0, seat_of(o.u(row, o.col("owner"))));
    t.in_hand = static_cast<int>(o.u(row, o.col("zone"))) == f.z_hand;
    read_face(f, row, &t.card, &t.back);
    const rmwork::CardFace *face = t.card >= 0 ? f.book.face(t.card, t.back) : nullptr;
    if (t.card >= 0) t.name = face ? face->name : f.s.cards().name(t.card, t.back);
    t.token = o.u(row, o.col("token")) != 0;
    t.face_down = o.u(row, o.col("face_down")) != 0;
    t.tapped = o.u(row, o.col("tapped")) != 0;
    t.sick = o.u(row, o.col("sick")) != 0;
    t.phased_out = o.u(row, o.col("phased_out")) != 0;
    t.attacking = o.u(row, o.col("attacking")) != 0;
    t.blocking = o.u(row, o.col("blocking")) != 0;
    t.blocked = o.u(row, o.col("blocked")) != 0;
    const uint64_t types = o.u(row, o.col("types"));
    const rmwork::TypeBits &bits = f.book.bits();
    t.creature = (types & bits.creature) != 0;
    t.land = (types & bits.land) != 0 && !t.creature;
    t.planeswalker = (types & bits.planeswalker) != 0;
    t.colors = bits.wubrg(o.u(row, o.col("colors")));
    t.has_pt = t.creature;
    t.power = static_cast<int>(o.s(row, o.col("power")));
    t.toughness = static_cast<int>(o.s(row, o.col("toughness")));
    if (t.has_pt) {
        const bool printed = face && face->has_pt && !t.face_down;
        const int pp = t.face_down ? 2 : printed ? face->power : t.power;
        const int pt = t.face_down ? 2 : printed ? face->toughness : t.toughness;
        t.printed_power = pp;
        t.printed_toughness = pt;
        t.pt_differs = (t.face_down || printed) && (pp != t.power || pt != t.toughness);
    }
    t.damage = static_cast<int>(o.u(row, o.col("damage")));
    t.deathtouch_damage = o.u(row, o.col("deathtouch_damage")) != 0;
    t.attached_to = o.u(row, o.col("attached_to"));
    read_tags(f, row, &t);
    return t;
}

// Each object's counters: loyalty as its loyalty, the other kinds as badges.
static void read_counters(const Frame &f, std::vector<rmwork::TableObject> *objects) {
    std::map<uint64_t, size_t> at;
    for (size_t i = 0; i < objects->size(); ++i) at[(*objects)[i].handle] = i;
    const agents::Table &c = f.counter;
    const int c_obj = c.col("obj"), c_kind = c.col("kind"), c_count = c.col("count");
    for (size_t r = 0; r < c.rows(); ++r) {
        auto it = at.find(c.u(r, c_obj));
        if (it == at.end()) continue;
        rmwork::TableObject &o = (*objects)[it->second];
        const std::string &kind = c.text(r, c_kind);
        const int n = static_cast<int>(c.u(r, c_count));
        if (kind == "LOYALTY") {
            o.loyalty = n;
        } else if (n > 0) {
            o.counters.push_back({counter_label(kind), n});
        }
    }
}

static void read_subtypes(const Frame &f, std::vector<rmwork::TableObject> *objects) {
    std::map<uint64_t, size_t> at;
    for (size_t i = 0; i < objects->size(); ++i) at[(*objects)[i].handle] = i;
    const agents::Table &st = f.subtype;
    const int c_obj = st.col("obj"), c_sub = st.col("subtype");
    for (size_t r = 0; r < st.rows(); ++r) {
        auto it = at.find(st.u(r, c_obj));
        if (it != at.end()) (*objects)[it->second].subtypes.push_back(f.s.enum_name("subtype", st.u(r, c_sub)));
    }
}

// The designations an object row carries, as short labels.
static void read_tags(const Frame &f, size_t row, rmwork::TableObject *o) {
    const agents::Table &t = f.obj;
    static const std::pair<const char *, const char *> FLAGS[] = {{"monstrous", "Monstrous"}, {"exerted", "Exerted"},
        {"prepared", "Prepared"}, {"bestowed", "Bestowed"}, {"evoked", "Evoked"}, {"escaped", "Escaped"},
        {"unearthed", "Unearthed"}};
    for (const auto &flag : FLAGS) {
        const int c = t.col(flag.first);
        if (c >= 0 && t.u(row, c)) o->tags.push_back(flag.second);
    }
    const int named = t.col("chosen_name");
    if (named >= 0 && !t.none(row, named)) o->tags.push_back("Named: " + f.s.cards().name(t.card(row, named)));
    const int type = t.col("chosen_type");
    if (type >= 0 && !t.none(row, type)) o->tags.push_back("Type: " + f.s.enum_name("subtype", t.u(row, type)));
    const int color = t.col("chosen_color");
    if (color >= 0 && t.u(row, color) > 0)
        o->tags.push_back("Colour: " + humanize(f.s.enum_name("color", t.u(row, color) - 1)));
    const int x = t.col("entered_x");
    if (x >= 0 && t.u(row, x) > 0) o->tags.push_back("X=" + std::to_string(t.u(row, x)));
}

// A land that may stand in a stack of identical lands: untapped, identified, with nothing on it.
static bool stackable(const rmwork::TableObject &o) {
    return o.land && o.card >= 0 && !o.tapped && !o.phased_out && !o.face_down && o.counters.empty() && o.loyalty < 0 &&
           o.damage == 0 && o.tags.empty() && !o.attacking && !o.blocking;
}

static bool same_land(const rmwork::TableObject &a, const rmwork::TableObject &b) {
    return a.card == b.card && a.back == b.back && a.controller == b.controller && a.token == b.token;
}

// Battlefield objects (in `order`) into their rows' piles: each host with its attachments, the
// identical lands stacked.
static void place_piles(rmwork::TableModel *m, const std::vector<int> &order) {
    std::map<uint64_t, int> on_field;
    for (int i : order) on_field[m->objects[static_cast<size_t>(i)].handle] = i;
    std::map<int, std::vector<int>> attachments;
    std::vector<int> hosts;
    for (int i : order) {
        const rmwork::TableObject &o = m->objects[static_cast<size_t>(i)];
        auto host = o.attached_to ? on_field.find(o.attached_to) : on_field.end();
        if (host != on_field.end() && host->second != i) {
            attachments[host->second].push_back(i);
        } else {
            hosts.push_back(i);
        }
    }
    // An attachment's own attachments join the root host's pile.
    for (int h : hosts) {
        std::vector<int> &list = attachments[h];
        for (size_t k = 0; k < list.size(); ++k) {
            auto sub = attachments.find(list[k]);
            if (sub == attachments.end() || sub->first == h) continue;
            for (int a : sub->second) list.push_back(a);
            sub->second.clear();
        }
    }
    for (int h : hosts) {
        const rmwork::TableObject &o = m->objects[static_cast<size_t>(h)];
        const bool top = o.controller == m->top;
        const rmwork::TableRowId row = o.land ? (top ? rmwork::ROW_TOP_LANDS : rmwork::ROW_BOTTOM_LANDS)
                                              : (top ? rmwork::ROW_TOP_OTHER : rmwork::ROW_BOTTOM_OTHER);
        std::vector<rmwork::TablePile> &piles = m->rows[row];
        const std::vector<int> &att = attachments[h];
        if (att.empty() && stackable(o)) {
            bool joined = false;
            for (rmwork::TablePile &p : piles) {
                if (!p.land_stack) continue;
                if (same_land(m->objects[static_cast<size_t>(p.objects[0])], o)) {
                    p.objects.push_back(h);
                    joined = true;
                    break;
                }
            }
            if (joined) continue;
            rmwork::TablePile p;
            p.objects.push_back(h);
            p.land_stack = true;
            piles.push_back(p);
            continue;
        }
        rmwork::TablePile p;
        p.objects.push_back(h);
        for (int a : att) p.objects.push_back(a);
        piles.push_back(p);
    }
    // A "stack" of one land is a plain pile.
    for (auto &row : m->rows)
        for (rmwork::TablePile &p : row)
            if (p.land_stack && p.objects.size() == 1) p.land_stack = false;
}

static void read_players(const Frame &f, rmwork::TableModel *m) {
    for (int s = 0; s < 2; ++s) {
        m->players[static_cast<size_t>(s)].seat = s;
        m->players[static_cast<size_t>(s)].name = seat_label(s);
    }
    const agents::Table &p = f.player;
    for (size_t r = 0; r < p.rows(); ++r) {
        const int s = seat_of(p.u(r, p.col("player")));
        if (s < 0) continue;
        rmwork::TablePlayer &pl = m->players[static_cast<size_t>(s)];
        pl.life = p.s(r, p.col("life"));
        pl.hand = static_cast<int>(p.u(r, p.col("hand")));
        pl.library = static_cast<int>(p.u(r, p.col("library")));
        pl.graveyard = static_cast<int>(p.u(r, p.col("graveyard")));
        pl.exile = static_cast<int>(p.u(r, p.col("exile")));
        pl.monarch = p.u(r, p.col("monarch")) != 0;
        pl.city_blessing = p.u(r, p.col("city_blessing")) != 0;
        const uint64_t speed = p.u(r, p.col("speed"));
        pl.speed = speed > 0 ? static_cast<int>(speed) - 1 : -1;
    }
    const agents::Table &c = f.pcounter;
    for (size_t r = 0; r < c.rows(); ++r) {
        const int s = seat_of(c.u(r, c.col("player")));
        if (s < 0) continue;
        rmwork::TablePlayer &pl = m->players[static_cast<size_t>(s)];
        const std::string &kind = c.text(r, c.col("kind"));
        const int n = static_cast<int>(c.u(r, c.col("count")));
        if (kind == "POISON") {
            pl.poison = n;
        } else if (kind == "ENERGY") {
            pl.energy = n;
        } else if (n > 0) {
            pl.counters.push_back({counter_label(kind), n});
        }
    }
    const agents::Table decks(f.s.decklists_frame(), f.s.schema(), "decklist");
    for (size_t r = 0; r < decks.rows(); ++r) {
        const int s = seat_of(decks.u(r, decks.col("player")));
        if (s >= 0) m->players[static_cast<size_t>(s)].deck = decks.text(r, decks.col("name"));
    }
    const agents::Table &e = f.emblem;
    for (size_t r = 0; r < e.rows(); ++r) {
        const int s = seat_of(e.u(r, e.col("controller")));
        const int cid = e.card(r, e.col("source_card"));
        if (s >= 0)
            m->players[static_cast<size_t>(s)].effects.push_back(
                "Emblem: " + (cid >= 0 ? f.s.cards().name(cid) : std::string("?")));
    }
    const agents::Table &pe = f.peffect;
    for (size_t r = 0; r < pe.rows(); ++r) {
        const int s = seat_of(pe.u(r, pe.col("player")));
        if (s >= 0) m->players[static_cast<size_t>(s)].effects.push_back(effect_label(f, r));
    }
}

// "Hexproof from blue (end of turn)", "Flash: Instant", "<card>'s trigger", "<card> effect".
static std::string effect_label(const Frame &f, size_t row) {
    const agents::Table &pe = f.peffect;
    const std::string kind = f.s.enum_name("player_effect_kind", pe.u(row, pe.col("kind")));
    const int cid = pe.card(row, pe.col("source_card"));
    const std::string card = cid >= 0 ? f.s.cards().name(cid) : std::string();
    std::string out;
    if (kind == "HEXPROOF_FROM") {
        out = "Hexproof from";
        const uint64_t colors = pe.u(row, pe.col("colors"));
        const std::vector<std::string> &names = f.s.enum_values("color");
        for (size_t i = 0; i < names.size() && i < 8; ++i)
            if (colors & (1ull << i)) out += " " + humanize(names[i]);
    } else if (kind == "CAST_AS_FLASH") {
        const std::string &filter = pe.text(row, pe.col("filter"));
        out = filter.empty() ? "Flash" : "Flash: " + filter;
    } else if (kind == "FLOATING_TRIGGER") {
        out = card.empty() ? "Delayed trigger" : card + " trigger";
    } else if (kind == "RESOLVED_EFFECT") {
        out = card.empty() ? "Effect" : card + " effect";
    } else {
        out = humanize(kind);
    }
    const std::string duration = f.s.enum_name("duration", pe.u(row, pe.col("duration")));
    if (duration == "END_OF_TURN") out += " (end of turn)";
    if (duration == "UNTIL_YOUR_NEXT_TURN") out += " (until next turn)";
    return out;
}

static void read_turn(const Frame &f, rmwork::TableModel *m) {
    const agents::Table &h = f.header;
    if (h.rows()) {
        m->sideboarding = h.u(0, h.col("sideboarding")) != 0;
        m->terminal = h.u(0, h.col("terminal")) != 0;
    }
    const agents::Table &t = f.turn;
    if (!t.rows()) return;
    m->turn = static_cast<int>(t.u(0, t.col("turn"))) + 1;
    m->active = seat_of(t.u(0, t.col("active")));
    const std::string step = f.s.enum_name("step", t.u(0, t.col("step")));
    const int stop = rmwork::stop_step_of(step);
    if (stop >= 0) {
        m->step = stop;
        m->step_name = step == "FIRST_STRIKE_DAMAGE" ? "First-strike damage"
                       : step == "FIRST_MAIN"        ? "Main 1"
                       : step == "SECOND_MAIN"       ? "Main 2"
                       : step == "BEGIN_COMBAT"      ? "Beginning of combat"
                                                     : humanize(step);
    }
    const std::string dn = f.s.enum_name("day_night", t.u(0, t.col("day_night")));
    m->day_night = dn == "DN_DAY" ? "Day" : dn == "DN_NIGHT" ? "Night" : "";
}

// An object's name as the frame shows it ("a face-down card" when it does not identify it);
// "a card" for a handle the frame does not list.
static std::string object_label(const Frame &f, uint64_t handle) {
    auto it = f.rows.find(handle);
    if (it == f.rows.end()) return "a card";
    int card = -1;
    bool back = false;
    read_face(f, it->second, &card, &back);
    if (card < 0) return "a face-down card";
    const rmwork::CardFace *face = f.book.face(card, back);
    return face ? face->name : f.s.cards().name(card, back);
}

// The object is being cast or activated and its X is not chosen yet: its cost still has X to announce.
static bool x_unannounced(const Frame &f, uint64_t handle) {
    const agents::Table &c = f.cost;
    for (size_t r = 0; r < c.rows(); ++r)
        if (c.u(r, c.col("for")) == handle && c.u(r, c.col("rem_x")) > 0) return true;
    return false;
}

static void read_stack(const Frame &f, const agents::DecisionView &dv, rmwork::TableModel *m) {
    const agents::Table &s = f.stack;
    for (size_t r : dv.stack_order()) {
        rmwork::TableStackItem item;
        item.handle = s.u(r, s.col("handle"));
        item.controller = std::max(0, seat_of(s.u(r, s.col("controller"))));
        const int kind = static_cast<int>(s.u(r, s.col("kind")));
        const uint64_t source = s.u(r, s.col("source"));
        const int src = source ? m->find(source) : -1;
        const bool sourceless = kind == f.k_triggered && source == 0 && s.card(r, s.col("ab_card")) < 0 &&
                                s.text(r, s.col("ab_key")).empty();
        if (src >= 0) {
            item.card = m->objects[static_cast<size_t>(src)].card;
            item.back = m->objects[static_cast<size_t>(src)].back;
        } else if (!sourceless) {
            item.card = s.card(r, s.col("source_card"));
            item.back = item.card >= 0 && (s.u(r, s.col("variant")) & f.back_face_bit) &&
                        f.book.face(item.card, true) != nullptr;
        }
        const rmwork::CardFace *face = item.card >= 0 ? f.book.face(item.card, item.back) : nullptr;
        const std::optional<std::string> name = dv.stack_card(r);
        item.name = name ? *name : kind == f.k_spell ? "Face-down spell" : "Ability";
        std::vector<std::string> parts;
        if (kind == f.k_activated) {
            parts.push_back("Activated ability");
        } else if (kind == f.k_triggered) {
            parts.push_back("Triggered ability");
        } else if (kind == f.k_copy || s.u(r, s.col("copy"))) {
            parts.push_back(face ? "Copy · " + face->type_line : "Copy");
        } else {
            parts.push_back(face ? face->type_line : "Spell");
        }
        const int64_t x = s.s(r, s.col("x"));
        if (x >= 0 && kind == f.k_spell && !x_unannounced(f, item.handle)) parts.push_back("X=" + std::to_string(x));
        if (const uint64_t k = s.u(r, s.col("kicked")))
            parts.push_back(k == 1 ? "kicked" : "kicked ×" + std::to_string(k));
        int modes = 0;
        for (size_t mr = 0; mr < f.smode.rows(); ++mr)
            if (f.smode.u(mr, f.smode.col("stack")) == item.handle) ++modes;
        if (modes > 0) parts.push_back(std::to_string(modes) + (modes == 1 ? " mode" : " modes"));
        for (const std::string &p : parts) item.text += (item.text.empty() ? "" : " · ") + p;
        std::vector<std::pair<uint64_t, size_t>> targets;
        const agents::Table &t = f.starget;
        for (size_t tr = 0; tr < t.rows(); ++tr)
            if (t.u(tr, t.col("stack")) == item.handle) targets.emplace_back(t.u(tr, t.col("order")), tr);
        std::sort(targets.begin(), targets.end());
        for (const auto &tt : targets) {
            const size_t tr = tt.second;
            rmwork::StackTarget st;
            if (static_cast<int>(t.u(tr, t.col("kind"))) == f.t_player) {
                st.player = true;
                st.seat = seat_of(t.u(tr, t.col("target_player")));
                st.label = seat_label(st.seat);
            } else if ((st.handle = t.u(tr, t.col("target"))) != 0) {
                st.label = object_label(f, st.handle);
            } else {
                const size_t l = static_cast<size_t>(t.u(tr, t.col("lki")));
                const int cid = !t.none(tr, t.col("lki")) && l < f.lki.rows() ? f.lki.card(l, f.lki.col("card")) : -1;
                st.label = (cid >= 0 ? f.s.cards().name(cid) : std::string("a card")) + " (gone)";
            }
            if (const uint64_t share = t.u(tr, t.col("share"))) st.label += " (" + std::to_string(share) + ")";
            item.targets.push_back(st);
        }
        m->stack.push_back(item);
    }
}

static void read_combat(const Frame &f, rmwork::TableModel *m) {
    const agents::Table &c = f.combat;
    const int c_att = c.col("attacker"), c_blk = c.col("blocker");
    for (size_t r = 0; r < c.rows(); ++r) {
        const uint64_t blocker = c.u(r, c_blk);
        if (blocker != 0) m->blocks.push_back({blocker, c.u(r, c_att)});
    }
}

static void read_menu(const Frame &f, rmwork::TableModel *m) {
    const agents::Table &t = f.menu;
    const int c_src = t.col("source"), c_tgt = t.col("target"), c_sp = t.col("source_player"),
              c_tp = t.col("target_player");
    for (size_t r = 0; r < t.rows(); ++r)
        m->menu.push_back({t.u(r, c_src), t.u(r, c_tgt), seat_of(t.u(r, c_sp)), seat_of(t.u(r, c_tp))});
}

namespace rmwork {

int TableModel::find(uint64_t handle) const {
    auto it = std::lower_bound(by_handle_.begin(), by_handle_.end(), std::make_pair(handle, -1));
    return it != by_handle_.end() && it->first == handle ? it->second : -1;
}

void TableModel::index() {
    by_handle_.clear();
    for (size_t i = 0; i < objects.size(); ++i) by_handle_.emplace_back(objects[i].handle, static_cast<int>(i));
    std::sort(by_handle_.begin(), by_handle_.end());
}

TableModel build_table_model(const agents::SeatStream &stream, const CardBook &book, const rmstate::DecodedFrame &frame,
    int viewer, int decider, bool pregame) {
    const Frame f(stream, book, frame);
    TableModel m;
    m.viewer = viewer;
    m.bottom = viewer == 1 ? 1 : 0;
    m.top = 1 - m.bottom;
    m.decider = decider;
    m.pregame = pregame;
    const agents::Table &o = f.obj;
    const int c_zone = o.col("zone"), c_owner = o.col("owner"), c_ts = o.col("timestamp");
    std::vector<std::pair<uint64_t, int>> field;  // (timestamp, object)
    for (size_t r = 0; r < o.rows(); ++r) {
        const int zone = static_cast<int>(o.u(r, c_zone));
        const bool hand = zone == f.z_hand && seat_of(o.u(r, c_owner)) == m.bottom;
        if (zone != f.z_battlefield && !hand) continue;
        m.objects.push_back(read_object(f, r));
        const int i = static_cast<int>(m.objects.size()) - 1;
        if (hand) {
            m.hand.push_back(i);
        } else {
            field.emplace_back(o.none(r, c_ts) ? UINT64_MAX : o.u(r, c_ts), i);
        }
    }
    read_counters(f, &m.objects);
    read_subtypes(f, &m.objects);
    m.index();
    std::stable_sort(field.begin(), field.end(),
        [](const std::pair<uint64_t, int> &a, const std::pair<uint64_t, int> &b) { return a.first < b.first; });
    std::vector<int> order;
    for (const auto &p : field) order.push_back(p.second);
    place_piles(&m, order);
    read_players(f, &m);
    read_turn(f, &m);
    const agents::DecisionView dv(stream, frame, m.bottom);
    read_stack(f, dv, &m);
    read_combat(f, &m);
    read_menu(f, &m);
    return m;
}

}  // namespace rmwork
