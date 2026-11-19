#include "sideboard_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>

#include "choices_view.h"
#include "evaluations.h"
#include "imgui_internal.h"
#include "modal.h"
#include "paths.h"
#include "table_layout.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

namespace {

// The deck's groups, in the order they are shown; a card goes to the first that fits it.
enum GroupKind {
    G_CREATURE,
    G_PLANESWALKER,
    G_BATTLE,
    G_INSTANT,
    G_SORCERY,
    G_ARTIFACT,
    G_ENCHANTMENT,
    G_LAND,
    G_OTHER,
    G_COUNT
};
const char *const GROUP_LABELS[G_COUNT] = {
    "Creatures", "Planeswalkers", "Battles", "Instants", "Sorceries", "Artifacts", "Enchantments", "Lands", "Other"};

// A pile's id for the magnifier: the card, and which section it is in.
uint64_t pile_id(int card, bool main) {
    return (static_cast<uint64_t>(card) + 1) * 2 + (main ? 0 : 1);
}

}  // namespace

static std::string number(double v, int decimals);
static int group_of(const rmwork::CardFace *face);
static std::string results_text(const rmwork::Match &m, int seat, int before_game);
static std::string short_agent(const std::string &spec);
static std::string moves_text(
    const rmwork::DeckCounts &from, const rmwork::DeckCounts &to, const std::function<std::string(int)> &names);
static std::string row_label(const rmwork::SideboardRow &r, const std::function<std::string(int)> &names);
static void count_field(int have, int want, const char *what, bool exact);
static float count_field_width(int have, int want, const char *what);
static void tag(ImDrawList *dl, ImVec2 at, const char *text, ImU32 bg, ImU32 fg, bool right);

static std::string number(double v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
    return buf;
}

static int group_of(const rmwork::CardFace *face) {
    if (!face) return G_OTHER;
    if (face->creature) return G_CREATURE;
    if (face->planeswalker) return G_PLANESWALKER;
    if (face->land) return G_LAND;
    const std::string &t = face->type_line;
    if (t.find("Battle") != std::string::npos) return G_BATTLE;
    if (t.find("Instant") != std::string::npos) return G_INSTANT;
    if (t.find("Sorcery") != std::string::npos) return G_SORCERY;
    if (t.find("Artifact") != std::string::npos) return G_ARTIFACT;
    if (t.find("Enchantment") != std::string::npos) return G_ENCHANTMENT;
    return G_OTHER;
}

// "won game 1, lost game 2" for `seat`, the games before `before_game` (0-based).
static std::string results_text(const rmwork::Match &m, int seat, int before_game) {
    std::string out;
    const std::vector<int> &w = m.result.game_winners;
    for (int g = 0; g < before_game && g < static_cast<int>(w.size()); ++g) {
        const int x = w[static_cast<size_t>(g)];
        const std::string n = std::to_string(g + 1);
        out +=
            (out.empty() ? "" : ", ") + (x < 0 ? "game " + n + " drawn" : (x == seat ? "won game " : "lost game ") + n);
    }
    return out;
}

// "nn:/runs/x/exports/final.ts.pt" -> "nn:final.ts.pt".
static std::string short_agent(const std::string &spec) {
    const size_t slash = spec.rfind('/');
    if (spec.compare(0, 3, "nn:") == 0 && slash != std::string::npos) return "nn:" + spec.substr(slash + 1);
    return spec;
}

// "+2 Smash to Smithereens, −1 Light Up the Stage": the main deck's moves from `from` to `to`.
static std::string moves_text(
    const rmwork::DeckCounts &from, const rmwork::DeckCounts &to, const std::function<std::string(int)> &names) {
    std::string plus, minus;
    std::map<int, int> all = from.main;
    for (const auto &kv : to.main) all[kv.first] += 0;
    for (const auto &kv : all) {
        const int d = to.main_of(kv.first) - from.main_of(kv.first);
        if (d > 0) plus += (plus.empty() ? "+" : ", +") + std::to_string(d) + " " + names(kv.first);
        if (d < 0)
            minus += (minus.empty() ? "\xE2\x88\x92" : ", \xE2\x88\x92") + std::to_string(-d) + " " + names(kv.first);
    }
    if (plus.empty() && minus.empty()) return "";
    return plus + (plus.empty() || minus.empty() ? "" : ", ") + minus;
}

static std::string row_label(const rmwork::SideboardRow &r, const std::function<std::string(int)> &names) {
    if (r.category == "SIDEBOARD_DONE") return "Done (keep this deck)";
    if (r.category == "SIDEBOARD_IN") return "in: " + names(r.card);
    if (r.category == "SIDEBOARD_OUT") return "cut: " + names(r.card);
    return r.category;
}

// "61 / 60 deck": the count large, good when it is the wanted one (or always neutral).
static void count_field(int have, int want, const char *what, bool exact) {
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.35f);
    const ImU32 c = !exact ? rmwork::tok().TEXT : have == want ? rmwork::tok().GOOD : rmwork::tok().BAD;
    ImGui::PushStyleColor(ImGuiCol_Text, c);
    ImGui::TextUnformatted(std::to_string(have).c_str());
    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::SameLine(0, 4);
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("/ %d %s", want, what);
}

static float count_field_width(int have, int want, const char *what) {
    const float big = ImGui::GetStyle().FontSizeBase * 1.35f;
    const float w1 = ImGui::GetFont()->CalcTextSizeA(big, 1e9f, 0, std::to_string(have).c_str()).x;
    char buf[64];
    std::snprintf(buf, sizeof buf, "/ %d %s", want, what);
    return w1 + 4 + ImGui::CalcTextSize(buf).x;
}

// A small label on a card: at `at`, its top-left corner (or top-right with `right`).
static void tag(ImDrawList *dl, ImVec2 at, const char *text, ImU32 bg, ImU32 fg, bool right) {
    const float size = std::max(9.0f, ImGui::GetFontSize() * 0.78f);
    const ImVec2 t = ImGui::GetFont()->CalcTextSizeA(size, 1e9f, 0, text);
    const ImVec2 a = right ? ImVec2(at.x - t.x - 6, at.y) : at;
    dl->AddRectFilled(a, ImVec2(a.x + t.x + 6, a.y + t.y + 2), bg, 2);
    dl->AddText(ImGui::GetFont(), size, ImVec2(a.x + 3, a.y + 1), fg, text);
}

namespace rmwork {

const SideboardGuides &SideboardView::guides() {
    if (!guides_read_) {
        guides_read_ = true;
        if (!guides_.load(repo_root() + "/python/sideboard_guides.json", &guides_error_))
            guides_error_ = "No league guide: " + guides_error_;
    }
    return guides_;
}

void SideboardView::update(ToolData &data, ReplaySession &s) {
    if (data.generation() == generation_ && s.match() == match_) return;
    generation_ = data.generation();
    if ((s.match() != match_ || s.cursor() != index_) && !send_.active) {
        plan_.clear();
        hint_.clear();
    }
    match_ = s.match();
    index_ = s.cursor();
    model_ = SideboardModel();
    if (!data.ok()) {
        model_.error = data.error();
        return;
    }
    const DecisionFrame &f = data.frame();
    const Match &m = *f.match;
    model_.ok = true;
    model_.view = f.view;
    model_.seat = f.view == VIEW_OMNI ? (f.info->decider >= 0 ? f.info->decider : 0) : f.view;
    const int seat = model_.seat;
    model_.phase = f.info->sideboarding;
    model_.seat_decides = model_.phase && f.info->decider == seat;
    model_.facts = read_sideboard_facts(*f.stream, *f.frame, seat);
    model_.lists = read_decklists(*f.stream);
    const DeckCounts &reg = model_.lists.lists[static_cast<size_t>(seat)];
    if (model_.lists.known[static_cast<size_t>(seat)]) {
        model_.deck_size = reg.main_total();
        model_.side_size = reg.side_total();
    }
    model_.on_play = model_.facts.plays_first == seat;
    SideboardState &st = model_.state;
    st.now = model_.facts.deck;
    st.start = st.now;
    st.swaps = model_.facts.swaps;
    st.drift = model_.facts.drift;
    st.pending_card = model_.facts.pending_card;
    if (!model_.seat_decides) {
        st.swaps = 0;
        st.drift = 0;
        st.pending_card = -1;
    }
    if (model_.phase && !model_.seat_decides) {
        const int other = f.info->decider;
        model_.note = std::string(seat_name(other)) + " is sideboarding (" + std::to_string(model_.facts.swaps) +
                      " swaps so far); " + view_name(f.view) + " does not see that deck. This is " + seat_name(seat) +
                      "'s own.";
    }
    // The phase's deck at its first decision, and the deck the frame after the phase shows.
    const int span = m.span_of(f.index);
    if (model_.phase && span >= 0) {
        const GameSpan &g = m.games[static_cast<size_t>(span)];
        std::string err;
        if (const std::shared_ptr<const rmstate::DecodedFrame> first = m.frame(g.first, f.view, &err)) {
            const SideboardFacts at = read_sideboard_facts(*f.stream, *first, seat);
            if (at.has_deck) st.start = at.deck;
        }
        const size_t end = g.first + g.count;
        const bool after =
            end < m.decisions.size() || (end == m.decisions.size() && m.terminal[static_cast<size_t>(f.view)].valid());
        if (after)
            if (const std::shared_ptr<const rmstate::DecodedFrame> next = m.frame(end, f.view, &err)) {
                const SideboardFacts at = read_sideboard_facts(*f.stream, *next, seat);
                if (at.has_deck) {
                    model_.have_end = true;
                    model_.end = at.deck;
                }
            }
    }
    if (model_.phase) model_.guide = guide_plan(guides(), model_.lists, seat, model_.on_play, f.stream->cards());
    if (model_.phase && !guides_.loaded()) model_.guide.why = guides_error_;
    if (model_.seat_decides) {
        model_.pi = ChoicesView::shown_pi(data);
        if (model_.pi.size() != model_.facts.menu.size()) model_.pi.clear();
        if (const featurize::PolicyEvalRecord *r = ChoicesView::shown_record(data))
            model_.policy = policy_label(r->id, r->version);
        if (model_.pi.empty()) {
            if (play_) {
                switch (play_->model_read().state()) {
                    case ModelReader::State::OFF:
                        model_.pi_note = "Model's Read is off for this game (New Game \xE2\x86\x92 Model's Read).";
                        break;
                    case ModelReader::State::NOT_CONNECTED:
                        model_.pi_note =
                            "Model's Read is not connected: no inference server at Settings \xE2\x86\x92 "
                            "Models' socket. The model's plan is unavailable.";
                        break;
                    case ModelReader::State::CONNECTING:
                        model_.pi_note = "Model's Read is connecting\xE2\x80\xA6";
                        break;
                    case ModelReader::State::READY:
                        model_.pi_note = "The model is reading this decision\xE2\x80\xA6";
                        break;
                    case ModelReader::State::FAILED:
                        model_.pi_note = "Model's Read stopped: " + play_->model_read().error();
                        break;
                }
            } else {
                model_.pi_note = "No evaluation of this decision was recorded.";
            }
        }
    } else if (model_.phase) {
        model_.pi_note = "Not the shown seat's decision.";
    }
    model_.results = results_text(m, seat, model_.facts.game);
    if (play_ && m.live) {
        model_.opponent =
            play_->config().opponent == Opponent::SCRIPTED ? "scripted" : short_agent(opponent_agent(play_->config()));
    } else {
        model_.opponent = agent_label(m, 1 - seat);
    }
}

std::string SideboardView::name_of(const DecisionFrame &f, int card) const {
    return f.stream ? f.stream->cards().name(card) : "#" + std::to_string(card);
}

CardLook SideboardView::look_of(const DecisionFrame &f, int card) const {
    CardLook look;
    look.face = f.book ? f.book->face(card, false) : nullptr;
    look.other = f.book ? f.book->face(card, true) : nullptr;
    if (look.face) {
        look.colors = look.face->colors;
        look.land = look.face->land;
        look.land_mana = look.face->land ? basic_land_mana(look.face->subtypes) : 0;
    }
    return look;
}

bool SideboardView::click_side(int card, std::string *why) {
    if (!interactive()) {
        *why = "not your sideboarding decision";
        return false;
    }
    return plan_.click_side(model_.state, card, why);
}

bool SideboardView::click_main(int card, std::string *why) {
    if (!interactive()) {
        *why = "not your sideboarding decision";
        return false;
    }
    return plan_.click_main(model_.state, card, why);
}

bool SideboardView::use_guide(std::string *why) {
    if (!interactive()) {
        *why = "not your sideboarding decision";
        return false;
    }
    const GuidePlan &g = model_.guide;
    if (!g.found) {
        *why = g.why;
        return false;
    }
    if (!g.unresolved.empty()) {
        *why = "the guide names " + g.unresolved + ", which is not in this 75";
        return false;
    }
    return plan_.set_target(model_.state, g.target, why);
}

bool SideboardView::apply_spec(const std::string &spec, std::string *why) {
    if (spec == "guide") return use_guide(why);
    if (!match_ || model_.view < 0) {
        *why = "no decision";
        return false;
    }
    const agents::CardNames &names = match_->stream(model_.view).cards();
    std::map<std::string, int> by_key;
    for (const std::map<int, int> *m : {&model_.state.now.main, &model_.state.now.side})
        for (const auto &kv : *m) by_key.emplace(card_key(names.name(kv.first)), kv.first);
    size_t at = 0;
    while (at <= spec.size()) {
        const size_t semi = std::min(spec.find(';', at), spec.size());
        const std::string item = spec.substr(at, semi - at);
        at = semi + 1;
        if (item.empty()) continue;
        const auto it = by_key.find(card_key(item.substr(1)));
        if ((item[0] != '+' && item[0] != '-') || it == by_key.end()) {
            *why = "'" + item + "' is not +Card or -Card of this 75";
            return false;
        }
        if (!(item[0] == '+' ? click_side(it->second, why) : click_main(it->second, why))) {
            *why = item + ": " + *why;
            return false;
        }
    }
    return true;
}

bool SideboardView::confirm(std::string *why) {
    if (!interactive()) {
        *why = send_.active ? "the plan is being sent" : "not your sideboarding decision";
        return false;
    }
    if (!plan_.balanced(model_.state)) {
        *why = "the plan is not balanced: every card brought in needs a cut";
        return false;
    }
    return start_sending(plan_.result(model_.state).main, why);
}

bool SideboardView::send_target(const std::map<int, int> &target, std::string *why) {
    if (!interactive()) {
        *why = "not your sideboarding decision";
        return false;
    }
    return start_sending(target, why);
}

bool SideboardView::start_sending(const std::map<int, int> &target, std::string *why) {
    if (model_.facts.game < 0 || model_.seat < 0) {
        *why = "no sideboarding phase";
        return false;
    }
    send_ = Sending();
    send_.active = true;
    send_.target = target;
    send_.seat = model_.seat;
    send_.game = model_.facts.game;
    error_.clear();
    result_.clear();
    hint_.clear();
    return true;
}

void SideboardView::dismiss() {
    error_.clear();
    result_.clear();
}

void SideboardView::finish(const std::string &error, const std::string &result) {
    send_.active = false;
    error_ = error;
    result_ = result;
    if (error.empty()) plan_.clear();
}

void SideboardView::step(PlaySession &p) {
    if (!send_.active) return;
    const Match *m = p.match();
    if (!m || !p.running()) {
        finish("The game ended before the plan was sent (" + std::to_string(send_.sent) + " replies sent).", "");
        return;
    }
    if (!p.deciding()) return;
    const long idx = p.pending_index();
    if (idx == send_.answered) return;
    std::string err;
    const std::shared_ptr<const rmstate::DecodedFrame> f = m->frame(static_cast<size_t>(idx), send_.seat, &err);
    if (!f) {
        finish("Decision " + std::to_string(idx + 1) + "'s frame could not be read: " + err, "");
        return;
    }
    const agents::SeatStream &stream = m->stream(send_.seat);
    const SideboardFacts facts = read_sideboard_facts(stream, *f, send_.seat);
    const auto names = [&stream](int c) { return stream.cards().name(c); };
    if (send_.answered >= 0) {
        if (!facts.sideboarding || facts.game != send_.game) {
            const std::string deck = std::to_string(facts.deck.main_total()) + " cards, " +
                                     std::to_string(facts.deck.side_total()) + " in the sideboard";
            if (main_matches(facts.deck, send_.target)) {
                finish("", "Sideboarding sent (" + std::to_string(send_.sent) + " replies): game " +
                               std::to_string(facts.game + 1) + "'s deck is as planned (" + deck + ").");
            } else {
                DeckCounts want;
                want.main = send_.target;
                finish("The phase ended before the plan was complete: game " + std::to_string(facts.game + 1) +
                           "'s deck differs from the plan by " + moves_text(facts.deck, want, names) + " (" + deck +
                           ").",
                    "");
            }
            return;
        }
        if (send_.last.done) {
            finish("Done was sent, but the engine asks again in the same phase.", "");
            return;
        }
        const int before = send_.before.main_of(send_.last_card), now = facts.deck.main_of(send_.last_card);
        const bool in = send_.last.what.compare(0, 3, "in:") == 0;
        if (in ? now <= before : now >= before) {
            finish("The engine did not apply \"" + send_.last.what + "\": the deck still has " + std::to_string(now) +
                       " copies of it. Nothing more was sent.",
                "");
            return;
        }
    }
    const SideboardStep st = next_sideboard_step(facts, send_.target, names);
    if (st.row < 0) {
        finish("The menu does not offer what the plan needs (" + st.error + "). Nothing more was sent; " +
                   std::to_string(send_.sent) + " replies had been.",
            "");
        return;
    }
    if (!p.reply(st.row, &err)) {
        finish("The engine refused \"" + st.what + "\": " + err, "");
        return;
    }
    send_.answered = idx;
    send_.last = st;
    send_.last_card = facts.menu[static_cast<size_t>(st.row)].card;
    send_.before = facts.deck;
    ++send_.sent;
}

// ── Drawing ──

std::vector<SideboardView::PileGroup> SideboardView::build_groups(const DecisionFrame &f) const {
    const SideboardState &s = model_.state;
    const DeckCounts res = plan_.result(s);
    const SideboardFacts &facts = model_.facts;
    std::map<int, double> pi_in, pi_out;
    int top = -1;
    for (size_t i = 0; i < model_.pi.size(); ++i) {
        const SideboardRow &r = facts.menu[i];
        if (r.category == "SIDEBOARD_IN") pi_in[r.card] = model_.pi[i];
        if (r.category == "SIDEBOARD_OUT") pi_out[r.card] = model_.pi[i];
        if (top < 0 || model_.pi[i] > model_.pi[static_cast<size_t>(top)]) top = static_cast<int>(i);
    }
    const SideboardRow *top_row = top >= 0 ? &facts.menu[static_cast<size_t>(top)] : nullptr;
    const std::map<int, int> &want = model_.guide.target;
    const bool guide = model_.guide.found && model_.guide.unresolved.empty() && model_.seat_decides;
    std::vector<PileGroup> groups(G_COUNT + 1);
    for (int g = 0; g < G_COUNT; ++g) groups[static_cast<size_t>(g)].label = GROUP_LABELS[g];
    std::map<int, int> main_cards = s.now.main, side_cards = s.now.side;
    for (const auto &kv : plan_.in()) main_cards[kv.first] += 0;
    for (const auto &kv : plan_.out()) side_cards[kv.first] += 0;
    for (const auto &kv : main_cards) {
        Pile p;
        p.card = kv.first;
        p.main = true;
        p.count = res.main_of(p.card);
        p.delta = plan_.planned_in(p.card) - plan_.planned_out(p.card);
        p.lock_in = s.in_locked(p.card);
        p.lock_out = s.out_locked(p.card);
        p.pending = s.drift == 1 && s.pending_card == p.card;
        p.guide = guide && p.count > (want.count(p.card) ? want.at(p.card) : 0);
        const auto pi = pi_out.find(p.card);
        if (pi != pi_out.end()) p.pi = pi->second;
        p.top = top_row && top_row->category == "SIDEBOARD_OUT" && top_row->card == p.card;
        groups[static_cast<size_t>(group_of(f.book ? f.book->face(p.card, false) : nullptr))].piles.push_back(p);
    }
    PileGroup &side = groups.back();
    for (const auto &kv : side_cards) {
        Pile p;
        p.card = kv.first;
        p.main = false;
        p.count = res.side_of(p.card);
        p.delta = plan_.planned_out(p.card) - plan_.planned_in(p.card);
        p.lock_out = s.out_locked(p.card);
        p.guide = guide && res.main_of(p.card) < (want.count(p.card) ? want.at(p.card) : 0);
        const auto pi = pi_in.find(p.card);
        if (pi != pi_in.end()) p.pi = pi->second;
        p.top = top_row && top_row->category == "SIDEBOARD_IN" && top_row->card == p.card;
        side.piles.push_back(p);
    }
    std::vector<PileGroup> out;
    for (PileGroup &g : groups) {
        std::sort(g.piles.begin(), g.piles.end(),
            [&](const Pile &a, const Pile &b) { return name_of(f, a.card) < name_of(f, b.card); });
        if (g.piles.empty() && &g != &side) continue;
        int n = 0;
        for (const Pile &p : g.piles) n += p.count;
        if (&g != &side) g.label += " (" + std::to_string(n) + ")";
        out.push_back(g);
    }
    if (!out.empty()) out.back().label.clear();  // the sideboard: its section heading names it
    return out;
}

// Places the groups' piles (cards `cw` wide) in rows `width` wide: a group starts a new row when it
// does not fit beside the one before, and a group wider than a row wraps under its own start. `at`
// gets each pile's top-left and `labels` each group's label position, relative to the origin;
// returns the height.
float SideboardView::flow(const std::vector<PileGroup> &groups, float cw, float width, std::vector<ImVec2> *at,
    std::vector<ImVec2> *labels) const {
    const float ch = std::round(cw * CARD_ASPECT), lh = ImGui::GetTextLineHeight();
    const float gap = std::round(cw * 0.1f), group_gap = gap * 3, row_gap = std::round(lh * 0.5f);
    const float cell_h = ch + lh + 4;  // the card and its π line
    float x = 0, y = 0, line_h = 0;
    for (const PileGroup &g : groups) {
        const float label_h = g.label.empty() ? 0 : lh + 2;
        const float n = static_cast<float>(g.piles.size());
        const float gw = std::max(n * cw + std::max(0.0f, n - 1) * gap, ImGui::CalcTextSize(g.label.c_str()).x);
        if (x > 0 && x + gw > width) {
            x = 0;
            y += line_h + row_gap;
            line_h = 0;
        }
        labels->push_back(ImVec2(x, y));
        float cx = x, cy = y + label_h;
        bool wrapped = false;
        for (size_t i = 0; i < g.piles.size(); ++i) {
            if (cx > x && cx + cw > width) {
                cx = x;
                cy += cell_h + row_gap;
                wrapped = true;
            }
            at->push_back(ImVec2(cx, cy));
            cx += cw + gap;
        }
        line_h = std::max(line_h, cy + cell_h - y);
        if (wrapped) {
            x = 0;
            y += line_h + row_gap;
            line_h = 0;
        } else {
            x = std::max(cx - gap, x + gw) + group_gap;
        }
    }
    return y + line_h;
}

void SideboardView::draw_pile(const DecisionFrame &f, const Pile &p, ImVec2 a, ImVec2 b, bool hovered) {
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float cw = b.x - a.x, r = cw * 0.05f;
    const bool empty = p.count == 0;
    dl->AddRectFilled(ImVec2(a.x + 1, a.y + 2), ImVec2(b.x + 1, b.y + 3), hex_color(0x000000, 120), r);
    renderer_->draw(dl, look_of(f, p.card), a, b, false, empty ? 0.3f : 1.0f);
    const char *times = "\xC3\x97";
    tag(dl, ImVec2(b.x - 3, a.y + 3), (times + std::to_string(p.count)).c_str(), hex_color(0x000000, 200),
        hex_color(0xFFFFFF), true);
    if (p.lock_in) tag(dl, ImVec2(a.x + 3, a.y + 3), "IN", tok().GOOD, hex_color(0x000000), false);
    if (p.lock_out) tag(dl, ImVec2(a.x + 3, a.y + 3), "OUT", tok().BAD, hex_color(0xFFFFFF), false);
    const float th = std::max(9.0f, ImGui::GetFontSize() * 0.78f) + 2;
    if (p.delta != 0) {
        const std::string d = (p.delta > 0 ? "+" : "\xE2\x88\x92") + std::to_string(std::abs(p.delta));
        tag(dl, ImVec2(a.x + 3, b.y - th - 3), d.c_str(), p.delta > 0 ? tok().GOOD : tok().BAD,
            p.delta > 0 ? hex_color(0x000000) : hex_color(0xFFFFFF), false);
    } else if (p.guide) {
        tag(dl, ImVec2(a.x + 3, b.y - th - 3), "guide", tok().ACCENT, hex_color(0x000000), false);
    }
    if (p.pending)
        dl->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), tok().GLOW, r + 2, ImDrawFlags_None, 3.0f);
    if (p.top)
        dl->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), tok().ACCENT, r + 2, ImDrawFlags_None, 2.5f);
    if (hovered)
        dl->AddRect(ImVec2(a.x - 4, a.y - 4), ImVec2(b.x + 4, b.y + 4), tok().LINK, r + 4, ImDrawFlags_None, 2.0f);
    // Under the card: the model's π of its row.
    const float lh = ImGui::GetTextLineHeight();
    if (p.pi >= 0) {
        const std::string v = number(p.pi, 3);
        const float tw = ImGui::CalcTextSize(v.c_str()).x;
        draw_bar(dl, ImVec2(a.x, b.y + 4 + lh * 0.3f), std::max(4.0f, cw - tw - 4), lh * 0.4f,
            static_cast<float>(std::min(1.0, p.pi * 4)), tok().ACCENT);
        dl->AddText(ImVec2(b.x - tw, b.y + 3), p.top ? tok().TEXT : tok().DIM, v.c_str());
    }
}

void SideboardView::draw_piles(
    const DecisionFrame &f, const std::vector<PileGroup> &groups, ImVec2 size, ImGuiKeyChord magnify) {
    if (groups.empty()) return;
    const std::vector<PileGroup> deck(groups.begin(), groups.end() - 1), side(groups.end() - 1, groups.end());
    const float lh = ImGui::GetTextLineHeightWithSpacing(), fs = ImGui::GetFontSize();
    const float heads = 2 * lh + fs;  // the two section headings and the gap between the sections
    const float width = size.x - ImGui::GetStyle().ScrollbarSize - 4;
    float cw = std::round(fs * 3.4f);
    std::vector<ImVec2> at_deck, at_side, lab_deck, lab_side;
    for (float w = std::round(fs * 8.5f); w >= fs * 3.4f; w -= 2) {
        at_deck.clear();
        at_side.clear();
        lab_deck.clear();
        lab_side.clear();
        const float h = flow(deck, w, width, &at_deck, &lab_deck) + flow(side, w, width, &at_side, &lab_side) + heads;
        cw = w;
        if (h <= size.y - 4) break;
    }
    at_deck.clear();
    at_side.clear();
    lab_deck.clear();
    lab_side.clear();
    const float deck_h = flow(deck, cw, width, &at_deck, &lab_deck);
    flow(side, cw, width, &at_side, &lab_side);
    const float ch = std::round(cw * CARD_ASPECT);
    const SideboardState &s = model_.state;
    const DeckCounts res = plan_.result(s);
    const bool live = interactive();
    ImGui::BeginChild("##piles", size, ImGuiChildFlags_None);
    const ImVec2 origin = ImGui::GetCursorScreenPos();
    // The section headings.
    std::string in_phase, cut;
    for (const auto &kv : s.now.main)
        if (s.in_locked(kv.first)) in_phase += (in_phase.empty() ? "" : ", ") + name_of(f, kv.first);
    std::map<int, int> cut_cards = s.start.main;
    for (const auto &kv : cut_cards)
        if (s.out_locked(kv.first)) cut += (cut.empty() ? "" : ", ") + name_of(f, kv.first);
    ImGui::TextUnformatted("Deck");
    ImGui::SameLine();
    std::string sub = std::to_string(res.main_total()) + " cards";
    if (model_.seat_decides) {
        if (!in_phase.empty()) sub += " \xC2\xB7 in this phase: " + in_phase + " (locked)";
        if (!cut.empty()) sub += " \xC2\xB7 cut: " + cut;
        if (live) sub += " \xC2\xB7 click a card to cut it";
    }
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", sub.c_str());
    ImGui::PopTextWrapPos();
    const ImVec2 deck_origin(origin.x, std::max(origin.y + lh, ImGui::GetCursorScreenPos().y));
    const ImVec2 side_head(origin.x, deck_origin.y + deck_h + fs);
    const ImVec2 side_origin(origin.x, side_head.y + lh);
    ImGui::SetCursorScreenPos(side_head);
    ImGui::TextUnformatted("Sideboard");
    ImGui::SameLine();
    ImGui::TextDisabled("%s",
        (std::to_string(res.side_total()) + " cards" + (live ? " \xC2\xB7 click one to bring it in" : "")).c_str());
    ImDrawList *dl = ImGui::GetWindowDrawList();
    uint64_t under = 0;
    const Pile *under_pile = nullptr;
    ImVec2 under_a, under_b;
    for (int section = 0; section < 2; ++section) {
        const std::vector<PileGroup> &gs = section == 0 ? deck : side;
        const std::vector<ImVec2> &at = section == 0 ? at_deck : at_side;
        const std::vector<ImVec2> &labels = section == 0 ? lab_deck : lab_side;
        const ImVec2 o = section == 0 ? deck_origin : side_origin;
        size_t k = 0;
        for (size_t gi = 0; gi < gs.size(); ++gi) {
            if (!gs[gi].label.empty())
                dl->AddText(ImVec2(o.x + labels[gi].x, o.y + labels[gi].y), tok().DIM, gs[gi].label.c_str());
            for (const Pile &p : gs[gi].piles) {
                const ImVec2 a(o.x + at[k].x, o.y + at[k].y), b(a.x + cw, a.y + ch);
                ++k;
                ImGui::SetCursorScreenPos(a);
                ImGui::PushID(static_cast<int>(pile_id(p.card, p.main)));
                ImGui::InvisibleButton("##pile", ImVec2(cw, ch), ImGuiButtonFlags_MouseButtonLeft);
                const bool hovered = ImGui::IsItemHovered();
                if (ImGui::IsItemClicked(ImGuiMouseButton_Left) && live) {
                    std::string why;
                    const bool ok = p.main ? plan_.click_main(s, p.card, &why) : plan_.click_side(s, p.card, &why);
                    hint_ = ok ? "" : name_of(f, p.card) + ": " + why;
                }
                ImGui::PopID();
                draw_pile(f, p, a, b, hovered);
                if (hovered) {
                    under = pile_id(p.card, p.main);
                    under_pile = &p;
                    under_a = a;
                    under_b = b;
                }
            }
        }
    }
    const float total = side_origin.y - origin.y + flow(side, cw, width, &at_side, &lab_side);
    ImGui::SetCursorScreenPos(ImVec2(origin.x, origin.y + total));
    ImGui::Dummy(ImVec2(1, 1));
    const bool over = ImGui::IsWindowHovered();
    ImGui::EndChild();
    mag_.update(under, over, magnify);
    if (under_pile && mag_.handle() == under) {
        const Pile &p = *under_pile;
        std::vector<std::string> lines;
        const CardFace *back = f.book ? f.book->face(p.card, true) : nullptr;
        lines.push_back(name_of(f, p.card) + (back ? " // " + back->name : ""));
        lines.push_back(std::to_string(res.main_of(p.card)) + " in the deck \xC2\xB7 " +
                        std::to_string(res.side_of(p.card)) + " in the sideboard");
        if (p.lock_in) lines.push_back("Brought in this phase: not cut again");
        if (p.lock_out) lines.push_back("Cut this phase: not brought back");
        if (p.pi >= 0) lines.push_back("\xCF\x80 " + number(p.pi, 3) + (p.main ? " (cut it)" : " (bring it in)"));
        draw_magnified_card(*renderer_, look_of(f, p.card), lines, under_a, under_b);
    } else if (under_pile && mag_.handle() == 0) {
        const Pile &p = *under_pile;
        std::string tip = name_of(f, p.card) + "\n" + std::to_string(res.main_of(p.card)) + " in the deck, " +
                          std::to_string(res.side_of(p.card)) + " in the sideboard";
        if (live) {
            std::string why;
            SideboardPlan probe = plan_;
            const bool ok = p.main ? probe.click_main(s, p.card, &why) : probe.click_side(s, p.card, &why);
            if (ok) {
                const bool cancel = p.main ? plan_.planned_in(p.card) > 0 : plan_.planned_out(p.card) > 0;
                tip += cancel   ? "\nClick: take back one planned copy"
                       : p.main ? "\nClick: cut one"
                                : "\nClick: bring one in";
            } else {
                tip += "\n" + why;
            }
        }
        if (magnify) tip += std::string("\nHold ") + ImGui::GetKeyChordName(magnify) + " or right-click to magnify";
        ImGui::SetTooltip("%s", tip.c_str());
    }
}

std::string SideboardView::status_line(const DecisionFrame &f) const {
    const SideboardState &s = model_.state;
    if (!model_.seat_decides) return "";
    if (send_.active) return "Sending the plan: " + std::to_string(send_.sent) + " replies sent\xE2\x80\xA6";
    if (!choosing_) return "";
    const std::function<std::string(int)> names = [this, &f](int c) { return name_of(f, c); };
    const std::string moves = moves_text(s.now, plan_.result(s), names);
    const int owed = plan_.ins() + s.drift - plan_.outs();
    if (plan_.empty() && s.drift == 1) return name_of(f, s.pending_card) + " came in: click a deck card to cut it";
    if (plan_.empty())
        return "Click a sideboard card to bring it in, then a deck card to cut; Confirm sends the swaps, then Done";
    if (owed > 0) return "Plan: " + moves + " \xC2\xB7 cut " + std::to_string(owed) + " more to balance";
    if (owed < 0) return "Plan: " + moves + " \xC2\xB7 bring in " + std::to_string(-owed) + " more to balance";
    return "Plan: " + (moves.empty() ? std::string("no change") : moves) + " \xC2\xB7 balanced";
}

void SideboardView::draw_header(const DecisionFrame &f) {
    const SideboardModel &md = model_;
    const int seat = md.seat;
    const bool play = play_ && f.match->live;
    std::string title;
    if (md.phase) {
        title = "Before game " + std::to_string(md.facts.game + 1);
    } else {
        title = "Game " + std::to_string(md.facts.game + 1);
    }
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(title.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("\xC2\xB7");
    ImGui::SameLine();
    seat_chip(seat);
    ImGui::SameLine();
    const std::string own_deck =
        md.lists.known[static_cast<size_t>(seat)] ? md.lists.name[static_cast<size_t>(seat)] : "?";
    ImGui::TextUnformatted(((play ? "you, " : std::string(seat_name(seat)) + ", ") + own_deck + " vs").c_str());
    ImGui::SameLine();
    seat_chip(1 - seat);
    ImGui::SameLine();
    const std::string opp_deck =
        md.lists.known[static_cast<size_t>(1 - seat)] ? md.lists.name[static_cast<size_t>(1 - seat)] : "?";
    ImGui::TextUnformatted(
        ((md.opponent.empty() ? std::string(seat_name(1 - seat)) : md.opponent) + ", " + opp_deck).c_str());
    std::string tail;
    if (!md.results.empty()) tail = md.results;
    if (md.phase && md.facts.plays_first >= 0)
        tail += (tail.empty() ? "" : ": ") + std::string(md.on_play ? "on the play" : "on the draw");
    if (!tail.empty()) {
        ImGui::SameLine();
        ImGui::TextDisabled("\xC2\xB7 %s", tail.c_str());
    }
    const DeckCounts res = plan_.result(md.state);
    // The counts, at the right of the line when they fit there, else on a line of their own.
    const int swaps = plan_.swaps_after(md.state);
    const float w = count_field_width(res.main_total(), md.deck_size, "deck") +
                    count_field_width(res.side_total(), md.side_size, "sideboard") +
                    (md.seat_decides ? count_field_width(swaps, SIDEBOARD_SWAP_CAP, "swaps") + 20 : 0) + 20;
    const float right = ImGui::GetWindowContentRegionMax().x;
    const float used = ImGui::GetItemRectMax().x - ImGui::GetWindowPos().x;
    if (used + 24 < right - w) ImGui::SameLine(right - w);
    count_field(res.main_total(), md.deck_size, "deck", true);
    ImGui::SameLine(0, 20);
    count_field(res.side_total(), md.side_size, "sideboard", true);
    if (md.seat_decides) {
        ImGui::SameLine(0, 20);
        count_field(swaps, SIDEBOARD_SWAP_CAP, "swaps", false);
    }
    // The second line: the plan, the note or what was sent.
    const std::string line = status_line(f);
    if (!md.note.empty()) {
        ImGui::TextDisabled("%s", md.note.c_str());
    } else if (!line.empty()) {
        const bool balanced = plan_.balanced(md.state) && !plan_.empty();
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const ImVec2 t = ImGui::CalcTextSize(line.c_str());
        ImGui::GetWindowDrawList()->AddRectFilled(
            ImVec2(p.x - 1, p.y - 1), ImVec2(p.x + t.x + 12, p.y + t.y + 3), tok().FIELD, 3);
        ImGui::GetWindowDrawList()->AddRect(
            ImVec2(p.x - 1, p.y - 1), ImVec2(p.x + t.x + 12, p.y + t.y + 3), balanced ? tok().GOOD : tok().GLOW, 3);
        ImGui::SetCursorScreenPos(ImVec2(p.x + 6, p.y + 1));
        ImGui::TextUnformatted(line.c_str());
    } else if (md.phase) {
        ImGui::TextDisabled("%s", play ? "An earlier decision: the toolbar's Latest goes to your decision"
                                       : "Replay: the deck as this decision saw it; the phase's swaps are under Plans");
    } else {
        ImGui::TextDisabled("%s", (std::string(seat_name(seat)) +
                                      "'s 75 in this game; a best of three's sideboarding opens here between games")
                                      .c_str());
    }
}

void SideboardView::draw_plans(const DecisionFrame &f) {
    const SideboardModel &md = model_;
    const std::function<std::string(int)> names = [this, &f](int c) { return name_of(f, c); };
    const DeckCounts res = plan_.result(md.state);
    if (md.phase) {
        const GuidePlan &g = md.guide;
        ImGui::TextUnformatted("League guide");
        if (g.found) {
            ImGui::SameLine();
            ImGui::TextDisabled("\xC2\xB7 %s vs %s \xC2\xB7 %s", g.own.c_str(), g.opp.c_str(),
                g.on_play ? "on the play" : "on the draw");
            const auto line = [&](const std::pair<std::string, int> &c, bool in) {
                const auto id = g.ids.find(c.first);
                bool met = false;
                if (id != g.ids.end()) {
                    const int want = g.target.count(id->second) ? g.target.at(id->second) : 0;
                    met = res.main_of(id->second) == want;
                }
                ImGui::PushStyleColor(ImGuiCol_Text, met ? tok().GOOD : tok().DIM);
                ImGui::TextUnformatted(met ? "\xE2\x9C\x93" : "\xE2\x97\x8B");
                ImGui::PopStyleColor();
                ImGui::SameLine();
                ImGui::Text("%s %d %s", in ? "in " : "out", c.second, c.first.c_str());
            };
            for (const auto &c : g.in) line(c, true);
            for (const auto &c : g.out) line(c, false);
            if (!g.other.empty()) note(g.other.c_str());
            if (g.holdout)
                note(
                    "A held-out matchup: the scripted tier keeps its deck here (the generalisation test); the "
                    "guide's row is shown all the same.");
            if (!g.unresolved.empty())
                colored_note(("The guide names " + g.unresolved + ", which this 75 does not have.").c_str(), tok().BAD);
            ImGui::BeginDisabled(!interactive() || !g.unresolved.empty());
            if (ImGui::Button("Use this plan")) {
                std::string why;
                hint_ = use_guide(&why) ? "" : "The guide's plan: " + why;
            }
            ImGui::EndDisabled();
        } else {
            note(g.why.c_str());
        }
        ImGui::Dummy(ImVec2(1, 4));
        ImGui::Separator();
        // The model.
        ImGui::TextUnformatted("Model");
        if (!md.policy.empty()) {
            ImGui::SameLine();
            ImGui::TextDisabled("\xC2\xB7 %s", md.policy.c_str());
        }
        if (md.pi.empty()) {
            note(md.pi_note.empty() ? "No evaluation." : md.pi_note.c_str());
        } else {
            std::vector<size_t> order(md.pi.size());
            for (size_t i = 0; i < order.size(); ++i) order[i] = i;
            std::stable_sort(order.begin(), order.end(), [&md](size_t a, size_t b) { return md.pi[a] > md.pi[b]; });
            const float bar_w = ImGui::CalcTextSize("0000000").x, num_w = ImGui::CalcTextSize("0.000").x;
            for (size_t k = 0; k < order.size() && k < 8; ++k) {
                const size_t i = order[k];
                ImGui::PushStyleColor(ImGuiCol_Text, k == 0 ? tok().ACCENT : tok().DIM);
                ImGui::TextUnformatted(k == 0 ? "\xE2\x96\xB8" : " ");
                ImGui::PopStyleColor();
                ImGui::SameLine();
                const float avail = ImGui::GetContentRegionAvail().x - bar_w - num_w - 16;
                const std::string label = row_label(md.facts.menu[i], names);
                const ImVec2 p = ImGui::GetCursorScreenPos();
                draw_fitted_text(ImGui::GetWindowDrawList(), ImGui::GetFont(), ImGui::GetFontSize(), p,
                    k == 0 ? tok().TEXT : tok().DIM, label, std::max(20.0f, avail));
                ImGui::Dummy(ImVec2(std::max(20.0f, avail), ImGui::GetTextLineHeight()));
                ImGui::SameLine();
                value_bar(static_cast<float>(md.pi[i]), bar_w, tok().ACCENT);
                ImGui::SameLine();
                ImGui::TextUnformatted(number(md.pi[i], 3).c_str());
            }
            note(
                "\xCF\x80 is the model's for the decision in front of you: it decides one swap at a time, so its plan "
                "past this decision is not known.");
        }
        ImGui::Dummy(ImVec2(1, 4));
        ImGui::Separator();
    }
    // The phase so far, the phase in all, the deck against its registered list.
    ImGui::TextUnformatted(md.phase ? "This phase" : "This deck");
    if (md.phase) {
        const std::string so_far = moves_text(md.state.start, md.state.now, names);
        ImGui::TextDisabled("Swaps so far");
        ImGui::TextWrapped("%s", so_far.empty() ? "none" : so_far.c_str());
        if (md.have_end) {
            const std::string all = moves_text(md.state.start, md.end, names);
            ImGui::TextDisabled("The phase in all");
            ImGui::TextWrapped("%s", all.empty() ? "no swaps (Done at once)" : all.c_str());
        }
    }
    if (md.lists.known[static_cast<size_t>(md.seat)]) {
        DeckCounts reg = md.lists.lists[static_cast<size_t>(md.seat)];
        const std::string vs = moves_text(reg, res, names);
        ImGui::TextDisabled("%s",
            md.phase && md.seat_decides ? "Against the registered list, once sent" : "Against the registered list");
        ImGui::TextWrapped("%s", vs.empty() ? "as registered" : vs.c_str());
    }
}

void SideboardView::draw_footer() {
    const SideboardState &s = model_.state;
    if (!error_.empty()) {
        colored_note(("Stopped: " + error_).c_str(), tok().BAD);
    } else if (!result_.empty()) {
        colored_note(result_.c_str(), tok().GOOD);
    } else if (!hint_.empty()) {
        colored_note(hint_.c_str(), tok().GLOW);
    }
    if (!model_.seat_decides || !choosing_) {
        if (!error_.empty() || !result_.empty()) {
            if (ImGui::Button("Dismiss")) dismiss();
        }
        return;
    }
    const bool balanced = plan_.balanced(s);
    std::string label;
    if (plan_.empty() && s.drift == 0) {
        label = "Done: keep this deck";
    } else {
        const int replies = plan_.ins() + plan_.outs() + (plan_.swaps_after(s) >= SIDEBOARD_SWAP_CAP ? 0 : 1);
        label = "Confirm: send " + std::to_string(replies) + (replies == 1 ? " reply" : " replies");
    }
    ImGui::BeginDisabled(send_.active || !balanced);
    ImGui::PushStyleColor(ImGuiCol_Button, tok().SEL);
    if (ImGui::Button(label.c_str())) {
        std::string why;
        if (!confirm(&why)) hint_ = why;
    }
    ImGui::PopStyleColor();
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(send_.active || plan_.empty());
    if (ImGui::Button("Undo")) undo();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) reset();
    ImGui::EndDisabled();
    if (!error_.empty() || !result_.empty()) {
        ImGui::SameLine();
        if (ImGui::Button("Dismiss")) dismiss();
    }
    ImGui::SameLine();
    ImGui::TextDisabled(
        "A card brought in this phase is not cut again, nor a cut card brought back; at most %d "
        "swaps.",
        SIDEBOARD_SWAP_CAP);
}

void SideboardView::draw(ToolData &data, ReplaySession &s, bool choosing, ImGuiKeyChord magnify) {
    choosing_ = choosing;
    update(data, s);
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (!model_.ok) {
        centered_text_block("Frame not readable", model_.error.c_str(), width);
        return;
    }
    const DecisionFrame &f = data.frame();
    if (!model_.facts.has_deck) {
        centered_text_block("No deck in this frame", "The frame carries no own_deck rows for this seat.", width);
        return;
    }
    draw_header(f);
    ImGui::Separator();
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float foot = ImGui::GetFrameHeightWithSpacing() * 2 + 6;
    const float body_h = std::max(80.0f, avail.y - foot);
    const float plans_w = std::max(ImGui::GetFontSize() * 17, std::min(ImGui::GetFontSize() * 26, avail.x * 0.28f));
    const float piles_w = std::max(120.0f, avail.x - plans_w - 8);
    const std::vector<PileGroup> groups = build_groups(f);
    draw_piles(f, groups, ImVec2(piles_w, body_h), magnify);
    ImGui::SameLine(0, 8);
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PANEL_2);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 6));
    ImGui::BeginChild("##plans", ImVec2(0, body_h), ImGuiChildFlags_Borders | ImGuiChildFlags_AlwaysUseWindowPadding);
    ImGui::PushTextWrapPos(0.0f);
    draw_plans(f);
    ImGui::PopTextWrapPos();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::Separator();
    draw_footer();
}

}  // namespace rmwork
