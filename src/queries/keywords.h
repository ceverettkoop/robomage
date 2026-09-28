#ifndef QUERIES_KEYWORDS_H
#define QUERIES_KEYWORDS_H

#include <string>
#include <vector>
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

// ── Keyword abilities (CR 702) ────────────────────────────────────────────────

// Does each instance of keyword `kw` function separately, so a second grant gives a second
// instance? True for the triggered keywords — Ward (CR 702.21a, 113.2c), Prowess (702.108b),
// Exalted (702.83a), Mobilize (702.181a): each instance triggers on its own. Other keywords are
// redundant in multiples, so a repeated grant of one adds nothing.
inline bool keyword_instances_stack(const std::string &kw) {
    return kw.rfind("Ward", 0) == 0 || kw == "Prowess" || kw == "Exalted" || kw.rfind("Mobilize", 0) == 0;
}

// Add one granted instance of keyword `kw` to a keyword list: always for a keyword whose
// instances stack (keyword_instances_stack), otherwise only if the list lacks it.
void add_keyword_instance(std::vector<std::string> &keywords, const std::string &kw);

// True if the creature carries the given keyword string (exact match).
inline bool creature_has_keyword(const Creature &cr, const char *kw) {
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

// True if keyword `kw` is currently SUPPRESSED on permanent `e` by an until-end-of-turn
// "loses <keyword>" effect (AB$ AnimateAll | RemoveKeywords$, Shadowspear — CR 613 layer 6).
// While suppressed the keyword is treated as absent regardless of how it was granted (printed,
// counter, continuous). Single gate consulted by every effective-keyword accessor below so the
// removal applies uniformly to creatures and noncreature permanents. Cleared at cleanup (514.2).
inline bool keyword_removed_eot(Entity e, const char *kw) {
    if (!global_coordinator.entity_has_component<Permanent>(e)) return false;
    const auto &removed = global_coordinator.GetComponent<Permanent>(e).removed_keywords_eot;
    return removed.find(kw) != removed.end();
}

// True if the permanent `e` is indestructible (CR 702.12b: it can't be destroyed —
// "destroy" effects don't destroy it, and it ignores the lethal-damage / deathtouch
// state-based actions, CR 704.5g/h). Reads the keyword from the object's effective
// keyword list: for a creature that is `Creature::keywords` (rebuilt each static pass
// from the printed list plus any granted keywords), otherwise the printed keywords on
// the face that's up (CR 712.8e) or the Token — so a non-creature permanent like an
// artifact land with `K:Indestructible` is covered too. Indestructible does NOT prevent sacrifice, exile,
// "put into graveyard", or the 0-toughness SBA (CR 704.5f); those callers do not consult
// this. Single source shared by the Destroy effects and the lethal-damage SBA.
bool is_indestructible(Entity e);

// True if the permanent `e` currently has the keyword `kw`, reading its EFFECTIVE
// keyword list the same way is_indestructible does: a creature's `Creature::keywords`
// (rebuilt each static pass from the printed list plus any granted keywords — Pump
// grants, continuous effects, keyword counters), otherwise the printed keywords of the
// face that's up (CR 712.8e) or the Token's. Single source for "does this permanent
// currently have keyword K"; targeting (Shroud/Hexproof, CR 702.18/702.11) and any future keyword query share it
// so they cannot drift on how a granted keyword is stored.
bool permanent_has_keyword(Entity e, const char *kw);

// The permanent `e`'s effective keyword list, read the same way permanent_has_keyword reads one
// keyword (a creature's Creature::keywords, else the face that's up or the Token's printed list),
// minus any keyword suppressed until end of turn. Used to snapshot a departing permanent's
// keywords into its last-known information (CR 608.2h).
std::vector<std::string> permanent_keywords(Entity e);

#endif /* QUERIES_KEYWORDS_H */
