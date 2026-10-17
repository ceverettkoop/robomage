"""The scripted agent's decisions, over an abstract view of one decision.

This module is the rule-based "scripted" agent itself: its skill tiers, presets and every
decision rule. It reads the game only through a **view** of the decision it is asked to make
and knows nothing about how the view was obtained, so one implementation serves both protocols:

- ``scripted_agent.VectorView`` reads the engine's BQUERY float vector (the RL stack's
  observation: ``scripted_agent.ScriptedAgent.act(obs, num_choices)``);
- ``rm_agent.FrameView`` reads one seat's decoded RMSTATE frame and the knowledge sidecar's
  ``k_*`` tables (``rm_agent.decide``).

Cards are named, never numbered: every card the rules key on is a name constant below, and every
view answers with card names (None where the view shows no card). Categories, steps and zone
references are the engine's enum names (the RMSTATE schema's ``action_category`` / ``step``
string tables; the vector view maps the BQUERY ints onto them). Card characteristics come from
the view's card knowledge (``view.cards``). Nothing here imports numpy or the BQUERY layout.

IMPORTANT — no multi-ply lookahead: Python cannot fork/snapshot/restore the C++
game state, so the agent reasons strictly single-ply.  Combat decisions are the
one place exact local lookahead is cheap (attacker/blocker P/T arithmetic) and
that is the only "simulation" performed.  Do not add rollout/minimax search that
assumes hypothetical engine queries — the engine does not expose them.

The decision functions are STATELESS pure functions of the (perspective-relative)
view, because they run inside parallel vec-env subprocesses and
diag/watch determinism depends on it. The only exceptions are small per-game,
per-agent sets that are themselves deterministic functions of the trajectory
(EXPLORE's novelty set and the GREEDY/HEURISTIC fruitless-activation stall
guard — see ``ScriptedCore``).

**The view** (both implementations; ``view.exact_refs`` says which joins it can make):

- menu: ``n``; ``cats`` (category names, ``n`` long); ``card(i)`` (the action's card name or
  None); ``nonnull_card(i)`` (the null-card test of the menu-shape rules); ``ctrl(i)`` (1 self,
  0 opponent, -1 none); ``zone(i)`` (the ActionRefZone name: ``NONE``, ``SELF_BATTLEFIELD``,
  ``PLAYER_OPP``, ...); ``ordinal(i)``; ``ref(i)`` (an identity key of the action's object);
  ``ref_perm(i)`` (that object as a battlefield permanent, or None);
  ``is_keyword_ability(i, keyword)`` (the action activates the ability a keyword built:
  ``Ninjutsu``, ``Equip``, ``Reconfigure``'s attach);
- players and turn: ``seat_is_a``, ``life``, ``opp_life``, ``library``, ``turn``, ``is_active``,
  ``in_main``, ``step``, ``pool(color)`` / ``pool_total`` (W, U, B, R, G, C), ``mulligans``;
- the pending decision's source: ``pending`` (its card name) and ``pending_perm`` (it as a
  battlefield permanent, when the view can say);
- objects: ``hand`` (own hand card names); ``my_perms`` / ``opp_perms`` (battlefield
  permanents, phased-out ones left out, CR 702.26b) and ``my_perms_all`` (phased-out included),
  each a ``Perm``-like object (``card``, ``token``, ``tapped``, ``sick``, ``creature``, ``land``,
  ``power``, ``toughness``, ``attacking``, ``blocking``, ``blocked``, ``loyalty``,
  ``other_counters``, ``attached``, ``has_attachment``, ``keywords``, ``mana_value``,
  ``art_ench`` (an artifact or an enchantment), ``legendary``); ``stack`` (stack objects
  with a card, top first: ``card``, ``mine``, ``spell``), ``stack_top`` (the top object, or
  None) and ``stack_empty``; ``graveyard_cards`` (both graveyards' card names), ``opp_graveyard``,
  ``opp_exile``, ``opp_exile_counters`` (``(name, counters)`` per opposing exiled card);
- memory: ``known_top(n)`` (the own library's top ``n`` cards, None where unknown),
  ``top_candidates(n)`` (per position the set of cards it can be, None where anything),
  ``opp_known_hand`` (card names, a name per copy), ``opp_hand_possible(name)`` (False when the
  card cannot be in the opponent's hand, True when it can, None when the view cannot tell),
  ``opp_deck`` (the opponent's registered list, name -> copies), ``own_library_has(name)``
  (whether the own library can hold the card; None when the view cannot tell);
- ``opp_runs_card(card, deck_name)``: whether the opponent's list holds the card (True / False,
  None when unknown);
- ``cards``: card knowledge by name (``mana_value``, ``cost_colors``, ``is_land``, ``power``,
  ``is_delve``, ``is_equipment``, ``has_miracle``, ``is_legendary``, ``land_colors``,
  ``deals_damage``, ``has_nonmana_activation``, ``is_artifact_or_enchantment``).

``exact_refs`` is true when the view joins an action to its own object (RMSTATE handles); the
vector view joins by card id where the rules always did (the first permanent with that card).
"""

import random
from dataclasses import dataclass
from enum import Enum


# ── Engine enum names (the RMSTATE schema's string tables) ───────────────────
_CAT_PASS = "PASS_PRIORITY"
_CAT_SEL_ATK = "SELECT_ATTACKER"
_CAT_CONF_ATK = "CONFIRM_ATTACKERS"
_CAT_SEL_BLK = "SELECT_BLOCKER"
_CAT_CONF_BLK = "CONFIRM_BLOCKERS"
_CAT_ACTIVATE = "ACTIVATE_ABILITY"
_CAT_CAST = "CAST_SPELL"
_CAT_TARGET = "SELECT_TARGET"
_CAT_LAND = "PLAY_LAND"
_CAT_OTHER = "OTHER_CHOICE"
_CAT_MULLIGAN = "MULLIGAN"
_CAT_BOTTOM_CARD = "BOTTOM_DECK_CARD"
_CAT_MANA_U = "MANA_U"
_CAT_MANA_R = "MANA_R"
_CAT_SEARCH = "SEARCH_LIBRARY"
_CAT_TOP_LIBRARY = "TOP_LIBRARY"
_CAT_SHUFFLE_YES = "SHUFFLE"
_CAT_PAYING = "PAYING_COSTS"
_CAT_DIG = "DIG_CHOICE"
_CAT_SB_IN = "SIDEBOARD_IN"
_CAT_SB_OUT = "SIDEBOARD_OUT"
_CAT_SB_DONE = "SIDEBOARD_DONE"
_CAT_SACRIFICE = "SACRIFICE_PERMANENT"
_CAT_RETURN = "RETURN_PERMANENT"
_CAT_CHOOSE_X = "CHOOSE_X"
_CAT_DISCARD = "DISCARD"
_CAT_CHOOSE_MODE = "CHOOSE_MODE"
_CAT_PAY_UNLESS = "PAY_UNLESS"
_CAT_NAME_CARD = "NAME_CARD"
_CAT_KEEP_LEGEND = "KEEP_LEGEND"
_CAT_CHOOSE_REPLACEMENT = "CHOOSE_REPLACEMENT"
_CAT_ATTACK_TARGET = "ATTACK_TARGET"
_CAT_BLOCK_TARGET = "BLOCK_TARGET"
_CAT_YESNO = "OPTIONAL_YESNO"
_CAT_SYLVAN = "SYLVAN_CHOICE"
_CAT_CHOOSE_CARD = "CHOOSE_CARD"
_CAT_COMPANION = "COMPANION"
_CAT_SHUFFLE_NO = "DONT_SHUFFLE"
_CAT_KEEP_HAND = "KEEP_HAND"

# ActionRefZone names (the zone an action's object or player is in, viewer-relative).
_REF_NONE = "NONE"
_REF_SELF_BATTLEFIELD = "SELF_BATTLEFIELD"
_REF_SELF_HAND = "SELF_HAND"
_REF_PLAYER_SELF = "PLAYER_SELF"
_REF_PLAYER_OPP = "PLAYER_OPP"

_STEP_DECLARE_BLOCKERS = "DECLARE_BLOCKERS"


class Skill(Enum):
    """Scripted-agent skill tiers.

    RANDOM    — picks uniformly among the legal actions (a weak floor opponent).
    GREEDY    — the original rule-based behaviour; the baseline opponent.
    HEURISTIC — GREEDY's scan with selected decision points overridden by
                single-ply heuristics (board eval, combat sim, smarter targeting).
    EXPLORE   — coverage-oriented fuzzer: a deterministic, seed-mixed weighted
                choice biased toward the actions that drive the most engine code
                (casting, activating, combat, varied targets/modes/X), plus a
                per-game novelty boost for not-yet-chosen cast/activate cards.
                Not meant to play well — meant so that sweeping a range of seeds
                exercises as many distinct engine paths as possible
                (see _explore_action).
    """
    RANDOM = "random"
    GREEDY = "greedy"
    HEURISTIC = "heuristic"
    EXPLORE = "explore"


@dataclass(frozen=True)
class AgentConfig:
    """Behaviour toggles for a scripted agent.

    HEURISTIC reuses GREEDY everywhere a toggle is off, so anything not explicitly
    improved falls back to today's proven behaviour (notably the Doomsday combo
    lines, which must never regress — see ``enable_combo_lines``).
    """
    skill: Skill = Skill.GREEDY
    use_combat_sim: bool = False        # profitable attacks/blocks via local P/T sim
    use_eval_targeting: bool = False    # removal -> biggest threat, burn -> lethal-only
    use_smart_mulligan: bool = False    # keep based on opening-hand land count
    use_tron_synergy: bool = False      # prioritize completing the Urza's Tron land set
    enable_combo_lines: bool = True     # keep deck-specific combo logic (Doomsday etc.)
    rng_seed: int | None = None         # seed for RANDOM's generator (reproducible)
    explore_profile: str = "default"    # EXPLORE weight profile: "default" (coverage)
                                        # or "patient" (big-mana; _EXPLORE_PATIENT_WEIGHTS)


# The "hard" configuration: heuristic play with combat simulation, evaluation-based
# targeting, and smart mulligans. This is ALSO the default a bare "scripted" resolves
# to — every default scripted opponent (the league/self-play scripted anchor, the
# mixed opponent pools, observe/baseline/analysis, the test harness) plays at this
# tier unless an explicit weaker suffix ("scripted:easy"/"greedy"/"random") is given.
_HARD_CONFIG = AgentConfig(skill=Skill.HEURISTIC, use_combat_sim=True,
                           use_eval_targeting=True, use_smart_mulligan=True,
                           use_tron_synergy=True)

# Canonical presets resolved from a spec-string suffix (see preset_config). The config
# object is read-only (the agent only reads it), so the three hard aliases may
# safely share one instance.
_PRESETS: dict[str, AgentConfig] = {
    "random":    AgentConfig(skill=Skill.RANDOM, enable_combo_lines=False),
    "easy":      AgentConfig(skill=Skill.GREEDY),
    "greedy":    AgentConfig(skill=Skill.GREEDY),
    "scripted":  _HARD_CONFIG,     # bare "scripted"/default now == hard (was greedy)
    "hard":      _HARD_CONFIG,
    "heuristic": _HARD_CONFIG,
    # Coverage fuzzer (see Skill.EXPLORE). Combo lines are intentionally OFF so it
    # never collapses into a fixed deck-specific line — it fans out across cards.
    "explore":   AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False),
    "fuzz":      AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False),
    # Big-mana explore variant (see _EXPLORE_PATIENT_WEIGHTS): develops mana and
    # holds expensive cards until they are castable. Plain explore never even sees
    # a 7-drop as a LEGAL option — this profile exists to create that availability.
    "explore:patient": AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False,
                                   explore_profile="patient"),
    "patient":         AgentConfig(skill=Skill.EXPLORE, enable_combo_lines=False,
                                   explore_profile="patient"),
}


def preset_config(spec: str = "scripted") -> AgentConfig:
    """The AgentConfig a spec string names.

    Accepts ``"scripted"``, ``"scripted:hard"``, ``"scripted:easy"``,
    ``"scripted:random"``, ``"scripted:explore"``, or the bare suffixes
    ``"greedy"``/``"hard"``/``"heuristic"``/``"easy"``/``"random"``/
    ``"explore"``/``"fuzz"``/``"explore:patient"``/``"patient"``
    (case-insensitive). ``explore``/``fuzz`` select the coverage fuzzer
    (Skill.EXPLORE) — vary the game ``--seed`` to fan it across engine paths.
    ``explore:patient`` (or bare ``patient``; ``scripted:explore:patient`` also
    resolves) is the fuzzer's big-mana profile: it develops mana and holds
    expensive cards until they are castable (see ``_EXPLORE_PATIENT_WEIGHTS``).
    """
    s = (spec or "scripted").strip().lower()
    suffix = s.split(":", 1)[1] if s.startswith("scripted:") else s
    if suffix in ("", "scripted"):
        suffix = "scripted"
    if suffix not in _PRESETS:
        raise ValueError(
            f"unknown scripted-agent spec {spec!r}; "
            f"valid suffixes: {sorted(_PRESETS)}"
        )
    return _PRESETS[suffix]


# Per-action-category weights for the EXPLORE tier (_explore_action). Higher weight =
# more likely to be chosen. The bias is toward the categories that exercise the most
# engine code (casting, activating, combat), with PASS kept low-but-nonzero so the game
# still advances, and the "finish this phase" confirms (attackers/blockers/sideboard)
# kept below their per-item counterparts so a few items get declared before confirming.
# Any category not listed (targets, X, modes, mana, search, dig, discard, sacrifice,
# name-a-card, ...) uses _EXPLORE_DEFAULT_WEIGHT — a plain uniform pick among that
# decision's options, so which target / X / mode is taken varies freely across seeds
# (one exception: the SEARCH_LIBRARY fail-to-find slot, see
# _EXPLORE_SEARCH_FAIL_WEIGHT).
_EXPLORE_DEFAULT_WEIGHT = 2.0
_EXPLORE_WEIGHTS: dict[str, float] = {
    _CAT_CAST:     8.0,   # cast a spell — stack, resolution, effects, triggers, SBAs
    _CAT_ACTIVATE: 6.0,   # activated abilities (mana, sacrifice, ETB-independent effects)
    _CAT_LAND:     4.0,   # play a land (fuels casts; land ETBs / fetch / man-lands)
    _CAT_SEL_ATK:  3.0,   # declare an attacker — combat triggers, damage, first strike
    _CAT_SEL_BLK:  3.0,   # declare a blocker — blocking, combat damage assignment
    _CAT_PASS:     1.0,   # advance priority / the game (low, but never zero)
    _CAT_CONF_ATK: 0.6,   # finish declaring attackers (declare a few first)
    _CAT_CONF_BLK: 0.6,   # finish declaring blockers
    _CAT_SB_DONE:  0.6,   # finish sideboarding (swap a few cards first, bo3)
    _CAT_COMPANION: 8.0,  # once-per-game and rarely offered — take it when legal so
                          # the companion machinery (and the card itself) gets fuzzed
}

# Novelty bias for the EXPLORE tier: a CAST_SPELL / ACTIVATE_ABILITY option whose
# card this agent has NOT yet chosen this game gets its category weight
# multiplied by this factor. Without it, expensive cards sharing a deck with
# cheaper lines were NEVER cast across a whole seed sweep (e.g. Meteor Sword,
# hard-cast Lorien Revealed) — the weighted draw kept landing on the cheap
# alternatives. First-time options winning ~3x more often fixes that while the
# already-seen cards keep their normal weight. Tracked per game in
# ``ScriptedCore._explore_seen`` (see ``new_game``).
_EXPLORE_NOVELTY_FACTOR = 3.0

# SEARCH_LIBRARY fail-to-find de-weight (relative to 1.0 per real pick; applies to
# BOTH explore profiles — a category-level tweak, not a patient one). A library
# search menu offers "Fail to find" alongside the real picks at the same uniform
# default weight, so a uniform draw failed ~1,200 tutors across a 441-game fuzz
# campaign — wasting coverage of the ChangeZone search/shuffle/ETB paths behind
# every fetch land and tutor. 0.3 keeps the fail branch reachable (the
# shuffle-without-find path still wants occasional fuzzing) while a real pick wins
# the large majority of searches. The fail slot is identified by its missing card,
# not by index: the engine builds it from Entity(0) — no card — while every real
# pick is a library card. Mandatory searches simply have no such slot, and a
# fail-only menu is a 1-choice decision handled before weighting.
_EXPLORE_SEARCH_FAIL_WEIGHT = 0.3

# ── "Patient" big-mana EXPLORE profile (explore:patient) ────────────────────
# The plain-explore weight draw only picks among LEGAL actions, and empirically
# (130+ explore games) expensive cards never became legal: explore never develops
# enough mana for a 7-drop, and cheap alternate lines with the same card (e.g.
# islandcycling Lorien Revealed) remove the card from hand long before it could be
# hard-cast. No weighting among legal options fixes availability — this profile
# does, by developing mana and holding expensive cards. Shares all the EXPLORE
# machinery (novelty bias on casts, seed-mixed global-random draw); only the
# weights below differ. Plain "explore" reads only _EXPLORE_WEIGHTS and is
# untouched by this table.
_EXPLORE_PATIENT_WEIGHTS: dict[str, float] = {
    _CAT_CAST:     8.0,   # base; scaled up by mana value (_PATIENT_CAST_MV_FACTOR)
                          # so a legal expensive cast dominates the draw
    _CAT_ACTIVATE: 1.0,   # demoted hard: cheap activations (cycling, sac lines)
                          # are what strip expensive cards from hand early
    _CAT_LAND:     100.0, # near-always take the land drop — mana development is
                          # the whole point of this profile
    _CAT_SEL_ATK:  3.0,   # keep combat as in plain explore so games stay decisive
    _CAT_SEL_BLK:  3.0,
    _CAT_PASS:     6.0,   # bank mana-development turns: an early cheap ACTIVATE
                          # now competes with a strong PASS instead of winning
    _CAT_CONF_ATK: 0.6,   # confirms below their per-item counterparts, as in
    _CAT_CONF_BLK: 0.6,   # _EXPLORE_WEIGHTS, so a few get declared first
    _CAT_SB_DONE:  0.6,
    _CAT_COMPANION: 8.0,  # as in _EXPLORE_WEIGHTS: once-per-game, take it when legal
}

# Patient-only CAST scaling: weight *= (1 + mana_value * factor). A legal MV-7
# cast gets 8x the pull of an MV-0 one, so once the mana finally exists the big
# card is cast almost immediately instead of losing the draw to cheap spells.
_PATIENT_CAST_MV_FACTOR = 1.0

# Patient mode gives NO novelty boost to ACTIVATE options. The novelty factor is
# what makes plain explore eagerly try an unseen islandcycle/sac activation the
# first turn it appears — exactly the card-stripping behaviour this profile must
# suppress. Casts keep the boost (a novel expensive cast should win the draw).
_PATIENT_ACTIVATE_NOVELTY = False

# Patient-only "hold" scaling for card-consuming activations: an ACTIVATE whose
# card is still in the acting player's HAND (cycling, Talon Gates-style
# from-hand lines) gets its weight DIVIDED by (1 + mana_value * factor) — the
# mirror image of the CAST scaling. A flat ACTIVATE demotion is not enough: the
# option reappears at every priority window, so even a small per-window pick
# probability compounds into "always cycled before turn 5" over a game. Scaling
# the demotion by MV holds exactly the cards this profile exists to hard-cast
# while leaving cheap from-hand activations barely affected. Battlefield
# activations (Wasteland, equip, ...) are untouched — their card is not in hand.
_PATIENT_HOLD_MV_FACTOR = 1.0


# ── The cards the rules key on (card names, as the card vocabulary spells them) ─
_WASTELAND = "Wasteland"
_AETHER_VIAL = "Aether Vial"
# All basic lands in the vocab. Used by the Wasteland gate ("does the opponent
# have a nonbasic land worth destroying"): a missing entry makes an opposing
# basic look nonbasic, so Wasteland fires with only its controller's own
# nonbasics as legal targets and destroys one of them.
_BASIC_LANDS = frozenset({"Mountain", "Forest", "Island", "Plains", "Swamp",
                          "Snow-Covered Island", "Wastes"})
# Every counterspell in the vocab: these are only cast with an opponent spell
# on the stack (a missing entry let the agent counter its OWN spell, the only
# legal target when nothing else is stacked).
_COUNTER_SPELLS = frozenset({"Force of Will", "Daze", "Counterspell", "Force of Negation",
                             "Consign to Memory", "Flusterstorm", "Spell Pierce",
                             "Spell Snare"})
_COUNTERSPELL = "Counterspell"
# Pure cantrips the hard tier lets resolve rather than countering (threat-type
# triage: a counter spent on the opponent's Ponder is a counter unavailable for
# their threat; everything NOT in this set — any creature/planeswalker/
# enchantment at any MV, and any other MV>=2 spell — stays counterable).
_COUNTER_EXEMPT_CANTRIPS = frozenset({"Ponder", "Brainstorm", "Mishra's Bauble",
                                      "Once Upon a Time", "Consider", "Preordain",
                                      "Careful Study", "Otherworldly Gaze"})
# Targeted removal the hard tier holds until a real threat exists — power >= 2,
# or any NONTOKEN 1-power creature (an unflipped Delver / Dragon's Rage
# Channeler is a real threat; a 1/1 token is not).
_TARGETED_REMOVAL = frozenset({"Swords to Plowshares", "Fatal Push", "Prismatic Ending"})
# Wrath of the Skies: symmetric X sweeper (destroys artifacts/creatures/
# enchantments with MV <= energy paid) — gated on the board actually favoring it.
_WRATH_OF_SKIES = "Wrath of the Skies"
# Death and Taxes: Solitude's ETB exiles a creature, so hard-cast it only with an
# opponent creature to hit. The One Ring's {T} draws a card per burden counter.
_SOLITUDE = "Solitude"
_THE_ONE_RING = "The One Ring"

# Doomsday deck cards
_DOOMSDAY = "Doomsday"
_DARK_RITUAL = "Dark Ritual"
_LOTUS_PETAL = "Lotus Petal"
_LED = "Lion's Eye Diamond"
_THASSAS_ORACLE = "Thassa's Oracle"
_STREET_WRAITH = "Street Wraith"
_EDGE_OF_AUTUMN = "Edge of Autumn"
_CONSIDER = "Consider"
_PONDER = "Ponder"
_BRAINSTORM = "Brainstorm"

# Maverick (green/white) deck cards
_GREEN_SUNS_ZENITH = "Green Sun's Zenith"
_KEEN_EYED_CURATOR = "Keen-Eyed Curator"
# Untapped creatures that tap for mana (count toward X for Green Sun's Zenith).
_MANA_DORKS = frozenset({"Birds of Paradise", "Ignoble Hierarch", "Noble Hierarch"})
# Lands that do NOT reliably produce mana when untapped, so they shouldn't count
# toward an affordable X: fetchlands (sacrifice for a land, no mana) and Gaea's
# Cradle (G per creature — zero with an empty board).
_UNRELIABLE_LANDS = frozenset({"Scalding Tarn", "Flooded Strand", "Polluted Delta",
                               "Wooded Foothills", "Misty Rainforest", "Windswept Heath",
                               "Bloodstained Mire", "Verdant Catacombs", "Gaea's Cradle"})
# Cantrips/digging that let us keep a one-land opening hand (they find more lands)
_KEEP_ONE_LANDERS = frozenset({_PONDER, _BRAINSTORM, "Once Upon a Time"})
# Self-controlled draw/cycling spells or abilities that, once on the stack, make it
# correct to crack Lion's Eye Diamond in response (float 3 mana that outlives the
# resolving draw, which then refills the discarded hand). Brainstorm is the canonical
# LED-response cantrip — crack LED, discard hand, then Brainstorm draws 3 fresh cards.
# Mishra's Bauble's sac-to-draw activation counts as the on-stack draw marker too.
_LED_DRAW_STACK = frozenset({_STREET_WRAITH, _EDGE_OF_AUTUMN, _CONSIDER, "Deep Analysis",
                             _PONDER, _BRAINSTORM, "Mishra's Bauble"})
_DOOMSDAY_DECK = frozenset({"Doomsday", "Thoughtseize", "Dark Ritual", "Lotus Petal",
                            "Lion's Eye Diamond", "Thassa's Oracle", "Personal Tutor",
                            "Street Wraith", "Edge of Autumn", "Cavern of Souls", "Consider",
                            "Duress", "Deep Analysis"})

# Tron deck cards. The three Urza lands each add extra colorless mana once the
# other two are also controlled (see Count$UrzaLands in src/components/ability.cpp);
# Planar Nexus is every nonbasic land type via a self-CDA (state_manager_statics.cpp),
# so a single Planar Nexus supplies whichever of the three types a real piece is
# missing. The Urza lands are kept in this order (the order _tron_choice tries them).
_TRON_LANDS = ("Urza's Mine", "Urza's Power Plant", "Urza's Tower")
_TRON_LAND_SET = frozenset(_TRON_LANDS)
_PLANAR_NEXUS = "Planar Nexus"
# Karn, the Great Creator wishes an artifact from the sideboard/exile into hand
# (-2). Mycosynth Lattice is the priority wish — with Karn in play it turns every
# opponent permanent into an artifact whose abilities Karn's static shuts off (the
# hard lock, incl. lands not tapping for mana). Cityscape Leveler is the follow-up
# beater once the lock (or a lack of Lattice) leaves nothing better to grab.
_KARN_GREAT_CREATOR = "Karn, the Great Creator"
_CITYSCAPE_LEVELER = "Cityscape Leveler"
_MYCOSYNTH_LATTICE = "Mycosynth Lattice"
# Candelabra of Tawnos untaps X target lands; it's a mana engine only when it
# untaps lands that make MORE THAN ONE mana, so we bias its targets to those.
# Ancient Tomb ({C}{C}) and Urza's Workshop (metalcraft: {C} per Urza's land)
# always/often produce >1; the three Urza's tron lands produce {C}{C}{C} each
# once the set is assembled. Planar Nexus is deliberately excluded — it only
# ever makes a single mana of any type, so untapping it gains nothing.
_CANDELABRA = "Candelabra of Tawnos"
_MULTI_MANA_LANDS = frozenset({"Ancient Tomb", "Urza's Workshop"}) | _TRON_LAND_SET

# Reanimator deck cards. Reanimate and Animate Dead are the "put a creature card
# from a graveyard onto the battlefield" spells; the fatties are the creatures those
# spells exist to cheat out (the deck's discard outlets put them in the graveyard
# first).
_REANIMATION_SPELLS = frozenset({"Reanimate", "Animate Dead"})
_REANIMATION_FATTIES = frozenset({"Griselbrand", "Archon of Cruelty", "Atraxa, Grand Unifier"})

# Lands deck cards. Thespian's Stage copying Dark Depths yields a zero-ice-counter
# Depths (copy effects don't copy counters, CR 706.2) whose "no ice counters"
# trigger makes Marit Lage — the deck's win condition. Urza's Saga's chapter-II
# ability builds Construct tokens, and Life from the Loam's dredge recurs
# Saga/Wasteland from the graveyard every turn.
_LIFE_FROM_LOAM = "Life from the Loam"
_EXPLORATION = "Exploration"
_URZAS_SAGA = "Urza's Saga"
_DARK_DEPTHS = "Dark Depths"
_THESPIANS_STAGE = "Thespian's Stage"
_TABERNACLE = "The Tabernacle at Pendrell Vale"
# Lands a sacrifice cost should never eat while another option exists — the
# combo pieces and the irreplaceable utility lands (a Crop Rotation that sacs
# Dark Depths to fetch Thespian's Stage has defeated itself).
_SAC_LAST_LANDS = frozenset({_DARK_DEPTHS, _THESPIANS_STAGE, _URZAS_SAGA, _TABERNACLE,
                             "Karakas", "Boseiju, Who Endures", "Blast Zone", "Maze of Ith"})


def _deck_named(deck, key):
    """True if `key` occurs anywhere in the deck name (e.g. 'league/car_doomsday').

    The uniform way archetype-specific behavior is keyed: doomsday (combo) and tron
    (big-mana ramp) decks are recognised this way."""
    return deck is not None and key in deck.lower()


def _is_land_card(v, card) -> bool:
    """True if the card is a land (its type line includes Land)."""
    return card is not None and v.cards.is_land(card)


def _card_mana_value(v, card) -> float:
    """Printed mana value of a card (X counts 0); 0 for no card or an unknown one."""
    return v.cards.mana_value(card) if card is not None else 0.0


def _hand_has_card(v, card) -> bool:
    """Whether the priority player's hand holds a copy of `card`."""
    return card in v.hand


def _is_doomsday_deck(v) -> bool:
    """Heuristic: a doomsday-deck-only card is in hand, on our battlefield
    (Cavern of Souls, Lotus Petal, Lion's Eye Diamond), or in a graveyard —
    post-Doomsday the hand is often cantrips and counters only, while the
    resolved Doomsday and the Dark Ritual that paid for it sit in the
    graveyard, and that is exactly when the post-pile rules must fire."""
    if any(c in _DOOMSDAY_DECK for c in v.hand):
        return True
    if any(p.card in _DD_EXCLUSIVE for p in v.my_perms_all):
        return True
    return _gy_has_any(v, _DD_EXCLUSIVE)


# Cards only the Doomsday deck runs — the battlefield / graveyard half of
# _is_doomsday_deck scans for these alone. The wider _DOOMSDAY_DECK hand
# set includes Dark Ritual and Lotus Petal, which reanimator also runs: a
# Ritual in reanimator's graveyard must not flip it onto the Doomsday rules
# for the rest of the game.
_DD_EXCLUSIVE = frozenset({
    "Doomsday",
    "Lion's Eye Diamond",
    "Thassa's Oracle",
    "Personal Tutor",
    "Street Wraith",
    "Edge of Autumn",
})


def _all_eligible_creatures_attacking(v) -> bool:
    """Return True if every untapped, non-sick creature self controls is attacking."""
    any_eligible = False
    for p in v.my_perms:
        if not p.creature:
            continue  # not a creature (land, or other permanent)
        if p.tapped or p.sick:
            continue  # can't attack
        any_eligible = True
        if not p.attacking:
            return False  # eligible but not yet attacking
    return any_eligible


def _opponent_has_nonbasic_land(v) -> bool:
    """Return True if opponent has at least one nonbasic land."""
    for p in v.opp_perms:
        if p.land and p.card is not None and p.card not in _BASIC_LANDS:
            return True
    return False


def _opponent_has_creature(v) -> bool:
    """Return True if opponent controls at least one creature."""
    return any(p.creature for p in v.opp_perms)


def _controls_card(v, card) -> bool:
    """True if self controls an unphased permanent named `card`."""
    return any(p.card == card for p in v.my_perms)


def _gy_has_any(v, cards: frozenset) -> bool:
    """True if EITHER graveyard holds a card in ``cards``. Both sides matter to the
    caller (Reanimate can take a creature from any graveyard)."""
    return not cards.isdisjoint(v.graveyard_cards)


def _tapped_multi_mana_lands(v) -> int:
    """Count self-controlled TAPPED lands that make more than one mana.

    These are the lands worth untapping with Candelabra of Tawnos — untapping a tapped
    Ancient Tomb / metalcraft Urza's Workshop / assembled Urza's tron land frees 2-3 mana
    apiece, whereas untapping a one-mana land (or any untapped land) gains nothing.
    """
    return sum(1 for p in v.my_perms if p.tapped and p.card in _MULTI_MANA_LANDS)


def _controlled_tron_pieces(v) -> set:
    """Urza's Mine/Power-Plant/Tower self currently controls. Planar Nexus is
    deliberately excluded — it's the substitute offered once a real piece is
    missing, not a piece itself."""
    return {p.card for p in v.my_perms if p.card in _TRON_LAND_SET}


def _tron_choice(v, cat: str, controlled: set) -> int | None:
    """Pick whichever offered ``cat``-category action completes the Tron set:
    a real Urza's Mine/Power-Plant/Tower self doesn't yet control, else Planar
    Nexus (every nonbasic land type via its self-CDA, so it substitutes for any
    missing piece — see state_manager_statics.cpp). Returns None if nothing
    offered helps, so the caller can fall back to its normal pick.
    """
    cats = v.cats
    for target in _TRON_LANDS:
        if target in controlled:
            continue
        for i, c in enumerate(cats):
            if c == cat and v.card(i) == target:
                return i
    for i, c in enumerate(cats):
        if c == cat and v.card(i) == _PLANAR_NEXUS:
            return i
    return None


def _opponent_has_spell_on_stack(v) -> bool:
    """Return True if at least one spell/ability on the stack is not controlled by self."""
    return any(not s.mine for s in v.stack)


def _self_stack_has_card(v, card) -> bool:
    """Return True if self controls a stack object with the given card."""
    return any(s.mine and s.card == card for s in v.stack)


def _opponent_threat_on_stack(v) -> bool:
    """Threat-type counter triage: True if a non-self stack object is worth a
    counterspell — anything that is NOT a known pure cantrip
    (_COUNTER_EXEMPT_CANTRIPS). Creatures/planeswalkers/enchantments at any
    MV and every other MV>=2 spell fall through the exemption and count."""
    return any(not s.mine and s.card not in _COUNTER_EXEMPT_CANTRIPS for s in v.stack)


def _self_has_draw_on_stack(v) -> bool:
    """Return True if self controls a cycling or draw ability/spell on the stack."""
    return any(s.mine and s.card in _LED_DRAW_STACK for s in v.stack)


def _hand_land_count(v) -> int:
    """Lands in the priority player's hand."""
    return sum(1 for c in v.hand if _is_land_card(v, c))


def _self_land_count(v) -> int:
    """Lands self controls (unphased)."""
    return sum(1 for p in v.my_perms if p.land)


# ── Doomsday combo constants/helpers ────────────────────────────────────────
# Doomsday keeps exactly 5 cards on top of the library (doomsday.txt ChangeNum$ 5).
# Thassa's Oracle wins the game when devotion-to-blue (>= 2 from its own UU) is
# >= the library size, so casting Oracle with <= 2 cards left is a guaranteed win.
_DD_PILE_SIZE = 5
_ORACLE_WIN_LIBRARY = 2

# Green Sun's Zenith costs {X}{G}. Hold it until at least this many untapped mana
# sources are available so it is never cast for X=0 (which can only fetch the
# 0-mana Dryad Arbor); 2 sources buy X=1, more buy a bigger creature.
_GSZ_MIN_MANA_SOURCES = 2

# Street Wraith cycling costs 2 life; never pay it below this so the agent can't
# cycle itself to death digging through the pile.
_STREET_WRAITH_MIN_LIFE = 3

# Permanents that can produce blue mana for Thassa's Oracle (UU). Cavern of Souls
# counts because Oracle is a creature spell; Lotus Petal makes any colour when
# sacrificed. We gate Doomsday on controlling at least two so the UU for Oracle
# is actually available after the combo resolves.
_BLUE_SOURCES_NEEDED = 2
_BLUE_SOURCES = frozenset({
    "Volcanic Island",    # Volcanic Island (U/R)
    "Tundra",   # Tundra (W/U)
    "Island",   # Island (U)
    "Undercity Sewers",   # Undercity Sewers (U/B)
    "Underground Sea",   # Underground Sea (U/B)
    "Cavern of Souls",   # Cavern of Souls (any colour for creature spells, e.g. Oracle)
    "Lotus Petal",   # Lotus Petal (sacrifice: one mana of any colour)
    "Tropical Island",  # Tropical Island (U/G) — wubg_doomsday runs 2; omitting it
          # undercounted blue sources and held Doomsday longer than needed
})

# ── Doomsday all-in mulligan (vs decks with no Force of Will) ───────────────
# Hand-based mana classification for the fast-kill keep test. A fetch counts
# for every colour the deck's duals let it find (Flooded Strand and Polluted
# Delta fetch Underground Sea, Verdant Catacombs fetches Sea or Tropical);
# Lotus Petal counts for both sides.
_DD_HAND_B_SOURCES = frozenset({
    "Swamp",
    "Undercity Sewers",   # Undercity Sewers (U/B)
    "Underground Sea",   # Underground Sea (U/B)
    "Scrubland",  # Scrubland (W/B)
    "Bloodstained Mire",   # Bloodstained Mire (fetches a Swamp-typed dual)
    "Verdant Catacombs",   # Verdant Catacombs (fetches Underground Sea)
    "Flooded Strand",    # Flooded Strand (fetches Underground Sea / Scrubland)
    "Polluted Delta",    # Polluted Delta (fetches Sea / Sewers)
    "Lotus Petal",
})
_DD_HAND_U_SOURCES = frozenset({
    "Island",
    "Tundra",   # Tundra (W/U)
    "Volcanic Island",    # Volcanic Island (U/R)
    "Undercity Sewers",   # Undercity Sewers (U/B)
    "Underground Sea",   # Underground Sea (U/B)
    "Cavern of Souls",   # Cavern of Souls (creature spells — pays for Oracle)
    "Tropical Island",  # Tropical Island (U/G)
    "Scalding Tarn",    # Scalding Tarn (fetches an Island-typed dual)
    "Misty Rainforest",    # Misty Rainforest (fetches an Island-typed dual)
    "Flooded Strand",
    "Polluted Delta",
    "Verdant Catacombs",   # Verdant Catacombs (fetches Tropical Island)
    "Lotus Petal",
})

# Instant-speed removal that can kill Thassa's Oracle mid-combo (in response
# to the cast or the win trigger) — what a Doomsday deck's Thoughtseize-style
# discard strips right after countermagic. The instant subset of
# _TARGETED_REMOVAL: Prismatic Ending is sorcery-speed, so it can't
# disrupt the kill and stays on the highest-MV fallback.
_DD_DISRUPTOR_REMOVAL = frozenset({
    "Swords to Plowshares",
    "Fatal Push",
})


def _dd_fast_hand(v, hand) -> bool:
    """Can this opening hand cast Doomsday by turn 1-3 and convert right
    after? Requirements: Doomsday itself; BBB reachable on at most three land
    drops plus Petals plus a Dark Ritual's net +2; a black-capable source to
    kick things off; and two blue-capable sources for Thassa's Oracle's UU —
    or Lion's Eye Diamond, which pays for Oracle by itself in the crack
    window. Sources are counted per-card across turns (the ritual land untaps
    before the Oracle turn), so one dual serving both colours is fine.
    Battlefield sources count too — normally none exist at the mulligan, but
    sculpted harness scenarios preset them."""
    if _DOOMSDAY not in hand:
        return False
    rituals = sum(1 for c in hand if c == _DARK_RITUAL)
    petals = sum(1 for c in hand if c == _LOTUS_PETAL)
    b_lands = sum(1 for c in hand
                  if c in _DD_HAND_B_SOURCES and c != _LOTUS_PETAL)
    bf_b = _count_untapped_sources(v, _DD_HAND_B_SOURCES)
    if min(b_lands, 3) + bf_b + petals == 0:
        return False
    if min(b_lands, 3) + bf_b + petals + 2 * min(rituals, 1) < 3:
        return False
    blue = (sum(1 for c in hand if c in _DD_HAND_U_SOURCES)
            + _count_untapped_sources(v, _DD_HAND_U_SOURCES))
    if _LED in hand:
        blue += 2
    return blue >= 2


def _led_crack_choice(v) -> int | None:
    """Lion's Eye Diamond: crack it for blue only in the Doomsday kill window. All of:

      1. a self-controlled draw/cycling spell or ability is on the stack, AND
      2. Thassa's Oracle is the known top card of the library (Doomsday has resolved
         and laid the pile — RearrangeTopOfLibrary makes the top cards known), AND
      3. the library is down to <= 2 cards (the pile is drawn down to the win).

    In that window the resolving draw puts Oracle into hand and the library hits the
    Oracle-win size; cracking LED floats UUU that outlives the draw and pays Oracle's
    UU. Gated this tightly, LED's "discard your hand" cost never sheds a card that
    still matters, so no empty-hand check is needed. Returns the MANA_U action index
    for LED, or None when the window / the offer isn't present.
    """
    if not _self_has_draw_on_stack(v):
        return None
    if v.known_top(1)[0] != _THASSAS_ORACLE:
        return None
    # The pending draw takes Oracle off the top, so the library is one card
    # above the win size while it sits on the stack (standard pile with LED
    # already in play: the Street Wraith draw fires at three cards).
    if v.library > _ORACLE_WIN_LIBRARY + 1:
        return None
    for i, c in enumerate(v.cats):
        if c == _CAT_MANA_U and v.card(i) == _LED:
            return i
    return None


def _count_untapped_sources(v, cards) -> int:
    """Untapped permanents self controls whose card is in ``cards``."""
    return sum(1 for p in v.my_perms if not p.tapped and p.card is not None and p.card in cards)


def _count_blue_sources(v) -> int:
    """Blue-mana sources self controls, tapped or not. The Doomsday gate asks
    whether Oracle's UU will exist by KILL time (next untap, or the LED crack
    window), so a source tapped mid-combo-turn — e.g. the Underground Sea
    that just paid for Dark Ritual — still counts. Counting only untapped
    sources made the ritual itself flip the Doomsday gate false."""
    return sum(1 for p in v.my_perms if p.card in _BLUE_SOURCES)


def _count_untapped_mana_sources(v) -> int:
    """Untapped mana-producers self controls.

    Counts every untapped land plus the untapped mana-dork creatures (Birds /
    Noble & Ignoble Hierarch). Used to decide whether Green Sun's Zenith can
    afford a meaningful X — its cost is {X}{G}, so N sources buy X = N-1.
    """
    count = 0
    for p in v.my_perms:
        if p.tapped:
            continue
        is_land = p.land and p.card not in _UNRELIABLE_LANDS
        is_dork = p.card in _MANA_DORKS
        if not (is_land or is_dork):
            continue
        # A creature mana-source (dork, or a creature-land like Dryad Arbor) can't
        # tap for mana while summoning-sick, so it can't pay toward X yet.
        if p.creature and p.sick:
            continue
        count += 1
    return count


# Free-draw enablers for the Doomsday pile: cycling that costs no mana (Street
# Wraith pays life, Edge of Autumn sacs a land), so a drawn one is immediately
# re-cycled to keep drawing down the pile the same turn.
_DD_FREE_DRAWS = frozenset({_STREET_WRAITH, _EDGE_OF_AUTUMN})
# Cheap cantrips: preferred pile filler (a drawn one is castable for more digging)
# but NOT counted toward Oracle's position — they cost mana, so the pile plan
# only relies on the free draws plus natural draw steps.
_DD_CANTRIPS = frozenset({_PONDER, _BRAINSTORM, _CONSIDER})


def _hand_free_draw_count(v) -> int:
    """Free-draw cyclers (Street Wraith / Edge of Autumn) in the priority
    player's hand. The Brainstorm same-turn pile plan needs at least one:
    Brainstorm's two put-backs cost two draws, so the pile's own cyclers
    always come up exactly one draw short without a cycler already in hand."""
    return sum(1 for c in v.hand if c in _DD_FREE_DRAWS)


# Blue sources a hand can put onto the battlefield WITHOUT a library search:
# the direct-play half of _DD_HAND_U_SOURCES. Fetch lands are excluded on
# purpose — after Doomsday lays the pile a fetch can't be cracked (the search
# shuffles the pile away), so a fetch in hand covers nothing for Oracle.
_DD_HAND_U_DIRECT = frozenset({"Island", "Tundra", "Volcanic Island", "Undercity Sewers",
                               "Underground Sea", "Cavern of Souls", "Tropical Island",
                               "Lotus Petal"})


def _hand_direct_blue_sources(v) -> int:
    """Blue sources in the priority player's hand that a land drop / a free
    Lotus Petal cast puts into play before the kill turn."""
    return sum(1 for c in v.hand if c in _DD_HAND_U_DIRECT)


def _petals_spent_on_black(v) -> int:
    """How many Lotus Petals on the battlefield Doomsday's BBB (or the Dark
    Ritual starting it) will consume. A Petal is in both the blue and the
    black source sets, but it is one mana once: the agent once cracked its
    only Petal for Dark Ritual, then held Doomsday because that same Petal
    had been counted as Oracle's second blue source."""
    petals = _count_controlled(v, _LOTUS_PETAL)
    if petals == 0:
        return 0
    black_lands = _count_black_sources(v) - petals
    black_needed = 1 if _hand_has_card(v, _DARK_RITUAL) else 3
    return max(0, min(petals, black_needed - black_lands))


def _dd_blue_for_oracle(v) -> bool:
    """Will Thassa's Oracle's UU exist by kill time? Two blue sources on the
    battlefield (Petals the BBB will eat excluded), or an LED that pays for
    Oracle by itself, or one blue source in play plus a directly playable one
    in hand: the kill lands on the next turn's draw step at the earliest, and
    the post-pile Edge / Wraith chain waits for that land drop
    (_dd_edge_cycle_safe) before drawing Oracle. Waiting for the second blue
    land BEFORE casting Doomsday was the main reason the pilot went off on
    turn 7-13 instead of turn 3-5."""
    on_board = _count_blue_sources(v) - _petals_spent_on_black(v)
    if on_board >= _BLUE_SOURCES_NEEDED or _led_covers_oracle(v):
        return True
    if on_board < 1:
        return False
    # One real source is enough when the second can come off the top of the
    # pile: the basic Island is still in the library, and the pile plan
    # (_dd_pile_wants_land) puts it there, drawn before the chain reaches
    # Oracle. Waiting for a second blue land drop instead cost a turn. Not
    # from a one-land board, though: a Wasteland on that land leaves no mana
    # at all for the kill.
    return (on_board + _hand_direct_blue_sources(v) >= _BLUE_SOURCES_NEEDED
            or (_dd_pile_land_available(v) and _count_lands(v) >= 2))


def _dd_pile_land_available(v) -> bool:
    """Is the basic Island (the pile's blue land) still in the library? The
    view's own-library knowledge when it has some; else, as it is a single
    copy: not on our battlefield, not in hand, not in a graveyard. (Exile is
    not checked — pre-Doomsday nothing of ours is exiled, and post-Doomsday
    the question no longer arises.)"""
    island = _DD_PILE_LANDS[0]
    known = v.own_library_has(island)
    if known is not None:
        return known
    return (not _controls_card(v, island)
            and not _hand_has_card(v, island)
            and not _gy_has_any(v, frozenset({island})))


def _led_covers_oracle(v) -> bool:
    """Lion's Eye Diamond pays Thassa's Oracle's UU on its own: cracked in the
    kill window (_led_crack_choice) it floats three blue that outlive the final
    cycle draw. So an LED in hand (castable for 0) or already on the
    battlefield satisfies the blue coverage the Doomsday gate demands even
    when fewer than two untapped blue sources are up."""
    return _hand_has_card(v, _LED) or _controls_card(v, _LED)


# Black-mana producers a Doomsday deck can control (Doomsday costs BBB; Dark
# Ritual needs one B to start). Lotus Petal makes any colour.
_DD_BLACK_SOURCES = frozenset({
    "Swamp",
    "Undercity Sewers",   # Undercity Sewers (U/B)
    "Underground Sea",   # Underground Sea (U/B)
    "Scrubland",  # Scrubland (W/B)
    "Lotus Petal",
})
# Cards a pre-pile surveil must keep on top rather than mill: the kill itself,
# the mana that casts it, and the pile's free draws.
_DD_COMBO_PIECES = frozenset({
    _DOOMSDAY, _DARK_RITUAL, _THASSAS_ORACLE,
    _LED, _LOTUS_PETAL, _STREET_WRAITH,
    _EDGE_OF_AUTUMN,
    "Personal Tutor",
})
# Fetch-land / land-search preference while piloting Doomsday: a dual serving
# BOTH combo colours first (Underground Sea untapped ahead of the enters-tapped
# Sewers), then whichever single colour the board is missing.
_DD_FETCH_BOTH_COLOURS = ("Underground Sea", "Undercity Sewers")
_DD_FETCH_BLACK = ("Scrubland", "Swamp")
_DD_FETCH_BLUE = ("Island", "Tundra", "Tropical Island")
# Smallest library each cantrip may be cast into: Brainstorm draws three,
# Ponder draws one, Consider may surveil its one card away and then draw.
# Drawing from an empty library loses the game.
_DD_CANTRIP_MIN_LIBRARY = {_BRAINSTORM: 3, _PONDER: 1, _CONSIDER: 2}


def _count_lands(v) -> int:
    """Unphased lands self controls, tapped or not."""
    return sum(1 for p in v.my_perms if p.land)


def _count_controlled(v, card) -> int:
    """How many unphased permanents named `card` self controls."""
    return sum(1 for p in v.my_perms if p.card == card)


# Our own spells on the stack that countermagic in hand must protect against
# an opponent's response: the kill, Doomsday, and the Ritual paying for it.
_DD_PROTECT_ON_STACK = (_THASSAS_ORACLE, _DOOMSDAY, _DARK_RITUAL)
# Spells a Doomsday deck may still cast off Petal mana while Doomsday waits in
# hand: the combo and its protection. Anything else (Tamiyo, Teferi, Quantum
# Riddler) must be paid by lands alone — three Petals once went to Ponder,
# Tamiyo and Teferi with Doomsday in hand and the black lands eight turns away.
_DD_PETAL_WORTHY = frozenset({
    _DOOMSDAY, _DARK_RITUAL, _THASSAS_ORACLE,
    _LOTUS_PETAL, _LED,
    _PONDER, _BRAINSTORM, _CONSIDER,
    "Veil of Summer",  # Veil of Summer (protection for the kill turn)
}) | _COUNTER_SPELLS
# Payment / pitch preference: mana sources and pitch fodder ranked worst-first.
# Cracking Petal / LED to pay for a cantrip wastes the combo's mana; pitching
# Oracle to Force of Will ends the game.
_DD_PAY_LAST = frozenset({_LOTUS_PETAL, _LED})


def _count_black_sources(v) -> int:
    """Black-mana producers self controls, tapped or not (the black-side twin
    of _count_blue_sources)."""
    return sum(1 for p in v.my_perms if p.card in _DD_BLACK_SOURCES)


def _dd_bbb_reachable(v) -> bool:
    """Can Doomsday's BBB be paid this turn or next? Black sources on the
    board plus the Lotus Petals in hand (free to cast), and a Dark Ritual in
    hand turns one of those into three."""
    black = _count_black_sources(v)
    black += sum(1 for c in v.hand if c == _LOTUS_PETAL)
    if black >= 1 and _hand_has_card(v, _DARK_RITUAL):
        black += 2
    return black >= 3


def _dd_combo_ready(v) -> bool:
    """Doomsday is castable this turn or next: BBB reachable and Oracle's UU
    covered. Gates the pre-Doomsday Brainstorm hold — while the mana isn't
    there the agent must dig with Brainstorm for the Ritual / Petal / black
    land, not sit on it for a kill it can't start (it once held Brainstorm
    for eight turns with three Doomsdays and Oracle in hand on two black
    sources and no Ritual)."""
    return _dd_bbb_reachable(v) and _dd_blue_for_oracle(v)


def _opponent_board_power(v) -> int:
    """Total power of the opponent's unphased creatures, summoning-sick ones
    included — all of them attack on the opponent's next turn."""
    return sum(p.power for p in v.opp_perms if p.creature)


def _dd_clock_allows_doomsday(v) -> bool:
    """Doomsday costs half our life (rounded up), and the kill lands on the NEXT
    draw step at the earliest unless a free-draw cycler plus Brainstorm are
    already in hand (the same-turn pile line). Hold it when the opponent's
    board would kill us in between — we survive their next attack at full life
    but not at half — since firing it then is a guaranteed loss. If the board
    kills us either way, go off: the combo is the only out."""
    power = _opponent_board_power(v)
    if power == 0:
        return True
    if (_hand_free_draw_count(v) >= 1
            and _hand_has_card(v, _BRAINSTORM)):
        return True
    life = v.life
    if life // 2 > power:
        return True
    return life <= power


def _dd_fetch_pick(v) -> int | None:
    """Land-search pick for a Doomsday deck: Underground Sea / Undercity Sewers
    (both colours) first, then the colour the board lacks — black when nothing
    makes B yet (Dark Ritual and Doomsday both stall without it), else blue
    until Oracle's UU is covered. Returns None when no preferred land is
    offered (a non-land search, or a menu of lands outside the table)."""
    offered: dict = {}
    for i, c in enumerate(v.cats):
        if c == _CAT_SEARCH:
            card = v.card(i)
            if card is not None and card not in offered:
                offered[card] = i
    if not offered:
        return None
    # Doomsday costs BBB: three black sources from lands alone, one plus a
    # Dark Ritual in hand. Blue is covered at Oracle's UU.
    black_needed = 1 if _hand_has_card(v, _DARK_RITUAL) else 3
    order = list(_DD_FETCH_BOTH_COLOURS)
    if _count_black_sources(v) < black_needed:
        order += _DD_FETCH_BLACK + _DD_FETCH_BLUE
    else:
        order += _DD_FETCH_BLUE + _DD_FETCH_BLACK
    for card in order:
        if card in offered:
            return offered[card]
    return None


def _dd_land_sac_key(v, card, tapped: bool):
    """Sort key for sacrificing one of our lands while piloting Doomsday —
    lower is better. In order: would it drop the UNTAPPED blue sources below
    Oracle's UU (the kill may be this very turn); would it drop the total blue
    sources below UU (next turn's kill); would it remove our only black source
    (pre-Doomsday, Dark Ritual and Doomsday still need it); then plain
    blue / black membership as tie-breaks so a colourless or off-colour land
    goes first."""
    is_blue = card in _BLUE_SOURCES
    is_black = card in _DD_BLACK_SOURCES
    untapped_blue = _count_untapped_sources(v, _BLUE_SOURCES)
    blue = _count_blue_sources(v)
    black = _count_black_sources(v)
    return (is_blue and not tapped and untapped_blue - 1 < _BLUE_SOURCES_NEEDED,
            is_blue and blue - 1 < _BLUE_SOURCES_NEEDED,
            is_black and black - 1 < 1,
            is_blue, is_black)


def _dd_edge_cycle_safe(v) -> bool:
    """Can Edge of Autumn's cycle eat one of our lands and still leave Thassa's
    Oracle's UU available — two untapped blue sources behind the sacrifice (or
    an LED covering the cast)? Post-Doomsday the alternative — waiting for the
    natural draw step to reach the next pile card — costs a turn, while
    drawing the pile down onto a board that can't cast Oracle strands it in
    hand until the deck-out. With no UU on the board at all the cycle is
    never worth it: the pile holds no lands, so only a drawn Petal or the
    turn's land drop can fix the mana, and every pile card should wait."""
    if _led_covers_oracle(v):
        return True
    untapped_blue = _count_untapped_sources(v, _BLUE_SOURCES)
    for p in v.my_perms:
        if not p.land:
            continue
        if p.card is None or p.card in _SAC_LAST_LANDS:
            continue
        untapped_blue_land = p.card in _BLUE_SOURCES and not p.tapped
        if untapped_blue - (1 if untapped_blue_land else 0) >= _BLUE_SOURCES_NEEDED:
            return True
    return False


def _dd_sacrifice_pick(v) -> int | None:
    """Sacrifice-cost pick for a Doomsday deck (Edge of Autumn's cycle): the
    land with the lowest _dd_land_sac_key — keep Thassa's Oracle's UU on the
    board above all (sacrificing the Island that was one of two blue sources
    leaves Oracle uncastable with the pile already drawn), then the only
    black source, then prefer an off-colour land. The menu doesn't say which
    offered land is tapped, so a land is treated as tapped only when EVERY
    copy we control is (the untapped-UU term is then moot for it). Protected
    utility lands are never offered up."""
    untapped = {p.card for p in v.my_perms if not p.tapped}
    best, best_key = None, None
    for i, c in enumerate(v.cats):
        if c != _CAT_SACRIFICE:
            continue
        card = v.card(i)
        if card in _SAC_LAST_LANDS:
            continue
        key = _dd_land_sac_key(v, card, card not in untapped)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def _dd_oracle_mana_choice(v, own_main: bool) -> int | None:
    """Thassa's Oracle kill window with the UU not yet available: Lotus Petal
    is a sacrifice mana ability the engine offers only as an explicit MANA_U
    action at priority (lands auto-pay). When Oracle is in hand, the library
    is at the win size, it's our main phase on an empty stack and the engine
    is NOT offering the cast, crack a Petal for blue so the next query offers
    it. Lion's Eye Diamond's MANA_U is excluded — its cost discards the hand,
    Oracle included (its own draw-on-stack window is _led_crack_choice)."""
    if not own_main or not v.stack_empty:
        return None
    if not _hand_has_card(v, _THASSAS_ORACLE):
        return None
    if v.library > _ORACLE_WIN_LIBRARY:
        return None
    cats = v.cats
    for i, c in enumerate(cats):
        if c == _CAT_CAST and v.card(i) == _THASSAS_ORACLE:
            return None
    for i, c in enumerate(cats):
        if c == _CAT_MANA_U and v.card(i) == _LOTUS_PETAL:
            return i
    return None


# The blue land the Doomsday pile carries on top when the board is short of
# blue sources (_dd_pile_wants_land): the basic Island ONLY — Wasteland-proof
# and a single copy, so the selection phase can never pick a second land. A
# Tundra fallback once got picked alongside it, and the extra land pushed the
# cyclers below Oracle.
_DD_PILE_LANDS = ("Island",)


def _dd_pile_wants_land(v) -> bool:
    """Should the pile carry a blue land on top? With only two blue sources
    Edge of Autumn can never be cycled safely (its sacrifice breaks Oracle's
    UU), so the chain stalls on natural draws — a turn slower, and one
    Wasteland on a dual ends the game. A third source drawn off the top of
    the pile makes Edge cyclable and survives the Wasteland. Not needed when
    a blue land is already in hand (the land drop is coming anyway) or LED
    pays for Oracle by itself."""
    return (not _led_covers_oracle(v)
            and _count_blue_sources(v) < 3
            and _hand_direct_blue_sources(v) == 0)


def _classify_top_library(v, want_land: bool = False):
    """Bucket the offered TOP_LIBRARY choices into
    (oracle_idx, led_idx, land_idx, free_draw_idxs, cantrip_idxs, filler_idxs).
    ``land_idx`` is the pile's blue land (see _DD_PILE_LANDS) when
    ``want_land``; otherwise lands are filler."""
    oracle_i, led_i, draw_is, cantrip_is, filler_is = None, None, [], [], []
    land_by_card: dict = {}
    for i, c in enumerate(v.cats):
        if c != _CAT_TOP_LIBRARY:
            continue
        card = v.card(i)
        if card == _THASSAS_ORACLE:
            if oracle_i is None:
                oracle_i = i
        elif card == _LED:
            if led_i is None:
                led_i = i
        elif want_land and card in _DD_PILE_LANDS and card not in land_by_card:
            land_by_card[card] = i
        elif card in _DD_FREE_DRAWS:
            draw_is.append(i)
        elif card in _DD_CANTRIPS:
            cantrip_is.append(i)
        else:
            filler_is.append(i)
    land_i = None
    for card in _DD_PILE_LANDS:
        if card in land_by_card:
            land_i = land_by_card[card]
            break
    return oracle_i, led_i, land_i, draw_is, cantrip_is, filler_is


def _dd_return_pick(v, cat) -> int:
    """Pick a card to give back from hand (Brainstorm's put-back to the top,
    or London-mulligan bottoming — ``cat`` selects which menu), protecting
    the Doomsday combo pieces and threats. Return, in preference order: an
    excess land (while the hand still holds >= 3, the same threshold as the
    discard rule), then a cheap cantrip, then any other non-combo card, then
    free-draw cyclers (the cycle chain re-draws whatever sits on top, so they
    are the least-bad combo piece for the put-back case), then Lion's Eye
    Diamond and Doomsday, with Thassa's Oracle only as the forced last
    resort."""
    land_heavy = _hand_land_count(v) >= 3
    tiers: list[list[int]] = [[], [], [], [], [], []]
    for i, c in enumerate(v.cats):
        if c != cat:
            continue
        card = v.card(i)
        if card == _THASSAS_ORACLE:
            tiers[5].append(i)
        elif card in (_LED, _DOOMSDAY):
            tiers[4].append(i)
        elif card in _DD_FREE_DRAWS:
            tiers[3].append(i)
        elif card in _DD_CANTRIPS:
            tiers[1].append(i)
        elif _is_land_card(v, card):
            tiers[0 if land_heavy else 2].append(i)
        else:
            tiers[2].append(i)
    for tier in tiers:
        if tier:
            return tier[0]
    return 0


def _doomsday_pile_pick(v) -> int:
    """Choose a card during Doomsday pile building (TOP_LIBRARY queries).

    A TOP_LIBRARY menu whose pending source is Brainstorm is its put-back
    (cards from HAND going on top), not pile building — that goes to
    _dd_return_pick so combo pieces are kept over junk.

    Doomsday resolves in two TOP_LIBRARY phases:

      (a) SELECTION — choose WHICH 5 cards to keep, from the whole library and
          graveyard (many options). Composition only; the order is fixed by the
          rearrange that follows: keep Thassa's Oracle, then every free-draw
          enabler (Street Wraith / Edge of Autumn), then cheap cantrips, then
          whatever fills the rest.

      (b) REARRANGE — order the (<= 5) kept cards. The engine fills the pile
          bottom-first (effect_rearrange_top_of_library.cpp), so the
          position-from-top being placed equals the number of remaining choices.

    Two pile plans:

    STANDARD, top -> bottom: [free draws..., Thassa's Oracle, rest...] with
    Oracle's position ADAPTIVE — directly under however many free-draw enablers
    the pile actually has (the old plan hardcoded Oracle 3rd behind two Street
    Wraiths, which stalls a list running only one Wraith, e.g. wubg_doomsday).
    The kill: each drawn enabler is immediately cycled (the post-Doomsday
    cycling rules), chaining down to Oracle; natural draw steps and any cantrip
    filler drain the remainder until the library hits the Oracle-win size.
    When the board is short of blue sources (_dd_pile_wants_land) a blue land
    sits on the very top: [land, free draws..., Thassa's Oracle, rest...] —
    the first draw delivers it, the next land drop makes Edge of Autumn
    cyclable and the kill Wasteland-proof.

    BRAINSTORM SAME-TURN (bs_line: Brainstorm in hand, LED offered, and a
    free-draw cycler already in hand), top -> bottom:
    [free draws..., Lion's Eye Diamond, Thassa's Oracle, rest...].
    Brainstorm lifts the top three pile cards (cyclers + LED), puts back two
    junk cards, LED is cast for 0, and the in-hand cycler chain eats the
    put-backs down to Oracle: the final cycle's draw trigger is the LED crack
    window (_led_crack_choice) — sac LED for UUU with Oracle still in the
    library, draw Oracle, cast it off the floating mana the same turn. The
    extra cycler in HAND is load-bearing: Brainstorm's two put-backs cost two
    draws, so a pile-only chain always comes up exactly one draw short.
    """
    if v.pending == _BRAINSTORM:
        return _dd_return_pick(v, _CAT_TOP_LIBRARY)

    # The land plan applies only while Doomsday itself is resolving (its
    # selection and rearrange menus); a Ponder rearrange keeps lands as filler.
    want_land = v.pending == _DOOMSDAY and _dd_pile_wants_land(v)
    oracle_i, led_i, land_i, draw_is, cantrip_is, filler_is = \
        _classify_top_library(v, want_land)
    bs_line = (led_i is not None
               and _hand_has_card(v, _BRAINSTORM)
               and _hand_free_draw_count(v) >= 1)
    if not bs_line and led_i is not None:
        # Without the Brainstorm line LED is ordinary filler, as before.
        filler_is = sorted(filler_is + [led_i])
        led_i = None
    n_choices = sum(1 for c in v.cats if c == _CAT_TOP_LIBRARY)

    # (a) SELECTION: Oracle, LED (bs_line only), the blue land, free draws,
    # cantrips, filler.
    if n_choices > _DD_PILE_SIZE:
        if oracle_i is not None:
            return oracle_i
        if led_i is not None:
            return led_i
        if land_i is not None:
            return land_i
        if draw_is:
            return draw_is[0]
        if cantrip_is:
            return cantrip_is[0]
        return 0

    # Small rearrange with no combo pieces (Ponder / Brainstorm): legacy behaviour.
    if oracle_i is None and not draw_is and led_i is None and land_i is None:
        return 0

    # (b) REARRANGE, bottom-first: position-from-top == remaining choice count.
    # The blue land (if any) is the top card; Oracle goes right under the
    # (remaining) free-draw enablers — with LED slotted between them on the
    # bs_line — positions above get the enablers, positions below get
    # filler-then-cantrips (cantrips shallower than dead filler, since a drawn
    # one digs further).
    pos = n_choices
    land_off = 1 if land_i is not None else 0
    led_pos = (land_off + len(draw_is) + 1) if led_i is not None else 0
    oracle_slot = land_off + len(draw_is) + (2 if led_i is not None else 1)
    oracle_pos = oracle_slot if oracle_i is not None else 0
    # Deepest combo-piece position: below it the pile is bury-order. With
    # neither Oracle nor LED among the remaining picks it is 0, i.e. every
    # position buries (the legacy oracle-absent behaviour).
    if oracle_i is not None:
        combo_floor = oracle_slot
    elif led_i is not None:
        combo_floor = led_pos
    else:
        combo_floor = 0
    if pos == 1 and land_i is not None:
        return land_i
    if pos == led_pos and led_i is not None:
        return led_i
    if pos == oracle_pos and oracle_i is not None:
        return oracle_i
    if pos > combo_floor:                    # below Oracle: bury the worst first
        if filler_is:
            return filler_is[0]
        if cantrip_is:
            return cantrip_is[0]
        if draw_is:
            return draw_is[0]
        if oracle_i is not None:
            return oracle_i
        return land_i if land_i is not None else 0
    # Above Oracle/LED: the free-draw chain.
    if draw_is:
        return draw_is[0]
    if cantrip_is:
        return cantrip_is[0]
    if oracle_i is not None:
        return oracle_i
    return land_i if land_i is not None else 0


# ── Reanimation / lands-combo helpers (card gated, deck-agnostic) ───────────
# Don't dredge (skip the draw, mill 3) once the library is this small — the
# lands deck grinds long games and a draw/deck-out loss is never acceptable.
_LOAM_DREDGE_MIN_LIBRARY = 10


def _reanimation_target_choice(v) -> int | None:
    """Target pick for Reanimate / Animate Dead: the highest-mana-value creature
    card offered, tie-broken toward self-owned (our own fatty beats stealing an
    equal one). Fixes the generic target rule's prefer-opponent polarity, which
    is meant for removal and is exactly backwards for reanimation. Returns None
    when no TARGET action is offered."""
    best_i, best_key = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_TARGET:
            continue
        card = v.card(i)
        if card is None:
            continue  # "no target" slot — never preferred
        key = (_card_mana_value(v, card), 1 if v.ctrl(i) == 1 else 0)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def _keep_legend_choice(v) -> int | None:
    """Legend-rule keep pick (704.5j) for duplicate Dark Depths: keep the copy
    with the FEWEST 'other' counters — the freshly-cloned Thespian's Stage at
    zero ice counters — so its "no ice counters" trigger fires and makes Marit
    Lage. Keeping the 10-counter original fizzles the combo. The duplicates
    share one card, so they are told apart through the action's object.
    Other legend conflicts return None (generic behaviour)."""
    best_i, best_ct = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_KEEP_LEGEND or v.card(i) != _DARK_DEPTHS:
            continue
        p = v.ref_perm(i)
        ct = p.other_counters if p is not None else float("inf")  # unresolvable — never preferred
        if best_ct is None or ct < best_ct:
            best_i, best_ct = i, ct
    return best_i


# ── Menu-shape helpers shared by the GREEDY and HEURISTIC tiers ─────────────
# Mana-ability offers inside an optional-payment menu (tap a source toward an
# "unless" cost) carry the MANA_<color> categories.
_MANA_CATS = frozenset({"MANA_W", "MANA_U", "MANA_B", "MANA_R", "MANA_G", "MANA_C"})
_POOL_RED = 3   # the pool's colour index (W/U/B/R/G/C)
_POOL_BLUE = 1

_TOXIC_DELUGE = "Toxic Deluge"
_CHAIN_LIGHTNING = "Chain Lightning"

# Mana an "unless [cost]" payment needs, keyed by the pending-decision source
# that imposes it (the menu shows the source, not the cost). Unlisted
# sources (The Tabernacle at Pendrell Vale's upkeep tax on each creature, ...)
# default to {1}. Chain Lightning's copy costs {R}{R}.
_UNLESS_MANA_COST = {
    "Daze": 1,
    "Spell Pierce": 2,
    _CHAIN_LIGHTNING: 2,
}


def _opp_max_toughness(v) -> int:
    """Highest toughness among the opponent's unphased creatures (0 if none)."""
    return max((p.toughness for p in v.opp_perms if p.creature), default=0)


def _x_ladder_choice(v) -> int:
    """An X-value ladder (CHOOSE_X with no card; index == X): the
    maximum X, the biggest effect the mana buys (Green Sun's Zenith fetches
    the biggest creature it can instead of the X=0 Dryad Arbor). Toxic
    Deluge's X is paid in LIFE and shrinks our creatures too: exactly enough
    to kill the opponent's toughest creature, never our whole life total."""
    if v.pending == _TOXIC_DELUGE:
        want = min(_opp_max_toughness(v), v.life - 1)
        return max(0, min(v.n - 1, want))
    return v.n - 1


def _pay_unless_choice(v) -> int:
    """An "unless [cost]" payment we are asked to make (Daze / Spell Pierce on
    our spell, The Tabernacle at Pendrell Vale's upkeep tax, Static Prison's
    energy, the Chain Lightning copy offer). The menu offers "Don't pay"
    (ordinal 0) plus the mana abilities that can fund the payment; "Pay"
    (ordinal 1) appears once the pool covers the cost. Paying keeps our spell
    or permanent, so we pay whenever we can: take Pay when offered, else tap
    one more source toward it — but only when the untapped sources on the
    menu plus the pool can reach the cost (_UNLESS_MANA_COST), so a payment
    that can never complete doesn't tap every land for nothing."""
    pay_i = decline_i = None
    mana: list[int] = []
    cats = v.cats
    for i in range(v.n):
        c = cats[i]
        if c == _CAT_PAY_UNLESS:
            if v.ordinal(i) == 1:
                pay_i = i
            else:
                decline_i = i
        elif c in _MANA_CATS:
            mana.append(i)
    if pay_i is not None:
        return pay_i
    pending = v.pending
    need = _UNLESS_MANA_COST.get(pending, 1)
    if pending == _CHAIN_LIGHTNING:
        mana = [i for i in mana if cats[i] == _CAT_MANA_R]
        have = v.pool(_POOL_RED)
    else:
        have = v.pool_total
    sources = {v.ref(i) for i in mana}
    if mana and have + len(sources) >= need:
        return mana[0]
    return decline_i if decline_i is not None else 0


def _resolution_cast_ok(v, card) -> bool:
    """Should we take an offer to cast ``card`` right now, during a resolution
    (Rift Bolt's suspend cast, a miracle cast, Amped Raptor's energy cast)?
    The offer is then-or-never and the engine already checked it is castable
    with legal targets, so take it — unless the only targets are our own:
    targeted removal with no opposing creature would hit our own creature,
    and a counterspell with no opposing spell on the stack would counter ours."""
    if card in _TARGETED_REMOVAL and not _opponent_has_creature(v):
        return False
    if card in _COUNTER_SPELLS and not _opponent_has_spell_on_stack(v):
        return False
    return True


def _is_yesno_menu(v) -> bool:
    return v.n == 2 and all(c == _CAT_YESNO for c in v.cats)


def _is_miracle_reveal(v) -> bool:
    """The miracle reveal prompt (CR 702.94a): a yes/no without cards whose
    pending source is a miracle card (the card just drawn, still in hand).
    Accepting only reveals it — the cast itself is offered afterwards by the
    miracle trigger, as a then-or-never cast offer."""
    if v.card(1) is not None:
        return False
    return v.pending is not None and v.cards.has_miracle(v.pending)


# ── GREEDY/HEURISTIC fruitless-activation stall guard ───────────────────────
# Action categories a resolution-time zone search emits (src/components/
# ability.cpp search_zone): SEARCH_LIBRARY for library searches, TOP_LIBRARY for
# search-to-top-of-library, CHOOSE_CARD for non-library picks (e.g. Aether
# Vial's put-a-creature-from-hand). When such a menu offers ONLY the
# "Fail to find" slot (a single choice with no card — the engine builds it from
# Entity(0), see _EXPLORE_SEARCH_FAIL_WEIGHT), the search cannot succeed. If the
# resolving item on top of the stack is a self-controlled ability, re-activating
# its source card right now would just fail again, so the hard tier records the
# fail against the card and skips its ACTIVATE for the rest of the TURN —
# not the game, because eligibility can change in ways a fail can't tell
# (Aether Vial's search keys on its charge-counter count: a fail at 0 counters
# says nothing about next turn's 1). After _FRUITLESS_ACTIVATION_LIMIT failed
# tries the card is dropped for the game: hard-scripted bw_dnt once activated
# Vial into a guaranteed fail-to-find every turn (~60 activations) to the
# decision cap — the one non-decisive game of a 441-game fuzz campaign — so
# retries must be bounded.
_SEARCH_MENU_CATS = frozenset((_CAT_SEARCH, _CAT_TOP_LIBRARY, _CAT_CHOOSE_CARD))
_FRUITLESS_ACTIVATION_LIMIT = 3


def _greedy_action(v,
                   fruitless: set | None = None,
                   hold_casts: frozenset | set | None = None,
                   hold_activations: frozenset | set | None = None,
                   doomsday: bool | None = None,
                   hold_idxs: set[int] | None = None) -> int:
    """
    Rule-based agent for test_minimal.dk (blue/red fetch-land deck).
    Works correctly for either Player A or Player B because the view
    is always the deciding player's perspective.

      - Never blocks (confirms immediately)
      - Attacks with every eligible creature each combat
      - Selects target 0 (opponent player or first offered spell/permanent)
      - Searches library: always finds the first offered card (never fails to find)
      - Casts every spell the moment it becomes affordable
      - Plays the first available land
      - Activates non-mana abilities (fetch lands, Wasteland destroy) during main phase
      - Taps mana during main phases, preferring the color the hand needs most
        (blue for Flying Men, Delver, Ponder, Daze, Counterspell, Air Elemental;
         red for Dragon's Rage Channeler, Lightning Strike)
      - Passes priority otherwise
    """
    cats = v.cats
    num_choices = v.n

    in_main_phase = v.in_main
    own_main = in_main_phase and v.is_active
    # Are we piloting Doomsday? The caller's deck-name identification when it
    # has one (``doomsday``); the hand-content heuristic only for unnamed
    # decks — that heuristic also fires on Thoughtseize / Dark Ritual / Lotus
    # Petal hands (reanimator, bug, wrb_energy), which must NOT inherit the
    # Doomsday fetch and pile rules.
    dd_deck = _is_doomsday_deck(v) if doomsday is None else doomsday
    # Doomsday state shared by the rules below. Once the five-card pile is laid
    # every library touch is part of the kill: the whole chain (cycles, the
    # LED crack, post-pile cantrips) runs in OUR MAIN PHASE only — Oracle is a
    # creature, so mana floated at the draw step is gone before it can be
    # cast — and the generic library-touching rules (surveil, activations,
    # fetches) stand down. With Oracle already in hand at the win size every
    # further draw is pointless and one closer to decking, so digging stops
    # and the main-phase cast rule wins.
    library = v.library
    dd_pile_laid = library <= _DD_PILE_SIZE and dd_deck
    oracle_kill_ready = (_hand_has_card(v, _THASSAS_ORACLE)
                         and library <= _ORACLE_WIN_LIBRARY)

    # 0a. Sideboarding: the scripted agent never sideboards, so it takes Done as
    # soon as the deck is balanced (Done is index 0 whenever it is offered). On the
    # IN-FIRST menu that is every decision it can reach, since it never opens a
    # swap. The fallback matters only if it inherits a phase mid-swap (Done is
    # withheld while a cut is outstanding): pick the first offered move, which is
    # necessarily the balancing cut, so the phase still terminates instead of
    # falling through to the combat/casting logic below with a sideboard menu in
    # hand.
    if any(c == _CAT_SB_DONE for c in cats):
        return 0
    if any(c in (_CAT_SB_IN, _CAT_SB_OUT) for c in cats):
        return next(i for i, c in enumerate(cats) if c in (_CAT_SB_IN, _CAT_SB_OUT))

    # 0. Mulligan: always keep — return the first non-mulligan action (the keep action)
    if any(c == _CAT_MULLIGAN for c in cats):
        for i, c in enumerate(cats):
            if c != _CAT_MULLIGAN:
                return i

    # 1. Confirm blockers immediately — never block
    for i, c in enumerate(cats):
        if c == _CAT_CONF_BLK:
            return i

    # 2. Attacker selection: select until all eligible creatures are attacking, then confirm.
    #    The game re-offers already-attacking creatures as SEL_ATK (for deselection), so we
    #    must check the battlefield state rather than blindly picking SEL_ATK every time.
    if any(c == _CAT_SEL_ATK for c in cats):
        if _all_eligible_creatures_attacking(v):
            for i, c in enumerate(cats):
                if c == _CAT_CONF_ATK:
                    return i
        else:
            for i, c in enumerate(cats):
                if c == _CAT_SEL_ATK:
                    return i

    # 3. Confirm attack declaration (fallback, e.g. no eligible attackers)
    for i, c in enumerate(cats):
        if c == _CAT_CONF_ATK:
            return i

    # 4. Select target — prefer non-self-controlled targets (ctrl 0 = an
    #    opponent's permanent/spell, -1 = a player target, also non-self).
    #    The C++ game sorts targets opponent-first so action 0 is usually correct,
    #    but guard against accidentally targeting own spells/permanents.
    for i, c in enumerate(cats):
        if c == _CAT_TARGET and v.ctrl(i) != 1:
            return i
    # Fallback: all targets are self-controlled — return first (shouldn't happen in practice).
    for i, c in enumerate(cats):
        if c == _CAT_TARGET:
            return i

    # 5. Search library — action 0 = fail to find; action 1+ = actual cards.
    #    For Doomsday pile building: prefer Thassa's Oracle, then draw spells.
    #    For Personal Tutor: prefer Doomsday.
    #    For fetch lands: pick action 1 (first land found).
    if any(c == _CAT_SEARCH for c in cats):
        # Prefer Thassa's Oracle when searching (Doomsday pile building)
        for i, c in enumerate(cats):
            if c == _CAT_SEARCH and v.card(i) == _THASSAS_ORACLE:
                return i
        # Prefer Doomsday when searching (Personal Tutor)
        for i, c in enumerate(cats):
            if c == _CAT_SEARCH and v.card(i) == _DOOMSDAY:
                return i
        # Doomsday deck land search: fetch the colours the combo needs
        # (see _dd_fetch_pick) instead of the first land offered.
        if dd_deck:
            pick = _dd_fetch_pick(v)
            if pick is not None:
                return pick
        return 1 if num_choices > 1 else 0

    # 5b. Paying costs (tapping lands for mana during spell/ability payment,
    #     or the pitch pick of an alternative cost such as Force of Will's
    #     "exile a blue card"). First available option, except that a
    #     Doomsday deck pays with lands before Petal / LED and pitches its
    #     combo pieces last (Thassa's Oracle never while anything else is
    #     offered).
    if any(c == _CAT_PAYING for c in cats):
        if dd_deck:
            best, best_key = 0, None
            for i, c in enumerate(cats):
                if c != _CAT_PAYING:
                    continue
                card = v.card(i)
                key = (card == _THASSAS_ORACLE,
                       card in _DD_COMBO_PIECES,
                       card in _DD_PAY_LAST,
                       card in _DD_CANTRIPS)
                if best_key is None or key < best_key:
                    best, best_key = i, key
            return best
        return 0

    # 5c. Delve exile count (a CHOOSE_X menu whose actions carry the delve spell's
    #     card; a plain X-cost ladder carries none). Exile the maximum — the
    #     cheapest cast, matching the old auto-payer (and for Murktide Regent also the
    #     biggest body). Every offered count is payable (the engine constrains the menu).
    if all(c == _CAT_CHOOSE_X for c in cats) and v.nonnull_card(0):
        return num_choices - 1

    # 5c'. X-value ladder (CHOOSE_X with no card): see _x_ladder_choice.
    if num_choices >= 1 and all(c == _CAT_CHOOSE_X for c in cats):
        return _x_ladder_choice(v)

    # 5c''. An "unless [cost]" payment (see _pay_unless_choice).
    if any(c == _CAT_PAY_UNLESS for c in cats):
        return _pay_unless_choice(v)

    # 5c'''. Sylvan Library: the card pick takes the first card offered; the
    #        pay-4-life-or-put-back prompt (no cards) puts the card back on
    #        top (ordinal 0) — never 4 life a card.
    if all(c == _CAT_SYLVAN for c in cats):
        if not v.nonnull_card(0):
            for i in range(num_choices):
                if v.ordinal(i) == 0:
                    return i
        return 0

    # 5d. Zone-change card pick (CHOOSE_CARD). An optional search-style pick (Aether
    #     Vial's put-a-creature-from-hand) leads with a card-less "Fail to find" slot —
    #     take the first REAL card instead, so Vial actually deploys a creature when
    #     one is eligible (declining was strictly worse: the guard below never
    #     blacklists a menu that has a real pick, so the agent just failed forever).
    #     Delve per-card exile picks are all-real menus, so they still take slot 0
    #     (the first pick may eat a card a smarter agent would keep, or skip an
    #     instant/sorcery Murktide would count — kept simple at this tier).
    if any(c == _CAT_CHOOSE_CARD for c in cats):
        # A surveil menu (keep on top / put in graveyard) while piloting
        # Doomsday keeps a combo piece — Consider and Undercity Sewers
        # surveils binned two Doomsdays and a Dark Ritual in one 30-game
        # sample — and keeps EVERY card once the pile is laid (a Sewers
        # surveil once binned Thassa's Oracle off the top of the pile).
        if dd_deck and num_choices == 2 and any(c == _CAT_TOP_LIBRARY for c in cats):
            seen = {v.card(i) for i in range(num_choices)}
            if dd_pile_laid or seen & _DD_COMBO_PIECES:
                for i, c in enumerate(cats):
                    if c == _CAT_TOP_LIBRARY:
                        return i
        for i, c in enumerate(cats):
            if c == _CAT_CHOOSE_CARD and v.card(i) is not None:
                return i
        return 0

    # 5e. Optional "may" trigger (OPTIONAL_YESNO, 0=Decline / 1=Accept). The tier's
    #     default is to decline (the final fallback returns 0), but Aether Vial's
    #     upkeep charge-counter trigger must be ACCEPTED or the Vial sits at 0
    #     counters forever and never deploys a creature. The resolving trigger is on
    #     top of the stack while the yes/no is asked, so the source is identifiable.
    if num_choices == 2 and all(c == _CAT_YESNO for c in cats):
        top = v.stack_top
        if top is not None and top.mine and not top.spell and top.card == _AETHER_VIAL:
            return 1

    # 5f. An offer to cast a card right now (OPTIONAL_YESNO whose accept option names
    #     the card): suspend's free cast when the last time counter comes off, a
    #     miracle cast, or Amped Raptor's energy cast (see _resolution_cast_ok).
    #     The miracle reveal before it (no card on accept) is accepted too —
    #     revealing only opens the cast offer.
    if _is_yesno_menu(v):
        card = v.card(1)
        if card is not None:
            return 1 if _resolution_cast_ok(v, card) else 0
        if _is_miracle_reveal(v):
            return 1

    # 6. Cast spells.
    #    Counter spells (Counterspell, Daze, Force of Will) require an opponent's spell
    #    on the stack; skip them when the stack holds only own spells or is empty.
    opponent_spell_on_stack = _opponent_has_spell_on_stack(v)

    # Priority: if opponent has a spell on stack and we have UU, cast Counterspell first
    # (unless the caller's triage put it on hold — a cantrip not worth countering).
    if (opponent_spell_on_stack and v.pool(_POOL_BLUE) >= 2
            and not (hold_casts and _COUNTERSPELL in hold_casts)):
        for i, c in enumerate(cats):
            if c == _CAT_CAST and v.card(i) == _COUNTERSPELL:
                return i

    # Crack Lion's Eye Diamond in the Doomsday kill window — a self draw is resolving,
    # Oracle is the known top card, and library <= 2 — to float UUU for the Thassa's
    # Oracle win. LED is offered as instant-speed MANA_* actions at priority (unlike
    # ordinary mana sources, which stay hidden and auto-pay), so it is picked here.
    if own_main:
        led = _led_crack_choice(v)
        if led is not None:
            return led

    # --- Doomsday deck rules ---
    has_doomsday_in_hand = _hand_has_card(v, _DOOMSDAY)

    # Win the game: cast Thassa's Oracle as soon as the library is small enough
    # that its devotion-to-blue (>= 2 from its own UU) meets/exceeds the library
    # size. "Mana is there" == the engine is offering Oracle as a legal cast;
    # when it isn't, a Lotus Petal cracked for blue can make it so.
    if library <= _ORACLE_WIN_LIBRARY:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and v.card(i) == _THASSAS_ORACLE:
                return i
        petal = _dd_oracle_mana_choice(v, own_main)
        if petal is not None:
            return petal

    # Cast Doomsday once Thassa's Oracle's UU will exist by kill time
    # (_dd_blue_for_oracle) and the opponent's board can't kill us at half
    # life before the kill lands (_dd_clock_allows_doomsday). Firing it
    # earlier just empties our library into a deck-out or halves us into
    # lethal, so we hold it until the mana to win is reachable and the clock
    # allows.
    # Never a SECOND Doomsday once the pile is laid: it re-piles the same
    # five cards for another half of our life (one game went 6 -> 2 -> 1
    # with Oracle in hand and the kill one cycle away).
    dd_ready = (_dd_blue_for_oracle(v) and _dd_clock_allows_doomsday(v)
                and not dd_pile_laid)
    for i, c in enumerate(cats):
        if c == _CAT_CAST and v.card(i) == _DOOMSDAY:
            if dd_ready:
                return i
            # else hold Doomsday until the blue mana to cast Oracle is in play

    # Only cast Dark Ritual if Doomsday is also in hand and we're ready to combo
    # (don't waste ritual mana without the payoff online). Main phase of our
    # own turn only: the mana pool empties at step boundaries, so a ritual cast
    # at upkeep (its first legal window — it's an instant) strands the BBB
    # before the sorcery-speed Doomsday can ever be cast with it.
    for i, c in enumerate(cats):
        if c == _CAT_CAST and v.card(i) == _DARK_RITUAL:
            if has_doomsday_in_hand and dd_ready and own_main:
                return i
            # else skip it

    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            card = v.card(i)
            if hold_casts and card in hold_casts:
                continue  # caller-computed hold (e.g. Reanimate with no fatty
                          # in a graveyard yet — see _heuristic_action)
            if card in _COUNTER_SPELLS and not opponent_spell_on_stack:
                continue
            if card == _DARK_RITUAL:
                continue  # handled above
            if card == _DOOMSDAY:
                continue  # handled above (held until blue mana for Oracle is online)
            if card == _THASSAS_ORACLE:
                continue  # only cast Oracle as the wincon (handled above)
            if card in _DD_FREE_DRAWS and dd_pile_laid:
                continue  # post-pile the cyclers are draws, never casts: a
                          # hard-cast Edge of Autumn searches the library and
                          # shuffles the pile away (the cycle rules below)
            if (dd_deck and has_doomsday_in_hand
                    and card not in _DD_PETAL_WORTHY
                    and _count_controlled(v, _LOTUS_PETAL) > 0
                    and _count_untapped_mana_sources(v) < _card_mana_value(v, card)):
                continue  # keep the Petals for Doomsday: a spell the lands
                          # alone can't pay would crack one
            if card == _BRAINSTORM and (
                    _self_stack_has_card(v, _DOOMSDAY)
                    or (has_doomsday_in_hand
                        and library > _DD_PILE_SIZE
                        and _dd_combo_ready(v))):
                continue  # hold Brainstorm: post-Doomsday it lifts the top of
                          # the pile (the same-turn kill's draw engine). The
                          # stack check matters — casting it in response to our
                          # own Doomsday resolves it FIRST, drawing from the
                          # un-piled library. The hold applies only while the
                          # combo is actually startable (black + UU on board,
                          # pile not yet laid): a mana-screwed hand digs with
                          # Brainstorm instead of sitting on it, and a spare
                          # Doomsday in hand after the pile is laid must not
                          # freeze the draw engine.
            if card in _DD_CANTRIPS and (
                    library < (_DD_CANTRIP_MIN_LIBRARY[card] if dd_deck else 3)
                    or _controls_card(v, _THASSAS_ORACLE)):
                continue  # never deck ourselves with a cantrip: drawing from
                          # an empty library is a loss, and with Oracle already
                          # on our battlefield its win trigger is pending — a
                          # greedy Brainstorm here resolves FIRST and kills us
                          # in response to our own winning trigger
            if (card in _DD_CANTRIPS and dd_pile_laid
                    and (not own_main or oracle_kill_ready
                         or (_count_untapped_sources(v, _BLUE_SOURCES) - 1
                             < _BLUE_SOURCES_NEEDED))):
                continue  # post-Doomsday a cantrip is a kill-chain step: our
                          # main phase only, never once Oracle is in hand at
                          # the win size (a draw-step Consider once milled the
                          # last card with Oracle already in hand), and never
                          # tapping the UU that casts Oracle this turn (two
                          # upkeep Brainstorms off both Underground Seas once
                          # left Oracle in hand with no blue mana)
            if (card == _GREEN_SUNS_ZENITH
                    and _count_untapped_mana_sources(v) < _GSZ_MIN_MANA_SOURCES):
                continue  # hold GSZ until it can pay X>=1 (avoids the X=0 Dryad Arbor cast)
            if card == _SOLITUDE and not _opponent_has_creature(v):
                continue  # Solitude's value is its ETB exile — don't hard-cast it
                          # as a vanilla 3/2 when there's no opposing creature to hit
            return i

    # 7. Play land (may unlock mana for a spell next query)
    for i, c in enumerate(cats):
        if c == _CAT_LAND:
            return i

    # 8. Activate non-mana abilities:
    #    - Cycling (Street Wraith, Edge of Autumn): cycle as part of the combo
    #    - Fetch lands: tap + pay 1 life + sacrifice → search for land
    #    - Wasteland: tap + sacrifice → destroy target nonbasic land

    # Cycle Street Wraith (pay 2 life, draw) only once we're going off: a small
    # library means we are drawing through the Doomsday pile toward Thassa's
    # Oracle. Cycling earlier would dump the Wraiths we need for the free-draw
    # chain and pull Oracle out of the pile, breaking the combo. Never pay the
    # 2 life below 3 life, so the dig can't kill us before Oracle resolves.
    # Only cycle on an empty stack: stacked cycles resolve LIFO, scrambling the
    # pile's draw order, and an unresolved LED spell must hit the battlefield
    # before the final cycle's draw opens its crack window.
    if (1 <= library <= _DD_PILE_SIZE
            and v.life >= _STREET_WRAITH_MIN_LIFE
            and v.stack_empty
            and own_main and not oracle_kill_ready):
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and v.card(i) == _STREET_WRAITH:
                return i

    # Cycle Edge of Autumn (sac a land, draw a card) only once we're going off,
    # like Street Wraith: pre-Doomsday every land is needed for BBB + UU. Only
    # on an empty stack (its draw is the LED crack window on the same-turn
    # line, so LED must have resolved) and never off an empty library (that
    # draw would be the game loss). Only when some land can go without
    # breaking Oracle's UU (_dd_edge_cycle_safe; _dd_sacrifice_pick then
    # chooses it) — otherwise the natural draw step reaches the next pile
    # card a turn later with the kill mana intact.
    if (1 <= library <= _DD_PILE_SIZE
            and own_main and not oracle_kill_ready
            and _dd_edge_cycle_safe(v)):
        for i, c in enumerate(cats):
            if (c == _CAT_ACTIVATE and v.card(i) == _EDGE_OF_AUTUMN
                    and v.stack_empty):
                return i

    # Always activate The One Ring's {T} draw ability whenever able — each use draws
    # a card per burden counter (pure card advantage), and it only taps once per turn.
    # Ungated by phase so it still fires on turns the agent has nothing else to do.
    # Never off an empty library: that draw is the game loss (with Thassa's
    # Oracle in hand the win at library 0 is handled above — just don't draw).
    library_empty = library == 0
    if not library_empty:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and v.card(i) == _THE_ONE_RING:
                return i

    if in_main_phase:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE:
                card = v.card(i)
                if (library_empty or dd_pile_laid) and card != _WASTELAND:
                    continue  # a draw-shaped activation (Clue, planeswalker
                              # card advantage) off an empty library loses on
                              # the spot; after Doomsday has laid the pile a
                              # fetch would shuffle it away and a Clue crack
                              # once tapped both blue lands the turn Oracle
                              # was due — the kill chain above is the only
                              # thing that touches the library or the mana
                if card == _WASTELAND and not _opponent_has_nonbasic_land(v):
                    continue  # no opponent nonbasic land to target — skip
                if card == _STREET_WRAITH:
                    continue  # only ever cycled via the gated combo rule above
                if card == _EDGE_OF_AUTUMN:
                    continue  # likewise: only via its stack-empty rule above,
                              # so its draw can serve as the LED crack window
                if card == _KEEN_EYED_CURATOR and not v.stack_empty:
                    continue  # don't pile copies onto the stack (each extra one
                              # fizzles when its graveyard target is already exiled)
                if card == _CANDELABRA and _tapped_multi_mana_lands(v) == 0:
                    continue  # nothing worth untapping — don't tap Candelabra for X=0
                if hold_activations and card in hold_activations:
                    continue  # caller-computed hold (e.g. Thespian's Stage with
                              # no Dark Depths to copy — see _heuristic_action)
                if hold_idxs and i in hold_idxs:
                    continue  # caller-computed per-option hold (an attached
                              # Equipment's equip, Karakas with no opposing
                              # legend, ... — see _activation_holds)
                if fruitless and card in fruitless:
                    continue  # stall guard: this card's search found NOTHING
                              # earlier this game (see _SEARCH_MENU_CATS) —
                              # re-activating it would just fail to find again
                return i

    # 9. (Removed - mana abilities no longer offered during normal priority in machine mode;
    #     mana is tapped automatically during cost payment via PAYING_COSTS.)

    # 10. Doomsday pile building (TOP_LIBRARY): select the 5 pile cards, then
    #     order them [free draws..., (LED,) Thassa's Oracle, rest...] for the
    #     cycle-chain kill; a Brainstorm put-back menu instead protects combo
    #     pieces. See _doomsday_pile_pick for both pile plans.
    if any(c == _CAT_TOP_LIBRARY for c in cats):
        return _doomsday_pile_pick(v)

    # 10b. Dig choice (Once Upon a Time): pick action 1 (first matching card) if available.
    if any(c == _CAT_DIG for c in cats):
        return 1 if num_choices > 1 else 0

    # 11. Other (uncategorized) choice: pick randomly.
    other_idxs = [i for i, c in enumerate(cats) if c == _CAT_OTHER]
    if other_idxs:
        return random.choice(other_idxs)

    # 12. Surveil / Delver reveal: both use two PASS_PRIORITY (0) choices whose
    #     source is the same library card.
    #     For surveil: action 1 = put in graveyard (good for delirium and Murktide).
    #     For Delver's PeekAndReveal: action 1 = reveal (safe; triggers transform
    #     only if the top card is an instant or sorcery, which is desirable).
    if num_choices == 2 and all(c == _CAT_PASS for c in cats[:2]):
        if v.nonnull_card(0) and v.same_card(0, 1):
            return 1

    # Default: pass priority
    return 0


# ── HEURISTIC helpers (single-ply; combat sim is the only exact lookahead) ──
# These read the view's permanents: a creature is a permanent with ``creature`` set.

# Opponent life at or below which a burn spell should go to the face.
_BURN_FACE_LIFE = 3


def _creatures(perms: list) -> list:
    """Creature permanents from a battlefield list."""
    return [p for p in perms if p.creature]


def _should_attack(ap: int, at: int, opp_blockers: list, alpha: bool) -> bool:
    """Single-ply attack decision for a vanilla attacker (P/T only).

    Attack unless an untapped opposing blocker would kill us while surviving
    (a pure loss).  Even trades and free hits are allowed.  ``alpha`` forces the
    attack when the team is lethal this turn.
    """
    if alpha:
        return True
    if not opp_blockers:
        return True
    for b in opp_blockers:
        if b.power >= at and b.toughness > ap:
            return False  # we die, the blocker lives — don't attack into it
    return True


def _passes(p, exclude_attacking=False, exclude_blocking=False, require_untapped=False,
            require_unsick=False) -> bool:
    """Whether permanent ``p`` passes the _lookup_pt filters."""
    if require_untapped and p.tapped:
        return False
    if require_unsick and p.sick:
        return False
    if exclude_attacking and p.attacking:
        return False
    if exclude_blocking and p.blocking:
        return False
    return True


def _lookup_pt(perms: list, card, **filters):
    """First creature named `card` passing the filters → (power, toughness)."""
    for p in perms:
        if p.card != card or not p.creature:
            continue
        if not _passes(p, **filters):
            continue
        return p.power, p.toughness
    return None


# _action_pt's answer for an action whose own object fails the filters.
_FILTERED = "filtered"


def _action_pt(v, i, perms, **filters):
    """(power, toughness) of the creature action ``i`` names: on a view with exact
    references its own object (``_FILTERED`` when that object fails the filters, None when it
    is no creature); else the first creature with its card that passes them (_lookup_pt)."""
    if v.exact_refs:
        p = v.ref_perm(i)
        if p is None or not p.creature:
            return None
        return (p.power, p.toughness) if _passes(p, **filters) else _FILTERED
    return _lookup_pt(perms, v.card(i), **filters)


def _best_opp_creature_target(v) -> int | None:
    """Among SELECT_TARGET actions, the opponent creature with the highest power.

    Opponent-controlled object targets are the ones with ctrl 0 (self is 1, a player /
    "no target" slot -1). Returns None if no opp creature is offered, so the caller
    can fall back to its usual pick.
    """
    best_i, best_pow = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_TARGET:
            continue
        if v.ctrl(i) != 0:
            continue  # self-controlled, or a player / "no target" slot
        pt = _action_pt(v, i, v.opp_perms)
        power = pt[0] if pt and pt is not _FILTERED else 0
        if best_pow is None or power > best_pow:
            best_i, best_pow = i, power
    return best_i


def _wrath_energy_plan(v) -> tuple[int, int, int]:
    """Best energy amount Y for Wrath of the Skies, with its board impact.

    Wrath destroys EACH artifact, creature, and enchantment with MV <= Y on
    BOTH sides (lands and planeswalkers are immune). Evaluate every candidate
    Y — the distinct MVs of the opponent's destroyable permanents, plus 0 —
    and pick the one maximizing (opponent permanents destroyed - own
    permanents destroyed), tie-broken by destroyed-power differential and
    then by the SMALLEST Y (cheaper X, and no collateral own losses beyond
    what the score already accepted). Tokens have mana value 0.

    Returns (best_y, count_net, power_net) so the caster can also gate on
    whether the sweep is worth it at all.
    """
    def destroyables(perms):
        out = []
        for p in perms:
            if p.land or p.loyalty > 0:
                continue
            out.append((int(p.mana_value), p.power))
        return out
    opp = destroyables(v.opp_perms)
    own = destroyables(v.my_perms)
    best_key, best = None, (0, 0, 0)
    for y in sorted({mv for mv, _ in opp} | {0}):
        count_net = (sum(1 for mv, _ in opp if mv <= y)
                     - sum(1 for mv, _ in own if mv <= y))
        power_net = (sum(pw for mv, pw in opp if mv <= y)
                     - sum(pw for mv, pw in own if mv <= y))
        key = (count_net, power_net, -y)
        if best_key is None or key > best_key:
            best_key, best = key, (y, count_net, power_net)
    return best


def _desired_block_count(v, attackers: list) -> int:
    """How many blockers to commit this combat (survival need vs. free-kill value)."""
    powers = sorted((a.power for a in attackers), reverse=True)
    my_life = v.life
    # Survival: fewest blocks (removing the biggest attackers) to drop incoming
    # damage below lethal.  We can't see which attackers are already blocked, so we
    # assume committed blockers absorbed the biggest ones (consistent each query).
    survival = 0
    while survival < len(powers) and sum(powers[survival:]) >= my_life:
        survival += 1
    # Value: free-kill blocks (a blocker that kills the attacker and survives).
    my_blockers = [p for p in _creatures(v.my_perms) if not p.tapped]
    used, value = set(), 0
    for a in sorted(attackers, key=lambda x: -x.power):
        for j, b in enumerate(my_blockers):
            if j in used:
                continue
            if b.power >= a.toughness and b.toughness > a.power:
                used.add(j)
                value += 1
                break
    return min(len(attackers), max(survival, value))


# ── Card desirability (scry / surveil / Ponder / put-backs / bottoming) ─────
# Cards whose alternative cost (pitch a card, return an Island, sacrifice
# Mountains, evoke) makes them castable for next to no mana.
_CHEAP_ALT_COST = frozenset({"Force of Will", "Force of Negation", "Daze", "Solitude",
                             "Fireblast"})


def _effective_mana_value(v, card) -> float:
    """The mana a card really costs to cast: a card with a cheap alternative
    cost (_CHEAP_ALT_COST) counts as 1, a delve spell (Murktide Regent)
    as half its printed mana value."""
    mv = _card_mana_value(v, card)
    if card in _CHEAP_ALT_COST:
        return min(mv, 1.0)
    if card is not None and v.cards.is_delve(card):
        return mv / 2
    return mv


def _card_want(v, card, lands_total: int) -> float:
    """How much we want to draw ``card`` soon (> 0 wanted, < 0 not), given
    ``lands_total`` lands in play plus in hand. Lands are wanted while the
    mana base is short (under 3, then less so up to 4) and dead weight past
    5; a spell is wanted
    when castable within a land drop, cheaper spells a little less (the
    bigger payoff first), and a card too expensive for now is unwanted.
    Countermagic (pitch / alternative costs) and delve spells count as cheap."""
    if card is None:
        return 0.0
    if _is_land_card(v, card):
        if lands_total < 3:
            return 2.0
        if lands_total < 4:
            return 1.0
        return 0.5 if lands_total < 5 else -1.0
    mv = _effective_mana_value(v, card)
    if mv > lands_total + 1:
        return -0.5
    return 1.0 + 0.1 * mv


def _lands_total(v) -> int:
    return _self_land_count(v) + _hand_land_count(v)


def _look_split_pick(v) -> int:
    """Scry / surveil (CR 701.22 / 701.25): each remaining card is offered
    twice — "on top" (TOP_LIBRARY; the next kept card goes under the ones
    already kept) and the other pile (BOTTOM_DECK_CARD for scry, CHOOSE_CARD
    = graveyard for surveil). Send the least wanted card away first while
    one falls short of _LOOK_KEEP_WANT, then keep the rest best-first, so
    the card we most want is drawn next. Surveil also bins reanimation
    targets — the graveyard is where Reanimate wants them."""
    keep: dict = {}
    away: dict = {}
    surveil = False
    for i in range(v.n):
        card = v.card(i)
        if v.cats[i] == _CAT_TOP_LIBRARY:
            keep.setdefault(card, i)
        elif v.cats[i] in (_CAT_CHOOSE_CARD, _CAT_BOTTOM_CARD):
            away.setdefault(card, i)
            surveil = v.cats[i] == _CAT_CHOOSE_CARD
    lands = _lands_total(v)

    def want(card):
        if surveil and card in _REANIMATION_FATTIES:
            return -2.0
        return _card_want(v, card, lands)
    if not keep:
        return 0
    worst = min(keep, key=want)
    if want(worst) < _LOOK_KEEP_WANT and worst in away:
        return away[worst]
    return keep[max(keep, key=want)]


# Least _card_want a scried / surveilled card needs to stay on top: a
# castable spell (>= 1.0) or a land while the mana base is short (>= 1.0)
# beats a random draw; a spare land (0.5) or a dead card doesn't.
_LOOK_KEEP_WANT = 0.75


def _is_look_split_menu(v) -> bool:
    return (any(c == _CAT_TOP_LIBRARY for c in v.cats)
            and any(c in (_CAT_CHOOSE_CARD, _CAT_BOTTOM_CARD) for c in v.cats))


def _least_wanted_pick(v, cat: str, lands_total: int) -> int:
    """The least wanted card among the ``cat`` actions: the one to put back
    (Brainstorm), bottom (London mulligan), or place deepest (Ponder's
    rearrange fills the deepest slot first, so the best card ends on top)."""
    best_i, best_w = 0, None
    for i, c in enumerate(v.cats):
        if c != cat:
            continue
        w = _card_want(v, v.card(i), lands_total)
        if best_w is None or w < best_w:
            best_i, best_w = i, w
    return best_i


def _shuffle_pick(v) -> int:
    """Ponder's "you may shuffle": shuffle away a top three that holds
    nothing we want (none with _card_want >= 0.5), else keep the arranged
    top. A position whose card is not known exactly counts by the best card
    it can still be (``top_candidates``), so a top the viewer knows only as
    candidates is kept when any of them is wanted."""
    lands = _lands_total(v)
    wants = []
    for cands in v.top_candidates(3):
        if cands:
            wants.append(max(_card_want(v, card, lands) for card in cands))
    want_shuffle = bool(wants) and max(wants) < 0.5
    target = _CAT_SHUFFLE_YES if want_shuffle else _CAT_SHUFFLE_NO
    for i, c in enumerate(v.cats):
        if c == target:
            return i
    return 0


# ── Name-a-card (Cabal Therapy, Pithing Needle, Disruptor Flute) ────────────
_CABAL_THERAPY = "Cabal Therapy"


def _name_card_pick(v) -> int:
    """Pick the name. Cabal Therapy (the candidates are the target's nonland
    cards) names what is most likely in their hand: a card we saw there
    (Thoughtseize, Cabal Therapy's own reveal) first, then a card that can
    still be in it (never one the viewer knows is elsewhere, when its view can
    tell), else the card with the most copies left unseen in their registered
    decklist. A lock piece (Pithing Needle / Disruptor Flute) names an
    opponent's permanent in play first, never one of ours, then the same
    unseen-copies rule."""
    deck = v.opp_deck
    seen: dict = {}
    for card in (v.opp_graveyard + v.opp_exile + [p.card for p in v.opp_perms]):
        seen[card] = seen.get(card, 0) + 1
    known_hand = set(v.opp_known_hand)
    opp_perms = {p.card for p in v.opp_perms}
    own_perms = {p.card for p in v.my_perms}
    therapy = v.pending == _CABAL_THERAPY
    best_i, best_key = 0, None
    for i in range(v.n):
        if v.cats[i] != _CAT_NAME_CARD:
            continue
        card = v.card(i)
        unseen = deck.get(card, 0) - seen.get(card, 0)
        possible = v.opp_hand_possible(card) is not False
        if therapy:
            key = (card in known_hand, possible, unseen, _card_mana_value(v, card))
        else:
            # A lock only stops non-mana activated abilities: naming a land
            # that just taps for mana locks nothing.
            key = (_has_nonmana_activation(v, card),
                   card in opp_perms and card not in own_perms, card not in own_perms,
                   card in known_hand, unseen)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def _has_nonmana_activation(v, card) -> bool:
    """True unless ``card`` is a land whose only activated abilities are mana
    abilities (basic land types or mana-ability lines): Pithing Needle can't stop
    those (its text exempts mana abilities, CR 605.1a)."""
    if not _is_land_card(v, card):
        return True
    return v.cards.has_nonmana_activation(card)


# ── Combat sub-choices ──────────────────────────────────────────────────────
def _block_target_pick(v) -> int:
    """Which attacker the just-declared blocker blocks (BLOCK_TARGET; the
    pending source is the blocker). A block that kills the attacker and
    survives first, then — when the unblocked damage would be lethal — the
    biggest attacker (absorb the most damage), then a trade with the biggest
    attacker it kills, else the biggest attacker."""
    blocker = None
    if v.exact_refs:
        p = v.pending_perm
        if p is not None and p.creature:
            blocker = (p.power, p.toughness)
    else:
        blocker = _lookup_pt(v.my_perms, v.pending)
    bp, bt = blocker if blocker else (0, 0)
    attackers = [p for p in _creatures(v.opp_perms) if p.attacking]
    lethal = sum(p.power for p in attackers) >= v.life
    best_i, best_key = 0, None
    for i in range(v.n):
        if v.cats[i] != _CAT_BLOCK_TARGET:
            continue
        a = v.ref_perm(i)
        ap, at = (a.power, a.toughness) if a is not None and a.creature else (0, 0)
        kills, survives = bp >= at, bt > ap
        key = (kills and survives, lethal and ap, kills, ap)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def _attack_target_pick(v) -> int:
    """Attack the opponent (ATTACK_TARGET's player option) rather than a
    planeswalker: damage to the player ends the game."""
    for i in range(v.n):
        if v.cats[i] == _CAT_ATTACK_TARGET and v.zone(i) == _REF_PLAYER_OPP:
            return i
    return 0


# Creatures that attack with a mandatory "draw a card" trigger: never swing
# with them off an empty library (a draw at 0 library is a loss — a draw of
# the game even when the same trigger kills the opponent).
_ATTACK_DRAWERS = frozenset({"Archon of Cruelty", "Uro, Titan of Nature's Wrath"})

# ── Ninjutsu (CR 702.49) ────────────────────────────────────────────────────
# Ninjas and their attacking power once in play (Kaito is a 3/4 creature
# during our turn).
_NINJA_POWER = {"Kaito, Bane of Nightmares": 3}
# Creatures worth returning to hand for their enters-the-battlefield value.
_ETB_VALUE = frozenset({"Orcish Bowmasters", "Baleful Strix", "Stoneforge Mystic",
                        "Recruiter of the Guard"})


def _ninjutsu_pick(v) -> int | None:
    """Ninjutsu offered while blockers are declared (declare-blockers step,
    our turn): swap an unblocked attacker for the ninja when the ninja hits
    harder than the weakest unblocked attacker, or that attacker is an ETB
    creature worth recasting. Only in the declare-blockers step — after
    combat damage the ninja arrives too late to deal any."""
    if v.step != _STEP_DECLARE_BLOCKERS or not v.is_active:
        return None
    unblocked = [p for p in _creatures(v.my_perms) if p.attacking and not p.blocked]
    for i in range(v.n):
        if v.cats[i] != _CAT_ACTIVATE or v.zone(i) != _REF_SELF_HAND:
            continue
        if not v.is_keyword_ability(i, "Ninjutsu"):
            continue
        card = v.card(i)
        power = _NINJA_POWER.get(card, v.cards.power(card) if card is not None else 0)
        if any(p.power < power or p.card in _ETB_VALUE for p in unblocked):
            return i
    return None


def _return_pick(v) -> int:
    """A return-to-hand cost pick (RETURN_PERMANENT): Daze's Island, Scryb
    Ranger's Forest, ninjutsu's unblocked attacker. A tapped land first (it
    had already made its mana); among creatures an ETB creature we get to
    recast, then the weakest nontoken — a returned token just vanishes."""
    best_i, best_key = 0, None
    for i in range(v.n):
        if v.cats[i] != _CAT_RETURN:
            continue
        p = v.ref_perm(i)
        if p is None:
            key = (False, False, True, 0)
        else:
            key = (p.tapped and p.land, p.card in _ETB_VALUE, not p.token, -p.power)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


# ── Equipment and other own-benefit targeting ───────────────────────────────
def _is_equipment(v, card) -> bool:
    return card is not None and v.cards.is_equipment(card)


def _equip_holds(v) -> set[int]:
    """Menu indices of equip / reconfigure activations to skip: moving an
    Equipment that is already attached just burns mana (the equip is offered
    again at every priority), and Reconfigure turns a growing Lion Sash
    creature back into an Equipment. An unattached
    Equipment is still equipped (onto the best creature, see
    _own_benefit_target_pick)."""
    holds: set[int] = set()
    for i in range(v.n):
        if v.cats[i] != _CAT_ACTIVATE:
            continue
        if not _is_equipment(v, v.card(i)):
            continue
        if v.is_keyword_ability(i, "Reconfigure"):
            holds.add(i)
        elif v.is_keyword_ability(i, "Equip"):
            p = v.ref_perm(i)
            if p is not None and p.attached:
                holds.add(i)
    return holds


_GUIDE_OF_SOULS = "Guide of Souls"
# Sources whose target is helped, not hurt, by the effect: pump / counters on
# a creature we control, or a card returned to us from our graveyard.
_OWN_BENEFIT_TARGETS = frozenset({
    _GUIDE_OF_SOULS,            # +1/+1 counters and flying on an attacker
    "Scythecat Cub",            # +1/+1 counter on a creature we control
    "Pre-War Formalwear",       # reanimates a creature card, then attaches
    "Mystic Sanctuary",         # an instant / sorcery back on top
})


def _own_benefit_target_pick(v) -> int | None:
    """Target pick for an effect that helps its target (equip, Guide of
    Souls' angel counters, Scythecat Cub's counter, graveyard returns):
    among our own options, a permanent: the creature that gets the most out
    of it — untapped and able to attack, highest power, no flying yet for
    Guide of Souls' flying counter; a card in our graveyard: the highest
    mana value. None when the pending source isn't such an effect, or when
    the menu also offers opposing objects — then it is a harmful effect of
    the same card (Meteor Sword's enters trigger destroys, Skateboard's
    taps), whose targeting stays with the generic rule."""
    pending = v.pending
    if not (_is_equipment(v, pending) or pending in _OWN_BENEFIT_TARGETS):
        return None
    if any(v.cats[i] == _CAT_TARGET and v.ctrl(i) != 1 for i in range(v.n)):
        return None
    best_i, best_key = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_TARGET or v.ctrl(i) != 1:
            continue
        p = v.ref_perm(i)
        if p is None:
            key = (0, _card_mana_value(v, v.card(i)), 0)
        else:
            already = _is_equipment(v, pending) and p.has_attachment
            flying = "Flying" in p.keywords
            key = (not already,
                   not (p.tapped or p.sick),
                   not (pending == _GUIDE_OF_SOULS and flying),
                   p.power)
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


# ── Activation gates (hard tier) ────────────────────────────────────────────
_KARAKAS = "Karakas"
_GOBLIN_BOMBARDMENT = "Goblin Bombardment"
_BLAST_ZONE = "Blast Zone"
_DAUTHI_VOIDWALKER = "Dauthi Voidwalker"
_KAITO = "Kaito, Bane of Nightmares"
_WITCH_ENCHANTER = "Witch Enchanter"   # its land face pays 3 life
# Griselbrand: "Pay 7 life: Draw seven cards." Only from a healthy life total
# (7 life out of 10 hands the opponent the game) and never into a deck-out.
_GRISELBRAND = "Griselbrand"
_GRISELBRAND_DRAW = 7
_GRISELBRAND_MIN_LIFE = 15
# Blast Zone's abilities in script order: mana, charge (X), destroy.
_BLAST_ZONE_CHARGE_ORD = 1
_BLAST_ZONE_DESTROY_ORD = 2
# Kaito, Bane of Nightmares' loyalty abilities: +1 emblem, 0 surveil+draw,
# -2 tap and stun.
_KAITO_SURVEIL_ORD = 1
_KAITO_STUN_ORD = 2

# Knight of Autumn's ETB modes in script order: two +1/+1 counters, destroy
# target artifact or enchantment, gain 4 life.
_KNIGHT_OF_AUTUMN = "Knight of Autumn"
_KNIGHT_COUNTERS_ORD = 0
_KNIGHT_DESTROY_ORD = 1
_KNIGHT_LIFE_ORD = 2
# At or below this life the Knight takes the 4 life over the counters.
_KNIGHT_LIFE_THRESHOLD = 8


def _knight_of_autumn_mode_pick(v) -> int | None:
    """Knight of Autumn's mode, announced as its ETB trigger goes on the
    stack (CR 603.3c): destroy an opposing artifact or enchantment when there
    is one (the mode is offered whenever ANY is on the battlefield, ours
    included); else the 4 life when low or when the Knight has already left
    the battlefield (its counters would go nowhere); else the counters."""
    by_ord = {v.ordinal(i): i for i in range(v.n)}
    if (_KNIGHT_DESTROY_ORD in by_ord
            and any(p.art_ench for p in v.opp_perms)):
        return by_ord[_KNIGHT_DESTROY_ORD]
    knight_here = any(p.card == _KNIGHT_OF_AUTUMN for p in v.my_perms)
    if _KNIGHT_LIFE_ORD in by_ord and (not knight_here
                                       or v.life <= _KNIGHT_LIFE_THRESHOLD):
        return by_ord[_KNIGHT_LIFE_ORD]
    return by_ord.get(_KNIGHT_COUNTERS_ORD, by_ord.get(_KNIGHT_LIFE_ORD))


def _blast_zone_plan(v) -> tuple[int, int] | None:
    """(current charge counters, the mana value worth blowing up) for our
    Blast Zone, or None when no mana value is worth it. Destroying every
    nonland permanent with MV == counters is symmetric, so pick the MV
    that takes 2+ more of their permanents than ours (or a 3+ power
    differential)."""
    zone = next((p for p in v.my_perms if p.card == _BLAST_ZONE), None)
    if zone is None:
        return None

    def by_mv(perms):
        out: dict[int, list[int]] = {}
        for p in perms:
            if p.land:
                continue
            mv = 0 if p.token else int(p.mana_value)
            out.setdefault(mv, []).append(p.power)
        return out
    opp, own = by_mv(v.opp_perms), by_mv(v.my_perms)
    best, best_key = None, None
    for mv, powers in opp.items():
        count_net = len(powers) - len(own.get(mv, []))
        power_net = sum(powers) - sum(own.get(mv, []))
        if not (count_net >= 2 or (count_net >= 1 and power_net >= 3)):
            continue
        key = (count_net, power_net, -mv)
        if best_key is None or key > best_key:
            best, best_key = mv, key
    if best is None:
        return None
    return zone.other_counters, best


def _activation_holds(v) -> set[int]:
    """Menu indices of activations the hard tier must not take this decision
    (on top of _equip_holds): Karakas without a targetable opposing
    legendary creature (it would bounce our own), Goblin Bombardment short of lethal (each
    activation sacrifices a creature for 1 damage), Blast Zone off its plan
    (_blast_zone_plan: charge only toward the target mana value, destroy only
    at it), Dauthi Voidwalker's sacrifice without an exiled void card
    worth more than the 3/2 itself, and Griselbrand's pay-7-life draw 7
    unless the life and the library can both afford it."""
    holds = _equip_holds(v)
    plan = None
    for i in range(v.n):
        if v.cats[i] != _CAT_ACTIVATE:
            continue
        card = v.card(i)
        if card == _KARAKAS:
            if not any(p.legendary and not {"Hexproof", "Shroud"} & set(p.keywords)
                       for p in _creatures(v.opp_perms)):
                holds.add(i)
        elif card == _GOBLIN_BOMBARDMENT:
            if v.opp_life > len(_creatures(v.my_perms)):
                holds.add(i)
        elif card == _BLAST_ZONE:
            plan = plan or _blast_zone_plan(v) or (0, -1)
            counters, target = plan
            ordinal = v.ordinal(i)
            if ((ordinal == _BLAST_ZONE_CHARGE_ORD and not (0 <= counters < target))
                    or (ordinal == _BLAST_ZONE_DESTROY_ORD and counters != target)):
                holds.add(i)
        elif card == _DAUTHI_VOIDWALKER:
            if _best_void_card(v) is None:
                holds.add(i)
        elif card == _GRISELBRAND:
            if (v.life < _GRISELBRAND_MIN_LIFE
                    or v.library <= _GRISELBRAND_DRAW + 3):
                holds.add(i)
    return holds


def _best_void_card(v):
    """Highest-mana-value nonland card the opponent owns in exile with a
    counter on it (Dauthi Voidwalker's void counters), if it beats keeping
    the Voidwalker (mana value 3+)."""
    best, best_mv = None, 2.0
    for card, counters in v.opp_exile_counters:
        if card is None or _is_land_card(v, card):
            continue
        if counters < 1:
            continue
        mv = _card_mana_value(v, card)
        if mv > best_mv:
            best, best_mv = card, mv
    return best


def _kaito_ability_pick(v) -> int | None:
    """Kaito, Bane of Nightmares' loyalty ability: -2 (tap and stun) on a
    real opposing threat (power 3+) while it leaves Kaito alive, else 0
    (surveil 2, draw if the opponent lost life) — the +1 emblem only pumps
    Ninjas, and Kaito is our only one."""
    options = {v.ordinal(i): i for i in range(v.n)
               if v.cats[i] == _CAT_ACTIVATE
               and v.card(i) == _KAITO
               and v.zone(i) == _REF_SELF_BATTLEFIELD}
    if not options:
        return None
    kaito = next((p for p in v.my_perms if p.card == _KAITO), None)
    threat = any(p.power >= 3 for p in _creatures(v.opp_perms))
    if threat and kaito is not None and kaito.loyalty > 2 and _KAITO_STUN_ORD in options:
        return options[_KAITO_STUN_ORD]
    return options.get(_KAITO_SURVEIL_ORD)


# ── Card choices ─────────────────────────────────────────────────────────────
def _tutor_pick(v) -> int | None:
    """A library search for a nonland card (Green Sun's Zenith, Recruiter of
    the Guard, Stoneforge Mystic): the highest mana value offered — the
    engine already capped it at what the search may find; printed power breaks
    ties. None for a land-only search (fetch lands keep their own rules)."""
    best_i, best_mv = None, -1.0
    for i in range(v.n):
        if v.cats[i] != _CAT_SEARCH:
            continue
        card = v.card(i)
        if card is None or _is_land_card(v, card):
            continue
        mv = _card_mana_value(v, card) + 0.01 * v.cards.power(card)
        if mv > best_mv:
            best_i, best_mv = i, mv
    return best_i


def _land_search_pick(v) -> int | None:
    """A land-only search (fetch lands, Knight of the Reliquary, ...): the land
    adding the most colours our hand's spells need and our lands don't make
    yet, then the most colours overall — never a third Plains while black
    spells wait in hand. Colours are W/U/B/R/G as 0-4; a land counts the
    colours its basic land types make. None when
    the menu offers a nonland card or no land at all."""
    have: set[int] = set()
    for p in v.my_perms:
        if p.land and p.card is not None:
            have |= v.cards.land_colors(p.card)
    need: set[int] = set()
    for card in v.hand:
        colors = v.cards.cost_colors(card)
        if colors is not None and not _is_land_card(v, card):
            need |= colors
    need -= have
    best_i, best_key = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_SEARCH:
            continue
        card = v.card(i)
        if card is None:
            continue
        if not _is_land_card(v, card):
            return None
        colors = v.cards.land_colors(card)
        key = (len(colors & need), len(colors))
        if best_key is None or key > best_key:
            best_i, best_key = i, key
    return best_i


def _sacrifice_pick(v) -> int | None:
    """Sacrifice the least valuable permanent offered: a token before a card,
    a tapped land before an untapped one, the weakest creature; never a
    _SAC_LAST_LANDS utility land while anything else is offered."""
    best_i, best_key = None, None
    for i in range(v.n):
        if v.cats[i] != _CAT_SACRIFICE:
            continue
        p = v.ref_perm(i)
        if p is None:
            key = (False, True, True, 0.0)
        else:
            key = (p.card in _SAC_LAST_LANDS,
                   not p.token,
                   not p.tapped,
                   p.power + p.toughness + p.mana_value)
        if best_key is None or key < best_key:
            best_i, best_key = i, key
    return best_i


# Opponent spells that win or dominate on resolution: countered on sight even
# by a Doomsday deck that otherwise saves its counters for its own kill.
_COMBO_THREATS = frozenset({_DOOMSDAY, _THASSAS_ORACLE}) | _REANIMATION_SPELLS


def _opponent_stack_has_any(v, cards) -> bool:
    return any(not s.mine and s.card in cards for s in v.stack)


# Damage dealt by targeted burn, keyed by source (default 3): a burn spell
# goes at a creature only if it kills it.
_BURN_DAMAGE = {_GOBLIN_BOMBARDMENT: 1, "Fireblast": 4, "Exquisite Firecraft": 4}


class ScriptedCore:
    """A stateless rule-based agent parameterised by an AgentConfig.

    ``decide(view)`` is a pure function of the (perspective-relative) view, except
    for the RANDOM tier which draws from a per-agent seeded generator so parallel
    vec-env subprocesses stay reproducible and independent of the global RNG, the
    EXPLORE tier which carries a per-game set of already-chosen cards for
    its novelty bias, and the GREEDY/HEURISTIC path which carries a per-game set
    of "fruitless" activation cards (the stall guard, see ``_hard_act``).
    Both per-game sets are cleared by ``new_game`` and are deterministic
    functions of the trajectory, so same-seed replays stay identical.
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config
        # Per-seat deck names (keyed by seat-is-A), pushed by callers that know
        # them via set_deck_names(); None = unknown, fall back to content-based
        # identification. Per-seat because one shared instance may drive both
        # players, each piloting a different deck.
        self._deck_names: dict[bool, str | None] = {True: None, False: None}
        # Per-agent generator: only consulted by the RANDOM tier.
        self._rng = random.Random(config.rng_seed)
        # Cards already cast/activated this game: only touched by the EXPLORE
        # tier (novelty bias). Reset per game via new_game().
        self._explore_seen: set = set()
        # Per-seat count of mulligans taken this game (London: the redrawn
        # hand is always 7 at the keep/mull decision, so the Doomsday all-in
        # mulligan's "never below 6" floor must count mulligans, not cards).
        # Zeroed on every keep, so a bo3's later games start fresh.
        self._mull_counts: dict[bool, int] = {True: 0, False: 0}
        # Stall guard (GREEDY/HEURISTIC only): per-seat map of card ->
        # [failed_search_count, turn_of_last_fail] for activated abilities whose
        # search found NOTHING. Keyed by seat (view.seat_is_a) so one shared
        # instance driving both players can't leak one seat's dead activation to
        # the other. Reset per game via new_game(). See _SEARCH_MENU_CATS / _hard_act.
        self._fruitless_activations: dict[bool, dict] = {}

    def new_game(self) -> None:
        """Reset per-game state at the start of a game.

        Called by ``runner.run_games`` (via ``ScriptedController.new_game``) before
        each game so a reused agent instance can't leak state across games. The
        per-game state is the EXPLORE novelty-bias set and the GREEDY/HEURISTIC
        fruitless-activation set; as a belt-and-braces for reuse paths that never
        call this (e.g. a scripted opponent living across vec-env episodes), both
        ``_explore_action`` and ``_hard_act`` also clear their own set whenever
        they see a pregame mulligan decision.
        """
        self._explore_seen.clear()
        self._fruitless_activations.clear()
        self._mull_counts = {True: 0, False: 0}

    def set_deck_names(self, deck_a: str | None, deck_b: str | None) -> None:
        """Tell the agent which deck each seat pilots.

        Callers that know the decks (runner.run_games, ModelVsScriptedEnv, the
        self-play scripted fallback, the RMSTATE controller from the DECKLISTS
        frame) push the names here so deck identification uses one name rule
        (_deck_named: 'doomsday'/'tron' anywhere in the name — league/car_doomsday
        and league/tron count). A matching name is authoritative; a missing or
        non-matching name falls back to content-based identification, so
        sculpted temp decks in the test harness keep working."""
        self._deck_names = {True: deck_a, False: deck_b}

    def _seat_deck(self, v) -> str | None:
        """Deck name piloted by the seat this view belongs to (None if unknown)."""
        return self._deck_names[bool(v.seat_is_a)]

    def _opp_deck(self, v) -> str | None:
        """Deck name piloted by this view's OPPONENT (None if unknown)."""
        return self._deck_names[not bool(v.seat_is_a)]

    def _deck_is_doomsday(self, v) -> bool:
        """Unified doomsday identification: a known deck name is authoritative
        either way; only an unnamed deck (sculpted harness decks) falls back to
        the content heuristic, which also fires on the Thoughtseize / Dark
        Ritual / Lotus Petal hands of reanimator, bug and wrb_energy."""
        deck = self._seat_deck(v)
        if deck is not None:
            return _deck_named(deck, "doomsday")
        return _is_doomsday_deck(v)

    def _deck_is_tron(self, v) -> bool:
        """Unified tron identification: deck name first, content fallback.

        With no name known the synergy stays available (the historical
        behaviour — each tron block self-gates on the view anyway); with a
        non-matching name, a Tron land controlled or in hand still counts."""
        deck = self._seat_deck(v)
        if deck is None or _deck_named(deck, "tron"):
            return True
        return (bool(_controlled_tron_pieces(v))
                or any(_hand_has_card(v, card) for card in _TRON_LANDS))

    def decide(self, v) -> int:
        """The action index for the decision ``v`` shows."""
        if self.config.skill is Skill.RANDOM:
            return self._random_action(v.n)
        if self.config.skill is Skill.EXPLORE:
            return self._explore_action(v)
        return self._hard_act(v)

    def _hard_act(self, v) -> int:
        """GREEDY/HEURISTIC entry: fruitless-activation stall guard + dispatch.

        The guard piggybacks on the fact that a resolution-time search query is
        answered while the resolving ability is still on top of the stack (the
        engine's search_zone runs inside the ability's resolution), so the view
        itself names the culprit — no cross-decision correlation state needed.
        When a search-category menu offers ONLY the fail-to-find slot, the
        search provably cannot succeed; if the top of the stack is a
        self-controlled ability, the fail is recorded against its source card
        and the greedy ACTIVATE scan skips that card for the rest of the
        TURN — retrying next turn, because eligibility can change invisibly
        (Aether Vial's search keys on its charge counters: a fail at 0 counters
        says nothing about 1). A card that fails
        _FRUITLESS_ACTIVATION_LIMIT times is dropped for the game, so the
        original stall (Vial re-activated into a guaranteed fail every turn to
        the decision cap) stays bounded at a few wasted activations.
        """
        # cats[0] is enough to classify the menus we care about: a mulligan
        # query opens with the KEEP_HAND action (index 0), and a fail-only search
        # menu has exactly one action.
        c0 = v.cats[0] if v.n else None
        if c0 in (_CAT_MULLIGAN, _CAT_KEEP_HAND):
            # A game always opens with a mulligan query, so seeing one means a
            # new game started: clear the guard even when nobody called
            # new_game() (same safety net as _explore_seen in _explore_action).
            self._fruitless_activations.clear()
        elif v.n == 1 and c0 in _SEARCH_MENU_CATS and v.card(0) is None:
            # Fail-only search menu (single choice, no card):
            # blame the resolving self-controlled ability on top of the stack.
            top = v.stack_top
            if (top is not None and top.card is not None and top.mine   # self-controlled
                    and not top.spell):                                 # ability, not spell
                seat = bool(v.seat_is_a)
                entry = self._fruitless_activations.setdefault(
                    seat, {}).setdefault(top.card, [0, -1])
                entry[0] += 1
                entry[1] = v.turn
        fruitless = None
        seat_fails = self._fruitless_activations.get(bool(v.seat_is_a))
        if seat_fails:
            turn = v.turn
            fruitless = {card for card, (n, t) in seat_fails.items()
                         if n >= _FRUITLESS_ACTIVATION_LIMIT or t == turn}
        if self.config.skill is Skill.HEURISTIC:
            return self._heuristic_action(v, fruitless)
        return _greedy_action(v, fruitless, doomsday=self._deck_is_doomsday(v))

    # ------------------------------------------------------------------
    # HEURISTIC decision points (everything else falls back to GREEDY).
    # ------------------------------------------------------------------

    def _heuristic_action(self, v, fruitless: set | None = None) -> int:
        cfg = self.config
        # Tron-synergy gating uses the unified deck identification (name rule,
        # content fallback) — computed once per decision for all blocks below.
        tron_synergy = cfg.use_tron_synergy and self._deck_is_tron(v)
        cats = v.cats
        num_choices = v.n

        # Smart mulligan
        if cfg.use_smart_mulligan and any(c == _CAT_MULLIGAN for c in cats):
            return self._mulligan_choice(v)

        if cfg.use_combat_sim:
            # Declare blockers (SEL_BLK + CONF_BLK offered together)
            if any(c == _CAT_CONF_BLK for c in cats):
                return self._block_choice(v)
            # "Which attacker to block" sub-query for the blocker just declared
            if any(c == _CAT_BLOCK_TARGET for c in cats):
                return _block_target_pick(v)
            # Declare attackers
            if any(c == _CAT_SEL_ATK for c in cats):
                return self._attack_choice(v)
        # Which player / planeswalker an attacker attacks
        if any(c == _CAT_ATTACK_TARGET for c in cats):
            return _attack_target_pick(v)

        # ── Prompts: optional triggers / costs, names, returns, library order ─
        dd = self._deck_is_doomsday(v)
        pending = v.pending
        if _is_yesno_menu(v):
            return self._yesno_choice(v)
        if any(c == _CAT_NAME_CARD for c in cats):
            return _name_card_pick(v)
        if any(c == _CAT_RETURN for c in cats):
            return _return_pick(v)
        # Scry / surveil, Ponder's rearrange and shuffle, Brainstorm's put-back
        # and London-mulligan bottoming by card desirability (_card_want). A
        # Doomsday deck keeps its own pile / put-back / surveil rules.
        if not dd:
            if _is_look_split_menu(v):
                return _look_split_pick(v)
            if any(c in (_CAT_SHUFFLE_YES, _CAT_SHUFFLE_NO) for c in cats):
                return _shuffle_pick(v)
            if (all(c == _CAT_TOP_LIBRARY for c in cats)
                    and pending in (_PONDER, _BRAINSTORM)):
                return _least_wanted_pick(v, _CAT_TOP_LIBRARY, _lands_total(v))
            if all(c == _CAT_BOTTOM_CARD for c in cats):
                return _least_wanted_pick(v, _CAT_BOTTOM_CARD, _hand_land_count(v))
        # Blast Zone's charge X: exactly the counters its plan still needs.
        if pending == _BLAST_ZONE and all(c == _CAT_CHOOSE_X for c in cats):
            plan = _blast_zone_plan(v)
            if plan is not None:
                return max(0, min(num_choices - 1, plan[1] - plan[0]))
        # Knight of Autumn's ETB mode (_knight_of_autumn_mode_pick).
        if pending == _KNIGHT_OF_AUTUMN and all(c == _CAT_CHOOSE_MODE for c in cats):
            pick = _knight_of_autumn_mode_pick(v)
            if pick is not None:
                return pick
        # Dauthi Voidwalker's pick: the most expensive void card.
        if pending == _DAUTHI_VOIDWALKER and any(c == _CAT_CHOOSE_CARD for c in cats):
            return max(range(num_choices),
                       key=lambda i: _card_mana_value(v, v.card(i)))
        # Ninjutsu: swap an unblocked attacker for the ninja (_ninjutsu_pick).
        if any(c == _CAT_ACTIVATE for c in cats):
            pick = _ninjutsu_pick(v)
            if pick is not None:
                return pick

        # ── Reanimation / lands-combo interceptions ─────────────────────────
        # All card / pending-source gated (no deck-name gate needed): each
        # rule can only fire on a menu that names its specific card, so no
        # other deck's decisions change. Kept out of GREEDY so the easy tier
        # stays byte-identical.

        # Legend rule: duplicate Dark Depths → keep the 0-counter clone so
        # Marit Lage happens (see _keep_legend_choice).
        if any(c == _CAT_KEEP_LEGEND for c in cats):
            pick = _keep_legend_choice(v)
            if pick is not None:
                return pick

        # Life from the Loam: take the dredge draw-replacement whenever the
        # library can afford the mill — Loam back to hand every turn is the
        # lands deck's engine (recast it to return Wasteland / Urza's Saga).
        if (any(c == _CAT_CHOOSE_REPLACEMENT for c in cats)
                and v.library > _LOAM_DREDGE_MIN_LIBRARY):
            for i, c in enumerate(cats):
                if c == _CAT_CHOOSE_REPLACEMENT and v.card(i) == _LIFE_FROM_LOAM:
                    return i

        # Beneficial-target polarity: the generic target rule prefers
        # opponent-controlled targets (right for removal, backwards for
        # effects that help their target). Keyed on the pending-decision
        # source so only the named cards change behaviour.
        if any(c == _CAT_TARGET for c in cats):
            # Reanimate / Animate Dead: the biggest creature card offered,
            # preferring our own graveyard.
            if pending in _REANIMATION_SPELLS:
                pick = _reanimation_target_choice(v)
                if pick is not None:
                    return pick
            # Thespian's Stage: copy our own Dark Depths (the Marit Lage line).
            if pending == _THESPIANS_STAGE:
                for i, c in enumerate(cats):
                    if c == _CAT_TARGET and v.ctrl(i) == 1 and v.card(i) == _DARK_DEPTHS:
                        return i
            # Life from the Loam: return the engine lands first (Urza's Saga
            # to redeploy, then Wasteland to keep stripping their mana).
            if pending == _LIFE_FROM_LOAM:
                for want in (_URZAS_SAGA, _WASTELAND):
                    for i, c in enumerate(cats):
                        if c == _CAT_TARGET and v.card(i) == want:
                            return i

        # London-mulligan bottoming: the Doomsday all-in mulligan keeps hands
        # for their specific combo pieces, so don't bottom blind (slot 0 could
        # be Doomsday itself) — give back junk by the same tiers as the
        # Brainstorm put-back. Other decks keep the legacy first-slot pick.
        if any(c == _CAT_BOTTOM_CARD for c in cats) and self._deck_is_doomsday(v):
            return _dd_return_pick(v, _CAT_BOTTOM_CARD)

        # Discard choice (lands-first conservative; replaces "first listed
        # card"). Own-hand discards, in priority order:
        #   1. a reanimation fatty — its home is the graveyard (the scripted
        #      agent never sideboards, so no post-board Show and Tell plan to
        #      protect);
        #   2. an excess land, while the hand still holds >= 3;
        #   3. the highest-MV card not castable any time soon
        #      (MV > lands in play + 1);
        #   4. the highest-MV card (lands are MV 0, so a spell goes first).
        # MV here is what the card really costs (_effective_mana_value: Force
        # of Will is a free spell, Murktide Regent delves).
        # A menu of OPPONENT-owned cards (our Thoughtseize-style pick) takes
        # their highest-MV card instead — strip the biggest threat. A
        # Doomsday deck instead strips whatever can disrupt the combo:
        # countermagic first, then instant-speed removal (Thassa's Oracle
        # dies to it mid-kill), then the highest-MV fallback.
        if any(c == _CAT_DISCARD for c in cats):
            own_dc, opp_dc = [], []
            for i, c in enumerate(cats):
                if c != _CAT_DISCARD:
                    continue
                card = v.card(i)
                (own_dc if v.ctrl(i) == 1 else opp_dc).append((i, card))
            for i, card in own_dc:
                if card in _REANIMATION_FATTIES:
                    return i
            if own_dc:
                if _hand_land_count(v) >= 3:
                    for i, card in own_dc:
                        if _is_land_card(v, card):
                            return i
                castable_mv = _self_land_count(v) + 1
                uncastable = [(_effective_mana_value(v, card), i) for i, card in own_dc
                              if _effective_mana_value(v, card) > castable_mv]
                if uncastable:
                    return max(uncastable)[1]
                return max((_effective_mana_value(v, card), i) for i, card in own_dc)[1]
            if opp_dc:
                if self._deck_is_doomsday(v):
                    for i, card in opp_dc:
                        if card in _COUNTER_SPELLS:
                            return i
                    for i, card in opp_dc:
                        if card in _DD_DISRUPTOR_REMOVAL:
                            return i
                return max((_card_mana_value(v, card), i) for i, card in opp_dc)[1]

        # Urza's Saga chapter-II ability: always make the Construct when
        # offered. The Saga's mana ability is auto-pay (never offered at
        # priority), so the only Saga ACTIVATE here is the {2},{T} token
        # ability — and the Saga sacrifices itself after chapter III, so the
        # window is short. Ungated by phase, like The One Ring.
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE and v.card(i) == _URZAS_SAGA:
                return i

        # Thespian's Stage: with our own Dark Depths in play, take the Clone
        # activation immediately (copy Depths → legend rule keeps the
        # 0-counter copy → Marit Lage). Without Depths, hold it — copying a
        # random land wastes the {2} and the tap.
        hold_activations = None
        if any(c == _CAT_ACTIVATE and v.card(i) == _THESPIANS_STAGE
               for i, c in enumerate(cats)):
            if _controls_card(v, _DARK_DEPTHS):
                for i, c in enumerate(cats):
                    if c == _CAT_ACTIVATE and v.card(i) == _THESPIANS_STAGE:
                        return i
            else:
                hold_activations = {_THESPIANS_STAGE}

        # Sacrifice-cost pick: never eat a combo piece or irreplaceable
        # utility land while any other option exists (Crop Rotation sacking
        # the Dark Depths it plays around defeats itself). A Doomsday deck
        # keeps Thassa's Oracle's UU and its black source on the board
        # (_dd_sacrifice_pick). Otherwise the least valuable permanent
        # (_sacrifice_pick: tokens, tapped lands, the weakest creature).
        if any(c == _CAT_SACRIFICE for c in cats):
            if dd:
                pick = _dd_sacrifice_pick(v)
                if pick is not None:
                    return pick
            pick = _sacrifice_pick(v)
            if pick is not None:
                return pick

        # Land-search preference (Crop Rotation, Expedition Map, any land
        # tutor — self-gating, the options only exist when the search can
        # fetch them): complete the missing Depths/Stage half, else grab
        # Tabernacle against a wide opposing board our own board shrugs at,
        # else Urza's Saga. Tron decks run none of these cards, so their own
        # search priorities (the tron blocks below) are unaffected.
        if any(c == _CAT_SEARCH for c in cats):
            search_prefs = []
            has_depths = _controls_card(v, _DARK_DEPTHS)
            has_stage = _controls_card(v, _THESPIANS_STAGE)
            if has_depths and not has_stage:
                search_prefs.append(_THESPIANS_STAGE)
            if has_stage and not has_depths:
                search_prefs.append(_DARK_DEPTHS)
            if any(v.card(i) == _TABERNACLE for i, c in enumerate(cats) if c == _CAT_SEARCH):
                gs_creatures = (len(_creatures(v.opp_perms)), len(_creatures(v.my_perms)))
                if gs_creatures[0] >= 3 and gs_creatures[1] <= 1:
                    search_prefs.append(_TABERNACLE)
            search_prefs.append(_URZAS_SAGA)
            for want in search_prefs:
                for i, c in enumerate(cats):
                    if c == _CAT_SEARCH and v.card(i) == want:
                        return i

        # Exploration first among casts: the extra land drop compounds every
        # later turn, so it beats any other one-shot play this turn.
        for i, c in enumerate(cats):
            if c == _CAT_CAST and v.card(i) == _EXPLORATION:
                return i

        # Land drop: complete the Depths/Stage pair when half is already down,
        # else lead with Urza's Saga (its chapters start ticking — mana, then
        # Constructs, then the tutor). Card gated: decks without these
        # lands in hand fall through unchanged.
        if any(c == _CAT_LAND for c in cats):
            land_prefs = []
            if _controls_card(v, _DARK_DEPTHS):
                land_prefs.append(_THESPIANS_STAGE)
            if _controls_card(v, _THESPIANS_STAGE):
                land_prefs.append(_DARK_DEPTHS)
            land_prefs.append(_URZAS_SAGA)
            for want in land_prefs:
                for i, c in enumerate(cats):
                    if c == _CAT_LAND and v.card(i) == want:
                        return i

        # Cast holds (see _cast_holds), threaded into the greedy fallback's
        # cast scan.
        cast_cards = {v.card(i) for i, c in enumerate(cats) if c == _CAT_CAST}
        hold_casts = self._cast_holds(v, cast_cards) or None

        # Wrath of the Skies X / energy-pay amounts: both arrive as bare
        # number ladders (index == amount), and both should be the planned
        # energy Y — X buys exactly the energy the sweep needs, and the pay
        # step spends exactly that much (the generic pickers defaulted these
        # to 0, sweeping nothing). Both the cast-time X ladder and the
        # resolution-time pay menu name Wrath as the pending-decision source.
        wrath_x_menu = pending == _WRATH_OF_SKIES
        if (num_choices >= 2 and wrath_x_menu
                and all(c in (_CAT_CHOOSE_X, _CAT_OTHER) for c in cats)):
            y, _, _ = _wrath_energy_plan(v)
            return min(y, num_choices - 1)

        # Tron synergy: Candelabra of Tawnos untaps X target lands. It's only a mana
        # engine when it untaps lands that make MORE THAN ONE mana (Ancient Tomb, a
        # metalcraft Urza's Workshop, the assembled Urza's tron lands), so pay X for
        # exactly the tapped ones and target those, instead of letting the generic
        # picker choose X=0 / the first land. The pending-decision source names the
        # ability making each choice even before it hits the stack, so it can't be
        # confused with any other X ability or land-targeting effect.
        if tron_synergy and pending == _CANDELABRA:
            # X value: untap as many of our tapped >1-mana lands as X allows.
            if all(c == _CAT_CHOOSE_X for c in cats):
                return min(num_choices - 1, _tapped_multi_mana_lands(v))
            # Targets: pick the >1-mana lands.
            if any(c == _CAT_TARGET for c in cats):
                for i, c in enumerate(cats):
                    if c == _CAT_TARGET and v.card(i) in _MULTI_MANA_LANDS:
                        return i

        # Solitude's ETB exiles "up to one" creature, so its target menu leads with a
        # "No target" slot the generic picker can fall into. We only cast Solitude with
        # an opponent creature to hit (see _greedy_action), so here take that creature —
        # the biggest one — rather than whiffing on the optional "No target".
        if any(c == _CAT_TARGET for c in cats) and pending == _SOLITUDE:
            pick = _best_opp_creature_target(v)
            if pick is not None:
                return pick

        # Effects that help their target (equip, Guide of Souls, Scythecat Cub,
        # graveyard returns) pick our best option; Goblin Bombardment (only
        # activated for lethal, see _activation_holds) goes face.
        if any(c == _CAT_TARGET for c in cats):
            pick = _own_benefit_target_pick(v)
            if pick is not None:
                return pick
            if pending == _GOBLIN_BOMBARDMENT:
                for i in range(num_choices):
                    if cats[i] == _CAT_TARGET and v.zone(i) == _REF_PLAYER_OPP:
                        return i

        # Eval-based targeting — skipped for combo decks (R1: never disturb Doomsday).
        if (cfg.use_eval_targeting and any(c == _CAT_TARGET for c in cats)
                and not dd):
            return self._target_choice(v)

        # Tron synergy: Karn, the Great Creator's -2 wishes an artifact from the
        # sideboard/exile into hand. Grab Mycosynth Lattice first — with Karn in play
        # it locks the opponent out (every permanent becomes an artifact Karn's static
        # silences) — and fall back to Cityscape Leveler as the beater once Lattice is
        # already in hand/play (so no longer offered from the wishboard). Its fetch is a
        # hidden reveal, so the choice can arrive as a search or a choose-card menu;
        # match by pending source + card across either.
        if tron_synergy and pending == _KARN_GREAT_CREATOR:
            for wish in (_MYCOSYNTH_LATTICE, _CITYSCAPE_LEVELER):
                for i in range(num_choices):
                    if v.card(i) == wish:
                        return i

        # Tron synergy: prioritize casting the Karn payoffs ahead of any other affordable
        # spell — Mycosynth Lattice first (completes the Karn lock), then Cityscape Leveler
        # (the premier threat / artifact-and-land destruction beater).
        if tron_synergy and any(c == _CAT_CAST for c in cats):
            for cast in (_MYCOSYNTH_LATTICE, _CITYSCAPE_LEVELER):
                for i, c in enumerate(cats):
                    if c == _CAT_CAST and v.card(i) == cast:
                        return i

        # Tron synergy: when activating Karn, the Great Creator, prefer his -2 (wish an
        # artifact — Mycosynth Lattice, then Cityscape Leveler — from the sideboard into hand) over the +1
        # (Animate). The two loyalty abilities share the ACTIVATE category and Karn's
        # card, differing only in menu order: the
        # engine emits them in script order, +1 Animate then -2 ChangeZone, so the LAST
        # Karn activation offered is the wish. Only force it when both are on the menu
        # (Karn has the 2 loyalty to spend); with just +1 left, greedy takes it.
        if tron_synergy:
            karn_acts = [i for i, c in enumerate(cats)
                         if c == _CAT_ACTIVATE and v.card(i) == _KARN_GREAT_CREATOR]
            if len(karn_acts) >= 2:
                return karn_acts[-1]

        # Tron synergy: once we control at least one of Urza's Mine/Power-Plant/
        # Tower, prefer playing a hand land that completes the set (a missing
        # real piece, else Planar Nexus) over any other offered land drop.
        if tron_synergy and any(c == _CAT_LAND for c in cats):
            controlled = _controlled_tron_pieces(v)
            if controlled:
                pick = _tron_choice(v, _CAT_LAND, controlled)
                if pick is not None:
                    return pick

        # Tron synergy: a library search (Expedition Map, fetch effects) finds
        # the Tron piece we're missing, or Planar Nexus as a substitute, ahead
        # of any other offered card.
        if tron_synergy and any(c == _CAT_SEARCH for c in cats):
            pick = _tron_choice(v, _CAT_SEARCH, _controlled_tron_pieces(v))
            if pick is not None:
                return pick

        # A tutor for a nonland card: the best card it may find (_tutor_pick);
        # a land search: the colours the hand needs (_land_search_pick).
        if not dd and any(c == _CAT_SEARCH for c in cats):
            pick = _tutor_pick(v)
            if pick is None:
                pick = _land_search_pick(v)
            if pick is not None:
                return pick

        # Activations to skip this decision (_activation_holds), and Kaito's
        # preferred loyalty ability (_kaito_ability_pick) — the others held.
        hold_idxs = None
        if any(c == _CAT_ACTIVATE for c in cats):
            hold_idxs = _activation_holds(v)
            kaito = _kaito_ability_pick(v)
            if kaito is not None:
                hold_idxs |= {i for i in range(num_choices)
                              if i != kaito and cats[i] == _CAT_ACTIVATE
                              and v.card(i) == _KAITO
                              and v.zone(i) == _REF_SELF_BATTLEFIELD}

        # Anything not explicitly improved: proven GREEDY behaviour (with the
        # holds computed above threaded through its scans).
        choice = _greedy_action(v, fruitless,
                                hold_casts=hold_casts,
                                hold_activations=hold_activations,
                                hold_idxs=hold_idxs,
                                doomsday=dd)
        # With nothing better to do in our own main phase, put the companion
        # into hand (it is cast like any other card afterwards).
        if (cats[choice] == _CAT_PASS and v.stack_empty
                and v.is_active and v.in_main):
            for i, c in enumerate(cats):
                if c == _CAT_COMPANION:
                    return i
        return choice

    def _yesno_choice(self, v) -> int:
        """An optional yes/no (OPTIONAL_YESNO; 0 = decline, 1 = accept).

        * A cast offer during a resolution (the accept option names the card:
          suspend, miracle, Amped Raptor): cast it unless it would only hit
          our own things (_resolution_cast_ok) or a cast hold applies to it.
        * The miracle reveal: accept (it opens the cast offer).
        * An optional cost while casting our own spell (kicker, replicate, a
          gift promise — the pending source is our spell on top of the
          stack): decline, keeping the spell at its plain cost.
        * "As it enters, you may pay 3 life" (Witch-Blessed Meadow): only with
          life to spare and in our own turn, when the mana is usable now.
        * Every other "you may" trigger — tutors (Stoneforge Mystic, Recruiter
          of the Guard), Uro's land, Aether Vial's counter, Cori-Steel
          Cutter's attach, Guide of Souls' energy, an opening-hand Leyline:
          accept; each is optional upside for us.
        """
        card = v.card(1)
        if card is not None:
            ok = (_resolution_cast_ok(v, card)
                  and card not in self._cast_holds(v, {card}))
            return 1 if ok else 0
        if _is_miracle_reveal(v):
            return 1
        pending = v.pending
        top = v.stack_top
        if top is not None and top.mine and top.spell and top.card == pending:
            return 0
        if pending == _WITCH_ENCHANTER:
            return 1 if (v.life >= 12 and v.is_active) else 0
        return 1

    def _cast_holds(self, v, cast_cards) -> set:
        """Cards among ``cast_cards`` (the casts on offer) the hard tier
        holds back this decision. Each rule is card gated: it only fires
        when that cast is actually offered."""
        hold_casts: set = set()
        # Reanimate/Animate Dead stay in hand until a creature worth cheating
        # out is in a graveyard (either side's — the engine only offers the
        # cast when SOME creature card is there, but a random 2-drop isn't
        # worth the card).
        if (not _REANIMATION_SPELLS.isdisjoint(cast_cards)
                and not _gy_has_any(v, _REANIMATION_FATTIES)):
            hold_casts |= _REANIMATION_SPELLS
        # Counter triage. A Doomsday deck's counters are combo PROTECTION:
        # hold them until our own Thassa's Oracle — the spell or its win
        # trigger (both stack objects carry Oracle's card) — is on the
        # stack, then counter ANY opponent spell that goes on top of it,
        # cantrip or not (a "cantrip" there is a response to the kill).
        # Other decks: an opponent spell is on the stack but nothing
        # threat-shaped (only exempt cantrips) — let it resolve and keep the
        # counter for their threat.
        if not _COUNTER_SPELLS.isdisjoint(cast_cards):
            if self._deck_is_doomsday(v):
                # ...and likewise fight for Doomsday itself and the Dark
                # Ritual paying for it — four Doomsdays were countered in one
                # 30-game sample while Force of Wills sat in hand — but only
                # while a SECOND counter stays back for Oracle: a countered
                # Doomsday costs a card, a countered Oracle with the pile
                # drawn costs the game (two wins turned into deck-outs when
                # the last counter went to Doomsday).
                n_counters = sum(1 for c in v.hand if c in _COUNTER_SPELLS)
                # An opposing combo spell (their Doomsday, Oracle or
                # reanimation) is countered on sight: letting it resolve
                # loses the game before our own kill matters.
                fight = _opponent_has_spell_on_stack(v) and (
                    _self_stack_has_card(v, _THASSAS_ORACLE)
                    or _opponent_stack_has_any(v, _COMBO_THREATS)
                    or (n_counters >= 2
                        and any(_self_stack_has_card(v, card)
                                for card in _DD_PROTECT_ON_STACK)))
                if not fight:
                    hold_casts |= _COUNTER_SPELLS
            elif not _opponent_threat_on_stack(v):
                hold_casts |= _COUNTER_SPELLS
        # Targeted removal (Swords/Push/Prismatic) waits for a real threat:
        # power >= 2, or a NONTOKEN 1-power creature (an unflipped Delver /
        # Dragon's Rage Channeler is worth killing; a 1/1 token is not).
        if not _TARGETED_REMOVAL.isdisjoint(cast_cards):
            threat = any(p.power >= 2 or (p.power >= 1 and not p.token)
                         for p in _creatures(v.opp_perms))
            if not threat:
                hold_casts |= _TARGETED_REMOVAL
        # Wrath of the Skies (symmetric X sweeper) only into a board where the
        # best energy amount nets a real sweep: 2+ more of their permanents
        # destroyed than ours, or a 4+ destroyed-power differential.
        if _WRATH_OF_SKIES in cast_cards:
            _, count_net, power_net = _wrath_energy_plan(v)
            if not (count_net >= 2 or power_net >= 4):
                hold_casts.add(_WRATH_OF_SKIES)
        return hold_casts

    @staticmethod
    def _confirm(cats, confirm_cat: str) -> int:
        for i, c in enumerate(cats):
            if c == confirm_cat:
                return i
        return 0

    def _mulligan_choice(self, v) -> int:
        """Keep a hand with a sane land count; otherwise mulligan.

        A one-land hand is keepable if it contains a cheap card-selection spell
        (Brainstorm, Ponder, Once Upon a Time) that can dig toward more lands.
        Hands with 5 or more lands are too land-heavy to keep.

        Doomsday all-in: piloting a Doomsday deck against an opponent whose
        decklist runs no Force of Will (nothing can stop the resolved combo),
        mulligan aggressively for a turn-1-3 kill hand (_dd_fast_hand),
        keeping ANY hand rather than go below 6 cards. London mulligan means
        the hand at this decision is ALWAYS 7 fresh cards — the "never below
        6" floor counts mulligans already taken (1 mulligan -> keeping now
        bottoms to 6), via the per-seat _mull_counts. An unknown opponent
        list counts as running Force of Will (stay on the standard rule).
        """
        cats = v.cats
        hand = v.hand
        if self._deck_is_doomsday(v):
            has_fow = v.opp_runs_card("Force of Will", self._opp_deck(v))
            if has_fow is False:
                seat = bool(v.seat_is_a)
                mulls = self._mull_counts.get(seat, 0)
                keep = mulls >= 1 or _dd_fast_hand(v, hand)
                self._mull_counts[seat] = 0 if keep else mulls + 1
                for i, c in enumerate(cats):
                    if (c != _CAT_MULLIGAN) == keep:
                        return i
                return 0
        lands = sum(1 for c in hand if _is_land_card(v, c))
        has_dig = any(c in _KEEP_ONE_LANDERS for c in hand)
        # Each mulligan costs a card, so the bar drops as they add up: a
        # 5-lander is kept going to six, any hand with a land going to five,
        # and anything at all below that.
        mulls = v.mulligans
        keepable = (2 <= lands <= 4 or (lands == 1 and has_dig)
                    or (mulls >= 1 and lands == 5)
                    or (mulls >= 2 and lands >= 1)
                    or mulls >= 3)
        # Mulligan query: index 0 = keep, any other index = mulligan.
        if keepable:
            for i, c in enumerate(cats):
                if c != _CAT_MULLIGAN:
                    return i
            return 0  # only keep offered
        for i, c in enumerate(cats):
            if c == _CAT_MULLIGAN:
                return i
        return 0

    def _attack_choice(self, v) -> int:
        cats = v.cats
        opp_blockers = [p for p in _creatures(v.opp_perms) if not p.tapped]
        my_attackers = [p for p in _creatures(v.my_perms) if not p.tapped and not p.sick]
        total_power = sum(p.power for p in my_attackers)
        alpha = total_power > 0 and total_power >= v.opp_life
        library_empty = v.library == 0
        for i, c in enumerate(cats):
            if c != _CAT_SEL_ATK:
                continue
            if library_empty and v.card(i) in _ATTACK_DRAWERS:
                continue  # its attack trigger would draw from an empty library
            pt = _action_pt(v, i, v.my_perms,
                            exclude_attacking=True, require_untapped=True, require_unsick=True)
            if pt is _FILTERED:
                continue  # already attacking (offered to deselect) or unable to attack
            if pt is None or _should_attack(pt[0], pt[1], opp_blockers, alpha):
                return i  # attack (or attack by default if P/T unknown)
        return self._confirm(cats, _CAT_CONF_ATK)

    def _block_choice(self, v) -> int:
        cats = v.cats
        attackers = [p for p in _creatures(v.opp_perms) if p.attacking]
        if not attackers:
            return self._confirm(cats, _CAT_CONF_BLK)
        committed = sum(1 for p in _creatures(v.my_perms) if p.blocking)
        desired = _desired_block_count(v, attackers)
        blk = [i for i, c in enumerate(cats) if c == _CAT_SEL_BLK]
        if committed < desired and blk:
            # Commit the sturdiest available blocker (highest toughness, then power).
            def quality(i):
                pt = _action_pt(v, i, v.my_perms, exclude_blocking=True, require_untapped=True)
                if pt is _FILTERED:
                    return -1     # already blocking (offered to deselect): never preferred
                return (pt[1] * 10 + pt[0]) if pt else 0
            return max(blk, key=quality)
        return self._confirm(cats, _CAT_CONF_BLK)

    def _target_choice(self, v) -> int:
        """Removal → biggest threat; burn → face only when lethal-ish.

        Targets are classified by the action's zone ref: the opponent player
        (PLAYER_OPP), a "No target" slot / ourselves (NONE /
        PLAYER_SELF — never preferred), or an opposing object (its power read
        through the action's object). Burn (a damage source that
        can also hit the player) goes face unless it kills a creature worth
        the card — power 3+ (_BURN_DAMAGE) — since face burn is the faster
        clock (in the burn mirror, burning 2-power creatures loses more
        matches than going face)."""
        cats = v.cats
        pending = v.pending
        burn = (pending is not None and v.cards.deals_damage(pending)
                and any(cats[i] == _CAT_TARGET and v.zone(i) == _REF_PLAYER_OPP
                        for i in range(len(cats))))
        damage = _BURN_DAMAGE.get(pending, 3)
        opp_objects, player_targets = [], []
        for i, c in enumerate(cats):
            if c != _CAT_TARGET:
                continue
            zone = v.zone(i)
            if zone == _REF_PLAYER_OPP:
                player_targets.append(i)
                continue
            if zone in (_REF_NONE, _REF_PLAYER_SELF) or v.ctrl(i) == 1:
                continue  # no target / ourselves / our own object — avoid
            p = v.ref_perm(i)
            power = p.power if p is not None else 0
            if burn and (p is None or not p.creature or p.toughness > damage
                         or power < 3):
                continue  # not a creature this burn kills and should kill
            opp_objects.append((i, power))
        if player_targets and v.opp_life <= _BURN_FACE_LIFE:
            return player_targets[0]
        if opp_objects:
            return max(opp_objects, key=lambda x: x[1])[0]
        if player_targets:
            return player_targets[0]
        # Fallback: first non-self target, mirroring GREEDY.
        for i, c in enumerate(cats):
            if c == _CAT_TARGET and v.ctrl(i) != 1:
                return i
        for i, c in enumerate(cats):
            if c == _CAT_TARGET:
                return i
        return 0

    def _random_index(self, n: int) -> int:
        """A uniform index below ``n`` from the agent's own generator."""
        return self._rng.randrange(n)

    def _random_action(self, num_choices: int) -> int:
        """Uniform choice over the legal actions.

        Every offered action is legal, and a confirm action is always among the
        choices during mandatory attacker/blocker declaration, so uniform picking
        terminates with probability 1 (the engine also truncates as a backstop).
        """
        if num_choices <= 1:
            return 0
        return self._random_index(num_choices)

    def _explore_action(self, v) -> int:
        """Coverage fuzzer: a game-seed-deterministic, coverage-biased weighted choice.

        Goal: sweeping a range of seeds should touch as many distinct engine paths as
        possible — the opposite of GREEDY/HEURISTIC, which converge on one strong line
        per state. Three ingredients:

        * **Coverage bias.** Each legal action is weighted by its category
          (``_EXPLORE_WEIGHTS``) so the pick leans toward the actions that drive the
          most engine code — casting spells, activating abilities, attacking/blocking —
          while still passing often enough to advance the game. Same-category menus
          (targets, X values, modes, search/dig picks) get uniform weight, so *which*
          target/mode/X is taken varies freely across seeds — except a library
          search's "Fail to find" slot, which is de-weighted to
          ``_EXPLORE_SEARCH_FAIL_WEIGHT`` so tutors/fetches usually complete.

        * **Novelty bias.** A cast/activate option whose card this agent has not
          yet chosen this game gets its weight multiplied by
          ``_EXPLORE_NOVELTY_FACTOR``, so expensive cards with cheaper alternatives
          in the same deck still get cast somewhere in a seed sweep instead of losing
          the draw every time. Options with no card are non-novel (no
          boost). Chosen cast/activate cards are recorded in ``_explore_seen``,
          which is per game: ``new_game()`` clears it, and the pregame mulligan
          decision clears it too as a safety net for agent instances reused across
          games without a ``new_game()`` call.

        * **Determinism, tied to the GAME seed.** The weighted draw comes from Python's
          global ``random`` — the same source GREEDY breaks ties with — which
          ``runner.run_games`` (and the test harness) seed to the engine ``--seed``
          before each game (``random.seed(seed)``). So the whole explore trajectory is
          reproducible for a given seed, and a different seed reshuffles both the deck
          (a different view at every step) and this draw, branching into a
          different line. No separate per-agent seed; the only cross-call state is the
          per-game novelty set, itself a deterministic function of the trajectory.

        Not a strong opponent — combo lines are off (see the "explore" preset), so it
        fans out across cards instead of collapsing into a deck's known line.

        The ``explore:patient`` preset (``AgentConfig.explore_profile ==
        "patient"``) runs this same machinery with a big-mana weight profile —
        land drops near-mandatory, PASS boosted over cheap activations, CAST
        weight scaled by mana value — so expensive cards that plain explore never
        even sees as legal options get developed toward and hard-cast (see
        ``_EXPLORE_PATIENT_WEIGHTS``).
        """
        num_choices = v.n
        if num_choices <= 1:
            return 0
        cats = v.cats
        # A game always opens with a mulligan query, so seeing one means a new game
        # started: clear the novelty set even when nobody called new_game() (e.g. a
        # scripted opponent reused across vec-env episodes). Repeated clears during
        # London mulligans are harmless — nothing is cast pregame.
        if any(c == _CAT_MULLIGAN for c in cats):
            self._explore_seen.clear()
        # The "patient" profile swaps in the big-mana weight table, scales CAST
        # weight by mana value, and withholds the novelty boost from ACTIVATE
        # (see the _EXPLORE_PATIENT_* constants). The default profile takes the
        # exact same code path with the original table, so plain explore
        # behaviour (and its RNG consumption) is unchanged.
        patient = self.config.explore_profile == "patient"
        table = _EXPLORE_PATIENT_WEIGHTS if patient else _EXPLORE_WEIGHTS
        weights = []
        for i, c in enumerate(cats):
            w = table.get(c, _EXPLORE_DEFAULT_WEIGHT)
            # De-weight a search's "Fail to find" (the card-less slot) so the
            # tutor/fetch usually finds — see _EXPLORE_SEARCH_FAIL_WEIGHT.
            if c == _CAT_SEARCH and v.card(i) is None:
                w *= _EXPLORE_SEARCH_FAIL_WEIGHT
            if c in (_CAT_CAST, _CAT_ACTIVATE):
                card = v.card(i)
                if card is not None and card not in self._explore_seen:
                    if c == _CAT_CAST or not patient or _PATIENT_ACTIVATE_NOVELTY:
                        w *= _EXPLORE_NOVELTY_FACTOR
                if patient and card is not None:
                    if c == _CAT_CAST:
                        w *= 1.0 + _card_mana_value(v, card) * _PATIENT_CAST_MV_FACTOR
                    elif _hand_has_card(v, card):
                        # Card-consuming from-hand activation (cycling etc.):
                        # hold the card — see _PATIENT_HOLD_MV_FACTOR.
                        w /= 1.0 + _card_mana_value(v, card) * _PATIENT_HOLD_MV_FACTOR
            weights.append(w)
        # Mulligan is a 2-way [keep, mulligan] choice — bias toward keeping (index 0)
        # so most games actually start, but still mulligan ~1/4 of the time to exercise
        # the mulligan + London-bottoming paths.
        if num_choices == 2 and all(c == _CAT_MULLIGAN for c in cats):
            weights = [3.0, 1.0]
        if sum(weights) <= 0:
            weights = [1.0] * num_choices
        choice = int(random.choices(range(num_choices), weights=weights, k=1)[0])
        # Record the chosen card so this game's later draws treat it as non-novel.
        if cats[choice] in (_CAT_CAST, _CAT_ACTIVATE):
            card = v.card(choice)
            if card is not None:
                self._explore_seen.add(card)
        return choice
