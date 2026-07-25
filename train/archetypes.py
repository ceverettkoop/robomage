"""Deck -> strategic-archetype metadata (Part 0 of the per-deck optimization plan).

Every deck the engine can load is tagged with one of a small number of *strategic
archetypes* (tempo, control, burn, ...) in ``bin/resources/decks/archetypes.json``.
The archetype pair of a matchup — (self archetype, opponent archetype) — indexes a
"value bucket", which later parts of the plan use to give the critic a per-matchup
final head so a burn race's value statistics stop fighting a control grind's.

Deliberately **stdlib-only** (no torch, no numpy) so the CLI spec, the TUI, and the
C++-side tooling can import it without paying for the ML stack.

Index order is a persisted contract: it is baked into checkpoints via the value-head
column layout and into the observation tail, so **append new archetypes at the end**
rather than reordering existing ones.
"""

import json
import os

# Strategic archetypes, in their canonical (index) order. APPEND ONLY — see above.
ARCHETYPES = [
    "tempo",          # 0
    "slow_midrange",  # 1
    "control",        # 2
    "burn",           # 3
    "lands",          # 4
    "doomsday",       # 5
    "reanimator",     # 6
]

# Fallback bucket for any deck not tagged in archetypes.json (ad-hoc / temp decks).
UNKNOWN = len(ARCHETYPES)
UNKNOWN_NAME = "unknown"

# Total archetype slots (the tagged ones plus 'unknown') and the number of
# (self archetype x opponent archetype) value buckets they span.
N_ARCH = UNKNOWN + 1
N_VALUE_BUCKETS = N_ARCH * N_ARCH

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DECKS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "decks")
ARCHETYPES_JSON = os.path.join(DECKS_DIR, "archetypes.json")

# Deck-spec prefix stripped when a caller hands us a path instead of a stem.
_DECKS_PREFIX = "resources/decks/"

_mapping_cache = None


def normalize_deck(deck):
    """Canonical decks/-relative stem for a deck spec, matching how the engine
    resolves ``--deck`` values (``decks/<stem>.dk``).

    Tolerates the spec variants callers pass around: a bare stem
    (``league/burn``), a trailing ``.dk``, OS-native separators, a leading
    ``./``, and a full/relative path through ``bin/resources/decks/``.
    Returns ``""`` for an empty/None spec.
    """
    if not deck:
        return ""
    stem = str(deck).strip().replace(os.sep, "/").replace("\\", "/")
    # A path through the decks dir (absolute or relative) -> keep the part below it.
    idx = stem.lower().rfind(_DECKS_PREFIX)
    if idx >= 0:
        stem = stem[idx + len(_DECKS_PREFIX):]
    while stem.startswith("./"):
        stem = stem[2:]
    stem = stem.lstrip("/")
    if stem.lower().endswith(".dk"):
        stem = stem[:-3]
    return stem.strip("/").lower()


def mapping():
    """The deck-stem -> archetype-name mapping from archetypes.json (cached).

    Keys are normalized stems; ``_``-prefixed keys (comments) are skipped. An
    unrecognized archetype name in the file is a loud error — a typo there would
    otherwise silently demote a deck to the 'unknown' bucket.
    """
    global _mapping_cache
    if _mapping_cache is not None:
        return _mapping_cache
    raw = {}
    if os.path.isfile(ARCHETYPES_JSON):
        with open(ARCHETYPES_JSON) as f:
            raw = json.load(f)
    out = {}
    for deck, arch in raw.items():
        if deck.startswith("_"):
            continue
        if arch not in ARCHETYPES:
            raise ValueError(
                f"{ARCHETYPES_JSON}: deck {deck!r} is tagged with unknown "
                f"archetype {arch!r}. Valid archetypes: {ARCHETYPES}")
        out[normalize_deck(deck)] = arch
    _mapping_cache = out
    return out


def arch_name(deck):
    """Archetype NAME for a deck spec, or ``'unknown'`` when untagged."""
    return mapping().get(normalize_deck(deck), UNKNOWN_NAME)


def arch_index(deck):
    """Archetype INDEX for a deck spec: ``ARCHETYPES`` position, else ``UNKNOWN``."""
    arch = mapping().get(normalize_deck(deck))
    return UNKNOWN if arch is None else ARCHETYPES.index(arch)


def bucket_index(self_deck, opp_deck):
    """Value-bucket index for a matchup: ``arch_index(self) * N_ARCH + arch_index(opp)``."""
    return arch_index(self_deck) * N_ARCH + arch_index(opp_deck)


def arch_name_at(index):
    """Archetype NAME for an archetype INDEX (``UNKNOWN`` -> ``'unknown'``)."""
    i = int(index)
    return ARCHETYPES[i] if 0 <= i < len(ARCHETYPES) else UNKNOWN_NAME


def bucket_archetypes(bucket):
    """Split a value-bucket index back into ``(self archetype, opp archetype)`` indices."""
    b = int(bucket)
    return b // N_ARCH, b % N_ARCH


def bucket_name(bucket):
    """Human/TensorBoard-friendly label for a value bucket: ``'burn_vs_control'``.

    The canonical tag for per-bucket diagnostics (training logs, baseline
    reports) so every report names a bucket the same way.
    """
    self_arch, opp_arch = bucket_archetypes(bucket)
    return f"{arch_name_at(self_arch)}_vs_{arch_name_at(opp_arch)}"


def decks_for_archetype(arch):
    """Every tagged deck stem belonging to ``arch`` (name or index), sorted."""
    name = ARCHETYPES[arch] if isinstance(arch, int) else arch
    return sorted(d for d, a in mapping().items() if a == name)


def untagged(decks):
    """The subset of ``decks`` with no archetypes.json entry (normalized stems)."""
    m = mapping()
    return [d for d in decks if normalize_deck(d) not in m]


def validate_roster(decks, context="league"):
    """Raise unless every deck in ``decks`` is tagged in archetypes.json.

    League/roster decks drive the value-bucket split, so an untagged one would
    quietly train in the 'unknown' bucket alongside every ad-hoc deck. Fail loudly
    at startup instead.
    """
    missing = untagged(decks)
    if missing:
        raise ValueError(
            f"{context}: deck(s) {missing} have no archetype in {ARCHETYPES_JSON}. "
            f"Tag each one with an archetype from {ARCHETYPES} (add an entry "
            f'"<stem>": "<archetype>", stems relative to decks/).')
