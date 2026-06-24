# MTG Comprehensive Rules — reference

The official **Magic: The Gathering Comprehensive Rules** are stored alongside this file as
[`mtg_comprehensive_rules.txt`](./mtg_comprehensive_rules.txt) (plain UTF-8 text, ~9.3k lines).

- **Version:** effective April 17, 2026 (file `MagicCompRules 20260417` from Wizards of the Coast).
- **Source:** <https://magic.wizards.com/en/rules> → the `.txt` download
  (`https://media.wizards.com/2026/downloads/MagicCompRules%2020260417.txt`).

This is the authoritative rules text. **Consult it whenever implementing or testing a game
mechanic** — it is the ground truth for how a mechanic is *supposed* to behave, independent of
how the current engine happens to implement it.

## How to read it efficiently (don't read the whole file)

The text is organized into numbered rules. Rule numbers are stable and greppable, so look up the
relevant section by number instead of loading the whole file. The top of the file contains a
table of contents listing every section.

Top-level sections:

| Section | Topic |
|---|---|
| 1xx | Game Concepts (mana, colors, objects, permanents, spells, abilities, targets, **117 timing & priority**, costs, damage, counters) |
| 2xx | Parts of a Card (name, mana cost, types, text, P/T, loyalty) |
| 3xx | Card Types (land, creature, artifact, enchantment, planeswalker, instant, sorcery, battle…) |
| 4xx | Zones (library, hand, battlefield, graveyard, stack, exile, command) |
| 5xx | Turn Structure (phases, steps; **5xx combat** — declare attackers/blockers, damage) |
| 6xx | Spells, Abilities, and Effects (casting, activating, resolving, replacement effects) |
| 7xx | Additional Rules (state-based actions 704, triggered abilities 603, layers 613, tokens, copying) |
| 8xx | Multiplayer Rules |
| 9xx | Casual Variants |
| — | **Glossary** (keyword & term definitions — e.g. Delve, Prowess, Flashback, Exalted) then Credits |

### Lookup recipes

```bash
# Find a rule by number (e.g. priority):
grep -nE "^117\." docs/mtg_comprehensive_rules.txt

# Read a specific rule and its subrules (e.g. combat damage 510):
grep -nE "^510\.[0-9]" docs/mtg_comprehensive_rules.txt

# Look up a keyword/mechanic in the glossary or its rule section:
grep -niE "^702\.[0-9]+\. (Delve|Prowess|Flashback)" docs/mtg_comprehensive_rules.txt
grep -ni "exalted" docs/mtg_comprehensive_rules.txt | head
```

Keyword abilities are defined in **rule 702** (with the short definition repeated in the
Glossary). State-based actions are **704**. The interaction of continuous effects (the "layers")
is **613**.
