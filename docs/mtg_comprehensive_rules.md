# MTG Comprehensive Rules — reference

The official **Magic: The Gathering Comprehensive Rules** are checked in next to this file as
[`mtg_comprehensive_rules.txt`](./mtg_comprehensive_rules.txt) (UTF-8 with BOM, CRLF line
endings, ~9.3k lines).

- **Version:** effective April 17, 2026 (`MagicCompRules 20260417`).
- **Source:** <https://magic.wizards.com/en/rules> → the `.txt` download
  (`https://media.wizards.com/2026/downloads/MagicCompRules%2020260417.txt`).

It is the ground truth for how a mechanic is *supposed* to behave, independent of the engine's
current implementation. **Consult it whenever implementing or testing a mechanic.**

## Reading it efficiently (don't load the whole file)

Rules are numbered and greppable; the table of contents at the top lists every rule title (so a
title grep hits the contents line first, then the rule itself).

| Section | Topic |
|---|---|
| 1xx | Game Concepts (mana, colors, objects, **111 tokens**, **117 timing & priority**, costs, damage, counters) |
| 2xx | Parts of a Card (name, mana cost, types, text, P/T, loyalty) |
| 3xx | Card Types (land, creature, artifact, enchantment, planeswalker, instant, sorcery, battle…) |
| 4xx | Zones (library, hand, battlefield, graveyard, stack, exile, command) |
| 5xx | Turn Structure (phases and steps; **combat 506–511**) |
| 6xx | Spells, Abilities, and Effects (601 casting, **603 triggered abilities**, 608 resolving, **613 layers**, 614–616 replacement/prevention) |
| 7xx | Additional Rules (701 keyword actions, **702 keyword abilities**, **704 state-based actions**, 707 copying, 712 double-faced cards) |
| 8xx | Multiplayer Rules (out of scope: the engine is two-player only) |
| 9xx | Casual Variants |
| — | **Glossary** (term and keyword definitions), then Credits |

### Lookup recipes

```bash
# A rule and its subrules (e.g. priority 117, combat damage 510):
grep -nE "^117\." docs/mtg_comprehensive_rules.txt
grep -nE "^510\.[0-9]" docs/mtg_comprehensive_rules.txt

# A keyword's rule section (702.x):
grep -nE "^702\.[0-9]+\. (Delve|Prowess|Flashback)" docs/mtg_comprehensive_rules.txt

# A glossary entry (term on its own line, definition below):
grep -nA3 "^Exalted" docs/mtg_comprehensive_rules.txt
```
