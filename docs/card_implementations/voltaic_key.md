# Voltaic Key  (vocab index 250)

## Oracle text
{1}, {T}: Untap target artifact.

(Artifact, mana cost {1}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/v/voltaic_key.txt`
- Key tags:
  - `A:AB$ Untap | Cost$ 1 T | ValidTgts$ Artifact` — {1}{T}: untap target artifact.

## Engine work
- none — fully covered by existing handlers:
  - Targeted `AB$ Untap`: `src/effects/effect_untap.cpp` untaps `ab.target` (proven by Candelabra
    of Tawnos, idx 242).
  - `ValidTgts$ Artifact` target legality and the `{1}{T}` activation cost are handled by the
    generic activated-ability/targeting path.

## Behavioral decisions
- none — behavior unambiguous.

## Tests
- Isolation (test_harness): A controls Voltaic Key + Grim Monolith. Cast Grafdigger's Cage tapping
  Grim Monolith (→ "Grim Monolith (T)"); activated Voltaic Key ({1} from Grim Monolith's leftover
  floating mana, tap Voltaic Key) targeting Grim Monolith — "Player A's Voltaic Key ability
  targeting Grim Monolith is on the stack / Resolving ability (category: Untap) / Grim Monolith
  untaps". Final board "Voltaic Key (T) | Grim Monolith" (Voltaic Key tapped, Grim Monolith
  untapped). PASS.
- Regression (test_harness --scripted, full games): RG deck with 4× Voltaic Key + Grim Monolith vs
  a creature deck, seeds 1-2 — both decisive (2 B wins), no draws, no non-fatal errors.

## Result
implemented
