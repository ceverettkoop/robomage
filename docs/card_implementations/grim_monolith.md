# Grim Monolith  (vocab index 248)

## Oracle text
This artifact doesn't untap during your untap step.
{T}: Add {C}{C}{C}.
{4}: Untap this artifact.

(Artifact, mana cost {2}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/g/grim_monolith.txt`
- Key tags:
  - `R:Event$ Untap | ValidCard$ Card.Self | ValidStepTurnToController$ You | Layer$ CantHappen`
    — self-referential "doesn't untap during your untap step" replacement (CR 614.1d).
  - `A:AB$ Mana | Cost$ T | Produced$ C | Amount$ 3` — {T}: add {C}{C}{C}.
  - `A:AB$ Untap | Cost$ 4` — {4}: untap this artifact (UNtargeted self-untap).

## Engine work
Two small, general gaps were filled (neither card-specific):

- **Untargeted `AB$ Untap` falls back to its source** (`src/effects/effect_untap.cpp`). The handler
  only untapped `ab.target`/`ab.targets`; an `AB$ Untap` with no `ValidTgts$` (Grim Monolith's
  "{4}: Untap this artifact.") left both empty and untapped nothing. Now, when no target is
  present, it untaps `ab.source`. Targeted untaps (Voltaic Key, Candelabra of Tawnos) are
  unchanged.
- **Self-referential `SKIP_UNTAP` replacement** (`src/parse.cpp` + `src/systems/replacement_effects.cpp`).
  The untap-prevention parse only built a `SKIP_UNTAP` replacement for a bare land-subtype
  `ValidCard$` (Choke: `ValidCard$ Island`). Added a parse branch for `event_is_untap &&
  layer_cant_happen && valid_card_self` (`ValidCard$ Card.Self`) that emits a `SKIP_UNTAP` with
  `applies_to_self_only = true`. The application side (`replacement_effects.cpp`) now matches
  `applies_to_self_only` by checking the source IS the permanent being untapped (`e == ev.entity`),
  alongside the existing subtype-filtered Choke path. `ValidStepTurnToController$ You` is implicit
  in a 2-player game (a permanent only untaps during its controller's untap step), so it needs no
  extra gating. Reused by any future "this permanent doesn't untap" artifact/permanent.
- Mechanics added (general): untargeted-Untap self-fallback; self-referential SKIP_UNTAP
  replacement.

## Behavioral decisions
- The `{T}: Add {C}{C}{C}` mana ability's `Amount$ 3` was already honored by the generic mana
  producer (`eval_mana_amount`); no change needed.
- `ValidStepTurnToController$ You` not parsed (cosmetic in 2-player scope) — see above.

## Tests
- Isolation (test_harness):
  - **Mana ability:** paying the {4} ability tapped Grim Monolith for mana — log "Player A
    activated Grim Monolith for 3(C)" confirms `Amount$ 3` colorless. Also paid a {1} spell from
    Grim Monolith alone. PASS.
  - **{4} untap (self):** activating "{4}: Untap" resolved "Resolving ability (category: Untap) /
    Grim Monolith untaps" — the untargeted untap acted on its source (verifies the new fallback).
    PASS.
  - **Doesn't untap during your untap step:** cast Grafdigger's Cage tapping Grim Monolith (turn 0);
    on TURN 2 (A's own turn, after A's untap step) the board still showed "Grim Monolith (T)" — it
    did NOT untap. PASS. (It correctly also stayed tapped through the opponent's untap step.)
- Regression (test_harness --scripted, full games): green/red Grim Monolith deck vs a creature deck,
  seeds 1-3 — all decisive (3 A wins; one ended by opponent decking), no draws, no non-fatal errors.

## Result
implemented
