# Paradox Engine (vocab index 272)

## Oracle text
Whenever you cast a spell, untap all nonland permanents you control.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/p/paradox_engine.txt`)

Key tags:
- `Types:Legendary Artifact`, `ManaCost:5`
- `T:Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ You | TriggerZones$ Battlefield | Execute$ TrigUntapAll | ...`
  — a plain, unfiltered "whenever you cast a spell" trigger.
- `SVar:TrigUntapAll:DB$ UntapAll | ValidCards$ Permanent.YouCtrl+nonLand` — the untap-all effect.

## Engine work
Two gaps, both filled with general handlers (no Paradox-Engine special-casing):

1. **Plain `Mode$ SpellCast` trigger binding** (`src/parse.cpp`). The SpellCast parse block had
   bindings only for the *filtered* variants (noncreature/colorless/spell-count/cmc/self); an
   unfiltered `ValidCard$ Card` SpellCast trigger fell through with `trigger_on == 0` and never
   fired. Added a general fallback after the specific branches: when `mode_is_spell_cast` and no
   specialized binding matched, bind `trigger_on = Events::SPELL_CAST` and gate on
   `trigger_valid_player_is_controller = valid_player_is_you` (`ValidActivatingPlayer$ You`).
   The existing general battlefield trigger scan (`state_manager_triggers.cpp`) already fires
   SPELL_CAST triggers and applies the controller gate via the event's `PLAYER` param, so with no
   extra `ValidCard$` filter the trigger fires on every spell the source's controller casts.

2. **`UntapAll` effect** (`src/effects/effect_untap_all.cpp`, new). Mirrors the other `*All`
   effects (`pump_all`, `destroy_all`): scans the battlefield via `is_battlefield_permanent`,
   matches each permanent against `ValidCards$` through the shared `permanent_matches_filter`
   (so `Permanent.YouCtrl+nonLand` — controller scoping and the `nonLand` negation — is honored
   generally), and clears `Permanent::is_tapped`. The `ValidCards$` key is consumed into
   `ab.valid_cards_filter` by the shared `parse_destroy_all` hook (no new parse hook needed).
   Registered as `EffectKind::UntapAll` in `effect_kind.{h,cpp}`, dispatched in
   `effect_table.cpp`, declared in `effects.h`.

## Rules basis
- CR 603 — triggered abilities; "whenever you cast a spell" fires on each qualifying cast and is
  put on the stack above the spell that triggered it.
- CR 701.21 — untap.
- The filter is matched relative to the ability's controller, so "nonland permanents you control"
  (`Permanent.YouCtrl+nonLand`) is correct in the 2-player engine.

## Behavioral decisions
- Modeled as a generic mass-untap keyed on the script's real `ValidCards$` filter rather than a
  Paradox-Engine-specific untap, so any future "untap all <filter>" card reuses it.
- Lands (and the opponent's permanents) are excluded purely by the filter — no hardcoded
  type/controller check at the call site.

## Limitations / notes
None affecting Paradox Engine. The trigger fires above the triggering spell on the stack, so the
untap resolves before the spell does (standard, and irrelevant to the untap result).

## Tests
Verified via `train/test_harness.py` (`--play`), seed 1:

Board: Player A controls Paradox Engine + Grim Monolith + Forest; Player B controls a Grizzly
Bears. A casts Grizzly Bears, paying with Forest (G) and Grim Monolith (C) — leaving both the
land and the mana rock tapped. On the cast:
- "Paradox Engine triggered" → "Grim Monolith untaps".
- Post-resolution board: `Paradox Engine | Grim Monolith | Forest (T)` — the **nonland** Grim
  Monolith untapped; the **land** Forest stayed tapped (the `nonLand` filter), and the
  opponent's permanents were unaffected (`YouCtrl`). PASS.

No draws, no fatal/non-fatal errors.

## Result
Done — registered in vocab (272), `train/card_costs.py` regenerated, clean build, scenario
verified.
