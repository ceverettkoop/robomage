# Blast Zone

## Oracle text
Blast Zone enters with a charge counter on it.

{T}: Add {C}.

{X}{X}, {T}: Put X charge counters on Blast Zone.

{3}, {T}, Sacrifice Blast Zone: Destroy each nonland permanent with mana value equal to the number
of charge counters on Blast Zone.

Land

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/b/blast_zone.txt`).
- **Key tags:**
  - `K:etbCounter:CHARGE:1` (enters with one charge counter).
  - `A:AB$ Mana | Cost$ T | Produced$ C`.
  - `A:AB$ PutCounter | Cost$ X X T | CounterType$ CHARGE | CounterNum$ X` (`SVar:X:Count$xPaid`).
  - `A:AB$ DestroyAll | Cost$ 3 T Sac<1/CARDNAME> | ValidCards$ Permanent.nonLand+cmcEQY`
    (`SVar:Y:Count$CardCounters.CHARGE`).

## Engine work
The mana / put-counter abilities already worked; the sacrifice-and-destroy needed the DestroyAll
handler to honour a `EQ` mana-value bound and a `nonLand` exclusion, and — because Blast Zone is
sacrificed as part of its own cost — to read its charge count from last-known information.

1. **cmc comparator + nonLand in DestroyAll** (`src/effects/effect_destroy_all.cpp`, CR 704/110.4a).
   The handler always applied `<=` and only recognized Artifact/Creature/Enchantment types
   (defaulting to "match every permanent", so a `Permanent.nonLand` filter would destroy lands too).
   Now it honours the parsed `cmc_op` (EQ for Blast Zone, LE for the legacy `cmcLEX` / Wrath-of-the-
   Skies path) via `apply_svar_op`, and excludes lands when the filter says `nonLand`.
   *Mechanics added (general): EQ/GE/LE mana-value bound and nonLand exclusion in DestroyAll.*

2. **Top-level DestroyAll `cmc<op>SVar` resolution** (`src/parse.cpp`). The `ValidCards$
   cmcEQY` → runtime `Count$` + comparator resolution lived only in the sub-ability parser
   (`parse_svar_ability`); Blast Zone's DestroyAll is a top-level `AB$` line, so its bound was never
   resolved. Extracted into a shared `resolve_destroyall_svars()` called from both the top-level
   (`parse_abilities`) and sub-ability paths.
   *Mechanics added (general): dynamic DestroyAll mana-value bound on activated (top-level) abilities.*

3. **Last-known counter count** (`src/classes/game.h`, `src/systems/orderer.cpp`,
   `src/components/ability.cpp`, CR 608.2h). `Count$CardCounters.CHARGE` on the DestroyAll's own
   source resolved to 0 because the source (Blast Zone) is sacrificed as part of the activation
   cost before the ability resolves. Added a typed-counter snapshot to `LastKnownInfo` (captured as
   a permanent leaves the battlefield) and a fallback in `evaluate_dynamic_amount`'s
   `Count$CardCounters` branch: when the source no longer has a `Permanent` component, read the
   last-known counter count.
   *Mechanics added (general): LKI counter counts for a source sacrificed as its own cost.*

## Behavioral decisions
- The `{X}{X}` counter pump and `{3}` destroy both tap Blast Zone, so they can't both be used in
  one turn (a single Blast Zone) — expected.
- The destroy is symmetric ("each nonland permanent"): it hits the controller's own qualifying
  permanents too.

## Tests
- Isolation (test_harness): preset Blast Zone (enters with 1 charge counter), plus a MV1 nonland
  (Birds of Paradise), a MV2 nonland (Grizzly Bears), and lands; activated the `{3},{T},Sac`
  ability.
  - Result (verified via a temporary debug print showing `cmc_bound=1 cmc_op=EQ nonland=1`):
    Birds of Paradise (MV1) destroyed; Grizzly Bears (MV2) survived; every land (Mountains,
    Forest) survived; Blast Zone went to the graveyard. Confirms the destroy is gated to MV
    **equal** to the charge count and never touches lands.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` (run once after all three cards).

## Result: implemented.
