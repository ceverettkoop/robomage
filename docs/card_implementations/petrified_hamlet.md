# Petrified Hamlet

```
Name:Petrified Hamlet
ManaCost:no cost
Types:Land
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigNameCard | TriggerDescription$ When this land enters, choose a land card name.
SVar:TrigNameCard:DB$ NameCard | Defined$ You | ValidCards$ Land | ValidDescription$ land card
S:Mode$ CantBeActivated | ValidCard$ Card.NamedCard | ValidSA$ Activated.!ManaAbility | Description$ Activated abilities of sources with the chosen name can't be activated unless they're mana abilities.
S:Mode$ Continuous | Affected$ Land.NamedCard | AddAbility$ ColorlessMana | Description$ Lands with the chosen name have "{T}: Add {C}."
SVar:ColorlessMana:AB$ Mana | Cost$ T | Produced$ C | SpellDescription$ Add {C}.
A:AB$ Mana | Cost$ T | Produced$ C | SpellDescription$ Add {C}.
```

Petrified Hamlet is a colorless land. On ETB its controller chooses a land card name; every
land with that name (any controller) gains `"{T}: Add {C}."` (CR 613.1f layer-6 ability grant).
It also has its own printed `{T}: Add {C}`.

Vocab index: **200** (`src/card_vocab.h`).

## What was already present

- The `T:Mode$ ChangesZone` self-ETB trigger and the `DB$ NameCard` effect category.
- `Mode$ Continuous` static parsing; the reusable `Affected$`-filter resolver
  `affected_permanents_for_static` (added for It That Heralds the End).
- The per-source named-card state `Permanent::chosen_name`, read by `match_named_card` statics
  (Disruptor Flute's RaiseCost / CantBeActivated, `active_raise_cost_for` /
  `rules_modifying.cpp`). The `CantBeActivated` line on this card therefore already worked.

## What was missing — the new mechanism

**General layer-6 `AddAbility$` continuous ability grant.** A `Mode$ Continuous` static can now
carry `AddAbility$ <SVar>`: the SVar's resolved `AB$` body is granted as a real activatable
ability to every permanent the `Affected$` filter designates, so the recipient's controller can
use it through the normal Permanent-abilities activation path.

### Files / functions

| Piece | Location |
|---|---|
| `StaticAbility::add_ability` (stores the resolved ability body) | `src/components/static_ability.h` |
| Parse `AddAbility$ <SVar>` (resolve SVar → body) | `src/parse.cpp` (next to `AddKeyword$`) |
| `parse_ability_body(body, type)` — public parser for one ability line | `src/parse.cpp` / `src/parse.h` |
| `Ability::granted_by_static` (source-static tag for de-dupe/removal) | `src/components/ability.h` |
| `StateManager::apply_layer6_ability_grants()` — **the grant mechanism** | `src/systems/state_manager_statics.cpp` |
| `ability_grant_targets()` / `strip_named_card_qualifier()` (NamedCard-aware affected set) | `src/systems/state_manager_statics.cpp` |
| Driver call (runs before the no-statics early-out) | `src/systems/state_manager_layers.cpp` |

### How the grant pass works (CR 613.1f, layer 6)

`apply_layer6_ability_grants()` runs every state-based-effects pass:

1. **Strip:** remove every ability marked `granted_by_static != 0` from all battlefield
   permanents. (Unlike keywords, `Permanent::abilities` is not rebuilt from base each pass, so
   the grant is rebuilt from scratch here.)
2. **Re-grant:** for each active `Continuous` static with a non-empty `add_ability` whose
   condition is met, resolve the affected permanents and attach a per-recipient copy of the
   parsed ability (tagged with the source static's entity). A `identical_activated_ability`
   check de-dupes so a land that already has `"{T}: Add {C}"` (e.g. Hamlet itself) is not given
   a duplicate.

It runs before the `g_active_statics.empty()` early-out so a grant left by a source that just
left the battlefield is cleaned up even when no statics remain. This makes the granted ability
appear/disappear as named lands and the granting source enter/leave the battlefield.

### NamedCard affected set

`Affected$ Land.NamedCard` is resolved by `ability_grant_targets`: it strips the `NamedCard`
qualifier (the shared filter evaluator has no NamedCard token), runs the rest of the filter
(`Land`, plus YouCtrl/etc.) through the reusable `affected_permanents_for_static`, then keeps
only permanents whose name equals the **source permanent's `chosen_name`** — the same per-source
named-card state `match_named_card` statics read, keeping grant and `CantBeActivated` consistent.

### Naming a land on ETB

`DB$ NameCard | Defined$ You | ValidCards$ Land` now routes through a new `defined_you` branch in
`effects::name_card` (`src/effects/effect_name_card.cpp`): the source's controller picks from the
land card names in their own deck (a new `only_lands` mode on `build_name_card_choices`), and the
choice is recorded on the **source permanent's `chosen_name`** (persistent), not the transient
global `cur_game.named_card` (which is cleared after the ability resolves). `ValidCards$` is now
parsed into `valid_cards_filter` for the `NameCard` category. The existing Cabal Therapy
`NameCard` path (opponent's nonland names → global) is unchanged.

## Tests (test_harness `--play`)

Mana abilities are auto-paid (hidden) in machine mode, so behavior was proven via a `{C}{C}` cost
(Kozilek's Command with X=0) that can only be paid by two colorless sources:

- **Named land gains `{T}: Add {C}`:** battlefield = Petrified Hamlet + Forest + Island; name
  Forest. Kozilek's Command `{X}{C}{C}` (X=0) cast **succeeds** — the second `{C}` can only come
  from the named Forest's granted ability (Hamlet supplies one `{C}`, Island none).
- **Non-named land does not get it:** name Forest with battlefield = Petrified Hamlet + 2 Islands.
  Kozilek's Command is **not offered** (Islands aren't named; only Hamlet makes `{C}`).
- **Grant removed when Hamlet leaves:** name Forest (2 Forests + Hamlet → 3 colorless), then
  opponent Wastelands Petrified Hamlet. After it is destroyed, Kozilek's Command is **no longer
  castable** — the Forests lost the granted `{C}`.
- **Hamlet's own `{T}: Add {C}` works:** battlefield = Petrified Hamlet + 4 Plains, name a card
  not on the field. Reality Smasher `{4}{C}` casts — the `{C}` pip comes from Hamlet's printed
  ability (Plains can't make `{C}` and weren't granted it).

No non-fatal errors and no unrecognized-ability-param warnings for Petrified Hamlet.

## Lingering uncertainty

The chosen name is stored per-source on `Permanent::chosen_name`, so multiple Petrified Hamlets
can name different lands independently (each grant keys off its own source's chosen name) — this
is correct, not an approximation. The candidate name set is restricted to the chooser's
own-deck land vocab cards (a decidable subset of "any land card," matching Disruptor Flute's
approach); naming a land that never appears in the chooser's deck is not offered.
