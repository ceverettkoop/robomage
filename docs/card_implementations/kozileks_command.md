# Kozilek's Command  (vocab index 137)

## Oracle text
Choose two —
- Target player creates X 0/1 colorless Eldrazi Spawn creature tokens with "Sacrifice this creature: Add {C}."
- Target player scries X, then draws a card.
- Exile target creature with mana value X or less.
- Exile up to X target cards from graveyards.

(Kindred Instant — Eldrazi, mana cost {X}{C}{C}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/k/kozileks_command.txt`
- Key tags:
  - `A:SP$ Charm | Choices$ DBToken,DBScry,DBExile,DBExileGraveyard | CharmNum$ 2 | Announce$ X`
    — a modal **"choose two"** instant (CR 700.2, 601.2b). `CharmNum$ 2` was already parsed
    into `Ability::charm_num`; the charm handler now honors values > 1.
  - `SVar:X:Count$xPaid` — every mode's magnitude is the X paid for the spell's {X} cost.
    `has_x_cost` is auto-detected from the `{X}` in ManaCost, so the X prompt fires at cast
    time and records `cur_game.x_paid`. `Announce$ X` is therefore informational here
    (added to the ignored-param list).
  - `SVar:DBToken:DB$ Token | ValidTgts$ Player | TokenAmount$ X | TokenScript$ c_0_1_eldrazi_spawn_sac | TokenOwner$ TargetedPlayer`
    — mode 1: the **targeted player** creates **X** Eldrazi Spawn tokens. The token script
    (`bin/resources/tokenscripts/c_0_1_eldrazi_spawn_sac.txt`) already existed and carries the
    intrinsic `AB$ Mana | Cost$ Sac<1/CARDNAME> | Produced$ C` sac-for-mana ability.
  - `SVar:DBScry:DB$ Scry | ScryNum$ X | ValidTgts$ Player | SubAbility$ DBDraw` and
    `SVar:DBDraw:DB$ Draw | Defined$ ParentTarget` — mode 2: the **targeted player** scries X,
    then draws a card (CR 701.18).
  - `SVar:DBExile:DB$ ChangeZone | ValidTgts$ Creature.cmcLEX | Origin$ Battlefield | Destination$ Exile`
    — mode 3: exile a target creature with mana value ≤ X (the existing ChangeZone handler).
  - `SVar:DBExileGraveyard:DB$ ChangeZone | TargetMin$ 0 | TargetMax$ X | Origin$ Graveyard | Destination$ Exile | ValidTgts$ Card`
    — mode 4: exile up to X target cards from graveyards (the existing graveyard-exile path).

## Engine work
All changes are general, keyed on tag intent (no card-specific branches).

- **Charm "choose N" (`src/effects/effect_charm.cpp`).** The handler picked exactly one mode.
  It now loops `Ability::charm_num` times, removing each chosen mode from the menu before the
  next pick (CR 601.2b: chosen modes must be different) and resolving each chosen mode's
  targets + effect immediately (matching Forge's top-to-bottom resolution). A choose-one card
  (`charm_num` default 1) is unaffected. Modes with no legal target are still filtered out; if
  fewer than N legal modes remain the spell resolves with what it could choose.
  *(Simplification, shared by every Charm card in this engine: modes are chosen at resolution
  rather than on cast. Documented across the charm docs.)*

- **`Count$xPaid` as a dynamic amount (`src/components/ability.cpp::evaluate_dynamic_amount`,
  `src/parse.cpp`).** Added an `xPaid` branch to `evaluate_dynamic_amount` (returns
  `cur_game.x_paid`), and routed any amount SVar resolving to `Count$xPaid` into
  `dynamic_amount_expr` in `parse_svar_ability`. `TokenAmount$` and `ScryNum$` were added to the
  amount-key set so they feed `amount` / `dynamic_amount_expr` like `NumCards$`/`Amount$`.

- **X value restored at resolution (`src/systems/stack_manager.cpp`).** When an instant/sorcery
  resolves, `cur_game.x_paid` is restored from the spell's recorded `Spell::x_paid` before the
  ability resolves, so a `Count$xPaid` amount reads the value *this* spell was cast with even if a
  later X-cost spell overwrote the global in between.

- **`cmcLEX` target bound (`src/components/ability.cpp::is_legal_target`).** The existing
  `cmcLE<N>` parse did `std::stoi` on the suffix, which threw on the non-numeric `cmcLEX`. It now
  reads `cur_game.x_paid` when the bound is `X` (and trims trailing `.`/`+` qualifiers). This gates
  the exile-creature mode: a creature with mana value > X is not a legal target (and the mode is
  not offered if no creature qualifies).

- **`TargetMax$ X` cap (`src/components/ability.h`, `src/parse.cpp`, `src/action_processor.cpp`).**
  A count-SVar `TargetMax$` that resolves to `Count$xPaid` sets a new
  `Ability::target_max_from_xpaid` flag; `select_target`'s multi-target loop then clamps its
  iteration count to `cur_game.x_paid` so "up to X target cards" exiles at most X (other count-SVar
  caps keep the prior "any number" fallback).

- **Token amount + owner (`src/effects/effect_token.cpp`, `src/components/ability_params.h`).**
  The Token handler creates `amount` (or `dynamic_amount_expr`-evaluated) tokens instead of always
  one, and `TokenOwner$ TargetedPlayer` (new `TokenParams::owner_is_target`) routes the tokens to
  the spell's targeted player rather than the caster.

- **Scry effect (`src/effects/effect_scry.cpp`, `effect_kind.{h,cpp}`, `effect_table.cpp`,
  `effects.h`).** New general `Scry` handler: the targeted player (ValidTgts$ Player; else the
  source's controller) looks at the top N cards (N = `ScryNum$`, here `Count$xPaid`) and chooses
  per card to keep on top or put on the bottom (CR 701.18). Cards left on top keep their relative
  order (the optional reorder-among-kept is omitted as a minor simplification). The `SubAbility$
  DBDraw` chains afterward inheriting the parent's target, so the same scried player draws — which
  is exactly what `Defined$ ParentTarget` means; no special ParentTarget handling was needed.

- **Draw default count (`src/effects/effect_draw.cpp`).** A `DB$ Draw` with no `NumCards$` now
  draws 1 (Forge default) instead of 0, fixing the "then draws a card" rider. No implemented Draw
  card specifies `NumCards$ 0` or a dynamic draw count, so this only corrects the missing-count case.

## Behavioral decisions (made in lieu of asking a human)
- **Modes honored as written; no retag.** Each of the four modes resolves through the handler that
  matches its actual `DB$` category (Token / Scry / ChangeZone), with amounts driven by the script's
  `Count$xPaid`. Nothing was repurposed into a different category.
- **Choose-two with distinct modes (CR 601.2b).** Each chosen mode is removed before the next pick;
  a mode with no legal target is not offered (e.g. the exile-creature mode is hidden when no creature
  has mana value ≤ X). With fewer than two legal modes the spell resolves with whatever it could pick.
- **`cmcLEX` / `TargetMax$ X` read the X actually paid** (`cur_game.x_paid`), restored from the
  resolving spell so it is not corrupted by an intervening X-cost cast.

## Tests
- Isolation (test_harness, X = 2 via two Ancient Tombs):
  - **Token + Scry/Draw:** chose mode 1 (target Player A → **two** 0/1 Eldrazi Spawn tokens created
    under Player A's control) and mode 2 (target Player A → scry 2: kept one card on top, put one on
    bottom, then **drew 1** card — the kept Forest). No non-fatal error.
  - **Exile creature + Token:** opponent has Grizzly Bears (MV 2); all four modes offered; chose
    "Exile target creature with mana value X or less" (Bears → exile) and the token mode (two spawns
    for Player A). Bears left play; tokens entered.
  - **`cmcLEX` gating (negative):** with X = 1, the exile-creature mode is **not offered** against a
    MV-2 Grizzly Bears (only token / scry / graveyard-exile remain).
  - **Graveyard-exile (empty yards):** the "Exile up to X target cards from graveyards" mode (TargetMin
    0) resolves cleanly doing nothing, then the second chosen mode (scry/draw) resolves — no fizzle,
    no non-fatal error.
- Regression (test_harness --scripted, full games): a deck with 4× Kozilek's Command + Ancient Tomb +
  Eldrazi Temple + Grizzly Bears + Forests, mirror match, seeds 1–6 — all six decisive (3 A wins, 3 B
  wins), no draws, no non-fatal errors, no crashes. Kozilek's Command was cast and resolved (choosing
  two modes) by both players in real games. Only the pre-existing cosmetic `Unrecognized ability param`
  warnings for unrelated cards (Delver of Secrets, Mishra's Bauble, Brainstorm, Cori-Steel Cutter)
  appear; none for Kozilek's Command.

## Result
implemented
