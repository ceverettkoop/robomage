# Eldrazi Temple  (vocab index 130)

## Oracle text
{T}: Add {C}.
{T}: Add {C}{C}. Spend this mana only to cast colorless Eldrazi spells or activate abilities of colorless Eldrazi.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/e/eldrazi_temple.txt`
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ C` — unrestricted {C} mana ability.
  - `A:AB$ Mana | Cost$ T | Produced$ C | Amount$ 2 | RestrictValid$ Spell.Eldrazi+Colorless,Activated.Eldrazi+Colorless+inZoneBattlefield`
    — the restricted {C}{C} ability (spend only on colorless Eldrazi spells / abilities of colorless Eldrazi).

## Engine work
- **Restricted-mana flag (general).** Added `Ability::restrict_to_colorless_eldrazi`
  (`src/components/ability.h`) and a parser branch in `src/parse.cpp` `RestrictValid` handling:
  a value containing both `Eldrazi` and `Colorless` sets the flag. This slots into the existing
  restricted-mana machinery (Cavern of Souls' `restrict_to_chosen_type_creature`, Abundant
  Countryside's `restrict_to_creature`). The trailing `Activated.Eldrazi+Colorless…` clause
  (abilities of colorless Eldrazi) is folded into the same flag; the engine's restricted-mana
  filter only fires for spell payment (`paid_for` = the spell being cast), which is the
  observable case, so no separate activated-ability path was needed.
- **Match predicate (general).** Added `colorless_eldrazi_restricted_mana_matches(paid_for)` in
  `src/mana_system.cpp`: true iff `paid_for` is a card with subtype Eldrazi and is colorless
  (no colored mana symbol and no explicit color override other than COLORLESS). Wired into all
  three restricted-mana filter sites: `collect_mana_legal_actions`, `can_afford_with_sources`,
  and `auto_pay_mana` (mirroring the two existing restricted-mana cases).
- **Devoid keyword (general, CR 702.114a).** Eldrazi Linebreaker — the only Eldrazi in the
  vocab and the natural test creature — is printed with a red mana symbol but is colorless via
  Devoid. Without Devoid the engine treated it as red, so the "colorless Eldrazi" filter would
  never match. Added a `K:Devoid` branch in `src/parse.cpp`'s keyword loop that sets
  `explicit_colors = {COLORLESS}` (the engine's existing "this card is colorless" marker, used
  by `color_identity_from`). This is a correct general implementation of the keyword, not a
  card-specific hack.
- **Multi-ability single-tap payment (general fix).** Eldrazi Temple is the first permanent
  with two same-color mana abilities of differing yield ({C} amount 1 vs {C}{C} amount 2).
  A permanent taps only once, so the payer must pick the larger applicable ability. Two spots
  were corrected in `src/mana_system.cpp`:
  - `can_afford_with_sources` first pass now counts the **maximum** amount among an entity's
    free same-color abilities (was: first ability only), so affordability reflects the best
    single activation.
  - `auto_pay_mana` now dedups same-entity/same-color abilities in `valid_sources`, keeping the
    highest-yield one, so the greedy payer realizes the affordability the gating check promised
    instead of burning the single tap on the smaller ability.
- Mechanics added (general, not card-specific): a third restricted-mana category
  (colorless-Eldrazi), the Devoid color keyword, and correct handling of permanents with
  multiple same-color mana abilities of different amounts.

## Behavioral decisions (made in lieu of asking a human)
- **"Colorless" test** uses the card's actual color (no colored mana symbols, no non-colorless
  explicit color), consistent with CR 105.2a (an object with no colored mana symbols and not
  defined as a color is colorless) and CR 702.114a (Devoid). Subtype Eldrazi is required in
  addition.
- **Restricted mana scope** is enforced at spell payment (CR 106.7 mana spending
  restrictions). The "activate abilities of colorless Eldrazi" half of the restriction is
  carried by the same flag but is only exercised when paying for a spell in the current engine;
  no colorless-Eldrazi activated ability with a mana cost exists in the vocab to exercise the
  other half, so it is unobservable today and left folded into the one flag rather than adding
  an unused code path.
- Otherwise behavior is unambiguous (a basic restricted mana ability).

## Tests
- Isolation (test_harness):
  - **Unrestricted {C}, any spell:** battlefield 1 Eldrazi Temple; cast Chalice of the Void
    (non-Eldrazi) → "Player A activated Eldrazi Temple for 1(C)", Chalice cast — the
    unrestricted ability pays a non-Eldrazi spell's generic, and the restricted {C}{C} is
    **not** used.
  - **Restricted {C}{C} pays a colorless Eldrazi:** battlefield 1 Eldrazi Temple + 1 Mountain;
    cast Eldrazi Linebreaker ({1}{C}{R}, Devoid→colorless Eldrazi) → "Player A activated Eldrazi
    Temple for 2(C)" + Mountain for {R}, Linebreaker enters. The restricted ability is required
    here (only 1 Temple), proving it is allowed for colorless Eldrazi.
  - **Restricted {C}{C} blocked for non-Eldrazi:** the Chalice case above shows the Temple
    taps for only 1(C) (the unrestricted ability) when paying a non-Eldrazi spell — the {C}{C}
    restricted ability is filtered out for the non-Eldrazi payee.
- Regression (observe via test_harness, scripted): mirror match of a deck running
  4 Eldrazi Temple / 4 Eldrazi Linebreaker / 4 Lightning Bolt / 4 Chalice of the Void /
  4 Mishra's Bauble / 40 Mountain across seeds 1–6 → every game a decisive winner, no draws,
  no non-fatal errors, no crashes; Eldrazi Temple repeatedly tapped for 2(C) to cast Eldrazi
  Linebreaker. (Lightning Strike was omitted from the test deck because it independently
  corrupts deck loading — a pre-existing issue unrelated to this card.)

## Result
implemented
