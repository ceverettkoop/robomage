# Sire of Seven Deaths (vocab index 167)

## Oracle text
First strike, vigilance
Menace, trample
Reach, lifelink
Ward—Pay 7 life.

(Ward—Pay 7 life: "Whenever this creature becomes the target of a spell or ability an
opponent controls, counter it unless that player pays 7 life.")

7-mana colorless 7/7 Eldrazi creature.

## Forge script (Source: pre-existing local — `bin/resources/cardsfolder/s/sire_of_seven_deaths.txt`)
Key tags:
- `ManaCost:7`, `Types:Creature Eldrazi`, `PT:7/7`
- `K:First Strike`, `K:Vigilance`, `K:Menace`, `K:Trample`, `K:Reach`, `K:Lifelink`
  — all six combat keywords were already handled by the engine.
- `K:Ward:PayLife<7>` — the new mechanic.

## Engine work (general pay-life Ward cost)
Ward already existed but only parsed a numeric **mana** cost (`std::stoi(kw_line.substr(colon+1))`),
so `PayLife<7>` made the numeric parse fail. The fix generalizes the Ward unless-cost to be
either mana or pay-life, honoring the script's real `K:Ward:PayLife<N>` tag (no retag):

- `src/components/carddata.h` — added `bool ward_is_life` alongside the existing `ward_cost`.
- `src/parse.cpp` (keyword loop) — `K:Ward:PayLife<N>` sets `ward_cost = N` and
  `ward_is_life = true`; any other `K:Ward:N` arg stays a numeric mana cost (unchanged path).
- `src/components/ability.h` — added `bool unless_cost_is_life` on `Ability` so a Ward `Counter`
  trigger can carry "pay this as life" through to resolution.
- `src/action_processor.cpp` (`trigger_ward_for_targets`) — copies `ward_is_life` onto the pushed
  Ward `Counter` ability's `unless_cost_is_life`; log line reads "Ward—Pay N life" vs "Ward {N}".
- `src/components/ability.cpp` (`run_unless_loop`) — added a `pay_as_life` parameter and a life
  branch: offers "Pay N life" only when the payer's life total >= N (CR 119.4), subtracts N life
  on payment, otherwise the spell/ability is countered. The existing mana branch is unchanged.
- `src/effects/effect_counter.cpp` — the Ward `Counter` effect passes `ab.unless_cost_is_life`
  into `run_unless_loop`, reusing the same "counter unless pay" (PAY_UNLESS) machinery.

The representation is general (a ward cost that is mana **or** pay-life), not a Sire-only hack;
any future `K:Ward:PayLife<N>` card works automatically.

## Behavioral decisions
- **CR 702.21a** — Ward is a triggered ability: "Whenever this permanent becomes the target of a
  spell or ability an opponent controls, counter that spell or ability unless that player pays
  [cost]." The trigger fires only for **opponent-controlled** spells/abilities (enforced by the
  existing `is_battlefield_permanent(tgt, opp)` controller check in `trigger_ward_for_targets`).
- **CR 119.4** — a player can pay life only if their life total is greater than or equal to the
  amount; you can't pay more life than you have. The pay-life branch gates the "Pay N life"
  option on `life_total >= N`. (CR 119.4b's "always pay 0 life" is moot here — `ward_cost > 0`.)

## Tests (`train/test_harness.py`, inline hands/battlefield + semantic `--play`)
- **(a) pay branch** — B casts Lightning Bolt at A's Sire; B chooses "Pay 7 life": B 20 → 13,
  Bolt resolves (Sire takes 3 damage, Bolt to graveyard). Confirmed.
- **(a) decline branch** — same setup, B chooses "Don't pay": Lightning Bolt is countered (to
  graveyard), B's life unchanged. Confirmed.
- **(b) own-spell** — A casts Lightning Bolt at A's **own** Sire: no Ward trigger fires, Bolt
  resolves normally (3 damage). Ward correctly fires only for opponents. Confirmed.
- **(c) combat keywords** — Sire attacks unblocked: deals 7, A gains 7 life (lifelink), attacker
  shows "(ATK)" and stays untapped (vigilance). Parsing all six K: lines did not break. Confirmed.
- **Regression** — scripted full games, Sire deck vs Lightning Bolt deck, seeds 1/2/3: each game
  ends with a decisive winner, no draws, no non-fatal errors / assertions.

## Result
Implemented. General pay-life Ward cost; clean `make HEADLESS=TRUE` build; all scenarios pass.
