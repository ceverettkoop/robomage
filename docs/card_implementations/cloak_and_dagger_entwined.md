# Cloak and Dagger, Entwined  (vocab index 268)

## Oracle text
Deathtouch, lifelink

When Cloak and Dagger enter, choose target opponent and up to one target creature they control.
They reveal their hand. You may exile a nonland card from their hand or the chosen creature until
Cloak and Dagger leave the battlefield.

## Forge script
- Source: provided — `bin/resources/cardsfolder/c/cloak_and_dagger_entwined.txt`
- Type: `Legendary Creature Human Hero`, mana cost `1 W B`, 2/2, keywords Deathtouch + Lifelink.
- Key tags (an ETB `ChangesZone` trigger chaining four sub-abilities):
  - `SVar:TrigRevealHand:DB$ RevealHand | ValidTgts$ Opponent | RememberRevealed$ True | SubAbility$ DBPump`
  - `SVar:DBPump:DB$ Pump | ValidTgts$ Creature.ControlledBy ParentTarget | TargetMin$ 0 | TargetMax$ 1 | RememberPumped$ True | SubAbility$ DBChangeZone`
  - `SVar:DBChangeZone:DB$ ChangeZone | Origin$ Hand,Battlefield | Destination$ Exile | Defined$ Remembered | ChangeNum$ 1 | Optional$ True | Duration$ UntilHostLeavesPlay | SubAbility$ DBCleanup`
  - `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True`

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.

## Engine work
The headline deliverable is a **general CR 603.6e linked exile-and-return mechanic**
(`Duration$ UntilHostLeavesPlay`), built to be reused by Sheltered by Ghosts and Static Prison.
Five pieces, each keyed on the tag's intended meaning:

1. **`Duration$ UntilHostLeavesPlay` parse** (`src/parse.cpp`). A new `Duration$` branch sets
   `Ability::duration_until_host_leaves` (the existing `Duration$ Permanent` Animate branch is
   untouched).

2. **The linked exile-and-return** (`src/effects/effect_change_zone.cpp`,
   `register_exile_until_host_leaves(host, card, origin)`). When a `ChangeZone | Destination$ Exile`
   with `duration_until_host_leaves` moves a card, this helper (a) records the card on the host's
   `Permanent::exiled_with` (so the last-known-info snapshot and "cards exiled with this" readers
   still see it) and (b) registers a **delayed trigger watching the host's departure from the
   battlefield** (the existing `DelayedTrigger { fire_on_leave_battlefield, watch_entity }`
   mechanism — the same one earthbend uses). The fire ability is a `ChangeZone` of that one card
   from Exile back to its recorded **origin** (HAND → owner's hand; BATTLEFIELD → battlefield under
   owner's control), seeded via `restore_remembered_exiled_with = {card}`. One trigger is
   registered per exiled card, so each carries its own origin even when a host exiles several. This
   is wired into BOTH:
   - the **targeted** ChangeZone path (a single target permanent — what Sheltered by Ghosts /
     Static Prison use; origin BATTLEFIELD), and
   - the **choose-from-remembered** path below (Cloak's hand-or-creature pick).

3. **Bounded/optional `Defined$ Remembered` exile** (`src/effects/effect_change_zone.cpp`). The
   blanket `Defined$ Remembered` move (Ajani — moves *every* remembered object) was unchanged; a new
   branch fires only when an explicit count/optionality is present
   (`ChangeNum$ N` → `change_num >= 0`, or `Optional$ True` → `optional_choice`). It presents a
   **choice** of which remembered object(s) to move: candidates are the remembered objects currently
   in an eligible `Origin$` zone (Hand or Battlefield), **nonland only** (honoring the oracle "a
   nonland card from their hand or the chosen creature" — the creature is itself nonland), and
   `Optional$ True` adds an "Exile nothing" decline.

4. **`RememberRevealed$ True`** (`src/effects/effect_reveal_hand.cpp`). The reveal now starts a fresh
   remembered set and fills it with the revealed hand — the candidate pool a later
   `Defined$ Remembered` exile picks from. (Previously `RememberRevealed` was an ignored param.)

5. **`RememberPumped$ True` + optional opponent-controlled Pump-as-selector**
   (`src/effects/effect_pump.cpp`). This Pump applies no stat change; it is purely an (optional)
   target-selector. The handler now filters to the opponent's creatures for
   `ValidTgts$ Creature.ControlledBy ParentTarget` (in the two-player engine the parent's "target
   opponent" is always the source's single opponent — CLAUDE.md two-player scope), offers a "Choose
   no creature" option when `TargetMin$ 0`, and on `RememberPumped$ True` **appends** the chosen
   creature to the remembered candidate pool.

`ValidTgtsDesc$` (cosmetic target-prompt prose) was added to the parser's ignored-params set.

## Behavioral decisions (made in lieu of asking a human)
- **The return is a delayed triggered ability** (CR 603.6e): the exiled card comes back when the
  host leaves the battlefield, going on the stack and resolving as a normal triggered ability. A
  battlefield permanent returns as a **fresh object** under its owner's control (re-entry triggers
  and summoning sickness apply); a hand card returns to its owner's hand.
- **Return under the owner's control** (not the host's controller): the fire ability's `source` is
  the returning card, so `owner` resolves from the card's `Zone.owner`.
- **Nonland candidate filter** comes from the oracle ("a nonland card … or the chosen creature");
  the script carries no `ValidCards$`, so it is applied in the choose-from-remembered branch (which
  is Cloak-specific — the reused single-permanent exilers go through the targeted path and are
  unaffected).
- **Two-player scope** (CLAUDE.md): "target opponent" and `ControlledBy ParentTarget` resolve to the
  single opponent of the source's controller.

## Tests
Isolation (`train/test_harness.py`, Cloak cast from A's hand; B has a creature + nonland cards):
- **(a) Exile the creature, then host leaves → creature returns to the battlefield.** Exiled
  `Grizzly Bears` (the chosen creature) leaving B's battlefield; B then Lightning Bolts Cloak →
  `Cloak and Dagger, Entwined is destroyed` → `Delayed trigger fires.` →
  `Grizzly Bears is moved to the battlefield`. Final board: Grizzly Bears back on **B's** battlefield
  (owner's control, summoning-sick fresh object). PASS.
- **(b) Exile a nonland hand card, then host leaves → card returns to hand.** Exiled `Lightning Bolt`
  from B's hand (B hand count 7→6); B kills Cloak with a second bolt →
  `Delayed trigger fires.` → `Lightning Bolt is moved to hand` (back in **B's** hand). PASS.
- **(c) Decline to exile** ("Exile nothing"): nothing exiled, resolution continues, no crash. PASS.
- The choose-from-remembered menu correctly offered all candidates together:
  `[0] Exile Lightning Bolt (from hand)  [1] Exile Grizzly Bears (the chosen creature)  [2] Exile nothing`.

Regression (`train/test_harness.py --scripted`, seeds 1–3): deck with 2× Cloak + Grizzly Bears /
Lightning Bolt / Plains / Swamp vs a Grizzly/Bolt/Forest/Mountain deck. All three games finished
decisively (A wins 1, B wins 2), no draws, no fatal/non-fatal errors. The scripted agent cast Cloak,
revealed the opponent's hand, and exiled a battlefield creature in real games. Build is clean (no
new `WARNING: Unrecognized ability param`). Temp decks cleaned up.

## Result
implemented
