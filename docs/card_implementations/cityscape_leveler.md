# Cityscape Leveler  (vocab index 287)

## Oracle text (per the in-repo Forge script)
Artifact Creature — Construct, mana cost `{8}`, 8/8, Trample.

When you cast this spell and whenever Cityscape Leveler attacks, destroy up to one target nonland
permanent. Its controller creates a tapped Powerstone token.

Unearth {8}.

> Note: this Forge script differs from the original *Brothers' War* printing (which is `{6}`, 6/6,
> "enters or attacks, destroy target nonland permanent an opponent controls. If you do, create a
> tapped Powerstone token"). Per the project rule the script is the source of truth — the engine
> honours these exact tags (cast-or-attacks; "up to one target nonland permanent"; the **destroyed
> permanent's controller** makes the Powerstone via `TokenOwner$ TargetedController`).

## Forge script
- Source: in-repo — `bin/resources/cardsfolder/c/cityscape_leveler.txt` (parsed as written, no retag).
- Key tags:
  - `K:Trample` (already supported).
  - `T:Mode$ SpellCast | ValidCard$ Card.Self | Execute$ TrigDestroy` — "when you cast this spell".
  - `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigDestroy | Secondary$ True` — "and whenever it attacks".
  - `SVar:TrigDestroy:DB$ Destroy | TargetMin$ 0 | TargetMax$ 1 | ValidTgts$ Permanent.nonland | SubAbility$ DBToken` — destroy up to one target nonland permanent.
  - `SVar:DBToken:DB$ Token | TokenTapped$ True | TokenScript$ c_a_powerstone | TokenOwner$ TargetedController` — its controller makes a tapped Powerstone.
  - `K:Unearth:8` — the new mechanic.
- Token: `bin/resources/tokenscripts/c_a_powerstone.txt` — a colorless `Artifact Powerstone` with
  `AB$ Mana | Cost$ T | Produced$ C | RestrictValid$ CantCastNonArtifactSpells` (the
  "can't be spent to cast a nonartifact spell" restriction is already supported).

## Rules (CR)
- **702.84 Unearth.** An activated ability of a card in a graveyard: pay the cost to return it to
  the battlefield; it gains haste; a delayed triggered ability exiles it at the beginning of the
  next end step; and if it would leave the battlefield it is exiled instead. Activate only as a
  sorcery.
- **603.7b** delayed triggered ability ("exile it at the beginning of the next end step").
- **603.6e / replacement** "if it would leave the battlefield, exile it instead."
- **702.19** Trample (pre-existing).
- **704.5f** zero-toughness state-based action (relevant to the noncreature-token bugfix below).

## Engine work
All mechanisms are general — a future card needs only `K:Unearth:<cost>` (Unearth) or
`TokenOwner$ TargetedController` (Powerstone routing).

### 1. Unearth (`K:Unearth:<cost>`) — graveyard-activated return with haste + delayed/redirected exile
The whole keyword is modeled as a synthetic graveyard-activated `ChangeZone`, reusing the existing
hand-activated-ability path (Cycling/Talon Gates) and the `Defined$ Self` ChangeZone path:
- **Parse** (`src/parse.cpp`): `K:Unearth:<cost>` synthesizes an `ACTIVATED` ability —
  `category=ChangeZone`, `activation_zone=Zone::GRAVEYARD`, `origin=Graveyard`,
  `destination=Battlefield`, `defined_self=true`, `sorcery_speed_only=true`, `is_unearth=true`
  (new flag on `Ability`, `src/components/ability.h`); the cost is parsed by the shared
  `parse_activation_cost`.
- **Legal action** (`src/systems/state_manager_actions.cpp`): a graveyard scan (mirroring the
  flashback/hand-activation scans) offers "Unearth <name>" as an `ACTIVATE_ABILITY` when the card
  is in the priority player's graveyard, the sorcery-speed window is open (main phase, own turn,
  empty stack), and the mana is payable.
- **Activation** (`src/action_processor.cpp`): the existing `ActivationZone$ Hand` branch of
  `process_activate_ability` was generalized to also accept `Zone::GRAVEYARD` — same flow (no
  targets, pay the mana, push the ability on the stack; the `defined_self` ChangeZone relocates the
  card itself, so the auto-consume-to-graveyard is skipped). Log line is zone-aware ("from
  graveyard").
- **Resolution → mark unearthed** (`src/effects/effect_change_zone.cpp`): when the `defined_self`
  ChangeZone lands the source on the battlefield and `is_unearth`, the card is added to a new
  one-shot set `Game::pending_unearthed` (`src/classes/game.h`).
- **Permanent finalize** (`src/systems/state_manager_statics.cpp`,
  `mark_unearthed_permanent`): when the `Permanent` is created and `pending_unearthed` is consumed,
  the helper (a) sets `Permanent::unearthed` (`src/components/permanent.h`), (b) grants haste
  (clears summoning sickness + adds the `Haste` keyword via `animate_added_keywords`, mirroring the
  earthbend-haste path), and (c) registers a one-shot **delayed triggered ability** that fires at
  `END_STEP_BEGAN` on the controller's end step and moves this exact permanent
  Battlefield → Exile (CR 603.7b).
- **Leaves-the-battlefield redirect** (`src/systems/orderer.cpp`, `Orderer::add_to_zone`): a small
  general redirect — any move of an `unearthed` permanent off the battlefield to a non-exile zone
  is rerouted to exile. Placed right after the replacement-effect dispatch, so the
  `CARD_CHANGED_ZONE` event reports the correct (exile) destination. The end-step delayed exile
  (already headed to exile) is a no-op for this redirect.

### 2. `TokenOwner$ TargetedController` — token routed to the target's controller
(`src/components/ability_params.h`, `src/effects/effect_token.cpp`.) New `TokenParams`
`owner_is_targeted_controller`, parsed by matching the exact value `"TargetedController"` *before*
the generic `"Targeted"` substring check. At resolution the token's owner/controller is the
**last-known controller** of `ab.target` (the just-destroyed permanent). The DBToken sub-ability
inherits the parent Destroy's target via the existing `bind_sub_target` (empty `Defined$` →
inherit). With no target chosen (the "up to one" destroy hit nothing), `ab.target == 0` and **no
token is created** — this implements the "if you do" gate.

### 3. Bugfix — noncreature tokens must not get a Creature component
(`src/components/token.cpp`, `bootstrap_token_components`.) Previously *every* token received a
`Creature`/`Damage` component and 0/0 base P/T, so the first noncreature token (the 0/0 Powerstone)
was destroyed by the zero-toughness SBA (CR 704.5f) the instant it entered. Now creature components
are bootstrapped only for tokens whose type line includes `Creature`. General fix for all
noncreature tokens (Powerstone/Treasure/Clue/Food/...).

## Tests
Isolation (`train/test_harness.py`, semantic `--play`, seed 1):
- **(a) Cast trigger → destroy + Powerstone.** Hand = Cityscape Leveler, 8 Mountains preset; opp
  has Grizzly Bears. On cast: trigger destroys Grizzly Bears; a **tapped** Powerstone token is
  created under **opp's** control (the destroyed permanent's controller, per
  `TokenOwner$ TargetedController`). Powerstone survives (no longer dies as a 0/0). **PASS.**
- **(b) Unearth.** Cityscape Leveler in A's graveyard, 8 Mountains preset. "Unearth Cityscape
  Leveler" is offered at sorcery speed; activating it returns it to the battlefield with haste —
  it **attacks the same turn** — and at the next end step it is **moved to exile** (not the
  graveyard). **PASS.**
- **(c) Attack trigger.** The unearthed/in-play Cityscape attacking fires the same
  destroy-up-to-one + Powerstone trigger (offering "No target" / a nonland-permanent target). **PASS.**
- **(d) Leaves-the-battlefield → exile.** Unearthed Cityscape attacks; opponent's Baleful Strix
  (deathtouch) blocks; Cityscape takes lethal deathtouch damage and is "destroyed", but is then
  "moved to exile" instead of going to the graveyard. **PASS.**

Regression (`train/test_harness.py --scripted`, seeds 1–3): a `4 Cityscape Leveler / 36 Mountain`
deck vs a mono-green deck. All three games finished **decisively** (A, A, A), **no draws**, no
non-fatal errors / asserts / tracebacks. (`train.py observe` not used — `torch` absent — so the
scripted regression ran directly through the harness, per CLAUDE.md.)

## Result
implemented
