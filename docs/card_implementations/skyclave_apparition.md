# Skyclave Apparition (vocab index 188)

## Oracle text
When Skyclave Apparition enters, exile up to one target nonland, nontoken permanent you don't
control with mana value 4 or less.
When Skyclave Apparition leaves the battlefield, the exiled card's owner creates an X/X blue
Illusion creature token, where X is the mana value of the exiled card.

`1 W W` — a 2/2 Kor Spirit.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/s/skyclave_apparition.txt`). Parsed as
written — no retag. Key tags:
- ETB: `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
  Execute$ TrigExile`
- `SVar:TrigExile:DB$ ChangeZone | TargetMin$ 0 | TargetMax$ 1 |
  ValidTgts$ Permanent.nonLand+!token+YouDontCtrl+cmcLE4 | Origin$ Battlefield |
  Destination$ Exile | RememberChanged$ True`
- LTB: `T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Any | ValidCard$ Card.Self |
  Execute$ TrigToken`
- `SVar:TrigToken:DB$ Token | TokenScript$ u_x_x_illusion | TokenOwner$ RememberedOwner |
  ConditionDefined$ Remembered | ConditionPresent$ Card.ExiledWithSource | TokenPower$ X |
  TokenToughness$ X | SubAbility$ DBCleanup`
- `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True`
- `SVar:X:Remembered$CardManaCost`

The token script `bin/resources/tokenscripts/u_x_x_illusion.txt` (blue Illusion, `*/*`) pre-exists.

## Engine work
The two triggers are standard ChangesZone triggers (ETB `Origin$ Any → Battlefield`, LTB
`Origin$ Battlefield → Any` self-trigger). The supporting framework already existed; four new
general pieces were added, plus reuse of several existing ones.

### Reused (pre-existing)
- **Up-to-one optional targeting** (`TargetMin$ 0 | TargetMax$ 1`): the optional-target path
  (`target_min == 0` ⇒ "No target" offered, zero picked is legal) already existed (Endurance's
  "up to one target player", the phases-out trigger). The ETB exile gets a "No target" /
  single-target menu for free.
- **Targeted `DB$ ChangeZone` Battlefield→Exile**: the targeted-move branch of
  `effects::change_zone` already moves a chosen permanent to exile and records it in the source
  permanent's `Permanent::exiled_with` list (Keen-Eyed Curator). 
- **`Remembered$CardManaCost`**: `evaluate_dynamic_amount` already returns the mana value of the
  first remembered card (`src/components/ability.cpp`, Birthing Ritual).
- **Leaves-the-battlefield self-trigger**: the `Card.Self` LTB ChangesZone look-back
  (`src/systems/state_manager_triggers.cpp`) already fires a self-trigger after the source has
  left (Flagstones / Thought-Knot Seer).
- **`DB$ Cleanup | ClearRemembered$ True`** and the `u_x_x_illusion` token script.

### New (general)
1. **Target-filter qualifiers `YouDontCtrl` and `!token`/`nonToken`** in `Ability::is_legal_target`
   (`src/components/ability.cpp`). `YouDontCtrl` is the explicit Forge spelling of "you don't
   control" and restricts the same way as `OppCtrl`; `!token`/`nonToken` reject a token permanent
   (`token` requires one). `nonLand` (via `type_set_passes_nontype`) and `cmcLE4` already worked.
   General over any target spec using these qualifiers.
2. **`RememberChanged$ True` in the targeted ChangeZone branch** (`effects::change_zone`,
   `src/effects/effect_change_zone.cpp`): the exiled target is now pushed into
   `cur_game.remembered_entities` (the self-move and search branches already did this), so the
   paired SVar / LTB ability can read the exiled card.
3. **`exiled_with` carried in last-known info** (`LastKnownInfo::exiled_with` in
   `src/classes/game.h`, populated in `src/systems/orderer.cpp`): when a permanent leaves the
   battlefield its `Permanent::exiled_with` is snapshotted before the SBA strips the Permanent
   component. The self-LTB trigger then copies this list onto the fired Ability
   (`Ability::restore_remembered_exiled_with`, `src/systems/state_manager_triggers.cpp`), and
   `Ability::resolve` restores it into `cur_game.remembered_entities` before any gate/SVar runs
   (CR 608.2h). This is what lets the LTB token's `Remembered$CardManaCost`, `RememberedOwner`,
   and `ExiledWithSource` gate all find the exiled card across the two separate trigger resolutions.
4. **`TokenOwner$ RememberedOwner` + `TokenPower$`/`TokenToughness$` from an SVar** in
   `effects::token` (`src/effects/effect_token.cpp`, params in `src/components/ability_params.h`,
   SVar resolution in `src/parse.cpp`'s sub-ability finalize): `RememberedOwner` makes the token
   owned and controlled by the **owner of the first remembered (exiled) card**; the
   `TokenPower$`/`TokenToughness$` SVar expressions (here `X = Remembered$CardManaCost`) override
   the script's `*/*` so the token enters as an `(MV)/(MV)`. Both are general (any
   `TokenOwner$ RememberedOwner` / X/X-from-SVar token-maker reuses them).
5. **`ConditionDefined$ Remembered | ConditionPresent$ Card.ExiledWithSource` gate** in
   `evaluate_present_condition` (`src/systems/state_manager_actions.cpp`): instead of merely
   counting remembered cards, the `Card.ExiledWithSource` form counts only remembered cards that
   are **still in the exile zone**, so the token is made only if the exiled card is still exiled.

## Behavioral decisions (CR)
- "exile up to one target …" is optional targeting (CR 115 / the `TargetMin$ 0` path): casting/
  ETB with no legal target, or choosing the "No target" option, exiles nothing and remembers
  nothing — so the later LTB makes no token.
- The token's controller/owner is the **owner of the exiled card** (`RememberedOwner`), not
  Skyclave's controller. Exiling your opponent's 3-MV creature and then having Skyclave die hands
  the opponent a 3/3 Illusion (CR 707 / the card text).
- Token size is the exiled card's mana value (CR 112.7, read from its card object). A 0-MV exiled
  permanent makes a 0/0 token that dies immediately to the zero-toughness SBA (CR 704.5f) — correct.
- If the exiled card has already left exile when Skyclave leaves, the `ExiledWithSource` condition
  fails and no token is created.
- `TriggerDescription$`/`SpellDescription$`/`TgtPrompt$` are cosmetic (ignored).

## Tests (test_harness.py, semantic `--play`, seed 1)
- **(a) ETB exile target legality** — A cast Skyclave; opponent battlefield had Endurance (3 MV
  creature), Forest (land), Reality Smasher (5 MV creature); A's own Birds of Paradise was in
  play. The ETB target menu offered **only** `No target` and `Endurance` — the land, the MV-5
  creature, and A's own permanent were **not** legal targets. **Pass.**
- **(b) exile + LTB token** — A exiled Endurance (`Endurance is moved to exile`), then bolted its
  own Skyclave. On Skyclave's death the LTB trigger resolved and created a **3/3 Illusion Token**
  on **Player B's** battlefield (Endurance's owner) — owner/controller = the opponent, P/T = MV 3.
  **Pass.**
- **(c) up-to-one, no legal target** — opponent had only a Forest + Reality Smasher (MV 5); the
  ETB menu offered **only `No target`**, nothing was exiled, and when Skyclave died the LTB trigger
  resolved but **created no token** (ExiledWithSource gate unmet). **Pass.**
- **(d) 0-MV exile** — A exiled Lotus Petal (0 MV); on Skyclave's death a **0/0 Illusion Token**
  was created and **died immediately** to the zero-toughness SBA. **Pass.**
- **Regression** — scripted full games `temp/skyclave_a` (4 Skyclave Apparition + Plains) vs
  `temp/skyclave_b` (Birds of Paradise / Lotus Petal / Collector Ouphe / Lightning Bolt / Forest),
  seeds 1/2/3 (and 5/7): all decisive (no draws), zero non-fatal errors/asserts. Skyclave casts
  and resolves cleanly in real games (the scripted agent declining the optional exile target is
  agent suboptimality, not an engine bug — the exile/token path is proven by the isolation tests).

## Result
Implemented. New reusable mechanics: `YouDontCtrl`/`!token` target-filter qualifiers;
`RememberChanged$` in the targeted exile path; `exiled_with` last-known-info snapshot +
`Ability::restore_remembered_exiled_with` restore; `TokenOwner$ RememberedOwner` and
`TokenPower$`/`TokenToughness$`-from-SVar token sizing; and the `ExiledWithSource` remembered
condition gate.
