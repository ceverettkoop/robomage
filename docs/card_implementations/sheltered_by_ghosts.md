# Sheltered by Ghosts  (vocab index 281)

## Oracle text
Enchantment — Aura. Mana cost {1}{W}. Enchant creature you control.

- When Sheltered by Ghosts enters, exile target nonland permanent an opponent controls until
  Sheltered by Ghosts leaves the battlefield.
- Enchanted creature gets +1/+0 and has lifelink and ward {2}.

## Forge script
- Source: in-repo — `bin/resources/cardsfolder/s/sheltered_by_ghosts.txt` (parsed as written,
  no retag).
- Type `Enchantment Aura`, mana cost `1 W`.
- Key tags:
  - `K:Enchant:Creature.YouCtrl:creature you control` — the Aura's enchant restriction.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
    Execute$ TrigExile` with
    `SVar:TrigExile:DB$ ChangeZone | Origin$ Battlefield | Destination$ Exile |
    ValidTgts$ Permanent.nonLand+OppCtrl | Duration$ UntilHostLeavesPlay` — the ETB exile.
  - `S:Mode$ Continuous | Affected$ Creature.EnchantedBy | AddPower$ 1 |
    AddKeyword$ Lifelink & Ward:2` — the static buff.

Note: the Oracle "exile up to one target" is implemented from the actual Forge script, which uses
a **required** target (`ValidTgts$ Permanent.nonLand+OppCtrl`, no `TargetMin$ 0`). With no legal
target the triggered ability simply does nothing (fizzles); the Aura still resolves and buffs.

## Rules (CR)
- **303** Auras: **303.4** an Aura spell requires a target it can enchant (601.2c), and on
  resolution it enters attached to that object (**303.4f/g**). **704.5n** — an Aura attached to an
  illegal object, or not attached to anything, is put into its owner's graveyard.
- **702.21** lifelink (damage dealt also causes its controller to gain that much life).
- **702.21x** ward — "whenever this permanent becomes the target of a spell or ability an opponent
  controls, counter it unless that player pays the ward cost" ({2} here).
- **611 / 613** layers — the +1/+0 (layer 7c) and the lifelink/ward grants (layer 6) come from the
  Aura's continuous static and are recomputed each SBA pass for the enchanted creature.
- Exile-until-leaves is a linked one-shot return (**the host's leave returns the exiled card**),
  handled by the pre-existing `UntilHostLeavesPlay` mechanism (Cloak and Dagger wave).

## Engine work
The ETB exile/return (`Duration$ UntilHostLeavesPlay`, `register_exile_until_host_leaves` in
`src/effects/effect_change_zone.cpp`), the granted ward {2} (`collect_ward_instances` already
reads granted `Ward:N` keywords, `src/action_processor.cpp`), and granted lifelink
(`source_has_keyword` over the effective keyword list, `src/components/damage.cpp`) were all
already in place. The genuine gap was **Aura attachment itself**: Sheltered by Ghosts is the first
Aura in the vocab, and the engine had no path to (a) target the enchanted object at cast, (b)
attach the resolved Aura, (c) resolve `Affected$ Creature.EnchantedBy` statics, or (d) fall the
Aura off when its host leaves. Implemented generally (no card-specific code), reusing the existing
equipment attachment link (`Permanent::equipped_to`):

1. **Parse the Enchant keyword** (`src/parse.cpp`): `K:Enchant:<ValidTgts>[:prompt]` →
   `CardData::enchant_filter` (`src/components/carddata.h`). The middle field is a standard target
   filter (`Creature.YouCtrl`).
2. **Cast-time targeting** (`src/action_processor.cpp`, CAST_SPELL): an Aura with an
   `enchant_filter` and no spell ability of its own builds a transient targeting `Ability` from the
   filter, selects the enchant target via the existing `select_target`, and records it in a new
   one-shot map `Game::pending_aura_target` ({aura → enchanted object}).
3. **Cast legality** (`src/systems/state_manager_actions.cpp`): an Aura is only offered if a legal
   object matching its `enchant_filter` exists (reuses `has_legal_targets`), per CR 601.2c.
4. **Attach on resolution** (`src/systems/state_manager_statics.cpp`, `apply_permanent_components`):
   when the Aura's `Permanent` is created, `pending_aura_target` sets `perm.equipped_to` to the
   enchanted object — reusing the equipment attachment field.
5. **`Affected$ Creature.EnchantedBy` statics** (`src/systems/state_manager_statics.cpp`): a shared
   helper `affected_is_attached_target()` now treats both `EquippedBy` (equipment) and
   `EnchantedBy` (Auras) as the single-target "attached creature" form, resolving to `equipped_to`.
   All three appliers (layer-6 keyword grant, layer-7b set-P/T, layer-7c additive P/T) and
   `affected_is_general_filter()` route through it, so the +1/+0, lifelink, and ward {2} land on
   the enchanted creature.
6. **Aura fall-off SBA** (`src/systems/state_manager.cpp`, 704.5n): a battlefield Aura
   (`enchant_filter` non-empty) whose `equipped_to` is 0 / no longer a battlefield permanent / no
   longer a creature is put into its owner's graveyard. When the host dies, the Aura leaves, which
   (via the linked `UntilHostLeavesPlay` return) brings the exiled permanent back.

**General:** any future Aura with `K:Enchant:<filter>` and an `Affected$ ...EnchantedBy` static is
now handled with no additional code.

## Behavioral decisions (made in lieu of asking a human)
- **704.5n "illegal object" scope.** The fall-off SBA checks the structural part of legality (no
  attachment / host left the battlefield / host is no longer a creature) — the common case (host
  dies or is bounced). A continuous re-check of the *full* enchant filter (e.g. losing control of a
  creature enchanted by "enchant creature **you control**") is not re-evaluated; in two-player it
  is a rare edge and out of scope for this card. Noted as a follow-up.
- **No-target ETB.** The exile trigger uses a required target; with no opponent nonland permanent it
  fizzles (existing trigger behavior) and the Aura still resolves and attaches — matches the Oracle
  "up to one target" net effect.

## Tests
Isolation (`train/test_harness.py`, seed 1, `--play` specs):
- **(Attach + buffs + exile).** A casts Sheltered by Ghosts on its Grizzly Bears, exiling the
  opponent's Grizzly Bears: "Sheltered by Ghosts is attached to Grizzly Bears", "gains 1/0 Lifelink
  & Ward:2", board **[3/2]**, opponent's Bears moved to exile. **PASS.**
- **(Lifelink unblocked).** The enchanted **3/2** attacks unblocked: "deals 3 damage to Player B",
  "Player A gains 3 life (lifelink)", life 20→23 / 20→17. **PASS.**
- **(Lifelink blocked — lethal-only).** Blocked by a 2/2: attacker assigns lethal **2**, gains
  **2** (correct — no trample). **PASS.**
- **(Ward {2}).** Opponent's Lightning Bolt targeting the enchanted creature triggers
  "Ward {2}: … may pay to counter", demanding payment. **PASS.**
- **(Aura leaves → exiled returns).** The enchanted creature dies in combat → "Sheltered by Ghosts
  is put into the graveyard (Aura not attached to a legal object)" → "Grizzly Bears is moved to the
  battlefield" (the exiled permanent returns to its owner). **PASS.**
- **(No exile target).** Opponent controls only lands: the Aura still casts, attaches, and buffs
  ([3/2], Lifelink & Ward:2); the ETB exile "ChangeZone fizzles". **PASS.**

Regression:
- Equipment still works after the `EquippedBy`→shared-helper refactor: Shadowspear equipped to a
  Grizzly Bears → "gains 1/1 Trample & Lifelink", **[3/3]**. **PASS.**
- `train/test_harness.py --scripted` delver vs mav, seeds 1/3/5/7: all decisive, no draws, no
  non-fatal errors. (`train.py observe` not used — `torch` absent — per CLAUDE.md.)

## Result
implemented
