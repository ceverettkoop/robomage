# Delver.dk — Unimplemented Cards & Implementation Plan

## Status Summary

| Card | Qty | Status | Notes |
|---|---|---|---|
| Lightning Bolt | 4 | ✅ Done | DealDamage parsed |
| Mountain | 1 | ✅ Done | Basic land, mana injected |
| Island | 1 | ✅ Done | Basic land, mana injected |
| Volcanic Island | 4 | Done
| Thundering Falls | 1 | 🟡 Partial | Should tap for mana TODO + ETB-tapped + surveil trigger |
| Scalding Tarn | 4 | ✅  | Fetch land |
| Flooded Strand | 1 | ✅  | Fetch land |
| Polluted Delta | 1 | ✅  | Fetch land |
| Wooded Foothills | 1 | ✅  | Fetch land |
| Misty Rainforest | 1 | ✅  | Fetch land | fetches - implemented
| Wasteland | 4 | ✅ | Colorless mana + destroy nonbasic |
| Brainstorm | 4 | ❌ | Draw 3 + put 2 back |
| Ponder | 4 | ❌ | Rearrange top 3 + draw |
| Force of Will | 4 | ❌ | Counter spell + alt cost |
| Daze | 4 | ❌ | Counter unless + alt cost (return Island) |
| Unholy Heat | 3 | ❌ | 2 or 6 damage with Delirium |
| Dragon's Rage Channeler | 4 | ❌ | Surveil trigger + Delirium pump |
| Delver of Secrets | 3 | ❌ | Transform, upkeep trigger |
| Murktide Regent | 3 | ❌ | Delve, Flying, +1/+1 counters |
| Brazen Borrower | 1 | ❌ | Flash, Flying, Adventure |
| Mishra's Bauble | 4 | ❌ | 0-cost artifact, tap+sac, delayed draw |
| Cori-Steel Cutter | 3 | ❌ | updated cards folder to have this one- this is a doozy - tokens, triggers |

--cut surgical
--ignoring sideboard

---

## Implementation Strategy

Everything is driven by extending the parser to handle more script syntax. No card names appear in game logic — the ML model identifies cards only via `card_vocab.h`.

---

### Tier 1 — Mana Base (prerequisite for all games)

**Dual lands (`Volcanic Island`, `Thundering Falls`)**
- Already have both land subtypes in `Types:` field
- `state_manager.cpp` currently injects one mana ability per basic land subtype found
- Fix: ensure the injection loop continues after finding the first match (currently may break or return early)
- Result: both U and R mana abilities on the same permanent, derived from subtypes

**Fetch lands (`Scalding Tarn`, `Flooded Strand`, `Polluted Delta`, `Wooded Foothills`, `Misty Rainforest`)**

Script format:
```
A:AB$ ChangeZone | Cost$ T PayLife<1> Sac<1/CARDNAME> | Origin$ Library | Destination$ Battlefield | ChangeType$ Island,Mountain
```
- Parser extension: `AB$` prefix → `ACTIVATED` ability type; parse `Cost$` tokens (`T`, `PayLife<N>`, `Sac<N/X>`) into a cost struct; parse `ChangeType$` as comma-separated land subtypes
- New effect in `action_processor.cpp`: search library for lands matching `ChangeType$` subtypes, present as `OTHER_CHOICE` list, put chosen one onto battlefield, shuffle library
- Activated abilities offered as legal actions when permanent is untapped and cost is payable

**Wasteland**

Two `AB$` lines:
```
A:AB$ Mana | Cost$ T | Produced$ C
A:AB$ Destroy | ValidTgts$ Land.nonBasic | Cost$ T Sac<1/CARDNAME>
```
- Colorless mana: extend mana system with `Colors::COLORLESS`, parse `Produced$ C`
- Destroy nonbasic: new `Destroy` ability category in `action_processor.cpp`; filter targets by `ValidTgts$ Land.nonBasic` at target-selection time

---

### Tier 2 — Draw Spells

**Brainstorm**
```
A:SP$ Draw | NumCards$ 3 | SubAbility$ ChangeZoneDB
SVar:ChangeZoneDB:DB$ ChangeZone | Origin$ Hand | Destination$ Library | ChangeNum$ 2 | Reorder$ True
```
- Draw 3 cards, then mandatory `OTHER_CHOICE` to select 2 cards from hand to put back on top of library
- Parser: support `SubAbility$` chain references resolved via `SVar:` map; support `DB$` prefix as a continuation effect
- New effect categories: `Draw` (move top N library cards to hand), `PutOnLibrary` (move selected hand cards to top of library)

**Ponder**
```
A:SP$ RearrangeTopOfLibrary | NumCards$ 3 | MayShuffle$ True | SubAbility$ DBDraw
SVar:DBDraw:DB$ Draw | NumCards$ 1
```
- Reveal top 3 to active player, offer `OTHER_CHOICE` for ordering, optional shuffle (yes/no choice), then draw 1
- Parser: `MayShuffle$ True` → flag on ability

---

### Tier 3 — Counterspells

**Force of Will** and **Daze** share the same infrastructure.

Both have:
```
A:SP$ Counter | ValidTgts$ Card
```
- Extend parser to recognize `Counter` ability category targeting spells on the stack
- New effect: remove target from stack, move to `Destination$` (graveyard by default)

**Daze `UnlessCost$ 1`**: after counter would resolve, give target's controller a chance to pay 1 generic mana; if they pay, the counter spell fizzles.

**Alternative costs** (`S:Mode$ AlternativeCost`):
- At cast time, offer the alternative cost as a second `CAST_SPELL` legal action
- FoW alt cost: `PayLife<1> ExileFromHand<1/Card.Blue+Other>` → pay 1 life + select a blue card from hand to exile
- Daze alt cost: `Return<1/Island>` → select a basic Island you control, return it to hand

---

### Tier 4 — Keywords and Continuous Effects

**Flying** (`K:Flying`)
- Set `has_flying = true` on Creature component (parsed from `K:` lines)
- Modify `declare_blockers` in `action_processor.cpp`: creature without flying cannot block creature with flying

**Flash** (`K:Flash`)
- Set `has_flash = true` on CardData (parsed from `K:` lines)
- Modify `determine_legal_actions` in `state_manager.cpp`: allow casting at instant speed (outside main phase / with non-empty stack)

**Delirium** (used by Dragon's Rage Channeler, Unholy Heat)
- New helper: count distinct card types among cards in a player's graveyard
- `S:Mode$ Continuous | Condition$ Delirium` → conditional static ability; evaluated each SBA pass
- Unholy Heat `SVar:X:Count$Compare Y GE4.6.2` → if graveyard type count >= 4, deal 6; else deal 2

---

### Tier 5 — Creatures with Triggered Abilities

**Dragon's Rage Channeler**
```
T:Mode$ SpellCast | ValidCard$ Card.nonCreature
S:Mode$ Continuous | Condition$ Delirium | AddPower$ 2 | AddToughness$ 2 | AddKeyword$ Flying
S:Mode$ MustAttack | ValidCreature$ Card.Self | Condition$ Delirium
```
- Spell-cast trigger: when owner casts noncreature spell, surveil 1
- **Surveil**: look at top card of library; `OTHER_CHOICE`: keep on top or put in graveyard
- Continuous: +2/+2 + Flying while Delirium; must attack each combat while Delirium

**Murktide Regent**
```
K:Delve
K:Flying
K:etbCounter:P1P1:X:no Condition
T:Mode$ ChangesZone | ValidCard$ Instant.YouOwn,Sorcery.YouOwn | Origin$ Graveyard
```
- `K:Delve` → at cast time, offer `OTHER_CHOICE` to exile instants/sorceries from graveyard; each reduces generic cost by 1
- ETB: add +1/+1 counters equal to number of instants/sorceries exiled with it
- Zone-change trigger: whenever instant/sorcery leaves owner's graveyard while Murktide is on battlefield, add +1/+1 counter
- New: +1/+1 counter tracking on Creature component

**Delver of Secrets**
```
AlternateMode:DoubleFaced
T:Mode$ Phase | Phase$ Upkeep
```
- Two-faced card: entity can be in front or back state; back face (Insectile Aberration) has different P/T and abilities
- Upkeep trigger: peek top card of library; if instant or sorcery, transform (switch to back face)

**Brazen Borrower**
```
K:Flash
K:Flying
S:Mode$ CantBlockBy | ValidAttacker$ Creature.withoutFlying
AlternateMode:Adventure
```
- Flash and Flying keywords (see Tier 4)
- Block restriction: can only block creatures with flying; parsed from `S:Mode$ CantBlockBy`
- Adventure: two castable faces — creature and instant (Petty Theft); after adventure resolves, card goes to exile and can later be cast as a creature from exile

---

### Tier 6 — Complex / Low Priority

**Mishra's Bauble**
```
ManaCost:0
A:AB$ PeekAndReveal | Cost$ T Sac<1/CARDNAME>
SVar:DelTrigSlowtrip:DB$ DelayedTrigger | NextTurn$ True | Mode$ Phase | Phase$ Upkeep
```
- 0-cost casting (`ManaCost:0`)
- Activated ability: look at top card of target player's library (not revealed to that player)
- Register a one-shot delayed trigger for next upkeep to draw 1 card

**Surgical Extraction**
```
ManaCost:BP
```
- Phyrexian black: pay B or 2 life at cast time
- Complex multi-zone exile chain via `SubAbility$` chain
- Low priority (1-of sideboard)

**Cori-Steel Cutter** — not in cardsfolder
- Create `bin/resources/cardsfolder/c/coristeel_cutter.txt` in Forge script format
- Equipment type; requires equip activated ability framework and "equipped creature" modifier system
- Implement after basic activated ability parsing is in place

---

## Parser Extensions Summary

All behavior flows from these parser additions — nothing is keyed to card names:

| Script Syntax | Meaning | Implementation |
|---|---|---|
| `AB$` prefix | Activated ability | Parse `Cost$`, offer as legal action when untapped and cost payable |
| `K:Flying` | Flying keyword | Set flag on Creature; restrict block eligibility |
| `K:Flash` | Flash | Set flag on CardData; allow instant-speed casting |
| `K:Delve` | Delve | Offer graveyard exile as cost reduction at cast time |
| `T:Mode$ Phase` | Upkeep trigger | Register trigger checked each Upkeep step |
| `T:Mode$ SpellCast` | Cast trigger | Register trigger checked when spell is cast |
| `T:Mode$ ChangesZone` | Zone-change trigger | ETB / leaves-battlefield triggers |
| `S:Mode$ Continuous \| Condition$` | Conditional static ability | Evaluate condition each SBA pass |
| `DB$` sub-ability | Chained effect | Resolve as continuation after parent effect |
| `SVar:` | Named variable | Resolve sub-ability chains |
| `Cost$ T PayLife Sac` | Compound cost | Validate and pay each component |
| `Produced$ C` | Colorless mana | Add colorless to mana pool |
| `AlternateMode:DoubleFaced` | Transform card | Dual-state entity |
| `AlternateMode:Adventure` | Adventure card | Two castable faces |

---

## Recommended Build Order

1. **Dual land mana** — fix state_manager subtype loop → unblocks mana base
2. **`AB$` parsing + fetch lands + Wasteland** → unblocks the land package
3. **Flying keyword** → unblocks all fliers in combat
4. **Draw / PutOnLibrary effects** → Brainstorm, Ponder
5. **Counter spell + UnlessCost** → Force of Will, Daze (without alt cost first)
6. **Triggered ability framework** → prerequisite for DRC, Delver, Murktide
7. **Surveil + Delirium** → Dragon's Rage Channeler, Unholy Heat, Thundering Falls
8. **Alt cost system** → Force of Will, Daze full implementations
9. **Transform + +1/+1 counters + Delve** → Delver of Secrets, Murktide Regent
10. **Flash + Adventure + Equipment** → Brazen Borrower, Cori-Steel Cutter
11. **Delayed triggers** → Mishra's Bauble
12. **Phyrexian mana + multi-zone search** → Surgical Extraction
