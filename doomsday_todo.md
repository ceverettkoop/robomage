# Doomsday Deck — Missing Cards & Engine TODO

## Context

The `doomsday.dk` deck has 18 cards not in the vocab. All 18 have card scripts in `bin/resources/cardsfolder/`. The scripts reference several ability categories, keywords, SVar expressions, and cost types that the C++ engine does not yet implement. This plan catalogs every gap.

## Cards Not in Vocab

All need entries in `src/card_vocab.h` (indices 53+) and a `train/card_costs.py` regen:

| # | Card | Script exists? |
|---|------|---------------|

| 2 | Thoughtseize | Yes | DONE untested
| 17 | Duress | Yes | DONE untested
| 1 | Doomsday | Yes | done untested
| 3 | Dark Ritual | Yes | untested
| 4 | Lotus Petal | Yes | untested
| 5 | Lion's Eye Diamond | Yes | untested
| 6 | Thassa's Oracle | Yes | untested
| 7 | Personal Tutor | Yes |
| 8 | Street Wraith | Yes |
| 9 | Edge of Autumn | Yes |
| 16 | Consider | Yes | may work?
| 18 | Deep Analysis | Yes |
| 15 | Cavern of Souls | Yes |




## Cards That Work Today (no new engine features needed)

These cards use only abilities/keywords already implemented:

- **Swamp** — basic land, mana injected by `apply_land_abilities`
- **Underground Sea** — dual land (Island Swamp), mana injected by subtypes
- **Bloodstained Mire** — fetchland, uses `AB$ ChangeZone` + `Sac`/`PayLife` costs (all implemented)
- **Verdant Catacombs** — same pattern as Bloodstained Mire
- **Ponder** — already in vocab and working
- **Brainstorm** — already in vocab and working
- **Force of Will** — already in vocab and working
- **Daze** — already in vocab and working
- **Polluted Delta** — already in vocab and working
- **Flooded Strand** — already in vocab and working
- **Misty Rainforest** — already in vocab and working
- **Island** — already in vocab and working
- **Personal Tutor** — `SP$ ChangeZone` from Library to Library top. Should work if `LibraryPosition$` is handled (need to verify)

## Engine Gaps — Unimplemented Ability Categories

### 1. `Discard` (Thoughtseize, Duress)
- **What:** Target player reveals hand, caster chooses a card matching filter, that card is discarded
- **Script fields:** `Mode$ RevealYouChoose`, `DiscardValid$ Card.nonLand` / `Card.nonCreature+nonLand`, `NumCards$ 1`
- **Needed:** New resolve() branch in `ability.cpp` that: shows opponent's hand, filters by DiscardValid, presents choice to caster, moves chosen card to graveyard
- **Also used by:** Lion's Eye Diamond cost (`Discard<0/Hand>` — discard entire hand as activation cost)

### 2. `LoseLife` (Doomsday, Thoughtseize sub-ability)
- **What:** Controller loses N life
- **Script fields:** `LifeAmount$ 2` or `LifeAmount$ Y` (SVar)
- **Needed:** New resolve() branch that decrements player life total

### 3. `ChangeZoneAll` (Doomsday sub-ability)
- **What:** Move ALL cards matching a filter from origin zone(s) to destination
- **Script fields:** `Origin$ Graveyard,Library`, `Destination$ Exile`, `ChangeType$ Card.IsNotRemembered`
- **Needed:** New resolve() branch; also requires `RememberChanged$` support (remembering entities selected by a prior ChangeZone in the chain) and `IsNotRemembered` filter
- **Depends on:** Expanding `cur_game.remembered_entity` (currently a single Entity) to a set/vector for Doomsday (which remembers 5 cards)

### 4. `WinsGame` (Thassa's Oracle sub-ability)
- **What:** Named player wins the game (checked conditionally)
- **Script fields:** `Defined$ You`, `ConditionCheckSVar$ Y`, `ConditionSVarCompare$ LEX`
- **Needed:** New resolve() branch + condition evaluation comparing two SVars
- **Depends on:** SVar `Count$Devotion.Blue` and `Count$InYourLibrary`

### 5. `ChooseType` (Cavern of Souls)
- **What:** Player chooses a creature type as the permanent enters
- **Script fields:** via `K:ETBReplacement:Other:ChooseCT`
- **Needed:** ETBReplacement keyword handling + ChooseType ability + storing chosen type on the permanent + `RestrictValid$ Spell.Creature+ChosenType` mana restriction

## Engine Gaps — Unimplemented Keywords

### 6. `Cycling` (Street Wraith, Edge of Autumn)
- **What:** Alternative activated ability from hand: pay cost, discard this card, draw a card
- **Costs:** `PayLife<2>` (Street Wraith), `Sac<1/Land>` (Edge of Autumn)
- **Needed:** Parse `K:Cycling:<cost>`, add cycling as a legal activated ability when card is in hand, handle the discard+draw effect

### 7. `Flashback` (Deep Analysis)
- **What:** Cast this card from graveyard for its flashback cost, then exile it instead of going to graveyard
- **Cost:** `1 U PayLife<3>`
- **Needed:** Parse `K:Flashback:<cost>`, allow casting from graveyard zone, exile on resolution instead of graveyard
- **Note:** `stack_manager.cpp:89` has `// TODO handle flashback etc`

### 8. `Landwalk` (Street Wraith — Swampwalk)
- **What:** Creature can't be blocked if defending player controls a Swamp
- **Needed:** Parse `K:Landwalk:<subtype>`, check during blocker legality in `determine_legal_actions`
- **Priority:** Low — Street Wraith is almost always cycled, rarely cast

## Engine Gaps — SVar Expressions

### 9. `Count$InYourLibrary` (Doomsday, Thassa's Oracle)
- Count cards in controller's library
- **Used in:** Doomsday (`RearrangeTopOfLibrary NumCards$ X`), Thassa's Oracle (Dig + win condition)

### 10. `Count$YourLifeTotal` with `/HalfUp` (Doomsday)
- Get controller's life total, divide by 2 rounding up
- **Used in:** `SVar:Y:Count$YourLifeTotal/HalfUp` for LoseLife amount

### 11. `Count$Devotion.Blue` (Thassa's Oracle)
- Count blue mana symbols in mana costs of permanents you control
- **Used in:** Dig amount for Oracle's ETB trigger

## Engine Gaps — Cost Parsing

### 12. `Discard<0/Hand>` cost (Lion's Eye Diamond)
- Discard entire hand as activation cost
- **Needed:** Parse this cost format in `parse.cpp`, enforce during activation in `mana_system.cpp` / `action_processor.cpp`

## Engine Gaps — Other

### 13. `RememberChanged$ True` + remembered entity set (Doomsday)
- Currently `cur_game.remembered_entity` is a single Entity
- Doomsday needs to remember 5 chosen cards, then exile everything else with `IsNotRemembered` filter
- **Needed:** Expand to `std::vector<Entity> remembered_entities` or `std::unordered_set<Entity>`

### 14. `ConditionSVarCompare$ LEX` — comparing two SVars (Thassa's Oracle)
- Current condition system compares an SVar against a literal integer
- Oracle needs: if `Count$Devotion.Blue >= Count$InYourLibrary`, win
- **Needed:** Allow the comparand to be another SVar expression, not just a constant

### 15. Undercity Sewers ETB trigger
- Uses `T: Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self`
- ETB triggers via `CARD_CHANGED_ZONE` event already work for permanents on the battlefield
- **Issue:** The trigger fires on the permanent that just entered — need to verify `trigger_only_self` + `Card.Self` parsing works for self-ETB (it likely does based on state_manager.cpp:808-810)
- The `R: Event$ Moved` replacement for enters-tapped IS already implemented (`ENTERS_TAPPED` in state_manager.cpp:131)

### 16. Surveil trigger from Undercity Sewers
- `DB$ Surveil | Amount$ 1` — Surveil is already implemented in resolve()
- Should work once ETB trigger fires correctly

## Implementation Priority (suggested order)

**Tier 1 — Unblocks most cards:**
1. Add all 18 cards to vocab + regen card_costs.py
2. `LoseLife` ability (simple, unblocks Doomsday + Thoughtseize)
3. `Discard` ability + `RevealYouChoose` mode (unblocks Thoughtseize, Duress)
4. `Count$InYourLibrary` and `Count$YourLifeTotal/HalfUp` SVars

**Tier 2 — Unblocks Doomsday combo:**
5. Expand remembered entities to a vector/set
6. `ChangeZoneAll` ability with `IsNotRemembered` filter
7. `RememberChanged$` support in ChangeZone

**Tier 3 — Unblocks Thassa's Oracle win:**
8. `Count$Devotion.Blue` SVar
9. `WinsGame` ability with dual-SVar condition comparison
10. Verify ETB trigger for Thassa's Oracle

**Tier 4 — Unblocks remaining cards:**
11. `Cycling` keyword (Street Wraith, Edge of Autumn)
12. `Discard<0/Hand>` cost parsing (Lion's Eye Diamond)
13. `Flashback` keyword (Deep Analysis)

**Tier 5 — Low priority / nice-to-have:**
14. `Landwalk` keyword (Street Wraith — rarely relevant)
15. Cavern of Souls (`ChooseType` + mana restriction + uncounterable) — complex, may defer

## Cards playable after each tier

- **After Tier 1:** Swamp, Underground Sea, Bloodstained Mire, Verdant Catacombs, Personal Tutor, Dark Ritual, Lotus Petal, Consider, Undercity Sewers, Thoughtseize, Duress (11 new + 6 already working = 17/26)
- **After Tier 2:** + Doomsday (18/26)
- **After Tier 3:** + Thassa's Oracle (19/26)
- **After Tier 4:** + Street Wraith, Edge of Autumn, Lion's Eye Diamond, Deep Analysis (23/26)
- **After Tier 5:** + Cavern of Souls (24/26, Street Wraith swampwalk also works)

## Verification

After each tier:
- `make clean && make`
- `train/.venv/bin/python train/train.py --diag` with a doomsday deck game
- `train/.venv/bin/python train/train.py --watch-scripted` to observe card interactions
