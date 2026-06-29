# Council's Judgment (vocab index 217)

## Oracle text
Will of the council — Starting with you, each player votes for a nonland permanent you don't
control. Exile each permanent with the most votes or tied for most votes. *(You may vote for an
object you secretly own but don't control.)*

(Sorcery — `{1}{W}{W}`.)

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/c/councils_judgment.txt`)
Key tags:
- `A:SP$ Vote | Defined$ Player | VoteSubAbility$ DBExile | VoteCard$ Permanent.nonLand+YouDontCtrl | VoteMessage$ for a nonland permanent you don't control`
  — the spell is a *will-of-the-council* vote (CR 701.32). `VoteCard$` is the set of objects that
  can be voted for (every nonland permanent the controller does not control). `VoteSubAbility$
  DBExile` names the SVar holding the effect applied to the winner(s).
- `SVar:DBExile:DB$ ChangeZone | Defined$ Remembered | Origin$ Battlefield | Destination$ Exile`
  — chained sub-ability: move the voted-for permanent(s) (the *remembered* object) from the
  battlefield to exile.

## Engine work
One new general mechanic, keyed on the script's real `SP$ Vote` tag (no retagging):

1. **`Vote` spell/ability effect** (`src/effects/effect_vote.cpp`, `effects::vote`). New
   `EffectKind::Vote` (`src/effects/effect_kind.{h,cpp}`, dispatched in
   `src/effects/effect_table.cpp`). The handler enumerates the battlefield permanents matching the
   `VoteCard$` filter (parsed into `Ability::vote_card_filter`, `src/components/ability.h`) via the
   shared `permanent_matches_filter` (`src/game_queries.h`), translating Forge's `YouDontCtrl`
   token to the evaluator's `OppCtrl` exactly as `Ability::is_legal_target` does. It offers those
   permanents to the spell's controller as a `CHOOSE_CARD` decision (value 44), records the chosen
   permanent into `cur_game.remembered_entities`, and returns `true` so the `VoteSubAbility$ DBExile`
   (parsed into `subabilities`, `Defined$ Remembered`) exiles it through the existing
   `effects::change_zone` Battlefield→Exile resolution.

2. **`VoteSubAbility$` / `VoteCard$` parsing** (`src/parse.cpp`). `VoteSubAbility$` is resolved in
   `parse_abilities` exactly like `SubAbility$`/`RepeatSubAbility$` — it names an SVar whose DB$
   body is parsed and pushed onto `subabilities`. `VoteCard$` is stored into
   `Ability::vote_card_filter` by `apply_param_to_ability`. The cosmetic `VoteMessage$` is added to
   the ignored-keys set.

## Behavioral decisions (CR cites)
- **Two-player reduction (user-directed).** The engine is **two-player only** (see CLAUDE.md
  scope); full multiplayer voting (CR 701.32 / the 800-series — three+ voters, ranges of
  influence) is out of scope. Per the user's design decision, the *will-of-the-council* vote is
  modeled as **"the spell's controller chooses one nonland permanent they don't control, and it is
  exiled."** This is the correct two-player outcome: you vote first, and with only one opponent any
  split/tie reduces to the single object you picked, so a one-pick choice exiles exactly the
  permanent the controller would force through the vote.
- **Choice, not a target (CR 115.10a / 701.32).** The permanent is *chosen* as the spell resolves,
  not declared as a target at cast time. Therefore an opponent's **hexproof** (CR 702.11b) or
  **shroud** (CR 702.18e) does **not** protect it — the handler enumerates candidates with
  `permanent_matches_filter` (which does not consult targeting keywords) and never calls
  `is_legal_target`. The spell is not "targeted," so it also does not fizzle for an illegal target.
- **Exile** (CR 406): the voted-for permanent goes to the exile zone via the standard
  Battlefield→Exile `ChangeZone`.
- **No eligible permanent** (CR 608.2 — do as much as possible): if the opponent controls no
  nonland permanent, there is nothing to vote for; the spell still resolves and does nothing.

## Tests (`train/test_harness.py`, seed 1)
- **(A) Basic vote → exile.** A casts Council's Judgment while A controls Plains + Noble Hierarch
  and B controls Noble Hierarch + Stoneforge Mystic + Forest. The vote menu offered **only B's two
  nonland permanents** (Noble Hierarch, Stoneforge Mystic) — not A's own permanents, not either
  Forest/Plains land. A voted Stoneforge Mystic → "Stoneforge Mystic is moved to exile"; B's
  battlefield left with Noble Hierarch + Forest. PASS.
- **(B) Hexproof/shroud bypass.** B activated Sylvan Safekeeper (sacrificing a Forest) to grant
  **Shroud** to B's Stoneforge Mystic ("Stoneforge Mystic gains Shroud until end of turn."). A's
  Council's Judgment still **offered the shrouded Stoneforge Mystic** in the vote menu and exiled
  it — confirming exile-by-vote is a choice, not a target. PASS.
- **(C) Zero legal choices.** B controls only Forests (lands). Council's Judgment resolves with
  "Will of the Council: no eligible permanent to vote for; nothing is exiled.", goes to A's
  graveyard, no choice prompted, no error. PASS.
- **(D) Determinism / regression.** Scripted-vs-scripted full games (a white maverick-style deck
  containing 2 Council's Judgment vs delver) over seeds 1–8 — every game produced a decisive
  winner (both A and B won across seeds), no draws, no crashes, and no non-fatal errors or
  non-cosmetic warnings.

## Result
Implemented. Council's Judgment lets its controller choose one nonland permanent they don't
control and exiles it, bypassing hexproof/shroud (it is a choice, not a target), per the
project's two-player reduction of the will-of-the-council vote (CR 701.32). The general `Vote`
effect, `VoteCard$` filter, and `VoteSubAbility$` chaining are reusable by future
will-of-the-council cards.
