# Hybrid mana (general engine feature)

General support for **hybrid mana symbols** in card mana costs (CR 107.4, 202.3f): both the
common **color hybrid** (`{W/U}`) and the **monocolored hybrid / "twobrid"** (`{2/W}`). Phyrexian
(`{W/P}`) is a separate, pre-existing path and is intentionally **out of scope** here (see below).

## What "correct" means (CR)
- **Color hybrid `{W/U}`** — one pip payable by EITHER one white OR one blue mana. Mana value
  contribution = **1** (CR 202.3f: a hybrid symbol's MV is the greatest of its component MVs → 1).
  The card is **both** of the two colors (CR 105.2 / 202.3f).
- **Monocolored hybrid `{2/W}`** ("twobrid") — one pip payable by EITHER two generic mana OR one
  white mana. Mana value contribution = **2**. The card is the symbol's color (white).

## Representation chosen
`ManaValue` (`std::multiset<Colors>`) — the engine's flat mana-cost type, where CMC is read as
`mana_cost.size()` in several places — cannot carry a per-pip "W or U" alternative. Rather than
re-type that hot path, hybrid pips are stored in a **parallel vector** on the card, exactly the way
Phyrexian pips already are:

```cpp
// src/components/carddata.h
struct HybridPip {
    std::vector<Colors> colors;  // color options (color-hybrid: 2; twobrid: 1)
    int generic_alt = 0;         // generic alternative count (twobrid: 2; color-hybrid: 0)
    int mana_value = 1;          // CMC contribution (1 color-hybrid, N twobrid)
};
std::vector<HybridPip> hybrid_mana;   // kept OUT of mana_cost, like phyrexian_mana
```

`mana_cost` holds only the **non-hybrid** pips; `hybrid_mana` holds the hybrid pips. This keeps every
existing `mana_cost` consumer untouched for non-hybrid cards and confines hybrid handling to the few
seams below.

## Where it's implemented
- **Parsing** — `parse_mana_cost` (`src/parse.cpp`). The cost string is first tokenized on spaces
  (Forge's encoding: `3 WU WU`, `2/B 2/R`); `parse_hybrid_token` recognizes a hybrid token and
  appends a `HybridPip`, and the remaining (non-hybrid) tokens fall through to the original
  per-character parse. Recognized forms: two adjacent color letters `WU`, slashed color pair `W/U`,
  and twobrid `<N>/<color>` (`2/W`). A `?/P` / `?P` token is left to the Phyrexian path.
- **Mana value (CMC)** — `card_mana_value(const CardData&)` (`src/game_queries.h`) = `mana_cost.size()`
  + Σ `hybrid_mana[i].mana_value`. All five CMC read sites now call it (`game_queries.cpp` card/
  permanent views; `effect_destroy`, `effect_destroy_all`, `effect_play`).
- **Color identity** — `card_colors` and `is_colorless_card` (`src/game_queries.h`) fold in each
  hybrid pip's colors, so `{W/U}` makes the card white **and** blue and a hybrid card is never
  colorless. `effective_colors` already routes through `card_colors`.
- **Affordability + payment** — `resolve_hybrid_cost` (`src/mana_system.{h,cpp}`). It enumerates the
  hybrid pips' payment options (each color = one mana of that color; a twobrid also offers its
  `generic_alt` generics), tries the **colored option first**, and returns true on the first
  assignment that — added to the flat base cost — is fully payable per the existing `can_pay_mana`
  simulate-payer. The chosen concrete flat cost is written back so the legality gate and the actual
  payment use the **same** assignment. Used by the cast-legality gate (`state_manager_actions.cpp`),
  the kicker/replicate affordability checks, and the machine/auto cast payment
  (`action_processor.cpp`). Interactive mode prompts the player per pip which option to pay (mirrors
  the Phyrexian prompt block). Because affordability is an exhaustive search over assignments,
  castability is exact: a spell is castable iff SOME hybrid assignment is payable.

## Out of scope / follow-ups
- **Phyrexian mana `{W/P}`.** Unchanged: it stays in `CardData::phyrexian_mana` with its own
  pay-color-or-2-life prompt. (As before, Phyrexian pips do not contribute to `card_mana_value` —
  a pre-existing matter left untouched so existing Phyrexian cards' CMC is unaffected.)
- **Hybrid pips in *secondary* costs** (flashback / escape / equip / activation costs). Only the main
  card cost captures `hybrid_mana` (the same restriction Phyrexian already has — `parse_mana_cost`'s
  hybrid-out is only wired for the printed card cost). No current vocab card needs hybrid in those.
- **Twobrid coverage.** Color hybrid is exercised by Yorion, Sky Nomad (vocab 289). No current vocab
  card uses the `{N/color}` twobrid form, so that branch is implemented + unit-reasoned but not
  exercised by a vocab card in play. The machine-mode cast-cost feature matrix
  (`train/card_costs.py`) counts a hybrid card's both colors (and a twobrid's generic + color) — a
  sensible, non-crashing row; unchanged here (no vocab change).
