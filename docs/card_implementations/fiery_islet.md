# Fiery Islet  (vocab index 309)
## Oracle text
{T}, Pay 1 life: Add {U} or {R}.
{1}, {T}, Sacrifice Fiery Islet: Draw a card.
## Forge script
- Source: pre-existing local
- Key tags: `A:AB$ Mana | Cost$ T PayLife<1> | Produced$ Combo U R`, `A:AB$ Draw | Cost$ 1 T Sac<1/CARDNAME> | NumCards$ 1`
## Engine work
- none — fully covered by existing handlers. This is a pure clone of Horizon Canopy (vocab idx36) — identical structure, only the produced colors differ (U/R vs Horizon Canopy's G/W). The `AB$ Mana` with `PayLife<1>` cost and `Combo` color choice, plus the `{1},{T},Sacrifice: Draw` ability (`Sac<1/CARDNAME>` cost), are all already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): casting a {R} spell (Lava Spike) paid by Fiery Islet logs "Player A pays 1 life" + "activated Fiery Islet for 1(R)". Activating the sac ability ({1} via a second land, {T}, Sacrifice) → "Player A sacrifices Fiery Islet" → draws a card; Fiery Islet to graveyard. (Proving card: Horizon Canopy idx36.)
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
