# Sunbaked Canyon  (vocab index 144)

## Oracle text
{T}, Pay 1 life: Add {R} or {W}.
{1}, {T}, Sacrifice Sunbaked Canyon: Draw a card.

## Forge script  (Source: pre-existing local; Key tags)
- `Types:Land`
- `A:AB$ Mana | Cost$ T PayLife<1> | Produced$ Combo R W` — pay-1-life dual mana ({R} or {W}).
- `A:AB$ Draw | Cost$ 1 T Sac<1/CARDNAME> | NumCards$ 1` — {1}, {T}, sacrifice: draw a card.

## Engine work  (none — covered by existing handlers)
Identical "horizon land" template to Horizon Canopy (vocab index 36), differing only in
the two mana colors (`Combo R W` vs Horizon Canopy's `Combo G W`):
- `AB$ Mana | Cost$ T PayLife<1> | Produced$ Combo <X> <Y>` — PayLife activation cost and
  the `Combo` flexible-color mana producer (color chosen on activation).
- `AB$ Draw | Cost$ 1 T Sac<1/CARDNAME>` — sacrifice-self activation cost (`sac_self`)
  with a Draw effect.
Every tag is exercised by the already-implemented Horizon Canopy.

## Behavioral decisions  (none)
Identical to Horizon Canopy.

## Tests
Skipped: proven by Horizon Canopy (identical horizon-land template, vocab index 36).
Clean `make HEADLESS=TRUE` build with the card registered.

## Result
Done. Covered card; verification skipped per the identical Horizon Canopy template.
