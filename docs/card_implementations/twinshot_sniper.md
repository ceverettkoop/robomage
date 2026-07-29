# Twinshot Sniper  (vocab index 317)
## Oracle text
Reach
When Twinshot Sniper enters, it deals 2 damage to any target.
Channel — {1}{R}, Discard Twinshot Sniper: It deals 2 damage to any target.
## Forge script
- Source: pre-existing local
- Key tags: `K:Reach`; `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDealDamage` with `SVar:TrigDealDamage:DB$ DealDamage | ValidTgts$ Any | NumDmg$ 2`; `A:AB$ DealDamage | Cost$ 1 R Discard<1/CARDNAME> | ValidTgts$ Any | NumDmg$ 2 | ActivationZone$ Hand | PrecostDesc$ Channel —`
## Engine work
- none — fully covered by existing handlers (ETB `ChangesZone` trigger → `DealDamage`; Channel `ActivationZone$ Hand` activated ability with `Discard<>` cost + `DealDamage`, same pattern as Boseiju/Eiganjo Channel)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): (1) ETB — cast Twinshot Sniper (3R), ETB dealt 2 damage to Player B (20 → 18). (2) Channel — activated the Channel ability from hand, discarded Twinshot Sniper, dealt 2 damage to Player B (→ 18), card went to graveyard.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
