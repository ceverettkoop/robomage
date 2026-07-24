# Boseiju, Who Endures  (vocab index 316)
## Oracle text
{T}: Add {G}.
Channel — {1}{G}, Discard Boseiju, Who Endures: Destroy target artifact, enchantment, or nonbasic land an opponent controls. That player may search their library for a land card with a basic land type, put it onto the battlefield, then shuffle. This ability costs {1} less to activate for each legendary creature you control.
## Forge script
- Source: pre-existing local
- Key tags: `A:AB$ Mana | Cost$ T | Produced$ G`; `A:AB$ Destroy | PrecostDesc$ Channel — | Cost$ 1 G Discard<1/CARDNAME> | ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl,Land.nonBasic+OppCtrl | SubAbility$ DBChangeZone | ReduceCost$ X | ActivationZone$ Hand` with `SVar:DBChangeZone:DB$ ChangeZone | Optional$ True | Origin$ Library | Destination$ Battlefield | ChangeType$ Land.hasABasicLandType | DefinedPlayer$ TargetedController` and `SVar:X:Count$Valid Creature.Legendary+YouCtrl`
## Engine work
- none — fully covered by existing handlers (Channel `ActivationZone$ Hand` activated ability with `Discard<>` cost, proven by Talon Gates idx50 / Eiganjo idx201; `Destroy` effect; sub-ability `ChangeZone` library search for the targeted player; `ReduceCost$` count-SVar reduction)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): Boseiju in hand (loaded via temp stacked deck to sidestep the comma name), Forests in play, opponent's Rishadan Port (nonbasic land) preset. Activated the Channel ability from hand → discarded Boseiju, paid {1}{G}, destroyed the opponent's Rishadan Port ("Rishadan Port is destroyed"), then the targeted player searched their library for a basic-typed land.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
