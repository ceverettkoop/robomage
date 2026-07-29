# Otherworldly Gaze  (vocab index 305)
## Oracle text
Surveil 3. (Look at the top three cards of your library, then put any number of them into your graveyard and the rest on top of your library in any order.)
Flashback {1}{U} (You may cast this card from your graveyard for its flashback cost. Then exile it.)
## Forge script
- Source: pre-existing local
- Key tags: `A:SP$ Surveil | Amount$ 3`, `K:Flashback:1 U`
## Engine work
- none — fully covered by existing handlers. The `SP$ Surveil` category and the `Flashback` keyword (cast from graveyard, then exile) are already handled.
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous.
## Tests
- Isolation (test_harness): cast from hand → "surveils 3" with per-card top/graveyard choice → resolves to graveyard. Then "Cast Otherworldly Gaze (flashback)" from graveyard → surveils 3 again (stack tagged `[flashback]`) → "Otherworldly Gaze is exiled (flashback)" → ends in exile.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
