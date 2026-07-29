# Rishadan Port  (vocab index 313)
## Oracle text
{T}: Add {C}.
{1}, {T}: Tap target land.
## Forge script
- Source: pre-existing local
- Key tags: `A:AB$ Mana | Cost$ T | Produced$ C`; `A:AB$ Tap | Cost$ 1 T | ValidTgts$ Land`
## Engine work
- none — fully covered by existing handlers (`AddMana` activated ability + `Tap` activated ability with `{1}` + tap cost and a Land target)
- Mechanics added: none
## Behavioral decisions
- none — behavior unambiguous
## Tests
- Isolation (test_harness): Rishadan Port + 2 Forests preset for A, opponent Island preset. Activated Rishadan Port's `{1},{T}: Tap target land`, paid {1} from a Forest, targeted the opponent's Island → "Island is tapped." Opponent's Island shown tapped.
- CI gate: pygen,vocab,smoke — 0 errors, no draws
## Result
implemented
