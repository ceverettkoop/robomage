TODO:
parse and display ability descriptions and targeting prompts

redo draw to allow for dredge and similar replacement effects
implement life from the loam as the example of dredge
confirm that draw is properly changing what card is known to be on top by the ML model

review ML process and analaysis tools

endurance evoke unimplemented

keen eyed curator buff from types exiled unimplemented? untested

dauthi cast from exile/void counter unimplemented

Pro color untested

Evoke unimplemented (Endurance): the K:Evoke:ExileFromHand<1/Card.Green+Other/...> keyword does not
produce an alternate cast — with no green mana the card simply can't be cast (no evoke option offered),
and the post-ETB sacrifice is also absent. The AltCost/ExileFromHand parser (parse.cpp:170-243) only
handles S:...AlternativeCost lines (Force of Negation pitch), not the K:Evoke: keyword. Endurance's ETB
graveyard-to-bottom works fine on a hardcast.

Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise

Known ML problems:
Does not know what's in exile

