TODO:
parse and display ability descriptions and targeting prompts
redo draw to allow for dredge
confirm that draw is not screwing up known on top machine flag

Known rules problems:
Pro color untested
Effect-granted MayPlay unimplemented (Dauthi Voidwalker): the {T},Sac activated ability to play an
exiled void-counter card for free does NOT work — it relies on DB$ Effect | StaticAbilities$ MayPlay |
RememberObjects$ ChosenCard (one-shot effect granting play permission), whose params are unrecognized at
parse. The replacement (exile opp cards with a void counter) DOES work.
(NOTE: the permanent static form S:Mode$ Continuous | MayPlay$ True DOES work — verified on Icetill
Explorer playing lands from its graveyard.)

Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise

Known ML problems:
Does not know what's in exile

