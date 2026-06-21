TODO:
parse and display ability descriptions and targeting prompts

evoke implemented (parsed from K:Evoke, pitch/mana/life cost; cast via alt cost adds a
synthetic ETB self-sacrifice trigger gated on Permanent::evoked). Works for the whole
pitch-evoke Incarnation family out of the box.
  - TODO: when a permanent has multiple simultaneous ETB triggers (e.g. evoked Endurance:
    its graveyard-bottom trigger + the evoke sacrifice trigger), the controller should
    choose the order they go on the stack (APNAP). Currently they are pushed in a fixed
    order and the player is not prompted.

keen eyed curator buff from types exiled unimplemented? untested

dauthi cast from exile/void counter unimplemented? untested

Pro color untested
rev
Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise

Known ML problems:
Does not know what's in exile

