TODO:
parse and display ability descriptions and targeting prompts

  - TODO: when a permanent has multiple simultaneous ETB triggers (e.g. evoked Endurance:
    its graveyard-bottom trigger + the evoke sacrifice trigger), the controller should
    choose the order they go on the stack (APNAP). Currently they are pushed in a fixed
    order and the player is not prompted.

keen eyed curator buff from types exiled untested

dauthi cast from exile/void counter unimplemented? untested

Pro color untested
rev
Engine stuff:
ML can only pay for spells after choosing them, this does not allow some rare cases of optimal behavior (e.g. floating mana) but should reduce noise

Known ML problems:
Does not know what's in exile

