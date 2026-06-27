# Alpha Deathclaw

```
Name:Alpha Deathclaw
ManaCost:4 B G
Types:Creature Lizard Mutant
PT:6/6
K:Menace
K:Trample
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDestroy | ...
T:Mode$ BecomeMonstrous | Secondary$ True | ValidCard$ Card.Self | TriggerZones$ Battlefield | Execute$ TrigDestroy | ...
SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Permanent
A:AB$ PutCounter | Cost$ 5 B G | Monstrosity$ 4
```

6/6 Menace, Trample. When it enters **or becomes monstrous**, destroy target permanent.
`{5}{B}{G}: Monstrosity 4` (CR 701.37: if it isn't monstrous, put four +1/+1 counters on it
and it becomes monstrous).

## Already present (reused)

- **Menace, Trample** — keyword abilities from the `K:` lines.
- **Self-ETB trigger** — `Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self`
  parses to a `CARD_CHANGED_ZONE` trigger with `trigger_only_self` (and is also caught by the
  leaves/enters self-trigger scan). Fires on both cast-from-hand and any other entry.
- **`DB$ Destroy | ValidTgts$ Permanent`** — the shared `TrigDestroy` SVar; the existing Destroy
  effect targets any permanent and re-checks legality at resolution.
- **`AB$ PutCounter`** — the existing +1/+1 counter-placement path (`effect_put_counter.cpp`).
- **`Activation$` gate machinery** (CR 602.5) — the same `activation_condition` /
  `activation_condition_met()` path added for Mox Opal's Metalcraft.

## New: general Monstrosity (CR 701.37) — reusable

"Monstrosity N" = *if this permanent isn't monstrous, put N +1/+1 counters on it and it becomes
monstrous* (701.37a). Monstrous is a per-permanent designation that lasts until the permanent
leaves the battlefield (701.37b). Implemented generally so any future Monstrosity card works:

1. **`Permanent::is_monstrous`** (`src/components/permanent.h`) — internal per-permanent bool,
   reset naturally when a new `Permanent` is created on (re-)entry. **Not** in the obs/state
   vector. Mutated only by the Monstrosity resolution; read only by the activation gate.

2. **`Monstrosity$ N` parse** (`parse_put_counter`, `src/effects/effect_put_counter.cpp`) — sets
   `CounterParams{type=P1P1, count=N}`, marks `Ability::is_monstrosity`, and installs the
   `activation_condition = "NotMonstrous"` gate. (Parser honors the script's `Monstrosity$` tag;
   nothing is retagged.)

3. **`NotMonstrous` activation gate** (`activation_condition_met`, `src/game_queries.h`) — a new
   named condition keyed on the gated ability's **source** permanent's `is_monstrous`. The gate
   helper gained a `source` entity parameter (defaulted to 0); the three call sites
   (`mana_system.cpp`, `state_manager_actions.cpp`, `action_processor.cpp`) now pass the source
   permanent. So a Monstrosity ability is illegal once its source is already monstrous (701.37a).

4. **Monstrosity resolution** (`put_counter` / `apply_monstrosity`,
   `src/effects/effect_put_counter.cpp`) — when `is_monstrosity`: if the source is already
   monstrous, do nothing (701.37a); otherwise set `is_monstrous`, fire the new `BECAME_MONSTROUS`
   event, then fall through to place the N +1/+1 counters on the source via the normal path.

5. **`BECAME_MONSTROUS` event** (`src/ecs/events.h`, id 17) — `Params: ENTITY=the permanent that
   became monstrous, PLAYER=its controller`. Fired by the Monstrosity resolution.

6. **`Mode$ BecomeMonstrous` trigger** (`parse_one_trigger`, `src/parse.cpp`) — parses to
   `trigger_on = BECAME_MONSTROUS` with `ValidCard$ Card.Self` reusing the standard
   `trigger_only_self` ENTITY check. `TriggerZones$ Battlefield` is the default and needs no
   special handling. (`Secondary$ True` is cosmetic — it just marks the same printed reminder
   text as the ETB trigger; both reference the same `TrigDestroy`.)

## Test evidence

- **Cast from hand** → "Alpha Deathclaw enters the battlefield" → ETB trigger → destroyed target
  Noble Hierarch.
- **Activate Monstrosity** (with {5}{B}{G}) → "Alpha Deathclaw becomes monstrous" → "Put 4 +1/+1
  counter(s)" (6/6 → 10/10) → BecomeMonstrous trigger → destroyed target permanent.
- **Activation gate** — once monstrous, "Activate Alpha Deathclaw" is no longer offered (a
  scripted re-activation fails loudly with no matching legal action). A second activation made
  while the first is *still on the stack* (source not yet monstrous) is legal, as CR allows; it
  does nothing on resolution per 701.37a.

## CR references

- 701.37a/b/c — Monstrosity / monstrous designation.
- 602.5 — "activate only if" activation restriction (the NotMonstrous gate).
- 122.1 / 613.4c — +1/+1 counters and P/T.
