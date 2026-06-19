# Claude Audit Issues

Engine redundancy / divergent-rule audit. Each issue is a place where one game
rule is implemented in more than one location, with the risk (or reality) that
the copies disagree. Findings corroborated by more than one independent audit
pass are marked ✓✓.

Status legend: ☐ open · ◐ in progress · ☑ done

---

## HIGH — rules that can actually diverge and cause bugs

### ☑ 1. Two parallel target-legality engines that have already drifted ✓✓
`build_valid_targets()` (`action_processor.cpp:410-557`, enumerates legal targets)
and `Ability::is_target_valid()` (`ability.cpp:653-728`, re-verifies at resolution)
are meant to be mirror images but have diverged. Both already carry in-code
`// TODO ... redundant` comments.

- `build_valid_targets` handles `Artifact`/`Enchantment`/`Permanent`/`cmcLE<N>`
  and skips `is_phased_out`; `is_target_valid` has none of these and falls
  through to `return false` → a `ValidTgts$ Artifact` spell is offered a target
  then **always fizzles** at resolution.
- Multi-target gap: `select_target` fills `targets[]` but `is_target_valid`
  re-checks only the single `target` field — secondary targets that became
  illegal are still acted on (`action_processor.cpp:998-1012`).
- `Pump` bypasses the pipeline entirely (`ability.cpp:845` excludes it, then
  re-rolls its own creature scan at `1148-1174`) — a 4th copy ignoring
  protection / phased-out / color / legendary filters.
- Color restrictions use three different notions of "what color is this".

**Fix:** one `is_legal_target(entity, caster)` predicate consumed by both
`build_valid_targets` (filter candidates through it) and `is_target_valid`
(run chosen target(s) through it, iterating `targets[]`).

**Resolved.** Added `Ability::is_legal_target(Entity, Zone::Ownership)`
(`ability.cpp`) as the single legality predicate. `is_target_valid` now iterates
`targets[]` and delegates to it; `build_valid_targets` keeps only candidate
selection + perspective ordering and delegates every legality decision to it.
Also fixed a latent bug surfaced by unification: `ValidTgts$ Any` no longer
treats non-creature artifacts/enchantments as legal targets ("any target" =
creature/player/planeswalker), which had let `Any` burn reach `deal_damage` on a
permanent with no `Damage` component. Verified: 30 crash-free scripted games,
Lightning Bolt to creature/player, and Abrade destroying an artifact (the
previously-always-fizzling path).

### ◐ 2. No "effective P/T" model — six paths mutate `Creature.power/toughness` in place ✓✓
> **Done — UNTESTED. Needs manual testing by user.** Implementation landed
> (uncommitted: `creature.h/.cpp`, `ability.cpp`, `state_manager.cpp`) composing
> P/T from separate signed contributions through one recompute path. Not yet
> verified in-engine; user to manually test (Pump up/down, counters, Exalted on a
> CDA creature, lethal-damage interaction) before this is marked ☑.

There is no `effective_power()`/`effective_toughness()`. Every effect bakes its
delta into one `uint32_t` field and must manually un-bake it: base set
(`state_manager.cpp:174`), ETB counters (`:194-196`), PutCounter
(`ability.cpp:1731-1733`), Prowess/Exalted (`ability.cpp:1075-1084`), Pump
(`ability.cpp:1183-1184`), static buffs add/revert (`state_manager.cpp:689-712`).

Concrete bugs:
- **Underflow:** toughness is `uint32_t`; a `-Y` pump or mis-ordered revert
  drops below 0 → wraps to ~4 billion (creature won't die).
- **Exalted toughness lost:** Exalted is stored in `prowess_bonus` and added to
  power+toughness (`ability.cpp:1082-1084`), but the characteristic-defining
  recompute re-adds prowess to power only, omitting toughness
  (`state_manager.cpp:605-606`). A CDA creature given Exalted silently loses the
  toughness boost on the next SBE pass.

**Fix:** compose P/T from separate contributions (base + counters + temp bonus +
static bonus) through one recompute path; use signed arithmetic so a transient
negative can't underflow.

### ☑ 3. Affordability decided in `determine_legal_actions`, paid by a different algorithm ✓
Legality uses `can_afford_with_sources`/`can_afford_with_delve`
(`state_manager.cpp:1511-1513`); payment uses the independent greedy
`auto_pay_mana` (`mana_system.cpp:473-634`). When they disagree the action shows
legal but payment fails — masked by a band-aid that suppresses the action after
2 payment failures (`action_processor.cpp:1143` + `state_manager.cpp:1474-1475`).

**Resolved.** `auto_pay_mana` now takes a `commit` flag: with `commit=false` it
runs the identical greedy algorithm over a *copied* mana pool, skipping every
write-only side effect (tap/sac/life/counters/delve zone moves/uncounterable
flag) — none of which feed the payment decision. New `can_pay_mana()` wraps that
simulate mode and is the single predicate now consumed by `determine_legal_actions`
for spell casts (incl. delve, replacing `can_afford_with_delve`), flashback, and
activated-ability mana costs. Legality and payment are the same code, so they
cannot disagree. Removed the now-dead `can_afford_with_delve`/`count_delve_fuel`.
The `payment_fail_counts` band-aid is left in place as a dormant backstop.
Verified: diag (10 games, 0 draws/crashes); Murktide Regent cast via Delve
(exiles 2 instants to pay the {5}, enters 5/5 from the 2 counters).

### ☑ 4. Five copies of the "pay colored pips then generic" mana algorithm ✓
`can_afford_pool` (`mana_system.cpp:38-54`), `spend_mana` (`76-92`),
`pay_partial` (`108-130`), `can_afford_with_sources` (`374-391`), and a fifth
variant in `auto_pay_mana` (`568-631`). They already diverge on generic-payment
color preference.

**Resolved.** Added one primitive `pay_from_pool(pool, cost) -> remainder`
(specific colors first, then generic from any remaining; returns the unpayable
portion). `spend_mana`, `pay_partial`, `can_afford_pool` (dry-run on a copy it
already made), and the auto-payer's pool spends all route through it, so the
spend rule lives in exactly one place. The flexible/hypothetical-mana variant in
`can_afford_with_sources` (models multi-color sources as wildcard mana) is
genuinely different and intentionally left distinct. Verified: diag (no
draws/crashes) and Murktide delve cast unchanged.

### ☐ 5. No canonical "push ability onto the stack" — 4 hand-rolled copies ✓
`action_processor.cpp:142-148` (activate from hand), `:365-373` (from
battlefield), `state_manager.cpp:1153-1167` (triggered), `:1029-1035` (delayed).
Each repeats the "init Zone as HAND so add_to_zone origin-removal is a no-op" hack.

### ☐ 6. `draw()` bypasses the canonical zone mover ✓
`Orderer::draw` (`orderer.cpp:280-312`) sets `location = HAND` and adjusts
`distance_from_top` by hand instead of calling `add_to_zone`
(`orderer.cpp:35-118`) → a normal draw never fires `CARD_CHANGED_ZONE` and never
updates the known-top-of-library ML feature.

---

## MEDIUM — duplicated logic, currently consistent but drift-prone

### ☐ 7. Token ETB component bootstrap built twice ✓✓✓
Permanent/Creature/Damage + timestamp + summoning-sick constructed in both
`ability.cpp:1758-1776` and `state_manager.cpp:74-94`; `resolve_token` skips
`apply_keyword_abilities`.

### ☐ 8. `enters_tapped` sourced from two unrelated fields in two functions
`CardData.replacement_effects` (`state_manager.cpp:133-142`) vs
`Ability.enters_tapped` (`ability.cpp:454-457`) → fetched taplands inconsistent.

### ☐ 9. Sac-spec (`;`-delimited) + return-cost matcher validated then re-executed ✓✓
`state_manager.cpp:1613-1706` vs `action_processor.cpp:104-135` & `254-285`
(the sac loop is copy-pasted within one function).

### ☐ 10. Activated-ability cost payment forked between hand and battlefield branches ✓✓
`action_processor.cpp:88-135` vs `232-285`; hand branch lacks tap/return/discard costs.

### ☐ 11. Stack-object post-resolution zone move in 3 disagreeing places
`stack_manager.cpp:80-99`, `:104-114`, `ability.cpp:1351-1365` — a countered
flashback spell goes to graveyard, not exile.

### ☐ 12. First/double-strike "does it matter" decided in two places
Step-skip scan (`game.cpp:187-192`) vs per-creature damage gate
(`state_manager.cpp:756-763`), separate keyword literals.

### ☐ 13. Delve reduction implemented 3× (planning vs execution)
`state_manager.cpp:1297-1322` vs `mana_system.cpp:478-500` & `678-696`.

### ☐ 14. RaiseCost + X-cost effective-cost loop written twice
`state_manager.cpp:1487-1507` vs `action_processor.cpp:1089-1095`; only legality
side handles `CantBeCast`.

### ☐ 15. Player life-loss from damage handled two ways
Combat path fires events + lifelink (`state_manager.cpp:818`); ability
`DealDamage` does neither (`ability.cpp:1054`); no `deal_damage_to_player()`.

### ☐ 16. `entity_name(Entity)` byte-identical static in two TUs
`action_processor.cpp:42-51` vs `state_manager.cpp:37-47`.

### ☐ 17. `opponent_of(player)` open-coded 5×
`action_processor.cpp:454,488,835`, `ability.cpp:1205`, `state_manager.cpp:1270`
— despite an existing `get_player_entity()` helper.

### ☐ 18. "is (non)basic land" type scan copy-pasted in 3 files
`ability.cpp:719-724`, `action_processor.cpp:528-534`, `state_manager.cpp:461-467`.

### ☐ 19. Cost-token parser reimplemented 3× in parse.cpp
`parse.cpp:321-352`, `361-376`, `812-867`; Flashback & Cycling silently drop
tokens the main `Cost$` honors.

### ☐ 20. `mana_symbol_str` private static duplicated, shadowing canonical `mana_symbol()`
`mana_system.cpp:146-156`, `action_processor.cpp:53-70`, plus a 3rd inline copy
for Phyrexian (`action_processor.cpp:1121`).

---

## LOW — cosmetic duplication / latent issues

- ☐ `creature_has_keyword` written ~7× (`state_manager.cpp:749`,
  `action_processor.cpp:790`, `damage.cpp:5`, plus inline scans).
- ☐ `get_stack()` sort comparator compares `a` to itself (`orderer.cpp:334-337`,
  should be `b`); `StackManager` does a correct min-scan instead.
- ☐ "enter battlefield + set controller" open-coded at 5 sites; Dauthi free-cast
  path (`ability.cpp:1243-1248`) forgets the timestamp.
- ☐ Timestamp `++` inline at 3 sites, no `next_timestamp()` allocator
  (`ability.cpp:1765`, `state_manager.cpp:82,143`).
- ☐ Summoning sickness cleared in 2 scopes (`game.cpp:135` vs `main.cpp:136-138`).
- ☐ Graveyard-contents scan loop hand-rolled 4×+ (`ability.cpp:161/273/499`,
  `state_manager.cpp:1298`); no `Orderer::get_graveyard(owner)`.
- ☐ Library/zone-search filter loop reimplemented 3× (`ability.cpp:142/303/489`).
- ☐ Activated-ability `life_cost` paid but never gated in legality
  (`action_processor.cpp:243-247` has a `//TODO VERIFY`; `state_manager.cpp:1654-1713`
  omits the life check spells/alt-costs get).
- ☐ `parse_mana_cost` mis-parses multi-digit generic (`parse.cpp:546-550`,
  iterates char-by-char so `{10}` = 1 generic).
