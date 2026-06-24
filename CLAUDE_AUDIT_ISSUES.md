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

### ☑ 5. No canonical "push ability onto the stack" — 4 hand-rolled copies ✓
`action_processor.cpp:142-148` (activate from hand), `:365-373` (from
battlefield), `state_manager.cpp:1153-1167` (triggered), `:1029-1035` (delayed).
Each repeats the "init Zone as HAND so add_to_zone origin-removal is a no-op" hack.

**Resolved.** Added `Orderer::push_ability_onto_stack(const Ability&, controller)`
(`orderer.cpp`) which owns the create-entity + HAND-zone + move-to-STACK + attach
sequence. All four sites now build the populated Ability and call it. Verified: Soul Warden
ETB trigger (a real stack trigger) — casting a creature fires "Soul Warden
triggered", the GainLife ability resolves off the stack, and it triggers on both
players' creatures entering; plus diag (10 games, no draws/crashes) and the
Murktide delve cast (spell-cast + delve path unaffected). (Murktide's enters-with
counters are an etbCounter *replacement* effect, not a stack trigger, so that
cast does not itself exercise these sites.)

### ☑ 6. `draw()` bypasses the canonical zone mover ✓
`Orderer::draw` (`orderer.cpp:280-312`) sets `location = HAND` and adjusts
`distance_from_top` by hand instead of calling `add_to_zone`
(`orderer.cpp:35-118`) → a normal draw never fires `CARD_CHANGED_ZONE` and never
updates the known-top-of-library ML feature.

**Resolved.** `draw` now collects the top `ct` cards (ascending distance) and
moves each via `add_to_zone(false, card, HAND)`, so a draw fires
`CARD_CHANGED_ZONE`, closes the library gap, and updates the known-top cache like
any other zone change. Dropped the manual `distance_from_top -= ct` (gap-closing
now handles it). Verified: diag (10 games, no draws/crashes) and a deterministic
`--no-shuffle` test drawing 8 cards in exact library order.

---

## MEDIUM — duplicated logic, currently consistent but drift-prone

### ☑ 7. Token ETB component bootstrap built twice ✓✓✓
Permanent/Creature/Damage + timestamp + summoning-sick constructed in both
`ability.cpp:1758-1776` and `state_manager.cpp:74-94`; `resolve_token` skips
`apply_keyword_abilities`.

**Resolved.** Added one `bootstrap_token_components(entity, token, controller,
&timestamp)` (`token.h`/new `token.cpp`) that attaches Permanent + Creature +
Damage from a Token, consuming the timestamp counter (by reference) only when it
actually creates the Permanent. Both `Ability::resolve_token` (immediate, so the
`Attach` subability sees the components) and `StateManager::apply_permanent_components`
(the SBE pass) now call it. The SBE pass still calls `apply_keyword_abilities`
afterward — and because it runs every pass on a battlefield token, `resolve_token`
not calling it directly is harmless (keywords land on the next tick). Fixed a
latent regression avoided in passing: the SBE site previously consumed
`timestamp++` only when adding the Permanent; the unified helper preserves that
(post-increments through the reference only on creation) so repeated passes over
an existing token don't advance the timestamp. Verified: diag (10 games, no
draws/crashes) and Cori-Steel Cutter (cast 2nd spell → token created → equipment
attached to it → token persists across SBE passes, attacks, deals damage).

### ☑ 8. `enters_tapped` sourced from two unrelated fields in two functions
`CardData.replacement_effects` (`state_manager.cpp:133-142`) vs
`Ability.enters_tapped` (`ability.cpp:454-457`) → fetched taplands inconsistent.

**Resolved.** The `Ability.enters_tapped` branch in `resolve_change_zone` was in
fact dead: it poked `Permanent.is_tapped` immediately after `add_to_zone`, but the
Permanent isn't created until the next SBE pass (`add_to_zone` doesn't make one),
so a fetch "onto the battlefield tapped" of a non-tapland (e.g. Edge of Autumn
fetching a basic) never tapped. Now the ChangeZone path records a one-shot
`cur_game.pending_enters_tapped` intent for the entity, and the single tapping
decision in `apply_permanent_components` ORs that in (and consumes it) alongside
the card's own `ENTERS_TAPPED` replacement effect — one source, one place. The
"enters tapped" log now fires once from that unified point.

Surfaced + fixed in passing (the only vocab card exercising this path is Edge of
Autumn, which was independently blocked): `matches_filter_spec` treated the
`Basic` in a `Land.Basic` search filter as a *color* qualifier, so the search
matched nothing and the fetch always failed to find. Added explicit
`Basic`/`nonBasic` supertype handling (checks for the `Basic` SUPERTYPE on the
card). Verified: Edge of Autumn now finds basics and the fetched Forest reports
"Forest enters tapped" (tapped on the battlefield); Thundering Falls (own
replacement) still enters tapped via the normal play path; diag 0 draws/crashes.

> **Build note:** this change adds a data member to `struct Game` (`game.h`). The
> Makefile has no header-dependency tracking, so an incremental `make` only
> rebuilds directly-edited `.cpp` files, leaving other TUs linked against the old
> `Game` layout → `cur_game` corruption (manifested as all-draws / early exit).
> A `make clean && make` is required after any struct-layout change.

### ☑ 9. Sac-spec (`;`-delimited) + return-cost matcher validated then re-executed ✓✓
**Resolved.** Added two shared `game_queries.h` helpers:
`permanent_matches_subtype_spec(perm, spec)` (the `;`-delimited subtype matcher)
and `controlled_permanents_matching(player, spec, entities)` (battlefield
permanents controlled by a player matching the spec). The six copy-pasted match
loops — four in legality (`state_manager_actions.cpp`, sac + return, battlefield +
hand) and two in payment (`action_processor.cpp`, hand + battlefield) — all
delegate to these now: legality checks `.empty()`, payment iterates the same list
to build the choice menu. One matcher, consumed by both sides. Verified: Knight of
the Reliquary (sac a Forest/Plains → fetch) and Scryb Ranger (return a Forest →
untap target) both pay the cost correctly.

### ☑ 10. Activated-ability cost payment forked between hand and battlefield branches ✓✓
**Resolved.** Extracted `pay_secondary_activation_costs(ability, source,
controller, orderer)` in `action_processor.cpp` covering life / sac-self /
sac-spec / return-to-hand / discard-self / discard-hand. Both the hand and
battlefield activation branches call it after paying mana, so the hand branch no
longer silently skips the return/discard/sac-self costs. (Tap + mana stay per
branch — they gate activation / can be cancelled with snapshot-restore.) Surfaced
+ fixed in passing: Faerie Macabre sets `discard_self_cost` *and* relied on the
hand branch's `!defined_self` auto-discard, so once the helper paid the discard
the auto-move would double-send it; the auto-move is now guarded with
`&& !ability.discard_self_cost`. Verified: Faerie Macabre discards exactly once
(single GY copy).

### ☑ 11. Stack-object post-resolution zone move — resolution + counter paths unified
The **resolution** copies were consolidated earlier (`ability.cpp::resolve()` no
longer moves zones; `stack_manager.cpp` is the single resolution-time decision).
**Now resolved on the counter side too:** the flashback→exile rule lives in one
helper `spell_cast_with_flashback(Entity)` (`game_queries.h`), consumed by both
`stack_manager` (resolution) and `effect_counter` (counter). A countered
flashback spell now goes to exile, not graveyard. Subtlety caught in testing:
`effect_counter` removes the `Spell` component before choosing the destination,
so the flashback flag is captured into `was_flashback` *before* removal. Verified:
flashback Deep Analysis countered → exiled (absent from both graveyards);
hardcast Deep Analysis countered → graveyard (no regression).

### ☑ 12. First/double-strike "does it matter" decided in two places
**Resolved.** Added one predicate `creature_deals_first_strike_damage(const
Creature&)` (`game_queries.h`) — true iff the creature has First Strike or Double
Strike. The step-skip scan (`game.cpp` DECLARE_BLOCKERS → does a first-strike
damage step matter) and the per-creature damage gate (`should_deal_damage`'s
`first_strike_only` branch in `state_manager_combat.cpp`) both call it, so the two
can no longer drift on the keyword literals. Verified: a first striker (Thalia)
attacking unblocked enters the First Strike Combat Damage step; a lone
non-first-striker (Grizzly Bears) skips straight to Combat Damage.

### ☑ 13. Delve reduction implemented 3× → unified to 2 shared helpers
The "planning" copy (`can_afford_with_delve`/`count_delve_fuel`) was already
removed under issue 3 (legality now runs the real payer in simulate mode). The two
remaining execution copies — the automatic payer (`auto_pay_mana`) and the
interactive prompt (`prompt_mana_payment`), both in `mana_system.cpp` — duplicated
both the "which graveyard card can Delve eat" scan and the "exile one card to pay a
generic pip" action. **Resolved.** Added two statics in `mana_system.cpp`:
`is_delve_eligible(Entity, controller)` (graveyard instant/sorcery owned by the
caster) and `delve_exile_one(...)` (exile + record in `delve_exiled` + drop one
GENERIC + log). Both payment paths delegate to them. Verified: delver mirror games
exile multiple instants via Delve and Murktide Regent casts via Delve, entering as
an 8/8 (4 instants exiled → 4 +1/+1 counters over the 4/4 base).

### ☑ 14. RaiseCost effective-cost loop written twice → one builder
`active_raise_cost_for` already centralised "how much"; the "insert that many
GENERIC pips into the base cost" loop was duplicated in legality
(`state_manager_actions.cpp`) and payment (`action_processor.cpp`). **Resolved.**
Added `effective_base_cost(const CardData&)` (`state_manager_statics.cpp`, declared
in `state_manager.h`) = `mana_cost` + RaiseCost surcharge (NOT the X choice, which
is interactive at cast time). Both sites consume it. The X-cost loop is execution-
only by design (legality uses the base cost), and `CantBeCast` is legality-only by
design (execution never re-checks an action that wasn't offered) — neither is a
duplicated copy. Verified: with Thalia (RaiseCost) in play, Player B's Lightning
Bolt is uncastable on a single Mountain ({1}{R}); with no Thalia it is castable
(R).

### ☑ 15. Player life-loss from damage — one `deal_damage_to_player()`
**Resolved.** Added `deal_damage_to_player(source, player_entity, amount)`
(`damage.cpp`/`damage.h`): reduces the player's life and grants the source's
controller lifelink if the source has it. The ability `DealDamage` path
(`effect_deal_damage.cpp`) now routes through it, so ability/burn damage from a
lifelink creature finally gains life (previously it did neither). The combat path
(`deal_combat_damage`) intentionally keeps its own subtraction + `apply_lifelink_if_any`
accumulation because combat lifelink must be applied *simultaneously* with damage
taken (rule 119.3) — documented at the helper. The combat-only
`COMBAT_DAMAGE_TO_PLAYER` event stays combat-specific (only Barrowgoyf-style
"deals combat damage" triggers consume it); no unconsumed general damage event was
added. Verified: Lightning Bolt to a player still deals 3 (20→17) with no spurious
life gain (the spell source has no Lifelink); diag 10 games, 0 draws/crashes.

### ☑ 16. `entity_name(Entity)` byte-identical static in two TUs
**Resolved.** `action_processor.cpp` now includes `state_manager_internal.h` and
its byte-identical private static (plus the stale forward declaration) was deleted,
so both TUs link to the single definition in `state_manager_statics.cpp`.

### ☑ 17. `opponent_of(player)` open-coded 5×
**Resolved.** The open-coded `(player == A) ? B : A` flips were replaced by the
existing `get_player_entity()` helper (`mana_system.h:18`), now called
consistently (`action_processor.cpp`, `state_manager_statics.cpp`, etc.).

### ☑ 18. "is (non)basic land" type scan copy-pasted in 3 files
**Resolved.** `state_manager_actions.cpp`'s two standalone land scans (play-from-hand
and play-from-graveyard legality) now call `is_land_card()`. For the "Basic"
*supertype* test (nonBasic-land filters) — which `is_basic_land_subtype()` does NOT
cover (that matches the six basic *subtype* names) — added a new
`has_basic_supertype(const std::set<Type>&)` to `game_queries.h` and adopted it at
all three `ability.cpp` sites (`matches_filter_spec` Basic/nonBasic qualifier, plus
the graveyard and battlefield target-validity nonBasic checks), collapsing the
hand-rolled supertype loops. Verified: Wasteland offers only nonbasic lands (Tundra,
itself) as targets — basics excluded — and destroys the targeted Tundra; diag 10
games, 0 draws/crashes.

### ☑ 19. Cost-token parser reimplemented 3× in parse.cpp
**Resolved.** Extracted the canonical `Cost$` token grammar into
`parse_activation_cost(const std::string&, Ability&)` (tap `T`, PayLife<N>,
Sac<.../...> incl. `CARDNAME`→sac-self, Discard<0/Hand|CARDNAME>, Return<N/Type>,
mana). The three sites now share it: the `Cost$` ability param, Cycling (which had
silently dropped tap/Discard/Return), and Flashback (which had dropped everything
but mana+life). Flashback maps the parsed temp Ability's `activation_mana_cost` +
`life_cost` onto its `flashback_mana_cost`/`flashback_alt_cost.life_cost` (the two
fields the cast path consumes) — so "1 U PayLife<3>" (Deep Analysis: both mana and
life) now flows through one token-by-token pass. Verified: Street Wraith's
`Cycling:PayLife<2>` is offered as an activatable ability; Deep Analysis is cast
from hand, then re-cast via flashback paying {1}{U} + 3 life and exiled; diag 10
games, 0 parse errors / draws / crashes.

### ☑ 20. `mana_symbol_str` private static duplicated, shadowing canonical `mana_symbol()`
**Resolved.** Promoted `const char *mana_symbol_str(Colors)` to a single canonical
definition in `classes/colors.{h,cpp}` (alongside the std::string `mana_symbol()`),
returning string literals so the `const char*` call convention stays valid. Deleted
the two byte-identical private statics in `mana_system.cpp` and
`action_processor.cpp`, and replaced the 3rd inline copy (the Phyrexian-mana color
name in `action_processor.cpp`) with a `mana_symbol_str()` call. Verified: mana
activation logs still render symbols correctly (`Island for 1(U)`,
`Volcanic Island for 1(R)`); diag 10 games, 0 draws/crashes.

---

## LOW — cosmetic duplication / latent issues

- ☑ `creature_has_keyword` — unified as `inline bool creature_has_keyword(const
  Creature&, const char*)` in `game_queries.h`; callers (combat etc.) use it.
- ☑ `get_stack()` sort comparator — now correctly compares `a` to `b`
  (`orderer.cpp:451-453`).
- ☐ "enter battlefield + set controller" open-coded at 5 sites; Dauthi free-cast
  path (`ability.cpp:1243-1248`) forgets the timestamp.
- ☐ Timestamp `++` inline at 3 sites, no `next_timestamp()` allocator
  (`ability.cpp:1765`, `state_manager.cpp:82,143`).
- ☐ Summoning sickness cleared in 2 scopes (`game.cpp:135` vs `main.cpp:136-138`).
- ◐ Graveyard-contents scan loop — `Orderer::get_graveyard(owner)` now exists
  (`orderer.cpp:431-440`), but `ability.cpp` still hand-rolls the loop in places.
- ☐ Library/zone-search filter loop reimplemented 3× (`ability.cpp:142/303/489`).
- ☑ Activated-ability `life_cost` now gated in legality — `can_afford_alt()`
  (`state_manager_actions.cpp:36-94`) checks `life_cost` before the action is
  offered, matching spell/alt-cost handling.
- ☑ `parse_mana_cost` multi-digit generic — the digit case now consumes the entire
  run of digits and `std::stoi`s it (so "10" = 10 generic), instead of adding one
  generic per digit char. Single-digit costs unchanged (diag 10 games, 0 draws).
