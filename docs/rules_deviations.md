# Rules deviations and known gaps

Where the engine knowingly departs from the Comprehensive Rules
([`mtg_comprehensive_rules.txt`](mtg_comprehensive_rules.txt)) or leaves a rules edge case
unimplemented. **Update this file when you knowingly deviate, leave a gap, or fix one listed here.**
Two-player games only (`CLAUDE.md`); multiplayer rules (CR 800+) are out of scope and not listed.

---

## 1. Intentional deviations (design decisions)

### 1.1 A cost's moved cards are narrated on different sides of the move
- **CR** none (narrative only). A cast logs each card its cost moves just before the move; an
  activation or triggered ability's cost just after it, so a line a replacement prints during the
  move lands on different sides. Moves, order and frames are the same. `src/cost_payment.cpp`: `move_items`.

### 1.2 Name-a-card offers a limited candidate set
- **CR 201.4:** any card in the Oracle card reference may be named.
- **Engine:** with open decklists (default) the candidates are the distinct vocab cards in the
  relevant deck(s), filtered by `ValidCards$`, uncapped: `CHOOSER_ONLY` one player's whole deck
  (Cabal Therapy: the target player's; Disruptor Flute: the opponent's), `BOTH_PLAYERS` both decks
  for land naming (Petrified Hamlet). Under `--closed-decklist` (`docs/state_format.md` register 11a)
  they are every vocab card (by front face) passing `ValidCards$`, in name order, read from the
  parse cache (`card_data_by_name`), so no entity or RNG draw changes.
- **Effect:** a card outside both decks (open) or outside the vocab (closed) can't be named; naming
  one would only bluff.
- **Code:** `src/name_card_choices.cpp`: `build_name_card_choices`; `src/effects/effect_name_card.cpp`;
  `src/card_db.cpp`: `card_data_by_name`.

### 1.3 Drawn games in a bo3 match
- **CR 104.4a, 103.1** (match procedure is MTR-style policy). A draw (`GAME_RESULT: <n> draw`)
  counts for neither player; the match continues until someone has won two games, and the drawn
  game's starting player starts again (103.1 under the forced-play policy, 2.1). Reaching
  `MAX_MATCH_GAMES` (10) is fatal. `src/game_driver.cpp`: `play_bo3_match`; `src/game_driver.h`.

### 1.4 Automatic mana payment (auto-mana)
- **CR 601.2g–h, 605, 107.4e:** the player chooses which mana abilities to activate, which pool
  mana to spend and how each hybrid symbol is paid.
- **Engine (auto-mana on, the default):** a greedy payer pays what the pool doesn't cover,
  preferring painless and single-colour sources, falling back to each source's highest-yield
  ability. Hybrid pips take the first payable assignment, unasked, unless the card reads the colors
  spent (Vibrance, `spell_reads_colors_spent`): those pips are asked among the payable colors while
  the payer still picks the generic part. Sunburst counts the colors of the sources the payer chose
  (Engineered Explosives). Ordinary mana abilities are not offered at priority; instant-speed
  sources and sources targeted by removal on the stack are, so mana can be floated.
  `--no-auto-mana` offers every mana ability and pays costs and hybrid pips by the player's choices.
- **In both modes:** generic pips are paid from the pool in enum order (W, U, B, R, G, C), after
  restricted mana usable on the payment and mana whose rider applies, before mana whose rider would
  be wasted (`spendable_unit`). TapToPay (Improvise) taps only for what mana sources can't pay, in
  entity order (under `--no-auto-mana` the permanents are payment rows). Convoke asks a creature at
  a time before mana is paid; each pays the first pip it can (colored before generic) that leaves
  the rest payable, so a two-color creature can't be aimed at its second color.
- **Effect:** no tapping decisions. With {W}{C} floating, paying {1} spends the {W}. A mana ability
  activated from the hand (Simian Spirit Guide) is spent last, after self-sacrificing sources, and
  stays offered at priority. A mana ability whose rider gives a keyword (Arena of Glory's second)
  is never chosen by the payer; offered at priority, its mana floats and the rider goes with the
  first spell it matches.
- **Code:** `src/mana_system.cpp`: `auto_pay_mana`, `auto_pay_mana_attempt`, `pay_from_pool`,
  `resolve_hybrid_cost`; `src/action_processor.cpp`: `run_cast_flow` (HYBRID step).

### 1.5 An illegal menace block is released, not rewound
- **CR 509.1b, 702.111b, 733.1:** an illegal block declaration is rewound as a whole. The engine
  removes a lone blocker on a menace attacker and the other blocks stand.
  `src/action_processor.cpp`: `release_illegal_menace_blockers`.

### 1.6 Interchangeable choices are collapsed; the sideboard menu is capped at 64
- **CR** none (action encoding). Menu entries making the same choice on interchangeable objects are
  dropped (two copies in a graveyard, or in a library when each seat sees both or neither). The
  RMSTATE menu has no width limit; a sideboarding menu wider than `SIDEBOARD_MENU_LIMIT` (64) is
  fatal. `src/input_logger.cpp`: `interchangeable_choices`, `distinct_choice_indices`;
  `src/queries/zones.cpp`: `interchangeable_cards`; `src/game_driver.cpp`.

### 1.7 Some prompts block and are never search roots
- **CR** none (the rules outcome is correct). These prompts read input inline and can't suspend:
  - choosing among two or more applicable replacement effects (CR 616.1);
  - the Mox Diamond discard;
  - an Aura's enchant pick when it is put onto the battlefield by a library search, a same-name
    move, a Remembered/ExiledWith return, the return of an "until" exile (within a zone move, CR
    610.3), Dig, DigUntil or ChangeZoneAll.

  They are not loop-safe, so no snapshot is saved or restored there (the `--selfcheck-state`
  snapshot check skips them; a search could not root there). Accepted as long as search doesn't break.
- **Code:** `src/systems/replacement_effects.cpp`: `choose_one`, `apply_discard_else_graveyard`;
  the `FrameCtx::blocking()` call sites in `src/effects/effect_change_zone.cpp`, `effect_dig*.cpp`,
  `effect_change_zone_all.cpp`; `src/snapshot.h`: `snapshot_set_loop_safe`.

### 1.8 Rulings the engine follows that are easy to mistake for bugs
- **CR 614.12:** a nonbasic land entering under Blood Moon / Magus of the Moon enters **untapped**
  (it won't have its "enters tapped" ability on the battlefield); Moonshadow cast under Humility gets
  no -1/-1 counters. `src/systems/replacement_effects.cpp`: `collect`.
- **CR 613.1f / 614.12:** Containment Priest under Humility exiles nothing (the replacement comes
  from a removed ability). `replacement_effects.cpp` (comment above `collect`).
- **CR 702.21a / 601.2c:** a permanent targeted by several "target" words of one spell or ability
  becomes the target once, so each Ward instance (and each BecomesTarget trigger) triggers once, as
  the Heroic/Valiant rulings read. `src/action_processor.cpp`: `chosen_targets_of`,
  `trigger_ward_for_targets`, `fire_targeting_hooks`.
- **CR 118.3 / 118.12:** "you may discard your hand" (Phyrexian Dragon Engine) is offered on an
  empty hand and, accepted, draws three: discarding a whole hand needs no resource (`hand_covers`)
  and "if you do" reads the choice. `src/cost_payment.cpp`: `offer_trigger_cost`.
- **CR 603.8:** Dark Depths' Marit Lage is created even though the sacrifice makes the "no ice
  counters" check false: a state trigger, not an intervening-if. `src/resolution.cpp`: `resolve_ability`.

---

## 2. Known gaps and unimplemented edge cases

### 2.1 Pregame and match procedure
- **CR 103.2b — the companion is revealed after mulligans** (FIAT_SETUP pregame stage), so the
  mulligan observation lacks the opponent's companion. `src/game_driver.cpp`: `pregame_fiat_setup`;
  `src/companion.cpp`: `setup_companions`.
- **CR 103.1 — no play/draw choice.** The previous game's loser plays first; in game 1 A is on the
  play. `rm_driver.py --on-the-play b|random` swaps the sides between engine seats instead.
  `src/game_driver.cpp`: `play_bo3_match`.
- **CR 100.2a / 100.4a — deck limits are not checked** (60 cards, sideboard ≤ 15, four copies), since
  harness decks are small. A malformed count hits `std::stoul` under `-fno-exceptions` and
  terminates silently. `src/classes/deck.cpp`: `Deck::parse_text`.

### 2.2 Combat
- **CR 510.1c — damage split among blockers is limited to lethal-sized picks.** A split is asked only
  when the attacker can't assign lethal to every blocker: "assign exactly lethal to blocker X" one
  pick at a time, the leftover on the last pick (or the first blocker if none). Otherwise it is
  automatic: lethal to each, the excess on the last blocker or trampling over.
  `src/action_processor.cpp`: `attacker_needs_assignment`, `run_damage_assignment`,
  `finish_pending_attacker`; `src/systems/state_manager_combat.cpp`: `deal_combat_damage`.
- **CR 702.7b/c — the second damage step reads current keywords.** A creature that gains first strike
  after the first-strike step deals no damage; one that loses it after dealing first-strike damage
  deals damage again. `state_manager_combat.cpp`: `should_deal_damage`.

### 2.3 Continuous effects and layers
- **CR 613.8 — dependency only for land-type setters.** A land-type setter (Blood Moon) suppresses
  the statics of the nonbasics it affects (Yavimaya, Cradle of Growth); every other within-layer
  order is timestamp order (613.7); `resolve_dependencies` is a no-op.
  `src/systems/state_manager_statics.cpp`: `apply_type_changing_effects`;
  `src/systems/state_manager_layers.cpp`: `resolve_dependencies`.
- **CR 613.7 — unearth's haste has no timestamp,** so ability removal (Humility) always removes it,
  even when older. `state_manager_statics.cpp`: `mark_unearthed_permanent`, `regrant_later_keywords`.
- **CR 613.7c / 613.1f — keyword counters and keyword-derived triggers (Prowess) count as older than
  any removal:** a later removal always strips them and they never return after one.
  `state_manager_statics.cpp`: `recompute_abilities`.
- **CR 611.3a / 613 — statics are re-applied at each state-based pass** (`apply_continuous_effects`),
  so at a decision mid-flow (a trigger's target choice, a parked prompt) a characteristic can lag the
  change just made (Urza's Saga's Construct's power after a sixth artifact). A plain
  `snapshot_restore` re-derives at once, so a restored decision can show the newer value; the
  `--selfcheck-state` snapshot check uses `snapshot_restore_exact`.
  `src/systems/state_manager_layers.cpp`; `src/snapshot.cpp`.
- **CR 614.12 — a token keeps its own entering replacements under ability removal:** the check reads
  `CardData`, which tokens lack. Latent. `state_manager_statics.cpp`: `entering_object_loses_abilities`.

### 2.4 Triggers and state-based actions
- **CR 603.2 / 603.3 — triggers are collected at the next SBA check,** not as each event happens. A
  source that leaves in the same resolution after its event misses a non-look-back trigger.
  Leaves-the-battlefield look-back (603.10a) treats all departures in one drained batch as
  simultaneous. `src/systems/state_manager.cpp`: `state_based_effects`;
  `src/systems/state_manager_triggers.cpp`: `collect_triggered_abilities`,
  `match_departed_watcher_triggers`.
- **CR 603.4 — `ResolvedLimit$` counts resolutions, not mana added** (as Forge counts it): a
  resolution counts once its "you may" is accepted and its intervening-if and targets held; at the
  limit the ability doesn't trigger and one on the stack does nothing. Carpet of Flowers accepted at
  X = 0 adds no mana but counts, so it doesn't trigger again that turn (its oracle says it would);
  declining doesn't count. `src/queries/activation.cpp`: `trigger_resolved_limit_reached`,
  `count_resolution`; `src/resolution.cpp`; `state_manager_triggers.cpp`.
- **CR 704.3 / 704.5j — one legend-rule conflict per SBA check** (APNAP, then by name); others
  resolve in later iterations, not simultaneously. `state_manager.cpp`: `choose_legend_rule_keep`.

### 2.5 Casting, targeting and resolution
- **CR 608.2b — a chained `Defined$ ParentTarget` of a player target gone illegal** finds no target,
  so a Draw reading it falls back to the controller (`affected_player`) instead of doing nothing.
  Latent (Kozilek's Command only). An object chain correctly finds nothing.
- **CR 115.7f, 707.10 — a copy of a dividing spell** choosing new targets (`run_target_select`) isn't
  held to the original's target count or shares. Latent: no copy effect reaches a
  `DividedAsYouChoose$` spell. `src/effects/effect_copy_spell.cpp`.
- **"Simultaneously" as two instructions (Goblin Welder):** `DB$ SacrificeAll`, then `DB$ ChangeZone`,
  one resolution with no check between, but two events in that order.
- **CR 601.2c / 602.2b — an activated ability's X isn't capped by its legal targets.** Candelabra of
  Tawnos can announce X above the land count; the target loop stops early instead of the activation
  being illegal. Spells have the cap (`spell_xpaid_target_cap`); an X a target's mana value must
  match is held to it (`x_leaves_a_target`, Lazotep Quarry). `src/action_processor.cpp`:
  `run_activation_flow` (X_LADDER), `run_target_select`.
- **CR 702.96a, 702.103a — overload and bestow are offered from the hand only.** An exile grant on
  Cyclonic Rift or Springheart Nantuko, and Bilbo's graveyard cast of Cyclonic Rift, offer the mana
  cost. `src/cast_options.cpp`.
- **CR 118.9a — a resolution's cast offers none of the card's own alternative costs** (morph's
  face-down cast included). `kResolutionCasts` holds the mana cost, what the effect names
  (`PlayCost$`, `WithoutManaCost$`) and, from the graveyard, a free-cast static's row: Bilbo can't
  pitch-cast Force of Will or flash back. `src/cast_options.cpp`.
- **`DB$ Play`'s `Amount$ All` is read as one card** (Beseech the Mirror, the one script, names one).
  `src/parse.cpp`.
- **CR 702.96c, 612 — overload changes a definition, not text.** `overload_spell` reads each
  `ValidTgts$` as `Defined$ Valid` on a face built at parse; an effect that reads no such set is a
  parse finding, so only ChangeZone has it (Cyclonic Rift). `src/queries/overlays.cpp`.
- **CR 702.102b — a fused spell takes timing and additional costs from the left half.**
  `combined_halves` joins names, costs, types, keywords and abilities; a right half's additional,
  alternative or kicker cost isn't carried, and a fused instant + sorcery would read as an instant.
  Wear/Tear has none of these. `src/queries/overlays.cpp`.
- **CR 709.4c — direct type reads see a split card's left half.** Filters, colors and
  `effective_types` read both halves (`card_characteristics`); `card_has_type` on a `CardData` reads
  the face given, so off the stack it is the left half. All vocab split cards have halves of one type.
- **CR 106.5 — a reflected mana ability with no color isn't offered** (Chrome Mox with nothing or a
  colorless card imprinted, Mox Amber with no colored legendary), so it can't be tapped for nothing.
  `src/mana_system.cpp`: `push_mana_source`.
- **CR 707.8a, 712.10 — a copy of a double-faced permanent has one face.** `copied_token` gives a
  token copy the face that's up, so it can't transform; an in-place copy (`copied_card`) likewise.
  Untested.
- **CR 707.2 — a spell copied after it left the stack is read as its whole card** (`copiable_card`),
  so a modal back face cast and countered copies its front. Latent.
  `src/queries/copies.cpp`.
- **CR 614.12a, 707.9 — a token copy makes no "as this enters" choice:** the choice is asked from a
  card's `CardData` in `apply_permanent_components`, so a token copy of Painter's Servant adds no
  color and of Cavern of Souls names no type. Latent. `src/systems/state_manager_statics.cpp`.
- **CR 715.3d — only a mana-cost grant casts an Adventure from exile;** another "you may play it"
  offers Petty Theft, and a free or resource grant the card alone. `exile_adventure_offered`.
- **CR 722.3c — a prepared copy has no card of its own in a frame:** the exiled copy's `card` is the
  preparation card with `transformed` set, flagged as a token (no knowledge-sidecar unit). Nothing in
  the vocab moves a card from exile there; if something did it would cease to exist (CR 704.5e) and
  the permanent would stay prepared with no copy.
- **CR 118.9d — Offspring is offered only on a regular or free-static cast.** Every cast runs the
  same announcement (kicker, replicate, additional life X, Delve, TapToPay), skipping only what
  belongs to an unpaid mana cost: its {X} (0, CR 107.3b) and its hybrid and Phyrexian pips.
  Offspring is a row of the hand cast and the free-cast static's alone, so a card cast from exile
  (Thundertrap Trainer, Agate Instigator) can't pay it. An alternative cost with {X} is refused
  (`cost_form_pays`, a finding). A cost increase on an alternative cost (Thalia on a pitched Force of
  Will) isn't lowered by reductions, and kicker's and replicate's mana join after the SetCost floor.
  `src/action_processor.cpp`: `run_cast_flow` (COST); `src/systems/state_manager_statics.cpp`:
  `floored_alt_mana_cost`.
- **CR 702.120a — escalate:** a further mode is offered only while its escalate cost is still
  payable on top (never made and reversed, CR 601.2e, 733); an alternative cost's pitch is reserved
  by count, not card. Escalate's mana joins after increases, reductions and the SetCost floor (as
  kicker's): an unused reduction doesn't reduce it and Trinisphere counts it on top. Escalate costs
  other than mana and `Discard<N/Filter>` (Collective Effort's tap) are not paid (`cost` findings).
  `src/action_processor.cpp`: `escalate_payable`, `apply_escalate`, `run_cast_flow`
  (ESCALATE_DISCARD); `src/parse.cpp`: `kw_escalate`.
- **CR 601.2f — a target-conditioned cost reduction is offered only through payable targets.**
  Mystical Dispute's reduction applies once targets are chosen; the cast is offered when its cost,
  reduced for the best single legal target, is payable, and each pick offers only candidates whose
  cost is payable, judged one target at a time (latent: it has one). Regular cost branch only.
  `src/action_processor.cpp`: `best_target_cost_reduction`, `target_cost_payable`,
  `apply_target_cost_reduction`; `state_manager_statics.cpp`: `target_cost_reduction_for`.
- **Free-cast statics apply only at priority, from the zones the engine casts from.** Omniscience's
  and Aluren's `MayPlayWithoutManaCost$` is offered from the hand, a cast-from-graveyard permission
  (Emry) and a play-from-exile permission for normal costs (Light Up the Stage); never with another
  alternative cost (CR 118.9a), during a resolution only from the graveyard (Bilbo), from a library
  top under a permission (Mystic Forge), never from the command zone. A MayPlay static that opens a
  zone itself (no `MayPlayDontGrantZonePermissions$`), or another play permission than these, is a
  `static_form` `MayPlay.<zone>` finding. `src/systems/rules_modifying.cpp`: `free_cast_grant`;
  `src/cast_options.cpp`: `free_cast_offered`; `src/parse.cpp`: `resolve_may_play_form`.
- **CR 605.3a — a mana ability that takes another permanent or energy is activated at priority
  alone** (Lazotep Quarry's sacrifice): its mana floats, and no payment taps it, as a payment runs
  blocking (`activate_mana_source`). `src/mana_system.cpp`: `mana_cost_paid_at_priority`.
- **Multi-pip Phyrexian dead end:** each pip's "Pay {C}" is checked against the cost so far, so a
  later pip can find neither 2 life nor mana; the cast is rewound cleanly (601.5). `run_cast_flow`
  (PHYREXIAN_PIP), `fail_cast_payment`.
- **CR 733.1 — a rewind doesn't reverse sacrificed mana sources** (Lotus Petal, Simian Spirit Guide),
  yet the pool is restored, so their mana is lost. Reachable only by the CLI payer's "Cancel
  casting"; the machine payer simulates before committing. `src/mana_system.cpp`:
  `snapshot_mana_state`, `restore_mana_state`.
- **`DB$ Animate`'s `Keywords$` grants nothing for `Duration$ UntilYourNextTurn`.** Permanent grants
  join the rest-of-game keywords (Sneak Attack), end-of-turn ones the until-end-of-turn keywords
  (Shallow Grave); an Enchant keyword (Animate Dead) is never granted. Latent.
  `src/effects/effect_animate.cpp`: `animate_permanent`.
- **Drop of Honey can't choose an indestructible creature tied for least power** (Oracle allows it,
  destroying nothing, CR 702.12b): the engine honours Forge's
  `Creature.leastPower+withoutIndestructible`. With only indestructible creatures at least power
  nothing is destroyed; `leastPower` compares every creature (`src/queries/filters.cpp`).
- **CR 106.6 — a spending restriction reads the printed card** (`CardData`), so a token of the chosen
  type or a permanent with changed types is judged wrongly. Latent. `src/mana_system.cpp`:
  `restricted_mana_matches`, `creature_restricted_mana_matches`,
  `colorless_eldrazi_restricted_mana_matches`.
- **Empower isn't in the checked-in CR (effective 2026-04-17).** From Theorist's Proxy's oracle:
  put N loyalty counters on a token of the subtype you control, creating it first if none. Token and
  counters arrive in one resolution (CR 704.5i never sees 0). With two (one phased in later) the
  controller chooses; a phased-out one isn't there (CR 702.26b); a copy has no counters and dies.
  Matches Forge's `EmpowerEffect` at the pin. A `Type$` with no `kEmpowerTokens` row is a finding.
  `src/effects/effect_empower.cpp`.
- **CR 701.42, 712.4 — a meld half is one plain face.** Phyrexian Dragon Engine's script has no back
  (the meld action is on the unprovisioned Mishra, Claimed by Gix), so nothing melds and it doesn't
  count as double-faced. A meld script with a back stays a `layout` finding. `src/parse.cpp`:
  `parse_card_file`.
- **CR 118.9 — a permission to cast from a graveyard offers the normal cost alone** (Gaea's Will:
  mana cost, either face's, or an Adventure; no pitch or evoke), and a modal double-faced card's land
  back can't be played from there (the land permission reads the front). `src/cast_options.cpp`:
  `kCastSections`; `src/queries/permissions.cpp`.

### 2.6 Zones, libraries and searches
- **CR 601.3e — a cast from the top of a library reads the front face and the face-down face**
  (mana cost, free under a free-cast static, or morph); no back face, split half, Adventure or other
  alternative cost. No vocab card Mystic Forge reaches needs one. `src/cast_options.cpp`: `kCastSections`.
- **CR 701.23 — search and shuffle sequencing** (`src/effects/effect_change_zone.cpp`: `change_zone`):
  1. A multi-card search (Buried Alive) shuffles once per pick (RNG consumption only).
  2. A "may search" (`ShuffleNonMandatory$ True`: Stoneforge Mystic, Ghost Quarter, …) has one
     fail-to-find row, read as declining: no shuffle. `Shuffle$ False` is not read.
  3. The shuffle keys on the first origin zone, so a multi-zone search listing Library second never
     shuffles. Latent (Doomsday has `Shuffle$ False`).
- **CR 701.23b — "any number of cards with the same name" finds them all** (Surgical Extraction, up
  to `ChangeNum$`), with no choice to find fewer. `change_zone_same_name`.
- **CR 111.8 — a departed token is held in place only by ChangeZone:** `change_zone_move` leaves it
  until the SBA removes it (CR 704.5d), so an exile-and-return in one resolution (Displacer Kitten,
  Eldrazi Confluence) returns nothing. `Orderer::add_to_zone` has no guard; no other mover reaches
  such a token. `change_zone_move`.
- **CR 401.4 — cards put into a library together are not ordered by their owner.** ChangeZoneAll
  (Terminus, Atraxa, Endurance) uses entity order (`RandomOrder$` shuffles); opponents see the batch
  order-hidden. Terminus moves permanents one at a time, A's before B's, a token before the owner's
  cards. `src/effects/effect_change_zone_all.cpp`: `move_all_cards`.
- **CR 101.4 / 303.4f — Show and Tell puts cards in sequentially:** the active player's card enters
  before the non-active player chooses, so a non-active Aura can enchant the just-entered creature
  and "enters" triggers see an order. `each_player_put_from_hand`.
- **Until-your-next-turn Animate reversal:** `revert_until_turn_animates` clears the rest-of-game
  animate fields and base P/T and strips Creature/Damage when the printed card isn't a creature, even
  if another Animate still applies. Latent (Karn, the Great Creator's +1 on an animated artifact).
  `src/effects/effect_animate.cpp`.
- **CR 404.3 — cards put into a graveyard together stay in move order,** unarranged: a mill or sweep
  one by one, a hand discarded at once in hand order (`discard_hand`, `src/cost_payment.cpp`;
  `Mode$ Hand`, `src/effects/effect_discard.cpp`). Only Shallow Grave reads the order.
  `src/systems/orderer.cpp`: `add_to_zone`.

### 2.6a Reveal windows (`src/queries/visibility.cpp`)
Reveal windows (`Game::reveal_windows`) feed the state format's `reveal` table and visibility
(`docs/state_format.md`): these change what a viewer is shown, not how the rules resolve.
- **CR 708.9 — a face-down object leaving is revealed by its move** (`zone_move`), no `reveal_start`
  first. `src/queries/face_down.cpp`: `reveal_as_it_leaves`.
- **CR 702.139a / 103.2b — the companion stays revealed** to both players while outside the game.
  `src/companion.cpp`: `setup_companion_for`.
- **CR 701.23a / 406.3 — the found card after the pre-move shuffle:** the shuffle ends the search's
  looks (CR 701.20d); the searcher's look at the found card is reopened, so a card then exiled face
  down (The Creation of Avacyn I) keeps its look. `change_zone`'s search loop.
- **CR 733.1 / 702.94b — a reversed cast doesn't restore a reveal** (`rewind_cast_move`). Reachable
  only by cancelling a miracle cast under `--offer-cancel` (a testing aid): the card would stay
  revealed until the miracle trigger leaves the stack. `src/systems/orderer.cpp`.
- **CR 401.5, 701.20e — a standing look is taken at decisions.** `sync_standing_looks` moves the
  window as each decision is asked, so a card on top only mid-resolution is never looked at, and a
  card a put-back buried stays looked at until the next decision. A seat may know less than it
  could, never more. No vocab card asks for that look. `src/queries/visibility.cpp`:
  `sync_standing_looks`; `src/input_logger.cpp`: `InputLogger::get_input`.

### 2.7 Script shortcuts that bypass the card's own tags
Two cards are recognised by a script fingerprint and handled by special-case code, against
`CLAUDE.md`'s "Parse script tags as intended". Each fingerprint matches one script today; a
different card with it would silently get the same behavior.
- **Sylvan Library** (CR 608.2c). Script: `AB$ ChooseCard` (`Cost$ Draw<2/You>`, `Amount$ 2` over
  `Card.YouOwn+DrawnThisTurn`), `DB$ RepeatEach` over `Card.ChosenCard`, body `DB$ ChangeZone` to the
  library with `UnlessCost$ PayLife<4>`. Engine: a trigger whose `Execute$` body has both
  `ChooseCard` and `DrawnThisTurn` becomes a synthetic `SylvanLibrary` category with a hard-coded
  handler. Fix: those three forms generically, then delete it. `src/parse.cpp`: `parse_one_trigger`;
  `src/effects/effect_sylvan_library.cpp`.
- **Green Sun's Zenith** (CR 608.2c, 608.2n). Script: `DB$ ChangeZone | Origin$ Stack | Destination$
  Library | Shuffle$ True | Defined$ Parent` moves the resolving spell. Engine: an SVar with those
  tags sets `CardData::shuffle_into_library` and is stripped; the resolved spell then goes to its
  owner's library, shuffled. Fix: read `Defined$ Parent` and the shuffle as a resolution step
  (`Spell::resolves_to` already carries a spell's own destination). `src/parse.cpp`:
  `parse_card_face_body`; `src/queries/spells.cpp`: `spell_destination`.

### 2.8 Script params of vocab cards the parser does not read
`robomage --parse-report @vocab` (`src/parse_report.h`) lists what the parser meets in a vocab
card's script and models nowhere; the findings are recorded in `tests/parse_report/vocab.json`
(checked by the `vocab` tier). A fix removes the finding there and the entry here.
- **Ajani, Nacatl Pariah — `ability_param:ImmediateTrigger.TriggerAmount`.** `TriggerAmount$
  Remembered$Amount` on his back face's reflexive ability; it triggers once, which is right, as the
  ability makes one token.

---

## 3. Out of scope because no vocab card needs it

Unimplemented, or implemented but unexercised, because no card in `src/card_vocab.h` needs them. A
candidate deck's needs are listed by the onboarding pipeline (`tools/onboard/README.md`,
`bin/resources/decks/candidates/REPORT.md`).

- **Toxic** (CR 702.164, 120.3g). Poison counters and the 704.5c SBA exist.
- **Battles** (CR 310, 120.3h): a permanent card type only; can't be damaged, attacked or defeated.
- **Infect / wither** (CR 702.90, 702.80): damage-result paths exist (`src/components/damage.cpp`),
  untested.
- **Handlers limited to the vocab's forms:** `gain_life` evaluates only `Targeted$CardPower` (other
  `LifeAmount$` expressions read the literal); `multiply_counter` always doubles +1/+1 counters,
  ignoring `CounterType$`; `parse_activation_cost` passes an unrecognised cost token to the mana
  parser (a `cost` finding).
- **Modal forms beyond "choose N" and "choose one or more"** (CR 700.2e–i): a mode chosen by another
  player, per-mode additional costs and pawprint modes. Only `CharmNum$`, `MinCharmNum$` (a
  CONFIRM_MODES row ends the choice; `0` reads as 1) and `CanRepeatModes$` (CR 700.2d) are read.
  Modes are announced only at the root of a stack object; a chained `DB$ Charm` would be fatal.
  `src/announce.cpp`: `run_announce`; `src/effects/effect_charm.cpp`.
- **Regeneration** (CR 701.19); `NoRegen$ True` (Snuff Out) is accepted and ignored
  (`src/parse.cpp`: `apply_param_to_ability`).
- **Face-down objects (CR 708):** Morph and Cloak only (`src/queries/face_down.h`); manifest,
  disguise and megamorph aren't built. CR 701.58g isn't: a cloaked instant or sorcery has no row (only
  `SetState`'s `TurnFaceUp`, for The Creation of Avacyn's exiled card). A cloaked creature with
  Phyrexian pips would pay them in color (`turns_up_as_cloaked`). No non-mana or X morph costs (CR
  702.37f), no "when turned face up" (CR 708.11). Showing the cost to all players (CR 702.37e,
  701.58b) isn't a step of its own. CR 708.5's control change is unexercised. A token can't be turned
  face down (`turn_face_down`). A card exiled face down keeps its printed characteristics for readers
  (CR 406.3a: none); only its identity is hidden. Two objects entering face down together show their
  entity order in timestamps (`docs/state_format.md` §6).
- **No machinery:** mutate, banding, splice, entwine, cumulative upkeep, dungeons/initiative.
