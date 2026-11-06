# Plan: the first open-vocabulary training run, with recent-event history

Goal: one R-NaD run over every playable deck, starting from the existing checkpoints, with the seat's
recent events as an input. It starts after `docs/mechanics_plan.md` M13: the engine, vocab and state format
must stop moving first, since a run's actors load them on every restart.

**Decided (2026-11-06).**

- History is on: `FeatureOptions::history_events` (`docs/featurizer.md` §3.7), the events of this turn and the
  previous one, newest first, at most 32, each with up to 8 cards. "Previous" is one turn back, which is the
  opponent's last turn from your own. That is the window wanted, so `EventHistory` doesn't change.
- A short paired run comes first: 5k steps on the ten original decks from one seed, history on against off, judged by
  `value_mse_gap`, held-out value MSE and a paired bench. It needs §2's items 1 to 3 and nothing from M13's decks.
- Old episodes are gone (`runs/*/episodes`, `runs/bc1/corpus/episodes`, 280 GB). The run needs none: R-NaD
  learns from its current policy. Checkpoints, decision logs, ladders and benchmarks stay. Gone with them:
  `run.corpus_dir` reuse of the B0 corpus (a run re-records it), the DAgger labels, and
  `runs/pilotB-prep/pretrain_critic.py`'s input, so a pilot B needs new B0 games.
- From `docs/open_vocab_plan.md`, unchanged: new decks enter through self-play from the current policy, with no
  teacher, no behaviour cloning and no deck-specific scripted logic. Open decklist. No archetypes.

## 1. Starting point

`run.start_from` the migrated `runs/league24h-sb/ckpt/fixed/m11.pt`, straight into B4 self-play, with `runs/pilotA.toml`
as the template. No new BC: hist1 (`docs/rnad_training_plan.md` §7) tested history under BC against a scripted
teacher that doesn't read history, so BC could not reward it, and its value memorisation comes from multi-pass
training. History is learned in R-NaD.

The checkpoint already fits: `layout_hash` is the same with and without `--history`, which is what a
checkpoint's `check_compatible` compares. In `m11` every history input is zero: `num_proj` over `hist_event` and
`hist_card`, their `type_emb` rows, their 101 `cat_emb` rows. So history tokens start silent, with one exception:
a zero-embedded token is still a real token, so it takes attention and counts in `Summary`'s mean (about 30
tokens on some 120). Step 3 of §2 measures that shift.

## 2. Work before the run

1. **`rmtrain` has no history option.** Only `benchmark.py` and `evaluate.py` take `--feature-history`. Add a run key
   (`run.feature_history`, not live, fixed at the start) carried to the tables export (`rmfeat --history`), the actors'
   schedule, `nn:`, evaluation, the ladder and `bench`. Check `retable`, `add-decks` and `accepted_spec_hashes` with
   it on: a history spec's hash differs from the checkpoint's tables. `docs/featurizer.md` §3.7's "refused" should be
   checked against what `load_checkpoint` does today, and the text corrected.
2. **`make check` with the opt-in tier.** Build `make actor`, set `$ROBOMAGE_TORCH_PYTHON`, and run the `actor` tier: it
   hasn't run during the mechanics plan. Add a history-on case: `nn:` bo3 games, a `--selfcheck-featurize` pass, an
   exact re-featurization of a `--replay`, and `check_rmtrain.py --smoke` with the key on.
3. **Entry shift.** On the open-vocab fixtures, `m11` with history on against off: max |Δ logit|, |Δ value| and how often the
   argmax moves. Then a paired `rmtrain bench` of both. Bar: no regression beyond two SEs. If the shift fails it,
   leave history tokens out of `Summary`'s mean at first, or raise their share with an `event_dropout` from 1 to 0 over
   the opening steps (BC has the option; R-NaD doesn't). Either is a model change, made once, before the run.
4. **Acceptance.** `learn/open_vocab_acceptance.py full`, with `tests/open_vocab/s0_rows.json` holding every card M7 to M13
   drifted, from the card digest's diff.
5. **Throughput.** History cost about 25% more per BC step and 9% more per episode. Measure an R-NaD step and the
   actor loop with it on, with the new decks' longer games if they are longer.

## 3. The league

Every onboarded deck but the held-out ones: up to about 35 decks, some 600 pairs against the 45 of the last run.
`add-decks`' mean weight doesn't scale that far, and pilotA's fair/combo weights name decks by hand.

Default, to confirm: every pair weight 1, combo-vs-combo 0.5 as in pilotA, written as explicit `league.pairs`. Start with all
decks at once, not in waves: an unseen card reads `unknown_id` until its first batch, and each batch's ledger marks both
lists, so all cards are seen within the first few steps. The zero-shot measure below comes from the held-out decks,
not from the league.

Sideboarding: `sb_explore` as in the last run. `scripted:hard+sb`'s guides cover the league matchups only, which matters
only for benchmarks against it.

## 4. Measuring it

- **Before training:** `rmtrain bench --decks league,heldout` on `m11`, the zero-shot baseline for every new deck.
- **During:** at fixed points, the same bench on a sampled schedule. 78 pairs take about 7 minutes a policy, so
  600 pairs need sampling: every held-out pair, every pair of the ten original decks, a seeded sample of the rest,
  the same sample each time so the common random numbers hold.
- **Generalisation** (`docs/open_vocab_plan.md` S8): `plays:heldout` and `faces:heldout`, paired against `m11`.
  Head-to-head ladders over `scripted:hard`, which pilots new decks with generic heuristics.
- **Memorisation:** `value_mse_gap`, with held-out value MSE.
- **No regression:** `within:league:original` against `m11`.

**Budget and stopping.** `league24h-sb` took 14,300 learner steps and 700k games in 24 hours, and needed about 13k
steps for +0.04. S8's minimum to see anything is 5k steps. With ten times the pairs and a quarter more cost per step,
plan on at least 20k steps, about two days, with a review at each 5k. Stop on a regression beyond two SEs in
`within:league:original` with no gain on the held-out decks, or when held-out gains stay inside two SEs over two
reviews.

## 5. Decisions needed

1. Held-out decks, before M13: at least one fair and one combo. `bant_control` and `show_and_tell` are held out today, but
   the candidates overlap them: `uwx_control` shares 48 of 60 main-deck cards with `bant_control`, `stoneblade` 44, `aluren`
   39 with `show_and_tell`. A held-out deck says little if a near copy trains. Low-overlap candidates: `storm` (combo, at most
   14 shared with any deck) and `dragon_stompy` (fair, at most 11).
2. The defaults in §3 and §4.

After this run, by the same paired benchmarks: S6, the in-model deck summary, and `id_dropout` 0.25, as
`docs/open_vocab_plan.md` S8 sets out. A recurrent core only if history shows a gain worth extending.
