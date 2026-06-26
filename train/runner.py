"""The single observation/decision loop shared by `train.py observe` and the
test harness.

There is exactly one game-driving loop in the project, here. Callers supply two
``Controller`` objects (scripted / model / action-list / interactive / auto-pass)
and the state-seeding parameters (decks, seed, pre-set battlefield, no-shuffle);
this module runs the engine via :class:`NarrativeEnv`, renders the unified
transcript, and tallies results. The test harness's job is reduced to *seeding
the state* (building stacked decks, battlefields, scenarios) and picking the
controllers — it no longer reimplements the loop.

Dependency-light on purpose (numpy + env + decode + the generated enum tables);
torch is only pulled in if a caller passes a model ``Controller``.
"""

import random

import numpy as np

import decode
from env import NarrativeEnv, STATE_SIZE, ACTION_CATEGORY_MAX, BINARY
from _enums import _CAT_NAMES, _STEP_NAMES


def _compact_line(decision, player, label, step_name, cats, action):
    """One-line RL-debug summary of a decision (the non-verbose default)."""
    chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
    all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]
    return (f"[{decision:4d}] P{player} ({label}/{player})  {step_name:<14}"
            f"  choices={len(cats)}  available={all_cats}  -> {action} ({chosen_cat})")


def run_games(controller_a, controller_b, *,
              label_a="A", label_b="B",
              binary_path=BINARY, deck_a=None, deck_b=None,
              n_games=1, bo3=False, seed=None, verbose=False,
              battlefield_a=None, battlefield_b=None,
              graveyard_a=None, graveyard_b=None, no_shuffle=False,
              max_decisions=None):
    """Run ``n_games`` between two controllers and render the transcript.

    ``controller_a``/``controller_b`` are :class:`opponents.Controller` objects;
    pass the *same* object for both sides for a single global decision-maker
    (the harness's action-list / interactive / auto-pass modes).  ``label_a`` /
    ``label_b`` are the display labels for each side ("Scripted", "Model", ...).

    Returns ``(wins, losses, draws)`` from Player A's perspective. A game that
    ends with no winner (e.g. the engine's step cap — a stall) counts as a draw
    and its full log is saved to ``draw_<n>.txt``; a game stopped early by
    ``max_decisions`` is reported as incomplete and not counted.
    """
    wins = losses = draws = incomplete = 0
    unit = "match" if bo3 else "game"

    for i in range(n_games):
        env = NarrativeEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                           bo3=bo3, battlefield_a=battlefield_a,
                           battlefield_b=battlefield_b,
                           graveyard_a=graveyard_a, graveyard_b=graveyard_b,
                           no_shuffle=no_shuffle)
        obs, _ = env.reset(seed=(seed + i) if seed is not None else None)
        # Seed Python's global RNG with the SAME per-game seed passed to the engine.
        # The scripted agent breaks ties on ambiguous OTHER_CHOICE prompts with
        # random.choice (e.g. the "pay {1} or be countered" / Sylvan Library
        # pay-or-return prompts); without this seeding those picks varied between
        # processes, making otherwise-deterministic scripted-vs-scripted games
        # non-reproducible across builds. Tying it to the engine seed makes the full
        # game deterministic.
        if seed is not None:
            random.seed(seed + i)
        done = False
        capped = False
        total_reward = 0.0
        decision = 0
        turn = 0
        prev_active_is_a = None       # None until the first non-mulligan query
        known_hand = {"A": [], "B": []}
        log_lines = []                # buffered so a draw can be dumped to file

        def emit(line):
            log_lines.append(line)
            print(line, flush=True)

        def emit_narrative(lines):
            if not lines:
                return
            block = decode.format_narrative_block(lines) if verbose else lines
            for ln in block:
                emit(ln)

        if n_games > 1:
            emit(f"--- {unit.capitalize()} {i + 1}/{n_games} ---")

        while not done:
            emit_narrative([ln for ln in env.flush_lines() if ln.strip()])

            if max_decisions is not None and decision >= max_decisions:
                capped = True
                break

            num_choices = env._num_choices
            priority_is_a = obs[32] > 0.5
            player = "A" if priority_is_a else "B"
            active_is_a = (obs[31] > 0.5) == priority_is_a
            step_name = _STEP_NAMES[int(np.argmax(obs[18:31]))]
            cats = np.round(
                obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
            is_mulligan = any(c == 11 for c in cats)

            known_hand[player] = decode.decode_hand(obs)

            if not is_mulligan and active_is_a != prev_active_is_a:
                turn += 1
                emit(f"--- Turn {turn} (Player {'A' if active_is_a else 'B'}) ---")
                emit(f"  PA: {', '.join(known_hand['A']) or '(empty)'}")
                emit(f"  PB: {', '.join(known_hand['B']) or '(empty)'}")
                prev_active_is_a = active_is_a

            controller = controller_a if priority_is_a else controller_b
            label = label_a if priority_is_a else label_b

            # Decode the menu once if either the transcript (verbose) or the
            # controller (e.g. PlayController, which matches specs against it)
            # needs it; otherwise skip the work for the model/scripted path.
            need_decoded = verbose or getattr(controller, "wants_decoded", False)
            decoded = (decode.decode_actions_from_obs(
                obs, num_choices, getattr(env, "_action_public", None),
                descriptions=getattr(env, "_action_descriptions", None))
                if need_decoded else None)

            action = controller.choose(obs, num_choices,
                                       action_masks=env.action_masks(),
                                       decoded_actions=decoded)

            if verbose:
                gs = decode.decode_game_state(obs[:STATE_SIZE])
                for ln in decode.format_decision_block(decision + 1, gs, decoded):
                    emit(ln)
                emit(decode.format_chosen_action(f"{label}/{player}", action, decoded))
            else:
                emit(_compact_line(decision, player, label, step_name, cats, action))

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            decision += 1

        emit_narrative([ln for ln in env.flush_lines() if ln.strip()])
        env.close()

        if capped:
            incomplete += 1
            print(f"\n=== stopped after {decision} decisions "
                  f"(max-decisions cap; no winner) ===", flush=True)
        elif total_reward > 0:
            wins += 1
            result = f"{label_a}/A wins"
        elif total_reward < 0:
            losses += 1
            result = f"{label_b}/B wins"
        else:
            draws += 1
            result = "DRAW (should not occur)"
            log_path = f"draw_{i + 1}.txt"
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines) + "\n")
            emit(f"\n=== DRAW (should not occur) — full log saved to {log_path} ===")

        if not capped:
            if n_games > 1:
                print(f"  {unit} {i + 1:2d}/{n_games}: {result}  "
                      f"(W:{wins} L:{losses} D:{draws})", flush=True)
            else:
                print(f"\n=== {result} ===", flush=True)

    if n_games > 1:
        total = wins + losses + draws
        win_pct = 100 * wins / total if total else 0
        extra = f", {incomplete} incomplete" if incomplete else ""
        print(f"\n{wins}W / {losses}L / {draws}D over {n_games} {unit}"
              f"{'es' if bo3 else 's'} ({win_pct:.1f}% Player A win rate){extra}",
              flush=True)

    return wins, losses, draws
