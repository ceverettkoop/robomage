"""Reconstruct browsable game records from AZ self-play shards.

The AZ trainer's shards (``train/az_data/gen/shard_*.npz``: parallel arrays
``obs (N, OBS_SIZE)``, ``pi (N, MAX_ACTIONS)``, ``z (N,)``, ``mask (N,
MAX_ACTIONS)``, plus the n-step TD columns ``q``/``explored``/``td_q (N,)``
this module does not need — one row per SEARCHED decision root, in
game-backfill order) carry no game/match ids, yet every row's obs embeds the
bo3 match context
(game number, win counters, sideboard flag) and the mover's seat. This module
recovers that structure and re-packs the rows into the per-game trace dicts
``analysis._collect_game_traces`` produces, so ``tui_analysis.py`` can browse
recorded self-play exactly like freshly simulated games (``--shards``).

The record unit is a full bo3 MATCH trace (matching the analysis collector,
whose env episode spans the match): steps are the VIEWPOINT seat's rows, the
other seat's rows become ``opp_actions`` one-liners, ``result`` is the match
outcome, and ``pi`` (the search's visit posterior) stands in for the policy
distribution. ``engine_seed``/``full_actions``/``prefix_len`` stay None for
training-pool shards, so the browser's whatif/replay paths self-disable
gracefully — but a GUI-play recording whose ``.rmplay`` sidecars are present
(see :func:`load_replay_sidecars`) gets them filled with the REAL seed and
action log, making those records exactly replayable (browser replay-to-step
MCTS analysis included).

Deliberately torch-free at import: numpy + the env layout constants + decode.
Net-based V(s) (``load_value_model``/``eval_values``) imports analysis/torch
lazily, only when a value net is actually requested.

Known reconstruction limits (inherent to the shard format, surfaced in the UI):
  * The played action is ``argmax(pi)`` — exact wherever self-play picked
    ``best_action()``, but each game's first ``temp_moves`` (default 20)
    searched decisions were SAMPLED from ``pi``, so there argmax is only the
    most probable reconstruction.
  * TRAINING-POOL shards record only searched decisions, so step sequences
    have gaps at the residual unsafe roots. (GUI-play recordings do NOT have
    this gap: shard_record writes every >1-choice decision, the unsearched
    ones as one-hot behavior rows with ``q = NaN``.)
  * Action labels are rebuilt from the obs metadata blocks (the engine's
    narrative description side-channel is not stored).
"""

import glob
import os

import numpy as np

import decode
from env import (MAX_ACTIONS, OBS_SIZE, _CUR_TURN_IDX, _IS_SIDEBOARD_IDX,
                 _MATCH_CTX_START, _SELF_IS_A_IDX)


def shard_sort_key(path):
    """Order pooled shards by write order: (mtime, numeric suffix n) from the
    shard_{ts}_{pid}_{n}.npz name (a lexicographic sort misorders n>=10)."""
    base = os.path.basename(path)
    try:
        n = int(base.rsplit("_", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        n = 0
    return (os.path.getmtime(path), n)


def load_shard_rows(data_dir):
    """Load every ``shard_*.npz`` under ``data_dir`` in WRITE order.

    Returns ``(obs, pi, z, mask, spans)`` where ``spans`` is a list of
    ``(path, start, end)`` row ranges, one per shard file. Rows are only
    contiguous per game WITHIN a file (each file is one worker's buffer), so
    segmentation must never cross a span boundary.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                   key=shard_sort_key)
    if not paths:
        raise FileNotFoundError(f"no shard_*.npz files in {data_dir}")
    obs, pi, z, mask, spans = [], [], [], [], []
    row = 0
    for p in paths:
        d = np.load(p)
        n = d["obs"].shape[0]
        if d["obs"].shape[1] != OBS_SIZE or d["mask"].shape[1] != MAX_ACTIONS:
            raise RuntimeError(
                f"shard layout mismatch in {p}: obs width {d['obs'].shape[1]} "
                f"(expected {OBS_SIZE}), mask width {d['mask'].shape[1]} "
                f"(expected {MAX_ACTIONS}) — regenerate the shards with this build")
        obs.append(d["obs"]); pi.append(d["pi"]); z.append(d["z"]); mask.append(d["mask"])
        spans.append((p, row, row + n))
        row += n
    return (np.concatenate(obs), np.concatenate(pi), np.concatenate(z),
            np.concatenate(mask), spans)


def _row_game_number(obs_row):
    return int(round(float(obs_row[_MATCH_CTX_START]) * 3.0))


def _row_turn(obs_row):
    return int(round(float(obs_row[_CUR_TURN_IDX]) * 50.0))


def segment_matches(obs, spans):
    """Split shard rows into match segments.

    Returns a list of matches; each match is a list of games; each game is a
    list of row indices. The engine stamps bo3 sideboard prompts with the
    UPCOMING game's number, so a sideboard run leads the game segment it
    sideboards into — which also matches az_selfplay's pricing (a sideboard
    sample's z is that next game's outcome). Segmentation never crosses a
    shard-file boundary. Within a file, a game_number increase starts the next
    game of the same match; a game_number decrease starts a new match; a turn
    reset without a game_number change (a bo1-style file, or two matches whose
    final/first games share a game_number) starts a new match too.
    """
    matches = []
    for _path, start, end in spans:
        cur_match, cur_game = [], []
        prev_gn = prev_turn = None
        prev_sb = False
        for i in range(start, end):
            gn = _row_game_number(obs[i])
            turn = _row_turn(obs[i])
            sb = bool(obs[i, _IS_SIDEBOARD_IDX] > 0.5)
            if prev_gn is not None:
                new_game = gn > prev_gn
                # Turn resets mid-game-number: a fresh game 0 following a
                # match that ended on game 0, or the first game after a
                # sideboard run keeping the finished game's number would have
                # bumped gn instead — so this only fires across matches.
                new_match = (gn < prev_gn
                             or (gn == prev_gn and turn < prev_turn
                                 and not sb and not prev_sb))
                if new_match:
                    cur_match.append(cur_game)
                    matches.append(cur_match)
                    cur_match, cur_game = [], []
                elif new_game:
                    cur_match.append(cur_game)
                    cur_game = []
            cur_game.append(i)
            prev_gn, prev_turn, prev_sb = gn, turn, sb
        if cur_game:
            cur_match.append(cur_game)
        if cur_match:
            matches.append(cur_match)
    return matches


def _match_result(games, obs, z, viewpoint_is_a):
    """Match outcome (+1/-1/0) for the viewpoint seat: the winner of the most
    games, each game's winner read from the first row's mover-perspective z
    (a leading sideboard row works too — its z is priced on this same game)."""
    won = lost = 0
    for rows in games:
        gw = 0.0
        for i in rows:
            zi = float(z[i])
            if zi == 0.0:
                continue
            mover_is_a = bool(obs[i, _SELF_IS_A_IDX] > 0.5)
            gw = zi if mover_is_a == viewpoint_is_a else -zi
            break
        won += gw > 0
        lost += gw < 0
    if won == lost:
        return 0.0
    return 1.0 if won > lost else -1.0


def build_match_records(obs, pi, z, mask, matches, viewpoint_is_a=True,
                        interp_fn=None, replay_docs=None):
    """Pack match segments into analysis-schema trace dicts.

    Steps are the viewpoint seat's rows; the other seat's rows are summarized
    into ``opp_actions``. ``values`` defaults to each step's game-outcome z
    (a flat per-game bar) — callers with a value net overwrite it via
    ``apply_net_values``. ``interp_fn`` (e.g. ``analysis._extract_interpretable``)
    fills ``interp_features``; None leaves them empty (the turning/clusters/
    shap views then have no data, everything else is unaffected).

    ``replay_docs`` (parallel to ``matches``, from :func:`load_replay_sidecars`)
    supplies each match's recorder sidecar: when present, the record gains the
    real ``engine_seed``/``full_actions``/``prefix_len`` (plus the absolute
    ``deck_a``/``deck_b``/``bo3_match`` under ``replay_decks``), so the
    browser's replay-to-step paths (whatif, MCTS analysis) work on the
    recording; without one they stay None and self-disable as before.

    Matches where the viewpoint seat never held a searched root are skipped
    (nothing to page through).
    """
    records = []
    for mi, games in enumerate(matches):
        doc = replay_docs[mi] if replay_docs else None
        steps, vals, zs, interp, actions, num_choices, probs, opp = \
            [], [], [], [], [], [], [], []
        g_prefix = []
        for rows in games:
            for i in rows:
                n = int(mask[i].sum())
                best = int(np.argmax(pi[i]))
                if bool(obs[i, _SELF_IS_A_IDX] > 0.5) == viewpoint_is_a:
                    steps.append(np.array(obs[i]))
                    mover_z = float(z[i])
                    vals.append(mover_z)
                    zs.append(mover_z)
                    actions.append(best)
                    num_choices.append(n)
                    probs.append(pi[i, :n].astype(np.float64))
                    if doc is not None:
                        g_prefix.append(doc["row_prefix"].get(i))
                    if interp_fn is not None:
                        interp.append(interp_fn(obs[i]))
                else:
                    acts = decode.decode_actions_from_obs(obs[i], n)
                    desc = acts[best]["description"] if best < n else f"#{best}"
                    opp.append({"desc": desc, "before_model_step": len(steps),
                                "clock": None})
        if not steps:
            continue
        replayable = (doc is not None and g_prefix
                      and all(p is not None for p in g_prefix))
        records.append({
            "observations": steps,
            "values": vals,
            "z": zs,          # exact per-step game outcome (values may be
            #                   overwritten by net V(s); this never is)
            "interp_features": interp,
            "actions": actions,
            "num_choices": num_choices,
            "action_probs": probs,
            "opp_actions": opp,
            "engine_seed": doc["engine_seed"] if replayable else None,
            "full_actions": list(doc["actions"]) if replayable else None,
            "prefix_len": g_prefix if replayable else None,
            "replay_decks": ({"deck_a": doc["deck_a"],
                              "deck_b": doc["deck_b"],
                              "bo3": doc["bo3"]} if replayable else None),
            "result": _match_result(games, obs, z, viewpoint_is_a),
            "model_is_a": viewpoint_is_a,
            "clock_remaining": None,
            "clock_bank": None,
            "opp_clock_bank": None,
            "shard": True,      # marks a reconstructed trace (UI caveat footer)
        })
    return records


def load_replay_sidecars(spans, matches):
    """Match each segmented match to its recorder ``.rmplay`` sidecar.

    A GUI-play recording (``shard_record.ShardRecorder`` with replay wiring)
    writes one same-stem ``.rmplay`` per shard file: engine seed, the match's
    full action list, and ``row_decision_idx`` mapping each shard row to its
    replay position. Returns a list parallel to ``matches``: None (no/invalid
    sidecar, or the file segmented into more than one match — then its rows
    cannot be attributed) or a dict with ``engine_seed``/``actions``/
    ``deck_a``/``deck_b``/``bo3`` and ``row_prefix`` rebased to GLOBAL row
    indices. Training-pool shards have no sidecars and load as all-None."""
    import gui_session_io

    per_span = {}
    for path, start, end in spans:
        side = os.path.splitext(path)[0] + gui_session_io.PLAY_EXT
        if not os.path.exists(side):
            continue
        try:
            doc = gui_session_io.load_replay(side)
        except (OSError, ValueError):
            continue
        idx = doc.get("row_decision_idx")
        if not isinstance(idx, list) or len(idx) != end - start:
            continue    # mid-update skew or foreign file: not trustworthy
        per_span[(start, end)] = doc

    # A span (file) maps cleanly only when it holds exactly one match.
    span_matches = {}
    for mi, games in enumerate(matches):
        first = games[0][0]
        for (start, end) in per_span:
            if start <= first < end:
                span_matches.setdefault((start, end), []).append(mi)

    docs = [None] * len(matches)
    for (start, end), mis in span_matches.items():
        if len(mis) != 1:
            continue
        doc = per_span[(start, end)]
        idx = doc["row_decision_idx"]
        docs[mis[0]] = {
            "engine_seed": doc["engine_seed"],
            "actions": doc["actions"],
            "deck_a": doc.get("deck_a"), "deck_b": doc.get("deck_b"),
            "bo3": bool(doc.get("bo3", True)),
            "row_prefix": {start + k: int(p) for k, p in enumerate(idx)},
        }
    return docs


def load_records(data_dir, viewpoint_is_a=True, limit=None, interp_fn=None):
    """The one-call front door: shards on disk -> browsable match records."""
    obs, pi, z, mask, spans = load_shard_rows(data_dir)
    matches = segment_matches(obs, spans)
    records = build_match_records(obs, pi, z, mask, matches,
                                  viewpoint_is_a=viewpoint_is_a,
                                  interp_fn=interp_fn,
                                  replay_docs=load_replay_sidecars(spans,
                                                                   matches))
    if limit is not None:
        records = records[:limit]
    return records


# ── Optional value net (lazy torch) ──────────────────────────────────────────

def load_value_model(spec):
    """Load the V(s) net for a model spec ('gen', a .zip/.pt path, az:/azraw:).

    Reuses analysis.py's loaders, so PPO checkpoints and AZNets both come back
    as an object exposing ``policy.predict_values(obs_t)``. Torch imports stay
    inside this call.
    """
    import analysis as an
    insp = an._inspection_spec(spec)
    if an._is_az_model_spec(insp):
        model, _path = an._load_az_analysis_model(insp)
        return model
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO
    return MaskablePPO.load(an._resolve_any_path(insp))


def apply_net_values(model, records, batch_size=256):
    """Overwrite each record's z-based ``values`` with the net's V(s) per step."""
    import torch
    for g in records:
        vals = []
        for lo in range(0, len(g["observations"]), batch_size):
            chunk = np.asarray(g["observations"][lo:lo + batch_size],
                               dtype=np.float32)
            obs_t = torch.as_tensor(chunk)
            with torch.no_grad():
                v = model.policy.predict_values(obs_t)
            vals.extend(float(x) for x in v.reshape(-1).cpu().numpy())
        g["values"] = vals
