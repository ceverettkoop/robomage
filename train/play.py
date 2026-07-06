"""
Play interactively against a trained RoboMage model.

The model is randomly assigned to Player A or B each game (pin with --player).
Text mode runs on the shared runner loop with a HumanController seat — enter
an action number or a semantic spec ('cast:bolt', 'pass'); --seed reproduces a
game. The --tui path delegates to tui_game.py; --gui drives the deprecated
raylib front end.

Usage:
    train/.venv/bin/python train/play.py --human-deck delver --model-deck burn
"""

import argparse
import subprocess
import numpy as np

from env import (ACTION_CATEGORY_MAX, BINARY, MAX_ACTIONS,
                 BIN_DIR, _ACTION_CARD_ID_NULL,
                 _HAND_START, MAX_HAND_SLOTS,
                 _BQUERY_STATE_BYTES, _BQUERY_CATS_BYTES, _BQUERY_IDS_BYTES, _BQUERY_CTRL_BYTES,
                 _BQUERY_PUB_BYTES,
                 _HAND_SLOT_SIZE, _BF_ID_IDX, _gather_costs)
from decode import card_from_id as _card_from_id

try:
    from card_costs import (_CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS)
except ImportError:
    from train.card_costs import (_CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS)

try:
    from sb3_contrib import MaskablePPO
    USE_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKABLE = False

# Mana-tap action categories → produced color (used by the GUI action labels).
_MANA_CAT_COLOR = {13: "W", 14: "U", 15: "B", 16: "R", 17: "G", 18: "C"}


# ── Decode helpers ────────────────────────────────────────────────────────────

# Card-name decoding (vocab lookup + Token sentinel + out-of-range handling)
# lives in decode.card_from_id — the single source of truth. Use _card_from_id.

def _action_label(cat: int, card_id_float: float) -> str:
    card = _card_from_id(card_id_float)
    if cat == 0:  return "pass priority"
    if cat == 2:  return f"attack with {card or '?'}"
    if cat == 3:  return "confirm attackers"
    if cat == 4:  return f"block with {card or '?'}"
    if cat == 5:  return "confirm blockers"
    if cat == 6:  return f"activate {card or '?'}"
    if cat == 7:  return f"cast {card or '?'}"
    if cat == 8:  return "select target"
    if cat == 9:  return f"play {card or '?'}"
    if cat == 10: return f"choose {card}" if card else "choose"
    if cat == 12: return f"bottom {card}" if card else "bottom card"
    if cat in _MANA_CAT_COLOR:
        return f"tap {card or '?'} for {{{_MANA_CAT_COLOR[cat]}}}"
    if cat == 19: return "fail to find" if card is None else f"find {card}"
    return f"action {cat}"


# ── Main play loop (text mode) ────────────────────────────────────────────────

def play(binary_path: str, model_path: str, human_deck: str = "delver",
         model_deck: str = "delver", human_player: str = None, seed: int = None):
    """Text-mode game against a trained model, on the shared runner loop.

    The human seat is an :class:`opponents.HumanController` — it renders the
    board and legal menu each decision and accepts an action number, a
    semantic spec (``cast:bolt``, ``target:bears@opp``, ``pass`` — the same
    grammar as the harness ``--play``), or 'quit'. The model's choices are
    announced as they happen; game narrative comes from the engine.
    """
    import runner
    from opponents import HumanController, ModelController

    model = MaskablePPO.load(model_path)

    if human_player is None:
        model_is_a = bool(np.random.random() < 0.5)
    else:
        model_is_a = human_player == "B"
    model_role = "A" if model_is_a else "B"
    human_role = "B" if model_is_a else "A"

    bot = ModelController(model, label="Model", deterministic=True)
    human = HumanController(label="You")

    print(f"=== Model (Player {model_role}, {model_deck}) vs "
          f"You (Player {human_role}, {human_deck}) ===", flush=True)
    print("(enter an action number or a spec like 'cast:bolt'; 'quit' to exit)\n",
          flush=True)

    def announce(d, action):
        if d.controller is bot:
            menu = d.menu()
            desc = (menu[action]["description"] if 0 <= action < len(menu)
                    else f"action {action}")
            print(f"[Model/{model_role}] {desc}", flush=True)

    runner.run_games(
        bot if model_is_a else human,
        human if model_is_a else bot,
        label_a="Model" if model_is_a else "You",
        label_b="You" if model_is_a else "Model",
        binary_path=binary_path,
        deck_a=model_deck if model_is_a else human_deck,
        deck_b=human_deck if model_is_a else model_deck,
        n_games=1, seed=seed, transcript="narrative", on_action=announce)


def play_gui(binary_path: str, model_path: str, human_player: str = None,
             human_deck: str = "delver", model_deck: str = "delver"):
    """Launch the raylib GUI window; model auto-responds on AI turns, human types in GUI text box.

    Architecture:
    - Binary runs with --machine --gui --player <human_player>
    - Model turns: binary emits BQUERY on stdout; Python reads it, predicts, writes to stdin
    - Human turns: binary displays choices in the GUI window and spins waiting for GUI text-box
      input; Python's reader thread collects narrative lines but sees no BQUERY, so the main
      thread simply waits for the next BQUERY to arrive after the human has acted
    - A background reader thread drains stdout continuously so Python never blocks on readline()
      while the binary is waiting for GUI input
    """
    import queue
    import threading

    model = MaskablePPO.load(model_path)

    if human_player is None:
        human_player = "A" if np.random.random() < 0.5 else "B"
    model_player = "B" if human_player == "A" else "A"
    deck_a = human_deck if human_player == "A" else model_deck
    deck_b = model_deck if human_player == "A" else human_deck
    print(f"=== You (Player {human_player}, {human_deck}) vs Model (Player {model_player}, {model_deck}) ===", flush=True)
    print("(Type your choice number into the GUI text box and press Enter)", flush=True)

    cmd = [binary_path, "--machine", "--gui", "--player", human_player,
           "--deck-a", deck_a, "--deck-b", deck_b]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=BIN_DIR,
    )

    # Items put in queue are either:
    #   str  — a decoded narrative line
    #   dict — a parsed BQUERY with keys: num_choices, state_arr, cat_arr, card_id_arr,
    #           ctrl_arr, pending_confirm
    #   None — EOF sentinel
    line_queue: queue.Queue = queue.Queue()

    _MANDATORY = {2, 3, 4, 5}
    _PAYLOAD = (_BQUERY_STATE_BYTES + _BQUERY_CATS_BYTES + _BQUERY_IDS_BYTES
                + _BQUERY_CTRL_BYTES + _BQUERY_PUB_BYTES)

    def _read_exactly(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("game process ended mid-payload")
            buf.extend(chunk)
        return bytes(buf)

    def _reader():
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.rstrip(b"\n")
                if line.startswith(b"BQUERY: "):
                    n = min(int(line[8:]), MAX_ACTIONS)
                    payload = _read_exactly(_PAYLOAD)
                    offset = 0
                    state_arr = np.frombuffer(payload[offset:offset + _BQUERY_STATE_BYTES],
                                              dtype=np.float32).copy()
                    offset += _BQUERY_STATE_BYTES
                    cats_int = np.frombuffer(payload[offset:offset + _BQUERY_CATS_BYTES],
                                             dtype=np.int32).copy()
                    offset += _BQUERY_CATS_BYTES
                    id_arr = np.frombuffer(payload[offset:offset + _BQUERY_IDS_BYTES],
                                           dtype=np.float32).copy()
                    offset += _BQUERY_IDS_BYTES
                    ctrl_arr = np.frombuffer(payload[offset:offset + _BQUERY_CTRL_BYTES],
                                             dtype=np.float32).copy()
                    offset += _BQUERY_CTRL_BYTES
                    pub_arr = np.frombuffer(payload[offset:offset + _BQUERY_PUB_BYTES],
                                            dtype=np.float32).copy()
                    cat_arr = (cats_int / ACTION_CATEGORY_MAX).astype(np.float32)
                    pending_confirm = any(cats_int[i] in _MANDATORY for i in range(n))
                    line_queue.put({"num_choices": n, "state_arr": state_arr,
                                    "cat_arr": cat_arr, "card_id_arr": id_arr,
                                    "ctrl_arr": ctrl_arr, "pub_arr": pub_arr,
                                    "pending_confirm": pending_confirm})
                else:
                    line_queue.put(line.decode("ascii", errors="replace"))
        except Exception:
            pass
        line_queue.put(None)

    threading.Thread(target=_reader, daemon=True).start()

    reward = 0.0
    try:
        while True:
            item = line_queue.get()
            if item is None:
                break

            if isinstance(item, str):
                if "Player A wins" in item:
                    reward = 1.0
                elif "Player B wins" in item:
                    reward = -1.0
                continue

            # BQUERY dict — model's turn
            num_choices  = item["num_choices"]
            state_arr    = item["state_arr"]
            cat_arr      = item["cat_arr"]
            card_id_arr  = item["card_id_arr"]
            ctrl_arr     = item["ctrl_arr"]
            pending_confirm = item["pending_confirm"]

            hand_ids = np.rint(
                state_arr[_HAND_START : _HAND_START + MAX_HAND_SLOTS * _HAND_SLOT_SIZE]
                * N_CARD_TYPES).astype(np.intp)
            hand_costs = _gather_costs(_CARD_COST_MATRIX, hand_ids)
            bf_ids = np.rint(state_arr[_BF_ID_IDX] * N_CARD_TYPES).astype(np.intp)
            bf_ability_costs = _gather_costs(_CARD_ABILITY_COST_MATRIX, bf_ids)

            obs = np.concatenate([
                state_arr, cat_arr, card_id_arr, ctrl_arr,
                hand_costs.flatten(), bf_ability_costs.flatten(),
            ])

            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            action, _ = model.predict(obs, action_masks=mask if USE_MASKABLE else None, deterministic=True)
            action = int(action)

            cat = int(round(float(cat_arr[action]) * ACTION_CATEGORY_MAX)) if action < num_choices else -1
            cid = float(card_id_arr[action]) if action < MAX_ACTIONS else float(_ACTION_CARD_ID_NULL)
            print(f"[Model/{model_player}] {_action_label(cat, cid)}", flush=True)

            game_action = -1 if (pending_confirm and action == num_choices - 1) else action
            proc.stdin.write(f"{game_action}\n".encode())
            proc.stdin.flush()
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()

    print()
    model_wins = (reward > 0 and model_player == "A") or (reward < 0 and model_player == "B")
    human_wins = (reward > 0 and human_player == "A") or (reward < 0 and human_player == "B")
    if model_wins:
        print(f"=== Model ({model_player}) wins! ===")
    elif human_wins:
        print("=== You win! ===")
    else:
        print("=== Draw ===")


if __name__ == "__main__":
    import os as _os

    # Flags come from cli_spec.PLAY_TOOL (single source shared with the TUI).
    from cli_spec import PLAY_TOOL, apply_to_parser
    parser = argparse.ArgumentParser()
    apply_to_parser(parser, PLAY_TOOL.subs[0])
    args = parser.parse_args()

    if args.scripted and not args.tui:
        parser.error("--scripted is only supported with --tui")

    model_path = args.model
    if args.scripted:
        # Scripted opponent: no checkpoint required (sentinel passed to tui_game.run).
        model_path = "scripted"
    elif model_path is None:
        # Checkpoints are per-deck (deck-pilot naming): one model pilots one deck
        # against any opponent, so the model the human faces is keyed on
        # --model-deck only, not the matchup. The shared resolver prefers
        # '{model_deck}__final.zip', then the newest '{model_deck}__v{steps}.zip'.
        from opponents import resolve_checkpoint
        model_path = resolve_checkpoint(args.model_deck)
        if not _os.path.exists(model_path):
            parser.error(f"No checkpoint found for deck '{args.model_deck}' "
                         f"(looked for {args.model_deck}__final.zip and "
                         f"{args.model_deck}__v*.zip under train/checkpoints/). "
                         f"Train a model piloting {args.model_deck} first, "
                         f"or use --model to specify a path, or --scripted for a rule-based opponent (TUI).")

    if args.tui:
        import tui_game
        tui_game.run(args.binary, model_path, human_player=args.player,
                     human_deck=args.human_deck, model_deck=args.model_deck,
                     bo3=not args.bo1)
    elif args.gui:
        play_gui(args.binary, model_path, human_player=args.player,
                 human_deck=args.human_deck, model_deck=args.model_deck)
    else:
        play(args.binary, model_path, human_deck=args.human_deck, model_deck=args.model_deck,
             human_player=args.player, seed=args.seed)
