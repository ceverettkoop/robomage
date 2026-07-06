"""
Play interactively against a trained RoboMage model.

The model is randomly assigned to Player A or B each game (pin with --player).
Text mode runs on the shared runner loop with a HumanController seat — enter
an action number or a semantic spec ('cast:bolt', 'pass'); --seed reproduces a
game. The --tui path delegates to tui_game.py.

Usage:
    train/.venv/bin/python train/play.py --human-deck delver --model-deck burn
"""

import argparse
import numpy as np

from env import BINARY
from decode import card_from_id as _card_from_id

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO

# Mana-tap action categories → produced color (used by the model announcer's action labels).
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
    else:
        play(args.binary, model_path, human_deck=args.human_deck, model_deck=args.model_deck,
             human_player=args.player, seed=args.seed)
