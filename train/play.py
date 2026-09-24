"""
Play interactively against a trained RoboMage model.

The seats are --player-a / --player-b: exactly one is the spec 'human' (you),
the other any opponents.make_controller spec (default: the generalist), with
--deck-a / --deck-b their decks. By default you are player A.
Text mode runs on the shared runner loop with a HumanController seat — enter
an action number or a semantic spec ('cast:bolt', 'pass', or 'concede' /
'concede:match' to resign); --seed reproduces a game. The --tui path delegates
to tui_game.py.

Usage:
    train/.venv/bin/python train/play.py --deck-a delver --deck-b burn
    train/.venv/bin/python train/play.py --player-a az:gen --player-b human \\
        --deck-a burn --deck-b delver
"""

import argparse

from env import BINARY

# ── Main play loop (text mode) ────────────────────────────────────────────────

def play(binary_path: str, model_path: str, human_deck: str = "delver",
         model_deck: str = "delver", human_player: str = "A", seed: int = None,
         bo3: bool = True):
    """Text-mode game against a trained model, on the shared runner loop.

    The human seat is an :class:`opponents.HumanController` — it renders the
    board and legal menu each decision and accepts an action number, a
    semantic spec (``cast:bolt``, ``target:bears@opp``, ``pass`` — the same
    grammar as the harness ``--play``), or 'quit'. ``concede`` /
    ``concede:match`` resign (CR 104.3a): the resolver hands back the negative
    sentinel and the runner steps it straight into the engine, so no clock or
    board plumbing is involved. The model's choices are announced as they
    happen; game narrative comes from the engine.
    """
    import runner
    from opponents import HumanController, ModelController

    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO
    model = MaskablePPO.load(model_path)

    model_is_a = human_player == "B"
    model_role = "A" if model_is_a else "B"
    human_role = "B" if model_is_a else "A"

    bot = ModelController(model, label="Model", deterministic=True)
    human = HumanController(label="You")

    print(f"=== Model (Player {model_role}, {model_deck}) vs "
          f"You (Player {human_role}, {human_deck}) ===", flush=True)
    print("(enter an action number or a spec like 'cast:bolt'; 'concede' or "
          "'concede:match' to resign; 'quit' to exit)\n", flush=True)

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
        n_games=1, bo3=bo3, seed=seed, transcript="narrative",
        on_action=announce)


if __name__ == "__main__":
    import os as _os

    # Flags come from cli_spec.PLAY_TOOL (single source shared with the TUI).
    from cli_spec import (PLAY_TOOL, append_spec_knob, apply_to_parser, is_bo3,
                          resolve_play_seats)
    parser = argparse.ArgumentParser()
    apply_to_parser(parser, PLAY_TOOL.subs[0])
    args = parser.parse_args()

    # The clocks live on the GameDriver, which only the board front ends run —
    # text mode drives the game through runner.run_games instead.
    if (args.human_clock is not None or args.hard_timeout) and not (args.tui or args.gui):
        parser.error("--human-clock/--hard-timeout need the TUI or GUI board")

    # Seats: exactly one player spec is 'human'; the other is the opponent
    # (None = the default generalist). Each seat pilots its own --deck-a/-b.
    try:
        human_player, model_path = resolve_play_seats(args.player_a, args.player_b)
    except ValueError as exc:
        parser.error(str(exc))
    human_deck, model_deck = ((args.deck_a, args.deck_b) if human_player == "A"
                              else (args.deck_b, args.deck_a))
    opp_flag = "--player-b" if human_player == "A" else "--player-a"

    is_ctrl_spec = bool(model_path) and model_path.lower().startswith(
        ("az:", "azraw:", "mcts:", "scripted"))
    is_search_spec = bool(model_path) and model_path.lower().startswith(
        ("az:", "mcts:"))
    if is_ctrl_spec and not (args.tui or args.gui):
        parser.error("controller specs (az:/azraw:/mcts:/scripted) need the TUI "
                     "or GUI board — text mode loads a PPO .zip directly")
    if args.sims is not None or args.worlds is not None:
        if not is_search_spec:
            parser.error("--sims/--worlds only apply to a search opponent "
                         f"({opp_flag} az:<ckpt> or mcts:<ckpt>)")
        # Append the knobs to the spec's query; appended-last wins over any
        # sims=/worlds= already present (later keys overwrite in the parser).
        if args.sims is not None:
            model_path = append_spec_knob(model_path, "sims", args.sims)
        if args.worlds is not None:
            model_path = append_spec_knob(model_path, "worlds", args.worlds)
    if args.think_time is not None:
        if not is_search_spec:
            parser.error("--think-time only applies to a search opponent "
                         f"({opp_flag} az:<ckpt> or mcts:<ckpt>)")
        # Wall-clock per-decision budget: the search runs as many sims as fit in
        # this many seconds. Appended last so it wins over any time= in the spec.
        model_path = append_spec_knob(model_path, "time", args.think_time)
    if args.search_procs is not None:
        if not is_search_spec:
            parser.error("--search-procs only applies to a search opponent "
                         f"({opp_flag} az:<ckpt> or mcts:<ckpt>)")
        # World-parallel search across N engine processes. Appended last so it
        # wins over any procs= already present in the spec.
        model_path = append_spec_knob(model_path, "procs", args.search_procs)
    elif is_search_spec and "procs=" not in model_path:
        # Interactive play defaults to a PARALLEL search (the spec grammar's own
        # default stays procs=1 for reproducible gates/eval): fan the worlds
        # across half the cores so a human's think time buys more sims. An
        # explicit procs= in the spec (or --search-procs above) always wins.
        from opponents import default_search_procs_for_spec
        model_path = append_spec_knob(
            model_path, "procs", default_search_procs_for_spec(model_path))
    if args.match_clock is not None:
        if not is_search_spec:
            parser.error("--match-clock only applies to a search opponent "
                         f"({opp_flag} az:<ckpt> or mcts:<ckpt>)")
        # Whole-match chess-clock bank; per-decision budgets are allocated from
        # it. Appended last so it wins over any clock= already in the spec.
        model_path = append_spec_knob(model_path, "clock", args.match_clock)
    if args.paced and args.no_paced:
        parser.error("--paced and --no-paced are mutually exclusive")
    if (args.paced or args.no_paced) and not is_search_spec:
        parser.error("--paced/--no-paced only apply to a search opponent "
                     f"({opp_flag} az:<ckpt> or mcts:<ckpt>)")
    if is_search_spec:
        # Paced default: ON whenever the opponent has a variable thinking budget
        # (a match clock or per-decision think time) — that is when response
        # timing would otherwise leak whether there was anything to think about.
        # Appended last so it wins over any paced= already in the spec.
        has_variable_budget = (args.match_clock is not None
                               or args.think_time is not None
                               or "clock=" in model_path or "time=" in model_path)
        if args.no_paced:
            model_path = append_spec_knob(model_path, "paced", 0)
        elif args.paced or has_variable_budget:
            model_path = append_spec_knob(model_path, "paced", 1)
    if model_path is None:
        # There is ONE generalist model ('gen'); it pilots whatever deck its
        # seat names. The default opponent is that single generalist, resolved
        # to 'gen__final.zip' (else the newest 'gen__v{steps}.zip').
        from opponents import resolve_checkpoint, GEN_STEM
        model_path = resolve_checkpoint(GEN_STEM)
        if not _os.path.exists(model_path):
            parser.error(f"No generalist checkpoint found "
                         f"(looked for {GEN_STEM}__final.zip and {GEN_STEM}__v*.zip "
                         f"under train/checkpoints/). Train the generalist first "
                         f"(train --deck-a {model_deck} --deck-b <opp>), "
                         f"or name the opponent with {opp_flag} (a checkpoint "
                         f"path, or {opp_flag} scripted for a rule-based "
                         f"opponent on the TUI/GUI board).")

    if args.gui:
        # --gui takes precedence over --tui (the launcher form pre-checks --tui).
        # PySide6 is an optional extra (train/requirements-gui.txt); if it isn't
        # installed, fall back to the TUI when --tui was also set, else error with
        # the install hint.
        try:
            import gui_main
        except ImportError:
            if args.tui:
                print("PySide6 not installed — falling back to the TUI board "
                      "(pip install -r train/requirements-gui.txt for the GUI).")
                import tui_game
                tui_game.run(args.binary, model_path, human_player=human_player,
                             human_deck=human_deck, model_deck=model_deck,
                             bo3=is_bo3(args), human_clock_s=args.human_clock,
                             hard_timeout=args.hard_timeout)
            else:
                parser.error("PySide6 not installed — pip install -r "
                             "train/requirements-gui.txt, or use --tui")
        else:
            import sys as _sys
            _sys.exit(gui_main.run(
                args.binary, model_path, human_player=human_player,
                human_deck=human_deck, model_deck=model_deck,
                bo3=is_bo3(args), analysis=args.analysis,
                record_shards=args.record_shards,
                human_clock_s=args.human_clock,
                hard_timeout=args.hard_timeout))
    elif args.tui:
        import tui_game
        tui_game.run(args.binary, model_path, human_player=human_player,
                     human_deck=human_deck, model_deck=model_deck,
                     bo3=is_bo3(args), human_clock_s=args.human_clock,
                     hard_timeout=args.hard_timeout)
    else:
        play(args.binary, model_path, human_deck=human_deck, model_deck=model_deck,
             human_player=human_player, seed=args.seed, bo3=is_bo3(args))
