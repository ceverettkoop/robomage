"""
Play interactively against a trained RoboMage model.

The seats are --player-a / --player-b: exactly one is the spec 'human' (you),
the other any opponents.make_controller spec (default: az:gen), with --deck-a /
--deck-b their decks. By default you are player A. Player A is on the play
in game 1; --on-the-play b swaps the two sides (agent and deck) so player B's
side starts on the engine's first seat, and --on-the-play random flips a coin
(seeded by --seed when given).

--board picks the front end: gui (the PySide6 app, the default — falls back to
the TUI when PySide6 is missing), tui (the Textual board), or text (a plain
transcript on the shared runner loop: enter an action number or a semantic
spec — 'cast:bolt', 'pass', or 'concede' / 'concede:match' to resign). With
--board gui and no seat/deck flags the GUI app opens on its welcome pane
(File ▸ New Session), which is what ./gui.sh does. Every flag works on every
board it can, and errors on a board that cannot honour it.

Usage:
    train/.venv/bin/python train/play.py --deck-a delver --deck-b burn
    train/.venv/bin/python train/play.py --board tui --player-a az:gen \\
        --player-b human --deck-a burn --deck-b delver
    train/.venv/bin/python train/play.py --board text --player-b scripted
"""

import argparse
import sys

from cli_spec import (BOARD_GUI, BOARD_TEXT, BOARD_TUI, DEFAULT_PLAY_OPPONENT,
                      PLAY_ANALYSIS_DESTS, PLAY_SESSION_DESTS, PLAY_TOOL,
                      SEARCH_KNOB_KEYS, apply_search_knobs, apply_to_parser,
                      explicit_dests, is_bo3, is_search_spec,
                      on_the_play_note, resolve_board, resolve_on_the_play,
                      resolve_play_seats, seat_play_sides, spec_query_keys)

# Flags that only reach a GameDriver board (gui/tui), not the text board.
_DRIVER_ONLY_DESTS = ("human_clock", "hard_timeout", "record_shards")


def build_parser():
    """play.py's parser: the flags come from cli_spec.PLAY_TOOL (the single
    source shared with the TUI form and the GUI launcher)."""
    parser = argparse.ArgumentParser(
        description="Play interactively against a trained model")
    apply_to_parser(parser, PLAY_TOOL.subs[0])
    return parser


def check_board_options(parser, args, explicit, board):
    """Error on a flag the chosen board cannot honour."""
    if board == BOARD_TEXT:
        bad = [d for d in _DRIVER_ONLY_DESTS if d in explicit]
        if bad:
            parser.error(f"{_flags(bad)} need the gui or tui board "
                         "(--board text drives the game through the runner "
                         "loop, which has no clocks or recorder)")
    if board != BOARD_GUI:
        bad = [d for d in ("analysis", *PLAY_ANALYSIS_DESTS) if d in explicit]
        if bad and not (bad == ["analysis"] and args.analysis is False):
            parser.error(f"{_flags(bad)} need the gui board (the analysis "
                         "window is a GUI window)")


def search_values(parser, args, explicit, spec):
    """The search-knob flag values to fold into the opponent ``spec``.

    Explicit flags always apply (appended last, they win over the spec's own
    query). A flag left at its default does not override a knob the spec
    already carries (so ``--player-b az:gen?worlds=2`` keeps its 2 worlds),
    and a spec or flag that sets sims drops the default match clock (the two
    are mutually exclusive). Errors on an explicit knob for a non-search
    opponent."""
    knob_dests = [d for d, _key in SEARCH_KNOB_KEYS]
    set_knobs = [d for d in knob_dests if d in explicit]
    if not is_search_spec(spec):
        if set_knobs:
            parser.error(f"{_flags(set_knobs)} only apply to a search opponent "
                         "(an az:<ckpt> or mcts:<ckpt> player spec)")
        return {}
    if "sims" in explicit and "match_clock" in explicit and args.match_clock:
        parser.error("--sims and --match-clock are mutually exclusive (a "
                     "clocked search is paced by the clock alone)")
    in_spec = spec_query_keys(spec)
    values = {}
    for dest, key in SEARCH_KNOB_KEYS:
        if dest in explicit or key not in in_spec:
            values[dest] = getattr(args, dest)
    if "match_clock" not in explicit and ("sims" in explicit
                                          or "sims" in in_spec):
        values["match_clock"] = None
    if values.get("paced") is None and in_spec & {"time", "clock"}:
        values["paced"] = True
    return values


def _ensure_rocm_env():
    """Any board may run a cuda forward in this process (a search opponent
    or the analysis evaluator on --search-device/--analysis-device cuda, or
    sb3 defaulting to the GPU), so the ROCm RDNA2 env defaults must be in
    place before the first cuda touch — same contract as train.py. Without
    torch there is no cuda to prepare."""
    try:
        from az_net import ensure_rocm_env
    except ImportError:
        return
    ensure_rocm_env()


def _flags(dests):
    return "/".join("--" + d.replace("_", "-") for d in dests)


def play_text(binary_path, opponent_spec, *, human_player="A",
              human_deck="delver", model_deck="delver", seed=None, bo3=True):
    """The text board: one match on the shared runner loop.

    The human seat is an :class:`opponents.HumanController` — it renders the
    board and legal menu each decision and accepts an action number, a
    semantic spec (``cast:bolt``, ``target:bears@opp``, ``pass`` — the same
    grammar as the harness ``--play``), or 'quit'. ``concede`` /
    ``concede:match`` resign (CR 104.3a): the resolver hands back the negative
    sentinel and the runner steps it straight into the engine. The opponent is
    any ``opponents.make_controller`` spec (a model, az:/mcts: search, a
    scripted tier); its choices are announced as they happen, game narrative
    comes from the engine."""
    import runner
    from opponents import HumanController, make_controller

    bot = make_controller(opponent_spec, deterministic=True)
    human = HumanController(label="You")
    model_is_a = human_player == "B"
    model_role = "A" if model_is_a else "B"
    human_role = "B" if model_is_a else "A"

    print(f"=== {opponent_spec} (Player {model_role}, {model_deck}) vs "
          f"You (Player {human_role}, {human_deck}) ===", flush=True)
    print("(enter an action number or a spec like 'cast:bolt'; 'concede' or "
          "'concede:match' to resign; 'quit' to exit)\n", flush=True)

    def announce(d, action):
        if d.controller is bot:
            menu = d.menu()
            desc = (menu[action]["description"] if 0 <= action < len(menu)
                    else f"action {action}")
            print(f"[Opponent/{model_role}] {desc}", flush=True)

    runner.run_games(
        bot if model_is_a else human,
        human if model_is_a else bot,
        label_a="Opponent" if model_is_a else "You",
        label_b="You" if model_is_a else "Opponent",
        binary_path=binary_path,
        deck_a=model_deck if model_is_a else human_deck,
        deck_b=human_deck if model_is_a else model_deck,
        n_games=1, bo3=bo3, seed=seed, transcript="narrative",
        on_action=announce)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    explicit = explicit_dests(parser, argv)

    board = resolve_board(args.board)
    if board == BOARD_GUI and not explicit & PLAY_SESSION_DESTS:
        # No seats or decks: the GUI app on its welcome pane (File ▸ New
        # Session opens the launcher dialogs, which carry their own settings).
        stray = sorted(explicit - {"board", "binary"})
        if stray:
            parser.error(f"{_flags(stray)} need a session — name the seats or "
                         "decks (--player-a/--player-b, --deck-a/--deck-b); "
                         "with none, --board gui opens the GUI app's welcome "
                         "pane, whose New Session dialogs carry these settings")
        _ensure_rocm_env()
        import gui_main
        return gui_main.run_launcher(args.binary)

    # Seats: exactly one player spec is 'human'; the other is the opponent.
    # Each seat pilots its own --deck-a/-b.
    try:
        human_player, opponent = resolve_play_seats(args.player_a,
                                                    args.player_b)
    except ValueError as exc:
        parser.error(str(exc))
    opponent = opponent or DEFAULT_PLAY_OPPONENT
    # --on-the-play: the engine always starts its seat A, so a player-B start
    # swaps the two (agent, deck) sides onto the other seats.
    first = resolve_on_the_play(args.on_the_play, args.seed)
    human_player, human_deck, model_deck = seat_play_sides(
        human_player, opponent, args.deck_a, args.deck_b, first)
    print(on_the_play_note(args.on_the_play, first, human_player, opponent,
                           human_deck, model_deck), flush=True)
    check_board_options(parser, args, explicit, board)
    spec = apply_search_knobs(opponent,
                              search_values(parser, args, explicit, opponent))
    from game_driver import resolve_opponent_spec
    try:
        spec = resolve_opponent_spec(spec)
    except ValueError as exc:
        parser.error(str(exc))

    _ensure_rocm_env()
    if board == BOARD_TEXT:
        return play_text(args.binary, spec, human_player=human_player,
                         human_deck=human_deck, model_deck=model_deck,
                         seed=args.seed, bo3=is_bo3(args))
    session = dict(human_player=human_player, human_deck=human_deck,
                   model_deck=model_deck, bo3=is_bo3(args),
                   human_clock_s=args.human_clock,
                   hard_timeout=args.hard_timeout, engine_seed=args.seed,
                   record_shards=args.record_shards)
    if board == BOARD_TUI:
        import tui_game
        return tui_game.run(args.binary, spec, **session)
    import gui_main
    from gui_game import analysis_opts
    return gui_main.run(args.binary, spec, analysis=analysis_opts(vars(args)),
                        **session)


if __name__ == "__main__":
    sys.exit(main())
