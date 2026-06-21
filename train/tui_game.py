"""Interactive RoboMage game board as a TUI — PLACEHOLDER.

This module will eventually reimplement the human-vs-model game interface
(currently `play.py`'s CLI loop and raylib GUI) as a full-screen Textual TUI:
a rendered battlefield, hand, stack, and prompts, driven over the engine's
`--machine` stdio protocol. It is invoked via `play.py --tui` and, in turn,
from the `tui.py` launcher's Play entry.

Nothing is implemented yet — `run()` is a stub so the wiring (`play.py --tui`
and the launcher) can be exercised end to end.
"""

import sys


def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver"):
    """Entry point called by `play.py --tui`. Not yet implemented."""
    print("tui_game: the TUI game board is not yet implemented.", file=sys.stderr)
    print(f"  would launch: binary={binary_path} model={model_path} "
          f"human_player={human_player} human_deck={human_deck} model_deck={model_deck}",
          file=sys.stderr)
    return 1
