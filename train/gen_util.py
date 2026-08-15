"""Shared helper for the codegen scripts (gen_enums / gen_card_costs /
gen_card_props / gen_archetypes).

Stdlib-only, like the generators themselves."""
import os


def write_if_changed(path, content):
    """Write `content` to `path` only when it differs from the file already there.

    Returns True if the file was created or its content changed, False if it was
    already up to date. Leaving an unchanged file untouched preserves its mtime,
    which matters because `make` regenerates the codegen on EVERY build: the C++
    mirror headers (src/gen/*.h) are #included by the engine, so rewriting them
    with identical bytes would bump their mtime and trigger a needless recompile
    cascade. write-if-changed keeps a no-op regeneration a true no-op."""
    try:
        with open(path, "r") as f:
            if f.read() == content:
                return False
    except FileNotFoundError:
        pass
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return True
