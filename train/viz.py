"""Headless-friendly visualization helpers for analysis.py.

Charts save to PNG by default (matplotlib Agg backend, which never needs a
display) so the tool works in headless cloud containers; pass ``show=True`` to
also open an interactive window where a display exists. Terminal-native
sparklines and bars cover the common views (value curves, card-importance
rankings) with no GUI at all.
"""

import os

_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_out")
_SPARK = "▁▂▃▄▅▆▇█"


def out_dir(args=None):
    """Resolve the chart/report output directory (--out, else analysis_out/)."""
    d = getattr(args, "out", None) if args is not None else None
    d = d or _DEFAULT_OUT
    os.makedirs(d, exist_ok=True)
    return d


def want_show(args=None):
    return bool(getattr(args, "show", False)) if args is not None else False


def pyplot(show=False):
    """Import matplotlib.pyplot with a backend safe for the environment.

    Uses Agg (file-only, never touches a display) unless ``show`` is requested,
    in which case an interactive backend is tried. Returns the pyplot module, or
    None if matplotlib is unavailable (caller should degrade to terminal output).
    """
    try:
        import matplotlib
    except ImportError:
        return None
    if show:
        for backend in ("TkAgg", "QtAgg", "MacOSX"):
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def save_or_show(plt, fig, name, args=None):
    """Save ``fig`` as a PNG under the output dir; optionally also display it.

    Returns the saved path (or None on failure) and prints it so the user — or
    Claude, reading the transcript — can open the artifact.
    """
    path = os.path.join(out_dir(args), name if name.endswith(".png") else name + ".png")
    saved = None
    try:
        fig.savefig(path, dpi=120, bbox_inches="tight")
        print(f"  [chart] saved {path}")
        saved = path
    except Exception as e:
        print(f"  [chart] could not save {name}: {e}")
    if want_show(args):
        try:
            plt.show()
        except Exception as e:
            print(f"  [chart] could not display: {e}")
    try:
        plt.close(fig)
    except Exception:
        pass
    return saved


# ── Terminal-native rendering ────────────────────────────────────────────────

def sparkline(values):
    """Render a sequence of numbers as a unicode sparkline string."""
    vals = [float(v) for v in values]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-9:
        return _SPARK[len(_SPARK) // 2] * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / span * (len(_SPARK) - 1) + 0.5)
        out.append(_SPARK[max(0, min(len(_SPARK) - 1, idx))])
    return "".join(out)


def hbar(value, maxval, width=30):
    """Render a non-negative value as a left-aligned ASCII bar of ``width``."""
    if maxval <= 0:
        return ""
    n = int(round(abs(value) / maxval * width))
    return "█" * max(0, min(width, n))


def diverging_bar(value, maxabs, half=18):
    """Render a signed value as a diverging bar around a center column.

    Positive grows right (█, "helps"); negative grows left (▒, "hurts"). Returns
    a fixed-width string of length ``2*half+1`` so rows line up in a column.
    """
    if maxabs <= 0:
        return " " * half + "│" + " " * half
    n = int(round(min(abs(value), maxabs) / maxabs * half))
    n = max(0, min(half, n))
    if value >= 0:
        return " " * half + "│" + "█" * n + " " * (half - n)
    return " " * (half - n) + "▒" * n + "│" + " " * half
