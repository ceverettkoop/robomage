"""Interactive RoboMage game board as a PySide6 (Qt) desktop window.

The graphical sibling of the Textual board (train/tui_game.py): same layout,
same content, rendered with Qt widgets instead of terminal cells. Both front
ends sit on the shared, front-end-agnostic engine loop and presentation helpers
in train/game_driver.py — this module only draws the board and marshals input.

Launch it via::

    train/.venv/bin/python train/play.py --gui --scripted --bo1

(`--gui` takes precedence over `--tui`; any opponent spec the TUI accepts —
scripted tiers, a checkpoint path, az:/mcts: search wrappers — works here too.)

Phase B is a fully playable skeleton with **text-placeholder** cards: the
CardWidget painter draws a rounded card with per-edge WUBRG borders, the card
name, P/T and counter chips, and status flags. Real Scryfall art is Phase C —
CardWidget.set_pixmap() is already wired so a downloaded pixmap can be slotted
in, and the window keeps a name -> [CardWidget] registry for the image pipeline
to swap art into. Cross-highlighting, the oracle-text popup, and the thinking
ticker are Phase D (right-button press/release are reserved no-op stubs here).

Wayland note: PySide6 wheels bundle both the wayland and xcb platform plugins.
If a Wayland session misbehaves (blank window, input glitches), force X11 with
``QT_QPA_PLATFORM=xcb``.

Headless sanity check: set ``ROBOMAGE_GUI_SMOKE=1`` to auto-quit cleanly shortly
after the first StateUpdate renders (exit 0), or ``ROBOMAGE_GUI_SMOKE=N`` (N>1)
to auto-play N human decisions — submitting the first legal action each time —
before quitting. Pair it with ``QT_QPA_PLATFORM=offscreen`` for a display-less run.
"""

import os
import threading

import numpy as np

from PySide6.QtCore import Qt, QObject, Signal, QTimer, QRectF, QSize
from PySide6.QtGui import (QColor, QFont, QPainter, QPen, QBrush, QPixmap,
                           QPainterPath, QKeySequence, QShortcut)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                               QVBoxLayout, QHBoxLayout, QScrollArea, QSplitter,
                               QListWidget, QListWidgetItem, QPlainTextEdit)

from env import _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE
import decode
import scryfall_cache
from game_driver import (GameDriver, build_session, decode_human_frame,
                         actions_for_card, stack_target_refs,
                         menu_label, prompt_text, hand_type_icon, _edge_colors,
                         _STEP_ABBR)

# ── Card geometry ─────────────────────────────────────────────────────────────
# Untapped cards are portrait at the real 63:88 Magic aspect; a tapped card is
# painted rotated 90° into a CARD_H×CARD_W landscape box so a row of them reflows
# just like a tabletop. Rows are a fixed height so 10+ permanents simply scroll.
CARD_W = 80
CARD_H = 112
CARD_RADIUS = 7
ROW_H = CARD_H + 22          # card + margins + horizontal scrollbar allowance
STACK_THUMB_W = 46
STACK_THUMB_H = 64
STACK_ROW_H = STACK_THUMB_H + 20

# Attacking creatures get a dashed border in this red — brighter/more saturated
# than any muted color-identity red so an attacker reads distinctly (mirrors the
# TUI's _ATTACK_BORDER).
_ATTACK_BORDER = "#ff2b2b"
_CARD_FILL = QColor("#1c1c24")
_CARD_TEXT = QColor("#e6e6ea")
_CHIP_BG = QColor("#0c0c10")


# ── QSS theme ─────────────────────────────────────────────────────────────────
# Dark terminal-ish palette matching the TUI: near-black background, light grey
# text, green "you" / red "opponent" accents, 1px panel borders, slightly
# lighter land rows, amber selection, a bold boosted prompt bar.
_QSS = """
QMainWindow, QWidget { background: #101014; color: #d8d8d8; }
QLabel#oppInfo  { color: #e05555; padding: 2px 4px; }
QLabel#selfInfo { color: #33d17a; padding: 2px 4px; }
QLabel#graveyards { color: #9a9aa2; padding: 2px 4px; }
QLabel#prompt {
    font-weight: bold; background: #24242e; color: #f0f0f4; padding: 4px 6px;
    border: 1px solid #2a2a33;
}
QLabel#stepCell { color: #6a6a72; padding: 0 4px; }
QLabel#stepCell[current="true"] {
    background: #d8d8d8; color: #101014; font-weight: bold;
}
QLabel#statusCell { color: #b8b8c0; padding: 0 6px; }
QScrollArea { border: 1px solid #2a2a33; }
QScrollArea#handRow { border: 1px solid #2f7a4a; }
QListWidget {
    border: 1px solid #2a2a33; background: #0c0c10;
    selection-background-color: #d8a12a; selection-color: #101014;
}
QListWidget::item:selected { background: #d8a12a; color: #101014; }
QPlainTextEdit {
    border: 1px solid #2a2a33; background: #0c0c10; color: #cfcfcf;
}
QSplitter::handle { background: #2a2a33; }
"""


# ── Card widget ───────────────────────────────────────────────────────────────

class CardWidget(QWidget):
    """A single text-placeholder card (battlefield permanent or hand card).

    The painter is permanent — it also serves as the loading/offline/token-miss
    fallback once Phase C adds art. Built from the decoded permanent dict (`perm`;
    None for a hand card) plus the card's vocab index, so it can draw the same
    P/T, counters and status flags the TUI's fmt_perm shows.

    Phase C: call set_pixmap(QPixmap) to slot in real Scryfall art. When a pixmap
    is set the painter draws it clipped to the card's rounded rect and drops the
    per-edge WUBRG borders (the card frame carries the color), keeping the name
    caption and all chips/flags/overlays on top. The owning window keeps a
    name -> [CardWidget] registry so the image pipeline can find every widget for
    a given card name and push a pixmap into it."""

    # Left click commits an action for this card; the window resolves it against
    # the live menu (see GameWindow._on_card_clicked).
    clicked = Signal(int, str, str)          # card_idx, controller, zone
    # Hover in/out — emits self on enter, None on leave. Unused in Phase B;
    # Phase D wires it to action<->card cross-highlighting.
    hovered = Signal(object)

    def __init__(self, name, card_idx, controller, zone, *, perm=None, icon="",
                 token_pt=None):
        super().__init__()
        self._name = name
        self._card_idx = card_idx
        self._controller = controller        # "self" | "opp"
        self._zone = zone                    # "battlefield" | "hand"
        self._icon = icon                    # hand-card type icon, else ""
        self._token_pt = token_pt            # (p, t) for a token, else None
        self._pixmap = None                  # Phase C art (full 'normal'), if set
        self._scaled = None                  # cached card-sized scale of _pixmap
        self._edge_colors = _edge_colors(decode.card_border_colors(card_idx))

        # Status / stat fields derived once from the permanent dict so paintEvent
        # stays cheap (a hand card has none of these).
        self._tapped = bool(perm and perm.get("tapped"))
        self._attacking = bool(perm and perm.get("attacking"))
        self._summoning_sick = bool(perm and perm.get("summoning_sick"))
        self._phased_out = bool(perm and perm.get("phased_out"))
        self._chip = _perm_stat_chip(perm) if perm else ""
        self._counters = _perm_counters_text(perm) if perm else ""
        self._flags = _perm_flags(perm) if perm else []

        # A tapped card lays landscape so the row reflows like a tabletop.
        self.setFixedSize(self.sizeHint())
        self.setMouseTracking(True)

    def sizeHint(self):
        return QSize(CARD_H, CARD_W) if self._tapped else QSize(CARD_W, CARD_H)

    def set_pixmap(self, pixmap):
        """Store Phase C card art (QPixmap) and repaint. A null/None pixmap
        reverts to the text placeholder. The full 'normal' art is scaled to the
        card size once (cached in _scaled) so paintEvent never rescales."""
        self._pixmap = pixmap
        self._scaled = None                  # invalidate cached scale
        self.update()

    def _card_pixmap(self):
        """The card-sized (portrait CARD_W x CARD_H) smooth scale of the stored
        art, computed once and cached; None when no usable art is set."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        if self._scaled is None:
            self._scaled = self._pixmap.scaled(
                QSize(int(CARD_W), int(CARD_H)),
                Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        return self._scaled

    # ----- input -----

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._card_idx, self._controller, self._zone)
        elif event.button() == Qt.RightButton:
            self._on_right_press()           # Phase D: hold to show oracle text

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self._on_right_release()

    def _on_right_press(self):
        """Reserved for the Phase D oracle-text popup (right-button hold)."""

    def _on_right_release(self):
        """Reserved for the Phase D oracle-text popup (right-button release)."""

    def enterEvent(self, event):
        self.hovered.emit(self)

    def leaveEvent(self, event):
        self.hovered.emit(None)

    # ----- painting -----

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Dim states apply to the whole card.
        if self._phased_out:
            painter.setOpacity(0.35)
        elif self._summoning_sick:
            painter.setOpacity(0.75)
        # A tapped card is drawn portrait then rotated 90° into its landscape box.
        if self._tapped:
            painter.translate(CARD_H, 0)
            painter.rotate(90)
        self._paint_card(painter, QRectF(0, 0, CARD_W, CARD_H))
        painter.end()

    def _paint_card(self, painter, rect):
        r = rect.adjusted(1.5, 1.5, -1.5, -1.5)
        path = QPainterPath()
        path.addRoundedRect(r, CARD_RADIUS, CARD_RADIUS)

        # Body: Phase C art (clipped to the rounded rect) or the dark placeholder.
        art = self._card_pixmap()
        if art is not None:
            painter.save()
            painter.setClipPath(path)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(r.toRect(), art)
            painter.restore()
        else:
            painter.fillPath(path, QBrush(_CARD_FILL))

        # Borders: dashed red when attacking, else the per-edge color identity
        # (placeholders only — real art carries the frame color).
        if self._attacking:
            pen = QPen(QColor(_ATTACK_BORDER))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        elif art is None:
            self._paint_edges(painter, r)

        # Name (wrapped, top). Over real art a translucent caption strip keeps
        # the name legible (unreadable directly on card art at this size).
        f = QFont(painter.font())
        f.setPointSizeF(7.0)
        painter.setFont(f)
        name_rect = QRectF(r.left() + 3, r.top() + 3, r.width() - 6, r.height() * 0.52)
        if art is not None:
            strip = name_rect.adjusted(-2, -2, 2, 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(12, 12, 16, 190))
            painter.drawRoundedRect(strip, 3, 3)
        painter.setPen(_CARD_TEXT)
        painter.drawText(name_rect, Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignTop,
                         self._name)

        # Hand-card type icon (top-right).
        if self._icon:
            icon_rect = QRectF(r.right() - 16, r.top() + 2, 14, 14)
            painter.drawText(icon_rect, Qt.AlignCenter, self._icon)

        # Status flags (small, below the name).
        if self._flags:
            painter.setPen(QColor("#c8a24a"))
            ff = QFont(painter.font())
            ff.setPointSizeF(6.0)
            painter.setFont(ff)
            flag_rect = QRectF(r.left() + 3, r.top() + r.height() * 0.55,
                               r.width() - 6, 12)
            painter.drawText(flag_rect, Qt.AlignHCenter | Qt.AlignVCenter,
                             " ".join(self._flags))

        # Counters chip (bottom-left) and P/T-or-loyalty chip (bottom-right).
        if self._counters:
            self._paint_chip(painter, self._counters, r, right=False)
        if self._chip:
            self._paint_chip(painter, self._chip, r, right=True)

    def _paint_edges(self, painter, r):
        """Paint the four color-identity border edges (see _edge_colors)."""
        top, right, bottom, left = self._edge_colors
        corners = {"tl": r.topLeft(), "tr": r.topRight(),
                   "br": r.bottomRight(), "bl": r.bottomLeft()}
        segs = [(top, corners["tl"], corners["tr"]),
                (right, corners["tr"], corners["br"]),
                (bottom, corners["br"], corners["bl"]),
                (left, corners["bl"], corners["tl"])]
        for color, a, b in segs:
            if not color:
                continue
            pen = QPen(QColor(color))
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawLine(a, b)

    def _paint_chip(self, painter, text, r, *, right):
        """Draw a small filled chip with `text` in a bottom corner of `r`."""
        f = QFont(painter.font())
        f.setPointSizeF(6.5)
        f.setBold(True)
        painter.setFont(f)
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 6
        h = fm.height() + 2
        x = (r.right() - w - 2) if right else (r.left() + 2)
        y = r.bottom() - h - 2
        chip = QRectF(x, y, w, h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_CHIP_BG))
        painter.drawRoundedRect(chip, 3, 3)
        painter.setPen(_CARD_TEXT)
        painter.drawText(chip, Qt.AlignCenter, text)


def _perm_stat_chip(p):
    """The bottom-right stat chip for a permanent: P/T (+damage) for a creature,
    else a loyalty chip for a planeswalker, else ''."""
    if "power" in p:
        chip = f"{p['power']}/{p['toughness']}"
        if "damage" in p:
            chip += f" ({p['damage']})"
        return chip
    if "loyalty" in p:
        return f"◆{p['loyalty']}"      # ◆ loyalty
    return ""


def _perm_counters_text(p):
    """Compact counter summary for the bottom-left chip (mirrors fmt_perm's
    counter rendering), or '' when the permanent has none."""
    if "counters" in p:
        return p["counters"]
    parts = []
    if "p1p1" in p:
        parts.append(f"{p['p1p1']:+d}")
    if "other_counters" in p:
        parts.append(f"c{p['other_counters']}")
    return ", ".join(parts)


def _perm_flags(p):
    """Status-flag chips for a permanent, matching fmt_perm's flag set."""
    flags = []
    if p.get("tapped"):
        flags.append("T")
    if p.get("attacking"):
        flags.append("ATK")
    if p.get("blocking"):
        flags.append("BLK")
    if p.get("blocked"):
        flags.append("BLOCKED")
    if p.get("summoning_sick"):
        flags.append("SICK")
    if p.get("phased_out"):
        flags.append("PHASED")
    return flags


# ── Stack item widget ─────────────────────────────────────────────────────────

class _StackThumb(QWidget):
    """A tiny placeholder thumbnail for a stack object. Phase C can push art in
    via set_pixmap (stack thumbs reuse the same image cache as board cards)."""

    def __init__(self, card_idx):
        super().__init__()
        self._card_idx = card_idx
        self._pixmap = None
        self._scaled = None
        self._edge = _edge_colors(decode.card_border_colors(card_idx))
        self.setFixedSize(STACK_THUMB_W, STACK_THUMB_H)

    def set_pixmap(self, pixmap):
        self._pixmap = pixmap
        self._scaled = None
        self.update()

    def _thumb_pixmap(self):
        if self._pixmap is None or self._pixmap.isNull():
            return None
        if self._scaled is None:
            self._scaled = self._pixmap.scaled(
                QSize(STACK_THUMB_W, STACK_THUMB_H),
                Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        return self._scaled

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        path = QPainterPath()
        path.addRoundedRect(r, 4, 4)
        art = self._thumb_pixmap()
        if art is not None:
            painter.setClipPath(path)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(r.toRect(), art)
        else:
            painter.fillPath(path, QBrush(_CARD_FILL))
            color = self._edge[0] or "#9a9aa2"
            pen = QPen(QColor(color))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.end()


class StackItemWidget(QWidget):
    """One object on the stack: a placeholder thumb + a description label using
    the same text the TUI shows. Carries its human-frame target refs so Phase D
    can highlight the targets on hover."""

    def __init__(self, entry, target_refs):
        super().__init__()
        self._target_refs = target_refs
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 8, 2)
        layout.setSpacing(6)
        self._thumb = _StackThumb(entry["card_idx"])
        layout.addWidget(self._thumb)
        label = QLabel(_stack_item_label(entry))
        label.setWordWrap(True)
        layout.addWidget(label, 1)


def _stack_item_label(e):
    """Human-readable stack-object label (verbatim from the TUI)."""
    kind = "spell" if e["is_spell"] else "ability"
    label = f"{e['name']} ({kind}, {e['controller']})"
    if e.get("targets"):
        label += " → " + "; ".join(e["targets"])
    return label


# ── Fixed-height, horizontal-only card row ────────────────────────────────────

class CardRow(QScrollArea):
    """A single board row: a horizontal-only scroll area of fixed height. Cards
    are fixed size, so a crowded row just scrolls sideways."""

    def __init__(self, height=ROW_H, *, land=False):
        super().__init__()
        self.setFixedHeight(height)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._layout = QHBoxLayout(self._inner)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.setWidget(self._inner)
        if land:
            # Land rows sit on a slightly lighter background (mirrors the TUI's
            # .bf-row.lands panel tint).
            self.setStyleSheet(
                "QScrollArea, QScrollArea > QWidget, QWidget { background: #16161c; }")

    def set_cards(self, widgets):
        """Replace the row's contents with `widgets` (list of card/thumb/label)."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        for w in widgets:
            self._layout.addWidget(w)


# ── Phase strip ───────────────────────────────────────────────────────────────

class PhaseStrip(QWidget):
    """The status + 13-step phase strip. The current step cell is reverse-
    highlighted via a dynamic QSS `current` property (repolished on change)."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 1, 4, 1)
        layout.setSpacing(0)
        self._status = QLabel()
        self._status.setObjectName("statusCell")
        layout.addWidget(self._status)
        self._cells = []
        for abbr in _STEP_ABBR:
            cell = QLabel(abbr)
            cell.setObjectName("stepCell")
            cell.setProperty("current", False)
            layout.addWidget(cell)
            self._cells.append(cell)
        layout.addStretch(1)

    def update_strip(self, cur_step, status_text):
        self._status.setText(status_text)
        for i, cell in enumerate(self._cells):
            want = (i == cur_step)
            if cell.property("current") != want:
                cell.setProperty("current", want)
                cell.style().unpolish(cell)
                cell.style().polish(cell)


# ── Worker -> UI bridge ───────────────────────────────────────────────────────

class DriverBridge(QObject):
    """Adapts GameDriver's sink callbacks (fired on the worker thread) to Qt
    signals. Qt queues cross-thread signal delivery onto the UI thread — the
    direct analog of Textual's post_message. The window connects each signal to
    its render slot."""

    state_update = Signal(object)
    log_lines = Signal(object)
    opp_thinking = Signal(bool)
    game_over = Signal(str)

    # DriverSink interface (called on the worker thread).
    def on_state(self, u):
        self.state_update.emit(u)

    def on_log(self, lines):
        self.log_lines.emit(list(lines))

    def on_opp_thinking(self, active):
        self.opp_thinking.emit(bool(active))

    def on_game_over(self, text):
        self.game_over.emit(text)


# ── Scryfall image provider ───────────────────────────────────────────────────

class ImageProvider(QObject):
    """Bridges the Qt-free scryfall_cache pipeline to the board widgets.

    `request(name, token_pt)` returns a ready QPixmap (from an in-memory cache or
    a synchronous disk hit) or None; on a miss it kicks off an async download and
    emits `image_ready(name)` on the UI thread once the file lands, so the window
    can re-resolve every widget showing that name and slot the art in. Only names
    that actually downloaded emit (a definitive miss stays a placeholder)."""

    image_ready = Signal(str)                # card name that just became available

    def __init__(self):
        super().__init__()
        self._pixmaps = {}                   # cache-key -> QPixmap (full 'normal')

    def request(self, name, token_pt=None):
        key = scryfall_cache._cache_key(name, token_pt)
        pm = self._pixmaps.get(key)
        if pm is not None:
            return None if pm.isNull() else pm
        path = scryfall_cache.get_cached(name, token_pt)
        if path:
            pm = QPixmap(path)
            self._pixmaps[key] = pm
            return None if pm.isNull() else pm
        # Not on disk: queue an async download; _on_fetched re-emits to the UI.
        scryfall_cache.fetch_async(name, self._on_fetched, token_pt)
        return None

    def _on_fetched(self, name, path):
        # Runs on the download worker thread. A queued signal marshals to the UI
        # thread, where request() will now find the freshly written file.
        if path:
            self.image_ready.emit(name)


# ── The window ────────────────────────────────────────────────────────────────

class GameWindow(QMainWindow):
    """The Qt game board. Owns the widget tree, the DriverBridge, and the
    GameDriver; renders each StateUpdate and delivers the human's picks back."""

    def __init__(self, session):
        super().__init__()
        self._opp_is_a = session.opp_is_a
        self._human_deck = session.human_deck
        self._opp_deck = session.opp_deck
        self._bo3 = session.bo3
        self._opp_label = session.opp_label

        # name -> [CardWidget] registry, rebuilt every StateUpdate; the image
        # pipeline uses it to find every widget showing a given card name.
        # _stack_thumbs is the parallel name -> [_StackThumb] registry (stack
        # thumbs aren't CardWidgets but take the same art).
        self._registry = {}
        self._stack_thumbs = {}
        self._provider = ImageProvider()
        self._actions = []
        self._awaiting = False
        self._thread = None

        # Smoke-test auto-drive (headless CI-less sanity check). See module docs.
        self._smoke_n = _smoke_n_from_env()
        self._smoke_seen = 0
        self._smoke_played = 0

        self._build_ui()

        seat = "B" if self._opp_is_a else "A"
        opp_seat = "A" if self._opp_is_a else "B"
        fmt = "Best of 3" if self._bo3 else "Single game"
        self.setWindowTitle(
            f"RoboMage · {fmt}  —  You (Player {seat}, {self._human_deck}) "
            f"vs {self._opp_label} (Player {opp_seat}, {self._opp_deck})")

        # Bridge + driver. The bridge lives on the UI thread, so its signal
        # emissions from the worker thread are queued to this thread's slots.
        self._bridge = DriverBridge()
        self._bridge.state_update.connect(self.on_state_update)
        self._bridge.log_lines.connect(self.on_log_lines)
        self._bridge.opp_thinking.connect(self.on_opp_thinking)
        self._bridge.game_over.connect(self.on_game_over)
        self._provider.image_ready.connect(self._on_image_ready)
        self._driver = GameDriver(
            env=session.env, opp_act=session.opp_act, opp_is_a=session.opp_is_a,
            is_model=session.is_model, opp_label=session.opp_label,
            bo3=session.bo3, sink=self._bridge,
            clock_fn=session.clock_fn, pace_idle=session.pace_idle)

        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)

        self._append_log(
            "Game starting...  Click a card or pick a numbered action. "
            "Keys: digits = pick, space = pass, p = autopass, ctrl+q = quit.")

    # ----- layout -----

    def _build_ui(self):
        # Board block: the same vertical order as the TUI's compose().
        board = QWidget()
        v = QVBoxLayout(board)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(3)

        self._phase = PhaseStrip()
        v.addWidget(self._phase)

        self._opp_info = QLabel()
        self._opp_info.setObjectName("oppInfo")
        v.addWidget(self._opp_info)

        # Opponent's rows are flipped (lands on top, permanents below) so the two
        # players' non-land rows sit adjacent across the stack.
        self._opp_lands = CardRow(land=True)
        self._opp_perms = CardRow()
        v.addWidget(self._opp_lands)
        v.addWidget(self._opp_perms)

        self._stack_row = CardRow(height=STACK_ROW_H)
        v.addWidget(self._stack_row)

        # Self: permanents on top, lands below — mirror of the opponent so the
        # non-land rows are the ones adjacent to the stack.
        self._self_perms = CardRow()
        self._self_lands = CardRow(land=True)
        v.addWidget(self._self_perms)
        v.addWidget(self._self_lands)

        self._hand_row = CardRow()
        self._hand_row.setObjectName("handRow")
        v.addWidget(self._hand_row)

        self._self_info = QLabel()
        self._self_info.setObjectName("selfInfo")
        v.addWidget(self._self_info)

        self._graveyards = QLabel()
        self._graveyards.setObjectName("graveyards")
        v.addWidget(self._graveyards)

        self._prompt = QLabel()
        self._prompt.setObjectName("prompt")
        self._prompt.setWordWrap(True)
        v.addWidget(self._prompt)

        # Bottom: action menu (~35%) + log.
        self._menu = QListWidget()
        self._menu.itemActivated.connect(self._on_item_activated)
        self._menu.installEventFilter(self)      # route digit/space/p keys
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self._menu)
        bottom.addWidget(self._log)
        bottom.setStretchFactor(0, 35)
        bottom.setStretchFactor(1, 65)
        bottom.setSizes([340, 640])

        # Vertical splitter: board block over the menu/log area.
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(board)
        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        self.setCentralWidget(splitter)
        self.resize(1120, 940)

    # ----- driver lifecycle -----

    def start(self):
        """Start the engine loop on a worker thread (called after show)."""
        self._thread = threading.Thread(target=self._driver.run,
                                        name="gui-driver", daemon=True)
        self._thread.start()

    def closeEvent(self, event):
        # Signal shutdown and join the worker before the caller closes the env.
        # request_quit sets _quitting so a loop blocked in a long opponent search
        # returns silently once the env (and engine pipe) tear down.
        self._driver.request_quit()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        super().closeEvent(event)

    # ----- render (UI thread) -----

    def on_state_update(self, u):
        obs = u.obs
        # decode_human_frame mirrors an opponent-perspective obs back to the
        # human's view so the board never flips while the opponent acts/decides.
        gs, mirrored = decode_human_frame(u)

        cur = int(np.argmax(
            obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))
        self._phase.update_strip(cur, self._phase_status(gs))
        self._opp_info.setText(
            self._info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self._self_info.setText(
            self._info_line("YOU", gs["self"], gs["self_library"]))
        self._graveyards.setText(
            f"Your GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:  {', '.join(gs['opp_graveyard']) or '—'}")

        # Rebuild the board; registries are refreshed from scratch each frame.
        self._registry = {}
        self._stack_thumbs = {}
        self._rebuild_stack(gs["stack"], mirrored)
        self._rebuild_bf(self._opp_perms, self._opp_lands, gs["opp_battlefield"], "opp")
        self._rebuild_bf(self._self_perms, self._self_lands, gs["self_battlefield"], "self")
        # A mirrored obs's "self_hand" is the OPPONENT's private hand — never
        # render it; the human's hand keeps its last own-perspective contents.
        if not mirrored:
            self._rebuild_hand(gs["self_hand"])

        # Action menu.
        self._actions = u.actions
        self._awaiting = u.human_turn
        self._menu.clear()
        self._menu.setEnabled(True)
        if u.human_turn:
            for a in u.actions:
                it = QListWidgetItem(f"{a['index']:>2}: {menu_label(a, self._opp_is_a)}")
                it.setData(Qt.UserRole, a["index"])
                self._menu.addItem(it)
            if u.actions:
                self._menu.setCurrentRow(0)
            self._prompt.setText(prompt_text(obs, u.num_choices, gs))
        elif self._driver._autopass:
            self._prompt.setText("Autopass — passing to next upkeep...")
        else:
            self._prompt.setText(f"{self._opp_label} is thinking...")

        self._smoke_seen += 1
        self._maybe_smoke(u.human_turn)

    def on_log_lines(self, lines):
        for line in lines:
            if line.strip():
                self._append_log(line)

    def on_opp_thinking(self, active):
        # Phase B: a plain label toggle (the elapsed-time ticker + clock bank is
        # Phase D). The menu prompt already shows "<opp> is thinking..." from the
        # state update; nothing extra to do here yet.
        pass

    def on_game_over(self, text):
        self._awaiting = False
        self._menu.clear()
        self._menu.setEnabled(False)
        self._prompt.setText(text + "  (Ctrl+Q to quit)")
        self._append_log(text)
        if self._smoke_n is not None:
            QTimer.singleShot(80, self.close)

    # ----- input -----

    def _on_item_activated(self, item):
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self._submit(int(idx))

    def _on_card_clicked(self, card_idx, controller, zone):
        if not self._awaiting:
            return
        matches = actions_for_card(self._actions, card_idx, controller, zone)
        if not matches:
            self._append_log(
                "No legal action for that card - use the numbered list.")
        elif len(matches) == 1:
            self._submit(matches[0]["index"])
        else:
            # Ambiguous -> don't guess: select the first option this card feeds
            # and list them (full multi-row highlight is Phase D).
            idxs = [a["index"] for a in matches]
            row = self._row_for_index(idxs[0])
            if row is not None:
                self._menu.setCurrentRow(row)
            self._append_log(
                f"That card has {len(idxs)} options: "
                f"{', '.join(str(i) for i in idxs)} - pick a number.")

    def eventFilter(self, obj, event):
        # Intercept digit/space/p in the menu so they drive the board instead of
        # the list's built-in typeahead; Enter still fires itemActivated.
        if obj is self._menu and event.type() == event.Type.KeyPress:
            if self._handle_key(event):
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if not self._handle_key(event):
            super().keyPressEvent(event)

    def _handle_key(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self._action_pass_zero()
            return True
        if key == Qt.Key_P:
            self._action_autopass()
            return True
        if Qt.Key_0 <= key <= Qt.Key_9:
            self._action_pick(key - Qt.Key_0)
            return True
        return False

    def _action_pick(self, digit):
        if self._awaiting and 0 <= digit < len(self._actions):
            self._submit(digit)

    def _action_pass_zero(self):
        """Spacebar = pass, but only when action 0 is the pass option (mirrors
        the TUI). Otherwise a no-op, so it can never fire a non-pass action."""
        if not self._awaiting:
            return
        if self._actions and self._actions[0]["category"] == 0:
            self._submit(0)

    def _action_autopass(self):
        """Pass priority every optional window until the next upkeep, stopping
        early for any mandatory decision. The driver auto-drives once engaged."""
        if not self._awaiting:
            return
        if not self._driver.engage_autopass():
            return
        self._append_log("[You] Autopass to next upkeep")
        self._awaiting = False
        self._menu.clear()
        self._prompt.setText("Autopass — passing to next upkeep...")

    def _submit(self, idx):
        if not self._awaiting:
            return
        if 0 <= idx < len(self._actions) and self._actions[idx]["category"] != 0:
            self._append_log(f"[You] {self._actions[idx]['description']}")
        self._awaiting = False
        self._menu.clear()
        self._prompt.setText("...")
        self._driver.submit(idx)

    # ----- helpers -----

    def _row_for_index(self, action_index):
        for row in range(self._menu.count()):
            if self._menu.item(row).data(Qt.UserRole) == action_index:
                return row
        return None

    def _append_log(self, text):
        self._log.appendPlainText(text)

    def _rebuild_stack(self, stack, mirrored):
        if not stack:
            label = QLabel("Stack: (empty)")
            label.setStyleSheet("color: #6a6a72;")
            self._stack_row.set_cards([label])
            return
        widgets = []
        for e in stack:
            w = StackItemWidget(e, stack_target_refs(e, mirrored))
            name = e["name"]
            self._stack_thumbs.setdefault(name, []).append(w._thumb)
            self._apply_image(w._thumb, name, None)
            widgets.append(w)
        self._stack_row.set_cards(widgets)

    def _rebuild_bf(self, perms_row, lands_row, perms, controller):
        """Fill one player's battlefield: non-lands into `perms_row`, lands into
        `lands_row`."""
        perms_row.set_cards(
            [self._mk_perm(p, controller) for p in perms if not p.get("is_land")])
        lands_row.set_cards(
            [self._mk_perm(p, controller) for p in perms if p.get("is_land")])

    def _rebuild_hand(self, hand):
        self._hand_row.set_cards(
            [self._mk_hand(c) for c in hand])

    def _mk_perm(self, p, controller):
        token_pt = self._token_pt(p)
        w = CardWidget(p["name"], p["card_idx"], controller, "battlefield",
                       perm=p, token_pt=token_pt)
        self._register(w, p["name"])
        self._apply_image(w, p["name"], token_pt)
        return w

    def _mk_hand(self, c):
        w = CardWidget(c["name"], c["card_idx"], "self", "hand",
                       icon=hand_type_icon(c["card_idx"]))
        self._register(w, c["name"])
        self._apply_image(w, c["name"], None)
        return w

    @staticmethod
    def _token_pt(p):
        """The Scryfall token lookup key for a permanent: (p, t) for a P/T token,
        (None, None) for a non-creature token (Clue/Food/Treasure), else None (a
        real named card)."""
        if p.get("card_idx") != decode._TOKEN_IDX:
            return None
        if "power" in p:
            return (p["power"], p["toughness"])
        return (None, None)

    def _apply_image(self, widget, name, token_pt):
        """Ask the provider for `name`'s art; slot it in immediately if ready.
        A miss leaves the placeholder; an async load arrives via image_ready."""
        pm = self._provider.request(name, token_pt)
        if pm is not None:
            widget.set_pixmap(pm)

    def _on_image_ready(self, name):
        """A download finished: push the art into every live widget for `name`."""
        for w in self._registry.get(name, []):
            pm = self._provider.request(w._name, w._token_pt)
            if pm is not None:
                w.set_pixmap(pm)
        for th in self._stack_thumbs.get(name, []):
            pm = self._provider.request(name, None)
            if pm is not None:
                th.set_pixmap(pm)

    def _register(self, widget, name):
        widget.clicked.connect(self._on_card_clicked)
        self._registry.setdefault(name, []).append(widget)

    def _phase_status(self, gs):
        active = "A" if gs["active_is_a"] else "B"
        prio = gs["priority_player"]
        return (self._match_strip(gs.get("match"))
                + f"Turn {gs['turn']} · Active {active} · Priority {prio}")

    def _match_strip(self, match):
        """Compact bo3 score prefix ("Game 2 · You 1-0 · "), or "" outside a
        best-of-three match (see the TUI's _match_strip for the rationale)."""
        if not self._bo3 or not match:
            return ""
        you, opp = match["self_wins"], match["opp_wins"]
        game_n = you + opp + 1
        prefix = f"Game {game_n} · You {you}-{opp}"
        if match.get("is_sideboard"):
            prefix += " · SIDEBOARD"
        return prefix + "  ·  "

    @staticmethod
    def _info_line(label, p, library):
        # Poison (☠) and energy (⚡) shown only when the player actually has some.
        counters = ""
        if p.get("poison", 0) > 0:
            counters += f"  ☠ {p['poison']}"
        if p.get("energy", 0) > 0:
            counters += f"  ⚡ {p['energy']}"
        return (f"{label}  ♥ {p['life']}{counters}  "
                f"mana [{decode.fmt_mana(p['mana'])}]  "
                f"hand {p['hand_count']}  lib {library}")

    # ----- smoke-test auto-drive -----

    def _maybe_smoke(self, human_turn):
        """Headless sanity driver: with ROBOMAGE_GUI_SMOKE=1 quit after the first
        render; with N>1 auto-submit the first legal action on each human turn
        for N decisions, then quit."""
        if self._smoke_n is None:
            return
        if self._smoke_n <= 1:
            QTimer.singleShot(150, self.close)
            return
        if not human_turn:
            return
        if self._smoke_played >= self._smoke_n:
            QTimer.singleShot(60, self.close)
            return
        self._smoke_played += 1
        QTimer.singleShot(15, self._smoke_submit)

    def _smoke_submit(self):
        if self._awaiting and self._actions:
            self._submit(0)


def _smoke_n_from_env():
    v = os.environ.get("ROBOMAGE_GUI_SMOKE")
    if not v:
        return None
    try:
        return max(1, int(v))
    except ValueError:
        return 1


# ── Entry point (called by play.py --gui) ─────────────────────────────────────

def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver", bo3=True):
    """Launch the Qt game board. Same signature/semantics as tui_game.run:
    `model_path` of None/"scripted" ⇒ rule-based opponent; any
    opponents.make_controller spec works; `bo3` (default True) plays a
    best-of-three match with sideboarding in one engine process."""
    session = build_session(binary_path, model_path, human_player=human_player,
                            human_deck=human_deck, model_deck=model_deck, bo3=bo3)
    app = QApplication.instance() or QApplication([])
    font = QFont("JetBrains Mono")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "monospace"])
    app.setFont(font)
    app.setStyleSheet(_QSS)

    window = GameWindow(session)
    try:
        window.show()
        window.start()
        app.exec()
    finally:
        # The front end owns closing the env (the driver leaves it open).
        session.env.close()
    return 0
