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
Add ``ROBOMAGE_ANALYSIS_SMOKE=1`` to also force the analysis window on (uniform
evaluator, no torch) and fail (exit 1) unless a live analysis run delivered
stats during the smoke.
"""

import html
import json
import os
import sys
import threading
import time

import numpy as np

from PySide6.QtCore import Qt, QObject, Signal, QTimer, QRectF, QSize
from PySide6.QtGui import (QColor, QFont, QPainter, QPen, QBrush, QPixmap,
                           QPainterPath, QKeySequence, QShortcut)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                               QVBoxLayout, QHBoxLayout, QScrollArea, QSplitter,
                               QListWidget, QListWidgetItem, QPlainTextEdit,
                               QToolButton, QSizePolicy, QDialog, QComboBox,
                               QFormLayout, QDialogButtonBox, QMessageBox,
                               QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox)

from env import _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE
import decode
import scryfall_cache
from game_driver import (GameDriver, build_session, decode_human_frame,
                         actions_for_card, action_zone, stack_target_refs,
                         menu_label, prompt_text, hand_type_icon, _edge_colors,
                         _STEP_ABBR)

# ── Card geometry ─────────────────────────────────────────────────────────────
# Untapped cards are portrait at the real 63:88 Magic aspect; a tapped card is
# painted rotated 90° into a CARD_H×CARD_W landscape box so a row of them reflows
# just like a tabletop. Rows are a fixed height so 10+ permanents simply scroll.
#
# Base size is scaled up ~15% versus the original 80x112: moving the graveyard
# line into the sidebar (see GraveyardPanel) and pinning the YOU/OPPONENT info
# lines to their natural height (see GameWindow._build_ui) frees vertical space
# in the board column, which goes back into taller rows and bigger cards rather
# than sitting empty.
_BASE_SCALE = 1.15
CARD_W = round(80 * _BASE_SCALE)
CARD_H = round(112 * _BASE_SCALE)
CARD_RADIUS = 7
# Per-row vertical chrome (contents margins + horizontal scrollbar allowance)
# baked into ROW_H: a row `h` px tall paints cards `h - ROW_CHROME` px tall. This
# is what lets a row scale its cards to whatever height the vertical splitter
# gives it (see CardRow.resizeEvent / CardWidget.set_card_height).
ROW_CHROME = 22
MIN_CARD_H = 44             # smallest a card is allowed to shrink to in a row
ROW_H = CARD_H + ROW_CHROME  # card + margins + horizontal scrollbar allowance
STACK_THUMB_W = round(46 * _BASE_SCALE)
STACK_THUMB_H = round(64 * _BASE_SCALE)
STACK_ROW_H = STACK_THUMB_H + 20
STACK_EMPTY_H = 22          # collapsed floor the stack pane can be dragged down to

# The hand row is 1.5x the standard row height; hand cards are scaled up to
# fill it (same aspect ratio, same narrow margin allowance as ROW_H) rather
# than sitting small and centered in extra blank space.
HAND_ROW_H = int(ROW_H * 1.5)
HAND_CARD_H = HAND_ROW_H - (ROW_H - CARD_H)
HAND_CARD_W = round(HAND_CARD_H * CARD_W / CARD_H)

# Attacking creatures get a dashed border in this red — brighter/more saturated
# than any muted color-identity red so an attacker reads distinctly (mirrors the
# TUI's _ATTACK_BORDER).
_ATTACK_BORDER = "#ff2b2b"
_CARD_FILL = QColor("#1c1c24")
_CARD_TEXT = QColor("#e6e6ea")
_CHIP_BG = QColor("#0c0c10")

# Cross-highlight accents (Phase D). Amber = "a menu action targets this card"
# (mirrors the TUI's $warning action-linked highlight); red = "a stack object
# targets this card / player" (the TUI's $error stack-target). Both are painted
# as a translucent tint plus a solid border so they read over card art too.
_HL_ACTION = QColor("#d8a12a")            # amber (action-linked)
_HL_STACK = QColor("#e05555")             # red (stack target)
_HL_ACTION_FILL = QColor(0xd8, 0xa1, 0x2a, 70)
_HL_STACK_FILL = QColor(0xe0, 0x55, 0x55, 70)
# Menu-row background used when a hovered card lights up the actions it feeds
# (the reverse direction of the card-linked highlight).
_MENU_HL_BG = QColor(0xd8, 0xa1, 0x2a, 70)


# ── QSS theme ─────────────────────────────────────────────────────────────────
# Dark terminal-ish palette matching the TUI: near-black background, light grey
# text, green "you" / red "opponent" accents, 1px panel borders, slightly
# lighter land rows, amber selection, a bold boosted prompt bar.
_QSS = """
QMainWindow, QWidget { background: #101014; color: #d8d8d8; }
QLabel#oppInfo  { color: #e05555; padding: 2px 4px; }
QLabel#selfInfo { color: #33d17a; padding: 2px 4px; }
QLabel#graveyards { color: #9a9aa2; padding: 2px 4px; }
QLabel#exile { color: #9a9aa2; padding: 2px 4px; }
QLabel#knownHand { color: #9a9aa2; padding: 2px 4px; }
QLabel#prompt {
    font-weight: bold; background: #24242e; color: #f0f0f4; padding: 4px 6px;
    border: 1px solid #2a2a33;
}
/* A stack object hovered on the board paints its player target red — the
   YOU/OPPONENT info line analog of the CardWidget stack-target tint. */
QLabel#oppInfo[stackTarget="true"], QLabel#selfInfo[stackTarget="true"] {
    background: #e05555; color: #101014; font-weight: bold;
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
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }
/* Oracle-text popup (frameless, always-on-top): amber-bordered dark panel. */
QWidget#oraclePopup { background: #16161c; border: 1px solid #d8a12a; }
QWidget#oraclePopup QLabel { background: transparent; border: none; color: #d8d8d8; }
/* Launcher (intro screen). */
QLabel#launcherTitle { font-size: 22px; font-weight: bold; color: #f0f0f4; }
QLabel#launcherSubtitle { color: #9a9aa2; padding-bottom: 4px; }
QGroupBox {
    border: 1px solid #2a2a33; margin-top: 10px; padding: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #b8b8c0; }
QComboBox {
    border: 1px solid #2a2a33; background: #0c0c10; color: #d8d8d8; padding: 3px 6px;
}
QComboBox QAbstractItemView {
    background: #0c0c10; color: #d8d8d8;
    selection-background-color: #d8a12a; selection-color: #101014;
}
QPushButton {
    border: 1px solid #2a2a33; background: #24242e; color: #f0f0f4; padding: 5px 12px;
}
QPushButton:hover { background: #2f2f3a; }
QPushButton:default { border: 1px solid #d8a12a; }
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
    # Hover in/out — emits self on enter, None on leave. Wired to the
    # action<->card cross-highlighting (the window lights up the menu rows this
    # card feeds) and drives the Q-hold oracle popup's "currently hovered card".
    hovered = Signal(object)
    # Right-button hold: show/hide this card's oracle-text popup. Emit self on
    # press so the window can read the card's id/name/token_pt.
    oracle_show = Signal(object)
    oracle_hide = Signal()

    def __init__(self, name, card_idx, controller, zone, *, perm=None, icon="",
                 token_pt=None, card_size=None):
        super().__init__()
        self._name = name
        self._card_idx = card_idx
        self._controller = controller        # "self" | "opp"
        self._zone = zone                    # "battlefield" | "hand"
        self._icon = icon                    # hand-card type icon, else ""
        self._token_pt = token_pt            # (p, t) for a token, else None
        self._pixmap = None                  # Phase C art (full 'normal'), if set
        self._scaled = None                  # cached card-sized scale of _pixmap
        self._highlight = None               # None | "action_linked" | "stack_target"
        self._edge_colors = _edge_colors(decode.card_border_colors(card_idx))
        # Untapped (portrait) card size; defaults to the standard CARD_W/CARD_H
        # but a larger box (e.g. hand cards) can be passed in. All paint metrics
        # (fonts, chips, radius) scale proportionally off this.
        self._card_w, self._card_h = card_size or (CARD_W, CARD_H)
        self._scale = self._card_w / CARD_W

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
        return (QSize(self._card_h, self._card_w) if self._tapped
                else QSize(self._card_w, self._card_h))

    def set_pixmap(self, pixmap):
        """Store Phase C card art (QPixmap) and repaint. A null/None pixmap
        reverts to the text placeholder. The full 'normal' art is scaled to the
        card size once (cached in _scaled) so paintEvent never rescales."""
        self._pixmap = pixmap
        self._scaled = None                  # invalidate cached scale
        self.update()

    def set_card_height(self, portrait_h):
        """Rescale the card to a target *portrait* height (keeping aspect ratio),
        then repaint. Called by CardRow when the row is resized in the vertical
        splitter so the cards grow/shrink to fill their panel instead of sitting
        in blank space. All paint metrics derive from _scale, so this is enough."""
        new_h = max(1, int(portrait_h))
        new_w = max(1, round(new_h * CARD_W / CARD_H))
        if (new_w, new_h) == (self._card_w, self._card_h):
            return
        self._card_w, self._card_h = new_w, new_h
        self._scale = self._card_w / CARD_W
        self._scaled = None                  # invalidate cached art scale
        self.setFixedSize(self.sizeHint())
        self.update()

    def set_highlight(self, kind):
        """Toggle a cross-highlight overlay: None, "action_linked" (amber — a
        hovered menu action targets this card) or "stack_target" (red — a hovered
        stack object targets it). Repaints with the highlight border/tint
        overriding the normal per-edge / attacking borders (precedence
        stack_target > action_linked > attacking > normal)."""
        if self._highlight != kind:
            self._highlight = kind
            self.update()

    def _card_pixmap(self):
        """The card-sized (portrait CARD_W x CARD_H) smooth scale of the stored
        art, computed once and cached; None when no usable art is set."""
        if self._pixmap is None or self._pixmap.isNull():
            return None
        if self._scaled is None:
            self._scaled = self._pixmap.scaled(
                QSize(self._card_w, self._card_h),
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
        """Right-button hold → ask the window to show this card's oracle popup."""
        self.oracle_show.emit(self)

    def _on_right_release(self):
        """Right-button release → dismiss the oracle popup."""
        self.oracle_hide.emit()

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
            painter.translate(self._card_h, 0)
            painter.rotate(90)
        self._paint_card(painter, QRectF(0, 0, self._card_w, self._card_h))
        painter.end()

    def _paint_card(self, painter, rect):
        r = rect.adjusted(1.5, 1.5, -1.5, -1.5)
        path = QPainterPath()
        radius = CARD_RADIUS * self._scale
        path.addRoundedRect(r, radius, radius)

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

        # Borders (precedence: stack_target > action_linked > attacking >
        # per-edge color identity). A cross-highlight paints a translucent tint
        # over the body (so card art still reads through) plus a solid border,
        # overriding both the attacking dashed border and the color-identity
        # edges; real art otherwise carries its own frame color, so the per-edge
        # WUBRG borders are placeholders-only.
        if self._highlight == "stack_target":
            self._paint_highlight(painter, path, _HL_STACK_FILL, _HL_STACK)
        elif self._highlight == "action_linked":
            self._paint_highlight(painter, path, _HL_ACTION_FILL, _HL_ACTION)
        elif self._attacking:
            pen = QPen(QColor(_ATTACK_BORDER))
            pen.setWidthF(2 * self._scale)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        elif art is None:
            self._paint_edges(painter, r)

        # Name (wrapped, top). Over real art a translucent caption strip keeps
        # the name legible (unreadable directly on card art at this size).
        f = QFont(painter.font())
        f.setPointSizeF(7.0 * self._scale)
        painter.setFont(f)
        name_rect = QRectF(r.left() + 3, r.top() + 3, r.width() - 6, r.height() * 0.52)
        if art is not None:
            # The strip hugs the wrapped text's actual extent — covering the
            # whole name_rect (52% of the card) darkened the art badly.
            br = painter.boundingRect(
                name_rect, Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignTop,
                self._name)
            strip = QRectF(br).adjusted(-3, -2, 3, 2)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(12, 12, 16, 165))
            painter.drawRoundedRect(strip, 3, 3)
        painter.setPen(_CARD_TEXT)
        painter.drawText(name_rect, Qt.TextWordWrap | Qt.AlignHCenter | Qt.AlignTop,
                         self._name)

        # Hand-card type icon (top-right) — placeholder cards only; real art
        # already shows what the card is.
        if self._icon and art is None:
            isz = 14 * self._scale
            icon_rect = QRectF(r.right() - isz - 2, r.top() + 2, isz, isz)
            painter.drawText(icon_rect, Qt.AlignCenter, self._icon)

        # Status flags (small, below the name).
        if self._flags:
            painter.setPen(QColor("#c8a24a"))
            ff = QFont(painter.font())
            ff.setPointSizeF(6.0 * self._scale)
            painter.setFont(ff)
            flag_rect = QRectF(r.left() + 3, r.top() + r.height() * 0.55,
                               r.width() - 6, 12 * self._scale)
            painter.drawText(flag_rect, Qt.AlignHCenter | Qt.AlignVCenter,
                             " ".join(self._flags))

        # Counters chip (bottom-left) and P/T-or-loyalty chip (bottom-right).
        if self._counters:
            self._paint_chip(painter, self._counters, r, right=False)
        if self._chip:
            self._paint_chip(painter, self._chip, r, right=True)

    def _paint_highlight(self, painter, path, fill, border):
        """Overlay a cross-highlight: translucent `fill` tint + solid `border`."""
        painter.fillPath(path, QBrush(fill))
        pen = QPen(border)
        pen.setWidthF(2 * self._scale)
        painter.setPen(pen)
        painter.drawPath(path)

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
            pen.setWidthF(3 * self._scale)
            painter.setPen(pen)
            painter.drawLine(a, b)

    def _paint_chip(self, painter, text, r, *, right):
        """Draw a small filled chip with `text` in a bottom corner of `r`."""
        f = QFont(painter.font())
        f.setPointSizeF(6.5 * self._scale)
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
    the same text the TUI shows. Carries its human-frame target refs so hovering
    it can highlight those targets on the board (see GameWindow._on_stack_hover)."""

    # Hover in/out — emits self on enter, None on leave.
    hovered = Signal(object)

    def __init__(self, entry, target_refs):
        super().__init__()
        self._target_refs = target_refs
        self.setMouseTracking(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 8, 2)
        layout.setSpacing(6)
        self._thumb = _StackThumb(entry["card_idx"])
        layout.addWidget(self._thumb)
        label = QLabel(_stack_item_label(entry))
        label.setWordWrap(True)
        layout.addWidget(label, 1)

    def enterEvent(self, event):
        self.hovered.emit(self)

    def leaveEvent(self, event):
        self.hovered.emit(None)


def _stack_item_label(e):
    """Human-readable stack-object label (verbatim from the TUI)."""
    kind = "spell" if e["is_spell"] else "ability"
    label = f"{e['name']} ({kind}, {e['controller']})"
    if e.get("targets"):
        label += " → " + "; ".join(e["targets"])
    return label


# ── Fixed-height, horizontal-only card row ────────────────────────────────────

class CardRow(QScrollArea):
    """A single board row: a horizontal-only scroll area whose height is driven by
    the enclosing vertical splitter. `preferred` is the row's initial height (used
    to seed the splitter); it never fixes the height. When the splitter resizes the
    row, resizeEvent rescales the row's cards to fill the new height, so dragging a
    panel taller means bigger cards (and shorter means smaller), not blank space. A
    crowded row still scrolls sideways since cards keep their aspect ratio."""

    def __init__(self, height=ROW_H, *, land=False, min_height=None):
        super().__init__()
        self.preferred = height
        # No setFixedHeight: the splitter owns the height. Keep a floor so a row
        # can't be dragged smaller than a legible card (the stack row passes a
        # lower floor so it can be collapsed to a thin strip).
        self.setMinimumHeight(min_height
                              if min_height is not None else MIN_CARD_H + ROW_CHROME)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Ignored)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._layout = QHBoxLayout(self._inner)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)
        self._layout.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.setWidget(self._inner)
        if land:
            # Land rows sit on a slightly lighter background (mirrors the TUI's
            # .bf-row.lands panel tint).
            self.setStyleSheet(
                "QScrollArea, QScrollArea > QWidget, QWidget { background: #16161c; }")

    def sizeHint(self):
        hint = super().sizeHint()
        hint.setHeight(self.preferred)
        return hint

    def _card_target_h(self):
        """Portrait card height that fills the current row height."""
        return max(MIN_CARD_H, self.height() - ROW_CHROME)

    def _rescale_cards(self):
        """Push the current row height into every scalable card in the row."""
        h = self._card_target_h()
        for i in range(self._layout.count()):
            w = self._layout.itemAt(i).widget()
            if hasattr(w, "set_card_height"):
                w.set_card_height(h)

    def resizeEvent(self, event):
        # Fires on window resize and on every splitter-handle drag; keep the cards
        # sized to the row so panels "adjust themselves accordingly".
        super().resizeEvent(event)
        self._rescale_cards()

    def set_cards(self, widgets):
        """Replace the row's contents with `widgets` (list of card/thumb/label)."""
        while self._layout.count():
            item = self._layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # hide, NOT setParent(None): a parentless widget becomes a
                # top-level window, which on macOS can flash on screen as a
                # small ghost window before deleteLater destroys it.
                w.hide()
                w.deleteLater()
        for w in widgets:
            self._layout.addWidget(w)
        self._rescale_cards()


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


# ── Oracle-text popup ─────────────────────────────────────────────────────────

# Oracle-popup card image target height (px); the plan's ~420 px. The full
# 'normal' 488×680 pixmap scales down to this sharply.
ORACLE_IMG_H = 420


class OraclePopup(QWidget):
    """Frameless, always-on-top card-inspect popup (the Qt analog of the TUI's
    #oracle banner): the card's full art over its name, mana cost and oracle text.

    Shown while the right mouse button is held over a card or while 'Q' is held
    (see GameWindow). It never steals focus or blocks input — a Qt.Tool window
    with WA_ShowWithoutActivating and no focus policy — so the board keeps its
    keyboard/mouse grab underneath. Art comes from the shared ImageProvider; a
    miss shows a placeholder panel and the popup subscribes to image_ready so a
    late download fills in while it is still open."""

    def __init__(self, provider, parent=None):
        super().__init__(parent, Qt.FramelessWindowHint | Qt.Tool |
                         Qt.WindowStaysOnTopHint)
        self.setObjectName("oraclePopup")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setFocusPolicy(Qt.NoFocus)
        self._provider = provider
        self._name = None
        self._token_pt = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignCenter)
        self._image.setFixedHeight(ORACLE_IMG_H)
        layout.addWidget(self._image)
        self._text = QLabel()
        self._text.setWordWrap(True)
        self._text.setTextFormat(Qt.RichText)
        self._text.setMaximumWidth(360)
        layout.addWidget(self._text)
        provider.image_ready.connect(self._on_image_ready)

    def show_card(self, card_idx, name, token_pt=None):
        self._name = name
        self._token_pt = token_pt
        cost = decode.fmt_mana_cost(decode.card_mana_cost(card_idx))
        oracle = decode.card_oracle_text(card_idx)
        head = f"<b>{html.escape(name)}</b>"
        if cost:
            head += (f"&nbsp;&nbsp;<span style='color:#d8a12a;'><b>"
                     f"{html.escape(cost)}</b></span>")
        if oracle:
            body = html.escape(oracle).replace("\n", "<br>")
        else:
            body = "<i>(no oracle text)</i>"
        self._text.setText(head + "<br>" + body)
        self._load_image()
        self.adjustSize()
        self._center_on_parent()
        self.show()
        self.raise_()

    def _load_image(self):
        pm = self._provider.request(self._name, self._token_pt)
        if pm is not None and not pm.isNull():
            self._image.setPixmap(pm.scaledToHeight(ORACLE_IMG_H,
                                                    Qt.SmoothTransformation))
        else:
            # A miss leaves the placeholder text; a later download arrives via
            # image_ready (only names that actually downloaded emit).
            self._image.setPixmap(QPixmap())
            self._image.setText("(loading image…)")

    def _on_image_ready(self, name):
        if self.isVisible() and name == self._name:
            self._load_image()
            self.adjustSize()
            self._center_on_parent()

    def _center_on_parent(self):
        parent = self.parentWidget()
        if parent is None:
            return
        c = parent.frameGeometry().center()
        self.move(c.x() - self.width() // 2, c.y() - self.height() // 2)


# ── The window ────────────────────────────────────────────────────────────────

class _CollapsiblePanel(QWidget):
    """Base for a right-hand sidebar section: a slim header with a collapse
    toggle over a content widget. Collapsing shrinks the panel to just the
    toggle strip (the vertical splitter honors the maximumHeight) so the
    panels below it get the extra room; the arrow button brings it back."""

    _COLLAPSED_H = 26

    def __init__(self, title, content, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._btn = QToolButton()
        self._btn.setArrowType(Qt.DownArrow)      # points toward the collapse
        self._btn.setAutoRaise(True)
        self._btn.setToolTip(f"Collapse/expand {title.lower()}")
        self._btn.clicked.connect(self.toggle)
        self._title = QLabel(title)
        header.addWidget(self._btn)
        header.addWidget(self._title)
        header.addStretch(1)
        v.addLayout(header)
        self._content = content
        v.addWidget(content)
        self._collapsed = False

    def toggle(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed):
        self._collapsed = collapsed
        self._content.setVisible(not collapsed)
        self._title.setVisible(not collapsed)
        self._btn.setArrowType(Qt.UpArrow if collapsed else Qt.DownArrow)
        self.setMaximumHeight(self._COLLAPSED_H if collapsed else 16777215)


class LogPanel(_CollapsiblePanel):
    """The game log, stacked in the right-hand sidebar above the action menu."""

    def __init__(self, parent=None):
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(2000)
        super().__init__("Log", self.text, parent)


class GraveyardPanel(_CollapsiblePanel):
    """Both players' graveyard contents, in the right-hand sidebar above the
    log (moved off the board to reclaim vertical space for the card rows)."""

    def __init__(self, parent=None):
        self._text = QLabel()
        self._text.setObjectName("graveyards")
        self._text.setWordWrap(True)
        super().__init__("Graveyards", self._text, parent)

    def set_text(self, text):
        self._text.setText(text)


class ExilePanel(_CollapsiblePanel):
    """Both players' exile-zone contents, in the right-hand sidebar."""

    def __init__(self, parent=None):
        self._text = QLabel()
        self._text.setObjectName("exile")
        self._text.setWordWrap(True)
        super().__init__("Exile", self._text, parent)

    def set_text(self, text):
        self._text.setText(text)


class KnownHandPanel(_CollapsiblePanel):
    """Opponent-hand cards whose identity the human currently KNOWS — the
    engine's belief state (revealed by Thoughtseize-style discard looks,
    tutors, bounces, ...), mirrored from the obs known-opp-hand block. The
    engine clears a slot when that card leaves their hand or hides again
    (shuffle), so this always reflects live knowledge, not history."""

    def __init__(self, parent=None):
        self._text = QLabel()
        self._text.setObjectName("knownHand")
        self._text.setWordWrap(True)
        super().__init__("Known opp hand", self._text, parent)

    def set_cards(self, names):
        self._text.setText(", ".join(names) if names else "—")


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
        # Flat list of every live CardWidget this frame (across both battlefields
        # + hand), so cross-highlighting can scan by card_idx/controller/zone.
        self._card_widgets = []
        self._provider = ImageProvider()
        self._actions = []
        self._awaiting = False
        self._thread = None

        # Cross-highlighting / oracle state (Phase D). _hovered_card is the
        # CardWidget the mouse is currently over (the Q-hold oracle anchor and the
        # card->menu highlight source); _stack_target_labels are the info-line
        # QLabels currently painted as a stack target.
        self._hovered_card = None
        self._stack_target_labels = []
        # Elapsed-seconds "opponent is thinking" ticker (a model/search opponent
        # only); started/stopped on the UI thread by on_opp_thinking.
        self._think_timer = None
        self._think_start = 0.0

        # Smoke-test auto-drive (headless CI-less sanity check). See module docs.
        self._smoke_n = _smoke_n_from_env()
        self._smoke_seen = 0
        self._smoke_played = 0
        self._smoke_analysis_waits = 0

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
        # Frameless oracle-text popup, parented to the window so it centers on it
        # and stays above it (see OraclePopup); shown on Q-hold / right-click-hold.
        self._oracle = OraclePopup(self._provider, self)
        self._driver = GameDriver(
            env=session.env, opp_act=session.opp_act, opp_is_a=session.opp_is_a,
            is_model=session.is_model, opp_label=session.opp_label,
            bo3=session.bo3, sink=self._bridge,
            clock_fn=session.clock_fn, pace_idle=session.pace_idle)

        # Analysis window (separate top-level window; F9 toggles). Built only
        # when the session was assembled with an AnalysisConfig — its worker
        # thread is idle until the first analyze, so construction is cheap.
        self._analysis = None
        if getattr(session, "analysis_cfg", None) is not None:
            from gui_analysis import AnalysisWindow
            self._analysis = AnalysisWindow(session.env, session.analysis_cfg,
                                            self._opp_is_a)
            QShortcut(QKeySequence("F9"), self, activated=self._toggle_analysis)
            # A search opponent's own per-decision SearchResult is free —
            # surface it in the analysis window (shown only when its reveal
            # toggle is on).
            ctrl = getattr(session, "controller", None)
            if ctrl is not None and hasattr(ctrl, "on_result"):
                ctrl.on_result = self._analysis.opp_search_sink()

        # App-wide filter so the arrow-keys→action-pane hop works from any
        # focused widget (log, scroll rows, ...), not just the window itself.
        QApplication.instance().installEventFilter(self)

        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        # >/< nudge the board/log division (parity with the TUI's resize_log).
        QShortcut(QKeySequence(Qt.Key_Greater), self,
                  activated=lambda: self._nudge_splitter(1))
        QShortcut(QKeySequence(Qt.Key_Less), self,
                  activated=lambda: self._nudge_splitter(-1))

        keys = ("Game starting...  Click a card or pick a numbered action. "
                "Keys: digits = pick, enter = confirm highlighted, space = pass, "
                "p = autopass, "
                "hold q or right-click a card = oracle text, </> = resize log, "
                "ctrl+q = quit.")
        if self._analysis is not None:
            keys += "  F9 = analysis window (F5 analyze, Shift+F5 stop)."
        self._append_log(keys)

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
        # Fixed vertical policy: a single-line label should hug its text, not
        # soak up the vbox's leftover space and sit padded above/below it.
        self._opp_info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        v.addWidget(self._opp_info)

        # The six card rows live in a vertical splitter so the human can drag the
        # dividers to re-apportion board space (e.g. give the hand more room, or
        # collapse the stack). Each row scales its cards to the height it's given,
        # so the panels "adjust themselves accordingly" as the handles move. The
        # phase strip and the YOU/OPPONENT info lines stay outside the splitter —
        # they're single-line labels, not resizable panels.
        # Opponent's rows are flipped (lands on top, permanents below) so the two
        # players' non-land rows sit adjacent across the stack.
        self._opp_lands = CardRow(land=True)
        self._opp_perms = CardRow()
        # Stack row gets a low floor so it can be dragged down to a thin strip when
        # the stack is empty (which it usually is).
        self._stack_row = CardRow(height=STACK_ROW_H, min_height=STACK_EMPTY_H)
        # Self: permanents on top, lands below — mirror of the opponent so the
        # non-land rows are the ones adjacent to the stack.
        self._self_perms = CardRow()
        self._self_lands = CardRow(land=True)
        self._hand_row = CardRow(height=HAND_ROW_H)
        self._hand_row.setObjectName("handRow")

        self._board_split = QSplitter(Qt.Vertical)
        self._board_split.setObjectName("boardSplit")
        self._board_split.setChildrenCollapsible(False)
        self._board_rows = (self._opp_lands, self._opp_perms, self._stack_row,
                            self._self_perms, self._self_lands, self._hand_row)
        for row in self._board_rows:
            self._board_split.addWidget(row)
        self._board_split.setSizes([r.preferred for r in self._board_rows])
        v.addWidget(self._board_split, 1)

        # Stack-pane auto-collapse: an empty stack shrinks to a thin strip and
        # pops back to the human's last divider position when something's cast.
        # _stack_expanded_h remembers that position; splitterMoved keeps it in
        # sync whenever the human drags the stack while it's expanded.
        self._stack_idx = self._board_rows.index(self._stack_row)
        self._stack_collapsed = False
        self._stack_expanded_h = STACK_ROW_H
        self._board_split.splitterMoved.connect(self._on_board_split_moved)

        self._self_info = QLabel()
        self._self_info.setObjectName("selfInfo")
        self._self_info.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        v.addWidget(self._self_info)

        self._prompt = QLabel()
        self._prompt.setObjectName("prompt")
        self._prompt.setWordWrap(True)
        v.addWidget(self._prompt)

        # Action menu. Lives in the right-hand panel below the log (see below),
        # not under the board.
        self._menu = QListWidget()
        self._menu.itemActivated.connect(self._on_item_activated)
        self._menu.installEventFilter(self)      # route digit/space/p/Q keys
        # Menu-row hover -> highlight the card(s) that action targets. itemEntered
        # needs mouse tracking on the view; the viewport Leave clears it.
        self._menu.setMouseTracking(True)
        self._menu.itemEntered.connect(self._on_menu_item_entered)
        self._menu.viewport().installEventFilter(self)
        self._log_panel = LogPanel()
        self._log = self._log_panel.text
        self._gy_panel = GraveyardPanel()
        self._exile_panel = ExilePanel()
        self._known_panel = KnownHandPanel()

        # Right-hand panel: graveyards, exile, and known-opp-hand (collapsible,
        # off-board info) stacked above the log, with the action menu at the
        # bottom.
        self._right = QSplitter(Qt.Vertical)
        self._right.addWidget(self._gy_panel)
        self._right.addWidget(self._exile_panel)
        self._right.addWidget(self._known_panel)
        self._right.addWidget(self._log_panel)
        self._right.addWidget(self._menu)
        self._right.setStretchFactor(0, 0)
        self._right.setStretchFactor(1, 0)
        self._right.setStretchFactor(2, 0)
        self._right.setStretchFactor(3, 3)
        self._right.setStretchFactor(4, 2)

        # Horizontal splitter: board | sidebar panel. Kept on the window so the
        # >/< shortcuts can nudge the right panel's width.
        self._hsplit = QSplitter(Qt.Horizontal)
        self._hsplit.addWidget(board)
        self._hsplit.addWidget(self._right)
        self._hsplit.setStretchFactor(0, 3)
        self._hsplit.setStretchFactor(1, 1)
        self._hsplit.setSizes([860, 380])
        self.setCentralWidget(self._hsplit)
        self.resize(1280, 940)

    # ----- driver lifecycle -----

    def start(self):
        """Start the engine loop on a worker thread (called after show)."""
        if self._analysis is not None:
            self._analysis.show()
        self._thread = threading.Thread(target=self._driver.run,
                                        name="gui-driver", daemon=True)
        self._thread.start()

    def _toggle_analysis(self):
        if self._analysis is None:
            return
        if self._analysis.isVisible():
            self._analysis.hide()
        else:
            self._analysis.show()

    def closeEvent(self, event):
        # Signal shutdown and join the worker before the caller closes the env.
        # request_quit sets _quitting so a loop blocked in a long opponent search
        # returns silently once the env (and engine pipe) tear down. The analysis
        # worker (and its detached engine) shuts down first — it must never see
        # the primary env's teardown as a divergence.
        if self._analysis is not None:
            self._analysis.shutdown()
            self._analysis.close()
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
        self._gy_panel.set_text(
            f"Your GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:  {', '.join(gs['opp_graveyard']) or '—'}")
        self._exile_panel.set_text(
            f"Your exile: {', '.join(gs.get('self_exile', [])) or '—'}\n"
            f"Opp exile:  {', '.join(gs.get('opp_exile', [])) or '—'}")
        # The known-opp-hand block is viewer-relative: a mirrored (opponent-
        # perspective) frame's block is what THEY know about YOUR hand, so only
        # own-perspective frames update the panel (mirrors the hand-row rule).
        if not mirrored:
            self._known_panel.set_cards(gs.get("opp_known_hand", []))

        # A fresh menu/board: drop any stale hover so cross-highlights don't
        # linger onto the newly-rebuilt widgets, and close the oracle popup (its
        # card may no longer be on the board).
        self._hovered_card = None
        self._hide_oracle()
        self._clear_stack_targets()

        # Rebuild the board; registries are refreshed from scratch each frame.
        self._registry = {}
        self._stack_thumbs = {}
        self._card_widgets = []
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
        elif u.passive:
            # A --broadcast-steps frame: the board/phase strip just advanced a
            # step with no one to act; the driver paces these ~0.2s apart.
            self._prompt.setText("Playing through steps…")
        elif self._driver._autopass:
            self._prompt.setText("Autopass — passing to next upkeep...")
        else:
            self._prompt.setText(f"{self._opp_label} is thinking...")

        # Passive frames are display-only, not decisions — never analysis roots.
        if self._analysis is not None and not u.passive:
            self._analysis.on_decision(u)

        self._smoke_seen += 1
        self._maybe_smoke(u.human_turn)

    def on_log_lines(self, lines):
        for line in lines:
            if line.strip():
                self._append_log(line)

    def on_opp_thinking(self, active):
        """Show/hide the elapsed-seconds "opponent is thinking" indicator around a
        model/search opponent's move (and the paced fake-think idle beats, which
        arrive on this same signal). Runs on the UI thread, so the QTimer ticker
        is safe to start/stop here. Mirrors the TUI's on_opp_thinking/_tick_think."""
        if active:
            self._think_start = time.monotonic()
            self._tick_think()
            if self._think_timer is None:
                self._think_timer = QTimer(self)
                self._think_timer.timeout.connect(self._tick_think)
                self._think_timer.start(1000)
        elif self._think_timer is not None:
            self._think_timer.stop()
            self._think_timer = None
            # Restore the static thinking prompt; the next state update refreshes
            # it to the human's menu (or the next think) momentarily.
            self._prompt.setText(f"{self._opp_label} is thinking...")

    def _tick_think(self):
        elapsed = int(time.monotonic() - self._think_start)
        bank = ""
        remaining = self._driver.clock_remaining()
        if remaining is not None:
            r = int(remaining)
            bank = f"  ·  bank {r // 60}:{r % 60:02d}"
        self._prompt.setText(
            f"⏳ {self._opp_label} is thinking…  ({elapsed}s){bank}")

    def on_game_over(self, text):
        self._awaiting = False
        self._menu.clear()
        self._menu.setEnabled(False)
        self._prompt.setText(text + "  (Ctrl+Q to quit)")
        self._append_log(text)
        if self._analysis is not None:
            self._analysis.on_game_over()
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
            # Ambiguous -> don't guess: amber-highlight every option this card
            # feeds, move the cursor to the first, and let the human disambiguate.
            idxs = [a["index"] for a in matches]
            self._set_menu_highlights(idxs)
            row = self._row_for_index(idxs[0])
            if row is not None:
                self._menu.setCurrentRow(row)
            self._append_log(
                f"That card has {len(idxs)} options: "
                f"{', '.join(str(i) for i in idxs)} - pick a number.")

    def eventFilter(self, obj, event):
        # Intercept digit/space/p/Q in the menu so they drive the board instead
        # of the list's built-in typeahead; Enter still fires itemActivated. The
        # menu viewport's Leave clears the menu-row->card cross-highlight.
        et = event.type()
        # Any arrow key, wherever focus is, jumps keyboard focus to the action
        # pane (installed app-wide); once the pane has focus the list handles
        # arrows natively for row navigation.
        if (et == event.Type.KeyPress
                and event.key() in (Qt.Key_Up, Qt.Key_Down,
                                    Qt.Key_Left, Qt.Key_Right)
                and not self._menu.hasFocus()):
            self._menu.setFocus(Qt.ShortcutFocusReason)
            return True
        if obj is self._menu:
            if et == event.Type.KeyPress:
                if self._oracle_key(event, True):
                    return True
                if self._handle_key(event):
                    return True
            elif et == event.Type.KeyRelease:
                if self._oracle_key(event, False):
                    return True
        elif obj is self._menu.viewport() and et == event.Type.Leave:
            self._highlight_cards_for_action(None)
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if self._oracle_key(event, True):
            return
        if not self._handle_key(event):
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if self._oracle_key(event, False):
            return
        super().keyReleaseEvent(event)

    def _oracle_key(self, event, pressed):
        """Q pressed/released → show/hide the hovered card's oracle popup. The
        isAutoRepeat guard means OS key-repeat while Q is held is ignored (the
        popup stays up on the real press until the real release)."""
        if event.key() != Qt.Key_Q or event.isAutoRepeat():
            return False
        if pressed:
            if self._hovered_card is not None:
                self._show_oracle(self._hovered_card)
        else:
            self._hide_oracle()
        return True

    def _handle_key(self, event):
        key = event.key()
        if key in (Qt.Key_Return, Qt.Key_Enter):
            self._action_submit_current()
            return True
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

    def _action_submit_current(self):
        """Enter = submit the currently highlighted menu row, wherever the
        keyboard focus is (the list's own Enter only fires when it has focus)."""
        if not self._awaiting:
            return
        item = self._menu.currentItem()
        if item is None:
            return
        idx = item.data(Qt.UserRole)
        if idx is not None:
            self._submit(int(idx))

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
        if self._analysis is not None:
            self._analysis.on_human_acted()
        self._driver.submit(idx)

    # ----- helpers -----

    def _row_for_index(self, action_index):
        for row in range(self._menu.count()):
            if self._menu.item(row).data(Qt.UserRole) == action_index:
                return row
        return None

    def _append_log(self, text):
        self._log.appendPlainText(text)

    def _on_board_split_moved(self, pos, index):
        """Track the human's stack-divider position while the stack is expanded, so
        the next auto-restore returns to exactly where they left it."""
        if not self._stack_collapsed:
            self._stack_expanded_h = self._board_split.sizes()[self._stack_idx]

    def _set_stack_collapsed(self, collapsed):
        """Collapse the stack pane to a thin strip (empty stack) or restore it to
        the human's remembered height (something on the stack). The freed/reclaimed
        space is traded with the self-permanents pane directly below it, so the
        rest of the board layout is left untouched."""
        if collapsed == self._stack_collapsed:
            return
        self._stack_collapsed = collapsed
        split = self._board_split
        sizes = split.sizes()
        i = self._stack_idx
        j = i + 1                            # self-perms pane below the stack
        floor_j = self._board_rows[j].minimumHeight()
        target = STACK_EMPTY_H if collapsed else self._stack_expanded_h
        # Never starve the neighbour below its own minimum.
        target = max(STACK_EMPTY_H, min(target, sizes[i] + sizes[j] - floor_j))
        delta = target - sizes[i]
        if delta == 0:
            return
        sizes[i] += delta
        sizes[j] -= delta
        split.setSizes(sizes)

    def _rebuild_stack(self, stack, mirrored):
        # The stack row's height is owned by the vertical board splitter now, so we
        # only swap its contents here (no setFixedHeight — that would lock the pane
        # and override the human's drag). An empty stack auto-collapses the pane;
        # anything on the stack restores it to the human's last divider position.
        self._set_stack_collapsed(not stack)
        if not stack:
            label = QLabel("Stack: (empty)")
            label.setStyleSheet("color: #6a6a72;")
            self._stack_row.set_cards([label])
            return
        widgets = []
        for e in stack:
            w = StackItemWidget(e, stack_target_refs(e, mirrored))
            w.hovered.connect(self._on_stack_hover)
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
                       icon=hand_type_icon(c["card_idx"]),
                       card_size=(HAND_CARD_W, HAND_CARD_H))
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
        widget.hovered.connect(self._on_card_hover)
        widget.oracle_show.connect(self._show_oracle)
        widget.oracle_hide.connect(self._hide_oracle)
        self._registry.setdefault(name, []).append(widget)
        self._card_widgets.append(widget)

    # ----- cross-highlighting (Phase D) -----

    def _on_card_hover(self, widget):
        """Mouse entered (widget) / left (None) a card. Track it as the oracle
        anchor and light up the menu actions it can drive."""
        if widget is None:
            self._hovered_card = None
            self._set_menu_highlights([])
            return
        self._hovered_card = widget
        if self._awaiting:
            idxs = [a["index"] for a in actions_for_card(
                self._actions, widget._card_idx, widget._controller, widget._zone)]
            self._set_menu_highlights(idxs)

    def _on_menu_item_entered(self, item):
        """Mouse moved onto a menu row — highlight the card(s) it targets."""
        self._highlight_cards_for_action(item.data(Qt.UserRole))

    def _highlight_cards_for_action(self, index):
        """Amber-highlight the card(s) the given action index refers to; a None
        index (mouse left the list) clears them. Matches on card id, controller,
        and — when the action carries a zone_ref — the exact board zone, so a
        hand card never lights up its same-named battlefield twin (mirrors the
        TUI's _highlight_perms_for_action)."""
        self._clear_action_linked()
        if index is None or not (0 <= index < len(self._actions)):
            return
        a = self._actions[index]
        if a["card_idx"] < 0:
            return
        want_ctrl = a["controller"]
        want_zone = action_zone(a)
        for w in self._card_widgets:
            if w._card_idx != a["card_idx"]:
                continue
            if want_ctrl is not None:
                w_want = "own" if w._controller == "self" else "opp"
                if w_want != want_ctrl:
                    continue
            if want_zone is not None and w._zone != want_zone:
                continue
            w.set_highlight("action_linked")

    def _set_menu_highlights(self, indices):
        """Amber-background the menu rows in `indices` (the reverse-direction
        cross-highlight: a hovered card lights up the actions it feeds). Clears
        all other rows. No-op outside a human decision."""
        if not self._awaiting:
            return
        hl = set(indices)
        for row in range(self._menu.count()):
            it = self._menu.item(row)
            on = it.data(Qt.UserRole) in hl
            it.setBackground(QBrush(_MENU_HL_BG) if on else QBrush())

    def _on_stack_hover(self, widget):
        """Mouse entered (widget) / left (None) a stack object — paint its
        announced targets red on the board and the player info line."""
        if widget is None:
            self._clear_stack_targets()
        else:
            self._highlight_stack_targets(widget._target_refs)

    def _highlight_stack_targets(self, refs):
        """Paint a hovered stack object's announced targets red: matching
        battlefield permanent(s), and the YOU/OPPONENT info line for a player
        target. `refs` are already in the human frame (is_self == YOU)."""
        self._clear_stack_targets()
        for ref in refs:
            if ref["is_player"]:
                label = self._self_info if ref["is_self"] else self._opp_info
                label.setProperty("stackTarget", True)
                label.style().unpolish(label)
                label.style().polish(label)
                self._stack_target_labels.append(label)
            elif ref["card_idx"] >= 0:
                want = "self" if ref["is_self"] else "opp"
                for w in self._card_widgets:
                    if w._card_idx == ref["card_idx"] and w._controller == want:
                        w.set_highlight("stack_target")

    def _clear_action_linked(self):
        for w in self._card_widgets:
            if w._highlight == "action_linked":
                w.set_highlight(None)

    def _clear_stack_targets(self):
        for w in self._card_widgets:
            if w._highlight == "stack_target":
                w.set_highlight(None)
        for label in self._stack_target_labels:
            label.setProperty("stackTarget", False)
            label.style().unpolish(label)
            label.style().polish(label)
        self._stack_target_labels = []

    # ----- oracle popup (Phase D) -----

    def _show_oracle(self, widget):
        self._oracle.show_card(widget._card_idx, widget._name, widget._token_pt)

    def _hide_oracle(self):
        self._oracle.hide()

    def _nudge_splitter(self, delta):
        """Grow (delta>0) / shrink (delta<0) the right-hand log+menu panel by
        shifting ~40 px between the board and right panes."""
        sizes = self._hsplit.sizes()
        if len(sizes) != 2:
            return
        step = 40 * delta
        left = max(320, sizes[0] - step)
        right = max(200, sizes[1] + step)
        self._hsplit.setSizes([left, right])

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
        if not (self._awaiting and self._actions):
            return
        # Analysis smoke: when this decision is being analyzed, hold the
        # auto-play until the first chunk of stats lands (re-check every 100ms,
        # ~15s total budget across the run) — submitting would cancel the run.
        if (self._analysis is not None
                and os.environ.get("ROBOMAGE_ANALYSIS_SMOKE")
                and self._analysis.pending_analyzable
                and not self._analysis.smoke_ok
                and self._smoke_analysis_waits < 150):
            self._smoke_analysis_waits += 1
            QTimer.singleShot(100, self._smoke_submit)
            return
        self._submit(0)


def _smoke_n_from_env():
    v = os.environ.get("ROBOMAGE_GUI_SMOKE")
    if not v:
        return None
    try:
        return max(1, int(v))
    except ValueError:
        return 1


# ── Launcher (intro screen) ───────────────────────────────────────────────────

# Where the launcher remembers the last-used options so they autofill next time.
_LAUNCHER_CONFIG = os.path.join(
    os.path.expanduser("~"), ".robomage", "gui_launcher.json")

# Opponent presets offered in the launcher's editable combo. The value is the
# raw spec passed straight through to build_session (== play.py's --model): "gen"
# is the one generalist model, "scripted*" the rule-based tiers, "az:/azraw:/
# mcts:" the search wrappers. The combo stays editable so a checkpoint path or a
# spec with knobs (e.g. "az:gen?sims=200") can be typed in.
_OPPONENT_PRESETS = [
    ("Generalist model (gen)", "gen"),
    ("Scripted — hard (heuristic)", "scripted:hard"),
    ("Scripted — easy (greedy)", "scripted:easy"),
    ("Scripted — random", "scripted:random"),
    ("MCTS search (az:gen)", "az:gen"),
    ("Raw AZ policy (azraw:gen)", "azraw:gen"),
]

# Evaluator presets for the analysis window's combo (editable, like the
# opponent combo). "az:gen" is the calibrated AZ value net (warm-started from
# the PPO generalist when no AZ checkpoint exists); "mcts:gen" the PPO heads;
# "uniform" a torch-free neutral evaluator (visits-only search).
_ANALYSIS_EVAL_PRESETS = [
    ("AZ net (az:gen) — calibrated win%", "az:gen"),
    ("PPO heads (mcts:gen)", "mcts:gen"),
    ("Uniform (no model)", "uniform"),
]

# Deck subfolders hidden from the launcher dropdowns (mirrors tui.py's scan).
_DECK_SCAN_EXCLUDE = frozenset({"temp", "not_used"})


def _scan_decks():
    """All .dk deck stems under bin/resources/decks/ (recursive), decks/-relative
    (e.g. 'delver', 'league/ur_delver'). Mirrors tui.py._scan_decks so the GUI
    launcher offers the same decks the TUI form does, without importing textual."""
    from cli_spec import REPO_ROOT
    decks_dir = os.path.join(REPO_ROOT, "bin", "resources", "decks")
    out = []
    for root, dirs, files in os.walk(decks_dir):
        dirs[:] = sorted(d for d in dirs if d not in _DECK_SCAN_EXCLUDE)
        rel_dir = os.path.relpath(root, decks_dir).replace(os.sep, "/")
        for fname in files:
            if fname.endswith(".dk"):
                stem = os.path.splitext(fname)[0]
                out.append(stem if rel_dir == "." else f"{rel_dir}/{stem}")
    return sorted(out, key=lambda rel: (rel.count("/"), rel))


def _load_launcher_config():
    try:
        with open(_LAUNCHER_CONFIG, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_launcher_config(cfg):
    try:
        os.makedirs(os.path.dirname(_LAUNCHER_CONFIG), exist_ok=True)
        with open(_LAUNCHER_CONFIG, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except OSError:
        pass                                 # persistence is best-effort


class LauncherDialog(QDialog):
    """Intro screen shown when the GUI is launched with no game arguments.

    Exposes the same game-running knobs as the TUI play form — the human's deck,
    the opponent's deck, the opponent controller, which seat the human takes, and
    the match format — and seeds every field from the last session's choices
    (persisted to ~/.robomage/gui_launcher.json). On Start it hands a plain options
    dict back to run_launcher, which assembles the session and shows the board."""

    def __init__(self, binary_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RoboMage — New Game")
        self.setModal(True)
        self._binary = binary_path
        cfg = _load_launcher_config()
        decks = _scan_decks()

        form = QFormLayout()
        form.setSpacing(8)

        self._human_deck = self._deck_combo(decks, cfg.get("human_deck", "delver"))
        self._model_deck = self._deck_combo(decks, cfg.get("model_deck", "delver"))
        form.addRow("Your deck", self._human_deck)
        form.addRow("Opponent deck", self._model_deck)

        self._opponent = QComboBox()
        self._opponent.setEditable(True)
        for label, spec in _OPPONENT_PRESETS:
            self._opponent.addItem(label, spec)
        self._set_opponent(cfg.get("opponent", "gen"))
        self._opponent.setToolTip(
            "Opponent controller: 'gen' (the generalist model), a scripted tier, "
            "an az:/azraw:/mcts: search spec, or an explicit checkpoint path.")
        form.addRow("Opponent", self._opponent)

        self._player = QComboBox()
        for label in ("Random", "A (you go first)", "B (opponent first)"):
            self._player.addItem(label)
        self._player.setCurrentIndex(
            {"": 0, "A": 1, "B": 2}.get(cfg.get("player", ""), 0))
        form.addRow("You play as", self._player)

        self._format = QComboBox()
        self._format.addItem("Best of three (with sideboarding)", True)
        self._format.addItem("Single game", False)
        self._format.setCurrentIndex(0 if cfg.get("bo3", True) else 1)
        form.addRow("Match format", self._format)

        box = QGroupBox("Game setup")
        box.setLayout(form)

        self._search_box = self._build_search_box(cfg)
        self._analysis_box = self._build_analysis_box(cfg)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start game")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        title = QLabel("RoboMage")
        title.setObjectName("launcherTitle")
        subtitle = QLabel("Choose your matchup, then Start game.")
        subtitle.setObjectName("launcherSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(box)
        layout.addWidget(self._search_box)
        layout.addWidget(self._analysis_box)
        layout.addWidget(buttons)
        self.setMinimumWidth(460)
        self._options = None

        # The search knobs are only meaningful for an az:/mcts: search opponent —
        # show that group only then, and re-check whenever the opponent changes.
        self._opponent.currentTextChanged.connect(self._update_search_visibility)
        self._update_search_visibility()

    def _build_search_box(self, cfg):
        """The az:/mcts:-only search-tuning group (hidden for other opponents).
        Each field's '(default)' sentinel (a spinbox at its minimum) means 'omit
        the knob'; set values are appended to the spec query the same way
        play.py's --sims/--worlds/--think-time/--search-procs/--match-clock do."""
        form = QFormLayout()
        form.setSpacing(8)
        self._sims = self._int_field(
            cfg.get("sims"), "MCTS simulations per decision (more = stronger, slower).")
        self._worlds = self._int_field(
            cfg.get("worlds"), "Determinized worlds per decision (sims split across them).")
        self._think_time = self._float_field(
            cfg.get("think_time"), "Wall-clock seconds per decision — runs as many "
            "sims as fit in this budget (overrides the sims terminator).")
        self._search_procs = self._int_field(
            cfg.get("search_procs"), "Engine processes to fan the worlds across "
            "(world-parallel search; default 1).")
        self._match_clock = self._float_field(
            cfg.get("match_clock"), "Whole-match thinking bank in seconds (chess "
            "clock; 1500 = 25 min for a bo3). Each decision draws a variable budget.")
        self._paced = QComboBox()
        self._paced.addItem("Default (on with a time/clock budget)", None)
        self._paced.addItem("On — mask response-timing tells", True)
        self._paced.addItem("Off — instant obvious decisions", False)
        self._paced.setCurrentIndex(
            {None: 0, True: 1, False: 2}.get(cfg.get("paced"), 0))
        form.addRow("Simulations", self._sims)
        form.addRow("Worlds", self._worlds)
        form.addRow("Think time (s)", self._think_time)
        form.addRow("Search procs", self._search_procs)
        form.addRow("Match clock (s)", self._match_clock)
        form.addRow("Paced responses", self._paced)
        box = QGroupBox("Search opponent settings")
        box.setLayout(form)
        return box

    def _build_analysis_box(self, cfg):
        """The analysis-window group: enable + evaluator + search knobs. Unlike
        the search box this applies to every opponent type (the analysis runs on
        its own detached engine), so it is always visible."""
        form = QFormLayout()
        form.setSpacing(8)
        self._analysis_enable = QCheckBox("Open the analysis window (F9 toggles in-game)")
        self._analysis_enable.setChecked(bool(cfg.get("analysis_enabled", True)))
        self._analysis_eval = QComboBox()
        self._analysis_eval.setEditable(True)
        for label, spec in _ANALYSIS_EVAL_PRESETS:
            self._analysis_eval.addItem(label, spec)
        self._set_combo_spec(self._analysis_eval,
                             cfg.get("analysis_evaluator", "az:gen"))
        self._analysis_eval.setToolTip(
            "Evaluator behind the analysis search: az:gen (AZ net, calibrated "
            "win%), mcts:gen (PPO heads), uniform (no model), or a checkpoint "
            "path.")
        self._analysis_worlds = self._int_field(
            cfg.get("analysis_worlds"),
            "Determinized worlds per analysis run (hidden-zone samples).")
        self._analysis_cap = QSpinBox()
        self._analysis_cap.setRange(-1, 1_000_000)
        self._analysis_cap.setSpecialValueText("(default)")
        self._analysis_cap.setValue(
            -1 if cfg.get("analysis_cap") is None else int(cfg["analysis_cap"]))
        self._analysis_cap.setToolTip(
            "Simulation cap per analysis run (0 = run until stopped; "
            "adjustable in the window too).")
        self._analysis_auto = QCheckBox("Auto-analyze each decision")
        self._analysis_auto.setChecked(bool(cfg.get("analysis_auto", True)))
        form.addRow(self._analysis_enable)
        form.addRow("Evaluator", self._analysis_eval)
        form.addRow("Worlds", self._analysis_worlds)
        form.addRow("Sims cap", self._analysis_cap)
        form.addRow(self._analysis_auto)
        box = QGroupBox("Analysis window (MCTS evaluation of your decisions)")
        box.setLayout(form)
        return box

    @staticmethod
    def _set_combo_spec(combo, spec):
        idx = combo.findData(spec)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(spec)

    @staticmethod
    def _combo_spec(combo):
        """The combo's spec: a preset's data when the visible text still matches
        that preset's label, else the raw typed text."""
        idx = combo.currentIndex()
        if idx >= 0 and combo.itemText(idx) == combo.currentText():
            return combo.itemData(idx)
        return combo.currentText().strip()

    @staticmethod
    def _int_field(value, tooltip):
        sb = QSpinBox()
        sb.setRange(0, 1_000_000)            # 0 == minimum == "(default)" sentinel
        sb.setSpecialValueText("(default)")
        sb.setValue(int(value) if value else 0)
        sb.setToolTip(tooltip)
        return sb

    @staticmethod
    def _float_field(value, tooltip):
        sb = QDoubleSpinBox()
        sb.setRange(0.0, 100_000.0)          # 0.0 == minimum == "(default)" sentinel
        sb.setDecimals(1)
        sb.setSingleStep(0.5)
        sb.setSpecialValueText("(default)")
        sb.setValue(float(value) if value else 0.0)
        sb.setToolTip(tooltip)
        return sb

    @staticmethod
    def _spin_value(sb):
        """A spinbox's value, or None when it's parked on its '(default)' sentinel."""
        return sb.value() if sb.value() != sb.minimum() else None

    def _update_search_visibility(self, *_):
        self._search_box.setVisible(self._is_search_spec(self._opponent_spec()))
        self.adjustSize()

    @staticmethod
    def _is_search_spec(spec):
        # az:/mcts: run a tree search (knobs apply); azraw: is the raw policy (no
        # search), so it — like scripted/gen — hides the search group.
        return (spec or "").strip().lower().startswith(("az:", "mcts:"))

    @staticmethod
    def _deck_combo(decks, current):
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(decks)
        # Keep the remembered value even if it isn't a scanned stem (e.g. a
        # hand-typed temp deck) by setting the edit text directly.
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            combo.setEditText(current)
        return combo

    def _set_opponent(self, spec):
        self._set_combo_spec(self._opponent, spec)

    def _opponent_spec(self):
        """The opponent spec: a preset's data when the visible text still matches
        that preset's label, else the raw typed text."""
        return self._combo_spec(self._opponent)

    def _search_knobs(self):
        """(key, value) query pairs for the set search fields (empty ones omitted).
        Keys match the spec query grammar make_controller parses (time=/procs=/…),
        mirroring how play.py appends --think-time/--search-procs/etc."""
        sims = self._spin_value(self._sims)
        worlds = self._spin_value(self._worlds)
        time_val = self._spin_value(self._think_time)
        procs = self._spin_value(self._search_procs)
        clock = self._spin_value(self._match_clock)
        pairs = [("sims", sims), ("worlds", worlds), ("time", time_val),
                 ("procs", procs), ("clock", clock)]
        pairs = [(k, v) for k, v in pairs if v is not None]
        # Paced: explicit On/Off wins; on Default, mirror play.py and turn pacing
        # on whenever the opponent has a variable time budget (think-time/clock).
        paced = self._paced.currentData()
        has_variable_budget = time_val is not None or clock is not None
        if paced is False:
            pairs.append(("paced", 0))
        elif paced is True or (paced is None and has_variable_budget):
            pairs.append(("paced", 1))
        return pairs

    @staticmethod
    def _with_query(spec, pairs):
        """Append `pairs` to a controller spec's ?k=v&… query (later keys win in
        make_controller's parser, so appending is always safe)."""
        if not pairs:
            return spec
        sep = "&" if "?" in spec else "?"
        return spec + sep + "&".join(f"{k}={v}" for k, v in pairs)

    def _on_accept(self):
        human_deck = self._human_deck.currentText().strip()
        model_deck = self._model_deck.currentText().strip()
        opponent = self._opponent_spec()
        if not human_deck or not model_deck:
            QMessageBox.warning(self, "Missing deck",
                                "Pick a deck for both you and the opponent.")
            return
        if not opponent:
            QMessageBox.warning(self, "Missing opponent",
                                "Pick or type an opponent.")
            return
        # A search opponent carries its tuning knobs in the spec query; other
        # opponents ignore the (hidden) fields entirely.
        spec = opponent
        if self._is_search_spec(opponent):
            spec = self._with_query(opponent, self._search_knobs())
        # Resolve the generalist up front so a missing checkpoint fails on the
        # launcher (with a clear message) rather than deep in build_session.
        try:
            model_path = _resolve_opponent_spec(spec)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(self, "Opponent unavailable", str(exc))
            return

        analysis = None
        if self._analysis_enable.isChecked():
            evaluator = self._combo_spec(self._analysis_eval) or "az:gen"
            cap = self._analysis_cap.value()
            analysis = dict(evaluator=evaluator,
                            worlds=self._spin_value(self._analysis_worlds),
                            max_sims=None if cap < 0 else cap,
                            auto=self._analysis_auto.isChecked())

        player = {0: None, 1: "A", 2: "B"}[self._player.currentIndex()]
        bo3 = bool(self._format.currentData())
        self._options = dict(binary=self._binary, model_path=model_path,
                             human_player=player, human_deck=human_deck,
                             model_deck=model_deck, bo3=bo3, analysis=analysis)
        # Persist the BASE opponent + individual knobs (not the composed spec) so
        # the fields autofill cleanly next session without double-appending.
        _save_launcher_config(dict(
            human_deck=human_deck, model_deck=model_deck, opponent=opponent,
            player=player or "", bo3=bo3,
            sims=self._spin_value(self._sims), worlds=self._spin_value(self._worlds),
            think_time=self._spin_value(self._think_time),
            search_procs=self._spin_value(self._search_procs),
            match_clock=self._spin_value(self._match_clock),
            paced=self._paced.currentData(),
            analysis_enabled=self._analysis_enable.isChecked(),
            analysis_evaluator=self._combo_spec(self._analysis_eval),
            analysis_worlds=self._spin_value(self._analysis_worlds),
            analysis_cap=None if self._analysis_cap.value() < 0
            else self._analysis_cap.value(),
            analysis_auto=self._analysis_auto.isChecked()))
        self.accept()

    def options(self):
        return self._options


def _resolve_opponent_spec(spec):
    """Turn the launcher's opponent spec into the model_path build_session wants.

    Scripted / search / play specs pass straight through (build_session +
    make_controller understand them). A model spec ('gen' or an explicit path) is
    resolved and existence-checked here so a missing generalist checkpoint is
    reported cleanly instead of crashing later in _load_model."""
    from opponents import is_scripted_spec, resolve_checkpoint
    s = spec.strip()
    low = s.lower()
    if is_scripted_spec(s) or low.startswith(("az:", "azraw:", "mcts:",
                                              "play:", "actions:", "human",
                                              "auto")):
        return s
    path = resolve_checkpoint(s)             # 'gen' -> newest gen snapshot path
    if not path or not os.path.exists(path):
        raise ValueError(
            f"No checkpoint found for opponent {s!r}. Train the generalist first "
            f"(train/train.py train --deck <deck> --opponent <opp>), pick a "
            f"scripted opponent, or type an explicit .zip path.")
    return path


# ── Entry point (called by play.py --gui) ─────────────────────────────────────

def _ensure_app():
    """The styled shared QApplication (created once)."""
    app = QApplication.instance() or QApplication([])
    font = QFont("JetBrains Mono")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setFamilies(["JetBrains Mono", "DejaVu Sans Mono", "monospace"])
    app.setFont(font)
    app.setStyleSheet(_QSS)
    return app


def _analysis_cfg_from(opts):
    """An AnalysisConfig from launcher/CLI analysis options, or None (disabled).

    `opts` is the launcher's analysis dict ({} = enabled with defaults, None =
    off). ``ROBOMAGE_ANALYSIS_SMOKE`` overrides everything with a small
    torch-free config so the headless smoke can exercise the pipeline."""
    from analysis_session import AnalysisConfig

    if os.environ.get("ROBOMAGE_ANALYSIS_SMOKE"):
        return AnalysisConfig(evaluator_spec="uniform", worlds=2,
                              chunk_sims=8, max_sims=32, auto_analyze=True)
    if opts is None:
        return None
    cfg = AnalysisConfig()
    if opts.get("evaluator"):
        cfg.evaluator_spec = opts["evaluator"]
    if opts.get("worlds"):
        cfg.worlds = int(opts["worlds"])
    if opts.get("max_sims") is not None:
        cfg.max_sims = int(opts["max_sims"])
    cfg.auto_analyze = bool(opts.get("auto", True))
    return cfg


def _run_session(session):
    """Show the board for an assembled session and run the Qt loop to completion."""
    window = GameWindow(session)
    try:
        window.show()
        window.start()
        _ensure_app().exec()
        if (os.environ.get("ROBOMAGE_ANALYSIS_SMOKE")
                and not (window._analysis is not None
                         and window._analysis.smoke_ok)):
            print("ANALYSIS SMOKE FAILED: no analysis stats arrived",
                  file=sys.stderr)
            return 1
    finally:
        # The front end owns closing the env (the driver leaves it open).
        session.env.close()
    return 0


def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver", bo3=True, analysis=False):
    """Launch the Qt game board. Same signature/semantics as tui_game.run:
    `model_path` of None/"scripted" ⇒ rule-based opponent; any
    opponents.make_controller spec works; `bo3` (default True) plays a
    best-of-three match with sideboarding in one engine process. `analysis`
    opens the analysis window (default config)."""
    _ensure_app()
    analysis_cfg = _analysis_cfg_from({} if analysis else None)
    session = build_session(binary_path, model_path, human_player=human_player,
                            human_deck=human_deck, model_deck=model_deck, bo3=bo3,
                            analysis=analysis_cfg is not None, step_pacing=True)
    session.analysis_cfg = analysis_cfg
    return _run_session(session)


def run_launcher(binary_path=None):
    """Show the intro screen, then launch the game the human configured. Returns 0
    on a clean exit (including Cancel). This is the no-arguments entry point."""
    from cli_spec import BINARY
    binary_path = binary_path or BINARY
    _ensure_app()
    dialog = LauncherDialog(binary_path)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0
    opts = dialog.options()
    analysis_cfg = _analysis_cfg_from(opts.get("analysis"))
    try:
        session = build_session(
            opts["binary"], opts["model_path"], human_player=opts["human_player"],
            human_deck=opts["human_deck"], model_deck=opts["model_deck"],
            bo3=opts["bo3"], analysis=analysis_cfg is not None, step_pacing=True)
        session.analysis_cfg = analysis_cfg
    except Exception as exc:                              # noqa: BLE001
        QMessageBox.critical(None, "Could not start game", str(exc))
        return 1
    return _run_session(session)


if __name__ == "__main__":
    # Run with no game arguments -> the intro/launcher screen. (play.py --gui is
    # the flagged entry point that skips the launcher and starts a game directly.)
    sys.exit(run_launcher())
