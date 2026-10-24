"""Trajectory episodes (docs/trajectory_format.md) read with the standard library only.

An episode is an RMTB file (``rmtb.py``) of kind ``episode``: one game of a match with the
sideboarding phase before it, every decision of the recorded seats in decision order. ``read``
returns an ``Episode``: the header and footer fields (``attrs``) and one ``Step`` per decision with
its tensors expanded to the featurizer's layout (``docs/featurizer.md`` §3: ``tok_num`` row-major
``[N, F]`` floats from the stored f16 values, ``tok_cat`` ``[N, C]``, ``tok_card`` ``[N, K]``,
``tok_ability`` ``[N, Q]``, ``tok_mask`` all ones, ``rel`` ``[R, 3]`` flat, ``act_tok`` the menu-type
tokens in order, ``act_mask`` all ones)::

    ep = rmtraj.read("trajectories/s123.g1.rmtraj")
    ep.attrs["result_a"], ep.attrs["end_reason"]
    for step in ep.steps:
        step.seat, step.game, step.decision, step.sideboarding, step.chosen
        step.mu                      # None when the deciding agent had no behaviour distribution
        step.tensors["tok_num"]      # flat list, N * F
        step.policies                # [(id, version, value, log_pi over the menu rows)]
        step.omniscient, step.targets  # None unless the episode is omniscient

Malformed input raises ``ValueError``. The C++ writer and reader are ``src/featurize/episode.{h,cpp}``;
the learner reads the same arrays with numpy (``rmtb.py``'s docstring).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rmtb  # noqa: E402

KIND = "episode"
VERSION = 3
VIEWER_OMNISCIENT = 2
TENSORS = ("tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "act_tok", "act_mask")


class Step:
    """One recorded decision."""

    def __init__(self):
        self.seat = self.game = self.decision = 0
        self.sideboarding = False
        self.chosen = 0
        self.mu = None
        self.policy_version = -1
        self.teacher = -1
        self.policies = []
        self.tensors = None
        self.omniscient = None
        self.targets = None


class Episode:
    """One episode: ``meta`` (the RMTB meta), ``attrs`` (header and footer), ``steps``."""

    def __init__(self, meta, steps):
        self.meta, self.attrs, self.steps = meta, meta.get("attrs", {}), steps


def _array(arrays, name):
    a = arrays.get(name)
    if a is None:
        raise ValueError("episode: no array %s" % name)
    return a.values


def _offsets(values, steps, total, name):
    if len(values) != steps + 1 or values[0] != 0 or values[-1] != total or \
            any(values[i + 1] < values[i] for i in range(steps)):
        raise ValueError("episode: %s does not tile its array" % name)
    return values


class _Tokens:
    """One featurization's compact arrays (``prefix`` "" for the seat's, "omni_" for the omniscient)."""

    def __init__(self, arrays, prefix, steps):
        g = lambda n: _array(arrays, prefix + n)  # noqa: E731
        self.tok_type, self.cat, self.card, self.ability = g("tok_type"), g("tok_cat"), g("tok_card"), g("tok_ability")
        self.num_index, self.num_value, self.rel = g("tok_num_index"), g("tok_num_value"), g("rel")
        self.tok_off = _offsets(g("tok_offsets"), steps, len(self.tok_type), prefix + "tok_offsets")
        self.rel_off = _offsets(g("rel_offsets"), steps, len(self.rel) // 3, prefix + "rel_offsets")
        self.num_off = _offsets(g("tok_num_offsets"), steps, len(self.num_index), prefix + "tok_num_offsets")
        self.pos = [0, 0, 0]  # the next categorical, card and ability value

    def expand(self, d, widths, dims, menu_type):
        F, C, K, Q = dims
        types = self.tok_type[self.tok_off[d]:self.tok_off[d + 1]]
        n = len(types)
        num, cat, card, ab = [0.0] * (n * F), [0] * (n * C), [-1] * (n * K), [-1] * (n * Q)
        for k in range(self.num_off[d], self.num_off[d + 1]):
            i = self.num_index[k]
            if not 0 <= i < n * F:
                raise ValueError("episode: a numeric position past its decision's tokens")
            num[i] = self.num_value[k]
        act = []
        for i, ty in enumerate(types):
            if not 0 <= ty < len(widths):
                raise ValueError("episode: a token type out of range")
            _, wc, wk, wq = widths[ty]
            for j, (src, dst, w, width) in enumerate(((self.cat, cat, wc, C), (self.card, card, wk, K),
                                                      (self.ability, ab, wq, Q))):
                at = self.pos[j]
                if at + w > len(src):
                    raise ValueError("episode: the per-type arrays end before their tokens' widths")
                dst[i * width:i * width + w] = src[at:at + w]
                self.pos[j] = at + w
            if ty == menu_type:
                act.append(i)
        rel = self.rel[3 * self.rel_off[d]:3 * self.rel_off[d + 1]]
        if any(not 0 <= rel[i] < n or not 0 <= rel[i + 1] < n for i in range(0, len(rel), 3)):
            raise ValueError("episode: a relation names no token of its decision")
        return {"tok_type": list(types), "tok_num": num, "tok_cat": cat, "tok_card": card, "tok_ability": ab,
                "tok_mask": [1] * n, "rel": list(rel), "act_tok": act, "act_mask": [1] * len(act)}

    def check_consumed(self):
        if self.pos != [len(self.cat), len(self.card), len(self.ability)]:
            raise ValueError("episode: the per-type arrays hold more values than their tokens' widths")


def read_bytes(data: bytes) -> Episode:
    """The ``Episode`` of an episode file's bytes."""
    meta, arrays = rmtb.read_bytes(data)
    attrs = meta.get("attrs", {})
    if meta.get("kind") != KIND:
        raise ValueError("not an episode file (kind %r)" % meta.get("kind"))
    if attrs.get("episode_version") != VERSION:
        raise ValueError("episode version %r, expected %d" % (attrs.get("episode_version"), VERSION))
    dims = tuple(attrs[k] for k in ("F", "C", "K", "Q"))
    menu_type, omniscient = attrs["menu_type"], bool(attrs["omniscient"])
    widths = arrays["type_widths"].rows()
    seat = _array(arrays, "seat")
    steps = len(seat)
    per = {n: _array(arrays, n) for n in ("game", "decision", "sideboarding", "chosen", "has_mu",
                                          "policy_version", "teacher")}
    if any(len(v) != steps for v in per.values()):
        raise ValueError("episode: the per-decision arrays differ in length")
    mu = _array(arrays, "mu")
    menu_off = _offsets(_array(arrays, "menu_offsets"), steps, len(mu), "menu_offsets")
    seat_tokens = _Tokens(arrays, "", steps)
    omni_tokens = _Tokens(arrays, "omni_", steps) if omniscient else None
    p = arrays["eval_policy"].shape[1] if "eval_policy" in arrays else 0
    if p:
        ev_id, ev_ver, ev_val, ev_lp = (_array(arrays, n) for n in ("eval_policy", "eval_version", "eval_value",
                                                                    "eval_log_pi"))
    if omniscient:
        lists = {}
        for n in ("opp_hand", "own_library", "opp_library"):
            lists[n] = (_array(arrays, n), _offsets(_array(arrays, n + "_offsets"), steps, len(arrays[n].values),
                                                     n + "_offsets"))
    out = []
    for d in range(steps):
        s = Step()
        s.seat, s.game, s.decision = seat[d], per["game"][d], per["decision"][d]
        s.sideboarding, s.chosen = bool(per["sideboarding"][d]), per["chosen"][d]
        s.policy_version, s.teacher = per["policy_version"][d], per["teacher"][d]
        mb, me = menu_off[d], menu_off[d + 1]
        if per["has_mu"][d]:
            s.mu = mu[mb:me]
        s.tensors = seat_tokens.expand(d, widths, dims, menu_type)
        if len(s.tensors["act_tok"]) != me - mb:
            raise ValueError("episode: decision %d's menu tokens are not its menu_offsets rows" % d)
        for k in range(p):
            if ev_id[d * p + k] >= 0:
                s.policies.append((ev_id[d * p + k], ev_ver[d * p + k], ev_val[d * p + k],
                                   [ev_lp[r * p + k] for r in range(mb, me)]))
        if omniscient:
            s.omniscient = omni_tokens.expand(d, widths, dims, menu_type)
            s.targets = {}
            for n, (values, off) in lists.items():
                s.targets[n] = values[off[d]:off[d + 1]]
        out.append(s)
    seat_tokens.check_consumed()
    if omni_tokens:
        omni_tokens.check_consumed()
    return Episode(meta, out)


def read(path: str) -> Episode:
    """The ``Episode`` of the episode file at ``path``."""
    with open(path, "rb") as fp:
        return read_bytes(fp.read())
