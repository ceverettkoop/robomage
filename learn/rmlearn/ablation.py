"""Ablations of a network's card inputs over recorded decisions: what the policy and the value lose
when a card's identity, or a group of card properties, is taken away (learn/ablate.py is the CLI).

    tables = spec.load_tables("runs/r1/tables")
    net = load_network("runs/r1/ckpt/fixed/m9.pt", tables)
    sample = Sample("runs/r1/episodes/selfplay", episodes=2000, per_episode=8, seed=1)
    res = identity_ablation(net, tables, sample)            # Result: one row per (decision, card)
    res = property_ablation(net, tables, sample)            # Result: one row per (decision, group)
    rows = res.table(by=("key",))                           # per card / group: means and standard errors
    res.save("out/dir"); res = Result.load("out/dir")       # result.npz + meta.json

A token's card part is, per card slot, the card's row: ``card_proj`` of its card-table row +
``text_proj`` of its oracle-text vector + its learned id row (model.CardRows). Both ablations
replace part of that row and run the unchanged network on the same recorded decisions:

**Identity** — a card's id row replaced by ``unknown_id``, in every slot of the decision that holds
the card. This is the input id-dropout trains (and the one an unseen card always reads), so the
network is on its training distribution; what moves is what it knows about that card beyond its
properties. Per decision: every id at once (key -1), and up to ``per_decision`` single cards.

**Properties** — a group of table columns (mana cost, types, keywords, the text vector, an ability's
category, …; ``default_groups``) replaced for every card: by the group's mean over the cards the
network has seen (``mode="mean"``), or by another seen card's values (``"resample"``: real values, the
wrong card's). The group ``id`` is the identity ablation of every card, for scale.

Each row compares the ablated network with the unablated one on one decision: ``tv`` (total
variation between the two policies, the share of probability moved), ``kl``, ``flip`` (the ablated
argmax is a row the unablated network ranked below its own), ``dlogp`` (the change in log π of the unablated argmax), ``dv`` (value change) and ``dse``
(the change in squared error against the game's result: positive when the ablated input carried
information about the outcome). Standard errors are clustered by episode.

torch is imported only by the functions that run a network, so results load, tabulate and plot
(``ablation_plot``) under any Python with numpy. On the GPU this must be the only process using it.
"""

import csv
import glob
import json
import os
import re
import time
import zlib

import numpy as np

from . import episodes as EP

ALL = -1                       # an identity row's key for "every card id at once"
METRICS = ("tv", "kl", "flip", "dlogp", "adv", "dv", "dse")
TIE = 1e-3                     # log π this close to the argmax's is the same choice (interchangeable menu rows)
DECISION_COLUMNS = ("episode", "step", "seat", "deck", "opp_deck", "game", "sideboarding", "menu", "tokens",
                    "value", "outcome", "entropy")
TABLES = ("card", "text", "ability", "part")

# The default property groups: name → per table the column-name prefixes (the name up to its first
# '=', ':' or '#') it takes. A table's named columns no group lists form `<table>.other`.
DEFAULT_GROUPS = (
    ("cost", {"card": ("mana_value", "cost_w", "cost_u", "cost_b", "cost_r", "cost_g", "cost_c", "cost_generic",
                       "cost_x", "card_pip.kind")}),
    ("colors", {"card": ("colors",)}),
    ("types", {"card": ("types", "supertypes")}),
    ("subtypes", {"card": ("card_subtype.subtype",)}),
    ("pt", {"card": ("power", "toughness", "loyalty")}),
    ("keywords", {"card": ("card_keyword.kw",)}),
    ("layout", {"card": ("face", "token", "layout", "n_abilities")}),
    ("ability.kind", {"ability": ("face", "ability_kind", "tap", "timing", "limit", "keyed", "n_parts",
                                  "charm_num")}),
    ("ability.category", {"ability": ("category",)}),
    ("ability.amount", {"ability": ("amount", "amount_kind", "power_mod", "toughness_mod", "pt_kind")}),
    ("ability.targets", {"ability": ("targets", "target_min", "target_max")}),
)
NOT_OTHER = {"card": ("present",)}      # named columns left out of `<table>.other` (constant over real cards)


# ── the network ──────────────────────────────────────────────────────────────────────────────

def load_network(path, tables):
    """The RMNet of a checkpoint over `tables`: a ``model.save_checkpoint`` file (a fixed point
    ``m<k>.pt``, ``final-model.pt``) or a learner checkpoint (``rnad-*.pt``: its online network)."""
    import torch
    from . import model as M
    from . import spec as S
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "learner" in data:
        cfg, hashes, sd = data["learner"]["model_config"], data["learner"]["hashes"], data["learner"]["model"]
    else:
        cfg, hashes, sd = data["config"], data["hashes"], data["state_dict"]
    S.require_compatible(hashes, tables, path)
    net = M.RMNet(M.ModelConfig.from_dict(cfg), tables)
    net.load_state_dict(sd)
    return net


def seen_cards(path):
    """The card ids a checkpoint's network has trained on (its ``cards.card_seen``), without building it."""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    sd = data["learner"]["model"] if "learner" in data else data["state_dict"]
    return np.nonzero(sd["cards.card_seen"].numpy()[:-1])[0]


class _Net:
    """A network in eval mode on a device, evaluated with a substituted card part."""

    def __init__(self, net, device):
        import torch
        self.torch = torch
        self.net = net.to(device).eval()
        self.device = device
        self.cards = net.cards
        self.n_cards = net.cards.n_cards
        self.checked = False

    def to_device(self, x):
        t = self.torch
        return {k: t.from_numpy(x[k]).to(self.device) for k in EP.INPUTS}

    def token_table(self, **subst):
        """``CardRows.token_table`` with the unknown-id rows, built over substituted frozen tables
        (`card_table`, `text_table`, `ability_table`, `part_table`: tensors of the buffers' shapes)."""
        c = self.cards
        saved = {k: getattr(c, k) for k in subst}
        try:
            for k, v in subst.items():
                setattr(c, k, v)
            return c.token_table(dropout_rows=True)
        finally:
            for k, v in saved.items():
                setattr(c, k, v)

    def forward(self, x, table, drop=None):
        """(log π [B, A], value [B]) of the decisions `x` over the token table `table`; `drop`
        [B, N, K] bool marks the card slots that read the unknown-id row."""
        from . import model as M
        net = self.net
        idx = x["tok_card"] if drop is None else x["tok_card"] + drop.to(x["tok_card"].dtype) * self.n_cards
        extra = M.gather_tokens(idx, x["tok_card"], x["tok_ability"], table, 2 * self.n_cards,
                                self.cards.n_abilities)
        h = net.encoder(x["tok_type"], x["tok_num"], x["tok_cat"], x["tok_mask"], x["rel"], x["rel_mask"], extra)
        g = net.summary(h, x["tok_mask"])
        logits = net.pointer(h, g, x["act_tok"], x["act_mask"])
        return M.masked_log_softmax(logits.float(), x["act_mask"]), net.value(g).float()

    def check(self, x, table):
        """Once: this path with nothing substituted is the network's own forward."""
        if self.checked:
            return
        from . import model as M
        out = self.net(*[x[k] for k in EP.INPUTS], aux=False)
        logp, value = self.forward(x, table)
        ref = M.masked_log_softmax(out["logits"].float(), x["act_mask"])
        m = x["act_mask"]
        err = max(float((logp - ref)[m].abs().max()), float((value - out["value"].float()).abs().max()))
        if not err < 1e-3:
            raise RuntimeError("ablation: the substituted forward differs from the network's by %g" % err)
        self.checked = True


def _effects(torch, base, abl, mask, z):
    """The METRICS of ablated (log π, value) against the base's, per decision, as numpy."""
    logp0, v0 = base
    logp1, v1 = abl
    p0, p1 = logp0.exp(), logp1.exp()
    fin = lambda a: a.masked_fill(~mask, 0.0)  # noqa: E731
    top = logp0.argmax(1, keepdim=True)
    dv = v1 - v0
    out = {"tv": 0.5 * (p0 - p1).abs().sum(1), "kl": (p0 * fin(logp0 - logp1)).sum(1),
           "flip": (logp0.gather(1, logp1.argmax(1, keepdim=True)) < logp0.gather(1, top) - TIE).squeeze(1).float(),
           "dlogp": (logp1.gather(1, top) - logp0.gather(1, top)).squeeze(1),
           "adv": dv.abs(), "dv": dv, "dse": (v1 - z) ** 2 - (v0 - z) ** 2}
    return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


# ── the decisions ────────────────────────────────────────────────────────────────────────────

class Sample:
    """A seeded sample of a recorded window's decisions: `episodes` episode files of `directory`
    (0: all), `per_episode` decisions of each (0: all). The same arguments give the same decisions,
    so two networks' results line up row for row.

    `decks`: only decisions whose deciding seat plays one of these decks (names as the episodes'
    ``deck_a``, e.g. ``league/elves``). `cards`: only decisions holding one of these card ids.
    `sideboarding`: None all, True only sideboarding decisions, False none of them."""

    def __init__(self, directory, episodes=2000, per_episode=8, seed=1, decks=None, cards=None, sideboarding=None,
                 batch=128):
        self.directory = directory
        self.files = sorted(glob.glob(os.path.join(directory, "*.rmtraj")))
        if not self.files:
            raise ValueError("no *.rmtraj episodes in %s" % directory)
        self.episodes, self.per_episode, self.seed, self.batch = episodes, per_episode, seed, batch
        self.decks = set(decks) if decks else None
        self.cards = np.asarray(sorted(cards), np.int64) if cards is not None and len(cards) else None
        self.sideboarding = sideboarding

    def describe(self):
        return {"directory": os.path.abspath(self.directory), "episodes": self.episodes,
                "per_episode": self.per_episode, "seed": self.seed, "decks": sorted(self.decks or []),
                "cards": [] if self.cards is None else self.cards.tolist(), "sideboarding": self.sideboarding,
                "window_files": len(self.files)}

    def _steps(self, ep):
        ok = np.ones(ep.steps, np.bool_)
        if self.decks is not None:
            seat_deck = np.array([ep.attrs.get("deck_a") in self.decks, ep.attrs.get("deck_b") in self.decks])
            ok &= seat_deck[np.asarray(ep.seat, np.int64)]
        if self.sideboarding is not None:
            ok &= np.asarray(ep.sideboarding, np.bool_) == self.sideboarding
        if self.cards is not None:
            f = ep.seat_feat
            step_of = np.repeat(np.arange(ep.steps), np.diff(f.card_off))
            holds = np.zeros(ep.steps, np.bool_)
            holds[step_of[np.isin(np.asarray(f.card, np.int64), self.cards)]] = True
            ok &= holds
        steps = np.nonzero(ok)[0]
        if self.per_episode and len(steps) > self.per_episode:
            rng = np.random.default_rng([self.seed, zlib.crc32(ep.name.encode())])
            steps = np.sort(rng.choice(steps, self.per_episode, replace=False))
        return steps

    def batches(self, spec_hashes=None):
        """Yields lists of (Episode, step), `batch` decisions each, over the sampled episodes."""
        order = np.random.default_rng(self.seed).permutation(len(self.files))
        items, taken = [], 0
        for i in order:
            path = self.files[i]
            if self.decks is not None:
                a = EP.episode_attrs(path)
                if a.get("deck_a") not in self.decks and a.get("deck_b") not in self.decks:
                    continue
            ep = EP.read_episode(path)
            if spec_hashes is not None and ep.spec_hash not in spec_hashes:
                continue
            steps = self._steps(ep)
            if not len(steps):
                continue
            items.extend((ep, int(d)) for d in steps)
            taken += 1
            while len(items) >= self.batch:
                yield items[:self.batch]
                items = items[self.batch:]
            if self.episodes and taken >= self.episodes:
                break
        if items:
            yield items


class _Recorder:
    """Accumulates a run's decision and sample rows."""

    def __init__(self):
        self.episodes, self.ep_index = [], {}
        self.decks, self.deck_index = [], {}
        self.dec = {k: [] for k in DECISION_COLUMNS}
        self.sam = {}
        self.n_dec = 0

    def _deck(self, name):
        if name not in self.deck_index:
            self.deck_index[name] = len(self.decks)
            self.decks.append(name)
        return self.deck_index[name]

    def decisions(self, items, x, base):
        """Records a batch's decisions; returns their row numbers [B]."""
        logp, value = base
        b = len(items)
        cols = {k: np.zeros(b, np.int64) for k in ("episode", "step", "seat", "deck", "opp_deck", "game",
                                                   "sideboarding")}
        z = np.zeros(b, np.float32)
        for j, (ep, d) in enumerate(items):
            if ep.name not in self.ep_index:
                self.ep_index[ep.name] = len(self.episodes)
                self.episodes.append(ep.name)
            seat = int(ep.seat[d])
            mine, theirs = ("deck_a", "deck_b") if seat == 0 else ("deck_b", "deck_a")
            cols["episode"][j], cols["step"][j], cols["seat"][j] = self.ep_index[ep.name], d, seat
            cols["deck"][j] = self._deck(str(ep.attrs.get(mine)))
            cols["opp_deck"][j] = self._deck(str(ep.attrs.get(theirs)))
            cols["game"][j] = int(ep.attrs.get("game", 0))
            cols["sideboarding"][j] = int(ep.sideboarding[d])
            z[j] = ep.result[seat]
        cols["menu"], cols["tokens"] = x["sizes"]["a"], x["sizes"]["n"]
        p = logp.exp()
        cols["value"], cols["outcome"] = value.cpu().numpy(), z
        cols["entropy"] = -(p * logp.masked_fill(p == 0, 0.0)).sum(1).cpu().numpy()
        for k in DECISION_COLUMNS:
            self.dec[k].append(np.asarray(cols[k]))
        rows = np.arange(self.n_dec, self.n_dec + b)
        self.n_dec += b
        return rows, z

    def samples(self, **cols):
        for k, v in cols.items():
            self.sam.setdefault(k, []).append(np.asarray(v))

    def result(self, kind, meta):
        cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}  # noqa: E731
        meta = dict(meta, kind=kind, episodes=self.episodes, decks=self.decks)
        return Result(kind, meta, cat(self.dec), cat(self.sam))


class _Progress:
    def __init__(self, what, every=10.0, quiet=False):
        self.what, self.every, self.quiet = what, every, quiet
        self.t0 = self.last = time.time()

    def __call__(self, decisions, rows, final=False):
        now = time.time()
        if self.quiet or (not final and now - self.last < self.every):
            return
        self.last = now
        print("%s: %d decisions, %d rows, %.0f s" % (self.what, decisions, rows, now - self.t0), flush=True)


def _card_meta(net, tables, since=None):
    """Per card id: its name, whether the network has seen it, its id row's distance from the
    unknown-card vector and, with `since` (an earlier checkpoint), whether that network had not."""
    c = net.cards
    seen = c.card_seen.cpu().numpy()[:-1]
    drift = (c.card_id.weight.detach() - c.unknown_id.detach()).norm(dim=1).cpu().numpy()[:-1]
    names = {int(i): n for i, n in tables.hashes.get("vocab", [])}
    names.setdefault(c.n_cards - 2, "(unregistered token)")     # TOKEN_SENTINEL, src/card_vocab.h
    out = {"names": names, "seen": np.nonzero(seen)[0].tolist(),
           "drift": {int(i): float(drift[i]) for i in np.nonzero(seen)[0]}}
    if since is not None:
        out["since"] = os.path.abspath(since)
        out["new"] = sorted(set(out["seen"]) - set(seen_cards(since).tolist()))
    return out


# ── identity ablation ────────────────────────────────────────────────────────────────────────

def identity_ablation(net, tables, sample, per_decision=8, cards=None, balance=True, device="cpu", forward_batch=256,
                      since=None, quiet=False):
    """Result of kind ``identity``: per sampled decision one row with every card id replaced by the
    unknown-card vector (key ``ALL``), and one per chosen card (key its id) with that card's alone.

    Up to `per_decision` of the decision's distinct seen cards are chosen (0: all of them; an unseen
    card already reads the unknown vector), restricted to the ids `cards` when given; with `balance`
    the cards with the fewest rows so far first, so rare cards are covered, else at random. Row
    columns: ``slots`` (the card slots replaced) and ``in_menu`` (the card is on a menu row: the
    decision is about it). `since`: an earlier checkpoint, to mark the cards first seen after it."""
    import torch
    m = _Net(net, device)
    rec = _Recorder()
    prog = _Progress("identity", quiet=quiet)
    rng = np.random.default_rng([sample.seed, 7])
    seen = net.cards.card_seen.cpu().numpy()
    allowed = seen.copy()
    if cards is not None:
        allowed &= np.isin(np.arange(len(seen)), np.asarray(list(cards), np.int64))
    counts = np.zeros(len(seen), np.int64)
    menu_type = tables.spec.menu_type
    rows_done = 0
    with torch.no_grad():
        table = m.token_table()
        for items in sample.batches(tables.accepted_spec_hashes()):
            xn = EP.expand(items)
            x = m.to_device(xn)
            m.check(x, table)
            base = m.forward(x, table)
            dec, z = rec.decisions(items, xn, base)
            zt = torch.from_numpy(z).to(device)
            card = xn["tok_card"]
            present = card >= 0
            eff = _effects(torch, base, m.forward(x, table, x["tok_card"] >= 0), x["act_mask"], zt)
            rec.samples(decision=dec, key=np.full(len(dec), ALL), slots=present.sum((1, 2)),
                        in_menu=np.zeros(len(dec), np.int64), **eff)
            # The single cards: (decision, card) pairs, evaluated `forward_batch` at a time.
            pb, pc = [], []
            for j in range(len(items)):
                ids = np.unique(card[j][present[j]])
                ids = ids[allowed[ids]]
                if per_decision and len(ids) > per_decision:
                    tie = rng.random(len(ids))
                    ids = ids[np.lexsort((tie, counts[ids] if balance else tie))[:per_decision]]
                counts[ids] += 1
                pb.extend([j] * len(ids))
                pc.extend(ids.tolist())
            pb, pc = np.asarray(pb, np.int64), np.asarray(pc, np.int64)
            for s in range(0, len(pb), forward_batch):
                jb = torch.from_numpy(pb[s:s + forward_batch]).to(device)
                jc = torch.from_numpy(pc[s:s + forward_batch]).to(device)
                xs = {k: v[jb] for k, v in x.items()}
                drop = xs["tok_card"] == jc.view(-1, 1, 1)
                eff = _effects(torch, (base[0][jb], base[1][jb]), m.forward(xs, table, drop), xs["act_mask"], zt[jb])
                on_menu = (drop.any(2) & (xs["tok_type"] == menu_type)).any(1)
                rec.samples(decision=dec[pb[s:s + forward_batch]], key=pc[s:s + forward_batch],
                            slots=drop.sum((1, 2)).cpu().numpy(), in_menu=on_menu.cpu().numpy().astype(np.int64),
                            **eff)
            rows_done += len(dec) + len(pb)
            prog(rec.n_dec, rows_done)
    prog(rec.n_dec, rows_done, final=True)
    meta = {"sample": sample.describe(), "per_decision": per_decision, "balance": bool(balance),
            "cards_filter": sorted(int(c) for c in cards) if cards is not None else [],
            "card": _card_meta(net, tables, since), "spec_hash": tables.spec.hash}
    return rec.result("identity", meta)


# ── property groups ──────────────────────────────────────────────────────────────────────────

class Group:
    """A property group: per table (``card``, ``text``, ``ability``, ``part``) the columns replaced,
    and whether the card ids are replaced too (``ids``)."""

    def __init__(self, name, cols=None, ids=False):
        self.name = name
        self.cols = {t: np.asarray(c, np.int64) for t, c in (cols or {}).items() if len(c)}
        self.ids = ids

    def describe(self):
        return {"name": self.name, "ids": self.ids, "columns": {t: int(len(c)) for t, c in self.cols.items()}}


def column_names(tables):
    """Per table its column names (the text table's are its dimensions' numbers)."""
    doc = tables.spec.doc["card_tables"]
    return {"card": list(doc["card"]["columns"]), "text": ["text%d" % i for i in range(tables.text.shape[1])],
            "ability": list(doc["ability"]["columns"]), "part": list(doc["part"]["columns"])}


def _prefix(name):
    return re.split(r"[=:#]", name, maxsplit=1)[0]


def default_groups(tables):
    """The default groups ({name: Group}, in order): DEFAULT_GROUPS over the spec's columns, each
    table's unlisted named columns as ``<table>.other``, ``text`` (the oracle-text vector), ``parts``
    (the part table), ``id`` (every card id) and the composites ``card.all`` (the structured card
    columns), ``card+text`` (everything about a card but its id), ``ability.all`` (ability and part
    tables) and ``all`` (every card input)."""
    names = column_names(tables)
    prefixes = {t: np.array([_prefix(n) for n in names[t]]) for t in TABLES}
    groups = {}
    used = {t: set(NOT_OTHER.get(t, ())) | {""} for t in TABLES}
    for gname, spec in DEFAULT_GROUPS:
        groups[gname] = Group(gname, {t: np.nonzero(np.isin(prefixes[t], p))[0] for t, p in spec.items()})
        for t, p in spec.items():
            used[t].update(p)
    for t in ("card", "ability"):
        rest = np.nonzero(~np.isin(prefixes[t], sorted(used[t])))[0]
        if len(rest):
            groups[t + ".other"] = Group(t + ".other", {t: rest})
    whole = {t: np.arange(len(names[t])) for t in TABLES}
    groups["text"] = Group("text", {"text": whole["text"]})
    groups["parts"] = Group("parts", {"part": whole["part"]})
    groups["id"] = Group("id", ids=True)
    groups["card.all"] = Group("card.all", {"card": whole["card"]})
    groups["card+text"] = Group("card+text", {"card": whole["card"], "text": whole["text"]})
    groups["ability.all"] = Group("ability.all", {"ability": whole["ability"], "part": whole["part"]})
    groups["all"] = Group("all", whole, ids=True)
    return groups


def custom_group(tables, name, spec):
    """A group from ``TABLE:REGEX[+TABLE:REGEX…]`` (the regex searched in the table's column names)."""
    names = column_names(tables)
    cols = {}
    for part in spec.split("+"):
        table, _, pattern = part.partition(":")
        if table not in names:
            raise ValueError("group %s: no table %r (one of %s)" % (name, table, ", ".join(TABLES)))
        hit = np.array([i for i, n in enumerate(names[table]) if re.search(pattern, n)], np.int64)
        if not len(hit):
            raise ValueError("group %s: no %s column matches %r" % (name, table, pattern))
        cols[table] = np.union1d(cols.get(table, np.zeros(0, np.int64)), hit)
    return Group(name, cols)


def split_groups(tables, spec, net):
    """One group per column of ``TABLE:REGEX`` that varies over the cards the network has seen
    (named ``TABLE:<column>``): e.g. ``card:^card_keyword`` gives every keyword its own group."""
    table, _, pattern = spec.partition(":")
    names = column_names(tables)
    if table not in names:
        raise ValueError("split %s: no table %r" % (spec, table))
    ref = _Rows(net, tables, None).ref[table]
    data = getattr(tables, table)[ref]
    out = []
    for i, n in enumerate(names[table]):
        if re.search(pattern, n) and data[:, i].min() != data[:, i].max():
            out.append(Group("%s:%s" % (table, n), {table: [i]}))
    if not out:
        raise ValueError("split %s: no varying column matches" % spec)
    return out


class _Rows:
    """Per table the reference rows (the seen cards': what a replaced value is drawn from) and the
    rows replaced (every real card's, or the cards `only`'s)."""

    def __init__(self, net, tables, only):
        doc = tables.spec.doc["card_tables"]["ability"]
        self.per_card, self.key_base = int(doc["abilities_per_card"]), int(doc["key_base"])
        n_real = net.cards.n_cards - 1
        seen = np.nonzero(net.cards.card_seen.cpu().numpy()[:-1])[0]
        if not len(seen):
            raise ValueError("the network has seen no card: no reference rows")
        vocab = np.array(sorted(int(i) for i, _ in tables.hashes.get("vocab", [])), np.int64)
        vocab = vocab[vocab < n_real] if len(vocab) else np.arange(n_real)
        self.only = None if only is None else np.asarray(sorted(only), np.int64)
        cards = vocab if self.only is None else self.only
        ab_present = (tables.ability != 0).any(1)
        ab_of = lambda ids: (ids[:, None] * self.per_card + np.arange(self.per_card)).ravel()  # noqa: E731
        keys = np.arange(self.key_base, len(ab_present))
        take = lambda rows: rows[ab_present[rows]]  # noqa: E731
        ab_ref = take(np.concatenate([ab_of(seen), keys]))
        ab_apply = take(ab_of(cards)) if self.only is not None else np.nonzero(ab_present)[0]
        parts = np.arange(len(tables.part_ability))
        self.ref = {"card": seen, "text": seen, "ability": ab_ref, "part": parts}
        self.apply = {"card": cards, "text": cards, "ability": ab_apply,
                      "part": parts[np.isin(tables.part_ability, ab_apply)]}


class _Substituter:
    """Builds a group's substituted tables on the network's device."""

    BUFFER = {"card": "card_table", "text": "text_table", "ability": "ability_table", "part": "part_table"}
    SPACE = {"card": "card", "text": "card", "ability": "ability", "part": "part"}

    def __init__(self, m, tables, only, mode, seed):
        if mode not in ("mean", "resample"):
            raise ValueError("mode %r: mean or resample" % mode)
        self.m, self.mode = m, mode
        self.rows = _Rows(m.net, tables, only)
        self.rng = np.random.default_rng([seed, 11])
        t = m.torch
        self.dev = lambda a: t.from_numpy(np.asarray(a, np.int64)).to(m.device)  # noqa: E731
        self.ref = {k: self.dev(v) for k, v in self.rows.ref.items()}
        self.apply = {k: self.dev(v) for k, v in self.rows.apply.items()}

    def tables(self, group):
        """{buffer name: tensor} for `group`; resampling draws a fresh donor per replaced row (the
        card and text tables' rows share theirs, so a card takes one other card's values)."""
        out, donors = {}, {}
        for table, cols in group.cols.items():
            src = getattr(self.m.cards, self.BUFFER[table])
            apply, ref = self.apply[table], self.ref[table]
            if not len(apply):
                continue
            cols_t = self.dev(cols)
            new = src.clone()
            if self.mode == "mean":
                fill = src[ref][:, cols_t].mean(0, keepdim=True).expand(len(apply), -1)
            else:
                space = self.SPACE[table]
                if space not in donors:
                    donors[space] = self.dev(self.rng.integers(0, len(ref), len(apply)))
                fill = src[ref[donors[space]]][:, cols_t]
            new[apply.unsqueeze(1), cols_t.unsqueeze(0)] = fill
            out[self.BUFFER[table]] = new
        return out


def property_ablation(net, tables, sample, groups=None, mode="mean", cards=None, device="cpu", forward_batch=256,
                      quiet=False):
    """Result of kind ``properties``: per sampled decision one row per group (key the group's index
    in ``meta["groups"]``) with the group's columns replaced, in every card's rows or, with `cards`
    (ids), in those cards' rows alone. `groups`: a list of Group (default ``default_groups``).
    Row column ``slots``: the decision's card and ability slots the group reaches (0: the ablation
    cannot have moved the decision)."""
    import torch
    m = _Net(net, device)
    groups = list(default_groups(tables).values()) if groups is None else list(groups)
    sub = _Substituter(m, tables, cards, mode, sample.seed)
    rec = _Recorder()
    prog = _Progress("properties", quiet=quiet)
    only = None if cards is None else torch.from_numpy(np.asarray(sorted(cards), np.int64)).to(device)
    card_side = [g.ids or "card" in g.cols or "text" in g.cols for g in groups]
    ab_side = ["ability" in g.cols or "part" in g.cols for g in groups]
    rows_done = 0
    with torch.no_grad():
        plain = m.token_table()
        for items in sample.batches(tables.accepted_spec_hashes()):
            xn = EP.expand(items)
            x = m.to_device(xn)
            m.check(x, plain)
            base = m.forward(x, plain)
            dec, z = rec.decisions(items, xn, base)
            zt = torch.from_numpy(z).to(device)
            hit = x["tok_card"] >= 0
            ab_hit = x["tok_ability"] >= 0
            if only is not None:
                hit &= torch.isin(x["tok_card"], only)
                ab_hit &= (x["tok_ability"] < sub.rows.key_base) & torch.isin(
                    torch.div(x["tok_ability"], sub.rows.per_card, rounding_mode="floor"), only)
            n_card, n_ab = hit.sum((1, 2)).cpu().numpy(), ab_hit.sum((1, 2)).cpu().numpy()
            for gi, g in enumerate(groups):
                table = m.token_table(**sub.tables(g)) if g.cols else plain
                parts = []
                for s in range(0, len(items), forward_batch):
                    xs = {k: v[s:s + forward_batch] for k, v in x.items()}
                    abl = m.forward(xs, table, hit[s:s + forward_batch] if g.ids else None)
                    parts.append(_effects(torch, (base[0][s:s + forward_batch], base[1][s:s + forward_batch]), abl,
                                          xs["act_mask"], zt[s:s + forward_batch]))
                eff = {k: np.concatenate([p[k] for p in parts]) for k in METRICS}
                rec.samples(decision=dec, key=np.full(len(dec), gi),
                            slots=n_card * card_side[gi] + n_ab * ab_side[gi], **eff)
            rows_done += len(dec) * len(groups)
            prog(rec.n_dec, rows_done)
    prog(rec.n_dec, rows_done, final=True)
    meta = {"sample": sample.describe(), "mode": mode, "groups": [g.describe() for g in groups],
            "cards_filter": sorted(int(c) for c in cards) if cards is not None else [],
            "card": _card_meta(net, tables), "spec_hash": tables.spec.hash}
    return rec.result("properties", meta)


# ── results ──────────────────────────────────────────────────────────────────────────────────

class Result:
    """An ablation's rows. ``decisions`` ({column: [D]}, DECISION_COLUMNS: ``deck`` / ``opp_deck``
    index ``meta["decks"]``, ``episode`` indexes ``meta["episodes"]``, ``value`` and ``entropy`` are
    the unablated network's, ``outcome`` the deciding seat's game result) and ``samples``
    ({column: [S]}: ``decision`` (a row of ``decisions``), ``key`` (identity: a card id or ``ALL``;
    properties: a group index), ``slots``, identity's ``in_menu``, and METRICS)."""

    def __init__(self, kind, meta, decisions, samples):
        self.kind, self.meta, self.decisions, self.samples = kind, meta, decisions, samples

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        arrays = {"d_" + k: v for k, v in self.decisions.items()}
        arrays.update({"s_" + k: v for k, v in self.samples.items()})
        np.savez_compressed(os.path.join(directory, "result.npz"), **arrays)
        with open(os.path.join(directory, "meta.json"), "w") as fp:
            json.dump(self.meta, fp, indent=1)

    @classmethod
    def load(cls, directory):
        with open(os.path.join(directory, "meta.json")) as fp:
            meta = json.load(fp)
        with np.load(os.path.join(directory, "result.npz")) as z:
            dec = {k[2:]: z[k] for k in z.files if k.startswith("d_")}
            sam = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        return cls(meta["kind"], meta, dec, sam)

    # Labels.

    def card_name(self, card_id):
        return self.meta["card"]["names"].get(str(card_id), self.meta["card"]["names"].get(card_id, "#%d" % card_id))

    def key_label(self, key):
        if self.kind == "properties":
            return self.meta["groups"][int(key)]["name"]
        return "(all ids)" if int(key) == ALL else self.card_name(int(key))

    def label(self, column, value):
        if column == "key":
            return self.key_label(value)
        if column in ("deck", "opp_deck"):
            return self.meta["decks"][int(value)]
        if column == "episode":
            return self.meta["episodes"][int(value)]
        return int(value)

    def column(self, name):
        """A column per sample row: a sample column, or its decision's."""
        if name in self.samples:
            return self.samples[name]
        return self.decisions[name][self.samples["decision"]]

    def code(self, column, value):
        """The stored integer of a label (a deck's, card's or group's name) or of an integer."""
        if isinstance(value, (int, np.integer)):
            return int(value)
        text = str(value)
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        if column in ("deck", "opp_deck"):
            return self.meta["decks"].index(text)
        if column == "key" and self.kind == "properties":
            return [g["name"] for g in self.meta["groups"]].index(text)
        if column == "key":
            return ALL if text in ("all", "(all ids)") else resolve_cards(self.meta["card"]["names"], [text])[0]
        raise ValueError("column %s takes integers" % column)

    def mask(self, where):
        """A [S] bool mask from {column: value | [values]} (labels or integers)."""
        ok = np.ones(len(self.samples["decision"]), np.bool_)
        for col, want in (where or {}).items():
            vals = want if isinstance(want, (list, tuple, set)) else [want]
            ok &= np.isin(self.column(col), [self.code(col, v) for v in vals])
        return ok

    def table(self, by=("key",), where=None, min_n=1, sort=None):
        """Rows grouped by the columns `by`: per group its labels, ``n`` (rows), ``episodes``,
        ``slots`` (mean), ``touched`` (the share of rows with a slot reached) and per metric its mean and
        ``<metric>_se`` (clustered by episode). An identity table by ``key`` alone adds the card's
        ``drift`` (its id row's distance from the unknown-card vector) and ``new`` (first seen after
        ``since``). `where` as ``mask``; `sort` a column (descending)."""
        ok = self.mask(where)
        keys = np.stack([self.column(b)[ok].astype(np.int64) for b in by], 1)
        if not len(keys):
            return []
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        g = len(uniq)
        n = np.bincount(inv, minlength=g).astype(np.float64)
        ep = self.column("episode")[ok].astype(np.int64)
        pair, pair_inv = np.unique(inv * (int(ep.max()) + 1) + ep, return_inverse=True)
        pair_group = pair // (int(ep.max()) + 1)
        stats = {"episodes": np.bincount(pair_group, minlength=g)}
        cols = {"slots": self.samples["slots"][ok].astype(np.float64),
                "touched": (self.samples["slots"][ok] > 0).astype(np.float64)}
        cols.update({k: self.samples[k][ok].astype(np.float64) for k in METRICS})
        for k, v in cols.items():
            mean = np.bincount(inv, weights=v, minlength=g) / n
            stats[k] = mean
            if k in METRICS:
                resid = np.bincount(pair_inv.reshape(-1), weights=v - mean[inv], minlength=len(pair))
                stats[k + "_se"] = np.sqrt(np.bincount(pair_group, weights=resid ** 2, minlength=g)) / n
        card = self.meta.get("card", {})
        new = set(card.get("new", []))
        rows = []
        for i in range(g):
            if n[i] < min_n:
                continue
            row = {b: self.label(b, uniq[i, j]) for j, b in enumerate(by)}
            row["n"] = int(n[i])
            row.update({k: float(v[i]) if k != "episodes" else int(v[i]) for k, v in stats.items()})
            if self.kind == "identity" and tuple(by) == ("key",) and uniq[i, 0] != ALL:
                row["card_id"] = int(uniq[i, 0])
                row["drift"] = card.get("drift", {}).get(str(uniq[i, 0]), card.get("drift", {}).get(int(uniq[i, 0])))
                if "new" in card:
                    row["new"] = int(int(uniq[i, 0]) in new)
            rows.append(row)
        if sort:
            rows.sort(key=lambda r: -(r.get(sort) or 0.0))
        return rows

    def base_summary(self):
        """The unablated network over the sampled decisions: count, value's mean squared error and
        correlation with the results, mean entropy and menu size."""
        d = self.decisions
        v, z = d["value"].astype(np.float64), d["outcome"].astype(np.float64)
        return {"decisions": int(len(v)), "episodes": int(len(self.meta["episodes"])),
                "value_mse": float(((v - z) ** 2).mean()),
                "value_corr": float(np.corrcoef(v, z)[0, 1]) if v.std() > 0 and z.std() > 0 else 0.0,
                "entropy": float(d["entropy"].mean()), "menu": float(d["menu"].mean())}


def resolve_cards(names, wanted):
    """Card ids for the names `wanted` against {id: name} (ids as int or str): an exact name
    (case-insensitive), else a unique substring, else an integer id."""
    table = {int(i): n for i, n in names.items()}
    out = []
    for w in wanted:
        w = w.strip()
        if re.fullmatch(r"\d+", w):
            out.append(int(w))
            continue
        exact = [i for i, n in table.items() if n.lower() == w.lower()]
        part = exact or [i for i, n in table.items() if w.lower() in n.lower()]
        if len(part) != 1:
            raise ValueError("card %r: %s" % (w, "no such card" if not part else "ambiguous: " + ", ".join(
                sorted(table[i] for i in part)[:8])))
        out.append(part[0])
    return out


DEFAULT_COLUMNS = ("n", "episodes", "tv", "flip", "kl", "adv", "dv", "dse")


def format_rows(rows, by, columns=DEFAULT_COLUMNS, top=None):
    """A text table of ``Result.table`` rows: the `by` labels, then `columns` (a metric as
    ``mean ± se``)."""
    rows = rows[:top] if top else rows
    head = list(by) + list(columns)
    body = []
    for r in rows:
        line = [str(r[b]) for b in by]
        for c in columns:
            v = r.get(c)
            if v is None:
                line.append("")
            elif c + "_se" in r:
                line.append("%+.4f ± %.4f" % (v, r[c + "_se"]) if c in ("dv", "dlogp", "dse") else
                            "%.4f ± %.4f" % (v, r[c + "_se"]))
            elif isinstance(v, float):
                line.append("%.3f" % v)
            else:
                line.append(str(v))
        body.append(line)
    width = [max(len(x[i]) for x in [head] + body) for i in range(len(head))]
    fmt = lambda x: "  ".join(c.ljust(width[i]) if i < len(by) else c.rjust(width[i])  # noqa: E731
                              for i, c in enumerate(x)).rstrip()
    return "\n".join([fmt(head), fmt(["-" * w for w in width])] + [fmt(x) for x in body])


def write_csv(path, rows, by):
    """`rows` as CSV: the `by` labels, then every other column."""
    if not rows:
        return
    rest = [k for k in rows[0] if k not in by]
    for r in rows[1:]:
        rest.extend(k for k in r if k not in by and k not in rest)
    with open(path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(by) + rest)
        w.writeheader()
        w.writerows(rows)
