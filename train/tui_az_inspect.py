"""AZ checkpoint inspector as a full-screen Textual TUI.

The interactive front end for :mod:`az_inspect` — the same views the CLI prints,
but with the checkpoint, the embedding matrix and the self-play sample loaded ONCE
and reused across views, and with the one thing a CLI cannot do: drilling through
the card embedding by clicking a neighbour and walking outward from it.

Two panes (a third, Probes, is added by the probe views):

  Embedding — neighbours of a card (click a neighbour to recentre on it, ``u``
              walks back), kNN label purity, k-means clusters, PCA scatter,
              per-card occurrence counts, action-category neighbours.
  Critic    — checkpoint overview, the per-matchup value-head column map,
              per-bucket calibration against recorded outcomes, and raw-net vs
              search-posterior divergence by action category.

Everything is computed from the checkpoint weights and the recorded self-play
shards; no game is played and the engine is never started.

Run from the repo root:
    train/.venv/bin/python train/tui_az_inspect.py
    train/.venv/bin/python train/tui_az_inspect.py --model gen__azv384000
    train/.venv/bin/python train/tui_az_inspect.py --no-shards      # weights only
"""

import argparse
import traceback

import numpy as np

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import (Footer, Header, Input, OptionList, Static,
                             TabbedContent, TabPane)
from textual.widgets.option_list import Option

import az_inspect as azi

# (view key, sidebar label). The key is dispatched in _render_worker.
_EMB_VIEWS = [
    ("neighbors", "Neighbours of a card"),
    ("structure", "Label purity (kNN)"),
    ("clusters", "Clusters (k-means)"),
    ("project", "PCA scatter"),
    ("occur", "Occurrences in self-play"),
    ("catemb", "Action-category embedding"),
]

_CRITIC_VIEWS = [
    ("overview", "Checkpoint overview"),
    ("buckets", "Value-head column map"),
    ("calib", "Calibration vs outcomes"),
    ("divergence", "Priors vs search"),
]

_PROBE_VIEWS = [
    ("state", "Decision + policy"),
    ("blocks", "Block attribution (state)"),
    ("blocksavg", "Block attribution (mean)"),
    ("swap", "Card-swap probe"),
    ("sweep", "Scalar sweeps"),
]

# Views that need the shard sample; offered but reported as unavailable under
# --no-shards rather than silently rendering something weaker.
_NEEDS_SHARDS = {"occur", "calib", "divergence"} | {k for k, _ in _PROBE_VIEWS}

_MAX_CARD_OPTIONS = 400
# Recorded decisions listed in the Probes sidebar. The sample is thousands of
# rows; the list is a picker, not a census (',' / '.' step through all of them).
_MAX_DECISION_OPTIONS = 250


class Loaded(Message):
    def __init__(self, path, status):
        self.path = path
        self.status = status
        super().__init__()


class Rendered(Message):
    def __init__(self, target, lines, cards=None):
        self.target = target
        self.lines = lines
        self.cards = cards          # [(vocab_idx, label)] for the drill-down list
        super().__init__()


class Failed(Message):
    def __init__(self, target, text):
        self.target = target
        self.text = text
        super().__init__()


class InspectApp(App):
    CSS = """
    .sidebar   { width: 34; border-right: solid $accent; }
    .views     { height: auto; max-height: 40%; border: round $primary; }
    .head      { height: 1; padding: 0 1; color: $accent; text-style: bold; }
    #status    { height: 2; padding: 0 1; color: $text-muted; }
    #emb-cards, #probe-decisions { height: 1fr; border: round $surface; }
    /* The swap view's site picker; hidden on the other probe views. */
    #probe-sites { height: 40%; border: round $primary; display: none; }
    /* Taller than the text pane above it: on the neighbours view the list IS
       the content, and the pane keeps only the card's header. Hidden (so the
       text pane takes the whole column) on every other view. */
    #emb-nn    { height: 3fr; border: round $primary; display: none; }
    .out-scroll { height: 1fr; overflow-x: auto; }
    .out       { padding: 0 1; width: auto; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "rerun", "Re-run view"),
        Binding("slash", "focus_filter", "Filter cards"),
        Binding("u", "back", "Back (card)"),
        Binding("comma", "step_row(-1)", "◀ decision"),
        Binding("full_stop", "step_row(1)", "decision ▶"),
    ]

    def __init__(self, args):
        super().__init__()
        self._args = args
        self._net = None
        self._path = None
        self._mat = None            # card embedding matrix
        self._sample = None         # recorded self-play sample (or None)
        self._counts = None         # per-card occurrence counts (or None)
        self._count_states = 0      # states actually decoded for those counts
        self._card = None           # current card index for the neighbours view
        self._history = []          # drill-down stack of card indices
        self._row = 0               # current recorded decision (probe views)
        self._site = None           # chosen card-identity site for the swap probe
        self._view = {"emb": "neighbors", "critic": "overview",
                      "probe": "state"}
        self._busy = False

    # ----- layout -----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(id="tabs"):
            with TabPane("Embedding", id="tab-emb"):
                with Horizontal():
                    with Vertical(classes="sidebar"):
                        yield Static("Loading checkpoint…", id="status")
                        yield Static("Views", classes="head")
                        yield OptionList(id="emb-views", classes="views")
                        yield Static("Cards", classes="head")
                        yield Input(placeholder="filter cards…", id="card-filter")
                        yield OptionList(id="emb-cards")
                    with Vertical():
                        with VerticalScroll(classes="out-scroll"):
                            yield Static(id="emb-out", markup=False,
                                         classes="out")
                        yield OptionList(id="emb-nn")
            with TabPane("Critic", id="tab-critic"):
                with Horizontal():
                    with Vertical(classes="sidebar"):
                        yield Static("Views", classes="head")
                        yield OptionList(id="critic-views", classes="views")
                    with VerticalScroll(classes="out-scroll"):
                        yield Static(id="critic-out", markup=False, classes="out")
            with TabPane("Probes", id="tab-probe"):
                with Horizontal():
                    with Vertical(classes="sidebar"):
                        yield Static("Views", classes="head")
                        yield OptionList(id="probe-views", classes="views")
                        yield Static("Recorded decisions", classes="head")
                        yield OptionList(id="probe-decisions")
                    with Vertical():
                        with VerticalScroll(classes="out-scroll"):
                            yield Static(id="probe-out", markup=False,
                                         classes="out")
                        yield OptionList(id="probe-sites")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "RoboMage · AZ inspect"
        self.sub_title = self._args.model
        for key, label in _EMB_VIEWS:
            self.query_one("#emb-views", OptionList).add_option(
                Option(label, id=key))
        for key, label in _CRITIC_VIEWS:
            self.query_one("#critic-views", OptionList).add_option(
                Option(label, id=key))
        for key, label in _PROBE_VIEWS:
            self.query_one("#probe-views", OptionList).add_option(
                Option(label, id=key))
        for pane in ("emb", "critic", "probe"):
            self.query_one(f"#{pane}-out", Static).update("Loading checkpoint…")
        self._load()

    # ----- loading -----

    @work(thread=True, group="load")
    def _load(self) -> None:
        a = self._args
        try:
            net, path = azi.load_net(a.model)
            self._net = net
            self._path = path
            self._mat = azi.card_embedding(net)
            status = [f"{path.split('/')[-1]}  "
                      f"steps={azi.checkpoint_meta(path).get('steps', '?')}"]
            if not a.no_shards:
                self._sample = azi.load_shard_sample(
                    a.shards, max_rows=a.max_rows, window=a.window, seed=a.seed)
                self._counts, self._count_states = azi.card_occurrences(
                    self._sample["obs"], limit=a.count_rows, seed=a.seed)
                status.append(f"{self._sample['obs'].shape[0]} decisions / "
                              f"{self._sample['n_shards']} shards, "
                              f"{self._count_states} decoded")
            else:
                status.append("weights only (--no-shards)")
            self.post_message(Loaded(path, "\n".join(status)))
        except Exception:
            self.post_message(Failed("emb", traceback.format_exc()))

    def on_loaded(self, message: Loaded) -> None:
        self.query_one("#status", Static).update(message.status)
        self._populate_cards("")
        self._populate_decisions()
        self._card = self._default_card()
        for pane in ("emb", "critic", "probe"):
            self.query_one(f"#{pane}-views", OptionList).highlighted = 0
        self._render("critic", "overview")
        self._render("probe", "state")
        self._render("emb", "neighbors")

    def _default_card(self):
        """Open on the most-played card when occurrence counts are available (it
        is the one whose embedding actually got trained), else the first card."""
        ids = azi.named_card_ids()
        if self._counts is None:
            return int(ids[0])
        return int(max(ids, key=lambda i: self._counts[i]))

    # ----- card list / drill-down -----

    def _populate_cards(self, needle: str) -> None:
        opts = self.query_one("#emb-cards", OptionList)
        opts.clear_options()
        needle = needle.strip().lower()
        ids = azi.named_card_ids()
        if self._counts is not None:
            ids = sorted(ids, key=lambda i: -self._counts[i])
        shown = 0
        for i in ids:
            name = azi.card_name(i)
            if needle and needle not in name.lower():
                continue
            seen = "" if self._counts is None else f"  ({int(self._counts[i])})"
            opts.add_option(Option(f"{name}{seen}", id=str(int(i))))
            shown += 1
            if shown >= _MAX_CARD_OPTIONS:
                break

    def _populate_decisions(self) -> None:
        """Label the first slice of sampled decisions by turn/step/outcome, so
        picking a probe target is not picking a bare row number."""
        opts = self.query_one("#probe-decisions", OptionList)
        opts.clear_options()
        if self._sample is None:
            return
        obs, z = self._sample["obs"], self._sample["z"]
        for r in range(min(_MAX_DECISION_OPTIONS, obs.shape[0])):
            turn = azi.decode.decode_turn(obs[r])
            step = azi.decode.decode_step(obs[r])
            opts.add_option(Option(f"#{r:<5} T{turn:<3} {step[:14]:<14} "
                                   f"z={z[r]:+.0f}", id=str(r)))

    def _populate_sites(self) -> None:
        """Card-identity sites of the current decision (the swap probe's menu)."""
        opts = self.query_one("#probe-sites", OptionList)
        opts.clear_options()
        if self._sample is None:
            return
        for i, (label, _, _) in enumerate(
                azi.card_id_sites(self._sample["obs"][self._row])):
            opts.add_option(Option(label, id=str(i)))

    def action_step_row(self, delta: int) -> None:
        if self._sample is None:
            return
        self._row = max(0, min(self._sample["obs"].shape[0] - 1,
                               self._row + int(delta)))
        self._site = None
        self._populate_sites()
        self._render("probe", self._view["probe"])

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "card-filter":
            self._populate_cards(event.value)

    def on_option_list_option_selected(
            self, event: OptionList.OptionSelected) -> None:
        oid = event.option.id
        which = event.option_list.id
        if which == "emb-views":
            self._render("emb", oid)
        elif which == "critic-views":
            self._render("critic", oid)
        elif which == "probe-views":
            if oid == "swap":
                self._populate_sites()
            self._render("probe", oid)
        elif which == "probe-decisions":
            self._row = int(oid)
            self._site = None
            self._populate_sites()
            self._render("probe", self._view["probe"])
        elif which == "probe-sites":
            self._site = int(oid)
            self._render("probe", "swap")
        elif which in ("emb-cards", "emb-nn"):
            if self._card is not None and oid != str(self._card):
                self._history.append(self._card)
            self._card = int(oid)
            self._render("emb", "neighbors")

    def action_back(self) -> None:
        if self._history:
            self._card = self._history.pop()
            self._render("emb", "neighbors")

    def action_focus_filter(self) -> None:
        self.query_one("#card-filter", Input).focus()

    def action_rerun(self) -> None:
        pane = self.query_one("#tabs", TabbedContent).active
        target = "emb" if pane == "tab-emb" else "critic"
        self._render(target, self._view[target])

    # ----- rendering -----

    def _render(self, target: str, key: str) -> None:
        if self._net is None or not key:
            return
        self._view[target] = key
        out = self.query_one(f"#{target}-out", Static)
        out.update(f"computing {key}…")
        self.query_one("#emb-nn", OptionList).display = (
            target == "emb" and key == "neighbors")
        self.query_one("#probe-sites", OptionList).display = (
            target == "probe" and key == "swap")
        self._render_worker(target, key)

    @work(thread=True, group="render", exclusive=True)
    def _render_worker(self, target: str, key: str) -> None:
        try:
            if key in _NEEDS_SHARDS and self._sample is None:
                self.post_message(Rendered(
                    target, [f"'{key}' needs recorded self-play; this session "
                             "was started with --no-shards."]))
                return
            lines, cards = self._compute(key)
            self.post_message(Rendered(target, lines, cards))
        except Exception:
            self.post_message(Failed(target, traceback.format_exc()))

    def _compute(self, key):
        """Run one view. Returns ``(lines, drill_down_cards_or_None)``."""
        a = self._args
        mat, counts = self._mat, self._counts
        if key == "neighbors":
            # The neighbours themselves live in the clickable list, so the text
            # pane keeps only the header (identity, row norm, times seen).
            idx = self._card
            nn = azi.nearest_cards(mat, idx, k=a.neighbors,
                                   candidates=self._candidates())
            lines = azi.render_neighbors(mat, idx, counts=counts, rows=False)
            lines += ["", azi.NEIGHBOR_HEADER]
            cards = [(j, azi.neighbor_row(j, cos, counts)) for j, cos in nn]
            return lines, cards
        if key == "structure":
            return azi.render_structure(mat, k=a.knn, counts=counts,
                                        min_seen=a.min_seen), None
        if key == "clusters":
            return azi.render_clusters(mat, k=a.clusters, seed=a.seed,
                                       counts=counts,
                                       min_seen=a.min_seen), None
        if key == "project":
            size = self.query_one("#emb-out", Static).container_size
            return azi.render_projection(mat, width=max(40, size.width - 2),
                                         height=max(12, size.height - 4),
                                         mark=a.mark, counts=counts,
                                         min_seen=a.min_seen), None
        if key == "occur":
            return azi.render_occurrences(counts, self._count_states,
                                          top_n=a.top), None
        if key == "catemb":
            return azi.render_category_embedding(self._net), None
        if key == "overview":
            return azi.render_overview(self._net, self._path, self._sample), None
        if key == "buckets":
            sb = (None if self._sample is None
                  else azi.obs_buckets(self._net, self._sample["obs"]))
            return azi.render_buckets(self._net, sb), None
        if key == "calib":
            return azi.render_calibration(
                azi.bucket_calibration(self._net, self._sample)), None
        if key == "divergence":
            return azi.render_divergence(
                azi.policy_divergence(self._net, self._sample,
                                      top_n=a.top)), None
        if key == "state":
            return azi.render_state(self._sample, self._row, net=self._net,
                                    top_n=a.top), None
        if key == "blocks":
            return azi.render_block_importance(
                azi.state_block_importance(self._net, self._sample, self._row,
                                           seed=a.seed),
                top_n=a.top, single=True), None
        if key == "blocksavg":
            return azi.render_block_importance(
                azi.block_importance(self._net, self._sample,
                                     n_rows=a.block_rows, donors=a.donors,
                                     seed=a.seed), top_n=a.top), None
        if key == "swap":
            sites = azi.card_id_sites(self._sample["obs"][self._row])
            if not sites:
                return ["this decision has no card on the battlefield or in "
                        "hand to swap — pick another (',' / '.')"], None
            if self._site is None or self._site >= len(sites):
                return ([f"{len(sites)} card-identity sites in decision "
                         f"{self._row} — pick one from the list below."], None)
            label, off, _ = sites[self._site]
            probe = azi.card_swap_probe(self._net,
                                        self._sample["obs"][self._row],
                                        self._sample["mask"][self._row], off)
            return azi.render_card_swap(probe, label, top_n=a.top,
                                        counts=counts), None
        if key == "sweep":
            return azi.render_sweeps(self._net, self._sample["obs"][self._row],
                                     self._sample["mask"][self._row]), None
        return [f"unknown view {key!r}"], None

    def _candidates(self):
        """Neighbour candidate pool, honouring --min-seen so warm-start noise
        rows do not crowd the list."""
        if self._counts is None or self._args.min_seen <= 0:
            return None
        return np.array([i for i in azi.named_card_ids()
                         if self._counts[i] >= self._args.min_seen])

    def on_rendered(self, message: Rendered) -> None:
        self.query_one(f"#{message.target}-out", Static).update(
            "\n".join(message.lines))
        if message.cards is not None:
            nn = self.query_one("#emb-nn", OptionList)
            nn.clear_options()
            for idx, label in message.cards:
                nn.add_option(Option(label, id=str(int(idx))))

    def on_failed(self, message: Failed) -> None:
        self.query_one(f"#{message.target}-out", Static).update(message.text)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="tui_az_inspect",
        description="Interactive AZ checkpoint inspector (weights + recorded "
                    "self-play; no games are played).")
    ap.add_argument("--model", default="gen",
                    help="AZ checkpoint spec: 'gen', a snapshot stem, or a path")
    ap.add_argument("--shards", default=azi.AZ_DATA_DIR,
                    help="directory of shard_*.npz self-play records")
    ap.add_argument("--no-shards", action="store_true",
                    help="weights only — skip loading recorded self-play")
    ap.add_argument("--max-rows", type=int, default=3000,
                    help="recorded decisions to sample (default: 3000)")
    ap.add_argument("--count-rows", type=int, default=800,
                    help="states decoded for occurrence counts (default: 800)")
    ap.add_argument("--window", type=int, default=None,
                    help="use only the newest N shards")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--min-seen", type=int, default=0,
                    help="drop cards seen fewer than N times from the embedding "
                         "views (their rows are untrained)")
    ap.add_argument("--neighbors", type=int, default=20,
                    help="neighbours listed per card")
    ap.add_argument("--knn", type=int, default=10,
                    help="k for the label-purity view")
    ap.add_argument("--clusters", type=int, default=8, help="k for k-means")
    ap.add_argument("--mark", default="color", choices=azi.LABEL_KINDS,
                    help="marker label for the PCA scatter")
    ap.add_argument("--top", type=int, default=25,
                    help="rows in the occurrence / divergence / probe tables")
    ap.add_argument("--block-rows", type=int, default=120,
                    help="states averaged by the mean block-attribution view")
    ap.add_argument("--donors", type=int, default=3,
                    help="donor states per block in that average")
    return ap


def main(argv=None):
    InspectApp(build_parser().parse_args(argv)).run()


if __name__ == "__main__":
    main()
