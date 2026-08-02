"""Regression for the AZ checkpoint inspector (az_inspect.py + tui_az_inspect.py).

Runs every view against a FRESH AZNet and a synthetic shard directory, so it
needs neither a trained checkpoint nor recorded self-play — the point is that the
views compute what they claim and stay pinned to the real observation layout,
not that any particular net is good.

Checks:

  layout     obs_blocks() partitions [0, OBS_SIZE) contiguously; the card
             embedding's padding row is row 0 and vocab card i is row i+1
  vocab      card resolution (exact / substring / ambiguous), labels, and the
             neighbour list's ordering + self-exclusion
  shards     load_shard_sample honours its row budget, and a shard recorded
             against a different obs width fails loudly instead of misframing
  occur      a card planted in every synthetic state is counted in every state
  critic     bucket_table spans N_VALUE_BUCKETS and agrees with archetypes;
             predict's priors are a distribution over LEGAL actions only
  probes     block attribution covers every block; the card-swap probe's own
             identity scores ΔV == 0 (the invariant that proves it is really
             swapping the right float); a life sweep writes the value the
             engine's normalizer would decode back
  diff       an identical checkpoint pair moves nothing; a perturbed card row
             surfaces as that card
  render     every render_* returns non-empty display lines
  tui        the Textual app's parser comes from the shared spec, and its view
             tables dispatch to real implementations

Run standalone:  train/.venv/bin/python train/test_az_inspect.py
Or as the opt-in ci tier:  train/.venv/bin/python train/ci_check.py --tier azinspect
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import az_inspect as azi
import archetypes
import decode
import env
from az_net import AZNet, N_VALUE_BUCKETS, obs_space_from_const

_EMBED_DIM = 32          # a small net: this test is about the views, not capacity
_N_SHARD_ROWS = 24
_N_SHARDS = 3
_PLANTED = "Lightning Bolt"

_failures = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        _failures.append(msg)


def make_net(seed=0):
    import torch
    torch.manual_seed(seed)
    net = AZNet(obs_space_from_const(), embed_dim=_EMBED_DIM)
    net.eval()
    return net


def synth_obs(rng, planted_idx):
    """One plausible observation: a step, life totals, a battlefield permanent,
    a planted hand card, and a matchup bucket. Not a legal game state — a state
    whose decoded fields are in range, which is what the views consume."""
    o = np.zeros(env.OBS_SIZE, dtype=np.float32)
    # Card-id floats default to the empty sentinel, not 0 (which is a real card).
    empty = -1.0 / azi.N_CARD_TYPES
    for s in range(env._PERM_SLOTS):
        for base in (env._SELF_PERM_START, env._OPP_PERM_START):
            o[base + s * env._PERM_SLOT_SIZE + env._PERM_CARD_OFF] = empty
    for s in range(env._HAND_SLOTS_TOTAL):
        o[env._HAND_START + s * env._HAND_SLOT_SIZE] = empty
    o[env._SELF_BLOCK_START + env._PB_LIFE] = rng.integers(1, 21) / 20.0
    o[env._OPP_BLOCK_START + env._PB_LIFE] = rng.integers(1, 21) / 20.0
    o[env._SELF_BLOCK_START + env._PB_HAND_CT] = 0.3
    o[env._STEP_ONEHOT_START + 3] = 1.0                    # First Main
    o[env._IS_ACTIVE_IDX] = 1.0
    o[env._CUR_TURN_IDX] = rng.integers(1, 12) / 50.0
    # A permanent on our battlefield, and the planted card in hand slot 0.
    o[env._SELF_PERM_START + env._PERM_CARD_OFF] = 0.0     # vocab card 0
    o[env._HAND_START] = planted_idx / azi.N_CARD_TYPES
    o[env.BUCKET_IDX] = float(rng.integers(0, N_VALUE_BUCKETS))
    return o


def synth_shards(path, rng, planted_idx):
    """A shard directory in the trainer's schema (obs / pi / z / mask)."""
    for k in range(_N_SHARDS):
        obs = np.stack([synth_obs(rng, planted_idx)
                        for _ in range(_N_SHARD_ROWS)])
        mask = np.zeros((_N_SHARD_ROWS, env.MAX_ACTIONS), dtype=bool)
        pi = np.zeros((_N_SHARD_ROWS, env.MAX_ACTIONS), dtype=np.float32)
        for r in range(_N_SHARD_ROWS):
            n = int(rng.integers(2, 6))
            mask[r, :n] = True
            w = rng.random(n)
            pi[r, :n] = w / w.sum()
        z = rng.choice([-1.0, 1.0], size=_N_SHARD_ROWS).astype(np.float32)
        np.savez_compressed(os.path.join(path, f"shard_x_{k}_0.npz"),
                            obs=obs, pi=pi, z=z, mask=mask)


def test_layout(net):
    blocks = azi.obs_blocks()          # raises if the chain is not contiguous
    check(sum(t - s for _, s, t in blocks) == env.OBS_SIZE,
          f"obs_blocks covers all {env.OBS_SIZE} floats in {len(blocks)} blocks")
    w = net.trunk.card_emb.weight.detach().cpu().numpy()
    check(np.allclose(w[0], 0.0),
          "card embedding row 0 is the zeroed padding row (padding_idx=0)")
    mat = azi.card_embedding(net)
    check(mat.shape == (azi.N_CARD_TYPES, _EMBED_DIM),
          f"card_embedding is (N_CARD_TYPES, D) = {mat.shape}")
    check(np.allclose(mat[7], w[8]),
          "card_embedding row i is table row i+1 (the padding offset)")
    return mat


def test_vocab(mat):
    bolt = azi.resolve_card(_PLANTED)
    check(azi.card_name(bolt) == _PLANTED, f"exact name resolves ({_PLANTED})")
    check(azi.resolve_card("lightning bolt") == bolt,
          "resolution is case-insensitive")
    for bad, why in ((_PLANTED + " nonesuch", "unknown card"),
                     ("", "empty query")):
        try:
            azi.resolve_card(bad)
            check(False, f"{why} raises")
        except ValueError:
            check(True, f"{why} raises")
    check(azi.card_primary_type(bolt) == "Instant",
          "primary type comes from the card script (Lightning Bolt = Instant)")
    check(azi.card_cost_colors(bolt) == "R" and azi.card_cmc(bolt) == 1,
          "cost colors / mana value come from the generated cost matrix")

    nn = azi.nearest_cards(mat, bolt, k=8)
    cos = [c for _, c in nn]
    check(len(nn) == 8, "nearest_cards returns k neighbours")
    check(all(i != bolt for i, _ in nn), "the query card excludes itself")
    check(cos == sorted(cos, reverse=True) and max(cos) <= 1.0001,
          "neighbours are sorted by descending cosine")

    ids = azi.named_card_ids()
    res = azi.knn_purity(mat, azi.card_labels("land", ids), ids, k=5)
    check(0.0 <= res["purity"] <= 1.0 and 0.0 <= res["baseline"] <= 1.0,
          "kNN purity and its chance baseline are probabilities")
    assign, centers = azi.kmeans(mat[ids][:80], 4, seed=0)
    check(len(assign) == 80 and set(assign) <= {0, 1, 2, 3},
          "k-means assigns every card to one of k clusters")
    coords, frac = azi.pca2(mat[ids][:80])
    check(coords.shape == (80, 2) and 0.0 <= frac[0] <= 1.0,
          "PCA returns 2D coordinates and an explained-variance fraction")


def test_shards(data_dir, planted_idx):
    s = azi.load_shard_sample(data_dir, max_rows=40, seed=0)
    total = _N_SHARDS * _N_SHARD_ROWS
    check(s["obs"].shape[1] == env.OBS_SIZE and s["obs"].shape[0] <= 40,
          f"sample honours the row budget ({s['obs'].shape[0]} <= 40 of {total})")
    check(s["mask"].shape[1] == env.MAX_ACTIONS and s["pi"].shape == s["mask"].shape,
          "pi/mask keep the MAX_ACTIONS width")

    with tempfile.TemporaryDirectory() as bad:
        np.savez_compressed(
            os.path.join(bad, "shard_bad_0_0.npz"),
            obs=np.zeros((2, env.OBS_SIZE - 3), dtype=np.float32),
            pi=np.zeros((2, env.MAX_ACTIONS), dtype=np.float32),
            z=np.zeros(2, dtype=np.float32),
            mask=np.zeros((2, env.MAX_ACTIONS), dtype=bool))
        try:
            azi.load_shard_sample(bad, max_rows=2)
            check(False, "a stale-layout shard is rejected")
        except RuntimeError as e:
            check("observation layout" in str(e),
                  "a stale-layout shard is rejected by message, not a shape error")

    counts, n = azi.card_occurrences(s["obs"], limit=8)
    check(counts[planted_idx] == n and n > 0,
          f"the planted hand card is counted in all {n} decoded states")
    check(counts.sum() > counts[planted_idx],
          "other cards in the synthetic state are counted too")
    return s


def test_critic(net, sample):
    rows = azi.bucket_table(net)
    check(len(rows) == N_VALUE_BUCKETS,
          f"bucket_table spans all {N_VALUE_BUCKETS} value buckets")
    check(rows[5]["name"] == archetypes.bucket_name(5),
          "bucket rows are named by archetypes.bucket_name")

    vals, priors = azi.predict(net, sample["obs"], sample["mask"])
    check(np.all(np.abs(vals) <= 1.0), "values are tanh-bounded to [-1, 1]")
    legal = sample["mask"]
    check(np.allclose(priors[legal].reshape(-1).sum(),
                      priors.sum(), atol=1e-5),
          "priors put no mass on illegal actions")
    check(np.allclose(priors.sum(axis=1), 1.0, atol=1e-4),
          "priors sum to 1 over each decision's legal actions")

    buckets = azi.obs_buckets(net, sample["obs"])
    check(all(b == net.obs_value_bucket(o)
              for b, o in zip(buckets, sample["obs"])),
          "obs_buckets matches AZNet.obs_value_bucket row by row")
    check(np.all((0 <= buckets) & (buckets < N_VALUE_BUCKETS)),
          "every sampled bucket is in range")

    cal = azi.bucket_calibration(net, sample)
    check(cal["n"] == sample["obs"].shape[0] and cal["overall_mse"] >= 0,
          "calibration reports an MSE over every sampled decision")
    check(sum(r["n"] for r in cal["rows"]) == cal["n"],
          "calibration's per-bucket counts partition the sample")

    div = azi.policy_divergence(net, sample)
    check(div["n"] > 0 and div["kl"] >= -1e-9,
          "policy divergence reports a non-negative mean KL")
    check(0.0 <= div["top1"] <= 1.0, "top-1 agreement is a fraction")


def test_probes(net, sample):
    blocks = azi.obs_blocks()
    imp = azi.state_block_importance(net, sample, 0, donors=4, seed=0)
    check(len(imp["rows"]) == len(blocks),
          f"single-state attribution covers every one of {len(blocks)} blocks")
    check(all(r["delta"] >= 0 for r in imp["rows"]),
          "attribution deltas are magnitudes")
    check(abs(imp["base"] - azi.state_value(net, sample["obs"][0],
                                            sample["mask"][0])) < 1e-6,
          "attribution's base V is the state's V")

    avg = azi.block_importance(net, sample, n_rows=6, donors=2, seed=0)
    check(len(avg["rows"]) == len(blocks) and avg["n"] == 6,
          "averaged attribution covers every block over the requested states")

    sites = azi.card_id_sites(sample["obs"][0])
    labels = [lab for lab, _, _ in sites]
    check(any(_PLANTED in lab for lab in labels),
          f"card_id_sites finds the planted {_PLANTED} in hand")
    hand_site = next(s for s in sites if _PLANTED in s[0])
    check(sample["obs"][0][hand_site[1]] * azi.N_CARD_TYPES
          == azi.resolve_card(_PLANTED),
          "the site's offset points at that card's identity float")

    probe = azi.card_swap_probe(net, sample["obs"][0], sample["mask"][0],
                                hand_site[1])
    check(len(probe["rows"]) == len(azi.named_card_ids()),
          "the swap probe scores every named card")
    same = [dv for idx, _, dv in probe["rows"] if idx == hand_site[2]]
    check(len(same) == 1 and abs(same[0]) < 1e-6,
          "swapping a card for ITSELF moves V by exactly 0")
    dvs = [dv for _, _, dv in probe["rows"]]
    check(dvs == sorted(dvs, reverse=True), "swap rows are ranked by ΔV")

    res = azi.sweep(net, sample["obs"][0], sample["mask"][0], "self_life")
    check(len(res["v"]) == len(res["values"]), "the sweep evaluates every value")
    idx, scale, values = azi.sweep_fields()["self_life"]
    row = np.array(sample["obs"][0], dtype=np.float32)
    row[idx] = values[-1] / scale
    check(decode._decode_player(row, env._SELF_BLOCK_START)["life"] == values[-1],
          "the sweep writes what the engine's normalizer decodes back")
    tr = azi.sweep_trend(res)
    check(tr["expected"] == +1 and tr["agrees"] in (True, False),
          "own-life sweeps carry an expected direction and a verdict")


def test_diff(net, tmp):
    a = os.path.join(tmp, "same_a__azv1.pt")
    b = os.path.join(tmp, "same_b__azv2.pt")
    net.save(a, 1)
    net.save(b, 2)
    d = azi.checkpoint_diff(a, b)
    check(not d["only_a"] and not d["only_b"] and not d["shape_diff"],
          "identical checkpoints share every tensor")
    check(max(t["delta"] for t in d["tensors"]) == 0.0,
          "identical checkpoints move nothing")

    import torch
    bolt = azi.resolve_card(_PLANTED)
    sd = torch.load(b, map_location="cpu")
    sd["trunk.card_emb.weight"][bolt + 1] += 5.0
    torch.save(sd, b)
    d = azi.checkpoint_diff(a, b)
    check(d["cards"] and d["cards"][0][1] == _PLANTED,
          f"a perturbed embedding row surfaces as {_PLANTED}")
    check(d["tensors"][0]["name"] == "trunk.card_emb.weight",
          "and as the tensor that actually changed")


def test_renders(net, path, sample, mat):
    counts, _ = azi.card_occurrences(sample["obs"], limit=6)
    bolt = azi.resolve_card(_PLANTED)
    sites = azi.card_id_sites(sample["obs"][0])
    renders = {
        "overview": lambda: azi.render_overview(net, path, sample),
        "neighbors": lambda: azi.render_neighbors(mat, bolt, k=5, counts=counts),
        "structure": lambda: azi.render_structure(mat, k=5),
        "clusters": lambda: azi.render_clusters(mat, k=3),
        "project": lambda: azi.render_projection(mat, width=40, height=10),
        "occur": lambda: azi.render_occurrences(counts, 6, top_n=5),
        "catemb": lambda: azi.render_category_embedding(net),
        "buckets": lambda: azi.render_buckets(
            net, azi.obs_buckets(net, sample["obs"])),
        "calib": lambda: azi.render_calibration(
            azi.bucket_calibration(net, sample)),
        "divergence": lambda: azi.render_divergence(
            azi.policy_divergence(net, sample)),
        "state": lambda: azi.render_state(sample, 0, net=net),
        "blocks": lambda: azi.render_block_importance(
            azi.state_block_importance(net, sample, 0, donors=2), single=True),
        "swap": lambda: azi.render_card_swap(
            azi.card_swap_probe(net, sample["obs"][0], sample["mask"][0],
                                sites[0][1]), sites[0][0], top_n=3),
        "sweep": lambda: azi.render_sweeps(net, sample["obs"][0],
                                           sample["mask"][0]),
    }
    bad = []
    for name, fn in renders.items():
        lines = fn()
        if not lines or not all(isinstance(x, str) for x in lines):
            bad.append(name)
    check(not bad, f"all {len(renders)} render_* views return display lines"
                   + (f" (bad: {bad})" if bad else ""))


def test_tui_wiring():
    import tui_az_inspect as tui
    from cli_spec import ALL_TOOLS, AZ_INSPECT_TOOL, iter_args
    check(AZ_INSPECT_TOOL in ALL_TOOLS,
          "the inspector is registered in cli_spec.ALL_TOOLS (./tui.sh menu)")
    args = tui.build_parser().parse_args([])
    spec_dests = {a.dest for a in iter_args(AZ_INSPECT_TOOL.subs[0])}
    check(spec_dests <= set(vars(args)),
          "the script's parser is built from the shared spec (no drift)")
    check(args.model == "gen", "the spec's defaults reach the parser")

    app = tui.InspectApp(args)
    check(app._args.shards == azi.AZ_DATA_DIR,
          "an unset --shards falls back to the recorded self-play directory")
    views = ([k for k, _ in tui._EMB_VIEWS] + [k for k, _ in tui._CRITIC_VIEWS]
             + [k for k, _ in tui._PROBE_VIEWS])
    check(len(views) == len(set(views)), "view keys are unique across the panes")
    # Every sidebar key must be a branch in _compute, else selecting it renders
    # "unknown view" at runtime — cheap to check against the dispatcher's own
    # string constants, and it catches a view added to one list only.
    handled = set(tui.InspectApp._compute.__code__.co_consts)
    check(set(views) <= handled,
          f"every view key is dispatched in _compute "
          f"(missing: {sorted(set(views) - handled)})")
    check(set(tui._NEEDS_SHARDS) >= {k for k, _ in tui._PROBE_VIEWS},
          "probe views are declared as needing recorded self-play")


async def _drive_app(app, pilot):
    """Wait for the load worker, then let every pane's initial render land."""
    for _ in range(400):                       # <= 80s: torch import + sampling
        await pilot.pause(0.2)
        if app._net is not None and app._counts is not None:
            break
    check(app._net is not None, "the app loads a checkpoint + shard sample")
    from textual.widgets import Static
    panes = ("emb", "critic", "probe")
    for _ in range(400):
        await pilot.pause(0.2)
        texts = {p: str(app.query_one(f"#{p}-out", Static).content) for p in panes}
        if not any(t.startswith("computing") for t in texts.values()):
            break
    # Each pane renders on load. A single shared worker group would leave two of
    # the three stuck at "computing…" forever (exclusive cancels its group), so
    # this is the check that keeps the per-pane groups honest.
    stuck = [p for p, t in texts.items() if t.startswith("computing")]
    check(not stuck, f"every pane renders its initial view (stuck: {stuck})")
    crashed = [p for p, t in texts.items() if "Traceback" in t]
    check(not crashed, f"no pane rendered a traceback (crashed: {crashed})")

    # A view switch in one pane still lands.
    app._render("emb", "structure")
    for _ in range(200):
        await pilot.pause(0.2)
        txt = str(app.query_one("#emb-out", Static).content)
        if not txt.startswith("computing"):
            break
    check("purity" in txt and "Traceback" not in txt,
          "switching views re-renders that pane")


def test_tui_end_to_end(net, tmp):
    """Drive the Textual app headlessly against a synthetic checkpoint+shards."""
    import asyncio
    import tui_az_inspect as tui

    ckpt_dir = os.path.join(tmp, "tui_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, "gen__azv1.pt")
    net.save(ckpt, 1)
    data_dir = os.path.join(tmp, "tui_shards")
    os.makedirs(data_dir, exist_ok=True)
    synth_shards(data_dir, np.random.default_rng(1), azi.resolve_card(_PLANTED))

    args = tui.build_parser().parse_args(
        ["--model", ckpt, "--shards", data_dir, "--max-rows", "16",
         "--count-rows", "6", "--block-rows", "3", "--donors", "1",
         "--neighbors", "5", "--knn", "4", "--clusters", "3", "--top", "5"])
    app = tui.InspectApp(args)

    async def run():
        async with app.run_test(size=(120, 40)) as pilot:
            await _drive_app(app, pilot)

    asyncio.run(run())


def main():
    print("az_inspect regression")
    rng = np.random.default_rng(0)
    planted = azi.resolve_card(_PLANTED)
    net = make_net()
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = os.path.join(tmp, "shards")
        os.makedirs(data_dir)
        synth_shards(data_dir, rng, planted)

        print("\n[layout]")
        mat = test_layout(net)
        print("\n[vocab + embedding]")
        test_vocab(mat)
        print("\n[shards]")
        sample = test_shards(data_dir, planted)
        print("\n[critic]")
        test_critic(net, sample)
        print("\n[probes]")
        test_probes(net, sample)
        print("\n[diff]")
        test_diff(net, tmp)
        print("\n[renders]")
        ckpt = os.path.join(tmp, "same_a__azv1.pt")
        test_renders(net, ckpt, sample, mat)
        print("\n[tui wiring]")
        test_tui_wiring()
        print("\n[tui end-to-end]")
        test_tui_end_to_end(net, tmp)

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print("az_inspect regression: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
