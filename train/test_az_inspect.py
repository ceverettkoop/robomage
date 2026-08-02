"""Regression for the AZ checkpoint inspector (az_inspect.py + tui_az_inspect.py).

Runs every view against a FRESH AZNet and a synthetic shard directory, so it
needs neither a trained checkpoint nor recorded self-play — the point is that the
views compute what they claim and stay pinned to the real observation layout,
not that any particular net is good.

Checks:

  layout     obs_blocks() partitions [0, OBS_SIZE) contiguously; the card
             embedding's padding row is row 0 and vocab card i is row i+1
  card props the frozen printed-property buffer: shape/values/padding, buffer
             not parameter, _embed_ids concat width, bit-identical through an
             optimizer step and a save/load round trip, DFC back-face canary,
             and props-space purity far above chance while a fresh identity
             table sits at chance
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
  weights    the weight-space views: PPO→AZNet key normalization (aliases
             dropped), first-layer column groups exactly tile every encoder's
             in-dim, the numpy entity-encoder mirror matches the torch module,
             spectra/value-geometry/zone/selectivity shapes, PopArt PPO-only
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
    # Compare against the extractor's identity width, NOT this test's net
    # _EMBED_DIM — the two being equal (32) was a coincidence that would mask
    # a width change in either.
    import extractor
    mat = azi.card_embedding(net)
    check(mat.shape == (azi.N_CARD_TYPES, extractor._CARD_EMBED_DIM),
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

    emb, rev, n = azi.card_occurrence_split(s["obs"], limit=8)
    check(emb[planted_idx] == n and n > 0,
          f"the planted hand card is counted in all {n} decoded states")
    check(emb.sum() > emb[planted_idx],
          "other cards in the synthetic state are counted too")
    # The revealed multi-hot never reaches card_emb, so it must not be folded
    # into the embedded count (it is also sticky for a whole match).
    check(rev.sum() == 0,
          "revealed-only counts are reported separately, not summed in")
    counts, n2 = azi.card_occurrences(s["obs"], limit=8)
    check(np.array_equal(counts, emb) and n2 == n,
          "card_occurrences returns the EMBEDDED column")
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


def test_card_props(net, tmp):
    """The frozen printed-property half of the card vector: right shape, right
    values, absent from the optimizer, and bit-identical through a training
    step and a save/load round trip."""
    import torch
    import extractor
    from card_props import _CARD_PROP_MATRIX, _PROP_NAMES, N_CARD_PROPS
    from az_net import decay_exempt_param_groups, load_az

    col = {n: i for i, n in enumerate(_PROP_NAMES)}
    props = net.trunk.card_props
    check(tuple(props.shape) == (azi.N_CARD_TYPES + 1, N_CARD_PROPS),
          f"card_props buffer is (N_CARD_TYPES + 1, P) = {tuple(props.shape)}")
    check(bool((props[0] == 0).all()),
          "card_props row 0 is the zero padding row")
    check("card_props" not in dict(net.named_parameters())
          and not any(k.endswith("card_props") for k in
                      dict(net.named_parameters())),
          "card_props is a buffer, not a parameter")

    # _embed_ids returns [identity | props] and the props tail is the matrix row.
    bolt = azi.resolve_card(_PLANTED)
    emb, present = net.trunk._embed_ids(
        torch.tensor([bolt / azi.N_CARD_TYPES, -1.0 / azi.N_CARD_TYPES]))
    ident_d = extractor._CARD_EMBED_DIM
    check(emb.shape[-1] == ident_d + N_CARD_PROPS,
          f"_embed_ids width is identity + props = {emb.shape[-1]}")
    check(bool(present[0]) and not bool(present[1]),
          "present mask still tracks the id, not the props")
    emb = emb.detach()
    check(np.allclose(emb[0, ident_d:].numpy(), _CARD_PROP_MATRIX[bolt]),
          "props tail equals the generated matrix row")
    check(bool((emb[1] == 0).all()),
          "empty slot embeds to all-zero in both halves")

    # Printed-fact spot checks straight through the buffer.
    check(emb[0, ident_d + col["type_instant"]] == 1.0
          and emb[0, ident_d + col["pip_r"]].item() > 0,
          f"{_PLANTED} props read instant + red pip")
    delver = azi.resolve_card("Delver of Secrets")
    check(_CARD_PROP_MATRIX[delver][col["kw_flying"]] == 0.0,
          "DFC canary: Delver does not inherit its back face's Flying")
    # CR 712.8e: a transform back face's mana value is the front face's; a
    # MODAL back face (CR 712.8d) keeps its own — 0 for the MDFC land.
    aberration = _CARD_PROP_MATRIX[azi.resolve_card("Insectile Aberration")]
    check(aberration[col["cmc"]] == _CARD_PROP_MATRIX[delver][col["cmc"]]
          and aberration[col["cmc"]] > 0,
          "transform back face inherits the front face's mana value")
    check(all(aberration[col[f"pip_{c}"]] == 0 for c in "wubrgc")
          and aberration[col["pip_generic"]] == 0,
          "...but has no mana COST pips of its own")
    meadow = _CARD_PROP_MATRIX[azi.resolve_card("Witch-Blessed Meadow")]
    check(meadow[col["cmc"]] == 0.0 and meadow[col["type_land"]] == 1.0,
          "modal back face keeps its own mana value (0 for the MDFC land)")
    tamiyo2 = _CARD_PROP_MATRIX[azi.resolve_card("Tamiyo, Seasoned Scholar")]
    check(tamiyo2[col["color_g"]] == 1.0 and tamiyo2[col["color_u"]] == 1.0,
          "back-face color indicator (Colors:green,blue) sets both colors")

    # One optimizer step that moves card_emb must not move card_props.
    train_net = make_net(seed=3)
    train_net.train()
    before = train_net.trunk.card_props.clone()
    emb_before = train_net.trunk.card_emb.weight.detach().clone()
    opt = torch.optim.Adam(decay_exempt_param_groups(train_net, 1e-2), lr=0.1)
    obs = torch.zeros(2, env.OBS_SIZE)
    obs[:, env._HAND_START] = bolt / azi.N_CARD_TYPES  # give card_emb gradient
    mask = torch.zeros(2, env.MAX_ACTIONS, dtype=torch.bool)
    mask[:, :3] = True
    logits, value = train_net(obs, mask)
    (logits[mask].sum() + value.sum()).backward()
    opt.step()
    check(not torch.equal(train_net.trunk.card_emb.weight, emb_before),
          "the step moved the trainable identity table")
    check(torch.equal(train_net.trunk.card_props, before),
          "...but left card_props bit-identical")

    # Save/load round trip carries the buffer bit-identically.
    p = os.path.join(tmp, "props_roundtrip__azv1.pt")
    train_net.save(p, 1)
    loaded = load_az(p)
    check(torch.equal(loaded.trunk.card_props, before),
          "save/load_az round-trips card_props bit-identically")

    # The frozen block provably encodes the ground-truth labels the structure
    # view measures — the purity ceiling printed facts alone provide.
    ids = azi.named_card_ids()
    pmat = azi.card_property_block(net)
    check(pmat.shape == (azi.N_CARD_TYPES, N_CARD_PROPS),
          "card_property_block drops the padding row")
    check(np.allclose(azi.card_matrix(net, "full"),
                      np.concatenate([azi.card_embedding(net), pmat], axis=1)),
          "card_matrix('full') is [identity | props]")
    for kind, floor in (("land", 0.9), ("type", 0.5), ("color", 0.4)):
        res = azi.knn_purity(pmat, azi.card_labels(kind, ids), ids, k=10)
        check(res["purity"] >= floor and res["lift"] > 0.15,
              f"props purity[{kind}] {res['purity']:.3f} "
              f"(chance {res['baseline']:.3f}) clears {floor}")
    # ...while the fresh random identity table stays at chance.
    res = azi.knn_purity(azi.card_embedding(net),
                         azi.card_labels("land", ids), ids, k=10)
    check(abs(res["lift"]) < 0.15,
          f"fresh identity table stays near chance on land "
          f"(lift {res['lift']:+.3f})")


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


def test_exposure(net, tmp):
    """Exposure must be an EXACT trained/not-trained split from weights alone."""
    import torch
    ckpt_dir = os.path.join(tmp, "exposure")
    os.makedirs(ckpt_dir, exist_ok=True)
    base = os.path.join(ckpt_dir, "gen__azv10.pt")
    later = os.path.join(ckpt_dir, "gen__azv20.pt")
    net.save(base, 10)
    net.save(later, 20)

    bolt = azi.resolve_card(_PLANTED)
    sd = torch.load(later, map_location="cpu")
    sd["trunk.card_emb.weight"][bolt + 1] += 0.25      # one card trained
    sd["value_head.weight"][3] += 0.5                  # one matchup trained
    torch.save(sd, later)

    exp = azi.card_exposure(later, checkpoint_dir=ckpt_dir)
    check(os.path.basename(exp["baseline"]) == "gen__azv10.pt"
          and exp["kind"] == "snapshot",
          "the baseline defaults to the previous snapshot")
    moved = np.nonzero(exp["moved"])[0]
    check(list(moved) == [bolt],
          f"exactly the perturbed card reads as trained (got {list(moved)})")
    check(exp["delta"][bolt] > 0 and (exp["delta"] >= 0).all(),
          "movement is a non-negative per-row distance")
    bmoved = np.nonzero(exp["bucket_moved"])[0]
    check(list(bmoved) == [3],
          f"exactly the perturbed value bucket reads as trained (got {list(bmoved)})")
    # An untouched row must be bit-identical, not merely close — that exactness
    # is what makes 'never trained' a fact rather than a threshold.
    check(exp["delta"][azi.resolve_card("Mountain")] == 0.0,
          "an untouched row moves by exactly 0")
    check(np.array_equal(azi.exposure_counts(exp), exp["delta"]),
          "exposure_counts exposes the movement vector for --min-seen")

    lines = azi.render_exposure(exp)
    check(any("1/330" in ln for ln in lines),
          "the exposure view reports the trained-row count")


def test_weight_views(net, tmp):
    """Weight-space views: family key normalization, first-layer column
    layout, spectra, PopArt, value geometry, zone embedding, selectivity."""
    import torch
    p = os.path.join(tmp, "wv__azv1.pt")
    net.save(p, 1)
    sd, _, kind = azi.load_weight_state(p)
    check(kind == "az" and "trunk.card_emb.weight" in sd,
          "an AZ .pt loads with kind 'az' and AZNet keys")

    fake = {
        "features_extractor.card_emb.weight": sd["trunk.card_emb.weight"],
        "pi_features_extractor.card_emb.weight": sd["trunk.card_emb.weight"],
        "vf_features_extractor.card_emb.weight": sd["trunk.card_emb.weight"],
        "mlp_extractor.value_net.0.weight": torch.zeros(2, 2),
        "value_net.weight": sd["value_head.weight"],
        "popart_mu": torch.zeros(N_VALUE_BUCKETS),
    }
    norm, k2 = azi.normalize_state_dict(fake)
    check(k2 == "ppo" and set(norm) == {"trunk.card_emb.weight",
                                        "value_body.0.weight",
                                        "value_head.weight", "popart_mu"},
          "PPO keys normalize into AZNet space (pi_/vf_ aliases dropped)")

    attr = azi.first_layer_attribution(sd)   # the width asserts run inside
    encs = {a["encoder"] for a in attr}
    check({"perm_encoder", "stack_encoder", "entity_encoder",
           "decklist_encoder", "revealed_encoder", "action_encoder"} <= encs,
          "firstlayer covers every trunk encoder incl. the per-action head")
    check(all(abs(sum(g["share"] for g in a["groups"]) - 1.0) < 1e-6
              for a in attr),
          "each encoder's group energy shares sum to 1")

    battr = azi.body_layer_attribution(sd)   # the remainder assert runs inside
    check({a["body"] for a in battr} == {"policy_body", "value_body"}
          and all(sum(g["width"] for g in a["groups"]) == a["in_dim"]
                  and len(a["arch"]) == 2 * archetypes.N_ARCH
                  for a in battr),
          "bodylayer tiles both bodies' in-dim and names every arch column")

    for layer in ("perm_encoder", "entity_encoder", "policy_body",
                  "value_body"):
        names, prefix = azi.layer_column_names(sd, layer)
        check(len(names) == sd[prefix + ".weight"].shape[1],
              f"layer_column_names labels every {layer} input column")
    prof = azi.unit_profile(sd, "value_body", 0, top=5)
    check(len(prof["pos"]) <= 5 and len(prof["neg"]) <= 5
          and all(v > 0 for _, v in prof["pos"])
          and all(v < 0 for _, v in prof["neg"]),
          "unit_profile splits a unit's row by sign")
    try:
        azi.unit_profile(sd, "no_such_layer", 0)
        check(False, "unit_profile rejects an unknown layer")
    except ValueError:
        check(True, "unit_profile rejects an unknown layer")

    b = azi.resolve_bucket(archetypes.bucket_name(3))
    check(b == 3 and azi.resolve_bucket("3") == 3,
          "resolve_bucket accepts a name and an index")
    conn = azi.bucket_input_connectivity(sd, b, top=7)
    check(sum(g["width"] for g in conn["groups"])
          == sd["value_body.0.weight"].shape[1]
          and abs(sum(g["share"] for g in conn["groups"]) - 1.0) < 1e-6
          and len(conn["top"]) == 7,
          "pathto connectivity tiles the value body input and shares sum to 1")

    acts = azi.entity_unit_activations(sd)
    ids = torch.arange(1, azi.N_CARD_TYPES + 1)
    with torch.no_grad():
        ref = net.trunk.entity_encoder(
            torch.cat([net.trunk.card_emb.weight[ids],
                       net.trunk.card_props[ids]], dim=1)).numpy()
    check(acts.shape == ref.shape and np.allclose(acts, ref, atol=1e-5),
          "entity_unit_activations matches the real torch entity_encoder")
    sel = azi.card_selectivity(sd)
    check(sel["n_units"] == acts.shape[1]
          and sel["n_dead"] + len(sel["units"]) == sel["n_units"],
          "card_selectivity partitions units into dead + ranked alive")

    spec = azi.weight_spectra(sd)
    check(bool(spec)
          and not any(r["name"].endswith("card_props") for r in spec)
          and all(r["eff"] <= min(r["shape"]) + 1e-6 for r in spec),
          "spectra: frozen buffer skipped, effective rank bounded by shape")

    geo = azi.value_head_geometry(sd)
    check(len(geo["rows"]) == N_VALUE_BUCKETS
          and len(geo["pairs"]) == geo["n_live"] * (geo["n_live"] - 1) // 2,
          "value geometry spans every bucket with all live pairs")

    zmat = azi.zone_embedding(sd)
    check(zmat.shape[0] == azi.N_REF_ZONES,
          "zone embedding has a row per ref zone")

    try:
        azi.popart_table(sd)
        check(False, "popart_table rejects an AZ checkpoint")
    except ValueError:
        check(True, "popart_table rejects an AZ checkpoint")
    pp = azi.popart_table({
        "popart_mu": torch.tensor([0.0, 0.5]),
        "popart_sigma": torch.tensor([1.0, 2.0]),
        "popart_count": torch.tensor([0.0, 12.0]),
    })
    check(len(pp["rows"]) == 2 and pp["n_trained"] == 1
          and pp["rows"][1]["trained"],
          "popart_table reads mu/sigma/count and flags trained buckets")

    for name, lines in (
            ("firstlayer", azi.render_first_layer(attr)),
            ("bodylayer", azi.render_body_layer(battr)),
            ("unit", azi.render_unit(prof)),
            ("pathto", azi.render_pathto(conn)),
            ("spectra", azi.render_spectra(spec)),
            ("popart", azi.render_popart(pp)),
            ("valuegeom", azi.render_value_geometry(geo)),
            ("zoneemb", azi.render_zone_embedding(zmat)),
            ("cardsel", azi.render_card_selectivity(sel))):
        check(bool(lines), f"render {name} returns display lines")


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
        "occur": lambda: azi.render_occurrences(counts, 6, top_n=5,
                                                revealed=counts * 0),
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
    # The default is weights-only: no shard load, no Probes pane, and the
    # sidebar offers only views this session can actually compute.
    check(not app._with_shards
          and app._panes == ("emb", "critic", "weights"),
          "the app defaults to weights-only (no Probes pane)")
    offered = [k for k, _ in tui.available_views(tui._EMB_VIEWS, False)
               ] + [k for k, _ in tui.available_views(tui._CRITIC_VIEWS, False)]
    check(not (set(offered) & tui._NEEDS_SHARDS),
          f"no shard-backed view is offered by default (got {offered})")
    check("exposure" in offered,
          "the weights-only exposure view replaces occurrences by default")
    wkeys = {k for k, _ in tui._WEIGHT_VIEWS}
    check([k for k, _ in tui.available_views(tui._WEIGHT_VIEWS, False)]
          == [k for k, _ in tui._WEIGHT_VIEWS]
          and not (wkeys & (tui._NEEDS_SHARDS | tui._NEEDS_NET)),
          "every Weights view is offered without shards and without an AZ net")
    with_shards = tui.InspectApp(tui.build_parser().parse_args(["--with-shards"]))
    check(with_shards._panes == ("emb", "critic", "weights", "probe"),
          "--with-shards restores the Probes pane")
    implied = tui.InspectApp(tui.build_parser().parse_args(["--shards", "/tmp/x"]))
    check(implied._with_shards,
          "naming an explicit --shards directory implies --with-shards")

    views = ([k for k, _ in tui._EMB_VIEWS] + [k for k, _ in tui._CRITIC_VIEWS]
             + [k for k, _ in tui._WEIGHT_VIEWS]
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


async def _drive_app(app, pilot, label):
    """Wait for the load worker, then let every pane's initial render land."""
    ready = (lambda: app._net is not None and app._counts is not None) \
        if app._with_shards else (lambda: app._net is not None)
    for _ in range(400):                       # <= 80s: torch import + sampling
        await pilot.pause(0.2)
        if ready():
            break
    check(app._net is not None, f"[{label}] the app loads a checkpoint")
    from textual.widgets import Static
    panes = app._panes
    for _ in range(400):
        await pilot.pause(0.2)
        texts = {p: str(app.query_one(f"#{p}-out", Static).content) for p in panes}
        if not any(t.startswith("computing") for t in texts.values()):
            break
    # Each pane renders on load. A single shared worker group would leave two of
    # the three stuck at "computing…" forever (exclusive cancels its group), so
    # this is the check that keeps the per-pane groups honest.
    stuck = [p for p, t in texts.items() if t.startswith("computing")]
    check(not stuck, f"[{label}] every pane renders its initial view "
                     f"(stuck: {stuck})")
    crashed = [p for p, t in texts.items() if "Traceback" in t]
    check(not crashed, f"[{label}] no pane rendered a traceback "
                       f"(crashed: {crashed})")

    # A view switch in one pane still lands.
    app._render("emb", "structure")
    for _ in range(200):
        await pilot.pause(0.2)
        txt = str(app.query_one("#emb-out", Static).content)
        if not txt.startswith("computing"):
            break
    check("purity" in txt and "Traceback" not in txt,
          f"[{label}] switching views re-renders that pane")


def test_tui_end_to_end(net, tmp):
    """Drive the Textual app headlessly against a synthetic checkpoint+shards,
    in BOTH modes: the weights-only default and the full --with-shards run."""
    import asyncio
    import tui_az_inspect as tui

    ckpt_dir = os.path.join(tmp, "tui_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Two snapshots so the weights-only run has a baseline for exposure.
    net.save(os.path.join(ckpt_dir, "gen__azv1.pt"), 1)
    ckpt = os.path.join(ckpt_dir, "gen__azv2.pt")
    net.save(ckpt, 2)
    data_dir = os.path.join(tmp, "tui_shards")
    os.makedirs(data_dir, exist_ok=True)
    synth_shards(data_dir, np.random.default_rng(1), azi.resolve_card(_PLANTED))

    common = ["--model", ckpt, "--max-rows", "16", "--count-rows", "6",
              "--block-rows", "3", "--donors", "1", "--neighbors", "5",
              "--knn", "4", "--clusters", "3", "--top", "5"]

    for label, extra in (("weights-only", []),
                         ("with-shards", ["--shards", data_dir])):
        app = tui.InspectApp(tui.build_parser().parse_args(common + extra))

        async def run(app=app, label=label):
            async with app.run_test(size=(120, 40)) as pilot:
                await _drive_app(app, pilot, label)

        asyncio.run(run())
        if not extra:
            check(app._sample is None and app._exposure is not None,
                  "[weights-only] no shards were read, but exposure loaded")


def test_tui_ppo_fallback(net, tmp):
    """Drive the app with load_net forced to miss: the normalized weight
    loader must serve the session (net=None), the Weights and embedding panes
    must compute from the state dict, and the net-gated views must explain
    themselves instead of crashing."""
    import asyncio
    import tui_az_inspect as tui
    from textual.widgets import Static

    ckpt = os.path.join(tmp, "fallback__azv1.pt")
    net.save(ckpt, 1)
    real_load_net = azi.load_net

    def _miss(*a, **k):
        raise FileNotFoundError("forced miss: no AZ checkpoint")

    azi.load_net = _miss
    try:
        app = tui.InspectApp(tui.build_parser().parse_args(["--model", ckpt]))

        async def run():
            async with app.run_test(size=(120, 40)) as pilot:
                for _ in range(400):
                    await pilot.pause(0.2)
                    if app._sd is not None:
                        break
                check(app._net is None and app._sd is not None,
                      "[fallback] the weight loader served the session")
                for _ in range(400):
                    await pilot.pause(0.2)
                    texts = {p: str(app.query_one(f"#{p}-out", Static).content)
                             for p in app._panes}
                    if not any(t.startswith("computing")
                               for t in texts.values()):
                        break
                stuck = [p for p, t in texts.items()
                         if t.startswith("computing")]
                crashed = [p for p, t in texts.items() if "Traceback" in t]
                check(not stuck and not crashed,
                      f"[fallback] every pane settles "
                      f"(stuck: {stuck}, crashed: {crashed})")
                check("needs an AZ checkpoint" in texts["critic"],
                      "[fallback] net-gated views explain instead of crashing")
                check("attribution" in texts["weights"],
                      "[fallback] the Weights pane computes from the state "
                      "dict")

        asyncio.run(run())
    finally:
        azi.load_net = real_load_net


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
        print("\n[card props]")
        test_card_props(net, tmp)
        print("\n[diff]")
        test_diff(net, tmp)
        print("\n[exposure]")
        test_exposure(net, tmp)
        print("\n[weight views]")
        test_weight_views(net, tmp)
        print("\n[renders]")
        ckpt = os.path.join(tmp, "same_a__azv1.pt")
        test_renders(net, ckpt, sample, mat)
        print("\n[tui wiring]")
        test_tui_wiring()
        print("\n[tui end-to-end]")
        test_tui_end_to_end(net, tmp)
        print("\n[tui ppo fallback]")
        test_tui_ppo_fallback(net, tmp)

    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + "; ".join(_failures))
        return 1
    print("az_inspect regression: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
