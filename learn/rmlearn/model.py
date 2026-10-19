"""The learner's network (docs/rnad_training_plan.md §4 L2) on the step-5 network contract.

    cfg = ModelConfig(d=256, layers=4)
    model = RMNet(cfg, tables)                     # tables: spec.load_tables(...)
    out = model(*inputs)                            # dict: logits [B,A], value [B], summary, aux_*
    out = model(*inputs, omni=omni_inputs, seat=seat)   # + critic_value (cfg.omniscient_critic)
    policy = model.inference_module(version=3)     # InferencePolicy: the contract's forward only

**Inputs** are the contract's ten padded tensors (docs/featurizer.md §6; ``batch.pad``):
``tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok, act_mask``.

**Architecture.** A token is the sum of
- its numeric slots through one linear map over the spec's (token type, slot) pairs (each type's
  slots scattered into their own columns, so the map is per type without a per-token weight gather);
- its type's embedding and the sum of its categorical slots' embeddings (one table over the global
  categorical id space, id 0 the pad);
- per card slot the card's row (frozen card-table and text-table rows projected, plus a learned
  card-id embedding; during training the id part is replaced, with probability ``id_dropout`` per
  slot, by a learned unknown-card vector) plus a per-slot vector;
- per ability slot the ability row (frozen ability-table row projected, its parts' projections
  averaged in) plus a per-slot vector.
An entity transformer (pre-LayerNorm, ``layers`` blocks) whose attention adds, per head and layer,
a learned bias for each relation type at (src, dst) and another at (dst, src), scatter-added into
the attention scores; padded tokens are masked as keys. A summary ``g`` (the masked mean and an
attention pool of the real tokens, mixed by an MLP) conditions a pointer head over the menu rows'
tokens (an MLP of the row's token, ``g`` and their product; padded rows -1e9) and the value head
(an MLP of ``g``, tanh).

**Auxiliary heads** (training only, never exported) predict the learner targets from ``g``: the
opponent's registered list and archetype (``METAGAME`` rows), its hand as a distribution over card
ids and each seat's library tops (``library_depth`` positions) as card ids. The card-id heads score
the same card rows the tokens read (a dot product), so they shape the card representation.

**Omniscient critic** (``omniscient_critic``; the plan's open issue 1): a second encoder of the
same design over the omniscient featurization of the same decision, with the deciding seat's
embedding added to every token (the omniscient frame is A-relative), and its own value head
(``critic_value``). It shares the card and ability rows with the policy.

Export-friendly: no control flow that depends on the data; gathers, scatter-adds and masks only.
``InferencePolicy`` holds the card and ability rows folded into buffers (constants of the weights)
and the four metadata strings; ``export.py`` writes it as TorchScript and AOTInductor ``.pt2``.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as Fn

NEG = -1e9


@dataclass
class ModelConfig:
    d: int = 192              # token width
    heads: int = 4
    layers: int = 4
    ff_mult: int = 4          # feed-forward width = ff_mult * d
    id_dropout: float = 0.1   # probability a card slot's id embedding is replaced by the unknown-card vector
    aux_heads: bool = True    # hidden-information heads (training only)
    omniscient_critic: bool = False
    critic_layers: int = 2    # the omniscient critic's encoder depth

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── blocks ────────────────────────────────────────────────────────────────────────────────────

class Layer(nn.Module):
    """A pre-LayerNorm transformer block with per-head relation biases."""

    def __init__(self, d: int, heads: int, ff: int, n_rel: int):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.n_rel = n_rel
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff1 = nn.Linear(d, ff)
        self.ff2 = nn.Linear(ff, d)
        # Per relation type a bias per head at (src, dst) (rows 0..n_rel-1) and at (dst, src) (rows n_rel..).
        self.rel_bias = nn.Embedding(2 * max(n_rel, 1), heads)
        nn.init.zeros_(self.rel_bias.weight)

    def forward(self, x, idx_fwd, idx_rev, rel_type, rel_w, key_col):
        b, n, d = x.shape
        h = self.heads
        y = self.ln1(x)
        qkv = self.qkv(y).view(b, n, 3, h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # The scores (the attention matrix is the memory-bound part, so no extra pass over it): the
        # padded keys masked through one more q·k dimension (1 · key_col: 0 for a real key, MASK for
        # a padded one), then the relation biases scatter-added in place at their (src, dst) and
        # (dst, src) cells.
        qa = torch.cat([q * (1.0 / math.sqrt(self.dh)), torch.ones_like(q[:, :, :, :1])], -1)
        ka = torch.cat([k, key_col.to(k.dtype).expand(b, h, n, 1)], -1)
        att = torch.matmul(qa, ka.transpose(-1, -2))
        r = idx_fwd.shape[1]
        vf = (self.rel_bias(rel_type) * rel_w).permute(0, 2, 1).to(att.dtype)          # [B, H, R]
        vr = (self.rel_bias(rel_type + self.n_rel) * rel_w).permute(0, 2, 1).to(att.dtype)
        flat = att.view(b, h, n * n)
        flat.scatter_add_(2, idx_fwd.unsqueeze(1).expand(b, h, r), vf)
        flat.scatter_add_(2, idx_rev.unsqueeze(1).expand(b, h, r), vr)
        att = torch.softmax(att, dim=-1)
        y = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b, n, d)
        x = x + self.out(y)
        return x + self.ff2(Fn.gelu(self.ff1(self.ln2(x))))


class TokenEmbed(nn.Module):
    """A token's numeric, type and categorical parts (the card and ability parts come gathered)."""

    def __init__(self, spec, d: int):
        super().__init__()
        counts = spec.numeric_counts()
        offsets = [0]
        for c in counts[:-1]:
            offsets.append(offsets[-1] + c)
        self.G = sum(counts)
        self.F = spec.F
        self.register_buffer("num_offset", torch.tensor(offsets, dtype=torch.int64), persistent=False)
        self.register_buffer("num_count", torch.tensor(counts, dtype=torch.int64), persistent=False)
        self.register_buffer("slot", torch.arange(spec.F, dtype=torch.int64), persistent=False)
        self.num_proj = nn.Linear(self.G, d, bias=False)
        self.type_emb = nn.Embedding(spec.n_types, d)
        self.cat_emb = nn.Embedding(spec.categorical_size, d, padding_idx=0)
        nn.init.normal_(self.num_proj.weight, std=1.0 / math.sqrt(8.0))
        nn.init.normal_(self.type_emb.weight, std=0.5)
        nn.init.normal_(self.cat_emb.weight, std=0.5)
        with torch.no_grad():
            self.cat_emb.weight[0].zero_()

    def forward(self, tok_type, tok_num, tok_cat):
        b, n = tok_type.shape
        # Each token's numeric slots scattered into its type's columns of a [B, N, G + 1] row (the
        # last column takes the slots a type does not use, always 0, and is dropped).
        col = self.num_offset[tok_type].unsqueeze(-1) + self.slot
        used = self.slot < self.num_count[tok_type].unsqueeze(-1)
        col = torch.where(used, col, torch.full_like(col, self.G))
        dense = torch.zeros(b, n, self.G + 1, dtype=tok_num.dtype, device=tok_num.device)
        dense = dense.scatter_add(2, col, tok_num)
        x = self.num_proj(dense[:, :, :self.G])
        return x + self.type_emb(tok_type) + self.cat_emb(tok_cat).sum(2)


class Encoder(nn.Module):
    """Token embedding + the entity transformer: [B, N, d] token states."""

    def __init__(self, spec, d: int, heads: int, layers: int, ff: int):
        super().__init__()
        self.embed = TokenEmbed(spec, d)
        self.n_rel = spec.n_rel
        self.layers = nn.ModuleList([Layer(d, heads, ff, spec.n_rel) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra):
        """`extra` [B, N, d]: the gathered card and ability parts (plus anything else per token)."""
        x = self.embed(tok_type, tok_num, tok_cat) + extra
        tok_w = tok_mask.to(x.dtype).unsqueeze(-1)
        x = x * tok_w
        b, n = tok_type.shape
        src = rel[:, :, 0]
        dst = rel[:, :, 1]
        idx_fwd = src * n + dst
        idx_rev = dst * n + src
        rel_type = rel[:, :, 2]
        rel_w = rel_mask.to(x.dtype).unsqueeze(-1)
        key_col = (1.0 - tok_mask.to(x.dtype)).view(b, 1, n, 1) * -30000.0
        for layer in self.layers:
            x = layer(x, idx_fwd, idx_rev, rel_type, rel_w, key_col)
        return self.ln_f(x)


class Summary(nn.Module):
    """g [B, d]: the masked mean and an attention pool of the real tokens, mixed by an MLP."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.mix = nn.Linear(2 * d, d)

    def forward(self, x, tok_mask):
        w = tok_mask.to(x.dtype).unsqueeze(-1)
        mean = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        s = self.score(x) + (1.0 - w) * -1e9
        pool = (torch.softmax(s, dim=1) * x).sum(1)
        return Fn.gelu(self.mix(torch.cat([mean, pool], -1)))


class PointerHead(nn.Module):
    """Logits [B, A] over the menu rows' tokens, conditioned on g; padded rows NEG."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(3 * d, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, x, g, act_tok, act_mask):
        b, a = act_tok.shape
        d = x.shape[2]
        acts = x.gather(1, act_tok.unsqueeze(-1).expand(b, a, d))
        ge = g.unsqueeze(1).expand(b, a, d)
        logits = self.fc2(Fn.gelu(self.fc1(torch.cat([acts, ge, acts * ge], -1)))).squeeze(-1).float()
        return torch.where(act_mask, logits, torch.full_like(logits, -1e9))


class ValueHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, g):
        return torch.tanh(self.fc2(Fn.gelu(self.fc1(g)))).squeeze(-1)


def gather_rows(idx, rows, none_row: int):
    """Σ over slots of rows[idx] ([B, N, S] → [B, N, d]), a -1 slot reading the zero row `none_row`."""
    i = torch.where(idx < 0, torch.full_like(idx, none_row), idx)
    return Fn.embedding(i, rows).sum(2)


def slot_vectors(idx, slot):
    """Σ over present slots of the slot's vector ([B, N, S] @ [S, d])."""
    return torch.matmul((idx >= 0).to(slot.dtype), slot)


# ── the learner's model ──────────────────────────────────────────────────────────────────────

class CardRows(nn.Module):
    """The frozen card / text / ability / part tables and their learned projections."""

    def __init__(self, tables, d: int):
        super().__init__()
        t = tables
        self.n_cards = t.card.shape[0]          # N_CARD_TYPES + 1; the last row is the none row
        self.n_abilities = t.ability.shape[0]   # + 1 zero row for -1
        self.register_buffer("card_table", torch.from_numpy(t.card.copy()), persistent=False)
        self.register_buffer("text_table", torch.from_numpy(t.text.copy()), persistent=False)
        self.register_buffer("ability_table", torch.from_numpy(t.ability.copy()), persistent=False)
        self.register_buffer("part_table", torch.from_numpy(t.part.copy()), persistent=False)
        self.register_buffer("part_ability", torch.from_numpy(t.part_ability.copy()), persistent=False)
        real = torch.ones(self.n_cards, 1)
        real[-1] = 0.0
        self.register_buffer("card_real", real, persistent=False)
        part_n = torch.zeros(self.n_abilities, 1).index_add(0, self.part_ability, torch.ones(t.part.shape[0], 1))
        self.register_buffer("part_n", part_n.clamp(min=1.0), persistent=False)
        self.card_proj = nn.Linear(t.card.shape[1], d)
        self.text_proj = nn.Linear(t.text.shape[1], d)
        self.card_id = nn.Embedding(self.n_cards, d)
        nn.init.normal_(self.card_id.weight, std=0.1)
        self.unknown_id = nn.Parameter(torch.zeros(d))
        self.ability_proj = nn.Linear(t.ability.shape[1], d)
        self.part_proj = nn.Linear(t.part.shape[1], d)
        self.card_slot = nn.Parameter(torch.randn(max(t.spec.K, 1), d) * 0.1)
        self.ability_slot = nn.Parameter(torch.randn(max(t.spec.Q, 1), d) * 0.1)

    def card_feature_rows(self):
        """[n_cards, d]: the tables' projection per card id (the none row 0), no id embedding."""
        return (self.card_proj(self.card_table) + self.text_proj(self.text_table)) * self.card_real

    def card_id_rows(self):
        return self.card_id.weight * self.card_real

    def card_rows(self):
        """[n_cards, d]: the full per-card row (features + id embedding), the none row 0."""
        return self.card_feature_rows() + self.card_id_rows()

    def ability_rows(self):
        """[n_abilities + 1, d]: per ability row its projection plus its parts' mean; a zero row last."""
        parts = self.part_proj(self.part_table)
        d = parts.shape[1]
        part_sum = torch.zeros(self.n_abilities, d, dtype=parts.dtype, device=parts.device)
        part_sum = part_sum.index_add(0, self.part_ability, parts)
        rows = self.ability_proj(self.ability_table) + part_sum / self.part_n
        return torch.cat([rows, torch.zeros(1, d, dtype=rows.dtype, device=rows.device)], 0)

    def gather(self, tok_card, tok_ability, id_dropout: float):
        """[B, N, d]: each token's card and ability parts; id-dropout when training."""
        feat = self.card_feature_rows()
        ids = self.card_id_rows()
        none = self.n_cards - 1
        x = gather_rows(tok_card, feat, none) + slot_vectors(tok_card, self.card_slot)
        i = torch.where(tok_card < 0, torch.full_like(tok_card, none), tok_card)
        id_part = Fn.embedding(i, ids)
        if self.training and id_dropout > 0:
            drop = (torch.rand(tok_card.shape, device=tok_card.device) < id_dropout) & (tok_card >= 0)
            id_part = torch.where(drop.unsqueeze(-1), self.unknown_id.to(id_part.dtype), id_part)
        x = x + id_part.sum(2)
        x = x + gather_rows(tok_ability, self.ability_rows(), self.n_abilities) + \
            slot_vectors(tok_ability, self.ability_slot)
        return x


class Critic(nn.Module):
    """The omniscient critic: its own encoder over the omniscient featurization, the deciding seat's
    embedding on every token, a summary and a value head."""

    def __init__(self, spec, cfg):
        super().__init__()
        self.encoder = Encoder(spec, cfg.d, cfg.heads, cfg.critic_layers, cfg.ff_mult * cfg.d)
        self.seat_emb = nn.Embedding(2, cfg.d)
        self.summary = Summary(cfg.d)
        self.value = ValueHead(cfg.d)

    def forward(self, cards, id_dropout: float, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel,
                rel_mask, seat):
        extra = cards.gather(tok_card, tok_ability, id_dropout) + self.seat_emb(seat).unsqueeze(1)
        x = self.encoder(tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra)
        return self.value(self.summary(x, tok_mask))


class RMNet(nn.Module):
    """The learner's model: policy logits, value, auxiliary heads and the optional omniscient critic."""

    def __init__(self, cfg, tables):
        super().__init__()
        spec = tables.spec
        self.cfg = cfg
        self.spec = spec
        self.hashes = dict(tables.hashes)
        d = cfg.d
        if d % cfg.heads:
            raise ValueError("d %d is not a multiple of heads %d" % (d, cfg.heads))
        self.cards = CardRows(tables, d)
        self.encoder = Encoder(spec, d, cfg.heads, cfg.layers, cfg.ff_mult * d)
        self.summary = Summary(d)
        self.pointer = PointerHead(d)
        self.value = ValueHead(d)
        if cfg.aux_heads:
            self.aux_list = nn.Linear(d, spec.n_lists)
            self.aux_archetype = nn.Linear(d, spec.n_archetypes)
            self.aux_hand = nn.Linear(d, d)
            self.aux_library = nn.Linear(d, 2 * spec.library_depth * d)
        self.critic = Critic(spec, cfg) if cfg.omniscient_critic else None

    def forward(self, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok, act_mask,
                omni=None, seat=None):
        """A dict: `logits` [B, A], `value` [B], `summary` [B, d]; with aux heads `aux_list`,
        `aux_archetype`, `aux_hand` [B, n_cards], `aux_own_library`, `aux_opp_library`
        [B, depth, n_cards] (logits; the none row NEG); with the critic and `omni` (the omniscient
        featurization's ten inputs) and `seat` [B] (0 = A), `critic_value` [B]."""
        extra = self.cards.gather(tok_card, tok_ability, self.cfg.id_dropout)
        x = self.encoder(tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra)
        g = self.summary(x, tok_mask)
        out = {"logits": self.pointer(x, g, act_tok, act_mask), "value": self.value(g), "summary": g}
        if self.cfg.aux_heads:
            out.update(self.aux(g))
        if self.critic is not None and omni is not None:
            o = list(omni)
            out["critic_value"] = self.critic(self.cards, self.cfg.id_dropout, o[0], o[1], o[2], o[3], o[4], o[5],
                                              o[6], o[7], seat)
        return out

    def aux(self, g):
        rows = self.cards.card_rows()                               # [n_cards, d]
        none = torch.zeros(rows.shape[0], dtype=torch.float32, device=g.device)   # f32: -1e9 is -inf in f16
        none[-1] = NEG
        b, d = g.shape
        depth = self.spec.library_depth
        lib = self.aux_library(g).view(b, 2, depth, d)
        lib_logits = torch.matmul(lib, rows.t()) / math.sqrt(d) + none
        return {"aux_list": self.aux_list(g), "aux_archetype": self.aux_archetype(g),
                "aux_hand": torch.matmul(self.aux_hand(g), rows.t()) / math.sqrt(d) + none,
                "aux_own_library": lib_logits[:, 0], "aux_opp_library": lib_logits[:, 1]}

    def inference_module(self, version=0):
        """The contract's forward only (InferencePolicy, CPU, eval), the card and ability rows folded."""
        return InferencePolicy(self, version)

    def param_count(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def aux_losses(out, targets):
    """Per auxiliary head its mean loss over the decisions that have a target (`targets` from
    batch.Targets.batch, on the outputs' device): cross-entropy for the list, archetype and each
    library position (-1 skipped), the hand as a distribution over card ids (empty hands skipped)."""
    losses = {}
    losses["list"] = Fn.cross_entropy(out["aux_list"], targets["opp_list"], ignore_index=-1) \
        if (targets["opp_list"] >= 0).any() else out["aux_list"].sum() * 0
    losses["archetype"] = Fn.cross_entropy(out["aux_archetype"], targets["opp_archetype"], ignore_index=-1) \
        if (targets["opp_archetype"] >= 0).any() else out["aux_archetype"].sum() * 0
    hand = targets["opp_hand"]
    n = hand.sum(-1)
    lp = Fn.log_softmax(out["aux_hand"].float(), -1)
    per = -(hand * lp).sum(-1) / n.clamp(min=1.0)
    losses["hand"] = (per * (n > 0)).sum() / (n > 0).sum().clamp(min=1)
    for key in ("own_library", "opp_library"):
        lg = out["aux_" + key]
        t = targets[key]
        losses[key] = Fn.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), t.reshape(-1), ignore_index=-1) \
            if (t >= 0).any() else lg.sum() * 0
    return losses


def masked_log_softmax(logits, act_mask):
    """log π over each decision's real menu rows; -inf at padded rows."""
    lg = torch.where(act_mask, logits, torch.full_like(logits, -float("inf")))
    return torch.log_softmax(lg, dim=-1)


def smoothed_log_policy(logits, act_mask, eps, sideboarding=None, eps_sideboarding=None):
    """log of (1 - ε)·π + ε·uniform-over-legal (R-NaD's smoothed π_0,reg, plan B3), -inf at padded
    rows. `eps` a float or a [B] tensor; with `sideboarding` [B] (bool) and `eps_sideboarding`, the
    sideboarding decisions take that ε instead."""
    logp = masked_log_softmax(logits.float(), act_mask)
    b = logits.shape[0]
    e = torch.as_tensor(eps, dtype=logp.dtype, device=logp.device).expand(b).clone()
    if sideboarding is not None and eps_sideboarding is not None:
        e = torch.where(sideboarding.to(torch.bool), torch.as_tensor(eps_sideboarding, dtype=e.dtype,
                                                                      device=e.device), e)
    e = e.unsqueeze(-1)
    n_legal = act_mask.sum(-1, keepdim=True).to(logp.dtype)
    out = torch.logaddexp(torch.log1p(-e) + logp, torch.log(e) - torch.log(n_legal))
    return torch.where(act_mask, out, torch.full_like(out, -float("inf")))


# ── the exported policy ──────────────────────────────────────────────────────────────────────

class InferencePolicy(nn.Module):
    """forward(ten contract inputs) -> (logits [B, A], value [B]); the card and ability rows folded
    into buffers. Carries spec_hash, vocab_hash, text_model and policy_version as string attributes
    (what export.py writes into each format's metadata)."""

    def __init__(self, model, version=0):
        super().__init__()
        self.spec_hash = str(model.hashes["spec_hash"])
        self.vocab_hash = str(model.hashes["vocab_hash"])
        self.text_model = str(model.hashes["text_model"])
        self.policy_version = str(int(version))
        self.F, self.C, self.K, self.Q = int(model.spec.F), int(model.spec.C), int(model.spec.K), int(model.spec.Q)
        src = model
        self.encoder = copy.deepcopy(src.encoder).cpu().float()
        self.summary = copy.deepcopy(src.summary).cpu().float()
        self.pointer = copy.deepcopy(src.pointer).cpu().float()
        self.value = copy.deepcopy(src.value).cpu().float()
        cards = src.cards
        with torch.no_grad():
            was = cards.training
            cards.eval()
            card_rows = cards.card_rows().detach().float().cpu().clone()
            ability_rows = cards.ability_rows().detach().float().cpu().clone()
            cards.train(was)
        self.register_buffer("card_rows", card_rows)
        self.register_buffer("ability_rows", ability_rows)
        self.register_buffer("card_slot", cards.card_slot.detach().float().cpu().clone())
        self.register_buffer("ability_slot", cards.ability_slot.detach().float().cpu().clone())
        self.none_card = int(card_rows.shape[0] - 1)
        self.none_ability = int(ability_rows.shape[0] - 1)
        self.eval()

    def forward(self, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok,
                act_mask):
        extra = gather_rows(tok_card, self.card_rows, self.none_card) + slot_vectors(tok_card, self.card_slot)
        extra = extra + gather_rows(tok_ability, self.ability_rows, self.none_ability) + \
            slot_vectors(tok_ability, self.ability_slot)
        x = self.encoder(tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra)
        g = self.summary(x, tok_mask)
        return self.pointer(x, g, act_tok, act_mask), self.value(g)


def save_checkpoint(model, path, extra=None):
    """Saves `model`'s config, hashes and state_dict (and `extra`) to `path` (torch.save)."""
    import os
    data = {"config": model.cfg.to_dict(), "hashes": dict(model.hashes),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        data["extra"] = extra
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, tables):
    """The RMNet a save_checkpoint file holds, built over `tables` (whose hashes must be the model's)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    for k in ("spec_hash", "vocab_hash", "text_model"):
        if data["hashes"].get(k) != tables.hashes[k]:
            raise ValueError("%s: the checkpoint's %s %s is not the tables' %s"
                             % (path, k, data["hashes"].get(k), tables.hashes[k]))
    model = RMNet(ModelConfig.from_dict(data["config"]), tables)
    model.load_state_dict(data["state_dict"])
    return model
