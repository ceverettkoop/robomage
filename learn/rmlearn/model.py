"""The learner's network (docs/rnad_training_plan.md §4 L2) on the step-5 network contract.

    cfg = ModelConfig(d=256, layers=4)
    model = RMNet(cfg, tables)                     # tables: spec.load_tables(...)
    out = model(*inputs)                            # dict: logits [B,A], value [B], summary, aux_*
    out = model(*inputs, omni=omni_inputs, seat=seat)   # + critic_value (cfg.omniscient_critic)
    policy = model.inference_module(version=3)     # InferencePolicy: the contract's forward only

**Inputs** are the contract's ten padded tensors (docs/featurizer.md §6; ``batch.pad``):
``tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok, act_mask``.

**Architecture.** A token is the sum of
- its numeric slots through one linear map over the spec's ``G`` columns (each type's slots scattered
  into its own block of the layout registry, so the map is per type without a per-token weight
  gather);
- its type's embedding and the sum of its categorical slots' embeddings (one table over the global
  categorical id space, id 0 the pad);
- per card slot the card's row (frozen card-table and text-table rows projected, plus a learned
  card-id embedding; during training the id part is replaced, with probability ``id_dropout`` per
  slot, by a learned unknown-card vector; a card no training batch has held, ``card_seen``, reads
  that vector in place of its id row everywhere, so an unseen card plays the path id-dropout
  trains) plus a per-slot vector;
- per ability slot the ability row (frozen ability-table row projected, its parts' projections
  averaged in) plus a per-slot vector.
An entity transformer (pre-LayerNorm, ``layers`` blocks) whose attention adds, per head and layer,
a learned bias for each relation type at (src, dst) and another at (dst, src), scatter-added into
the attention scores; padded tokens are masked as keys. A summary ``g`` (the masked mean and an
attention pool of the real tokens, mixed by an MLP) conditions a pointer head over the menu rows'
tokens (an MLP of the row's token, ``g`` and their product; padded rows -1e9) and the value head
(an MLP of ``g``, tanh).

**Auxiliary heads** (training only, never exported) predict the learner targets from ``g``: the
opponent's hand as a distribution over card ids and each seat's library tops (``library_depth``
positions) as card ids. The card-id heads score
the same card rows the tokens read (a dot product), so they shape the card representation (all but
the unknown-card vector, which they read detached).

**Omniscient critic** (``omniscient_critic``; the plan's open issue 1): a second encoder of the
same design over the omniscient featurization of the same decision, with the deciding seat's
embedding added to every token (the omniscient frame is A-relative), and its own value head
(``critic_value``). It shares the card and ability rows with the policy.

**Cost on the GPU** (the RX 6700 XT, rocBLAS; profiled, docs/rnad_training_plan.md §4 L4): a token's
card, ability and slot parts are one ``embedding_bag`` over a per-batch table (``token_table``: per
card slot every card's row plus the slot's vector; under id-dropout a second copy with the
unknown-card vector), its categorical and type parts another (bag gradients rather than one sorted
embedding gradient per table); every layer's relation biases are one gather; with gradients the
scores are one GEMM onto the additive bias (key mask + relation biases), without them (inference, the
R-NaD target network) the key mask rides an extra q·k column (padded to a width of 16) and the
relation biases are added in place; the softmax stays in the scores' precision; the token-wise
linear maps (``Linear``) compute their weight gradient in the orientation rocBLAS tiles well and the
width-1 maps are products and sums. All of it is the same function as the plain formulation.

Export-friendly: no control flow that depends on the data; gathers, scatter-adds and masks only.
``InferencePolicy`` holds the card and ability rows folded into a buffer (constants of the weights;
each row once, the slot vectors added at lookup) and the four metadata strings; ``export.py`` writes it as TorchScript and AOTInductor ``.pt2``.
"""

import copy
import math
from dataclasses import asdict, dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from . import spec as S

NEG = -1e9


@dataclass
class ModelConfig:
    d: int = 192              # token width
    heads: int = 4
    layers: int = 4
    ff_mult: int = 4          # feed-forward width = ff_mult * d
    id_dropout: float = 0.1   # probability a card slot's id embedding is replaced by the unknown-card vector
    #                           (the path unseen cards read; docs/open_vocab_plan.md S4, a training lever)
    aux_heads: bool = True    # hidden-information heads (training only)
    omniscient_critic: bool = False
    critic_layers: int = 2    # the omniscient critic's encoder depth

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── blocks ────────────────────────────────────────────────────────────────────────────────────

class _LinearFn(torch.autograd.Function):
    """y = x Wᵀ + b, its weight gradient computed as whichever of (gᵀ x) and (xᵀ g)ᵀ has the narrow
    side first: rocBLAS picks a tiny-tile kernel for the other orientation when the token count (the
    summed dimension) is large, several times slower. Autocast's casts are made here, once."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        dev = x.device.type
        if torch.is_autocast_enabled(dev):
            dt = torch.get_autocast_dtype(dev)
            x, weight = x.to(dt), weight.to(dt)
            bias = bias.to(dt) if bias is not None else None
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        with torch.autocast(dev, enabled=False):
            return Fn.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1]).to(x.dtype)
        gx = gw = gb = None
        if ctx.needs_input_grad[0]:
            gx = torch.mm(g2, w).view(x.shape)
        if ctx.needs_input_grad[1]:
            x2 = x.reshape(-1, x.shape[-1])
            gw = torch.mm(x2.t(), g2).t() if w.shape[0] >= w.shape[1] else torch.mm(g2.t(), x2)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            gb = g2.sum(0)
        return gx, gw, gb


class Linear(nn.Linear):
    """``nn.Linear`` (same parameters) whose training-mode backward is ``_LinearFn``'s."""

    def forward(self, x):
        if self.training and not torch.jit.is_scripting():
            return self._train_forward(x)
        return Fn.linear(x, self.weight, self.bias)

    @torch.jit.unused
    def _train_forward(self, x):
        return _LinearFn.apply(x, self.weight, self.bias)


class Layer(nn.Module):
    """A pre-LayerNorm transformer block with per-head relation biases."""

    def __init__(self, d: int, heads: int, ff: int, n_rel: int):
        super().__init__()
        self.heads = heads
        self.dh = d // heads
        self.scale = 1.0 / math.sqrt(self.dh)
        self.ext = (self.dh + 16) // 16 * 16     # the head width plus the key-mask column, padded
        self.n_rel = n_rel
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = Linear(d, 3 * d)
        self.out = Linear(d, d)
        self.ff1 = Linear(d, ff)
        self.ff2 = Linear(ff, d)
        # Per relation type a bias per head at (src, dst) (rows 0..n_rel-1) and at (dst, src) (rows n_rel..).
        self.rel_bias = nn.Embedding(2 * max(n_rel, 1), heads)
        nn.init.zeros_(self.rel_bias.weight)

    def forward(self, x, rel_idx, rel_vals, key_bias):
        """`rel_idx` [B, 2R]: the flat score cell (row · N + column) of each relation's bias;
        `rel_vals` [B, 2R, H]: this layer's bias there (0 for a padded relation); `key_bias`
        [B, 1, 1, N]: 0 at a real key, MASK at a padded one."""
        b, n, d = x.shape
        h = self.heads
        dh = self.dh
        qkv = self.qkv(self.ln1(x)).view(b, n, 3, h, dh).permute(2, 0, 3, 1, 4)
        r2 = rel_idx.shape[1]
        if torch.is_grad_enabled():
            # Heads folded into the GEMM batch; the scores as one GEMM onto the additive bias (in the
            # scores' precision): the key mask broadcast with the relation biases scatter-added at
            # their cells (into a copy, not a view, so autograd records no slice copy).
            qkv = qkv.reshape(3, b * h, n, dh)
            q, k, v = qkv[0], qkv[1], qkv[2]
            bias = key_bias.to(q.dtype).expand(b, h, n, n).reshape(b * h, n * n)
            bias.scatter_add_(1, rel_idx.repeat_interleave(h, 0),
                              rel_vals.to(q.dtype).permute(0, 2, 1).reshape(b * h, r2))
            att = torch.baddbmm(bias.view(b * h, n, n), q, k.transpose(1, 2), alpha=self.scale)
            # In the scores' precision (the kernel accumulates in f32), not autocast's f32 copy.
            att = torch._softmax(att, -1, False)
            y = torch.bmm(att, v).view(b, h, n, dh)
        else:
            # Without gradients (inference, exports) no pass over the scores but the GEMM's: the key
            # mask through one more q·k dimension (1 · the key's mask value; zero columns pad the width
            # to a multiple of 16, which rocBLAS tiles well), the relation biases added in place.
            q, k, v = qkv[0], qkv[1], qkv[2]
            pad = torch.zeros(b, h, n, self.ext - dh - 1, dtype=q.dtype, device=q.device)
            qa = torch.cat([q * self.scale, torch.ones_like(q[:, :, :, :1]), pad], -1)
            ka = torch.cat([k, key_bias.to(k.dtype).view(b, 1, n, 1).expand(b, h, n, 1), pad], -1)
            att = torch.matmul(qa, ka.transpose(-1, -2))
            att.view(b, h, n * n).scatter_add_(2, rel_idx.unsqueeze(1).expand(b, h, r2),
                                               rel_vals.to(att.dtype).permute(0, 2, 1))
            att = torch._softmax(att, -1, False)
            y = torch.matmul(att, v)
        y = y.transpose(1, 2).reshape(b, n, d)
        x = x + self.out(y)
        return x + self.ff2(Fn.gelu(self.ff1(self.ln2(x))))


class TokenEmbed(nn.Module):
    """A token's numeric, type and categorical parts (the card and ability parts come gathered)."""

    def __init__(self, spec, d: int):
        super().__init__()
        offsets, counts = spec.numeric_layout()
        self.G = int(spec.G)
        self.F = spec.F
        self.cat_size = int(spec.categorical_size)
        self.register_buffer("num_offset", torch.tensor(offsets, dtype=torch.int64), persistent=False)
        self.register_buffer("num_count", torch.tensor(counts, dtype=torch.int64), persistent=False)
        self.register_buffer("slot", torch.arange(spec.F, dtype=torch.int64), persistent=False)
        self.num_proj = Linear(self.G, d, bias=False)
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
        # The categorical slots' rows and the type's row in one bag (id 0, the pad, skipped).
        table = torch.cat([self.cat_emb.weight, self.type_emb.weight], 0)
        idx = torch.cat([tok_cat, (tok_type + self.cat_size).unsqueeze(-1)], 2).view(b * n, -1)
        return x + Fn.embedding_bag(idx, table, mode="sum", padding_idx=0).view(b, n, -1)


class Encoder(nn.Module):
    """Token embedding + the entity transformer: [B, N, d] token states."""

    def __init__(self, spec, d: int, heads: int, layers: int, ff: int):
        super().__init__()
        self.embed = TokenEmbed(spec, d)
        self.n_rel = spec.n_rel
        self.heads = heads
        self.layers = nn.ModuleList([Layer(d, heads, ff, spec.n_rel) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra):
        """`extra` [B, N, d]: the gathered card and ability parts (plus anything else per token)."""
        x = self.embed(tok_type, tok_num, tok_cat) + extra
        x = x * tok_mask.to(x.dtype).unsqueeze(-1)
        b, n = tok_type.shape
        src = rel[:, :, 0]
        dst = rel[:, :, 1]
        rel_type = rel[:, :, 2]
        # Each relation's bias row at (src, dst) and its reverse row at (dst, src), every layer's at
        # once from the layers' bias tables side by side (zero for a padded relation).
        rel_idx = torch.cat([src * n + dst, dst * n + src], 1)
        rows = torch.cat([rel_type, rel_type + self.n_rel], 1)
        tables: List[torch.Tensor] = []
        for layer in self.layers:
            tables.append(layer.rel_bias.weight)
        rel_vals = Fn.embedding(rows, torch.cat(tables, 1)) * rel_mask.repeat(1, 2).unsqueeze(-1).to(x.dtype)
        key_bias = ((~tok_mask).to(x.dtype) * -30000.0).view(b, 1, 1, n)
        h = self.heads
        i = 0
        for layer in self.layers:
            x = layer(x, rel_idx, rel_vals[:, :, i * h:(i + 1) * h], key_bias)
            i += 1
        return self.ln_f(x)


class Summary(nn.Module):
    """g [B, d]: the masked mean and an attention pool of the real tokens, mixed by an MLP."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.mix = Linear(2 * d, d)

    def forward(self, x, tok_mask):
        w = tok_mask.to(x.dtype).unsqueeze(-1)
        mean = (x * w).sum(1) / w.sum(1).clamp(min=1.0)
        # The width-1 linear maps as products and sums (a GEMM with one output column tiles badly).
        s = (x * self.score.weight[0].to(x.dtype)).sum(-1, keepdim=True) + self.score.bias + (1.0 - w) * -1e9
        pool = (torch.softmax(s, dim=1) * x).sum(1)
        return Fn.gelu(self.mix(torch.cat([mean, pool], -1)))


class PointerHead(nn.Module):
    """Logits [B, A] over the menu rows' tokens, conditioned on g; padded rows NEG."""

    def __init__(self, d: int):
        super().__init__()
        self.fc1 = Linear(3 * d, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, x, g, act_tok, act_mask):
        b, a = act_tok.shape
        d = x.shape[2]
        acts = x.gather(1, act_tok.unsqueeze(-1).expand(b, a, d))
        ge = g.unsqueeze(1).expand(b, a, d)
        hid = Fn.gelu(self.fc1(torch.cat([acts, ge, acts * ge], -1)))
        logits = (hid * self.fc2.weight[0].to(hid.dtype)).sum(-1, dtype=torch.float32) + self.fc2.bias.float()
        return torch.where(act_mask, logits, torch.full_like(logits, -1e9))


class ValueHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, 1)

    def forward(self, g):
        return torch.tanh(self.fc2(Fn.gelu(self.fc1(g)))).squeeze(-1)


def gather_tokens(card_idx, tok_card, tok_ability, table, card_rows: int, n_abilities: int):
    """[B, N, d]: per token the sum over its present card slots of the card's row plus the slot's
    vector, and over its present ability slots of the ability's row plus the slot's vector, as one
    bag over `table` (``CardRows.token_table``: per card slot `card_rows` rows, then per ability slot
    `n_abilities` rows, a zero row last). `card_idx` is each card slot's row (`tok_card`, or a
    substitute row), read where `tok_card` ≥ 0; a -1 slot adds nothing."""
    b, n, k = tok_card.shape
    q = tok_ability.shape[2]
    pad = table.shape[0] - 1
    card_off = torch.arange(k, dtype=tok_card.dtype, device=tok_card.device) * card_rows
    ability_off = torch.arange(q, dtype=tok_ability.dtype, device=tok_ability.device) * n_abilities + k * card_rows
    idx = torch.cat([torch.where(tok_card >= 0, card_idx + card_off, torch.full_like(tok_card, pad)),
                     torch.where(tok_ability >= 0, tok_ability + ability_off, torch.full_like(tok_ability, pad))], 2)
    return Fn.embedding_bag(idx.view(b * n, -1), table, mode="sum", padding_idx=pad).view(b, n, -1)


# ── the learner's model ──────────────────────────────────────────────────────────────────────

class CardRows(nn.Module):
    """The frozen card / text / ability / part tables and their learned projections.

    **Seen cards** (docs/open_vocab_plan.md S4). ``card_seen`` [n_cards] (bool, saved in the
    state_dict) marks the card ids some training batch has held (``mark_seen``, called by the
    training loops on each batch before its forward). A seen card's id part is its ``card_id`` row;
    an unseen card's is ``unknown_id``, the vector id-dropout trains, in the training rows, the aux
    heads' rows and the exports' folded rows alike. A card's first marking copies ``unknown_id`` into
    its id row, so it goes on from the path it played unseen; until then its row is zero (never
    read). A state_dict without ``card_seen`` (a network made before the mask) is refused: such a
    network read every id row as a card's, and its rows do not tell which cards it trained on (the aux
    heads' softmax moved all of them), so neither "all seen" nor "none seen" is its function;
    ``set_seen`` (S5's migration) or ``load_checkpoint(seen=...)`` states its seen set."""

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
        nn.init.zeros_(self.card_id.weight)
        self.unknown_id = nn.Parameter(torch.zeros(d))
        self.register_buffer("card_seen", torch.zeros(self.n_cards, dtype=torch.bool))
        self.ability_proj = nn.Linear(t.ability.shape[1], d)
        self.part_proj = nn.Linear(t.part.shape[1], d)
        self.card_slot = nn.Parameter(torch.randn(max(t.spec.K, 1), d) * 0.1)
        self.ability_slot = nn.Parameter(torch.randn(max(t.spec.Q, 1), d) * 0.1)

    def card_feature_rows(self):
        """[n_cards, d]: the tables' projection per card id (the none row 0), no id embedding."""
        return (self.card_proj(self.card_table) + self.text_proj(self.text_table)) * self.card_real

    def card_id_rows(self, train_unknown: bool = True):
        """[n_cards, d]: per card id its id part — its ``card_id`` row when seen, else ``unknown_id``
        (detached unless `train_unknown`) — the none row 0. (A blend by a fresh float copy of the mask,
        not a ``where`` on the buffer, so marking after a forward never invalidates that forward's
        backward.)"""
        seen = self.card_seen.unsqueeze(1).to(self.card_id.weight.dtype)
        unknown = self.unknown_id if train_unknown else self.unknown_id.detach()
        return (self.card_id.weight * seen + unknown * (1.0 - seen)) * self.card_real

    def card_rows(self, train_unknown: bool = True):
        """[n_cards, d]: the full per-card row (features + id part, ``card_id_rows``), the none row 0."""
        return self.card_feature_rows() + self.card_id_rows(train_unknown)

    @torch.no_grad()
    def mark_seen(self, card_ids):
        """Marks every card id in `card_ids` (a tensor or array of any shape; negative entries, the
        absent slots, skipped) seen. A newly seen card's id row is set to ``unknown_id``, so its row is
        unchanged at the marking and trains from there. No host synchronisation."""
        ids = torch.as_tensor(card_ids, device=self.card_seen.device).reshape(-1).long()
        ids = torch.where(ids >= 0, ids, torch.full_like(ids, self.n_cards - 1))
        hit = torch.zeros_like(self.card_seen).index_fill_(0, ids, True)
        hit[-1] = False                                         # the none row is never a card
        w = self.card_id.weight
        new = (hit & ~self.card_seen).unsqueeze(1)
        w.copy_(torch.where(new, self.unknown_id.to(w.dtype).expand_as(w), w))
        self.card_seen.logical_or_(hit)

    @torch.no_grad()
    def set_seen(self, card_ids):
        """Sets the seen mask to exactly the ids in `card_ids` (any iterable or array of ids) and zeroes
        every other card's id row (never read; a later first marking sets it). For a network trained
        before the mask existed: the ids its training actually saw (S5's migration)."""
        ids = torch.as_tensor(list(card_ids) if not torch.is_tensor(card_ids) else card_ids,
                              dtype=torch.int64).reshape(-1)
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= self.n_cards - 1):
            raise ValueError("set_seen: card ids lie in [0, %d)" % (self.n_cards - 1))
        mask = torch.zeros(self.n_cards, dtype=torch.bool)
        mask[ids] = True
        self.card_seen.copy_(mask.to(self.card_seen.device))
        w = self.card_id.weight
        w.mul_(self.card_seen.unsqueeze(1).to(w.dtype))

    def seen_ids(self):
        """The seen card ids, ascending (a list)."""
        return torch.nonzero(self.card_seen).squeeze(1).cpu().tolist()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        if prefix + "card_seen" not in state_dict:
            error_msgs.append("%scard_seen is missing: the network predates the seen-card mask "
                              "(docs/open_vocab_plan.md S4), and its card_id rows do not tell which cards it saw; "
                              "migrate it (S5) or state the ids its training saw (load_checkpoint(seen=...), "
                              "CardRows.set_seen)" % prefix)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def ability_rows(self):
        """[n_abilities + 1, d]: per ability row its projection plus its parts' mean; a zero row last."""
        parts = self.part_proj(self.part_table)
        d = parts.shape[1]
        part_sum = torch.zeros(self.n_abilities, d, dtype=parts.dtype, device=parts.device)
        part_sum = part_sum.index_add(0, self.part_ability, parts)
        rows = self.ability_proj(self.ability_table) + part_sum / self.part_n
        return torch.cat([rows, torch.zeros(1, d, dtype=rows.dtype, device=rows.device)], 0)

    def token_table(self, dropout_rows: bool = False):
        """[rows, d] f32, ``gather_tokens``' table: per card slot the card rows (``card_rows``: an
        unseen card's with the unknown-card vector) plus the slot's vector (with `dropout_rows` followed
        by the same with the unknown-card vector in place of every id part), per ability slot the
        ability rows plus the slot's vector, and a zero row."""
        feat = self.card_feature_rows().float()
        cards = feat + self.card_id_rows()
        if dropout_rows:
            cards = torch.cat([cards, feat + self.unknown_id], 0)
        d = feat.shape[1]
        abilities = self.ability_rows()[:-1].float()
        return torch.cat([(cards.unsqueeze(0) + self.card_slot.unsqueeze(1)).reshape(-1, d),
                          (abilities.unsqueeze(0) + self.ability_slot.unsqueeze(1)).reshape(-1, d),
                          torch.zeros(1, d, dtype=feat.dtype, device=feat.device)], 0)

    def gather(self, tok_card, tok_ability, id_dropout: float):
        """[B, N, d]: each token's card and ability parts; id-dropout when training (a dropped slot
        reads its card's unknown-id row)."""
        drop_on = self.training and id_dropout > 0
        card_idx = tok_card
        if drop_on:
            drop = (torch.rand(tok_card.shape, device=tok_card.device) < id_dropout) & (tok_card >= 0)
            card_idx = tok_card + drop.to(tok_card.dtype) * self.n_cards
        card_rows = 2 * self.n_cards if drop_on else self.n_cards
        return gather_tokens(card_idx, tok_card, tok_ability, self.token_table(drop_on), card_rows, self.n_abilities)


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
            self.aux_hand = nn.Linear(d, d)
            self.aux_library = nn.Linear(d, 2 * spec.library_depth * d)
        self.critic = Critic(spec, cfg) if cfg.omniscient_critic else None

    def forward(self, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok, act_mask,
                omni=None, seat=None, aux=True):
        """A dict: `logits` [B, A], `value` [B], `summary` [B, d]; with aux heads (and `aux`)
        `aux_hand` [B, n_cards], `aux_own_library`, `aux_opp_library` [B, depth, n_cards] (logits;
        the none row NEG); with the critic and `omni` (the omniscient featurization's ten inputs)
        and `seat` [B] (0 = A), `critic_value` [B]. ``aux=False`` skips the aux heads (R-NaD reads
        none of them)."""
        extra = self.cards.gather(tok_card, tok_ability, self.cfg.id_dropout)
        x = self.encoder(tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra)
        g = self.summary(x, tok_mask)
        out = {"logits": self.pointer(x, g, act_tok, act_mask), "value": self.value(g), "summary": g}
        if self.cfg.aux_heads and aux:
            out.update(self.aux(g))
        if self.critic is not None and omni is not None:
            o = list(omni)
            out["critic_value"] = self.critic(self.cards, self.cfg.id_dropout, o[0], o[1], o[2], o[3], o[4], o[5],
                                              o[6], o[7], seat)
        return out

    def aux(self, g):
        # Every id is a candidate, so the unseen ones' rows (unknown_id each) would train the unknown-card
        # vector toward "unlikely" from hundreds of unseen and empty ids: it enters detached here, and
        # only the tokens' path (id-dropout) shapes it.
        rows = self.cards.card_rows(train_unknown=False)            # [n_cards, d]
        none = torch.zeros(rows.shape[0], dtype=torch.float32, device=g.device)   # f32: -1e9 is -inf in f16
        none[-1] = NEG
        b, d = g.shape
        depth = self.spec.library_depth
        lib = self.aux_library(g).view(b, 2, depth, d)
        lib_logits = torch.matmul(lib, rows.t()) / math.sqrt(d) + none
        return {"aux_hand": torch.matmul(self.aux_hand(g), rows.t()) / math.sqrt(d) + none,
                "aux_own_library": lib_logits[:, 0], "aux_opp_library": lib_logits[:, 1]}

    def inference_module(self, version=0):
        """The contract's forward only (InferencePolicy, CPU, eval), the card and ability rows folded."""
        return InferencePolicy(self, version)

    def param_count(self, trainable_only=True):
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


def aux_losses(out, targets):
    """Per auxiliary head its mean loss over the decisions that have a target (`targets` from
    batch.Targets.batch, on the outputs' device): cross-entropy for each library position (-1
    skipped), the hand as a distribution over card ids (empty hands skipped)."""
    losses = {}
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
    sideboarding decisions take that ε instead. An ε of 0 gives the raw log π; the padded rows are
    taken out before the mix, so its gradient stays finite there too."""
    logp = masked_log_softmax(logits.float(), act_mask).masked_fill(~act_mask, 0.0)
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
    into one buffer (``folded_rows``). Carries spec_hash, vocab_hash, text_model, layout_hash, vocab
    (the vocab manifest's JSON text) and policy_version as string attributes (what export.py writes
    into each format's metadata; also tables_digest): the model manifest of the tables it was folded from, which an
    export must match exactly (spec.check_compatible, exact).

    ``folded_rows`` holds each card's row once, each distinct ability row once (every ability
    address no ability has — no features, no parts — shares one row: they project alike) and the
    per-slot vectors, not a copy of the rows per slot as the training model's ``token_table``: a
    token's card slot k adds its card's row and slot vector k, an ability slot its row (through
    ``ability_index``) and its slot vector, as one bag (``gather_folded``). The same function as the
    training model's, the sums in another order."""

    def __init__(self, model, version=0):
        super().__init__()
        self.spec_hash = str(model.hashes["spec_hash"])
        self.vocab_hash = str(model.hashes["vocab_hash"])
        self.text_model = str(model.hashes["text_model"])
        self.layout_hash = str(model.hashes.get("layout_hash", ""))
        self.tables_digest = str(model.hashes.get("tables_digest", ""))
        self.vocab = S.vocab_text(model.hashes.get("vocab", []))
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
            folded, ability_index = _fold_rows(cards)
            cards.train(was)
        self.register_buffer("folded_rows", folded)
        self.register_buffer("ability_index", ability_index)
        self.n_cards = int(cards.n_cards)
        self.n_abilities = int(cards.n_abilities)
        self.slot_at = int(folded.shape[0] - 1 - self.K - self.Q)
        self.none_card = int(cards.n_cards - 1)
        self.none_ability = int(cards.n_abilities)
        self.eval()

    def forward(self, tok_type, tok_num, tok_cat, tok_card, tok_ability, tok_mask, rel, rel_mask, act_tok,
                act_mask):
        extra = gather_folded(tok_card, tok_ability, self.folded_rows, self.ability_index, self.slot_at, self.K)
        x = self.encoder(tok_type, tok_num, tok_cat, tok_mask, rel, rel_mask, extra)
        g = self.summary(x, tok_mask)
        return self.pointer(x, g, act_tok, act_mask), self.value(g)


def _fold_rows(cards):
    """(``folded_rows`` [n_cards + distinct ability rows + K + Q + 1, d] f32 on the CPU: the card
    rows (``card_rows``: an unseen card's features plus ``unknown_id``), the distinct ability rows,
    the card slots' and the ability slots' vectors, a zero row; ``ability_index`` [n_abilities] i64: per ability row its row in ``folded_rows``)."""
    card_rows = cards.card_rows().detach().float().cpu()
    ability_rows = cards.ability_rows()[:-1].detach().float().cpu()
    has_parts = torch.zeros(cards.n_abilities, dtype=torch.bool)
    has_parts[cards.part_ability.cpu()] = True
    used = (cards.ability_table.cpu() != 0).any(1) | has_parts
    keep = torch.nonzero(used).squeeze(1)
    empty = torch.nonzero(~used).squeeze(1)
    if empty.numel():
        keep = torch.cat([keep, empty[:1]])
    position = torch.full((cards.n_abilities,), keep.numel() - 1, dtype=torch.int64)
    position[keep] = torch.arange(keep.numel(), dtype=torch.int64)
    d = card_rows.shape[1]
    folded = torch.cat([card_rows, ability_rows[keep], cards.card_slot.detach().float().cpu(),
                        cards.ability_slot.detach().float().cpu(), torch.zeros(1, d)], 0).clone()
    return folded, position + card_rows.shape[0]


def gather_folded(tok_card, tok_ability, table, ability_index, slot_at: int, n_card_slots: int):
    """[B, N, d]: ``gather_tokens``' sums over ``InferencePolicy.folded_rows``: per present card slot
    k the card's row and the slot vector at `slot_at` + k, per present ability slot q the ability's
    row (`ability_index`) and the slot vector at `slot_at` + `n_card_slots` + q; a -1 slot adds
    nothing (the zero row last is the bag's padding)."""
    b, n, k = tok_card.shape
    q = tok_ability.shape[2]
    pad = table.shape[0] - 1
    card_slot = torch.arange(k, dtype=tok_card.dtype, device=tok_card.device) + slot_at
    ability_slot = torch.arange(q, dtype=tok_ability.dtype, device=tok_ability.device) + slot_at + n_card_slots
    c_on, a_on = tok_card >= 0, tok_ability >= 0
    padded = torch.full_like(tok_card, pad)
    cards = torch.stack([torch.where(c_on, tok_card, padded),
                         torch.where(c_on, card_slot.expand(b, n, k), padded)], 3).view(b, n, 2 * k)
    padded = torch.full_like(tok_ability, pad)
    rows = ability_index[tok_ability.clamp(min=0)]
    abilities = torch.stack([torch.where(a_on, rows, padded),
                             torch.where(a_on, ability_slot.expand(b, n, q), padded)], 3).view(b, n, 2 * q)
    idx = torch.cat([cards, abilities], 2)
    return Fn.embedding_bag(idx.view(b * n, -1), table, mode="sum", padding_idx=pad).view(b, n, -1)


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


def load_checkpoint(path, tables, seen=None):
    """The RMNet a save_checkpoint file holds, built over `tables` (the checkpoint's manifest must be
    compatible with theirs, spec.check_compatible: the vocab may have grown). The model carries the
    tables' manifest from then on. A file without ``cards.card_seen`` (made before the seen-card mask)
    is refused unless `seen` gives the card ids its training saw (``CardRows.set_seen``); `seen` is
    refused for a file that has the mask."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    S.require_compatible(data["hashes"], tables, path)
    model = RMNet(ModelConfig.from_dict(data["config"]), tables)
    sd = data["state_dict"]
    if seen is not None:
        if "cards.card_seen" in sd:
            raise ValueError("%s carries its seen-card mask; seen= is for a checkpoint without one" % path)
        sd = dict(sd)
        sd["cards.card_seen"] = model.cards.card_seen
    model.load_state_dict(sd)
    if seen is not None:
        model.cards.set_seen(seen)
    return model
