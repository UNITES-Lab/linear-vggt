"""Register-guided sparse attention: anchor + top-K frame attention with token eviction.

Every query frame t reads: frame 0 (anchor), up to K further historical frames, and itself. All tokens of the
chosen frames form one renormalized softmax.
  reg    - frames ranked by Reg->Reg routing mass: current register queries attend (one softmax) to the register
           keys of frames 0..t; source-register mass is summed per frame and averaged over query registers and heads.
  dense  - all historical frames (equals the teacher's frame-causal attention; reference for tests and analysis)
Selection is a hard, non-differentiable choice; Q/K/V/proj still receive gradients through the attention over the
selected frames.

The module wraps (and reuses the parameters of) a teacher `Attention`. Two paths give the same result: a
full-sequence path for training/batched inference (each query frame gathers its selected frames, so memory is
linear in the frame count), and a one-frame cached path for streaming.

Token eviction (evict_budget, default None = off; the token indexer of the paper): when a frame enters the history
it keeps only its camera/register tokens and the round(evict_budget x patches) patch tokens that its own queries
attend to most (softmax within the frame, summed over heads and queries). Frame 0 (anchor) and the query frame
itself stay whole. Routing reads only register keys, which are always kept, so the routed frames do not change.
The choice of tokens is hard (no gradient); training with eviction lets Q/K/V adapt to it. The full-sequence,
segment (forward_evicted with a carried state) and cached paths give the same result.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

POLICIES = ("reg", "dense")


class AnchorTopKAttention(nn.Module):
    def __init__(self, attention, policy="reg", top_k=2, num_registers=4, evict_budget=None):
        super().__init__()
        if policy not in POLICIES or top_k < 0 or num_registers < 1:
            raise ValueError("Invalid policy/budget/register count")
        if evict_budget is not None and not 0 < evict_budget <= 1:
            raise ValueError("evict_budget must be in (0, 1]")
        self.base = attention
        self.policy = policy
        self.top_k = top_k
        self.num_registers = num_registers
        self.evict_budget = evict_budget
        self.last_selected = None

    def extra_repr(self):
        evict = "" if self.evict_budget is None else f", evict_budget={self.evict_budget}"
        return f"policy={self.policy}, top_k={self.top_k}, num_registers={self.num_registers}{evict}"

    def select(self, mass, history):
        """mass: [B, >=history] routing mass. Returns sorted historical frame ids [B, count]."""
        b, device = mass.shape[0], mass.device
        if history == 0:
            return torch.zeros(b, 0, device=device, dtype=torch.long)
        if self.policy == "dense":
            return torch.arange(history, device=device).expand(b, -1)
        count = min(self.top_k, history - 1)     # frames 1..history-1 compete for the top-k picks
        extra = torch.argsort(mass[:, 1:history], dim=-1, descending=True, stable=True)[:, :count] + 1
        anchor = torch.zeros(b, 1, device=device, dtype=torch.long)
        return torch.cat((anchor, extra), dim=1).sort(dim=-1).values

    def _qkv(self, x, pos):
        a = self.base
        b, n, _ = x.shape
        q, k, v = a.qkv(x).reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        if a.rope is not None:
            if pos is None:
                raise ValueError("RoPE requires positions")
            q, k = a.rope(q, pos), a.rope(k, pos)
        return q, k, v

    def _routing_mass(self, q_reg, k_reg):
        """q_reg: [B,H,S,r,D] current registers; k_reg: [B,H,F,r,D]. Returns [B,S,F] (unmasked logits' softmax
        must be restricted by the caller when S > 1)."""
        scale = self.base.scale
        return torch.einsum("bhsrd,bhfud->bhsruf", q_reg.float(), k_reg.float()) * scale

    def _output(self, y):
        a = self.base
        b, _, n, _ = y.shape
        return a.proj_drop(a.proj(y.transpose(1, 2).reshape(b, n, -1)))

    def forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
        if use_cache:
            return self._forward_cached(x, pos, past_key_values)
        if attn_mask is None:
            raise ValueError("The full-sequence path needs the aggregator's frame-causal mask")
        return self._forward_full(x, pos, attn_mask)

    def _forward_full(self, x, pos, attn_mask):
        a, r = self.base, self.num_registers
        b, n, c = x.shape
        # Frame length from the frame-causal mask: row 0 may attend exactly frame 0.
        p = int((attn_mask.reshape(-1, n, n)[0, 0] == 0).sum())
        if p <= r or n % p:
            raise ValueError("Could not infer frame length from the frame-causal mask")
        if self.evict_budget is not None:
            return self.forward_evicted(x, pos, p)[0]
        s = n // p
        q, k, v = self._qkv(x, pos)
        h, d = a.num_heads, a.head_dim
        regs = slice(1, 1 + r)
        q_reg = q.reshape(b, h, s, p, d)[:, :, :, regs]
        k_reg = k.reshape(b, h, s, p, d)[:, :, :, regs]
        logits = self._routing_mass(q_reg, k_reg)                        # [B,H,S,r,r,F]
        future = torch.arange(s, device=x.device)[None, :] > torch.arange(s, device=x.device)[:, None]
        logits = logits.masked_fill(future[None, None, :, None, None, :], -torch.inf)
        prob = logits.flatten(-2).softmax(-1).unflatten(-1, (r, s))      # softmax over all visible registers
        mass = prob.sum(-2).mean((1, 3))                                   # [B,S,F]
        keep = torch.eye(s, device=x.device, dtype=torch.bool).expand(b, s, s).clone()
        selected = []
        with torch.no_grad():
            for t in range(s):
                chosen = self.select(mass[:, t], t)
                keep[:, t].scatter_(1, chosen, True)
                selected.append(chosen)
        self.last_selected = selected
        return self._output(self._attend_selected(q, k, v, keep, p))

    def _attend_selected(self, q, k, v, keep, p):
        """Each query frame attends only the frames it keeps (keep: [B,S,F] bool over the F key frames;
        F = S, or more when k/v also hold earlier frames, as in truncated BPTT). Equals the masked dense
        attention over all N tokens, but memory grows linearly with S instead of quadratically, and the
        unmasked calls can use the fused SDPA kernels."""
        a = self.base
        b, h, n, d = q.shape
        s = n // p
        counts = keep.sum(-1)                                              # [B,S] frames read per query frame
        if not (counts == counts[:1]).all():
            raise ValueError("Frame budgets differ across the batch")
        qf = q.reshape(b, h, s, p, d)
        kf, vf = (t.reshape(b, h, -1, p, d) for t in (k, v))
        fold = lambda t, g: t.transpose(1, 2).reshape(b * g, h, -1, d)  # [B,H,G,...] -> [B*G,H,L,D]
        out = [None] * s
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for c in counts[0].unique().tolist():
                frames = torch.nonzero(counts[0] == c).flatten()          # query frames that read c frames
                g = len(frames)
                ids = torch.argsort((~keep[:, frames]).to(torch.uint8), dim=-1, stable=True)[..., :c]  # [B,G,c]
                gather = ids.reshape(b, 1, g * c, 1, 1).expand(b, h, g * c, p, d)
                sk = kf.gather(2, gather).reshape(b, h, g, c * p, d)
                sv = vf.gather(2, gather).reshape(b, h, g, c * p, d)
                y = F.scaled_dot_product_attention(fold(qf[:, :, frames], g), fold(sk, g), fold(sv, g),
                                                   dropout_p=a.attn_drop.p if self.training else 0.)
                y = y.reshape(b, g, h, p, d)
                for j, t in enumerate(frames.tolist()):
                    out[t] = y[:, j]
        return torch.stack(out, dim=2).reshape(b, h, n, d)

    def _attend_masked(self, q, k, v, keep, p):
        """Reference for _attend_selected (tests): dense attention over all tokens with a frame mask."""
        token_keep = keep.repeat_interleave(p, 1).repeat_interleave(p, 2)  # [B,N,N]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=token_keep[:, None],
                                              dropout_p=self.base.attn_drop.p if self.training else 0.)

    @torch.no_grad()
    def kept_tokens(self, q, k, p):
        """Token eviction: per frame, the token ids [B,S,T] it keeps in the history: its camera/register tokens,
        then (sorted) the patch tokens its own queries attend to most. q, k: [B,H,S*P,D], normalised + rotated."""
        b, h, n, d = q.shape
        s, special = n // p, 1 + self.num_registers
        patches = p - special
        count = max(1, round(self.evict_budget * patches))
        qf, kf = q.reshape(b, h, s, p, d), k.reshape(b, h, s, p, d)
        score = torch.stack([(qf[:, :, t].float() @ kf[:, :, t].float().transpose(-1, -2) * self.base.scale)
                             .softmax(-1).sum((1, 2)) for t in range(s)], dim=1)               # [B,S,P]
        top = score[..., special:].topk(count, dim=-1).indices.sort(-1).values + special
        return torch.cat((torch.arange(special, device=q.device).expand(b, s, special), top), dim=-1)

    def forward_evicted(self, x, pos, p, state=None):
        """Full-sequence path with token eviction over the frames of x [B,S*P,C], after the history in `state`
        (None at the sequence start, else the state returned for the earlier frames: anchor K/V [B,H,P,D] and
        the kept K/V of every earlier frame [B,H,F,T,D], K normalised + rotated). Returns (output, state up to
        and including these frames); the returned state is detached (truncated BPTT carries it)."""
        a, r = self.base, self.num_registers
        b, n, _ = x.shape
        s, h, d = n // p, a.num_heads, a.head_dim
        q, k, v = self._qkv(x, pos)
        kf, vf = k.reshape(b, h, s, p, d), v.reshape(b, h, s, p, d)
        ids = self.kept_tokens(q, k, p)
        gather = ids[:, None, :, :, None].expand(b, h, s, ids.shape[-1], d)
        kept_k, kept_v = kf.gather(3, gather), vf.gather(3, gather)                    # [B,H,S,T,D]
        if state is None:
            anchor_k, anchor_v = kf[:, :, 0], vf[:, :, 0]
        else:
            anchor_k, anchor_v = state[0].to(k.dtype), state[1].to(v.dtype)
            kept_k = torch.cat((state[2].to(k.dtype), kept_k), dim=2)
            kept_v = torch.cat((state[3].to(v.dtype), kept_v), dim=2)
        f = kept_k.shape[2] - s                                                           # earlier frames
        q_reg = q.reshape(b, h, s, p, d)[:, :, :, 1:1 + r]
        logits = self._routing_mass(q_reg, kept_k[:, :, :, 1:1 + r])                   # [B,H,S,r,r,F+S]
        query_frame = f + torch.arange(s, device=x.device)
        future = torch.arange(f + s, device=x.device)[None, :] > query_frame[:, None]
        logits = logits.masked_fill(future[None, None, :, None, None, :], -torch.inf)
        mass = logits.flatten(-2).softmax(-1).unflatten(-1, (r, f + s)).sum(-2).mean((1, 3))   # [B,S,F+S]
        with torch.no_grad():
            chosen = [self.select(mass[:, t], f + t) for t in range(s)]
        self.last_selected = chosen
        y = self._attend_evicted(q, kf, vf, (anchor_k, anchor_v), (kept_k, kept_v), chosen)
        state = tuple(t.detach().to(v.dtype) for t in (anchor_k, anchor_v, kept_k, kept_v))
        return self._output(y), state

    def _attend_evicted(self, q, kf, vf, anchor, kept, chosen):
        """Query frame t reads the whole anchor, the kept tokens of its other chosen frames, and itself whole.
        Query frames with the same number of chosen frames share one fused SDPA call."""
        b, h, n, d = q.shape
        s, p = kf.shape[2], kf.shape[3]
        qf = q.reshape(b, h, s, p, d)
        tokens = kept[0].shape[3]
        fold = lambda t, g: t.transpose(1, 2).reshape(b * g, h, -1, d)  # [B,H,G,...] -> [B*G,H,L,D]
        counts = [c.shape[1] for c in chosen]
        whole = 1                            # chosen[t][:, 0] is the whole anchor frame 0
        out = [None] * s
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            for c in sorted(set(counts)):
                frames = [t for t in range(s) if counts[t] == c]
                g, index = len(frames), torch.tensor(frames, device=q.device)
                parts = []
                for full, compact, self_frames in zip(anchor, kept, (kf, vf)):
                    part = []
                    if c and whole:
                        part.append(full[:, :, None].expand(b, h, g, p, d))
                    if c > whole:
                        m = c - whole
                        others = torch.stack([chosen[t][:, whole:] for t in frames], dim=1)   # [B,G,m]
                        gather = others.reshape(b, 1, g * m, 1, 1).expand(b, h, g * m, tokens, d)
                        part.append(compact.gather(2, gather).reshape(b, h, g, m * tokens, d))
                    part.append(self_frames[:, :, index])
                    parts.append(torch.cat(part, dim=3))
                y = F.scaled_dot_product_attention(fold(qf[:, :, index], g), fold(parts[0], g), fold(parts[1], g),
                                                   dropout_p=self.base.attn_drop.p if self.training else 0.)
                y = y.reshape(b, g, h, p, d)
                for j, t in enumerate(frames):
                    out[t] = y[:, j]
        return torch.stack(out, dim=2).reshape(b, h, n, d)

    def _forward_cached(self, x, pos, past_key_values):
        if self.evict_budget is not None:
            return self._forward_cached_evicted(x, pos, past_key_values)
        a, r = self.base, self.num_registers
        b, n, c = x.shape
        if n <= r:
            raise ValueError("Expected camera, register, and patch tokens")
        # Raw (pre-norm, pre-RoPE) K/V, as in the teacher Attention. Register keys are also kept
        # normalized and rotated: both ops act per token with the same positions in every frame,
        # so each frame's routing keys are computed once, when the frame arrives.
        q, k, v = a.qkv(x).reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        rk = a.k_norm(k[:, :, 1:1+r])                                       # [B,H,r,D]
        if a.rope is not None:
            if pos is None:
                raise ValueError("RoPE requires positions")
            rk = a.rope(rk, pos[:, 1:1+r])
        cache = past_key_values if past_key_values is not None else SparseKVCache()
        cache.append(k, v, rk)
        k, v, rk = cache.k, cache.v, cache.reg_k
        frames = cache.frames
        q = a.q_norm(q)
        if a.rope is not None:
            q = a.rope(q, pos)
        logits = self._routing_mass(q[:, :, None, 1:1+r], rk)              # [B,H,1,r,r,F]
        mass = logits.flatten(-2).softmax(-1).unflatten(-1, (r, frames)).sum(-2).mean((1, 3))[:, 0]
        selected = self.select(mass, frames - 1)
        self.last_selected = selected.detach()
        count = selected.shape[1]
        indices = torch.cat((selected, torch.full((b, 1), frames - 1, device=x.device, dtype=torch.long)), dim=1)
        gather = indices[:, None, :, None, None].expand(b, a.num_heads, count + 1, n, a.head_dim)
        sk = a.k_norm(k.gather(2, gather).reshape(b, a.num_heads, (count + 1) * n, a.head_dim))
        sv = v.gather(2, gather).reshape(b, a.num_heads, (count + 1) * n, a.head_dim)
        if a.rope is not None:
            sk = a.rope(sk, pos.repeat(1, count + 1, 1))
        y = F.scaled_dot_product_attention(q, sk, sv, dropout_p=a.attn_drop.p if self.training else 0.)
        return self._output(y), cache

    def _forward_cached_evicted(self, x, pos, past_key_values):
        """One streaming frame with token eviction (batch size 1). Keys are stored normalised and rotated (the kept
        tokens' positions differ per frame), in v's dtype, as SDPA uses them under autocast."""
        a, r = self.base, self.num_registers
        b, n, c = x.shape
        if b != 1:
            raise ValueError("Streaming token eviction supports batch size 1")
        q, k, v = a.qkv(x).reshape(b, n, 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)
        if a.rope is not None:
            q, k = a.rope(q, pos), a.rope(k, pos)
        cache = past_key_values if past_key_values is not None else EvictedKVCache()
        cache.reg_k.append(k[:, :, 1:1 + r].float())   # routing keys at full precision, as in the plain path
        keep = self.kept_tokens(q, k, n)[0, 0]
        k = k.to(v.dtype)
        frames = cache.frames + 1
        logits = self._routing_mass(q[:, :, None, 1:1 + r], cache.reg_k.view)
        mass = logits.flatten(-2).softmax(-1).unflatten(-1, (r, frames)).sum(-2).mean((1, 3))[:, 0]
        selected = self.select(mass, frames - 1)
        self.last_selected = selected.detach()
        parts_k, parts_v = [], []
        # compact buffers hold frames 1..F-1; frame 0 is the whole anchor
        others = selected[0, 1:] - 1 if selected.shape[1] else selected[0, :0]
        if cache.anchor is not None:
            parts_k.append(cache.anchor[0])
            parts_v.append(cache.anchor[1])
        if len(others):
            parts_k.append(cache.k.view.index_select(2, others).reshape(b, a.num_heads, -1, a.head_dim))
            parts_v.append(cache.v.view.index_select(2, others).reshape(b, a.num_heads, -1, a.head_dim))
        y = F.scaled_dot_product_attention(q, torch.cat(parts_k + [k], dim=2), torch.cat(parts_v + [v], dim=2),
                                           dropout_p=a.attn_drop.p if self.training else 0.)
        if cache.frames == 0:
            cache.anchor = (k, v)
        else:
            cache.k.append(k.index_select(2, keep))
            cache.v.append(v.index_select(2, keep))
        cache.frames = frames
        return self._output(y), cache


class _Growable:
    """[B,H,capacity,T,D] buffer that grows by max(16, frames // 8) slots, like SparseKVCache."""

    def __init__(self):
        self.frames, self.buffer = 0, None

    def append(self, x):  # x [B,H,T,D]
        if self.buffer is None:
            self.buffer = x.new_empty(x.shape[:2] + (16,) + x.shape[2:])
        if self.frames == self.buffer.shape[2]:
            bigger = self.buffer.new_empty(self.buffer.shape[:2] + (self.frames + max(16, self.frames // 8),)
                                           + self.buffer.shape[3:])
            bigger[:, :, :self.frames] = self.buffer
            self.buffer = bigger
        self.buffer[:, :, self.frames] = x
        self.frames += 1

    @property
    def view(self):
        return self.buffer[:, :, :self.frames]


class EvictedKVCache:
    """Streaming history with token eviction: the anchor's K/V whole, the kept K/V of frames 1..F-1, and the
    register routing keys of every frame. `frames` counts the streamed frames (the aggregator reads it)."""

    def __init__(self):
        self.frames, self.anchor = 0, None
        self.k, self.v, self.reg_k = _Growable(), _Growable(), _Growable()  # k/v hold frames 1..F-1

    def nbytes(self):
        total = sum(t.numel() * t.element_size() for t in (self.anchor or ()))
        return total + sum(g.view.numel() * g.view.element_size() for g in (self.k, self.v, self.reg_k)
                           if g.buffer is not None)


class SparseKVCache:
    """Streaming history of one sparse layer: raw K/V [B,H,F,N,D] and routing register keys [B,H,F,r,D].

    The buffers hold `capacity` frame slots, of which the first `frames` are valid. When they are
    full, they grow by max(16, frames // 8) slots, so on average each new frame copies O(1) frames of
    history instead of the whole history (a torch.cat per frame). `frames` is also what the
    aggregator reads to count the streamed frames. The memory overhead is at most 1/8 of the history
    (or 16 frames).
    """

    def __init__(self):
        self.frames, self._k, self._v, self._reg_k = 0, None, None, None

    def append(self, k, v, reg_k):
        """k, v: [B,H,N,D]; reg_k: [B,H,r,D] for the new frame."""
        new = (k, v, reg_k)
        if self._k is None:
            capacity = 16
            self._k, self._v, self._reg_k = (x.new_empty(x.shape[:2] + (capacity,) + x.shape[2:]) for x in new)
        elif (self._k.shape[:2] != k.shape[:2] or self._k.shape[3:] != k.shape[2:]
              or self._reg_k.shape[3:] != reg_k.shape[2:]):
            raise ValueError("Incompatible history cache")
        if self.frames == self._k.shape[2]:
            capacity = self.frames + max(16, self.frames // 8)
            grown = []
            for buffer in (self._k, self._v, self._reg_k):
                bigger = buffer.new_empty(buffer.shape[:2] + (capacity,) + buffer.shape[3:])
                bigger[:, :, :self.frames] = buffer
                grown.append(bigger)
            self._k, self._v, self._reg_k = grown
        for buffer, x in zip((self._k, self._v, self._reg_k), new):
            buffer[:, :, self.frames] = x
        self.frames += 1

    @property
    def k(self):
        return self._k[:, :, :self.frames]

    @property
    def v(self):
        return self._v[:, :, :self.frames]

    @property
    def reg_k(self):
        return self._reg_k[:, :, :self.frames]
