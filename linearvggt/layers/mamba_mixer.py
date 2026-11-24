"""Official Mamba-2 (state-spaces/mamba) as a StreamVGGT global-attention replacement.

The mixer IS an unmodified `mamba_ssm.Mamba2` module; this file only adds
  * MOHAWK-style initialization from a teacher attention layer,
  * the aggregator's global-attention call interface, and
  * a one-frame streaming step built from the official kernels.

Configuration (MOHAWK mapping, expand=1): d_inner = dim, headdim = teacher head_dim,
nheads = teacher heads, ngroups = nheads, d_state = head_dim. The fused in_proj
rows are ordered [z | x | B | C | dt]; x <- V, B <- K, C <- Q * head_dim^-0.5,
out_proj <- teacher proj. Teacher QK LayerNorm and 2D RoPE have no counterpart in
the standard block and are dropped. Other initialization choices:
  * conv1d starts as identity (last tap = 1), followed by Mamba2's own SiLU;
  * z rows are zero with a constant bias, so the gate is a per-token constant that
    the gated RMSNorm (norm_before_gate=False) cancels: out = RMSNorm(y) at init;
  * dt rows are zero; dt_bias gives a per-token decay exp(-dt) with the requested
    half-life (A = -1). In SSD, dt also scales the input, so the initial state is a
    normalized (~ln 2) running average instead of an unbounded sum;
  * D (skip) = 0: the teacher attention has no skip path.

The scan is token-causal, including within a frame (the teacher is frame-causal).
Full-sequence calls use the official fused `mamba_split_conv1d_scan_combined`
path (forward and backward). Streaming carries the conv tail and the SSM state.
"""
import math
from typing import NamedTuple

import torch
from einops import rearrange
from torch import nn

try:
    from mamba_ssm import Mamba2
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from causal_conv1d import causal_conv1d_fn
except ImportError as error:  # pragma: no cover - import guard for machines without the build
    raise ImportError(
        "The official mamba_ssm and causal_conv1d packages are required: pip install mamba-ssm causal-conv1d") from error


class MambaCache(NamedTuple):
    conv_state: torch.Tensor  # [B, conv_dim, d_conv-1], channel-last strides (causal_conv1d requirement)
    ssm_state: torch.Tensor   # [B, nheads, headdim, d_state], fp32
    frames: int


class MambaMixer(nn.Module):
    # Inference-only diagnostic: multiplies the per-token step dt (after softplus), i.e. both the decay
    # exp(dt*A) and the input scale. dt_scale = train_tokens_per_frame / eval_tokens_per_frame keeps the
    # per-frame memory the same when evaluating at a higher resolution than training. 1.0 = unchanged.
    dt_scale = 1.0
    # Inference-only, per head (None = off): dt_head_scale [nheads] multiplies dt like dt_scale but per head
    # (decay and input); a_head_scale [nheads] multiplies A only, i.e. changes the decay exp(dt*A) and leaves
    # the input term dt*B*x unchanged.
    dt_head_scale = None
    a_head_scale = None

    def __init__(self, dim, num_heads, d_state=None, d_conv=4, chunk_size=256, initial_half_life_tokens=512.):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        head_dim = dim // num_heads
        self.config = dict(dim=dim, num_heads=num_heads, d_state=d_state or head_dim, d_conv=d_conv,
                           chunk_size=chunk_size, initial_half_life_tokens=initial_half_life_tokens)
        self.mamba = Mamba2(d_model=dim, d_state=d_state or head_dim, d_conv=d_conv, expand=1,
                            headdim=head_dim, ngroups=num_heads, rmsnorm=True, norm_before_gate=False,
                            bias=True, conv_bias=True, chunk_size=chunk_size)

    @classmethod
    def from_attention(cls, attention, **kwargs):
        dim = attention.qkv.in_features
        mixer = cls(dim, attention.num_heads, **kwargs).to(attention.qkv.weight.device)
        mixer.initialize_from_attention(attention)
        return mixer.to(dtype=attention.qkv.weight.dtype)

    @torch.no_grad()
    def initialize_from_attention(self, attention):
        m = self.mamba
        dim, h, d = m.d_model, m.nheads, m.headdim
        if (attention.num_heads, attention.head_dim, attention.qkv.in_features) != (h, d, dim) or m.d_state != d:
            raise ValueError("MOHAWK initialization needs d_state == head_dim and matching heads/width")
        wq, wk, wv = attention.qkv.weight.float().chunk(3, dim=0)
        bq, bk, bv = (attention.qkv.bias.float().chunk(3) if attention.qkv.bias is not None
                      else (torch.zeros(dim, device=wq.device),) * 3)
        scale = attention.scale
        z = slice(0, dim)
        x = slice(dim, 2 * dim)
        b = slice(2 * dim, 3 * dim)
        c = slice(3 * dim, 4 * dim)
        dt = slice(4 * dim, 4 * dim + h)
        weight, bias = m.in_proj.weight, m.in_proj.bias
        weight.zero_()
        bias.zero_()
        weight[x], bias[x] = wv, bv
        weight[b], bias[b] = wk, bk          # groups == heads, so B/C are per-head like K/Q
        weight[c], bias[c] = wq * scale, bq * scale
        bias[z] = 1.0                         # constant gate; cancelled by the gated RMSNorm
        m.conv1d.weight.zero_()
        m.conv1d.weight[:, 0, -1] = 1.0
        m.conv1d.bias.zero_()
        dt_init = math.log(2) / self.config["initial_half_life_tokens"]
        m.dt_bias.fill_(dt_init + math.log(-math.expm1(-dt_init)))  # inverse softplus
        m.A_log.zero_()                       # A = -exp(A_log) = -1
        m.D.zero_()
        m.norm.weight.fill_(1.0)
        m.out_proj.weight.copy_(attention.proj.weight)
        m.out_proj.bias.copy_(attention.proj.bias)

    def new_cache(self, batch, device, dtype):
        m = self.mamba
        conv_dim = m.d_ssm + 2 * m.ngroups * m.d_state
        conv = torch.zeros(batch, m.d_conv - 1, conv_dim, device=device, dtype=dtype).transpose(1, 2)
        ssm = torch.zeros(batch, m.nheads, m.headdim, m.d_state, device=device, dtype=torch.float32)
        return MambaCache(conv, ssm, 0)

    def _split(self, u):
        m = self.mamba
        z, xBC, dt = torch.split(m.in_proj(u), [m.d_ssm, m.d_ssm + 2 * m.ngroups * m.d_state, m.nheads], dim=-1)
        return z, xBC, dt

    def step(self, u, cache):
        """One frame [B, L, D] with carried conv/SSM state, using the official kernels."""
        m = self.mamba
        z, xBC, dt = self._split(u)
        if xBC.stride(1) % 8:  # channel-last causal_conv1d needs 8-aligned strides (holds for dim=1024)
            xBC = xBC.contiguous()
        conv_state = cache.conv_state.to(xBC.dtype)
        xBC, conv_state = causal_conv1d_fn(
            xBC.transpose(1, 2), rearrange(m.conv1d.weight, "d 1 w -> d w"), bias=m.conv1d.bias,
            initial_states=conv_state, return_final_states=True, activation="silu")
        x, B, C = torch.split(xBC.transpose(1, 2), [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
        dt_args = dict(dt_bias=m.dt_bias, dt_softplus=True)
        if self.dt_scale != 1.0 or self.dt_head_scale is not None:
            dt = torch.nn.functional.softplus(dt.float() + m.dt_bias.float()) * self.dt_scale
            if self.dt_head_scale is not None:
                dt = dt * self.dt_head_scale.to(dt.device)
            dt_args = dict(dt_bias=None, dt_softplus=False)
        A = -torch.exp(m.A_log.float())
        if self.a_head_scale is not None:
            A = A * self.a_head_scale.to(A.device)
        y, ssm_state = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=m.headdim), dt, A,
            rearrange(B, "b l (g n) -> b l g n", g=m.ngroups), rearrange(C, "b l (g n) -> b l g n", g=m.ngroups),
            chunk_size=m.chunk_size, D=m.D, z=None, **dt_args,
            initial_states=cache.ssm_state, return_final_states=True)
        y = m.norm(rearrange(y, "b l h p -> b l (h p)"), z)
        return m.out_proj(y), MambaCache(conv_state, ssm_state.float(), cache.frames + 1)

    def forward(self, x, pos=None, attn_mask=None, past_key_values=None, use_cache=False):
        """Global-attention interface. `pos` is unused (no RoPE in standard Mamba2).

        `attn_mask` is the aggregator's frame-causal mask; the scan is stricter
        (token-causal), so it is accepted and ignored. Cached calls must be one frame.
        """
        if past_key_values is not None and not use_cache:
            raise ValueError("A recurrent state requires use_cache=True")
        if not use_cache:
            return self.mamba(x)
        cache = past_key_values if past_key_values is not None else self.new_cache(x.shape[0], x.device, x.dtype)
        if cache.ssm_state.shape[0] != x.shape[0]:
            raise ValueError("Incompatible Mamba cache batch size")
        return self.step(x, cache)

    def mixing_rows(self, x, pos=None, rows=None):
        """Selected rows of the per-head SSD matrix M (over post-conv x), for Stage 1.

        y_i = sum_{j<=i} C_i.B_j * exp(sum_{m=j+1..i} dt_m A) * dt_j * x_j  (D and gate excluded).
        Returns [B, H, len(rows), N] in fp32, comparable to softmax attention rows.
        """
        m = self.mamba
        with torch.autocast(device_type=x.device.type, enabled=False):
            _, xBC, dt = self._split(x.float())
            xBC = m.act(m.conv1d(xBC.transpose(1, 2))[..., :xBC.shape[1]].transpose(1, 2))
            _, B, C = torch.split(xBC, [m.d_ssm, m.ngroups * m.d_state, m.ngroups * m.d_state], dim=-1)
            n = x.shape[1]
            if rows is None:
                rows = torch.arange(n, device=x.device)
            B = rearrange(B, "b l (g s) -> b g l s", g=m.ngroups)
            C = rearrange(C, "b l (g s) -> b g l s", g=m.ngroups)
            heads_per_group = m.nheads // m.ngroups
            B, C = B.repeat_interleave(heads_per_group, 1), C.repeat_interleave(heads_per_group, 1)
            dt = torch.nn.functional.softplus(dt + m.dt_bias.float()).transpose(1, 2)  # [B, H, N]
            if m.dt_limit != (0.0, float("inf")):
                dt = dt.clamp(*m.dt_limit)
            log_a = dt * -torch.exp(m.A_log.float())[None, :, None]
            prefix = log_a.cumsum(-1)
            valid = torch.arange(n, device=x.device)[None, :] <= rows[:, None]
            decay = (prefix[:, :, rows, None] - prefix[:, :, None, :]).masked_fill(~valid, -torch.inf).exp()
            return (C[:, :, rows] @ B.transpose(-1, -2)) * decay * dt[:, :, None, :]
