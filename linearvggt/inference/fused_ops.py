"""Inference-only Triton kernels for the streaming student (no autograd).

Both kernels keep the precision of the eager path: fp32 math for residual/LayerNorm/RoPE, bf16 only where autocast
would round anyway (inputs of the next Linear / SDPA). Multiplies and adds that PyTorch rounds separately use
mul.rn / add.rn so they are not contracted into FMAs. The LayerNorm statistics use a different reduction order than
PyTorch's kernel, so results match to fp32 rounding, not bitwise.

add_scale_layernorm(x, y, gamma, norm, out_dtype) -> (x + y * gamma, LayerNorm(x + y * gamma).to(out_dtype))
qk_norm_rope(qkv, H, stack_qk_norm(q_norm, k_norm), eps, cos, sin, positions) -> q, k [B, H, N, D] bf16 after QK-norm + 2D RoPE
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _add_scale_ln_kernel(X, Y, G, W, B, XO, H, C, eps, HAS_RES: tl.constexpr, HAS_GAMMA: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < C
    x = tl.load(X + row * C + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        y = tl.load(Y + row * C + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_GAMMA:
            y = _mul_rn(y, tl.load(G + cols, mask=mask, other=0.0))
        x = _add_rn(x, y)
        tl.store(XO + row * C + cols, x, mask=mask)
    mean = tl.sum(x, axis=0) / C
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    h = d * rstd * w + b
    tl.store(H + row * C + cols, h.to(H.dtype.element_ty), mask=mask)


def add_scale_layernorm(x, y, gamma, norm, out_dtype, x_out=None):
    """x: [..., C] fp32 residual; y: [..., C] (bf16 sublayer output) or None; gamma: LayerScale [C] or None;
    norm: nn.LayerNorm. Returns (new residual fp32 (x itself if y is None), normalised tensor in out_dtype)."""
    c = x.shape[-1]
    x2 = x.reshape(-1, c)
    if not x2.is_contiguous():
        raise ValueError("residual must be contiguous")
    rows = x2.shape[0]
    h = torch.empty(x.shape, device=x.device, dtype=out_dtype)
    has_res = y is not None
    if has_res:
        y2 = y.reshape(-1, c)
        if not y2.is_contiguous():
            y2 = y2.contiguous()
        x_out = torch.empty_like(x) if x_out is None else x_out
    else:
        y2, x_out = x2, x
    _add_scale_ln_kernel[(rows,)](
        x2, y2, gamma if gamma is not None else x2, norm.weight, norm.bias, x_out, h, c, norm.eps,
        HAS_RES=has_res, HAS_GAMMA=gamma is not None, BLOCK=triton.next_power_of_2(c), num_warps=4)
    return x_out, h


@triton.jit
def _qk_norm_rope_kernel(QKV, NW, NB, POS, COS, SIN, OUT, BHND, N, eps,
                         HAS_NORM: tl.constexpr, HAS_ROPE: tl.constexpr, H: tl.constexpr, D: tl.constexpr):
    token = tl.program_id(0).to(tl.int64)       # over B * N
    which = tl.program_id(1)                    # 0: q, 1: k
    b = token // N
    n = token % N
    heads = tl.arange(0, H)[:, None]
    d = tl.arange(0, D)[None, :]
    HALF: tl.constexpr = D // 2                 # features per spatial direction
    QUARTER: tl.constexpr = D // 4
    half = d // HALF
    j = d % HALF
    partner = half * HALF + (j + QUARTER) % HALF
    base = QKV + token * 3 * H * D + which * H * D
    t = tl.load(base + heads * D + d).to(tl.float32)
    tp = tl.load(base + heads * D + partner).to(tl.float32)
    if HAS_NORM:
        mean = tl.sum(t, axis=1)[:, None] / D
        dt = t - mean
        var = tl.sum(dt * dt, axis=1)[:, None] / D
        rstd = 1.0 / tl.sqrt(var + eps)
        t = dt * rstd * tl.load(NW + which * D + d) + tl.load(NB + which * D + d)
        tp = (tp - mean) * rstd * tl.load(NW + which * D + partner) + tl.load(NB + which * D + partner)
    if HAS_ROPE:
        py = tl.load(POS + token * 2)
        px = tl.load(POS + token * 2 + 1)
        p = tl.where(half == 0, py, px)
        c = tl.load(COS + p * HALF + j)
        s = tl.load(SIN + p * HALF + j)
        rot = tl.where(j < QUARTER, -tp, tp)
        t = _add_rn(_mul_rn(t, c), _mul_rn(rot, s))
    offs = which * BHND + ((b * H + heads) * N + n) * D + d
    tl.store(OUT + offs, t.to(OUT.dtype.element_ty))


def stack_qk_norm(q_norm, k_norm):
    """Packed [2, D] weight/bias of the q and k LayerNorms for qk_norm_rope (build once)."""
    return torch.stack((q_norm.weight, k_norm.weight)).contiguous(), torch.stack((q_norm.bias, k_norm.bias)).contiguous()


def qk_norm_rope(qkv, num_heads, norm=None, eps=1e-5, cos=None, sin=None, positions=None, out_dtype=torch.bfloat16):
    """qkv: [B, N, 3*C] contiguous (Linear output). norm: stack_qk_norm(...) or None (no QK-norm). cos/sin: RoPE
    tables [P, D/2] fp32 (RotaryPositionEmbedding2D frequency components) or None; positions [B, N, 2] long (y, x).
    Returns q, k as [B, H, N, D] in out_dtype (views of one [2, B, H, N, D] buffer)."""
    bsz, n, c3 = qkv.shape
    d = c3 // 3 // num_heads
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    out = torch.empty(2, bsz, num_heads, n, d, device=qkv.device, dtype=out_dtype)
    has_norm, has_rope = norm is not None, cos is not None
    _qk_norm_rope_kernel[(bsz * n, 2)](
        qkv, norm[0] if has_norm else qkv, norm[1] if has_norm else qkv,
        positions if has_rope else qkv, cos if has_rope else qkv, sin if has_rope else qkv,
        out, bsz * num_heads * n * d, n, eps,
        HAS_NORM=has_norm, HAS_ROPE=has_rope, H=num_heads, D=d, num_warps=4)
    return out[0], out[1]
