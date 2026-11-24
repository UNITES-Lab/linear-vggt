"""Fast streaming inference for LinearVGGT (inference only; mutates the model's head/patch-embed methods).

The per-frame computation is the aggregator + heads of inference.streaming.predict_images, re-written as one program so that
it can run as a few CUDA graphs and with optional fused kernels:

  graphs  The frame is split at the sparse-attention layers (their KV cache grows) into segments of fixed-shape
          work. Each segment is one CUDA graph; recurrent-mixer states live in static buffers. The sparse-attention
          cores and the camera head (growing trunk KV cache) run eagerly between the graphs.
  exact   Optimisations that leave every output bitwise unchanged:
          - Linear/Conv2d weights of the aggregator kept in bf16 (autocast casts them to bf16 on every call anyway);
          - DPT-head and DINO positional embeddings cached per resolution (were recomputed every frame);
          - only the intermediate layers that the heads read are concatenated (DPT layers + the last one);
          - RoPE cos/sin rows gathered once per resolution instead of in every attention call;
          - sparse layers cache keys after QK-norm + RoPE (bf16, as SDPA consumes them), so each frame normalises
            and rotates only its own keys instead of all selected history frames.
  fused   Triton kernels (fused_ops): residual add + LayerScale + LayerNorm (+ bf16 cast for the next Linear), and
          QK-norm + 2D RoPE + bf16 cast. Same precision as the eager path (fp32 math, bf16 where autocast rounds);
          LayerNorm statistics differ from PyTorch's at fp32 rounding level, so outputs are close, not bitwise.

  streamer = FastStreamer(model, exact=True, fused=True, graphs=True)
  streamer.reset()
  for frame in frames:                  # [1, 1, 3, H, W] in [0, 1]
      out = streamer.step(frame)        # dict(depth, depth_conf, points, point_conf, pose); tensors are fresh copies
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from linearvggt.layers.anchor_sparse_attention import AnchorTopKAttention, SparseKVCache
from streamvggt.layers.layer_scale import LayerScale
from streamvggt.layers.vision_transformer import BlockChunk
from .fused_ops import add_scale_layernorm, qk_norm_rope, stack_qk_norm


def _rotate(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class FastStreamer:
    def __init__(self, model, exact=True, fused=False, graphs=True):
        self.model, self.exact, self.fused, self.graphs = model, exact, fused, graphs
        agg = self.agg = model.aggregator
        if exact and any(isinstance(b.attn, AnchorTopKAttention) and b.attn.evict_budget is not None
                         for b in agg.global_blocks):  # checked before the model is changed (bf16 casts below)
            raise ValueError("FastStreamer does not implement token eviction; set evict_budget to None in the sparse "
                             "layers first (LinearVGGT.enable_fast_stream does this)")
        for module in agg.modules():                       # autocast would cast these to bf16 on every call
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.to(torch.bfloat16)
        self.dino = agg.patch_embed
        self.dino_blocks = [b for blk in self.dino.blocks
                            for b in (blk if isinstance(blk, BlockChunk) else [blk]) if not isinstance(b, nn.Identity)]
        self.sparse_layers = [i for i, b in enumerate(agg.global_blocks) if isinstance(b.attn, AnchorTopKAttention)]
        self.mixer_layers = [i for i, b in enumerate(agg.global_blocks) if hasattr(b.attn, "step")]
        if len(self.sparse_layers) + len(self.mixer_layers) != agg.depth:
            raise ValueError("Every global block must be a recurrent mixer or a sparse attention layer")
        read = set(model.depth_head.intermediate_layer_idx) | set(model.point_head.intermediate_layer_idx)
        self.keep = read | {agg.depth - 1} if exact else set(range(agg.depth))
        if exact:
            _cache_dpt_pos_embed(model.depth_head)
            _cache_dpt_pos_embed(model.point_head)
            _cache_dino_pos_embed(self.dino)
        if fused:
            self.qk_norm = {id(a): stack_qk_norm(a.q_norm, a.k_norm)
                            for a in [b.attn for b in agg.frame_blocks]
                            + [agg.global_blocks[i].attn.base for i in self.sparse_layers]}
        for i in self.mixer_layers:                        # knobs read inside the step: keep them on the GPU
            m = agg.global_blocks[i].attn
            for name in ("dt_head_scale", "a_head_scale"):
                if torch.is_tensor(getattr(m, name, None)):
                    setattr(m, name, getattr(m, name).to(agg.camera_token.device))
        self.resolution, self.prepared = None, OrderedDict()
        self.reset()

    # ------------------------------------------------------------------ setup
    STATE = ("pos", "cos", "sin", "rope_rows", "img", "special", "mixer_state", "graph_list", "h_static", "y_static",
             "outputs")
    MAX_RESOLUTIONS = 3                                    # prepared (and possibly captured) resolutions kept

    def _use(self, images, capture):
        """Switch to the prepared state of this resolution (preparing it if new); capture its graphs if asked."""
        key = tuple(images.shape[-2:])
        if self.resolution != key:
            if self.frame:
                raise ValueError("Resolution changed within a sequence")
            if self.resolution is not None:
                self.prepared[self.resolution] = {k: getattr(self, k) for k in self.STATE}
            if key in self.prepared:
                self.prepared.move_to_end(key)
                for k, v in self.prepared.pop(key).items():
                    setattr(self, k, v)
                self.resolution = key
            else:
                while len(self.prepared) >= self.MAX_RESOLUTIONS:
                    self.prepared.popitem(last=False)
                self._prepare(images)
            self.reset()
        if capture and self.graph_list is None:
            self._capture()

    def _prepare(self, images):
        agg, dev = self.agg, images.device
        h, w = images.shape[-2:]
        p = agg.patch_size
        pos = agg.position_getter(1, h // p, w // p, device=dev) + 1
        pos = torch.cat([torch.zeros(1, agg.patch_start_idx, 2, device=dev, dtype=pos.dtype), pos], dim=1)
        self.pos = pos.contiguous()
        cos, sin = agg.rope._compute_frequency_components(agg.frame_blocks[0].attn.head_dim // 2,
                                                          agg.rope.MAX_POSITIONS, dev, torch.float32)
        self.cos, self.sin = cos, sin
        self.rope_rows = [(F.embedding(pos[..., i], cos)[:, None], F.embedding(pos[..., i], sin)[:, None])
                          for i in (0, 1)]                 # exactly the rows RotaryPositionEmbedding2D gathers
        self.img = images.clone()
        self.special = torch.zeros(1, agg.patch_start_idx, agg.camera_token.shape[-1], device=dev)
        self.mixer_state = {i: agg.global_blocks[i].attn.new_cache(1, dev, torch.float32) for i in self.mixer_layers}
        self.graph_list, self.h_static, self.y_static, self.outputs = None, {}, {}, None
        self.resolution = (h, w)

    def reset(self):
        self.frame, self.sparse_cache, self.camera_cache = 0, {}, None
        if self.resolution is not None:
            for state in self.mixer_state.values():
                for t in state:
                    if torch.is_tensor(t):
                        t.zero_()

    # ------------------------------------------------------------------ building blocks
    def _res_norm(self, x, y, ls, norm, out_dtype=torch.bfloat16):
        """(x + ls(y), norm(x + ls(y))); y None: (x, norm(x)); norm None: no norm."""
        if self.fused and norm is not None:
            gamma = ls.gamma if isinstance(ls, LayerScale) else None
            return add_scale_layernorm(x, y, gamma, norm, out_dtype)
        if y is not None:
            x = x + ls(y)
        return x, (norm(x) if norm is not None else None)

    def _block(self, blk, x, h, attn, next_norm, next_dtype=torch.bfloat16):
        x, h2 = self._res_norm(x, attn(blk.attn, h), blk.ls1, blk.norm2)
        return self._res_norm(x, blk.mlp(h2), blk.ls2, next_norm, next_dtype)

    def _rope(self, t, rope, rows=slice(None)):
        if not self.exact:
            return rope(t, self.pos[:, rows])
        (cy, sy), (cx, sx) = self.rope_rows
        v, hz = t.chunk(2, dim=-1)
        v = (v * cy[:, :, rows]) + (_rotate(v) * sy[:, :, rows])
        hz = (hz * cx[:, :, rows]) + (_rotate(hz) * sx[:, :, rows])
        return torch.cat((v, hz), dim=-1)

    def _qkv(self, a, h):
        """q, k (after QK-norm + RoPE, as SDPA receives them), v, and the raw qkv view [B, N, 3, H, D]."""
        b, n, c = h.shape
        qkv = a.qkv(h)
        qkv5 = qkv.view(b, n, 3, a.num_heads, a.head_dim)
        v = qkv5[:, :, 2].transpose(1, 2)
        if self.fused:
            q, k = qk_norm_rope(qkv, a.num_heads, self.qk_norm[id(a)], a.q_norm.eps, self.cos, self.sin, self.pos)
        else:
            q, k = qkv5.permute(2, 0, 3, 1, 4).unbind(0)[:2]
            q, k = self._rope(a.q_norm(q), a.rope), self._rope(a.k_norm(k), a.rope)
        return q, k, v, qkv5

    def _frame_attn(self, a, h):
        b, n, c = h.shape
        q, k, _, qkv5 = self._qkv(a, h)
        v = qkv5[:, :, 2].transpose(1, 2) if self.fused else qkv5.permute(2, 0, 3, 1, 4)[2]
        y = F.scaled_dot_product_attention(q, k, v)
        return a.proj(y.transpose(1, 2).reshape(b, n, c))

    def _mixer(self, i):
        def run(m, h):
            state = self.mixer_state[i]
            out, new = m.step(h, state)
            for old, value in zip(state, new):
                if torch.is_tensor(old):
                    old.copy_(value)
            return out
        return run

    def _sparse_core(self, i, h):
        """One streamed frame of AnchorTopKAttention._forward_cached; caches keys after QK-norm + RoPE."""
        attn = self.agg.global_blocks[i].attn
        if not self.exact:
            y, self.sparse_cache[i] = attn(h, pos=self.pos, past_key_values=self.sparse_cache.get(i), use_cache=True)
            return y
        a, r = attn.base, attn.num_registers
        b, n, c = h.shape
        q, k, v, qkv5 = self._qkv(a, h)
        regs = slice(1, 1 + r)
        if self.fused:                                     # routing uses fp32 register queries/keys as in eager
            q_reg = self._rope(a.q_norm(qkv5[:, regs, 0].transpose(1, 2)), a.rope, regs)
            rk = self._rope(a.k_norm(qkv5[:, regs, 1].transpose(1, 2)), a.rope, regs)
        else:
            q_reg = q[:, :, regs]
            rk = self._rope(a.k_norm(qkv5[:, :, 1].transpose(1, 2)[:, :, regs]), a.rope, regs)
        cache = self.sparse_cache.setdefault(i, SparseKVCache())
        cache.append(k.to(v.dtype), v, rk)
        frames = cache.frames
        logits = attn._routing_mass(q_reg[:, :, None], cache.reg_k)
        mass = logits.flatten(-2).softmax(-1).unflatten(-1, (r, frames)).sum(-2).mean((1, 3))[:, 0]
        selected = attn.select(mass, frames - 1)
        count = selected.shape[1]
        indices = torch.cat((selected, torch.full((b, 1), frames - 1, device=h.device, dtype=torch.long)), dim=1)
        gather = indices[:, None, :, None, None].expand(b, a.num_heads, count + 1, n, a.head_dim)
        sk = cache.k.gather(2, gather).reshape(b, a.num_heads, (count + 1) * n, a.head_dim)
        sv = cache.v.gather(2, gather).reshape(b, a.num_heads, (count + 1) * n, a.head_dim)
        y = F.scaled_dot_product_attention(q, sk, sv)
        return attn._output(y)

    # ------------------------------------------------------------------ one frame as a program
    def _program(self):
        """Generator over one frame. Yields (layer, h) at each sparse layer and expects its attention output back;
        returns (features, depth, depth_conf, points, point_conf)."""
        agg, model = self.agg, self.model
        images = (self.img - agg._resnet_mean) / agg._resnet_std
        x = self.dino.prepare_tokens_with_masks(images.reshape(-1, *images.shape[-3:]))
        blocks = self.dino_blocks
        x, h = self._res_norm(x, None, None, blocks[0].norm1)
        for j, blk in enumerate(blocks):
            last = j == len(blocks) - 1
            x, h = self._block(blk, x, h, lambda m, t: m(t), self.dino.norm if last else blocks[j + 1].norm1,
                               torch.float32 if last else torch.bfloat16)
        tokens = torch.cat([self.special, h[:, 1 + self.dino.num_register_tokens:]], dim=1)
        bsz, n, c = tokens.shape
        x, h = self._res_norm(tokens, None, None, agg.frame_blocks[0].norm1)
        features = [None] * agg.depth
        for i in range(agg.depth):
            fb, gb = agg.frame_blocks[i], agg.global_blocks[i]
            x, h = self._block(fb, x, h, self._frame_attn, gb.norm1)
            frame_out = x
            nxt = agg.frame_blocks[i + 1].norm1 if i + 1 < agg.depth else None
            if i in self.mixer_layers:
                x, h = self._block(gb, x, h, self._mixer(i), nxt)
            else:
                y = yield i, h
                x, h2 = self._res_norm(x, y, gb.ls1, gb.norm2)
                x, h = self._res_norm(x, gb.mlp(h2), gb.ls2, nxt)
            if i in self.keep:
                features[i] = torch.cat([frame_out, x], dim=-1).reshape(bsz, 1, n, 2 * c)
        with torch.autocast("cuda", enabled=False):
            features = [f.float() if f is not None else None for f in features]
            depth, depth_conf = model.depth_head(features, images=self.img, patch_start_idx=agg.patch_start_idx)
            points, point_conf = model.point_head(features, images=self.img, patch_start_idx=agg.patch_start_idx)
        return features, depth, depth_conf, points, point_conf

    def _run_eager(self):
        program = self._program()
        try:
            request = next(program)
            while True:
                request = program.send(self._sparse_core(*request))
        except StopIteration as stop:
            return stop.value

    def _capture(self):
        """Warm up eagerly (Triton/cuBLAS/cuDNN init, cached embeddings), then capture one graph per segment."""
        self.reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._run_eager()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        self.reset()
        program, graphs, send = self._program(), [], None
        while True:
            graph = torch.cuda.CUDAGraph()
            done = False
            with torch.cuda.graph(graph):
                try:
                    request = program.send(send) if graphs else next(program)
                except StopIteration as stop:
                    self.outputs, done = stop.value, True
            graphs.append(graph)
            graph.replay()                                 # make the static outputs valid for the next segment
            if done:
                break
            layer, h = request
            self.h_static[layer] = h
            y = self._sparse_core(layer, h)
            self.y_static[layer] = send = y.clone()
        self.graph_list = graphs
        torch.cuda.synchronize()
        self.reset()

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def step(self, images, graphs=None):
        """images [1, 1, 3, H, W] in [0, 1] -> dict of this frame's predictions (fresh tensors).
        graphs: replay CUDA graphs (default: the constructor's setting); False runs the same program eagerly."""
        if images.shape[0] != 1 or images.shape[1] != 1:
            raise ValueError("FastStreamer streams one frame of batch size 1")
        graphs = self.graphs if graphs is None else graphs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self._use(images, capture=graphs)
            agg = self.agg
            which = 0 if self.frame == 0 else 1
            special = torch.cat([agg.camera_token[:, which], agg.register_token[:, which]], dim=1)
            self.img.copy_(images)
            self.special.copy_(special)
            if graphs:
                for graph, layer in zip(self.graph_list, self.sparse_layers + [None]):
                    graph.replay()
                    if layer is not None:
                        self.y_static[layer].copy_(self._sparse_core(layer, self.h_static[layer]))
                features, depth, depth_conf, points, point_conf = self.outputs
            else:
                features, depth, depth_conf, points, point_conf = self._run_eager()
            with torch.autocast("cuda", enabled=False):
                pose, self.camera_cache = self.model.camera_head.stream_step(features[-1], self.camera_cache)
        self.frame += 1
        return dict(depth=depth.clone(), depth_conf=depth_conf.clone(), points=points.clone(),
                    point_conf=point_conf.clone(), pose=pose.clone())

    MIN_GRAPH_FRAMES = 16    # shorter sequences run the same program eagerly (capture costs ~1-2 s per resolution)

    @torch.no_grad()
    def predict(self, images, output_buffers=None):
        """Drop-in for inference.streaming.predict_images(model, images, streaming=True, output_buffers): one fresh
        sequence [1, T, 3, H, W]. Must run under bf16 autocast, the precision the eager path is evaluated in."""
        if not (torch.is_autocast_enabled() and torch.get_autocast_gpu_dtype() == torch.bfloat16):
            raise RuntimeError("FastStreamer reproduces the bf16-autocast eager path; call it under bf16 autocast")
        if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] < 1:
            raise ValueError("FastStreamer expects one [1, T, 3, H, W] sequence")
        graphs = self.graphs and images.shape[1] >= self.MIN_GRAPH_FRAMES
        self.reset()
        outputs = []
        for t, chunk in enumerate(images.split(1, dim=1)):
            frame = self.step(chunk, graphs=graphs)
            if output_buffers is None:
                outputs.append(frame)
            else:
                for key, value in frame.items():
                    output_buffers[key][:, t].copy_(value[:, 0], non_blocking=True)
        self.reset()
        if output_buffers is not None:
            torch.cuda.synchronize(images.device)
            return output_buffers
        return {key: torch.cat([o[key] for o in outputs], dim=1) for key in outputs[0]}


def _cache_dpt_pos_embed(head):
    """DPTHead._apply_pos_embed recomputes a sinusoidal embedding (float64 sin/cos) on every call; cache it."""
    original, cache = head._apply_pos_embed, {}

    def apply(x, W, H, ratio=0.1):
        key = (x.shape[1:], W, H, ratio, x.dtype, x.device)
        if key not in cache:
            cache[key] = original(torch.zeros_like(x[:1]), W, H, ratio)   # 0 + pos_embed == pos_embed exactly
        return x + cache[key].expand(x.shape[0], -1, -1, -1)

    head._apply_pos_embed = apply


def _cache_dino_pos_embed(dino):
    """DinoVisionTransformer.interpolate_pos_encoding re-interpolates the positional table every frame; cache it."""
    original, cache = dino.interpolate_pos_encoding, {}

    def interpolate(x, w, h):
        key = (x.shape[1], x.dtype, x.device, w, h)
        if key not in cache:
            cache[key] = original(x, w, h)
        return cache[key]

    dino.interpolate_pos_encoding = interpolate
