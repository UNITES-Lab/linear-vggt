"""LinearVGGT: a StreamVGGT student with 3 Mamba2 layers : 1 register-guided sparse attention layer.

Only the global-attention mixers change. Residuals, LayerScale, norms, MLPs, all frame-attention blocks, the
encoder and the heads keep the teacher architecture. Default layout: sparse attention at global blocks
3, 7, 11, 15, 19, 23 and a Mamba2 mixer everywhere else.

Checkpoint config (`hybrid_config`): "mamba_layers" and "sparse_layers" (global block indices), "mamba" (mixer
arguments), "sparse" (frame selection: policy, top_k, num_registers) and, optionally,
"evict": {"budget": b, "layers": [...]}: the share of patch tokens that each past frame keeps in those sparse layers.
"""
import copy
from pathlib import Path

import torch
from torch import nn

from streamvggt.models.streamvggt import StreamVGGT, StreamVGGTOutput
from ..inference.streaming import predict_images
from ..layers.anchor_sparse_attention import AnchorTopKAttention
from ..layers.mamba_mixer import MambaMixer

FORMAT = "streamvggt_mamba2_hybrid_v1"


def default_layout(depth=24, period=4):
    sparse = [i for i in range(depth) if i % period == period - 1]
    return dict(mamba_layers=[i for i in range(depth) if i not in sparse], sparse_layers=sparse)


def make_config(depth=24, mamba_layers=None, sparse_layers=None, policy="reg", top_k=2, num_registers=4,
                d_state=None, d_conv=4, chunk_size=256, initial_half_life_tokens=512.,
                evict_budget=None, evict_layers=None):
    layout = default_layout(depth)
    mamba_layers = sorted(layout["mamba_layers"] if mamba_layers is None else mamba_layers)
    sparse_layers = sorted(layout["sparse_layers"] if sparse_layers is None else sparse_layers)
    if set(mamba_layers) & set(sparse_layers) or any(i < 0 or i >= depth for i in mamba_layers + sparse_layers):
        raise ValueError("Mamba and sparse layers must be distinct global block indices")
    config = dict(mamba_layers=mamba_layers, sparse_layers=sparse_layers,
                  sparse=dict(policy=policy, top_k=top_k, num_registers=num_registers))
    config["mamba"] = dict(d_state=d_state, d_conv=d_conv, chunk_size=chunk_size,
                           initial_half_life_tokens=initial_half_life_tokens)
    if evict_budget is not None:
        layers = sparse_layers if evict_layers is None else sorted(evict_layers)
        if not set(layers) <= set(sparse_layers):
            raise ValueError("evict_layers must be sparse layers")
        config["evict"] = dict(budget=evict_budget, layers=layers)
    return config


def layer_evict_budget(config, layer):
    """Token-eviction budget of a sparse layer (None: keeps every token)."""
    evict = config.get("evict")
    return evict["budget"] if evict and layer in evict["layers"] else None


def check_config(config):
    """This release implements the final model only: Mamba2 mixers and anchored sparse attention."""
    if config.get("mixer", "mamba2") != "mamba2":
        raise ValueError(f"Checkpoints with {config['mixer']!r} recurrent layers are not supported by this release")
    if config["sparse"].get("anchor", True) is not True:
        raise ValueError("Checkpoints trained without the anchor frame are not supported by this release")


def build_mixers(aggregator, config, initialize=True, share_sparse=True):
    """Replacement mixers keyed by global block index (str), built from the CURRENT teacher mixers.

    Sparse layers wrap the teacher Attention object (`share_sparse`) or a deep copy
    of it (needed when they are trained while the teacher is still used as target).
    Mamba2 layers are new modules, MOHAWK-initialized from the teacher when `initialize`.
    """
    check_config(config)
    mixers = nn.ModuleDict()
    kwargs = config["mamba"]
    sparse = {k: v for k, v in config["sparse"].items() if k != "anchor"}
    for i in config["mamba_layers"]:
        attn = aggregator.global_blocks[i].attn
        if initialize:
            mixers[str(i)] = MambaMixer.from_attention(attn, **kwargs)
        else:
            mixers[str(i)] = MambaMixer(attn.qkv.in_features, attn.num_heads, **kwargs).to(
                device=attn.qkv.weight.device, dtype=attn.qkv.weight.dtype)
    for i in config["sparse_layers"]:
        attn = aggregator.global_blocks[i].attn
        mixers[str(i)] = AnchorTopKAttention(attn if share_sparse else copy.deepcopy(attn), **sparse,
                                             evict_budget=layer_evict_budget(config, i))
    return mixers


def install(aggregator, mixers):
    for i, module in mixers.items():
        aggregator.global_blocks[int(i)].attn = module


class LinearVGGT(StreamVGGT):
    def __init__(self, config=None, initialize=False):
        super().__init__()
        self.hybrid_config = config or make_config(self.aggregator.depth)
        install(self.aggregator, build_mixers(self.aggregator, self.hybrid_config, initialize))

    def enable_fast_stream(self, exact=True, fused=True, graphs=True):
        """Inference only: route streaming predict_images through inference.fast_stream.FastStreamer (CUDA graphs,
        exact optimisations, fused kernels). Call after every other inference knob is set; it keeps bf16 copies of
        the aggregator's Linear/Conv2d weights and caches the heads' positional embeddings."""
        from linearvggt.inference.fast_stream import FastStreamer
        sparse = [self.aggregator.global_blocks[i].attn for i in self.hybrid_config["sparse_layers"]]
        if any(layer.evict_budget is not None for layer in sparse):
            # FastStreamer does not implement token eviction (the token indexer). Without it, each cached frame
            # keeps all its tokens: the same frames are selected, and the cache uses about four times the memory.
            for layer in sparse:
                layer.evict_budget = None
            print("LinearVGGT fast stream: token eviction is off (not implemented in FastStreamer)", flush=True)
        self.fast_streamer = FastStreamer(self, exact=exact, fused=fused, graphs=graphs)
        print(f"LinearVGGT fast stream: exact={exact} fused={fused} graphs={graphs}", flush=True)
        return self

    def set_sparse(self, policy=None, top_k=None):
        """Change the frame selection ("reg" or "dense") or its budget at inference time (no parameter change)."""
        for i in self.hybrid_config["sparse_layers"]:
            module = self.aggregator.global_blocks[i].attn
            module.policy = policy or module.policy
            module.top_k = module.top_k if top_k is None else top_k

    @torch.no_grad()
    def predict_images(self, images, streaming=True, output_buffers=None):
        return predict_images(self, images, streaming, output_buffers)

    @torch.no_grad()
    def inference(self, frames, query_points=None, past_key_values=None):
        if query_points is not None or past_key_values is not None:
            raise NotImplementedError("This student API supports fresh depth/point/pose sequences; use predict_images")
        prediction = self.predict_images(torch.stack([f["img"] for f in frames], dim=1), streaming=True)
        results = [dict(depth=prediction["depth"][:, i], depth_conf=prediction["depth_conf"][:, i],
                        pts3d_in_other_view=prediction["points"][:, i], conf=prediction["point_conf"][:, i],
                        camera_pose=prediction["pose"][:, i]) for i in range(len(frames))]
        return StreamVGGTOutput(ress=results, views=frames)


def build_student(teacher_weights, config=None, device="cpu"):
    """Teacher weights -> hybrid student with MOHAWK-initialized recurrent layers."""
    model = StreamVGGT()
    state = torch.load(teacher_weights, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state.get("model", state), strict=True)
    config = config or make_config(model.aggregator.depth)
    install(model.aggregator, build_mixers(model.aggregator, config, initialize=True))
    student = LinearVGGT(config)
    student.load_state_dict(model.state_dict(), strict=True)
    return student.to(device)


def export_student(model, config, path, metadata=None):
    """Self-contained checkpoint; reload does not need the teacher."""
    for i in config["mamba_layers"]:
        if not isinstance(model.aggregator.global_blocks[i].attn, MambaMixer):
            raise ValueError(f"Global block {i} is not a MambaMixer")
    for i in config["sparse_layers"]:
        if not isinstance(model.aggregator.global_blocks[i].attn, AnchorTopKAttention):
            raise ValueError(f"Global block {i} is not AnchorTopKAttention")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=FORMAT, hybrid_config=config, metadata=metadata or {},
                   model={k: v.detach().cpu() for k, v in model.state_dict().items()})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_student(path, device="cpu", policy=None, top_k=None):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("format") != FORMAT:
        raise ValueError(f"Expected a {FORMAT} checkpoint")
    model = LinearVGGT(payload["hybrid_config"])
    model.load_state_dict(payload["model"], strict=True)
    model.set_sparse(policy, top_k)
    model.checkpoint_metadata = payload.get("metadata", {})
    return model.eval().requires_grad_(False).to(device)


HybridStreamVGGT = LinearVGGT  # name used during development


def from_pretrained(name_or_path, device="cpu", filename="linearvggt.pt", **kwargs):
    """Load a released checkpoint from a local file or from a Hugging Face Hub repository id."""
    path = Path(name_or_path)
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=str(name_or_path), filename=filename)
    return load_student(path, device, **kwargs)
