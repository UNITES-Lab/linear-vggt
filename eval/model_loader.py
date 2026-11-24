"""Model loading shared by the evaluation launchers (mv_recon, video_depth, monodepth, relpose).

  LinearVGGT   a released or self-trained checkpoint (linearvggt.models.linearvggt.load_student)
  StreamVGGT   the teacher, through its official streaming path (KV caches)

Both return, per frame, a dict with depth, depth_conf, pts3d_in_other_view, conf and camera_pose
(pose encoding in the VGGT convention: camera-from-world, first frame = world).
"""
import torch

MODEL_NAMES = ("LinearVGGT", "StreamVGGT")


def add_model_args(parser):
    parser.add_argument("--model_name", default="LinearVGGT", choices=MODEL_NAMES)
    parser.add_argument("--fast_stream", action="store_true",
                        help="LinearVGGT: streaming inference through FastStreamer (CUDA graphs + fused Triton kernels)")
    return parser


def load_eval_model(args, device="cuda"):
    """Build the model selected by the arguments of add_model_args (plus --weights)."""
    if args.model_name == "StreamVGGT":
        if args.fast_stream:
            raise ValueError("--fast_stream is only implemented for LinearVGGT")
        from streamvggt.models.streamvggt import StreamVGGT
        model = StreamVGGT()
        state = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=True)
        return model.eval().to(device)
    from linearvggt.models.linearvggt import load_student
    model = load_student(args.weights, device)
    if args.fast_stream:
        model.enable_fast_stream()
    return model


@torch.no_grad()
def run_inference(model, views):
    """Streaming inference on a list of views (dicts with "img" in [0, 1]); returns dict(views, pred)."""
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.autocast("cuda", dtype=dtype):
        output = model.inference(views)
    return dict(views=output.views, pred=output.ress)


def prefetch_files(paths, workers=32):
    """Read the files once in parallel, so that a sequential image loader hits a warm file-system cache."""
    from concurrent.futures import ThreadPoolExecutor

    def touch(path):
        with open(path, "rb") as f:
            while f.read(1 << 20):
                pass
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(touch, paths))
