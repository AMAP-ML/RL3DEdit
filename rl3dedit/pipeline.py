"""RL3DEdit inference pipeline (diffusers backend).

The released checkpoint (exander/RL3DEdit :: rl3dedit-flux-kontext.safetensors) is in
original (BFL) single-file format. We load it with diffusers' `from_single_file` and plug it
into a `FluxKontextPipeline`. VAE / text encoders come from the base
`black-forest-labs/FLUX.1-Kontext-dev` repo (gated: accept its license first).
"""
import hashlib
import torch
from PIL import Image

from .grid import tile_3x3, split_3x3

# Kontext resolution buckets (W, H), all divisible by 16 — must match training.
KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]

WEIGHTS_REPO = "exander/RL3DEdit"
WEIGHTS_FILE = "rl3dedit-flux-kontext.safetensors"
BASE_REPO = "black-forest-labs/FLUX.1-Kontext-dev"


def _pick_bucket(w, h):
    ar = w / h
    return min(KONTEXT_RESOLUTIONS, key=lambda wh: abs(wh[0] / wh[1] - ar))


def _row_seed(base_seed, instruction):
    h = int(hashlib.md5(instruction.encode()).hexdigest()[:8], 16)
    return base_seed ^ h


class RL3DEditPipeline:
    def __init__(self, pipe):
        self.pipe = pipe

    @classmethod
    def from_pretrained(cls, dit_path=None, base_repo=BASE_REPO, device="cuda:0",
                        torch_dtype=torch.bfloat16, cpu_offload=True):
        """Build the FLUX-Kontext pipeline with the RL3DEdit transformer.

        dit_path  : RL3DEdit transformer .safetensors. If None, downloaded from
                    Hugging Face `exander/RL3DEdit`.
        base_repo : repo (or local diffusers dir) providing VAE + text encoders,
                    default `black-forest-labs/FLUX.1-Kontext-dev` (gated — accept
                    its license on HF first).
        cpu_offload : enable model CPU offload to fit in ~24-32GB VRAM.
        """
        from diffusers import FluxKontextPipeline, FluxTransformer2DModel

        if dit_path is None:
            from huggingface_hub import hf_hub_download
            print(f"[rl3dedit] downloading RL3DEdit transformer from {WEIGHTS_REPO} ...")
            dit_path = hf_hub_download(repo_id=WEIGHTS_REPO, filename=WEIGHTS_FILE)

        print("[rl3dedit] loading transformer via from_single_file ...")
        transformer = FluxTransformer2DModel.from_single_file(
            dit_path, torch_dtype=torch_dtype)

        print(f"[rl3dedit] building FluxKontextPipeline from {base_repo} "
              "(gated; accept its license on HF if this fails) ...")
        pipe = FluxKontextPipeline.from_pretrained(
            base_repo, transformer=transformer, torch_dtype=torch_dtype)

        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        return cls(pipe)

    def edit(self, image, instruction, seed=42, num_inference_steps=28,
             guidance_scale=2.5):
        """Edit a 3x3 grid.

        image : a PIL 3x3 grid, or a list of 9 PIL views (auto-tiled).
        Returns the edited 3x3 grid (PIL.Image).
        """
        if isinstance(image, (list, tuple)):
            grid = tile_3x3(list(image))
        else:
            grid = image.convert("RGB")

        tw, th = _pick_bucket(*grid.size)
        if grid.size != (tw, th):
            grid = grid.resize((tw, th), Image.LANCZOS)

        out = self.pipe(
            image=grid,
            prompt=instruction,
            height=th, width=tw,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(_row_seed(seed, instruction)),
        )
        return out.images[0]

    def edit_and_split(self, image, instruction, **kw):
        """Edit and also return the 9 individual edited views."""
        grid = self.edit(image, instruction, **kw)
        return grid, split_3x3(grid)
