"""RL3DEdit inference CLI.

Edit a 3x3 multi-view grid with a text instruction.

Examples:
  # from a pre-tiled 3x3 grid image
  python inference.py --input grid.jpg --instruction "remove the stone base beneath the bear statue" --enhance --output out.jpg

  # from a directory of 9 views (sorted by filename)
  python inference.py --views_dir views/ --instruction "remove the stone base" --output out.jpg --save_views out_views/
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl3dedit import (
    RL3DEditPipeline, enhance_instruction, load_views_from_dir, split_3x3,
)
from PIL import Image


def main():
    ap = argparse.ArgumentParser(description="RL3DEdit 3x3 multi-view editing")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="a pre-tiled 3x3 grid image")
    src.add_argument("--views_dir", help="directory of exactly 9 view images")

    ap.add_argument("--instruction", required=True, help="editing instruction")
    ap.add_argument("--enhance", action="store_true",
                    help="rewrite the instruction for 3x3/3D-consistency "
                         "(LLM if OPENAI_API_KEY set, else template)")
    ap.add_argument("--no_llm", action="store_true",
                    help="with --enhance, force template (skip LLM even if key set)")
    ap.add_argument("--output", default="output.jpg", help="output grid path")
    ap.add_argument("--save_views", default=None,
                    help="optional dir to also save the 9 edited views")

    ap.add_argument("--dit_path", default=None,
                    help="local RL3DEdit transformer .safetensors (default: download from HF)")
    ap.add_argument("--base_repo", default="black-forest-labs/FLUX.1-Kontext-dev",
                    help="repo or local diffusers dir for VAE + text encoders")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no_cpu_offload", action="store_true",
                    help="disable model CPU offload (needs more VRAM, runs faster)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=2.5)
    args = ap.parse_args()

    # load input
    if args.views_dir:
        image = load_views_from_dir(args.views_dir)  # list of 9
        print(f"[rl3dedit] loaded 9 views from {args.views_dir}")
    else:
        image = Image.open(args.input).convert("RGB")
        print(f"[rl3dedit] loaded grid {args.input} {image.size}")

    # enhance instruction
    instruction = args.instruction
    if args.enhance:
        instruction = enhance_instruction(instruction, use_llm=not args.no_llm)
    print(f"[rl3dedit] instruction: {instruction}")

    # build pipeline + run
    pipe = RL3DEditPipeline.from_pretrained(
        dit_path=args.dit_path, base_repo=args.base_repo, device=args.device,
        cpu_offload=not args.no_cpu_offload)
    grid = pipe.edit(image, instruction, seed=args.seed,
                     num_inference_steps=args.steps, guidance_scale=args.guidance)

    grid.save(args.output, quality=95)
    print(f"[rl3dedit] saved edited grid -> {args.output}")

    if args.save_views:
        os.makedirs(args.save_views, exist_ok=True)
        for i, v in enumerate(split_3x3(grid)):
            v.save(os.path.join(args.save_views, f"view_{i}.png"))
        print(f"[rl3dedit] saved 9 views -> {args.save_views}")


if __name__ == "__main__":
    main()
