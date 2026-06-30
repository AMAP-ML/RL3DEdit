"""Instruction enhancement for RL3DEdit.

The model was trained on editing instructions phrased for a 3x3 multi-view grid,
e.g. "Remove the rectangular stone base beneath the bear statue in the 3x3 grid,
ensuring 3D consistency across all nine views ...". A short user instruction like
"remove the stone base" works best when rewritten into that distribution.

Two modes:
  * LLM rewrite (recommended) via any OpenAI-compatible endpoint, configured by
    environment variables — no keys are hardcoded.
  * Template fallback (no API needed): wraps the instruction with the grid/3D
    consistency phrasing.
"""
import os

_SYSTEM = (
    "You rewrite a user's image-editing instruction so it targets a 3x3 grid that "
    "tiles 9 views of the same 3D scene. Keep the user's intent EXACTLY; only adapt "
    "the phrasing. The rewrite MUST: (1) be a single English sentence; (2) mention "
    "that the edit applies to the 3x3 grid; (3) require the edit to stay 3D-consistent "
    "across all nine views (consistent position/orientation/scale); (4) ask to keep "
    "everything else (geometry, background) unchanged. Do not invent new edits. "
    "Output only the rewritten instruction, no preamble."
)

_TEMPLATE = (
    "{instr} in the 3x3 grid, maintaining consistent 3D positioning, orientation, "
    "and scale across all nine views, while keeping the rest of the scene and "
    "background unchanged."
)


def _template_enhance(user_text):
    instr = user_text.strip().rstrip(".")
    return _TEMPLATE.format(instr=instr)


def enhance_instruction(user_text, use_llm=True, verbose=True):
    """Return an enhanced editing instruction.

    If use_llm and OPENAI_API_KEY is set, rewrite via an OpenAI-compatible API:
      OPENAI_API_KEY        (required for LLM mode)
      OPENAI_BASE_URL       (optional, for custom/compatible endpoints)
      RL3DEDIT_ENHANCE_MODEL (optional, default "gpt-4o-mini")
    Otherwise fall back to a deterministic template.
    """
    if not use_llm:
        return _template_enhance(user_text)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        if verbose:
            print("[enhance] OPENAI_API_KEY not set -> using template fallback. "
                  "Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL, "
                  "RL3DEDIT_ENHANCE_MODEL) for LLM rewriting.")
        return _template_enhance(user_text)

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )
        model = os.environ.get("RL3DEDIT_ENHANCE_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user_text.strip()},
            ],
            temperature=0.3,
        )
        out = resp.choices[0].message.content.strip()
        if verbose:
            print(f"[enhance] LLM ({model}) -> {out}")
        return out or _template_enhance(user_text)
    except Exception as e:
        if verbose:
            print(f"[enhance] LLM rewrite failed ({e}); using template fallback.")
        return _template_enhance(user_text)
