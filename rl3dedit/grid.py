"""3x3 multi-view grid utilities.

The model edits a 3x3 grid that tiles 9 views (row-major) of the same scene.
These helpers convert between a list of 9 views and a single grid image.
"""
from pathlib import Path
from PIL import Image

GRID = 3  # 3x3


def tile_3x3(images):
    """Tile 9 PIL images into a single 3x3 grid (row-major).

    All views are resized to the first view's size (the model was trained on
    equal-sized cells). Returns one PIL.Image.
    """
    if len(images) != 9:
        raise ValueError(f"tile_3x3 expects exactly 9 images, got {len(images)}")
    images = [im.convert("RGB") for im in images]
    cw, ch = images[0].size
    images = [im if im.size == (cw, ch) else im.resize((cw, ch), Image.LANCZOS)
              for im in images]
    grid = Image.new("RGB", (cw * GRID, ch * GRID))
    for i, im in enumerate(images):
        r, c = divmod(i, GRID)
        grid.paste(im, (c * cw, r * ch))
    return grid


def split_3x3(grid):
    """Split a 3x3 grid image back into 9 views (row-major).

    Matches the cell-cropping used in evaluation: cw, ch = W//3, H//3.
    """
    W, H = grid.size
    cw, ch = W // GRID, H // GRID
    views = []
    for i in range(9):
        r, c = divmod(i, GRID)
        views.append(grid.crop((c * cw, r * ch, c * cw + cw, r * ch + ch)))
    return views


def load_views_from_dir(dir_path):
    """Load 9 views from a directory, sorted by filename.

    Accepts common image extensions. Raises if not exactly 9 are found.
    """
    d = Path(dir_path)
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if len(files) != 9:
        raise ValueError(
            f"expected 9 view images in {dir_path}, found {len(files)}: "
            f"{[f.name for f in files]}")
    return [Image.open(f).convert("RGB") for f in files]
