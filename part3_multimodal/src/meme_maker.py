"""Classic meme renderer: bold uppercase top/bottom text with a black outline, auto-sized and wrapped."""
from __future__ import annotations

import textwrap
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

REPO = Path(__file__).resolve().parents[2]
FONT_CANDIDATES = [
    REPO / "part3_multimodal" / "assets" / "Anton-Regular.ttf",          # optional bundled meme font
    Path("C:/Windows/Fonts/impact.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),       # Docker image (fonts-dejavu-core)
    Path("/usr/share/fonts/truetype/msttcorefonts/Impact.ttf"),
]
MAX_SIDE = 900


@lru_cache(maxsize=64)
def _font(size: int):
    for p in FONT_CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def _fit(draw, text, max_w, max_h, start):
    """Largest font size (and wrapping) so the text block fits in max_w x max_h."""
    size = start
    while size >= 14:
        font = _font(size)
        avg_char = max(1, draw.textlength("ABCDEFGHIJKLMNOPQRSTUVWXYZ", font=font) / 26)
        width_chars = max(8, int(max_w / avg_char))
        lines = textwrap.wrap(text, width=width_chars) or [""]
        line_h = font.getbbox("Ay")[3] + size // 8
        if max(draw.textlength(l, font=font) for l in lines) <= max_w and line_h * len(lines) <= max_h:
            return font, lines, line_h
        size -= 2
    font = _font(14)
    return font, textwrap.wrap(text, width=40)[:3], font.getbbox("Ay")[3] + 2


def _draw_block(draw, text, img_w, y0, max_h, from_bottom=False):
    if not text:
        return
    margin = int(img_w * 0.04)
    font, lines, line_h = _fit(draw, text.upper(), img_w - 2 * margin, max_h, start=max(28, img_w // 9))
    stroke = max(2, font.size // 12)
    total = line_h * len(lines)
    y = y0 - total if from_bottom else y0
    for line in lines:
        w = draw.textlength(line, font=font)
        draw.text(((img_w - w) / 2, y), line, font=font, fill="white", stroke_width=stroke, stroke_fill="black")
        y += line_h


def make_meme(img: Image.Image, top: str = "", bottom: str = "") -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    if min(img.size) < 320:                               # tiny crops: upscale so the text stays readable
        s = 320 / min(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.LANCZOS)
    out = img.copy()
    d = ImageDraw.Draw(out)
    w, h = out.size
    pad = int(h * 0.03)
    _draw_block(d, top, w, pad, max_h=int(h * 0.30))
    _draw_block(d, bottom, w, h - pad, max_h=int(h * 0.30), from_bottom=True)
    return out


# ---------------------------------------------------------------- modern style: white caption band above the photo
BAR_FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),           # Docker image (fonts-dejavu-core)
]


@lru_cache(maxsize=32)
def _bar_font(size: int):
    for p in BAR_FONT_CANDIDATES:
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size=size)


def make_bar_meme(img: Image.Image, lines: list[str]) -> Image.Image:
    """Twitter/Instagram-style meme: black sentence-case text on a white band, photo underneath."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    target_w = 720
    s = target_w / img.width
    img = img.resize((target_w, max(1, int(img.height * s))), Image.LANCZOS)
    if img.height > 900:                                  # keep tall photos reasonable
        img = img.crop((0, (img.height - 900) // 2, target_w, (img.height - 900) // 2 + 900))
    font = _bar_font(40)
    pad, gap = 28, 10
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    char_w = max(1, probe.textlength("abcdefghijklmnopqrstuvwxyz", font=font) / 26)
    wrapped = [w for line in lines if line.strip() for w in (textwrap.wrap(line.strip(), width=int((target_w - 2 * pad) / char_w)) or [""])]
    line_h = font.getbbox("Ay")[3] + gap
    band_h = pad * 2 + line_h * max(1, len(wrapped)) - gap
    out = Image.new("RGB", (target_w, band_h + img.height), "white")
    d = ImageDraw.Draw(out)
    y = pad
    for line in wrapped:
        d.text((pad, y), line, font=font, fill="#111111")
        y += line_h
    out.paste(img, (0, band_h))
    return out


def render(img: Image.Image, caption: dict) -> Image.Image:
    """Draw a caption dict from captioner.generate_caption in its style ("classic" or "bar")."""
    if caption.get("style") == "bar":
        return make_bar_meme(img, [caption.get("top", ""), caption.get("bottom", "")])
    return make_meme(img, caption.get("top", ""), caption.get("bottom", ""))


if __name__ == "__main__":
    import sys
    m = make_meme(Image.open(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "top text",
                  sys.argv[3] if len(sys.argv) > 3 else "bottom text")
    m.save(sys.argv[4] if len(sys.argv) > 4 else "meme_test.jpg", quality=92)
