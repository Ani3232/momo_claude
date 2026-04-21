# view_image.py
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
import base64
import io
import ollama
from rich.console import Console
from setup import workspace
console = Console()


# ── style helpers ──────────────────────────────────────────────────────
_CORAL   = "#C8603A"
_BULLET  = f"[{_CORAL}]⬤[/{_CORAL}]"
_NEST    = "[dim]  ⎿[/dim]"
# Formats that go through Playwright (rendered in a browser first)
BROWSER_EXTENSIONS = {".svg", ".html", ".htm"}

# Formats loaded directly as images
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

ALL_EXTENSIONS = BROWSER_EXTENSIONS | IMAGE_EXTENSIONS

VISION_MODEL = "gemma4:31b-cloud"


def _render_with_playwright(p: Path, viewport_width: int = 1280, viewport_height: int = 900) -> Image.Image:
    """
    Open any HTML or SVG file in a headless Chromium browser,
    take a full-page PNG screenshot, and return it as a PIL Image.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": viewport_width, "height": viewport_height})
        page.goto(f"file:///{p.resolve()}", wait_until="networkidle")
        png_bytes = page.screenshot(full_page=True)
        browser.close()

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def _compress(img: Image.Image, max_tokens: int) -> tuple[str, str]:
    max_pixels = 1024 * 1024
    quality = 85
    b64 = ""
    resized = img

    for _ in range(6):
        w, h = img.size
        scale = (max_pixels / (w * h)) ** 0.5
        if scale < 1.0:
            resized = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            resized = img

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode()

        if len(b64) // 3 <= max_tokens:
            break

        max_pixels = int(max_pixels * 0.55)
        quality = max(30, quality - 15)

    meta = f"[{resized.size[0]}x{resized.size[1]}px | quality={quality}]"
    return b64, meta


class ViewImageInput(BaseModel):
    path: str = Field(
        description="Absolute or workspace-relative path to the image or HTML/SVG file"
    )
    prompt: str = Field(
        default="",
        description=(
            "What to look for in the image. Be specific. "
            "E.g. 'Check if padding is consistent and identify alignment issues' "
            "or 'What colors are used in the chart and do they match the target palette?'"
        )
    )
    max_tokens: int = Field(
        default=50000,
        description="Compression budget. Use 100000 for high detail, 20000 for quick check"
    )
    viewport_width: int = Field(
        default=1280,
        description="Browser viewport width in pixels, used when rendering HTML/SVG files"
    )
    viewport_height: int = Field(
        default=900,
        description="Browser viewport height in pixels, used when rendering HTML/SVG files"
    )


def view_image(
    path: str,
    prompt: str = "",
    max_tokens: int = 50000,
    viewport_width: int = 1280,
    viewport_height: int = 900,
) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = Path(workspace) / p
    if not p.exists():
        return f"Error: file not found at {p}"

    ext = p.suffix.lower()
    if ext not in ALL_EXTENSIONS:
        return f"Error: unsupported format '{p.suffix}'"

    if ext in BROWSER_EXTENSIONS:
        try:
            img = _render_with_playwright(p, viewport_width, viewport_height)
        except Exception as e:
            return f"Error: failed to render '{p.name}' in browser — {e}"
    else:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            return f"Error: failed to open image '{p.name}' — {e}"

    b64, meta = _compress(img, max_tokens)

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": prompt or "Describe this image in detail. Cover layout, colors, spacing, and all visible elements.",
            "images": [b64]
        }]
    )
    console.print(f"{_BULLET} [bold]Viewing Image[/bold] [dim]{prompt[:40]}...[/dim]")
    console.print(f"{_NEST} [dim]viewing image {path}[/dim]")

    description = response.message.content
    return f"Image analysis {meta}:\n\n{description}"


view_image_tool = StructuredTool(
    name="view_image",
    func=view_image,
    args_schema=ViewImageInput,
    description=(
        "View and analyze an image or web file using a vision model. "
        "Returns a detailed description based on your prompt. "
        "Always call list_directory first to find the file. "
        "Use the prompt argument to direct what the vision model should focus on — "
        "the more specific your prompt, the more useful the analysis. "
        "Supports raster formats (PNG, JPEG, WEBP, BMP, TIFF) and browser-rendered "
        "formats (SVG, HTML) — the latter are screenshotted via headless Chromium "
        "before analysis, so what the vision model sees is always a PNG of the "
        "actual rendered output."
    )
)