"""
Discovery script — intercepts all JSON API calls made by Caliente and PlayDoit
while navigating to their football odds pages.

Run this locally (not in the cloud container):
    pip install playwright playwright-stealth
    playwright install chromium
    python scripts/discover_odds_endpoints.py

Output:
    scripts/discovered/caliente_calls.json
    scripts/discovered/caliente_screenshot.png
    scripts/discovered/playdoit_calls.json
    scripts/discovered/playdoit_screenshot.png

Then share those files so we can build the real scraper.
"""

import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import async_playwright, Request, Response

try:
    from playwright_stealth import stealth_async
    HAS_STEALTH = True
except ImportError:
    HAS_STEALTH = False
    print("WARNING: playwright-stealth not installed. Cloudflare may block the request.")

OUT_DIR = Path(__file__).parent / "discovered"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = [
    {
        "name": "caliente",
        "url": "https://www.caliente.mx/deportes/futbol",
        "wait_for": 8,
    },
    {
        "name": "playdoit",
        "url": "https://www.playdoit.mx/sports/football",
        "wait_for": 8,
    },
]

# Keywords that suggest an odds/events/markets API response
_INTERESTING = re.compile(
    r"odds|market|event|fixture|selection|match|price|sport|league|competition|bet",
    re.IGNORECASE,
)


async def discover(target: dict) -> None:
    name = target["name"]
    print(f"\n{'='*60}")
    print(f"  Investigating: {name}  ({target['url']})")
    print(f"{'='*60}")

    captured: list[dict] = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,  # visible window so Cloudflare sees a real browser
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="es-MX",
            timezone_id="America/Mexico_City",
        )
        page = await context.new_page()

        if HAS_STEALTH:
            await stealth_async(page)

        # ── Intercept responses ──────────────────────────────────────────
        async def on_response(response: Response) -> None:
            ct = response.headers.get("content-type", "")
            if "json" not in ct:
                return
            url = response.url
            try:
                body = await response.json()
            except Exception:
                return

            body_str = json.dumps(body)
            if not _INTERESTING.search(url) and not _INTERESTING.search(body_str[:500]):
                return  # skip clearly unrelated calls (analytics, auth, etc.)

            entry = {
                "url":     url,
                "method":  response.request.method,
                "status":  response.status,
                "headers": dict(response.request.headers),
                "response_preview": body if _small(body) else _trim(body),
            }
            captured.append(entry)
            print(f"  [JSON] {response.status}  {url}")

        page.on("response", on_response)

        # ── Navigate ─────────────────────────────────────────────────────
        print(f"  Navigating … (browser window will open)")
        try:
            await page.goto(target["url"], wait_until="domcontentloaded", timeout=30_000)
        except Exception as exc:
            print(f"  Navigation warning: {exc}")

        # Wait for dynamic content to load
        print(f"  Waiting {target['wait_for']}s for dynamic content …")
        await asyncio.sleep(target["wait_for"])

        # ── Screenshot ───────────────────────────────────────────────────
        shot_path = OUT_DIR / f"{name}_screenshot.png"
        await page.screenshot(path=str(shot_path), full_page=False)
        print(f"  Screenshot saved → {shot_path}")

        # ── Try scrolling to trigger lazy-loaded odds ─────────────────────
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight / 2)")
        await asyncio.sleep(3)

        await browser.close()

    # ── Save results ──────────────────────────────────────────────────────
    out_path = OUT_DIR / f"{name}_calls.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(captured, f, indent=2, ensure_ascii=False)

    print(f"\n  Captured {len(captured)} JSON calls → {out_path}")
    if not captured:
        print("  !! No JSON calls captured — likely blocked by Cloudflare.")
        print("     Try: open the page manually in the browser window that appeared,")
        print("     solve any CAPTCHA, then re-run the script.")


def _small(obj) -> bool:
    return len(json.dumps(obj)) < 4_000


def _trim(obj) -> dict | list | str:
    """Keep only the first 2 items if it's a list, truncate strings."""
    if isinstance(obj, list):
        return obj[:2]
    if isinstance(obj, dict):
        return {k: v for i, (k, v) in enumerate(obj.items()) if i < 8}
    return str(obj)[:500]


async def main() -> None:
    for target in TARGETS:
        await discover(target)
    print("\nDone. Share the files in scripts/discovered/ to build the scraper.")


if __name__ == "__main__":
    asyncio.run(main())
