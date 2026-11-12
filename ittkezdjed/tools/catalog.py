"""Regenerates ittkezdjed/catalog/<scene>.json: what each built-in scene holds that a site can address
by name (materials, meshes, top-level actors, environment, render settings, content items).

    uv run --with playwright==1.56.0 python -I D:/AI/Unreal/ittkezdjed/tools/catalog.py [scene ...]

It serves the repository root on a free local port and opens tools/catalog.html in headless
Chromium with WebGPU. web/pkg must be built (./build-web.ps1). The browser is HELIOS_CHROMIUM if set,
else Playwright's cached Chromium, else the installed Chrome.
"""
import functools
import glob
import http.server
import json
import os
import sys
import threading
import time

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(HERE)
ROOT = os.path.dirname(KIT)
SCENES = sys.argv[1:] or ["valley", "cathedral", "forest", "gallery"]
ARGS = ["--enable-unsafe-webgpu", "--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"]


class Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def chromium():
    if os.environ.get("HELIOS_CHROMIUM"):
        return os.environ["HELIOS_CHROMIUM"]
    cached = sorted(glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\ms-playwright\chromium-*\chrome-win64\chrome.exe")))
    return cached[-1] if cached else None


def main():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=ROOT))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/ittkezdjed/tools/catalog.html"
    out_dir = os.path.join(KIT, "catalog")
    os.makedirs(out_dir, exist_ok=True)
    with sync_playwright() as p:
        exe = chromium()
        browser = p.chromium.launch(headless=True, args=ARGS, **({"executable_path": exe} if exe else {"channel": "chrome"}))
        for scene in SCENES:
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            t0 = time.time()
            page.goto(f"{base}?scene={scene}")
            page.wait_for_function("window.catalog !== undefined", timeout=180_000)
            catalog = page.evaluate("window.catalog")
            page.close()
            if "error" in catalog:
                print(f"{scene}: ERROR {catalog['error']} {errors}")
                continue
            with open(os.path.join(out_dir, f"{scene}.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(catalog, f, indent=1, ensure_ascii=False)
                f.write("\n")
            print(f"{scene}: {len(catalog['materials'])} materials, {len(catalog['meshes'])} meshes, "
                  f"{catalog['actorCount']} actors, {catalog['lights']} lights ({time.time() - t0:.0f} s)")
        browser.close()
    server.shutdown()


if __name__ == "__main__":
    main()
