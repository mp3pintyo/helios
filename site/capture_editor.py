"""Records the showcase site's editor media in headless Chromium with WebGPU.

usage: uv run --with playwright --with pillow python -I site/capture_editor.py [stills|video|all] [--out DIR]

Serves web/ (build it first with ./build-web.ps1) on a random local port, opens editor.html and
drives it through `globalThis.heliosEditor` and real mouse input. Stills land in
<out>/stills/editor_*.png (+ .webp), the video in <out>/videos/editor_tour.mp4 (+ poster .webp);
<out> defaults to site/media-src. Needs ffmpeg. HELIOS_CHROMIUM overrides the Chromium binary.
"""
import base64
import os
import random
import shutil
import socket
import subprocess
import sys
import time

from PIL import Image
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding='utf-8')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(REPO, 'site', 'media-src')
CHROMIUM = os.environ.get('HELIOS_CHROMIUM', os.path.expandvars(r'%LOCALAPPDATA%\ms-playwright\chromium-1248\chrome-win64\chrome.exe'))
ARGS = ['--enable-unsafe-webgpu', '--enable-gpu', '--ignore-gpu-blocklist', '--use-angle=d3d11']
FFMPEG = shutil.which('ffmpeg') or r'c:\Apps\ffmpeg-shared\bin\ffmpeg.exe'
SIZE = {'width': 1600, 'height': 900}

# Page-side helpers: the editor object, the engine module and screen positions of world points.
HELPERS = """
globalThis.cap = {
  ed: () => globalThis.heliosEditor,
  engine: () => import('./pkg/helios_demo.js'),
  // The first instance whose mesh name matches, nearest to the camera among those on screen.
  find(pattern) {
    const ed = this.ed(), re = new RegExp(pattern, 'i'), rect = ed.canvas.getBoundingClientRect();
    let best = null;
    for (let i = 0; i < ed.state.instances.mesh.length; i++) {
      if (!re.test(ed.meshName(i)) || !ed.exists('i:' + i)) continue;
      const t = ed.instance(i)?.transform?.t || ed.instance(i)?.t;
      if (!t) continue;
      const p = ed.view.project(t);
      if (!p || p[0] < rect.width * 0.2 || p[0] > rect.width * 0.8 || p[1] < rect.height * 0.15 || p[1] > rect.height * 0.85) continue;
      const d = Math.hypot(t[0] - ed.view.eye[0], t[1] - ed.view.eye[1], t[2] - ed.view.eye[2]);
      if (!best || d < best.d) best = { i, d, t };
    }
    return best;
  },
  // Page coordinates of a world point.
  screen(t) {
    const ed = this.ed(), rect = ed.canvas.getBoundingClientRect(), p = ed.view.project(t);
    return p ? [rect.left + p[0], rect.top + p[1]] : null;
  },
};
"""


def free_port():
    while True:
        port = random.randint(18000, 18999)
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port


def webp(png, quality=88):
    Image.open(png).convert('RGB').save(png[:-4] + '.webp', quality=quality, method=6)


class Session:
    def __init__(self, browser, base):
        self.context = browser.new_context(viewport=SIZE)
        self.page = self.context.new_page()
        self.base = base
        self.errors = []
        self.page.on('console', lambda m: m.type == 'error' and self.errors.append(m.text))
        self.page.on('pageerror', lambda e: self.errors.append(str(e)))
        self.frames = []
        self.cdp = None

    def open(self, scene):
        self.page.goto(f'{self.base}/editor.html?scene={scene}')
        self.page.wait_for_function('globalThis.heliosEditorMounted === true', timeout=240000)
        self.page.evaluate(HELPERS)
        self.page.wait_for_function('globalThis.heliosEditor.view !== undefined && globalThis.heliosEditor.view !== null', timeout=30000)
        self.page.wait_for_timeout(2500)

    def js(self, code, arg=None):
        return self.page.evaluate(code, arg)

    def shot(self, name):
        out = os.path.join(ROOT, 'stills')
        os.makedirs(out, exist_ok=True)
        png = os.path.join(out, name + '.png')
        self.page.screenshot(path=png)
        webp(png)
        print('still', name)

    def record(self, on):
        """Starts / stops a screencast of the page (JPEG frames with their timestamps)."""
        if on:
            self.cdp = self.context.new_cdp_session(self.page)

            def frame(params):
                self.frames.append((params['metadata']['timestamp'], params['data']))
                try:
                    self.cdp.send('Page.screencastFrameAck', {'sessionId': params['sessionId']})
                except Exception:
                    pass  # a frame that arrives while the page closes

            self.cdp.on('Page.screencastFrame', frame)
            self.cdp.send('Page.startScreencast', {'format': 'jpeg', 'quality': 92, 'maxWidth': SIZE['width'], 'maxHeight': SIZE['height']})
        else:
            self.cdp.send('Page.stopScreencast')

    def drag(self, start, end, steps=40, pause=25):
        page = self.page
        page.mouse.move(*start)
        page.mouse.down()
        for k in range(1, steps + 1):
            f = k / steps
            page.mouse.move(start[0] + (end[0] - start[0]) * f, start[1] + (end[1] - start[1]) * f)
            page.wait_for_timeout(pause)
        page.mouse.up()


def select_sculpture(s, click=False):
    hit = s.js("() => cap.find('^sculpture')")
    if not hit:
        raise RuntimeError('no sculpture on screen')
    if click:
        x, y = s.js('(t) => cap.screen(t)', hit['t'])
        s.page.mouse.click(x, y)
        s.page.wait_for_timeout(600)
    if s.js('() => cap.ed().selection.length') == 0:
        s.js('(i) => cap.ed().select(["i:" + i])', hit['i'])
    return hit


def add_flowers(s, fly_ms=0):
    """Drops a 'Flowers in meadows' PCG volume on the meadow in the foreground, then
    brings the camera down into the meadow (over `fly_ms`, or at once)."""
    s.js("""async (ms) => {
      const ed = cap.ed();
      const box = ed.canvas.getBoundingClientRect();
      // The meadow in the foreground of the valley's opening view.
      const at = await ed.placementPoint(box.width * 0.45, box.height * 0.85);
      ed.addPcg('flowers-in-meadows', at);
      await new Promise((r) => setTimeout(r, 1500));
      ed.select([]);
      const rect = ed.canvas.getBoundingClientRect();
      const d = ed.view.ray(rect.width / 2, rect.height / 2).dir;
      const n = Math.hypot(d[0], d[2]), fx = d[0] / n, fz = d[2] / n;
      const pose = (back, h, ahead, ah) => [at[0] - fx * back, h, at[2] - fz * back, at[0] + fx * ahead, ah, at[2] + fz * ahead];
      const from = pose(26, 8, 0, 0), to = pose(7, 1.5, 4, 0.2);
      const m = await cap.engine();
      const start = performance.now();
      await new Promise((done) => {
        const step = () => {
          const f = ms > 0 ? Math.min((performance.now() - start) / ms, 1) : 1;
          const e = f * f * (3 - 2 * f);
          m.heliosLookAt(...from.map((v, k) => v + (to[k] - v) * e));
          if (f < 1) requestAnimationFrame(step); else done();
        };
        step();
      });
    }""", fly_ms)
    s.page.wait_for_timeout(2500)


def stills(browser, base):
    s = Session(browser, base)
    s.open('valley')
    select_sculpture(s)
    s.page.wait_for_timeout(1500)
    s.shot('editor_valley')
    s.js('() => cap.ed().select([])')
    add_flowers(s)
    s.shot('editor_pcg')

    s.open('gallery')
    s.js("""() => {
      const ed = cap.ed();
      const lava = (ed.state.graphs || []).find((g) => /lava/i.test(g.name));
      ed.graphEditor.open(lava ? lava.name : undefined);
    }""")
    s.page.wait_for_timeout(2500)
    s.shot('editor_graph')

    s.open('forest')
    hit = s.js("() => cap.find('conifer needles')") or s.js("() => cap.find('conifer')")
    if hit:
        s.js('(i) => cap.ed().select(["i:" + i])', hit['i'])
    s.page.wait_for_timeout(2000)
    s.shot('editor_forest')
    report(s)
    s.context.close()


def video(browser, base):
    tmp = os.path.join(ROOT, 'tmp_editor_video')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    frames = []
    # Segment 1: the valley — click the sculpture, drag it along its X arrow, undo, then a PCG meadow.
    s = Session(browser, base)
    s.open('valley')
    s.record(True)
    s.page.wait_for_timeout(800)
    hit = select_sculpture(s, click=True)
    s.page.wait_for_timeout(1000)
    centre = s.js('(t) => cap.screen(t)', hit['t'])
    tip = s.js('(t) => cap.screen([t[0] + 1, t[1], t[2]])', hit['t'])
    if centre and tip:
        dx, dy = tip[0] - centre[0], tip[1] - centre[1]
        n = max((dx * dx + dy * dy) ** 0.5, 1e-3)
        dx, dy = dx / n, dy / n
        grab = (centre[0] + dx * 55, centre[1] + dy * 55)
        s.drag(grab, (grab[0] + dx * 160, grab[1] + dy * 160), steps=60)
    s.page.wait_for_timeout(1200)
    s.js('() => cap.ed().undo()')
    s.page.wait_for_timeout(800)
    s.js('() => cap.ed().select([])')
    add_flowers(s, fly_ms=3500)
    s.page.wait_for_timeout(500)
    s.record(False)
    report(s)
    frames.append(s.frames)
    s.context.close()
    # Segment 2: the gallery with the lava material graph open (the lava flows in the viewport).
    s = Session(browser, base)
    s.open('gallery')
    s.record(True)
    s.page.wait_for_timeout(1000)
    s.js("""() => {
      const ed = cap.ed();
      const lava = (ed.state.graphs || []).find((g) => /lava/i.test(g.name));
      ed.graphEditor.open(lava ? lava.name : undefined);
    }""")
    s.page.wait_for_timeout(5000)
    s.record(False)
    report(s)
    frames.append(s.frames)
    s.context.close()

    # Frames keep their own timestamps: each is shown until the next one arrives.
    listing = []
    for segment in frames:
        for k, (stamp, data) in enumerate(segment):
            path = os.path.join(tmp, f'f{len(listing):05d}.jpg')
            open(path, 'wb').write(base64.b64decode(data))
            duration = segment[k + 1][0] - stamp if k + 1 < len(segment) else 1 / 30
            listing.append((path, max(duration, 0.001)))
    concat = os.path.join(tmp, 'frames.txt')
    with open(concat, 'w') as f:
        for path, duration in listing:
            f.write(f"file '{path}'\nduration {duration:.4f}\n")
        f.write(f"file '{listing[-1][0]}'\n")
    out = os.path.join(ROOT, 'videos')
    os.makedirs(out, exist_ok=True)
    mp4 = os.path.join(out, 'editor_tour.mp4')
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', concat,
                    '-vf', 'fps=30,scale=1600:900', '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', mp4], check=True)
    poster = os.path.join(out, 'editor_tour.png')
    shutil.copy2(listing[len(listing) // 6][0], poster[:-4] + '.jpg')
    Image.open(poster[:-4] + '.jpg').save(poster)
    os.remove(poster[:-4] + '.jpg')
    webp(poster, 80)
    shutil.rmtree(tmp, ignore_errors=True)
    print('video editor_tour', len(listing), 'frames', os.path.getsize(mp4) // 1024, 'KiB')


def report(s):
    if s.errors:
        print('console errors:', *s.errors[:10], sep='\n  ')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'all'
    port = free_port()
    server = subprocess.Popen([sys.executable, 'serve.py', str(port), 'web'], cwd=REPO,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0:
                    break
            time.sleep(0.1)
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=CHROMIUM, headless=True, args=ARGS)
            base = f'http://127.0.0.1:{port}'
            if what in ('stills', 'all'):
                stills(browser, base)
            if what in ('video', 'all'):
                video(browser, base)
            browser.close()
    finally:
        server.terminate()
