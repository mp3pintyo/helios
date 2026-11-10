"""Assembles the showcase site into site/dist: the page, the wasm engine, the editor and the media.

usage: python site/build.py [MEDIA_DIR]

MEDIA_DIR defaults to site/media-src (written by site/capture.py): stills/*.webp, videos/*.mp4 +
poster *.webp and models/*.glb (glTF models the page loads into the engine; a model missing
there is taken from HELIOS_ASSETS/<name>/<name>.glb). `web/pkg` must be built first
(./build-web.ps1). The editor (web/editor.html + web/editor/) is shipped next to the page, so the
page can link to editor.html.
dist/index.html is the page content as published (the host adds the document skeleton);
dist/preview.html wraps it in an equivalent skeleton for local testing (python serve.py 8080 site/dist).
"""
import os
import re
import shutil
import sys

SITE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SITE)
MEDIA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SITE, 'media-src')
ASSETS = os.environ.get('HELIOS_ASSETS', os.path.join(ROOT, 'assets'))
DIST = os.path.join(SITE, 'dist')

SKELETON = ('<!doctype html><html><head><meta charset="utf-8"><link rel="icon" href="data:,">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}'
            'body{margin:0;font-size:14px}img{max-width:100%}[hidden]{display:none!important}</style>'
            '</head><body>{content}</body></html>')

page = open(os.path.join(SITE, 'index.html'), encoding='utf-8').read()
shutil.rmtree(DIST, ignore_errors=True)
os.makedirs(os.path.join(DIST, 'media'))
shutil.copytree(os.path.join(ROOT, 'web', 'pkg'), os.path.join(DIST, 'pkg'))
shutil.copy2(os.path.join(ROOT, 'web', 'editor.html'), os.path.join(DIST, 'editor.html'))
shutil.copytree(os.path.join(ROOT, 'web', 'editor'), os.path.join(DIST, 'editor'))
open(os.path.join(DIST, 'index.html'), 'w', encoding='utf-8').write(page)
open(os.path.join(DIST, 'preview.html'), 'w', encoding='utf-8').write(SKELETON.replace('{content}', page))


def source(name):
    """Where a referenced media file comes from, or None."""
    for sub in ('stills', 'videos', 'models'):
        path = os.path.join(MEDIA, sub, name)
        if os.path.exists(path):
            return path
    stem, ext = os.path.splitext(name)
    path = os.path.join(ASSETS, stem, name)
    return path if ext == '.glb' and os.path.exists(path) else None


# Copy exactly the media the page references (commented-out sections, such as placeholders for
# chapters still being built, don't count).
live = re.sub(r'<!--.*?-->', '', page, flags=re.S)
wanted = sorted(set(re.findall(r'media/([\w.-]+)', live)))
missing = []
for name in wanted:
    src = source(name)
    if src:
        shutil.copy2(src, os.path.join(DIST, 'media', name))
    else:
        missing.append(name)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DIST) for f in fs)
print(f'dist: {len(wanted) - len(missing)} media files, {total / 2**20:.1f} MiB total')
if missing:
    print('missing:', ', '.join(missing))
