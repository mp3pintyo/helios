"""Assembles the showcase site into site/dist: the page, the wasm engine and the media.

usage: python site/build.py [MEDIA_DIR]

MEDIA_DIR defaults to site/media-src (written by site/capture.py): stills/*.webp and
videos/*.mp4 + poster *.webp. `web/pkg` must be built first (./build-web.ps1).
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
DIST = os.path.join(SITE, 'dist')

SKELETON = ('<!doctype html><html><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{padding-top:env(safe-area-inset-top);padding-bottom:env(safe-area-inset-bottom)}'
            'body{margin:0;font-size:14px}img{max-width:100%}[hidden]{display:none!important}</style>'
            '</head><body>{content}</body></html>')

page = open(os.path.join(SITE, 'index.html'), encoding='utf-8').read()
shutil.rmtree(DIST, ignore_errors=True)
os.makedirs(os.path.join(DIST, 'media'))
shutil.copytree(os.path.join(ROOT, 'web', 'pkg'), os.path.join(DIST, 'pkg'))
open(os.path.join(DIST, 'index.html'), 'w', encoding='utf-8').write(page)
open(os.path.join(DIST, 'preview.html'), 'w', encoding='utf-8').write(SKELETON.replace('{content}', page))

# Copy exactly the media the page references.
wanted = sorted(set(re.findall(r'media/([\w.-]+)', page)))
missing = []
for name in wanted:
    for sub in ('stills', 'videos'):
        src = os.path.join(MEDIA, sub, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DIST, 'media', name))
            break
    else:
        missing.append(name)
total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(DIST) for f in fs)
print(f'dist: {len(wanted) - len(missing)} media files, {total / 2**20:.1f} MiB total')
if missing:
    print('missing:', ', '.join(missing))
