"""Publishes site/dist to the gh-pages branch of origin (GitHub Pages).

usage: python site/deploy_pages.py

Run site/build.py first. GitHub Pages serves files as-is, so the page goes out as
dist/preview.html (the page wrapped in its document skeleton) renamed to index.html.
The branch is rebuilt from scratch on every deploy (one commit, force-pushed), so the
media history does not pile up in the repository.
"""
import os
import shutil
import subprocess
import sys
import tempfile

SITE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SITE)
DIST = os.path.join(SITE, 'dist')


def git(*args, cwd):
    subprocess.run(['git', *args], cwd=cwd, check=True)


if not os.path.exists(os.path.join(DIST, 'preview.html')):
    sys.exit('site/dist is missing; run python site/build.py first')

remote = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=ROOT, check=True,
                        capture_output=True, text=True).stdout.strip()

with tempfile.TemporaryDirectory() as out:
    shutil.copytree(DIST, out, dirs_exist_ok=True)
    os.replace(os.path.join(out, 'preview.html'), os.path.join(out, 'index.html'))
    open(os.path.join(out, '.nojekyll'), 'w').close()
    git('init', '-q', '-b', 'gh-pages', cwd=out)
    git('add', '-A', cwd=out)
    git('commit', '-q', '-m', 'Deploy showcase site', cwd=out)
    git('push', '-f', remote, 'gh-pages', cwd=out)
print('pushed gh-pages')
