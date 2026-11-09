"""Captures the showcase website's media from the native Helios build.

usage: python site/capture.py [stills|videos|all] [--only NAME[,NAME...]] [--size WxH] [--out DIR]

Stills land in media/stills/<name>.png (+ .webp), videos in media/videos/<name>.mp4 (+ poster .webp).
Every still is a fixed camera and frame count, so reruns are reproducible.
"""
import os
import shutil
import subprocess
import sys

from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Output directory (stills/, videos/): --out DIR, default site/media-src.
ROOT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else os.path.join(REPO, 'site', 'media-src')
EXE = os.path.join(REPO, 'target', 'release', 'helios.exe')
# glTF test assets (Sponza, DamagedHelmet from the Khronos glTF Sample Assets).
ASSETS = os.environ.get('HELIOS_ASSETS', r'D:\AI\helios-work\assets')
FFMPEG = shutil.which('ffmpeg') or r'c:\Apps\ffmpeg-shared\bin\ffmpeg.exe'
SPONZA = os.path.join(ASSETS, 'Sponza', 'Sponza.gltf')
HELMET = os.path.join(ASSETS, 'DamagedHelmet', 'DamagedHelmet.glb')

LAKE_DOWN = ['--eye', '-5,6,10', '--at', '-5,-1,24']
AISLE = ['--eye', '31,1.5,10.5', '--at', '38,3,13.5']
GLASS = ['--eye', '14,1.7,10.25', '--at', '-0.6,3,10.25']
SPONZA_VIEW = ['--gltf', SPONZA, '--eye', '-9,1.6,0', '--at', '9,3.5,0']

# name: arguments. Pairs share a prefix and end in _off / _on.
STILLS = {
    # Hero shots
    'hero_valley': ['--scene', 'valley'],
    'hero_cathedral': ['--scene', 'cathedral'],
    'hero_forest': ['--scene', 'forest'],
    'hero_cathedral_night': ['--scene', 'cathedral', '--sun', '200,-30'],
    'hero_valley_night': ['--scene', 'valley', '--sun', '200,-30'],
    'hero_valley_dawn': ['--scene', 'valley', '--sun', '95,4'],
    'hero_lake_high_sun': ['--scene', 'valley', '--sun', '52,50'] + LAKE_DOWN,
    'hero_sponza': SPONZA_VIEW,
    'hero_helmet': ['--gltf', HELMET],
    # Virtual geometry (debug views)
    'geo_clusters': ['--scene', 'forest', '--view', '1'],
    'geo_triangles': ['--scene', 'cathedral', '--view', '2'],
    'geo_lod': ['--scene', 'valley', '--view', '3'],
    'geo_raster_path': ['--scene', 'forest', '--view', '9'],
    # Shadows
    'shadow_pages': ['--scene', 'valley', '--view', '8'],
    'shadow_local_off': ['--scene', 'cathedral', '--sun', '200,-30', '--keys', 'Digit1'],
    'shadow_local_on': ['--scene', 'cathedral', '--sun', '200,-30'],
    'shadow_local_pages': ['--scene', 'cathedral', '--sun', '200,-30', '--view', '15'],
    # Lumen
    'gi_off': ['--scene', 'cathedral', '--keys', 'KeyG'],
    'gi_on': ['--scene', 'cathedral'],
    'gi_indirect': ['--scene', 'cathedral', '--view', '7'],
    'gi_aisle_indirect': ['--scene', 'cathedral', '--view', '7'] + AISLE,
    'gi_cards': ['--scene', 'cathedral', '--view', '14'],
    'gi_sdf': ['--scene', 'cathedral', '--view', '12'],
    'gi_reflections': ['--scene', 'cathedral', '--view', '13'],
    'gi_surface_cache': ['--scene', 'cathedral', '--view', '11'],
    # Atmosphere
    'clouds_off': ['--scene', 'valley', '--keys', 'KeyC'],
    'clouds_on': ['--scene', 'valley'],
    'fog_off': ['--scene', 'cathedral', '--keys', 'KeyF'],
    'fog_on': ['--scene', 'cathedral'],
    'water_off': ['--scene', 'valley', '--sun', '52,50', '--keys', 'KeyU'] + LAKE_DOWN,
    'water_on': ['--scene', 'valley', '--sun', '52,50'] + LAKE_DOWN,
    'water_under_up': ['--scene', 'valley', '--sun', '52,50', '--eye', '-5,0.4,18', '--at', '-5,3,26'],
    'water_under_side': ['--scene', 'valley', '--sun', '52,50', '--eye', '-5,0.4,18', '--at', '-5,0.2,30'],
    'water_night': ['--scene', 'valley', '--sun', '200,-30'] + LAKE_DOWN,
    # Materials
    'translucency_layers_off': ['--scene', 'cathedral', '--keys', 'Slash'] + GLASS,
    'translucency_layers_on': ['--scene', 'cathedral'] + GLASS,
    'texture_pool16_raw': SPONZA_VIEW + ['--texture-pool-mb', '16', '--no-texture-compression'],
    'texture_pool16_bc': SPONZA_VIEW + ['--texture-pool-mb', '16'],
    # Camera
    'tsr_100': ['--scene', 'forest'],
    'tsr_67': ['--scene', 'forest', '--screen-percentage', '67'],
    'tsr_50': ['--scene', 'forest', '--screen-percentage', '50'],
    'dof': ['--scene', 'cathedral', '--keys', 'KeyB,KeyX,KeyX'],
}

# name: (arguments, seconds). Recorded at 60 fps, encoded at 30.
VIDEOS = {
    'tour_valley': (['--scene', 'valley', '--keys', 'KeyY'], 20),
    'tour_cathedral': (['--scene', 'cathedral', '--keys', 'KeyY'], 20),
    'tour_forest': (['--scene', 'forest', '--keys', 'KeyY'], 20),
    'wind_forest': (['--scene', 'forest', '--eye', '2,1.6,-4', '--at', '6,2.5,4'], 8),
    'lake_waves': (['--scene', 'valley', '--sun', '52,50'] + LAKE_DOWN, 8),
    'day_cycle': (['--scene', 'valley', '--keys', 'KeyJ'], 20),
}


def run(args, size, out_png, frames=120, dump=None, count=0):
    cmd = [EXE, *args, '--size', size, '--frames', str(frames), '--screenshot', out_png]
    if dump:
        cmd += ['--dump-dir', dump, '--dump-count', str(count)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0 or not os.path.exists(out_png):
        print('FAILED', ' '.join(args), r.stderr[-800:])
        return False
    return True


def webp(png, quality=88):
    Image.open(png).convert('RGB').save(png[:-4] + '.webp', quality=quality, method=6)


def stills(size, only):
    out = os.path.join(ROOT, 'stills')
    os.makedirs(out, exist_ok=True)
    for name, args in STILLS.items():
        if only and name not in only:
            continue
        png = os.path.join(out, name + '.png')
        if run(args, size, png):
            webp(png)
            print('still', name)


def videos(size, only):
    out = os.path.join(ROOT, 'videos')
    os.makedirs(out, exist_ok=True)
    for name, (args, seconds) in VIDEOS.items():
        if only and name not in only:
            continue
        frames = seconds * 60
        dump = os.path.join(ROOT, 'tmp_' + name)
        shutil.rmtree(dump, ignore_errors=True)
        poster = os.path.join(out, name + '.png')
        if not run(args, size, poster, frames=frames + 30, dump=dump, count=frames):
            continue
        mp4 = os.path.join(out, name + '.mp4')
        # Dumped frames keep their absolute frame number (the last `frames` of the run).
        first = min(int(f[6:10]) for f in os.listdir(dump) if f.startswith('frame_'))
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-framerate', '60', '-start_number', str(first),
                        '-i', os.path.join(dump, 'frame_%04d.png'),
                        '-vf', 'fps=30', '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', mp4], check=True)
        # Keep every video under the artifact host's 15 MB file limit.
        if os.path.getsize(mp4) > 14 * 2**20:
            smaller = mp4[:-4] + '_small.mp4'
            subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', mp4, '-c:v', 'libx264', '-preset', 'slow', '-crf', '25',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', smaller], check=True)
            os.replace(smaller, mp4)
        webp(poster, 80)
        shutil.rmtree(dump, ignore_errors=True)
        print('video', name, os.path.getsize(mp4) // 1024, 'KiB')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    only = set()
    size = '1920x1080'
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
    if '--size' in sys.argv:
        size = sys.argv[sys.argv.index('--size') + 1]
    if what in ('stills', 'all'):
        stills(size, only)
    if what in ('videos', 'all'):
        videos(size, only)
