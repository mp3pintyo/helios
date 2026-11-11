"""Captures the showcase website's media from the native Helios build.

usage: python site/capture.py [stills|videos|models|docs|all] [--only NAME[,NAME...]] [--size WxH] [--out DIR]

`docs` cuts the round-8 before/after pairs whose "before" side only a pre-fix build could render
(and the editor's browser screenshots) out of the comparison images in docs/round8/.

Stills land in media/stills/<name>.png (+ .webp), videos in media/videos/<name>.mp4 (+ poster .webp),
glTF models the page loads into the engine in media/models/. Every still is a fixed camera and
frame count, so reruns are reproducible. The editor's screenshots and video are recorded in the
browser by site/capture_editor.py.
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
ASSETS = os.environ.get('HELIOS_ASSETS', os.path.join(REPO, 'assets'))
FFMPEG = shutil.which('ffmpeg') or r'c:\Apps\ffmpeg-shared\bin\ffmpeg.exe'
SPONZA = os.path.join(ASSETS, 'Sponza', 'Sponza.gltf')
HELMET = os.path.join(ASSETS, 'DamagedHelmet', 'DamagedHelmet.glb')
# Fox: Khronos glTF Sample Assets (model PixelMannen, CC0; rigging and animation tomkranis, CC BY 4.0).
FOX = os.path.join(ASSETS, 'Fox', 'Fox.glb')
# Models the page itself fetches (copied to media/models).
MODELS = [FOX]

LAKE_DOWN = ['--eye', '-5,6,10', '--at', '-5,-1,24']
AISLE = ['--eye', '31,1.5,10.5', '--at', '38,3,13.5']
GLASS = ['--eye', '14,1.7,10.25', '--at', '-0.6,3,10.25']
# South aisle looking west: the windows' coloured patches on the piers and the floor (round 7).
GLASS_AISLE = ['--eye', '45,1.7,10.3', '--at', '5,4,10.8']
SPONZA_VIEW = ['--gltf', SPONZA, '--eye', '-9,1.6,0', '--at', '9,3.5,0']
# Round 4 views (the page's presets use the same cameras).
NIGHT = ['--sun', '200,-30']
ML_AISLE = ['--scene', 'cathedral', '--lights', '1000'] + NIGHT + ['--eye', '15,1.7,9', '--at', '35,0.8,12.5']
CAMPFIRE = ['--scene', 'forest'] + NIGHT + ['--eye', '3.5,1.4,1.0', '--at', '7,1.0,-1']
CANOPY = ['--scene', 'forest', '--eye', '0,30,0', '--at', '200,0,-60', '--fov', '18']
PAINT = ['--scene', 'gallery', '--eye', '-7.2,1.9,3.4', '--at', '-7.2,1.5,0']
STONE_PATH = ['--scene', 'valley', '--sun', '200,28', '--eye', '-27,1.2,41', '--at', '-31,0.1,48']
STONE_GRAZE = ['--scene', 'valley', '--sun', '200,28', '--eye', '-27.5,0.35,41.5', '--at', '-31,0.12,49']
BRICKS = ['--scene', 'gallery', '--eye', '0.9,0.45,10.9', '--at', '0,0,9.2']
MUSHROOMS = ['--scene', 'gallery', '--eye', '-5.9,0.3,10.4', '--at', '-6.5,0.08,9.2']
# Round 8 views (the page's presets use the same cameras).
WC_AISLE = ['--scene', 'cathedral', '--eye', '22,1.7,11.5', '--at', '45,4,11']
PUDDLES = ['--scene', 'gallery', '--gi', 'lite', '--eye', '0,0.6,6', '--at', '0,0.5,0']
ML_NIGHT = ['--scene', 'cathedral', '--lights', '300', '--megalights'] + NIGHT
GLASS_CANDLES = ['--scene', 'cathedral', '--lights', '1008'] + NIGHT + ['--eye', '0.8,1.6,7.0', '--at', '2.0,0.4,12.0', '--fov', '34']
LAKE_PANELS = ['--scene', 'valley', '--eye', '-2.8,3.2,27', '--at', '-1.5,2.6,30.6']
SMOKE = ['--scene', 'forest', '--campfire-smoke', '8', '--sun', '143,40', '--eye', '11,4,2.5', '--at', '7.5,0,-2']
FOG_NAVE = ['--scene', 'cathedral', '--lantern', '--lights', '300'] + NIGHT
PHOTO_NAVE = ['--scene', 'cathedral'] + NIGHT + ['--fov', '40', '--dof', '1.2,6']
GRASS = ['--scene', 'forest', '--eye', '4.6,0.4,-1.6', '--at', '5.7,0.12,-2.7']
FERNS = ['--scene', 'forest', '--eye', '5.2,0.7,1.2', '--at', '6.2,0.0,0.2']
CLIFF = ['--scene', 'valley', '--eye', '7,1.8,-25', '--at', '-3.9,-3,-38.9']
ARCH = ['--scene', 'valley', '--eye', '20.7,1.8,-9.8', '--at', '34,-5.5,-22']
CAVE = ['--scene', 'valley', '--eye', '-14,2.2,-28', '--at', '-12.5,1.8,-40']
RIVER = ['--scene', 'valley', '--eye', '-2,3,-6', '--at', '-9,0,-20']
# Beauty stills in photo mode: 64 frames accumulated (sub-pixel samples; the scene clock stands).
PHOTO = ['--photo', '64']

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
    # Round 7: sunlight through the stained glass, white / coloured by the panes
    'glass_shadow_off': ['--scene', 'cathedral', '--no-glass-shadows'] + GLASS_AISLE,
    'glass_shadow_on': ['--scene', 'cathedral'] + GLASS_AISLE,
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
    # Round 4 · dynamic world
    'dyn_invalidation': ['--scene', 'cathedral', '--view', '16'],
    'dyn_raft': ['--scene', 'valley'] + LAKE_DOWN,
    # Skeletal animation
    'fox_lit': ['--gltf', FOX],
    'fox_weights': ['--gltf', FOX, '--view', '17'],
    # MegaLights (the candle-lit cathedral)
    'megalights_off': ML_AISLE,
    'megalights_on': ML_AISLE + ['--megalights'],
    'megalights_samples': ML_AISLE + ['--megalights', '--view', '18'],
    # Lumen Lite
    'gi_lumen_indirect': ['--scene', 'cathedral', '--gi', 'lumen', '--view', '7'],
    'gi_lite_indirect': ['--scene', 'cathedral', '--gi', 'lite', '--view', '7'],
    'lite_probes': ['--scene', 'cathedral', '--gi', 'lite', '--view', '20'],
    # Volumes and particles (the smoke needs a few seconds to rise)
    'campfire_off': (CAMPFIRE + ['--keys', 'Quote'], 300),
    'campfire_on': (CAMPFIRE, 300),
    'incense': (['--scene', 'cathedral', '--eye', '66.2,1.8,0.4', '--at', '68.4,2.4,1.9'], 400),
    'dust_shafts': (['--scene', 'cathedral', '--eye', '20,1.7,-4', '--at', '30,7,6'], 200),
    'volume_steps': (CAMPFIRE + ['--view', '21'], 300),
    # Nanite Foliage
    'canopy_voxels_off': CANOPY + ['--keys', 'Insert'],
    'canopy_voxels_on': CANOPY,
    'foliage_voxels': CANOPY + ['--view', '22'],
    'foliage_assemblies': ['--scene', 'forest', '--view', '23'],
    # Substrate and material graphs
    'substrate_off': PAINT + ['--keys', 'PageUp'],
    'substrate_on': PAINT,
    'gallery': ['--scene', 'gallery'],
    'gallery_wax': ['--scene', 'gallery'] + NIGHT + ['--eye', '-1.9,1.5,1.8', '--at', '-1.45,1.35,0'],
    'gallery_features': ['--scene', 'gallery', '--view', '24'],
    # Nanite tessellation (the valley's cobblestone path and displaced boulders)
    'tess_path_off': STONE_PATH + ['--no-tessellation'],
    'tess_path_on': STONE_PATH,
    'tess_graze_off': STONE_GRAZE + ['--no-tessellation'],
    'tess_graze_on': STONE_GRAZE,
    'tess_depth': STONE_PATH + ['--view', '25'],
    # Rounds 5 and 6: graph displacement in the gallery, Lumen following it (the variants bake on
    # worker threads after loading, hence the longer runs), the carried lantern. `lumen_disp_off`
    # is the same view as `lumen_disp_on` rendered by a round-5 build (main 6eb1c4d).
    'tess_bricks_off': BRICKS + ['--no-tessellation'],
    'tess_bricks_on': BRICKS,
    'tess_mushrooms_off': MUSHROOMS + ['--no-tessellation'],
    'tess_mushrooms_on': MUSHROOMS,
    'tess_cloth': ['--scene', 'gallery', '--eye', '4.4,0.6,10.9', '--at', '6.3,0.15,9.0'],
    'tess_tufts': (['--scene', 'gallery', '--eye', '1.9,1.1,15.6', '--at', '0,0.1,13.4'], 300),
    'lumen_disp_on': (MUSHROOMS + ['--view', '14'], 400),
    'lantern_night': (['--scene', 'cathedral', '--lantern', '--sun', '200,-25', '--eye', '11.4,1.2,-0.6', '--at', '12.9,0.5,0.5'], 300),
    # Round 8. Lumen world cache: the south aisle's indirect light, cards + voxels / world cache,
    # at a fixed exposure (auto exposure would even out the difference); the nave's cells (27).
    'wc_off': (WC_AISLE + ['--world-cache', 'off', '--view', '7', '--exposure', '0'], 300),
    'wc_on': (WC_AISLE + ['--view', '7', '--exposure', '0'], 300),
    'wc_cells': (['--scene', 'cathedral', '--view', '27'], 300),
    # Lumen Lite 5.8: the gallery's puddles, traced reflections / screen traces + irradiance field.
    'lite_refl_off': (PUDDLES + ['--lite-traced-reflections'], 300),
    'lite_refl_on': (PUDDLES, 300),
    # MegaLights' froxel volume: the fog and glass take shadowed candlelight (300 candles).
    'ml_volume_off': (ML_NIGHT + ['--ml-no-volume'], 300),
    'ml_volume_on': (ML_NIGHT, 300),
    # Candlelight through the north aisle's stained-glass screens: white / coloured by the panes.
    'glass_candles_off': (GLASS_CANDLES + ['--no-glass-shadows'], 300),
    'glass_candles_on': (GLASS_CANDLES, 300),
    # AVBOIT: the lake's crossing panels, weighted blended + depth peeling / adaptive voxel OIT.
    'avboit_off': (LAKE_PANELS + ['--wboit'], 200),
    'avboit_on': (LAKE_PANELS, 200),
    # Shadows: the overview (26) in the windy forest; the smoky campfire's volume shadow.
    'vsm_overview': ['--scene', 'forest', '--view', '26'],
    'smoke_shadow_off': (SMOKE + ['--no-volume-shadows'], 420),
    'smoke_shadow_on': (SMOKE, 420),
    # Fog screen-space scattering in the candle-lit nave with the lantern; photo mode's bokeh.
    'fog_glow_off': (FOG_NAVE, 300),
    'fog_glow_on': (FOG_NAVE + ['--fog-scattering', '4'], 300),
    'photo_realtime': (PHOTO_NAVE, 300),
    'photo_accum': (PHOTO_NAVE + ['--photo', '256'], 300),
    # Grounded foliage: short-range AO off / on at the grass tuft; ferns by the fire (photo).
    'ao_off': (GRASS + ['--no-short-range-ao'], 200),
    'ao_on': (GRASS, 200),
    'ferns_photo': (FERNS + PHOTO, 200),
    # Mesh terrain, the river and the cliff rock (photo mode).
    'cliff': (CLIFF + PHOTO, 200),
    'arch': (ARCH + PHOTO, 200),
    'cave_river': (CAVE + PHOTO, 200),
    'river_close': (RIVER + PHOTO, 200),
}

# Round 8 before/after pairs of fixes reported by the user: the "before" side was rendered by the
# build before the fix, so both sides are cut from the comparison images in docs/round8/ (the
# label strip at the top left is cropped away). name: (image, before box, after box) -> name_off/on.
DOC_PAIRS = {
    'grounded': ('foliage-grounding/grass-before-after.jpg', (32, 36, 768, 450), (840, 36, 1576, 450)),
    'campfire_plants': ('foliage-grounding/campfire-night-before-after.jpg', (32, 36, 768, 450), (840, 36, 1576, 450)),
    'rock': ('rockmat/default-before-after.jpg', (32, 36, 768, 450), (832, 36, 1568, 450)),
    'cave_fix': ('riverfix/cave-before-after.jpg', (32, 36, 1168, 675), (32, 711, 1168, 1350)),
}
# Browser screenshots of the round-8 editor (the Playwright runs of docs/round8/editor-hierarchy.md).
DOC_STILLS = {
    'editor_actor_drag': 'editor-hierarchy/browser-drag-campfire.jpg',
    'editor_outliner': 'editor-hierarchy/browser-outliner-trees.jpg',
    'editor_component': 'editor-hierarchy/browser-component.jpg',
}

# name: (arguments, seconds). Recorded at 60 fps, encoded at 30.
VIDEOS = {
    'tour_valley': (['--scene', 'valley', '--keys', 'KeyY'], 20),
    'tour_cathedral': (['--scene', 'cathedral', '--keys', 'KeyY'], 20),
    'tour_forest': (['--scene', 'forest', '--keys', 'KeyY'], 20),
    'wind_forest': (['--scene', 'forest', '--eye', '2,1.6,-4', '--at', '6,2.5,4'], 8),
    'lake_waves': (['--scene', 'valley', '--sun', '52,50'] + LAKE_DOWN, 8),
    'day_cycle': (['--scene', 'valley', '--keys', 'KeyJ'], 20),
    # Round 4
    'tour_gallery': (['--scene', 'gallery', '--keys', 'KeyY'], 20),
    'chandeliers_candles': (['--scene', 'cathedral', '--lights', '300', '--megalights'] + NIGHT + ['--eye', '8,1.7,0', '--at', '40,7,0'], 12),
    'campfire_night': (CAMPFIRE, 12),
    # Survey, then a crossfade to the walk 4.5 s in.
    'fox_clips': (['--gltf', FOX, '--keys', 'Minus', '--keys-at', '300'], 10),
    # Round 8: a Numpad1 gust ring sweeping the clearing's grass and ferns (2 s in); the river.
    'wind_gust': (['--scene', 'forest', '--eye', '3.0,2.6,2.5', '--at', '7.0,0.0,-2.0', '--keys', 'Numpad1', '--keys-at', '150'], 8),
    'river_flow': (RIVER, 8),
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
        args, frames = args if isinstance(args, tuple) else (args, 120)
        png = os.path.join(out, name + '.png')
        if run(args, size, png, frames=frames):
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
        limit_size(mp4)
        webp(poster, 80)
        shutil.rmtree(dump, ignore_errors=True)
        print('video', name, os.path.getsize(mp4) // 1024, 'KiB')


def limit_size(mp4, limit=14 * 2**20):
    """Re-encodes a video with rising CRF until it fits the artifact host's 15 MB file limit
    (foliage in the wind compresses poorly)."""
    for crf in (25, 28, 31):
        if os.path.getsize(mp4) <= limit:
            return
        smaller = mp4[:-4] + f'_crf{crf}.mp4'
        subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', mp4, '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', smaller], check=True)
        if os.path.getsize(smaller) < os.path.getsize(mp4):
            os.replace(smaller, mp4)
        else:
            os.remove(smaller)


def docs(only):
    """Cuts the round-8 before/after pairs and editor screenshots out of docs/round8/."""
    out = os.path.join(ROOT, 'stills')
    os.makedirs(out, exist_ok=True)
    src = os.path.join(REPO, 'docs', 'round8')
    for name, (image, before, after) in DOC_PAIRS.items():
        if only and name not in only:
            continue
        im = Image.open(os.path.join(src, image)).convert('RGB')
        for suffix, box in (('_off', before), ('_on', after)):
            im.crop(box).save(os.path.join(out, name + suffix + '.webp'), quality=90, method=6)
        print('pair', name)
    for name, image in DOC_STILLS.items():
        if only and name not in only:
            continue
        Image.open(os.path.join(src, image)).convert('RGB').save(os.path.join(out, name + '.webp'), quality=85, method=6)
        print('still', name)


def models():
    out = os.path.join(ROOT, 'models')
    os.makedirs(out, exist_ok=True)
    for path in MODELS:
        shutil.copy2(path, out)
        print('model', os.path.basename(path))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    only = set()
    size = '1920x1080'
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
    if '--size' in sys.argv:
        size = sys.argv[sys.argv.index('--size') + 1]
    if what in ('models', 'all'):
        models()
    if what in ('stills', 'all'):
        stills(size, only)
    if what in ('videos', 'all'):
        videos(size, only)
    if what in ('docs', 'all'):
        docs(only)
