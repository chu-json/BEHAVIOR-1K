"""
Visual perturbations for robustness tests of the camera-only policy. Each one can be applied to a running
environment and undone again, so a single sim process can test every condition one after another.

    PERTURBATIONS = {name: Perturbation}      # apply(env) / restore(env)

What is changed, by condition:
    light_<k>x              intensity of every USD light in the stage multiplied by k. In Rs_int the lighting that
                            matters is 8 warm DiskLights (intensity 10,000 each) that live inside the ceiling
                            objects. The dome light ("skybox", intensity 2,500) is nearly irrelevant indoors:
                            sweeping it from 0 to 25,000 changed the mean image brightness by under 1 of 148.
    wall_texture            the diffuse (colour) texture of all 7 wall objects -> a generated brick pattern
    table_texture           the diffuse texture of the table -> a generated wood-grain pattern
    red_apple               multiplicative colour tint on the apple's material, turning the green apple red

Why generated textures: the dataset's own wall/table textures are UV *atlases* (the black regions are unused
atlas space), so swapping one object's atlas onto another garbles it. A uniform pattern image has no such problem.

Rendering needs time to converge after a light change (about 80 frames); WARMUP_RENDERS says how many frames to
render after applying each perturbation before measuring anything.
"""
import os

import cv2
import numpy as np
import torch as th

TEXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "textures")
TABLE_NAME = "breakfast_table_skczfi_0"
WARMUP_RENDERS = {"light": 100, "default": 10}  # frames to render after applying a perturbation
LIGHT_TYPES = ("DiskLight", "SphereLight", "RectLight", "CylinderLight", "DistantLight", "DomeLight")
RED_TINT = th.tensor([1.0, 0.10, 0.10])  # multiplies the albedo: green apple (70, 84, 27) -> about (70, 8, 3)


def make_textures(out_dir=TEXTURE_DIR):
    """Writes brick.png and wood.png (1024x1024) and returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 1024
    # --- brick: rows of 128x48 px bricks with a half-brick offset on alternate rows, light mortar in between ---
    brick = np.full((n, n, 3), (190, 190, 185), np.uint8)  # mortar (BGR)
    for row, y in enumerate(range(0, n, 48)):
        for x in range(-64 * (row % 2), n, 128):
            tone = rng.uniform(0.75, 1.15)
            color = np.array([50, 70, 150]) * tone  # BGR: red-brown
            cv2.rectangle(brick, (x + 4, y + 4), (x + 124, y + 44), tuple(float(c) for c in color), -1)
    brick = np.clip(brick.astype(np.float32) + rng.normal(0, 6, brick.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "brick.png"), brick)
    # --- wood: long dark grain lines on a warm brown base ---
    yy = np.arange(n)[:, None]
    xx = np.arange(n)[None, :]
    grain = np.sin(yy * 0.35 + 6 * np.sin(xx * 0.004 + yy * 0.01)) * 0.5 + 0.5
    grain = grain * 0.6 + cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 3) * 0.8
    wood = np.stack([40 + 30 * grain, 80 + 55 * grain, 130 + 80 * grain], axis=-1)  # BGR
    cv2.imwrite(os.path.join(out_dir, "wood.png"), np.clip(wood, 0, 255).astype(np.uint8))
    return os.path.join(out_dir, "brick.png"), os.path.join(out_dir, "wood.png")


def _objects(env, category):
    return [o for o in env.scene.objects if o.category == category]


def _set_texture(materials, path):
    import omnigibson.lazy as lazy

    for m in materials:
        m.set_input(inp="diffuse_texture", val=lazy.pxr.Sdf.AssetPath(path))


class Perturbation:
    def __init__(self, name, apply_fn, restore_fn, description):
        self.name, self._apply, self._restore, self.description = name, apply_fn, restore_fn, description
        self._saved = None

    def apply(self, env):
        import omnigibson as og

        self._saved = self._apply(env)
        for _ in range(WARMUP_RENDERS["light" if self.name.startswith("light") else "default"]):
            og.sim.render()  # let the renderer's temporal accumulation converge to the new look

    def restore(self, env):
        self._restore(env, self._saved)
        self._saved = None


def _light_scale(scale):
    def apply(env):
        import omnigibson as og

        saved = []
        with og.sim.editing_usd():  # raw USD edits must happen inside this context
            for prim in og.sim.stage.Traverse():
                if prim.GetTypeName() in LIGHT_TYPES:
                    attr = prim.GetAttribute("inputs:intensity")
                    saved.append((attr, attr.Get()))
                    attr.Set(saved[-1][1] * scale)
        assert saved, "no lights found in the stage"
        return saved

    def restore(env, saved):
        import omnigibson as og

        with og.sim.editing_usd():
            for attr, value in saved:
                attr.Set(value)

    return apply, restore


def _texture_swap(select_materials, texture_path):
    def apply(env):
        mats = select_materials(env)
        saved = [(m, m.get_input(inp="diffuse_texture").resolvedPath) for m in mats]
        _set_texture(mats, texture_path)
        return saved

    def restore(env, saved):
        for m, path in saved:
            _set_texture([m], path)

    return apply, restore


def _wall_materials(env):
    return [m for o in _objects(env, "walls") for m in o.materials]


def _table_materials(env):
    return list(env.scene.object_registry("name", TABLE_NAME).materials)


def _apple_materials(env):
    return list(env.scene.object_registry("name", "apple").materials)


def _tint_apple(env):
    mats = _apple_materials(env)
    saved = [(m, m.diffuse_tint.clone()) for m in mats]
    for m in mats:
        m.diffuse_tint = RED_TINT
    return saved


def _untint_apple(env, saved):
    for m, tint in saved:
        m.diffuse_tint = tint


def build_perturbations():
    """Built lazily: the texture paths and OmniGibson imports need the simulator to be running."""
    brick, wood = make_textures()
    p = {}
    for scale in (0.5, 2.0, 0.25, 4.0):
        a, r = _light_scale(scale)
        p[f"light_{scale:g}x"] = Perturbation(f"light_{scale:g}x", a, r, f"all light intensities x{scale:g}")
    a, r = _texture_swap(_wall_materials, brick)
    p["wall_texture"] = Perturbation("wall_texture", a, r, "walls -> generated brick pattern")
    a, r = _texture_swap(_table_materials, wood)
    p["table_texture"] = Perturbation("table_texture", a, r, "table -> generated wood grain")
    p["red_apple"] = Perturbation("red_apple", _tint_apple, _untint_apple, f"apple tint {RED_TINT.tolist()}")
    return p
