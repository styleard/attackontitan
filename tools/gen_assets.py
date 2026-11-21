#!/usr/bin/env python3
"""Generates geo models, animations and textures for the giants (original designs).
Run:  python3 tools/gen_assets.py   (needs: pillow, numpy)"""
import json, os, random
import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "src/main/resources")
ASSETS = os.path.join(RES, "assets/titanmod")
MIND_SCALE = 0.4  # mindless giant is exported 0.4x (6 blocks) directly in the geo
H = 9  # upper body is lifted by H so the whole model is ~15 blocks tall

# ---------------------------------------------------------------- noise helpers
def noise(rng, w, h, scale):
    sw, sh = max(2, w // scale + 2), max(2, h // scale + 2)
    a = rng.random((sh, sw)).astype(np.float32)
    img = Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0

def fbm(rng, w, h):
    n = 0.5 * noise(rng, w, h, 16) + 0.3 * noise(rng, w, h, 8) + 0.2 * noise(rng, w, h, 3)
    return n

# ---------------------------------------------------------------- face painters
class Painter:
    def __init__(self, variant, seed):
        self.v = variant
        self.rng = np.random.default_rng(seed)
        self.skin = np.array([206, 148, 118] if variant == "warrior" else [196, 160, 132], dtype=np.float32)
        self.hair = np.array([150, 108, 48], dtype=np.float32)

    def skin_face(self, w, h, face, muscle=False, outline=False, dark=1.0):
        r = self.rng
        n = fbm(r, w, h)
        base = np.ones((h, w, 3), dtype=np.float32) * self.skin * dark
        base *= (0.86 + 0.28 * n)[..., None]
        # edge ambient occlusion
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ex = np.minimum(xx, w - 1 - xx) / max(1, w / 2)
        ey = np.minimum(yy, h - 1 - yy) / max(1, h / 2)
        ao = 0.78 + 0.22 * np.clip(np.minimum(ex, ey) * 3, 0, 1)
        base *= ao[..., None]
        if muscle:
            warp = noise(r, w, h, 6) * 5
            lines = np.sin((xx * 0.9 + warp) * 1.1)
            base *= (1 - 0.10 * (lines > 0.9))[..., None]
            shade = 0.9 + 0.2 * np.sin(yy / max(1, h) * np.pi)
            base *= shade[..., None]
            # subtle red flesh tone
            base[..., 0] *= 1.04
            base[..., 2] *= 0.95
        if outline and w > 3 and h > 3:
            base[0, :] *= 0.62; base[-1, :] *= 0.62
            base[:, 0] *= 0.62; base[:, -1] *= 0.62
        # faint veins
        if muscle and w > 8:
            vein = (noise(r, w, h, 4) > 0.82) & (noise(r, w, h, 9) > 0.55)
            base[vein] *= np.array([0.85, 0.8, 0.95])
        return self._rgba(base)

    def hair_face(self, w, h, face):
        r = self.rng
        col = r.random(w).astype(np.float32)
        streak = np.tile(col, (h, 1))
        n = noise(r, w, h, 5)
        base = np.ones((h, w, 3), dtype=np.float32) * self.hair
        base *= (0.65 + 0.45 * streak * 0.6 + 0.4 * n * 0.5)[..., None]
        base[..., 0] *= 1.05
        yy = np.mgrid[0:h, 0:w][0].astype(np.float32)
        base *= (1.0 - 0.18 * yy / max(1, h))[..., None]
        return self._rgba(base)

    def flat(self, w, h, rgb, noise_amt=0.08):
        n = noise(self.rng, w, h, 3)
        base = np.ones((h, w, 3), dtype=np.float32) * np.array(rgb, dtype=np.float32)
        base *= (1 - noise_amt + noise_amt * 2 * n)[..., None]
        return self._rgba(base)

    def teeth(self, w, h):
        img = self.flat(w, h, [236, 230, 210], 0.06)
        a = np.asarray(img).copy()
        for x in range(0, w, 3):
            a[:, x, :3] = (a[:, x, :3] * 0.7).astype(np.uint8)
        return Image.fromarray(a)

    def eye(self, w, h, glow):
        if glow:
            base = np.zeros((h, w, 3), dtype=np.float32)
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            d = np.sqrt(((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2)
            core = np.clip(1.2 - d, 0, 1)
            base[..., 0] = 40 + 120 * core
            base[..., 1] = 190 + 65 * core
            base[..., 2] = 70 + 110 * core
            # thin dark pupil slit
            cx = w // 2
            base[:, cx:cx + max(1, w // 8)] *= 0.45
            return self._rgba(base)
        base = np.ones((h, w, 3), dtype=np.float32) * np.array([225, 222, 205], dtype=np.float32)
        px = int(w * 0.62)
        base[max(0, h // 2 - 1):h // 2 + 2, px:px + max(1, w // 4)] = [15, 12, 12]
        base[0, :] *= 0.6; base[-1, :] *= 0.6
        return self._rgba(base)

    @staticmethod
    def _rgba(rgb):
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        a = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
        return Image.fromarray(np.concatenate([rgb, a], axis=2), "RGBA")

# ---------------------------------------------------------------- model builder
FACES = ["north", "south", "east", "west", "up", "down"]

def face_dims(size, face):
    w, h, d = size
    if face in ("north", "south"): return w, h
    if face in ("east", "west"): return d, h
    return w, d

class Model:
    def __init__(self, name):
        self.name = name
        self.bones = {}   # name -> dict(pivot, parent, cubes)
        self.order = []

    def bone(self, name, pivot, parent=None):
        self.bones[name] = dict(name=name, pivot=pivot, parent=parent, cubes=[])
        self.order.append(name)

    def cube(self, bone, lo, hi, style="skin", **kw):
        size = [hi[i] - lo[i] for i in range(3)]
        self.bones[bone]["cubes"].append(dict(origin=list(lo), size=size, style=style, kw=kw))

    def mirror_cube(self, bone, lo, hi, style="skin", **kw):
        self.cube(bone, [-hi[0], lo[1], lo[2]], [-lo[0], hi[1], hi[2]], style, **kw)

def build(variant):
    W = variant == "warrior"
    m = Model(variant if False else ("warrior_giant" if W else "mindless_giant"))
    y = lambda v: v + H
    m.bone("body", [0, y(105), 0])
    m.bone("chest", [0, y(125), 0], "body")
    m.bone("head", [0, y(186), 0], "chest")
    m.bone("jaw", [0, y(188), 10], "head")
    hw = 14 if W else 17           # head half width
    hz = 15 if W else 17
    m.bone("eye_l", [hw * 0.55, y(201.5), -hz - 0.5], "head")
    m.bone("eye_r", [-hw * 0.55, y(201.5), -hz - 0.5], "head")
    if W:
        m.bone("hair_back", [0, y(213), 15], "head")
    for s, sx in (("l", 1), ("r", -1)):
        m.bone("arm_" + s, [sx * 49, y(169), 0], "chest")
        m.bone("forearm_" + s, [sx * 50, y(121), 0], "arm_" + s)
    for s, sx in (("l", 1), ("r", -1)):
        m.bone("leg_" + s, [sx * 18, y(105), 0])
        m.bone("shin_" + s, [sx * 18, 66, 0], "leg_" + s)

    # pelvis / torso
    m.cube("body", [-33, y(98), -17], [33, y(126), 17], "skin", muscle=True)
    m.cube("chest", [-30, y(125), -17], [30, y(156), 17], "skin", muscle=True)
    m.cube("chest", [-40, y(155), -19], [40, y(187), 19], "skin", muscle=True)
    m.cube("chest", [-36, y(135), 19], [36, y(185), 25], "skin", muscle=True, outline=True)   # back
    if W:
        m.cube("chest", [1, y(157), -24], [38, y(180), -19], "skin", muscle=True, outline=True)
        m.cube("chest", [-38, y(157), -24], [-1, y(180), -19], "skin", muscle=True, outline=True)
        for i in range(3):
            lo = y(127 + i * 10)
            m.cube("chest", [1, lo, -21], [17, lo + 9, -17], "skin", muscle=True, outline=True)
            m.cube("chest", [-17, lo, -21], [-1, lo + 9, -17], "skin", muscle=True, outline=True)
    else:
        m.cube("chest", [-36, y(124), -32], [36, y(160), -17], "skin", muscle=False, outline=False, dark=0.97)  # belly
        m.cube("chest", [-30, y(158), -24], [30, y(178), -19], "skin", muscle=False)
    # neck + traps
    m.cube("chest", [-11, y(186), -11], [11, y(199), 11], "skin", muscle=True)
    m.cube("chest", [7, y(183), -13], [31, y(194), 13], "skin", muscle=True)
    m.cube("chest", [-31, y(183), -13], [-7, y(194), 13], "skin", muscle=True)

    # head
    top = y(222) if W else y(226)
    bot = y(195)
    m.cube("head", [-hw, bot, -hz], [hw, top, hz], "skin", head=True)
    m.cube("head", [-2.5, y(201), -hz - 3], [2.5, y(209), -hz], "skin", dark=0.93)       # nose
    m.cube("head", [-hw + 1, y(213), -hz - 2], [hw - 1, y(217), -hz], "brow" if W else "skin", dark=0.85)
    m.cube("head", [-hw + 3.5, y(196), -hz - 1], [hw - 3.5, y(200), -hz], "teeth")       # upper teeth
    m.cube("jaw", [-hw + 2, y(189), -hz], [hw - 2, y(196), hz - 3], "skin", jaw=True)
    m.cube("jaw", [-hw + 3.5, y(193), -hz - 1], [hw - 3.5, y(197), -hz], "teeth")       # lower teeth
    # eyes (small cubes around bone pivot)
    for b, sx in (("eye_l", 1), ("eye_r", -1)):
        cx = sx * hw * 0.55
        m.cube(b, [cx - 3.5, y(199), -hz - 1.5], [cx + 3.5, y(204), -hz], "eye", glow=W)
    if W:
        m.cube("head", [-15, y(221), -16], [15, y(227), 16], "hair")
        m.cube("head", [-15, y(216), -17], [15, y(222), -14], "hair")
        m.cube("head", [-18, y(180), -5], [-14, y(222), 17], "hair")
        m.cube("head", [14, y(180), -5], [18, y(222), 17], "hair")
        m.cube("hair_back", [-17, y(150), 15], [17, y(222), 22], "hair")
        m.cube("hair_back", [-12, y(128), 19], [12, y(152), 24], "hair")

    # arms
    for s, sx in (("l", 1), ("r", -1)):
        c = (m.cube if sx > 0 else m.mirror_cube)
        mc = lambda b, lo, hi, st="skin", **k: (m.cube(b, lo, hi, st, **k) if sx > 0 else m.mirror_cube(b, lo, hi, st, **k))
        mc("arm_" + s, [36, y(166), -14], [62, y(190), 14], muscle=True)           # delt
        mc("arm_" + s, [40, y(122), -12], [60, y(168), 12], muscle=True)           # upper arm
        mc("arm_" + s, [42, y(142), -17], [58, y(164), -12], muscle=True, outline=True)  # bicep
        mc("arm_" + s, [42, y(136), 12], [58, y(162), 15], muscle=True)            # tricep
        mc("forearm_" + s, [41, y(84), -10], [59, y(122), 10], muscle=True)
        mc("forearm_" + s, [43, y(92), -13], [57, y(116), -10], muscle=True, outline=True)
        mc("forearm_" + s, [40, y(70), -11], [60, y(86), 11])                        # hand
        mc("forearm_" + s, [41, y(62), -10], [59, y(72), 4])                         # fingers
    # legs
    for s, sx in (("l", 1), ("r", -1)):
        mc = lambda b, lo, hi, st="skin", **k: (m.cube(b, lo, hi, st, **k) if sx > 0 else m.mirror_cube(b, lo, hi, st, **k))
        mc("leg_" + s, [4, 66, -15], [32, y(105), 15], muscle=True)
        mc("leg_" + s, [6, 72, -19], [30, y(97), -15], muscle=True, outline=True)
        mc("leg_" + s, [5, 90, 15], [31, y(105), 19], muscle=True)
        mc("shin_" + s, [7, 10, -12], [29, 66, 13], muscle=True)
        mc("shin_" + s, [8, 28, 13], [28, 62, 18], muscle=True, outline=True)
        mc("shin_" + s, [6, 52, -16], [30, 68, -12], muscle=True)
        mc("shin_" + s, [4, 0, -34], [32, 10, 12])
        mc("shin_" + s, [5, 0, -42], [31, 6, -34])
    return m

# ---------------------------------------------------------------- atlas + export
def make_assets(variant):
    m = build(variant)
    P = Painter(variant, 7 if variant == "warrior" else 11)
    rects = []
    for bname in m.order:
        for ci, c in enumerate(m.bones[bname]["cubes"]):
            for f in FACES:
                w, h = face_dims(c["size"], f)
                rects.append([bname, ci, f, max(1, round(w)), max(1, round(h))])
    # shelf pack
    S = 512
    while True:
        order = sorted(rects, key=lambda r: -r[4])
        x = y = rowh = 0; ok = True; pos = {}
        for r in order:
            w, h = r[3] + 1, r[4] + 1
            if x + w > S: x = 0; y += rowh; rowh = 0
            if y + h > S: ok = False; break
            pos[(r[0], r[1], r[2])] = (x, y, r[3], r[4]); x += w; rowh = max(rowh, h)
        if ok: break
        S *= 2
    tex = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    bones_json = []
    sc = MIND_SCALE if variant == "mindless" else 1.0
    for bname in m.order:
        b = m.bones[bname]
        cubes_json = []
        for ci, c in enumerate(b["cubes"]):
            kw, st = c["kw"], c["style"]
            uv = {}
            for f in FACES:
                px, py, w, h = pos[(bname, ci, f)]
                uv[f] = {"uv": [px, py], "uv_size": [w, h]}
                dark = kw.get("dark", 1.0)
                if st == "skin":
                    img = P.skin_face(w, h, f, kw.get("muscle", False), kw.get("outline", False), dark)
                    if kw.get("head") and f == "north":
                        img = paint_face(P, img, variant)
                    if kw.get("head") and f == "down":
                        img = P.flat(w, h, [70, 12, 16])
                    if kw.get("jaw") and f == "up":
                        img = P.flat(w, h, [150, 50, 60])
                elif st == "hair": img = P.hair_face(w, h, f)
                elif st == "brow": img = P.flat(w, h, [60, 38, 22], 0.2)
                elif st == "teeth": img = P.teeth(w, h)
                elif st == "eye":
                    img = P.eye(w, h, kw.get("glow", False))
                    if kw.get("glow"): glow.paste(img, (px, py))
                tex.paste(img, (px, py))
            cubes_json.append({"origin": [round(v * sc, 3) for v in c["origin"]],
                               "size": [round(v * sc, 3) for v in c["size"]], "uv": uv})
        bj = {"name": bname, "pivot": [round(v * sc, 3) for v in b["pivot"]]}
        if b["parent"]: bj["parent"] = b["parent"]
        if cubes_json: bj["cubes"] = cubes_json
        bones_json.append(bj)
    geo = {"format_version": "1.12.0", "minecraft:geometry": [{
        "description": {"identifier": "geometry." + m.name, "texture_width": S, "texture_height": S,
                        "visible_bounds_width": 8, "visible_bounds_height": 18, "visible_bounds_offset": [0, 8, 0]},
        "bones": bones_json}]}
    os.makedirs(os.path.join(ASSETS, "geo"), exist_ok=True)
    os.makedirs(os.path.join(ASSETS, "textures/entity"), exist_ok=True)
    json.dump(geo, open(os.path.join(ASSETS, "geo", m.name + ".geo.json"), "w"), indent=1)
    tex.save(os.path.join(ASSETS, "textures/entity", m.name + ".png"))
    if variant == "warrior":
        glow.save(os.path.join(ASSETS, "textures/entity", m.name + "_glowmask.png"))
    return m, S

def paint_face(P, img, variant):
    """Face details on the head's front: cheeks, nostrils, mouth line."""
    a = np.asarray(img).copy().astype(np.float32)
    h, w = a.shape[:2]
    # eye-socket shading
    for cx in (w * 0.22, w * 0.78):
        x0, x1 = int(cx - w * 0.2), int(cx + w * 0.2)
        y0, y1 = int(h * 0.2), int(h * 0.46)
        a[y0:y1, max(0, x0):x1, :3] *= 0.8
    # nose shadow
    a[int(h * 0.50):int(h * 0.60), int(w * 0.38):int(w * 0.62), :3] *= 0.85
    # mouth line (dark red) near the jaw hinge
    my = h - 3 if variant == "warrior" else h - 6
    a[my:my + 2, 1:w - 1, :3] = [80, 16, 20]
    if variant != "warrior":
        a[my - 2:my, 2:w - 2, :3] = [95, 25, 28]
    # cheek lines
    a[int(h * 0.6):int(h * 0.85), 2, :3] *= 0.8
    a[int(h * 0.6):int(h * 0.85), w - 3, :3] *= 0.8
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")

# ---------------------------------------------------------------- animations
def kf(d): return {str(k): v for k, v in d.items()}

def anims(m, W):
    has = lambda n: n in m.bones
    A = {}
    def add(anim, bone, chan, frames):
        if not has(bone): return
        A.setdefault(anim, {"bones": {}})["bones"].setdefault(bone, {})[chan] = kf(frames)
    def meta(name, length, loop):
        A.setdefault(name, {"bones": {}})
        A[name]["animation_length"] = length
        A[name]["loop"] = loop

    # ---------- idle
    meta("idle", 4.0, True)
    lean = 0 if W else 9
    add("idle", "chest", "scale", {0: [1, 1, 1], 2: [1.012, 1.02, 1.012], 4: [1, 1, 1]})
    add("idle", "chest", "rotation", {0: [lean, 0, 0], 2: [lean - 1.5, 0, 0], 4: [lean, 0, 0]})
    hz = 0 if W else 7
    add("idle", "head", "rotation", {0: [0, 0, hz], 1: [0, 4, hz], 2: [0, 0, hz], 3: [0, -4, hz], 4: [0, 0, hz]})
    ax = 0 if W else -14
    add("idle", "arm_l", "rotation", {0: [ax, 0, 3], 2: [ax + 2, 0, 5], 4: [ax, 0, 3]})
    add("idle", "arm_r", "rotation", {0: [ax, 0, -3], 2: [ax + 2, 0, -5], 4: [ax, 0, -3]})
    add("idle", "hair_back", "rotation", {0: [0, 0, 0], 1: [2.5, 0, 0], 2: [0, 0, 0], 3: [-2.5, 0, 0], 4: [0, 0, 0]})
    j = 4 if W else 26
    add("idle", "jaw", "rotation", {0: [j, 0, 0], 2: [j + 3, 0, 0], 4: [j, 0, 0]} if W else
        {0: [j, 0, 0], 1: [j + 8, 0, 0], 2: [j, 0, 0], 3: [j + 6, 0, 0], 4: [j, 0, 0]})
    amp = 1.2 if W else 2.2
    for e in ("eye_l", "eye_r"):
        add("idle", e, "position", {0: [0, 0, 0], 0.8: [amp, 0, 0], 1.6: [-amp, 0, 0], 2.4: [0, amp * 0.6, 0],
                                     3.2: [amp, -amp * 0.4, 0], 4: [0, 0, 0]})
        add("idle", e, "scale", {0: [1, 1, 1], 3.0: [1, 1, 1], 3.08: [1, 0.1, 1], 3.2: [1, 1, 1], 4: [1, 1, 1]})

    # ---------- walk
    meta("walk", 1.2, True)
    sw = 28 if W else 24
    add("walk", "leg_l", "rotation", {0: [-sw, 0, 0], 0.6: [sw, 0, 0], 1.2: [-sw, 0, 0]})
    add("walk", "leg_r", "rotation", {0: [sw, 0, 0], 0.6: [-sw, 0, 0], 1.2: [sw, 0, 0]})
    add("walk", "shin_l", "rotation", {0: [5, 0, 0], 0.3: [5, 0, 0], 0.9: [45, 0, 0], 1.2: [5, 0, 0]})
    add("walk", "shin_r", "rotation", {0: [45, 0, 0], 0.3: [5, 0, 0], 0.6: [5, 0, 0], 1.2: [45, 0, 0]})
    aw = 25 if W else 15
    add("walk", "arm_l", "rotation", {0: [aw + ax, 0, 3], 0.6: [-aw + ax, 0, 3], 1.2: [aw + ax, 0, 3]})
    add("walk", "arm_r", "rotation", {0: [-aw + ax, 0, -3], 0.6: [aw + ax, 0, -3], 1.2: [-aw + ax, 0, -3]})
    add("walk", "body", "position", {0: [0, 0, 0], 0.3: [0, -3, 0], 0.6: [0, 0, 0], 0.9: [0, -3, 0], 1.2: [0, 0, 0]})
    add("walk", "chest", "rotation", {0: [lean, 4, 0], 0.6: [lean, -4, 0], 1.2: [lean, 4, 0]} if W else
        {0: [lean, 5, 5], 0.6: [lean, -5, -5], 1.2: [lean, 5, 5]})
    add("walk", "head", "rotation", {0: [0, 0, hz], 0.6: [0, 0, -hz if not W else 0], 1.2: [0, 0, hz]})
    add("walk", "jaw", "rotation", {0: [j, 0, 0], 1.2: [j, 0, 0]})
    add("walk", "hair_back", "rotation", {0: [4, 0, 0], 0.6: [-4, 0, 0], 1.2: [4, 0, 0]})

    # ---------- attack
    meta("attack", 0.9, False)
    if W:
        add("attack", "arm_r", "rotation", {0: [0, 0, -3], 0.25: [60, 0, -10], 0.45: [-110, 0, 0], 0.9: [0, 0, -3]})
        add("attack", "forearm_r", "rotation", {0: [0, 0, 0], 0.25: [20, 0, 0], 0.45: [-25, 0, 0], 0.9: [0, 0, 0]})
        add("attack", "chest", "rotation", {0: [0, 0, 0], 0.25: [0, -15, 0], 0.45: [0, 20, 0], 0.9: [0, 0, 0]})
        add("attack", "jaw", "rotation", {0: [4, 0, 0], 0.3: [4, 0, 0], 0.45: [30, 0, 0], 0.9: [4, 0, 0]})
    else:
        for a in ("arm_l", "arm_r"):
            add("attack", a, "rotation", {0: [ax, 0, 0], 0.3: [-150, 0, 0], 0.5: [-25, 0, 0], 0.9: [ax, 0, 0]})
        add("attack", "chest", "rotation", {0: [lean, 0, 0], 0.3: [-8, 0, 0], 0.5: [25, 0, 0], 0.9: [lean, 0, 0]})
        add("attack", "jaw", "rotation", {0: [j, 0, 0], 0.5: [45, 0, 0], 0.9: [j, 0, 0]})
        add("attack", "head", "rotation", {0: [0, 0, hz], 0.3: [-15, 0, 0], 0.5: [15, 0, 0], 0.9: [0, 0, hz]})

    # ---------- roar (warrior transformation scream)
    if W:
        meta("roar", 3.0, False)
        add("roar", "jaw", "rotation", {0: [4, 0, 0], 0.4: [42, 0, 0], 2.5: [42, 0, 0], 3.0: [4, 0, 0]})
        add("roar", "head", "rotation", {0: [0, 0, 0], 0.4: [-24, 0, 0], 2.5: [-24, 0, 0], 3.0: [0, 0, 0]})
        add("roar", "chest", "scale", {0: [1, 1, 1], 0.4: [1.08, 1.04, 1.08], 2.5: [1.08, 1.04, 1.08], 3.0: [1, 1, 1]})
        add("roar", "chest", "rotation", {0: [0, 0, 0], 0.4: [-8, 0, 0], 2.5: [-8, 0, 0], 3.0: [0, 0, 0]})
        add("roar", "arm_l", "rotation", {0: [0, 0, 3], 0.4: [-10, 0, 40], 2.5: [-10, 0, 40], 3.0: [0, 0, 3]})
        add("roar", "arm_r", "rotation", {0: [0, 0, -3], 0.4: [-10, 0, -40], 2.5: [-10, 0, -40], 3.0: [0, 0, -3]})
        for e in ("eye_l", "eye_r"):
            add("roar", e, "scale", {0: [1, 1, 1], 0.4: [1.25, 1.25, 1], 2.5: [1.25, 1.25, 1], 3.0: [1, 1, 1]})
        add("roar", "hair_back", "rotation", {0: [0, 0, 0], 0.5: [10, 0, 0], 1.5: [-6, 0, 0], 3.0: [0, 0, 0]})
    return {"format_version": "1.8.0", "animations": A}

def icon_assets():
    os.makedirs(os.path.join(ASSETS, "textures/item"), exist_ok=True)
    os.makedirs(os.path.join(ASSETS, "textures/mob_effect"), exist_ok=True)
    # serum vial 16x16
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0)); px = im.load()
    glass, glassd, liq, liql, cork = (200, 225, 235, 255), (120, 150, 165, 255), (60, 230, 110, 255), (170, 255, 190, 255), (140, 95, 55, 255)
    for x in range(6, 10):
        px[x, 1] = cork; px[x, 2] = cork
    for x in range(6, 10): px[x, 3] = glassd
    for yy in range(4, 14):
        for x in range(4, 12):
            edge = x in (4, 11) or yy == 13
            px[x, yy] = glassd if edge else (liq if yy > 6 else (230, 240, 245, 120))
    for yy in range(7, 13): px[5, yy] = liql
    for x in (4, 11): px[x, 4] = (0, 0, 0, 0)
    im.save(os.path.join(ASSETS, "textures/item/giant_serum.png"))
    # effect icons 18x18
    for name, col in (("giant_power", (70, 235, 120)), ("giant_form", (235, 150, 60))):
        e = Image.new("RGBA", (18, 18), (0, 0, 0, 0)); p = e.load()
        for yy in range(18):
            for x in range(18):
                d = ((x - 8.5) ** 2 + (yy - 8.5) ** 2) ** 0.5
                if d < 8: p[x, yy] = (20, 30, 25, 255) if d > 6.6 else (*col, 255)
        # simple fist/eye shape
        for x in range(5, 13):
            p[x, 8] = (255, 255, 255, 255); p[x, 9] = (255, 255, 255, 255)
        for x in range(7, 11):
            p[x, 6] = (255, 255, 255, 255); p[x, 11] = (255, 255, 255, 255)
        e.save(os.path.join(ASSETS, "textures/mob_effect", name + ".png"))

if __name__ == "__main__":
    os.makedirs(os.path.join(ASSETS, "animations"), exist_ok=True)
    for var in ("warrior", "mindless"):
        m, S = make_assets(var)
        W = var == "warrior"
        an = anims(m, W)
        if not W:
            for a in an["animations"].values():
                for bd in a["bones"].values():
                    if "position" in bd:
                        bd["position"] = {k: [x * MIND_SCALE for x in v] for k, v in bd["position"].items()}
        json.dump(an, open(os.path.join(ASSETS, "animations", m.name + ".animation.json"), "w"), indent=1)
        print(m.name, "texture", S)
    icon_assets()
