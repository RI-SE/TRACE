# ===============================
# SIGNAL ENGINE
# ===============================

from dataclasses import dataclass
import math
import jsonschema

CONTEXT_SCHEMA = {
    "type": "object",
    "required": ["ready", "vehicles", "pair", "duration_seconds", "context_version"],
    "properties": {
        "ready": {"type": "boolean"},
        "context_version": {
            "type": "string",
            "const": "timeline_v2"
        },
        "duration_seconds": {
            "type": "number",
            "exclusiveMinimum": 0
        },
        "vehicles": {
            "type": "object",
            "required": ["V1", "V2"],
            "properties": {
                "V1": {"$ref": "#/definitions/vehicle"},
                "V2": {"$ref": "#/definitions/vehicle"}
            }
        },
        "pair": {
            "type": "object",
            "required": ["timeline"],
            "properties": {
                "timeline": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "distance": {"type": "number"},
                            "delta_s": {"type": "number"},
                            "delta_d": {"type": "number"}
                        },
                        "required": ["distance", "delta_s", "delta_d"]
                    }
                }
            }
        }
    },

    "definitions": {

        "vehicle": {
            "type": "object",
            "required": ["timeline"],
            "properties": {
                "timeline": {
                    "type": "object",
                    "minProperties": 2,
                    "additionalProperties": {
                        "$ref": "#/definitions/frame"
                    }
                }
            }
        },

        "frame": {
            "type": "object",
            "required": ["region_id", "world", "sd"],
            "properties": {

                "region_id": {
                    "anyOf": [
                        {"type": "integer"},
                        {"type": "null"}
                    ]
                },

                "world": {
                    "type": "object",
                    "required": ["x", "y"],
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"}
                    }
                },

                "sd": {
                    "type": "object",
                    "required": ["s", "d"],
                    "properties": {
                        "s": {"type": "number"},
                        "d": {"type": "number"}
                    }
                }
            }
        }
    }
}


CONTEXT_VERSION = "timeline_v2"

REQUIRED_TOP_LEVEL = {
    "ready": bool,
    "vehicles": dict,
    "pair": dict,
    "duration_seconds": (int, float),
    "context_version": str,
}

REQUIRED_VEHICLE = {
    "timeline": dict,
}

REQUIRED_FRAME = {
    "region_id": (int, type(None)),
    "world": object,   # expects .x, .y OR (x,y)
    "sd": dict,
}

REQUIRED_SD = {
    "s": (int, float),
    "d": (int, float),
}

_VALIDATOR = jsonschema.Draft7Validator(CONTEXT_SCHEMA)

@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float

@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float

def _dbg(msg, data=None):
    try:
        print(f"[CTX-DEBUG] {msg}")
        if data is not None:
            if isinstance(data, dict):
                print(f"  keys: {list(data.keys())[:10]}")
                print(f"  len: {len(data)}")
            else:
                print(f"  {data}")
    except Exception:
        pass

def validate_context(ctx: dict):
    if not isinstance(ctx, dict):
        raise RuntimeError("Context must be dict")

    errors = list(_VALIDATOR.iter_errors(ctx))
    if errors:
        raise RuntimeError(f"Schema validation failed: {errors[0].message}")

    _validate_timeline_alignment(ctx)

    return True


def _validate_timeline_alignment(ctx):
    try:
        v1 = ctx["vehicles"]["V1"]["timeline"]
        v2 = ctx["vehicles"]["V2"]["timeline"]
        p  = ctx["pair"]["timeline"]
    except Exception:
        raise RuntimeError("Missing timelines")

    try:
        k1 = sorted(map(int, v1.keys()))
        k2 = sorted(map(int, v2.keys()))
        kp = sorted(map(int, p.keys()))
    except Exception:
        raise RuntimeError("Timeline keys must be int")

    if not (k1 == k2 == kp):
        raise RuntimeError("Timeline mismatch between V1/V2/pair")

def _normalize_world(world):
    if not world:
        return {"x": float("nan"), "y": float("nan")}

    if isinstance(world, dict):
        return {
            "x": float(world.get("x", float("nan"))),
            "y": float(world.get("y", float("nan")))
        }

    if isinstance(world, (list, tuple)) and len(world) >= 2:
        return {
            "x": float(world[0]),
            "y": float(world[1])
        }

    if hasattr(world, "x") and hasattr(world, "y"):
        return {
            "x": float(world.x),
            "y": float(world.y)
        }

    return {"x": float("nan"), "y": float("nan")}

def normalize_context(ctx: dict):
    if not ctx:
        return ctx

    if "context_version" not in ctx:
        _dbg("[FIX] injecting context_version")
        ctx["context_version"] = "timeline_v2"

    if "duration_seconds" not in ctx or not ctx.get("duration_seconds"):
        _dbg("[FIX] injecting default duration_seconds")
        ctx["duration_seconds"] = 1.0
    

    # ---------- keys → int ----------
    for vid in ("V1", "V2"):
        tl = ctx["vehicles"][vid].get("timeline", {})
        ctx["vehicles"][vid]["timeline"] = {
            int(k): v for k, v in tl.items()
        }

    pt = ctx["pair"].get("timeline", {})
    ctx["pair"]["timeline"] = {
        int(k): v for k, v in pt.items()
    }

    # ---------- world normalization ----------
    for vid in ("V1", "V2"):
        for f in ctx["vehicles"][vid]["timeline"].values():
            f["world"] = _normalize_world(f.get("world"))

    # ---------- sd normalization ----------
    for vid in ("V1", "V2"):
        for f in ctx["vehicles"][vid]["timeline"].values():
            sd = f.get("sd") or {}
            f["sd"] = {
                "s": float(sd.get("s", float("nan"))),
                "d": float(sd.get("d", float("nan")))
            }

    # ---------- align timelines ----------
    v1 = ctx["vehicles"]["V1"]["timeline"]
    v2 = ctx["vehicles"]["V2"]["timeline"]
    p  = ctx["pair"]["timeline"]

    common = sorted(set(v1) & set(v2) & set(p))

    ctx["vehicles"]["V1"]["timeline"] = {k: v1[k] for k in common}
    ctx["vehicles"]["V2"]["timeline"] = {k: v2[k] for k in common}
    ctx["pair"]["timeline"] = {k: p[k] for k in common}

    # ---------- NaN normalization (pair) ----------
    for f in ctx["pair"]["timeline"].values():
        for m in ("distance", "delta_s", "delta_d"):
            val = f.get(m)
            if not isinstance(val, (int, float)):
                f[m] = float("nan")

    # ---------- duration meta ----------
    n = len(common)
    dur = ctx.get("duration_seconds", 0)

    if n >= 2 and dur > 0:
        ctx.setdefault("meta", {})
        ctx["meta"]["dt"] = dur / (n - 1)
        ctx["meta"]["frame_count"] = n

    return ctx

def extract_sd_sequences(ctx: dict):
    """
    Extracts aligned SD sequences for V1 and V2.

    Returns:
        {
            "V1": [{s, d, t}, ...],
            "V2": [{s, d, t}, ...]
        }
    """
    vehicles = ctx.get("vehicles", {})
    out = {}

    for vid in ("V1", "V2"):
        tl = vehicles.get(vid, {}).get("timeline", {})
        seq = []

        for k in sorted(tl.keys()):
            f = tl[k]
            sd = f.get("sd") or {}

            seq.append({
                "s": float(sd.get("s", float("nan"))),
                "d": float(sd.get("d", float("nan"))),
                "t": float(k)
            })

        out[vid] = seq

    return out

def compute_derivatives(sd_seq):
    """
    Computes longitudinal derivatives:
    speed, acceleration, jerk
    """
    speeds, accs, jerks = [], [], []

    for i in range(1, len(sd_seq)):
        p0, p1 = sd_seq[i-1], sd_seq[i]
        dt = p1["t"] - p0["t"]

        if dt <= 0:
            continue

        ds = p1["s"] - p0["s"]
        speeds.append(ds / dt)

    for i in range(1, len(speeds)):
        accs.append(speeds[i] - speeds[i - 1])

    for i in range(1, len(accs)):
        jerks.append(accs[i] - accs[i - 1])

    return {
        "speed": speeds,
        "acc": accs,
        "jerk": jerks,
    }

def compute_lateral_derivatives(sd_seq):
    """
    Computes lateral velocity, acceleration
    """
    d_vals = [p["d"] for p in sd_seq if p.get("d") is not None]

    if len(d_vals) < 2:
        return {"lat_vel": [], "lat_acc": []}

    lat_vel = [
        d_vals[i] - d_vals[i-1]
        for i in range(1, len(d_vals))
    ]

    lat_acc = [
        lat_vel[i] - lat_vel[i-1]
        for i in range(1, len(lat_vel))
    ]

    return {
        "lat_vel": lat_vel,
        "lat_acc": lat_acc,
    }

def enrich_pairwise_timeline(ctx: dict):
    """
    Enrich existing ctx["pair"]["timeline"] with:

        lat_distance
        closing_rate
        lateral_rate

    This is SAFE:
    - Does not overwrite existing fields
    - Adds only derived signal-level data
    """

    pair = ctx.get("pair", {}).get("timeline", {})
    if not pair:
        return ctx

    keys = sorted(pair.keys())

    prev = None

    for k in keys:
        p = pair[k]

        dist = p.get("distance")
        ds   = p.get("delta_s")
        dd   = p.get("delta_d")

        # ---- lateral distance ----
        if isinstance(dd, (int, float)):
            p["lat_distance"] = abs(dd)
        else:
            p["lat_distance"] = float("nan")

        # ---- rates ----
        if prev and isinstance(dist, (int, float)) and isinstance(prev.get("distance"), (int, float)):
            dt = float(k - prev["k"])

            if dt > 0:
                p["closing_rate"] = (prev["distance"] - dist) / dt

                prev_dd = prev.get("delta_d")
                if isinstance(prev_dd, (int, float)) and isinstance(dd, (int, float)):
                    p["lateral_rate"] = (dd - prev_dd) / dt
                else:
                    p["lateral_rate"] = 0.0
            else:
                p["closing_rate"] = 0.0
                p["lateral_rate"] = 0.0
        else:
            p["closing_rate"] = 0.0
            p["lateral_rate"] = 0.0

        p["_k"] = k  # internal helper
        prev = {"distance": dist, "delta_d": dd, "_k": k}

    # cleanup internal helper
    for k in keys:
        pair[k].pop("_k", None)

    return ctx

def extract_pair_series(ctx: dict):
    """
    Converts pair timeline → ordered list

    Returns:
        [
            {
                t, distance, ds, dd,
                lat_distance,
                closing_rate,
                lateral_rate
            }
        ]
    """

    pt = ctx.get("pair", {}).get("timeline", {})
    out = []

    for k in sorted(pt.keys()):
        p = pt[k]

        out.append({
            "t": float(k),
            "distance": p.get("distance"),
            "ds": p.get("delta_s"),
            "dd": p.get("delta_d"),
            "lat_distance": p.get("lat_distance"),
            "closing_rate": p.get("closing_rate"),
            "lateral_rate": p.get("lateral_rate"),
        })

    return out

# =========================================================
# KINEMATICS + PAIRWISE 
# =========================================================
class Kinematics:

    @staticmethod
    def compute_velocity_from_timeline(timeline, dt=1.0):
        """
        Adds:
            vx, vy, speed
            fs (frenet ds if available)
        """
        keys = sorted(timeline.keys())

        for i in range(1, len(keys)):
            k0 = keys[i - 1]
            k1 = keys[i]

            p0 = timeline[k0]
            p1 = timeline[k1]

            # ---- WORLD VELOCITY ----
            if "world" in p0 and "world" in p1:
                w0 = p0.get("world") or {}
                w1 = p1.get("world") or {}

                x0, y0 = float(w0.get("x", 0.0)), float(w0.get("y", 0.0))
                x1, y1 = float(w1.get("x", 0.0)), float(w1.get("y", 0.0))

                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt

                p1["vx"] = vx
                p1["vy"] = vy
                p1["speed"] = math.hypot(vx, vy)

            # ---- FRENET (GLOBAL S) ----
            if p0.get("s_global") is not None and p1.get("s_global") is not None:
                p1["frenet_velocity"] = {
                    "ds": (p1["s_global"] - p0["s_global"]) / dt
                }


class PairwiseComputer:

    @staticmethod
    def from_timelines(tl1, tl2):

        timeline = {}
        keys = sorted(set(tl1) & set(tl2))

        for k in keys:
            v1 = tl1.get(k)
            v2 = tl2.get(k)

            if not v1 or not v2 or not (v1.get("world") and v2.get("world")):
                timeline[k] = {
                    "distance": float("nan"),
                    "delta_s": float("nan"),
                    "delta_d": float("nan"),
                }
                continue
            w1 = v1.get("world") or {}
            w2 = v2.get("world") or {}

            x1, y1 = float(w1.get("x", 0)), float(w1.get("y", 0))
            x2, y2 = float(w2.get("x", 0)), float(w2.get("y", 0))

            dist = math.hypot(x2 - x1, y2 - y1)

            sd1 = v1.get("sd") or {}
            sd2 = v2.get("sd") or {}

            ds = float(sd2.get("s", float("nan")) - sd1.get("s", float("nan")))
            dd = float(sd2.get("d", float("nan")) - sd1.get("d", float("nan")))

            timeline[k] = {
                "distance": dist,
                "delta_s": ds,
                "delta_d": dd,
            }

        return {"timeline": timeline}


class VehicleTrajectory:
    def __init__(self, start_xy, end_xy):
        if start_xy is None or end_xy is None:
            raise ValueError("start_xy and end_xy must not be None")

        if len(start_xy) != 2 or len(end_xy) != 2:
            raise ValueError("start_xy and end_xy must be (x, y)")

        self.start = (float(start_xy[0]), float(start_xy[1]))
        self.end = (float(end_xy[0]), float(end_xy[1]))

        self.topology = {}
        self.ref_xy = None

    def attach_topology(self, mask=None, meta=None, ref_xy=None):
        self.ref_xy = ref_xy
        self.topology = {
            "mask": mask,
            "meta": meta,
            "ref_xy": ref_xy,
        }

    @property
    def direction(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        mag = (dx**2 + dy**2) ** 0.5
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    @property
    def length(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx**2 + dy**2) ** 0.5

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "ref_xy": self.ref_xy,
        }


@dataclass
class SignalConfig:
    same_lane_dd: float = 1.2
    aligned_ds: float = 1.0
    lateral_threshold: float = 0.5
    longitudinal_threshold: float = 15.0
    precomp_dist_px: float = 80.0
    precomp_ttc_frames: float = 3.0


class SignalComputer:
    def __init__(self, config: SignalConfig = None):
        self.cfg = config or SignalConfig()

    def compute_frenet_relation(self, sd1, sd2):
        if not sd1 or not sd2:
            return {}, {}

        ds = float(sd2["s"] - sd1["s"])
        dd = float(sd2["d"] - sd1["d"])

        cfg = self.cfg

        return (
            {"delta_s": ds, "delta_d": dd},
            {
                "same_lane": abs(dd) < cfg.same_lane_dd,
                "ahead": ds > cfg.aligned_ds,
                "behind": ds < -cfg.aligned_ds,
                "aligned": abs(ds) <= cfg.aligned_ds,
                "left": dd > cfg.lateral_threshold,
                "right": dd < -cfg.lateral_threshold,
            },
        )

    def compute_distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def compute_ttc(self, s1, s2, v1, v2):
        if v1 is None or v2 is None:
            return None

        rel_speed = v1 - v2
        if abs(rel_speed) < 1e-3:
            return None

        return (s2 - s1) / rel_speed

    def is_interesting_pair(self, d, ttc, sd1, sd2):
        cfg = self.cfg

        same_lane_sd = False
        close_longitudinal = False

        if sd1 and sd2:
            ds = sd2["s"] - sd1["s"]
            dd = abs(sd2["d"] - sd1["d"])

            same_lane_sd = dd < cfg.same_lane_dd
            close_longitudinal = abs(ds) < cfg.longitudinal_threshold

        return (
            same_lane_sd
            or close_longitudinal
            or (d is not None and d <= cfg.precomp_dist_px)
            or (ttc is not None and ttc <= cfg.precomp_ttc_frames)
        )


# =========================================================
# CONTEXT BUILDER 
# =========================================================
class ContextBuilder:

    def __init__(self, road, config=None):
        self.road = road
        self.sc = SignalComputer(config)

    def _compute_s_global(self, timeline):
        """
        Build continuous longitudinal coordinate across lanes.
        """

        keys = sorted(timeline.keys())

        offset = 0.0
        prev_sd = None
        prev_region = None

        for k in keys:
            f = timeline[k]
            sd = f.get("sd")
            region = f.get("region_id")

            if not sd:
                continue

            s_val = sd.get("s")

            if s_val is None or not isinstance(s_val, (int, float)) or math.isnan(s_val):
                prev_region = region
                continue

            if prev_sd and region != prev_region:
                if prev_sd.get("s_global") is not None:
                    offset = prev_sd["s_global"]

            f["s_global"] = offset + s_val

            prev_sd = sd
            prev_region = region

        
    def build_vehicle_context(self, trajectory, dt=1.0):

        if not trajectory:
            return {"timeline": {}}

        timeline = {}

        for item in trajectory:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                # GUI format: (t, x, y)
                t, x, y = item

                timeline[int(t)] = {
                    "world": {
                        "x": float(x),
                        "y": float(y)
                    },
                    "region_id": None,
                    "sd": {
                        "s": float("nan"),
                        "d": float("nan")
                    },
                    "s_global": None
                }

            elif isinstance(item, (list, tuple)) and len(item) == 2:
                # expected format: (t, frame_dict)
                t, frame = item

                w = frame.get("world") or {}
                sd = frame.get("sd") or {}

                timeline[int(t)] = {
                    "world": {
                        "x": float(w.get("x", float("nan"))),
                        "y": float(w.get("y", float("nan"))),
                    },
                    "region_id": frame.get("region_id"),
                    "sd": {
                        "s": float(sd.get("s", float("nan"))),
                        "d": float(sd.get("d", float("nan"))),
                    },
                    "s_global": None,
                }

        self._compute_s_global(timeline)
        Kinematics.compute_velocity_from_timeline(timeline, dt)

        return {"timeline": timeline}



    def enrich_vehicle(self, ctx, dt=1.0):
        if not ctx or "timeline" not in ctx:
            return ctx

        Kinematics.compute_velocity_from_timeline(ctx["timeline"], dt)
        return ctx

    def build_pairwise_auto(self, ctx1, ctx2):
        return PairwiseComputer.from_timelines(
            ctx1["timeline"], ctx2["timeline"]
        )

    def build_pairwise(self, ctx1, ctx2):

        timeline = {}
        frames = sorted(set(ctx1["timeline"]) & set(ctx2["timeline"]))

        for t in frames:
            v1 = ctx1["timeline"][t]
            v2 = ctx2["timeline"][t]

            if not v1.get("world") or not v2.get("world"):
                timeline[t] = {
                    "distance": float("nan"),
                    "delta_s": float("nan"),
                    "delta_d": float("nan"),
                    "lane_relation": {},
                    "ttc_s": None,
                    "interesting": False,
                }
                continue

            w1 = v1.get("world") or {}
            w2 = v2.get("world") or {}

            x1, y1 = float(w1.get("x", 0)), float(w1.get("y", 0))
            x2, y2 = float(w2.get("x", 0)), float(w2.get("y", 0))

            dist = self.sc.compute_distance(x1, y1, x2, y2)

            sd1 = v1.get("sd") or {}
            sd2 = v2.get("sd") or {}

            ds = float(sd2.get("s", float("nan")) - sd1.get("s", float("nan")))
            dd = float(sd2.get("d", float("nan")) - sd1.get("d", float("nan")))

            relation = {}
            if self.road and v1.get("region_id") and v2.get("region_id"):
                try:
                    relation = self.road.compute_lane_relation(
                        v1["region_id"], v2["region_id"]
                    )
                except Exception:
                    pass

            fv1 = v1.get("frenet_velocity", {})
            fv2 = v2.get("frenet_velocity", {})

            ttc = self.sc.compute_ttc(
                v1.get("s_global"),
                v2.get("s_global"),
                fv1.get("ds"),
                fv2.get("ds"),
            )

            timeline[t] = {
                "distance": dist,
                "delta_s": ds,
                "delta_d": dd,
                "lane_relation": relation,
                "ttc_s": ttc,
                "interesting": self.sc.is_interesting_pair(dist, ttc, sd1, sd2),
            }

        return {"timeline": timeline}
