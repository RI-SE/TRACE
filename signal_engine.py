# ===============================
# SIGNAL ENGINE (compact)
# ===============================

from dataclasses import dataclass
import math
import jsonschema

# ---------------------------------------------------------
# Context schema (legacy timeline_v2; map_scenario_context
# is validated structurally in validate_context)
# ---------------------------------------------------------

CONTEXT_SCHEMA = {
    "type": "object",
    "required": ["ready", "vehicles", "pair", "duration_seconds", "context_version"],
    "properties": {
        "ready": {"type": "boolean"},
        "context_version": {
            "type": "string",
            "enum": ["timeline_v2", "map_scenario_context_v1"]
        },
        "duration_seconds": {"type": "number", "exclusiveMinimum": 0},
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
                    "additionalProperties": {"$ref": "#/definitions/frame"}
                }
            }
        },
        "frame": {
            "type": "object",
            "required": ["region_id", "world", "sd"],
            "properties": {
                "region_id": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "world": {
                    "type": "object",
                    "required": ["x", "y"],
                    "properties": {"x": {"type": "number"}, "y": {"type": "number"}}
                },
                "sd": {
                    "type": "object",
                    "required": ["s", "d"],
                    "properties": {"s": {"type": "number"}, "d": {"type": "number"}}
                }
            }
        }
    }
}

LEGACY_CONTEXT_VERSION = "timeline_v2"
MAP_SCENARIO_CONTEXT_VERSION = "map_scenario_context_v1"

_VALIDATOR = jsonschema.Draft7Validator(CONTEXT_SCHEMA)


@dataclass(frozen=True)
class PixelPoint:
    x: float
    y: float


@dataclass(frozen=True)
class WorldPoint:
    x: float
    y: float


# ---------------------------------------------------------
# small helpers
# ---------------------------------------------------------

def _f(v, default=float("nan")):
    """Safe float: None/str/missing -> default (NaN)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _normalize_world(world):
    if isinstance(world, dict):
        return {"x": _f(world.get("x")), "y": _f(world.get("y"))}
    if isinstance(world, (list, tuple)) and len(world) >= 2:
        return {"x": _f(world[0]), "y": _f(world[1])}
    if hasattr(world, "x") and hasattr(world, "y"):
        return {"x": _f(world.x), "y": _f(world.y)}
    return {"x": float("nan"), "y": float("nan")}


# ---------------------------------------------------------
# validation
# ---------------------------------------------------------

def validate_context(ctx: dict):
    if not isinstance(ctx, dict):
        raise RuntimeError("Context must be dict")

    if ctx.get("kind") == "map_scenario_context":
        missing = [
            k for k in ("actors", "interaction_topology", "scope")
            if k not in ctx
        ]
        if missing:
            raise RuntimeError(
                "Missing MapScenarioContext fields: " + ", ".join(missing)
            )
        return True

    errors = list(_VALIDATOR.iter_errors(ctx))
    if errors:
        raise RuntimeError(f"Schema validation failed: {errors[0].message}")
    _validate_timeline_alignment(ctx)
    return True


def _validate_timeline_alignment(ctx):
    try:
        k1 = sorted(map(int, ctx["vehicles"]["V1"]["timeline"]))
        k2 = sorted(map(int, ctx["vehicles"]["V2"]["timeline"]))
        kp = sorted(map(int, ctx["pair"]["timeline"]))
    except Exception:
        raise RuntimeError("Missing timelines or non-int keys")
    if not (k1 == k2 == kp):
        raise RuntimeError("Timeline mismatch between V1/V2/pair")


# ---------------------------------------------------------
# normalization
# ---------------------------------------------------------

def _set_meta(ctx, n):
    meta = ctx.setdefault("meta", {})
    meta["frame_count"] = n
    if n >= 2 and ctx["duration_seconds"] > 0:
        meta["dt"] = ctx["duration_seconds"] / (n - 1)


def _norm_vehicle_timeline(tl):
    """Copy frames (no aliasing), int keys, normalized world/sd."""
    norm = {}
    for k, frame in (tl or {}).items():
        frame = dict(frame)
        frame["world"] = _normalize_world(frame.get("world"))
        sd = frame.get("sd") or {}
        frame["sd"] = {"s": _f(sd.get("s")), "d": _f(sd.get("d"))}
        norm[int(k)] = frame
    return norm


def normalize_context(ctx):
    if not ctx:
        return ctx

    if "context_version" not in ctx:
        ctx["context_version"] = (
            MAP_SCENARIO_CONTEXT_VERSION
            if ctx.get("kind") == "map_scenario_context"
            else LEGACY_CONTEXT_VERSION
        )
    if not ctx.get("duration_seconds"):
        ctx["duration_seconds"] = 1.0

    # ---------------- map_scenario_context v1 ----------------
    if ctx.get("kind") == "map_scenario_context":
        actors = ctx.setdefault("actors", {})
        n = 0
        for actor in ("A", "B"):
            a = actors.setdefault(actor, {})
            a["timeline"] = _norm_vehicle_timeline(a.get("timeline"))
            if actor == "A":
                n = len(a["timeline"])

        interaction = ctx.setdefault("interaction_facts", {})
        itl = interaction.get("timeline") or {}
        interaction["timeline"] = {int(k): v for k, v in itl.items()}

        ctx.setdefault("meta", {})
        _set_meta(ctx, n)
        return ctx

    # ---------------- legacy timeline_v2 ----------------
    vehicles = ctx.setdefault("vehicles", {})
    for vid in ("V1", "V2"):
        v = vehicles.setdefault(vid, {})
        v["timeline"] = _norm_vehicle_timeline(v.get("timeline"))

    pair = ctx.setdefault("pair", {})
    norm_p = {}
    for k, p in (pair.get("timeline") or {}).items():
        p = dict(p)
        for m in ("distance", "delta_s", "delta_d"):
            if not isinstance(p.get(m), (int, float)):
                p[m] = float("nan")
        norm_p[int(k)] = p
    pair["timeline"] = norm_p

    common = sorted(
        set(vehicles["V1"]["timeline"])
        & set(vehicles["V2"]["timeline"])
        & set(norm_p)
    )
    for vid in ("V1", "V2"):
        vehicles[vid]["timeline"] = {
            k: vehicles[vid]["timeline"][k] for k in common
        }
    pair["timeline"] = {k: norm_p[k] for k in common}

    _set_meta(ctx, len(common))
    return ctx


# ---------------------------------------------------------
# kinematics + pairwise
# ---------------------------------------------------------

class Kinematics:

    @staticmethod
    def compute_velocity_from_timeline(timeline, dt=1.0):
        """Adds vx/vy/speed and frenet_velocity.ds per frame.

        Uses the ACTUAL frame delta (k1-k0) so dropped frames don't
        skew velocities; skips frames with non-finite coordinates.
        With dt=1.0 units are per-frame (pass dt=1/fps for m/s).
        """
        keys = sorted(timeline.keys())

        for i in range(1, len(keys)):
            p0, p1 = timeline[keys[i - 1]], timeline[keys[i]]
            dtk = keys[i] - keys[i - 1]
            if dtk <= 0:
                continue
            step = dtk * dt

            w0 = p0.get("world") or {}
            w1 = p1.get("world") or {}
            x0, y0 = _f(w0.get("x")), _f(w0.get("y"))
            x1, y1 = _f(w1.get("x")), _f(w1.get("y"))

            if (
                math.isfinite(x0) and math.isfinite(y0)
                and math.isfinite(x1) and math.isfinite(y1)
            ):
                vx = (x1 - x0) / step
                vy = (y1 - y0) / step
                p1["vx"] = vx
                p1["vy"] = vy
                p1["speed"] = math.hypot(vx, vy)

            g0, g1 = p0.get("s_global"), p1.get("s_global")
            if g0 is not None and g1 is not None:
                p1["frenet_velocity"] = {"ds": (g1 - g0) / step}


def _pair_sample(v1, v2, relation=None):
    """One pair-frame: world distance + frenet deltas (NaN-safe)."""
    w1 = v1.get("world") or {}
    w2 = v2.get("world") or {}
    dist = math.hypot(
        _f(w2.get("x")) - _f(w1.get("x")),
        _f(w2.get("y")) - _f(w1.get("y")),
    )
    sd1 = v1.get("sd") or {}
    sd2 = v2.get("sd") or {}
    s1, s2 = _f(sd1.get("s")), _f(sd2.get("s"))
    d1, d2 = _f(sd1.get("d")), _f(sd2.get("d"))
    return {
        "distance": dist,
        "delta_s": s2 - s1,
        "delta_d": d2 - d1,
        "lane_relation": relation or {},
        "ttc_s": None,
        "interesting": False,
    }


class PairwiseComputer:

    @staticmethod
    def from_timelines(tl1, tl2):
        return {
            "timeline": {
                k: _pair_sample(tl1[k], tl2[k])
                for k in sorted(set(tl1) & set(tl2))
                if tl1.get(k) and tl2.get(k)
            }
        }


@dataclass
class SignalConfig:
    same_lane_dd: float = 1.2
    aligned_ds: float = 1.0
    lateral_threshold: float = 0.5
    longitudinal_threshold: float = 15.0
    # WORLD meters (was precomp_dist_px=80.0 - a pixel-era value that
    # was applied to post-homography meter distances, flagging almost
    # every pair as interesting)
    interest_dist_m: float = 8.0
    # frames (frenet ds is per-frame because Kinematics runs with dt=1.0)
    interest_ttc_frames: float = 3.0


class SignalComputer:

    def __init__(self, config: SignalConfig = None):
        self.cfg = config or SignalConfig()

    def compute_frenet_relation(self, sd1, sd2):
        if not sd1 or not sd2:
            return {}, {}
        ds = _f(sd2.get("s")) - _f(sd1.get("s"))
        dd = _f(sd2.get("d")) - _f(sd1.get("d"))
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

    @staticmethod
    def compute_distance(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def compute_ttc(s1, s2, v1, v2):
        if None in (s1, s2, v1, v2):
            return None
        rel_speed = v1 - v2
        if abs(rel_speed) < 1e-3:
            return None
        return (s2 - s1) / rel_speed

    def is_interesting_pair(self, d, ttc, sd1, sd2):
        cfg = self.cfg
        same_lane = False
        close_longitudinal = False

        if sd1 and sd2:
            s1, s2 = sd1.get("s"), sd2.get("s")
            d1, d2 = sd1.get("d"), sd2.get("d")
            if all(isinstance(v, (int, float)) for v in (s1, s2, d1, d2)):
                same_lane = abs(d2 - d1) < cfg.same_lane_dd
                close_longitudinal = abs(s2 - s1) < cfg.longitudinal_threshold

        return (
            same_lane
            or close_longitudinal
            or (
                isinstance(d, (int, float)) and math.isfinite(d)
                and d <= cfg.interest_dist_m
            )
            or (
                isinstance(ttc, (int, float)) and math.isfinite(ttc)
                and ttc <= cfg.interest_ttc_frames
            )
        )


# ---------------------------------------------------------
# context builder
# ---------------------------------------------------------

class ContextBuilder:

    def __init__(self, road, config=None):
        self.road = road
        self.sc = SignalComputer(config)

    def _compute_s_global(self, timeline):
        """Build a continuous longitudinal coordinate.

        Handles BOTH input conventions:
        - lane-local s (resets on lane change): a DROP in s across a
          lane change is a reset -> carry progress over so g stays
          continuous;
        - already-global s (what the viewer's _traj_to_builder_format
          produces): s never drops -> offset stays untouched, g == s.

        (Old version read prev_sd["s_global"] from the raw sd dict,
        which never has that key, so the offset was dead and this
        silently reduced to g == s.)
        """
        offset = 0.0
        prev_s = None
        prev_region = None
        prev_g = None

        for k in sorted(timeline):
            f = timeline[k]
            s = (f.get("sd") or {}).get("s")
            region = f.get("region_id")

            if not isinstance(s, (int, float)) or math.isnan(s):
                f["s_global"] = None
                continue

            if (
                prev_s is not None
                and region is not None
                and region != prev_region
                and s < prev_s - 1e-6
            ):
                offset = (prev_g - s) if prev_g is not None else 0.0

            f["s_global"] = offset + s
            prev_s, prev_region, prev_g = s, region, f["s_global"]

    def build_vehicle_context(self, trajectory, dt=1.0):
        if not trajectory:
            return {"timeline": {}}

        timeline = {}

        for item in trajectory:
            if not isinstance(item, (list, tuple)):
                continue

            if len(item) == 3:
                # GUI format: (t, x, y)
                t, x, y = item
                timeline[int(t)] = {
                    "world": {"x": _f(x), "y": _f(y)},
                    "region_id": None,
                    "sd": {"s": float("nan"), "d": float("nan")},
                    "s_global": None,
                }
            elif len(item) == 2:
                # builder format: (t, frame_dict)
                t, frame = item
                w = frame.get("world") or {}
                sd = frame.get("sd") or {}
                timeline[int(t)] = {
                    "world": {"x": _f(w.get("x")), "y": _f(w.get("y"))},
                    "region_id": frame.get("region_id"),
                    "sd": {"s": _f(sd.get("s")), "d": _f(sd.get("d"))},
                    "s_global": None,
                }

        self._compute_s_global(timeline)
        Kinematics.compute_velocity_from_timeline(timeline, dt)

        return {"timeline": timeline}

    def enrich_vehicle(self, ctx, dt=1.0):
        if ctx and "timeline" in ctx:
            Kinematics.compute_velocity_from_timeline(ctx["timeline"], dt)
        return ctx

    def build_pairwise_auto(self, ctx1, ctx2):
        return PairwiseComputer.from_timelines(
            ctx1["timeline"], ctx2["timeline"]
        )

    def build_pairwise(self, ctx1, ctx2):
        frames = sorted(set(ctx1["timeline"]) & set(ctx2["timeline"]))

        # (r1, r2) pairs repeat for long stretches - cache the road
        # lookup instead of calling compute_lane_relation per frame.
        rel_cache = {}
        out = {}

        for t in frames:
            v1 = ctx1["timeline"][t]
            v2 = ctx2["timeline"][t]

            r1 = v1.get("region_id")
            r2 = v2.get("region_id")

            relation = {}
            if r1 and r2 and self.road is not None:
                key = (r1, r2)
                relation = rel_cache.get(key)
                if relation is None:
                    try:
                        relation = self.road.compute_lane_relation(r1, r2) or {}
                    except Exception:
                        relation = {}
                    rel_cache[key] = relation

            sample = _pair_sample(v1, v2, relation)

            fv1 = v1.get("frenet_velocity") or {}
            fv2 = v2.get("frenet_velocity") or {}
            sample["ttc_s"] = self.sc.compute_ttc(
                v1.get("s_global"), v2.get("s_global"),
                fv1.get("ds"), fv2.get("ds"),
            )
            sample["interesting"] = self.sc.is_interesting_pair(
                sample["distance"], sample["ttc_s"],
                v1.get("sd"), v2.get("sd"),
            )

            out[t] = sample

        return {"timeline": out}


# ---------------------------------------------------------
# trajectories
# ---------------------------------------------------------

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
        self.topology = {"mask": mask, "meta": meta, "ref_xy": ref_xy}

    @property
    def direction(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        mag = (dx ** 2 + dy ** 2) ** 0.5
        if mag < 1e-6:
            return (0.0, 0.0)
        return (dx / mag, dy / mag)

    @property
    def length(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (dx ** 2 + dy ** 2) ** 0.5

    def as_dict(self):
        return {"start": self.start, "end": self.end, "ref_xy": self.ref_xy}


# =========================================================
# LEGACY timeline_v2 helpers
# (kept for API compatibility - nothing in the
#  map_scenario_context flow uses them; delete this
#  section if no other module imports these names)
# =========================================================

def extract_sd_sequences(ctx: dict):
    out = {}
    for vid in ("V1", "V2"):
        tl = (ctx.get("vehicles") or {}).get(vid, {}).get("timeline", {})
        out[vid] = [
            {
                "s": _f((f.get("sd") or {}).get("s")),
                "d": _f((f.get("sd") or {}).get("d")),
                "t": float(k),
            }
            for k, f in sorted(tl.items(), key=lambda kv: int(kv[0]))
        ]
    return out


def compute_derivatives(sd_seq):
    speeds = []
    for i in range(1, len(sd_seq)):
        dt = sd_seq[i]["t"] - sd_seq[i - 1]["t"]
        if dt <= 0:
            continue
        speeds.append((sd_seq[i]["s"] - sd_seq[i - 1]["s"]) / dt)
    accs = [speeds[i] - speeds[i - 1] for i in range(1, len(speeds))]
    jerks = [accs[i] - accs[i - 1] for i in range(1, len(accs))]
    return {"speed": speeds, "acc": accs, "jerk": jerks}


def compute_lateral_derivatives(sd_seq):
    d_vals = [p["d"] for p in sd_seq if p.get("d") is not None]
    lat_vel = [d_vals[i] - d_vals[i - 1] for i in range(1, len(d_vals))]
    lat_acc = [lat_vel[i] - lat_vel[i - 1] for i in range(1, len(lat_vel))]
    return {"lat_vel": lat_vel, "lat_acc": lat_acc}


def enrich_pairwise_timeline(ctx: dict):
    """Adds lat_distance / closing_rate / lateral_rate to pair frames.

    Fixed: old version read prev["k"] while storing "_k" (KeyError on
    the 2nd frame) and wrote a "_k" helper into every frame.
    """
    pair = ctx.get("pair", {}).get("timeline", {})
    if not pair:
        return ctx

    prev_k = None
    prev_dist = None
    prev_dd = None

    for k in sorted(pair):
        p = pair[k]
        dist = p.get("distance")
        dd = p.get("delta_d")

        p["lat_distance"] = (
            abs(dd) if isinstance(dd, (int, float)) else float("nan")
        )

        if (
            prev_k is not None
            and isinstance(dist, (int, float))
            and isinstance(prev_dist, (int, float))
        ):
            dtk = k - prev_k
            if dtk > 0:
                p["closing_rate"] = (prev_dist - dist) / dtk
                p["lateral_rate"] = (
                    (dd - prev_dd) / dtk
                    if isinstance(dd, (int, float)) and isinstance(prev_dd, (int, float))
                    else 0.0
                )
            else:
                p["closing_rate"] = 0.0
                p["lateral_rate"] = 0.0
        else:
            p["closing_rate"] = 0.0
            p["lateral_rate"] = 0.0

        prev_k, prev_dist, prev_dd = k, dist, dd

    return ctx


def extract_pair_series(ctx: dict):
    pt = ctx.get("pair", {}).get("timeline", {})
    return [
        {
            "t": float(k),
            "distance": p.get("distance"),
            "ds": p.get("delta_s"),
            "dd": p.get("delta_d"),
            "lat_distance": p.get("lat_distance"),
            "closing_rate": p.get("closing_rate"),
            "lateral_rate": p.get("lateral_rate"),
        }
        for k, p in sorted(pt.items(), key=lambda kv: int(kv[0]))
    ]