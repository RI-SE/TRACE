# ============================================================
# scenario_engine.py 
# ============================================================
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import numpy as np
import copy
from types import SimpleNamespace
from collections import Counter
import hashlib

from lstm_single_bridge import LSTMSingleBridge
from collections import deque
from dataclasses import dataclass, field
from typing import Any


def cfg_get(ctx: Dict[str, Any], key: str, default=None):
    val = (ctx.get("config") or {}).get(key, default)

    if val is None:
        return None

    try:
        return float(val)
    except Exception:
        return default

def _normalize_event_features(event):

    feat = (
        event.get("features")
        or {}
    )

    feat.setdefault(
        "min_dist",
        float("inf")
    )

    feat.setdefault(
        "mean_dist",
        float("inf")
    )

    feat.setdefault(
        "max_closing_rate",
        0.0
    )

    feat.setdefault(
        "mean_closing_rate",
        0.0
    )

    feat.setdefault(
        "max_lateral_rate",
        0.0
    )

    feat.setdefault(
        "lateral_variation",
        0.0
    )

    feat.setdefault(
        "duration",
        0.0
    )

    event["features"] = feat

    return event

def build_path_descriptor(path, meta):

    lanes = set()
    overlaps = set()
    junctions = set()

    for rid in path:

        entry = meta.get(str(rid), {})

        if not isinstance(entry, dict):
            continue

        if entry.get("type") == "overlap":
            overlaps.add(rid)

            for m in entry.get("members", []) or []:
                try:
                    lanes.add(int(m))
                except Exception:
                    pass
        else:
            lanes.add(rid)

        # direct junction reference
        if entry.get("junction_id") is not None:
            junctions.add(str(entry["junction_id"]))

        # shared junction relationships observed in metadata
        for field in (
            "shares_upcoming_junction_with",
            "shares_prev_junction_with"
        ):
            for j in entry.get(field, []) or []:
                junctions.add(str(j))

    return {
        "lanes": lanes,
        "overlaps": overlaps,
        "junctions": junctions
    }

def get_reachable(start_nodes, graph, max_depth=3):

    visited = set()
    queue = deque((n, 0) for n in start_nodes)

    while queue:

        node, depth = queue.popleft()

        if node in visited:
            continue

        visited.add(node)

        if depth >= max_depth:
            continue

        for nxt in graph.get(node, ()):
            queue.append((nxt, depth + 1))

    return visited

@dataclass(slots=True)
class EventRecord:

    scenario_id: str
    event_id: int
    event_type: str
    start_t: float
    end_t: float
    actors: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class EventFingerprint:
    event_type: str
    duration_norm: float
    actor_count: int
    context_tags: tuple[str, ...]

class SafeJSONEncoder(json.JSONEncoder):
    """Handles inf, nan, numpy/pandas types safely for JSON export."""
    def default(self, obj):
        if isinstance(obj, float) and math.isinf(obj):
            return "Infinity"
        if isinstance(obj, float) and math.isnan(obj):
            return "NaN"
        if hasattr(obj, 'tolist'):  # numpy/pandas compatibility
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# ------------- Graph helpers -------------

def _succ_set(meta: Dict[str, Any], rid: int) -> Set[int]:
    e = meta.get(str(rid), {}) if meta else {}
    return {int(s) for s in (e.get("successors", []) or []) if isinstance(s, (int, str)) and str(s).isdigit()}


def _get_adjacent(meta: Dict[str, Any], rid: int) -> Set[int]:
    try:
        e = meta.get(str(rid), {})
        if not isinstance(e, dict):
            return set()
        if e.get("type") not in ("lane", "connecting_lane"):
            return set()
        out = set()
        for a in e.get("adjacent", []) or []:
            try:
                aid = int(a)
            except Exception:
                continue
            de = meta.get(str(aid), {})
            if isinstance(de, dict) and de.get("type") in ("lane", "connecting_lane"):
                out.add(aid)
        return out
    except Exception:
        return set()


def _extract_window_features(ctx, t_start, t_end, actor=None):

    cache = ctx.setdefault("_runtime", {}).setdefault("feature_cache", {})
    key = f"{actor}|{round(t_start,3)}|{round(t_end,3)}"

    if key in cache:
        return cache[key]

    engine = ctx.get("derived", {}).get("engine", {})
    pair_ts = engine.get("pair_time_series", [])

    window = [
        p
        for p in pair_ts
        if isinstance(p.get("t"), (int, float))
        and t_start <= p["t"] <= t_end
    ]

    if not window:
        feat = {
            "min_dist": float("inf"),
            "mean_dist": float("inf"),
            "max_closing_rate": 0.0,
            "mean_closing_rate": 0.0,
            "max_lateral_rate": 0.0,
            "lateral_variation": 0.0,
            "duration": max(
                0.0,
                t_end - t_start
            ),
            "min_distance": float("inf"),
            "mean_distance": float("inf")
        }
        cache[key] = feat
        return feat

    dists = [p["dist"] for p in window if isinstance(p.get("dist"), (int, float))]
    closings = [p["closing_rate"] for p in window if isinstance(p.get("closing_rate"), (int, float))]
    lat_rates = [p["lateral_rate"] for p in window if isinstance(p.get("lateral_rate"), (int, float))]

    feat = {
        "min_dist":min(dists) if dists else None,
        "mean_dist":
            float(np.mean(dists))
            if dists else None,
        "max_closing_rate":
            max(closings)
            if closings else None,
        "mean_closing_rate":
            float(np.mean(closings))
            if closings else None,
        "max_lateral_rate":
            max(abs(x) for x in lat_rates)
            if lat_rates else None,
        "lateral_variation":
            float(np.std(lat_rates))
            if lat_rates else None,
        "duration":
            t_end - t_start,
        "min_distance":
            min(dists) if dists else None,
        "mean_distance":
            float(np.mean(dists))
            if dists else None,
    }

    cache[key] = feat
    return feat



# ============================================================
# TOPOLOGY RESOLVER (ENGINE-ONLY SEMANTICS)
# ============================================================

def enrich_with_topology(ctx):
    ctx.setdefault(
        "route_semantics",
        {}
    )
    ctx.setdefault(
        "interaction_topology",
        {}
    )
    return ctx



def build_reachability_graph(meta: Dict[str, Any]) -> Dict[int, Set[int]]:
    """
    Pure successor graph (NO adjacency).
    """
    graph: Dict[int, Set[int]] = {}

    if not meta:
        return graph

    for rid_str, entry in meta.items():
        try:
            rid = int(rid_str)
        except Exception:
            continue

        if not isinstance(entry, dict):
            continue

        if entry.get("type") not in ("lane", "connecting_lane"):
            continue

        succ = set()
        for s in entry.get("successors", []) or []:
            try:
                succ.add(int(s))
            except Exception:
                continue

        graph[rid] = succ

    return graph


def is_valid_topology_transition(
    from_set: Set[int],
    to_set: Set[int],
    meta: Dict[str, Any],
    reach: Dict[int, Set[int]]
) -> bool:
    """
    Checks if movement between regions is valid:
    - successor flow OR
    - adjacent (lane-change)
    """
    for a in from_set:
        succ = reach.get(a, set())
        adj = _get_adjacent(meta, a)

        for b in to_set:
            if b in succ:
                return True

            if b in adj:
                return True

            a_meta = meta.get(str(a), {})
            b_meta = meta.get(str(b), {})

            if (
                a_meta.get("junction_id") is not None
                and
                a_meta.get("junction_id")
                ==
                b_meta.get("junction_id")
            ):
                return True

    return False


def compute_headway_sd(v1_sd, v2_sd):
    gaps = [
        abs(p2.get("s_global", p2.get("s"))-p1.get("s_global", p1.get("s")))
        for p1, p2 in zip(v1_sd, v2_sd)
        if (
            p1.get("s_global", p1.get("s")) is not None
            and
            p2.get("s_global", p2.get("s")) is not None
        )
    ]
    if not gaps:
        return None, None
    return min(gaps), float(np.mean(gaps))



# ============================================================
# geometry & sampling
# ============================================================

def extract_region_sequence(vehicle):

    tl = vehicle.get("timeline", {})

    regions = [
        (
            frame.get("region_id")
            or frame.get("laneId")
            or frame.get("resolved_lane_id")
        )
        for _, frame in sorted(
            tl.items(),
            key=lambda x: int(x[0])
        )
    ]

    regions = [r for r in regions if r is not None]

    sequence = []

    for r in regions:
        if not sequence or sequence[-1] != r:
            sequence.append(r)

    return {
        "full": regions,
        "sequence": sequence,
        "start": sequence[0] if sequence else None,
        "end": sequence[-1] if sequence else None
    }

def compute_kinematics(sd_seq):

    if len(sd_seq) < 2:
        return {
            "speeds": [],
            "avg_speed": 0.0,
            "max_speed": 0.0,
            "distance": 0.0
        }

    speeds = []
    distance = 0.0

    MAX_REASONABLE_SPEED = 25.0   # 90 km/h

    for i in range(1, len(sd_seq)):

        p0 = sd_seq[i - 1]
        p1 = sd_seq[i]

        dt = max(p1["t"] - p0["t"], 1e-3)

        s0 = p0.get("s_global", p0["s"])
        s1 = p1.get("s_global", p1["s"])

        ds = s1 - s0

        v = ds / dt

        if abs(v) > MAX_REASONABLE_SPEED:
            continue

        speeds.append(v)
        distance += abs(ds)

    return {
        "speeds": speeds,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "max_speed": float(max(speeds)) if speeds else 0.0,
        "distance": float(distance)
    }


def compute_pairwise_time_series(v1_sd, v2_sd):

    out = []

    for p1, p2 in zip(v1_sd, v2_sd):

        if (
            p1.get("x") is not None
            and p1.get("y") is not None
            and p2.get("x") is not None
            and p2.get("y") is not None
        ):
            dist = math.hypot(
                p2["x"] - p1["x"],
                p2["y"] - p1["y"]
            )
        else:
            dist = abs(
                p2.get("s_global", p2["s"])
                -
                p1.get("s_global", p1["s"])
            )

        entry = {
            "t": p1["t"],
            "ds":
                p2.get("s_global", p2["s"])
                -
                p1.get("s_global", p1["s"]),
            "dd":
                p2["d"] - p1["d"],
            "dist": dist,
            "lat_dist": abs(p2["d"] - p1["d"])
        }

        if out:

            dt = p1["t"] - out[-1]["t"]

            if dt > 0:
                prev = out[-1]

                entry["closing_rate"] = (
                    prev["dist"] - entry["dist"]
                ) / dt

                entry["lateral_rate"] = (
                    entry["dd"] - prev["dd"]
                ) / dt
            else:
                entry["closing_rate"] = 0.0
                entry["lateral_rate"] = 0.0

        else:
            entry["closing_rate"] = 0.0
            entry["lateral_rate"] = 0.0

        out.append(entry)

    return out


# ============================================================
# PATH RELATION
# ============================================================


def _is_non_interactive_relation(rel):

    return rel in {
        "independent",
        "unknown"
    }

def sanitize_ttc(ttc):

    if ttc is None:
        return None

    if not math.isfinite(ttc):
        return None

    if abs(ttc) > 60:
        return None

    return float(ttc)



# ============================================================
# RULE MODELS
# ============================================================
@dataclass
class RuleEntry:
    type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    actor: str = "both"

@dataclass
class ScenarioRules:
    scenario_name: str
    duration_seconds: float
    vehicle1: Dict[str, Any]
    vehicle2: Dict[str, Any]
    derived_rules: List[RuleEntry] = field(default_factory=list)
    notes: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        return {
            "scenario_name": self.scenario_name,
            "duration_seconds": self.duration_seconds,
            "vehicles": {
                "vehicle1": self.vehicle1,
                "vehicle2": self.vehicle2,
            },
            "derived_rules": [
                {
                    "type": r.type,
                    "parameters": r.parameters,
                    "confidence": r.confidence,
                    "actor": r.actor,
                }
                for r in self.derived_rules
            ],
            "events": self.events,
            "notes": self.notes,
        }

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, cls=SafeJSONEncoder)

# ============================================================
# templates & DSL
# ============================================================

def compute_speeds(sd):

    return [
        (
            p1.get("s_global", p1["s"])
            -
            p0.get("s_global", p0["s"])
        )
        /
        max(p1["t"] - p0["t"], 1e-3)

        for p0, p1 in zip(sd[:-1], sd[1:])
    ]


def load_templates(path="scenario_gorules.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def extract_sd_sequences(ctx: dict):
    actors = ctx.get(
        "actors",
        {}
    )

    vehicles = {

        "V1":
            actors.get(
                "A",
                {}
            ),

        "V2":
            actors.get(
                "B",
                {}
            )
    }
    out = {}

    for vid in ("V1", "V2"):
        tl = vehicles.get(vid, {}).get("timeline", {})

        seq = []
        keys = sorted(tl, key=int)

        for k in keys:
            frame = tl[k]
            sd = frame.get("sd") or {}

            s = sd.get("s")
            d = sd.get("d")

            if s is None or d is None:
                continue

            lane_id = (
                frame.get("lane_id")
                or frame.get("laneId")
                or frame.get("resolved_lane_id")
                or frame.get("region_id")
            )

            t_val = frame.get("t")

            if not isinstance(t_val, (int, float)):

                default_dt = cfg_get(
                    ctx,
                    "default_frame_dt",
                    0.1
                )

                t_val = (
                    len(seq)
                    * default_dt
                )

            s_global = frame.get("s_global")

            try:
                s_global = float(
                    s if s_global is None else s_global
                )
            except (TypeError, ValueError):
                s_global = float(s)

            seq.append({
                "s": float(s),
                "s_global": s_global,
                "d": float(d),
                "t": float(t_val),
                "lane_id": lane_id,
                "region_id": frame.get("region_id"),
                "x": frame.get("world", {}).get("x"),
                "y": frame.get("world", {}).get("y")
            })

        out[vid] = sorted(seq, key=lambda x: x["t"])

    return out

def build_waiting_events(ctx: dict):

    events = []

    wait_speed = cfg_get(
        ctx,
        "waiting_speed_threshold_mps",
        1.0
    )

    min_wait = cfg_get(
        ctx,
        "min_wait_duration_s",
        2.0
    )

    engine = (
        ctx.get("derived", {})
           .get("engine", {})
    )

    actors = {
        "vehicle1": engine.get("v1_sd", []),
        "vehicle2": engine.get("v2_sd", [])
    }

    for actor_id, sd in actors.items():

        if len(sd) < 2:
            continue

        start_idx = None

        for i in range(1, len(sd)):

            p0 = sd[i - 1]
            p1 = sd[i]

            dt = max(
                p1["t"] - p0["t"],
                1e-3
            )

            speed = abs(
                (
                    p1.get("s_global", p1["s"])
                    -
                    p0.get("s_global", p0["s"])
                ) / dt
            )

            is_waiting = (
                speed <= wait_speed
            )

            if is_waiting:

                if start_idx is None:
                    start_idx = i - 1

            else:

                if start_idx is not None:

                    t_start = sd[start_idx]["t"]
                    t_end = sd[i - 1]["t"]

                    duration = t_end - t_start

                    if duration >= min_wait:

                        events.append(
                            {
                                "type": "waiting",
                                "participants": [actor_id],
                                "t_start": float(t_start),
                                "t_end": float(t_end),
                                "metrics": {},
                                "features": {
                                    "duration": duration,
                                    "waiting_duration_s": duration,
                                    "stopped_duration_s": duration,
                                    "min_dist": float("inf"),
                                    "mean_dist": float("inf"),
                                    "max_closing_rate": 0.0,
                                    "mean_closing_rate": 0.0,
                                    "max_lateral_rate": 0.0
                                }
                            }
                        )

                    start_idx = None

        if start_idx is not None:

            t_start = sd[start_idx]["t"]
            t_end = sd[-1]["t"]

            duration = t_end - t_start

            if duration >= min_wait:

                events.append(
                    {
                        "type": "waiting",
                        "participants": [actor_id],
                        "t_start": float(t_start),
                        "t_end": float(t_end),
                        "metrics": {},
                        "features": {
                            "duration": duration,
                            "waiting_duration_s": duration,
                            "stopped_duration_s": duration,
                            "min_dist": float("inf"),
                            "mean_dist": float("inf"),
                            "max_closing_rate": 0.0,
                            "mean_closing_rate": 0.0,
                            "max_lateral_rate": 0.0                            
                        }
                    }
                )

    return events


def build_lane_change_events(ctx: dict, v_sd: list, regions: list, actor: str):

    events = []

    if not v_sd or not regions:
        return events

    N = min(len(v_sd), len(regions))
    v_sd = v_sd[:N]
    regions = regions[:N]

    meta = ctx.get("meta", {})

    current_start = None

    for i in range(1, N):
        r_prev = regions[i - 1]
        r_curr = regions[i]

        if r_prev is None or r_curr is None:
            continue

        is_lane_change = _is_lane_change_transition(r_prev, r_curr, meta)

        if is_lane_change:
            if current_start is None:
                current_start = i - 1
        else:
            if current_start is not None:
                start_idx = current_start
                end_idx = i  

                #  find first actual transition inside segment
                transition_idx = None

                for j in range(start_idx + 1, end_idx + 1):
                    if regions[j] != regions[j - 1]:
                        if _is_lane_change_transition(regions[j - 1], regions[j], meta):
                            transition_idx = j
                            break

                # fallback 
                if transition_idx is None:
                    continue

                from_lane = regions[transition_idx - 1]
                to_lane = regions[transition_idx]

                t_start = v_sd[start_idx]["t"]
                t_end = v_sd[end_idx]["t"]

                features = _extract_window_features(ctx, t_start, t_end)

                events.append({
                    "type": "lane_change",
                    "actor": actor,
                    "t_start": t_start,
                    "t_end": t_end,
                    "region": from_lane,
                    "metrics": {
                        "from_lane": from_lane,
                        "to_lane": to_lane
                    },
                    "features": features
                })

                current_start = None
    #  finalize last segment 
    if current_start is not None:
        start_idx = current_start
        end_idx = N - 1
        transition_idx = None
        for j in range(start_idx + 1, end_idx + 1):
            if regions[j] != regions[j-1]:
                if _is_lane_change_transition(regions[j-1], regions[j], meta):
                    transition_idx = j
                    break

        if transition_idx is None:
            return events

        from_lane = regions[transition_idx - 1]
        to_lane = regions[transition_idx]

        t_start = v_sd[start_idx]["t"]
        t_end = v_sd[end_idx]["t"]
        features = _extract_window_features(ctx, t_start, t_end)

        events.append({
            "type": "lane_change",
            "actor": actor,
            "t_start": t_start,
            "t_end": t_end,
            "region": from_lane,
            "metrics": {
                "from_lane": from_lane,
                "to_lane": to_lane
            },
            "features": features
        })

    return events


# ============================================================
# CONTEXT BUILDER
# ============================================================

def enrich_with_signal_metrics(ctx: dict) -> dict:
    if not isinstance(ctx, dict):
        return {}

    ctx.setdefault("derived", {})

    signals = ctx.setdefault("derived", {}).setdefault("signals", {})
    engine = ctx.setdefault("derived", {}).setdefault("engine", {})

    actors = ctx.get(
        "actors",
        {}
    )

    vehicles = {

        "V1":
            actors.get("A", {}),

        "V2":
            actors.get("B", {})
    }

    v1 = vehicles.get("V1", {})
    v2 = vehicles.get("V2", {})

    # -------------------------
    # SD extraction
    # -------------------------
    sd = extract_sd_sequences(ctx)
    v1_sd, v2_sd = sd.get("V1", []), sd.get("V2", [])

    N = min(len(v1_sd), len(v2_sd))
    v1_sd, v2_sd = v1_sd[:N], v2_sd[:N]

    engine["v1_sd"], engine["v2_sd"] = v1_sd, v2_sd

    # -------------------------
    # kinematics
    # -------------------------
    engine["v1_kinematics"] = compute_kinematics(v1_sd)
    engine["v2_kinematics"] = compute_kinematics(v2_sd)

    signals["v1_avg_speed"] = engine["v1_kinematics"]["avg_speed"]
    signals["v2_avg_speed"] = engine["v2_kinematics"]["avg_speed"]
    signals["v1_max_speed"] = engine["v1_kinematics"]["max_speed"]
    signals["v2_max_speed"] = engine["v2_kinematics"]["max_speed"]
    signals["v1_distance"] = engine["v1_kinematics"]["distance"]
    signals["v2_distance"] = engine["v2_kinematics"]["distance"]

    # -------------------------
    # pairwise time series (CORE)
    # -------------------------
    pair_ts = compute_pairwise_time_series(v1_sd, v2_sd)
    engine["pair_time_series"] = pair_ts

    if pair_ts:
        dists = [p["dist"] for p in pair_ts]
        signals["min_distance"] = min(dists)
        signals["max_distance"] = max(dists)

        close_rates = [p["closing_rate"] for p in pair_ts]
        signals["avg_closing_rate"] = float(np.mean(close_rates))

    else:
        signals["min_distance"] = float("inf")
        signals["avg_closing_rate"] = 0.0

    # -------------------------
    # lane change signals (NO decisions)
    # -------------------------
    def extract_regions(vehicle):
        tl = vehicle.get("timeline", {})
        return [
            f.get("region_id")
            for k, f in sorted(tl.items(), key=lambda x: int(x[0]))
        ]

    v1_regions = extract_regions(v1)
    v2_regions = extract_regions(v2)

    # --- compute independently ---
    lc_events_v1 = build_lane_change_events(ctx, v1_sd, v1_regions, "vehicle1")
    lc_events_v2 = build_lane_change_events(ctx, v2_sd, v2_regions, "vehicle2")

    # --- assign cleanly ---
    engine["lane_change_events"] = lc_events_v1 + lc_events_v2


    #  derive counts from events (SINGLE SOURCE OF TRUTH)

    ctx.setdefault(
        "derived",
        {}
    )

    engine["lane_change_events"] = (
        lc_events_v1 +
        lc_events_v2
    )

    ctx.setdefault("derived", {})

    ctx["derived"]["v1_lane_change_count"] = len(lc_events_v1)
    ctx["derived"]["v2_lane_change_count"] = len(lc_events_v2)

    ctx["derived"]["v1_lane_change"] = (
        len(lc_events_v1) > 0
    )

    ctx["derived"]["v2_lane_change"] = (
        len(lc_events_v2) > 0
    )

    return ctx

# ============================================================
# PROBABILISTIC BEHAVIOR FEATURES
# ============================================================
def _compute_behavioral_probabilities(ctx: dict):

    engine = (
        ctx.get("derived", {})
           .get("engine", {})
    )

    pair_ts = engine.get(
        "pair_time_series",
        []
    )

    if len(pair_ts) < 2:
        return {}

    distances = [
        p["dist"]
        for p in pair_ts
        if isinstance(
            p.get("dist"),
            (int, float)
        )
    ]

    closing_rates = [
        p["closing_rate"]
        for p in pair_ts
        if isinstance(
            p.get("closing_rate"),
            (int, float)
        )
    ]

    lateral_rates = [
        abs(p["lateral_rate"])
        for p in pair_ts
        if isinstance(
            p.get("lateral_rate"),
            (int, float)
        )
    ]

    ds = [
        p["ds"]
        for p in pair_ts
        if isinstance(
            p.get("ds"),
            (int, float)
        )
    ]

    sign_flip_count = sum(
        1
        for i in range(1, len(ds))
        if ds[i-1] * ds[i] < 0
    )

    return {
        "min_distance_raw":
            min(distances)
            if distances else None,

        "mean_closing_rate_raw":
            float(np.mean(closing_rates))
            if closing_rates else 0.0,

        "max_lateral_rate_raw":
            max(lateral_rates)
            if lateral_rates else 0.0,

        "sign_flip_count":
            sign_flip_count
    }

def _compute_instability_signals(ctx: dict) -> dict:

    engine = ctx.get("derived", {}).get("engine", {})
    signals = {}

    def process(sd, prefix):
        if len(sd) < 5:
            return

        # LATERAL
        d_vals = [p["d"] for p in sd if p.get("d") is not None]
        if d_vals:
            mean_d = sum(d_vals) / len(d_vals)
            dev = [abs(d - mean_d) for d in d_vals]
            if dev:
                signals[f"{prefix}_lat_instability"] = float(max(dev))

        # LONGITUDINAL
        speeds = compute_speeds(sd)
        if len(speeds) > 3:
            acc = [(speeds[i] - speeds[i-1]) for i in range(1, len(speeds))]
            if acc:
                signals[f"{prefix}_lon_instability"] = float(max(abs(a) for a in acc))

    process(engine.get("v1_sd", []), "v1")
    process(engine.get("v2_sd", []), "v2")

    return signals

def _resolve_conflicts(rules: List[RuleEntry], config: dict) -> List[RuleEntry]:
    if not config:
        return rules

    conflict_map = config.get("conflict_resolution", {})
    if not conflict_map:
        return rules

    # collect all rule subtypes safely
    types = [
        (r.parameters or {}).get("subtype")
        for r in rules
        if r.parameters
    ]

    filtered = []

    for r in rules:
        params = r.parameters or {}
        rtype = params.get("subtype")

        # no subtype → keep safely
        if not rtype:
            filtered.append(r)
            continue

        conflicts = conflict_map.get(str(rtype).upper(), [])

        # normalize once for speed + correctness
        type_set = {str(t).lower() for t in types if t}

        if any(str(c).lower() in type_set for c in conflicts):
            continue

        filtered.append(r)

    return filtered


def build_empty_engine_context(
        map_ctx):

    return {

        "kind":
            "engine_context",

        "version":
            "1.0",

        "source_kind":
            map_ctx.get(
                "kind"
            ),

        "map_context":
            map_ctx,

        "facts": {},

        "rule_inputs": {},

        "events": [],

        "behaviors": [],

        "insights": []
    }

def build_engine_insight_context(
    v1=None,
    v2=None,
    duration: float = 0.0,
    *,
    context=None,
    **kwargs
):

    if not isinstance(context, dict):
        raise ValueError(
            "MapScenarioContext required"
        )

    if context.get("kind") != (
        "map_scenario_context"
    ):
        raise ValueError(
            "Expected map_scenario_context"
        )

    if not context.get("ready"):
        raise ValueError(
            "Context not ready"
        )

    ctx = copy.deepcopy(context)

    actors = ctx.get(
        "actors",
        {}
    )

    if "A" not in actors:
        raise ValueError(
            "Actor A missing"
        )

    if "B" not in actors:
        raise ValueError(
            "Actor B missing"
        )

    ctx.setdefault("derived", {})
    ctx = enrich_with_signal_metrics(ctx)
    rule_inputs = _derive_rule_inputs(ctx)
    ctx["derived"]["rule_inputs"] = rule_inputs
    ctx.update(rule_inputs)

    return ctx
# ============================================================
# Events 
# ============================================================


def detect_events(ctx, events=None):
    events = list(events) if events else []
    e = ctx.get("derived", {}).get("engine", {})

    lat_thr = cfg_get(ctx, "proximity_lat_dist", None)
    dist_thr = cfg_get(ctx, "proximity_dist", float("inf"))
    enter_dist = cfg_get(
        ctx,
        "proximity_enter_dist",
        dist_thr
    )

    exit_dist = cfg_get(
        ctx,
        "proximity_exit_dist",
        enter_dist + 0.5
    )    

    min_event_duration = cfg_get(
        ctx,
        "min_event_duration",
        0.0
    )

    # -------------------------
    # Lane change
    # -------------------------
    for lc in e.get("lane_change_events", []):
        t0 = lc.get("t_start")
        t1 = lc.get("t_end", t0)

        if not isinstance(t0, (int, float)):
            continue
        if not isinstance(t1, (int, float)):
            t1 = t0

        features = _extract_window_features(ctx, t0, t1)
        interaction = _derive_interaction_features(ctx, t0, t1)
        events.append({
            "type": "lane_change",
            "participants": [lc.get("actor")],
            "t_start": float(t0),
            "t_end": float(t1),
            "region": lc.get("region"),
            "metrics": lc.get("metrics", {}),
            "features": features,
            "interaction": interaction
        })

    # --------------------------------------------------
    # MERGED PROXIMITY EVENTS (segment-based)
    # --------------------------------------------------
    pair_ts = e.get("pair_time_series", [])

    active_start = None
    active_points = []

    inside_event = False

    for p in pair_ts:

        d = p.get("dist")
        lat_d = p.get("lat_dist")

        currently_active = inside_event

        if not isinstance(d, (int, float)):
            currently_active = False

        elif (
            lat_thr is not None
            and isinstance(lat_d, (int, float))
            and lat_d > lat_thr
        ):
            currently_active = False

        else:

            if not inside_event:

                if d <= enter_dist:
                    currently_active = True

            else:

                if d > exit_dist:
                    currently_active = False

        # ------------------------------------
        # segment closed
        # ------------------------------------
        if inside_event and not currently_active:

            t0 = active_start
            t1 = active_points[-1]["t"]

            features = _extract_window_features(
                ctx,
                t0,
                t1
            )

            interaction = _derive_interaction_features(
                ctx,
                t0,
                t1
            )

            if (t1 - t0) >= min_event_duration:

                events.append({
                    "type": "proximity",
                    "participants": [
                        "vehicle1",
                        "vehicle2"
                    ],
                    "t_start": float(t0),
                    "t_end": float(t1),
                    "metrics": {},
                    "features": features,
                    "interaction": interaction
                })

            active_start = None
            active_points = []

        # ------------------------------------
        # segment started
        # ------------------------------------
        if not inside_event and currently_active:

            active_start = p["t"]
            active_points = []

        if currently_active:
            active_points.append(p)

        inside_event = currently_active


    # close last segment
    if active_start is not None and active_points:
        t0 = active_start
        t1 = active_points[-1]["t"]

        features = _extract_window_features(ctx, t0, t1)
        interaction = _derive_interaction_features(ctx, t0, t1)
        if (t1 - t0) >= min_event_duration:
            events.append({
                "type": "proximity",
                "participants": ["vehicle1", "vehicle2"],
                "t_start": float(t0),
                "t_end": float(t1),
                "metrics": {},
                "features": features,
                "interaction": interaction
            })
    waiting_events = build_waiting_events(ctx)

    if waiting_events:
        events.extend(waiting_events)


    return events




# ============================================================
# EVENTS
# ============================================================


def _build_event_timeline(ctx):

    timeline = []
    for e in ctx.get("events", []):
        t0 = e.get("t_start")
        if t0 is None:
            t0 = e.get("start_t")
        t1 = e.get("t_end")
        if t1 is None:
            t1 = e.get("end_t")
        if t1 is None:
            t1 = t0
        if t0 is None:
            continue
        timeline.append({
            **e,
            "t_start": float(t0),
            "t_end": float(t1)
        })
    return timeline

# ============================================================
# GRAPH-BASED TEMPORAL SEQUENCE DETECTOR
# ============================================================

class EventGraph:
    """
    Builds a temporal graph of events.

    Nodes = events
    Edges = temporal relations
    """

    def __init__(self, events: list[dict]):
        self.events = sorted(events, key=lambda e: e.get("t_start", 0))
        self.graph = {i: [] for i in range(len(self.events))}
        self._build_graph()

    def _build_graph(self):
        for i, e1 in enumerate(self.events):
            for j in range(i + 1, len(self.events)):
                e2 = self.events[j]

                if self._is_before(e1, e2):
                    self.graph[i].append(("before", j))

                if self._is_overlap(e1, e2):
                    self.graph[i].append(("overlaps", j))

    @staticmethod
    def _is_before(e1, e2):
        t1 = e1.get("t_end")
        t2 = e2.get("t_start")
        return (
            isinstance(t1, (int, float)) and
            isinstance(t2, (int, float)) and
            t1 <= t2
        )

    @staticmethod
    def _is_overlap(e1, e2):
        a0, a1 = e1.get("t_start"), e1.get("t_end")
        b0, b1 = e2.get("t_start"), e2.get("t_end")

        if None in (a0, a1, b0, b1):
            return False

        return not (a1 < b0 or b1 < a0)


# ------------------------------------------------------------
# MATCHER
# ------------------------------------------------------------

class TemporalGraphMatcher:

    def __init__(self, ctx: dict):
        self.ctx = ctx
        self.events = ctx.get("event_timeline", [])
        self.graph = EventGraph(self.events)

    def match(self, rule: dict) -> list:
        """
        rule format:

        {
          "name": "cutin_sequence",
          "steps": [
            {"name": "A", "type": "lane_change"},
            {"name": "B", "type": "proximity"},
            {"name": "C", "type": "avoidance"}
          ],
          "relations": [
            {"from": "A", "to": "B", "type": "before", "within": 2.0},
            {"from": "B", "to": "C", "type": "before", "within": 2.0}
          ],
          "produce": {
            "type": "unsafe_sequence",
            "actor": "both"
          }
        }
        """

        steps = rule.get("steps", [])
        relations = rule.get("relations", [])

        if not steps:
            return []

        # --------------------------------------------------
        # 1. candidate filtering (by type)
        # --------------------------------------------------
        candidates = {}

        for s in steps:
            name = s["name"]
            etype = s.get("type")

            candidates[name] = [
                (idx, e)
                for idx, e in enumerate(self.events)
                if e.get("type") == etype
            ]

            if not candidates[name]:
                return []

        # --------------------------------------------------
        # 2. DFS search
        # --------------------------------------------------
        results = []
        step_names = [s["name"] for s in steps]

        def dfs(depth, current_match):
            if depth == len(step_names):
                results.append(dict(current_match))
                return

            step = step_names[depth]

            for idx, event in candidates[step]:

                if not self._valid_transition(current_match, step, idx, rule):
                    continue

                current_match[step] = idx
                dfs(depth + 1, current_match)
                del current_match[step]

        dfs(0, {})

        # --------------------------------------------------
        # 3. convert to RuleEntry
        # --------------------------------------------------
        out = []

        for match in results:
            events_seq = [self.events[i] for i in match.values()]

            t0 = min(e["t_start"] for e in events_seq if e.get("t_start") is not None)
            t1 = max(e["t_end"] for e in events_seq if e.get("t_end") is not None)

            subtype = rule.get("produce", {}).get("type", "sequence")

            out.append(
                RuleEntry(
                    type="sequence_event",
                    actor=rule.get("produce", {}).get("actor", "both"),
                    confidence=1.0,
                    parameters={
                        "subtype": subtype,
                        "event_type": "sequence",
                        "start_t": t0,
                        "end_t": t1,
                        "steps": list(match.keys())
                    }
                )
            )

        return out

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------
    def _valid_transition(self, current, new_step, new_idx, rule):

        for rel in rule.get("relations", []):

            src = rel["from"]
            dst = rel["to"]

            if dst != new_step:
                continue

            if src not in current:
                continue

            i1 = current[src]
            i2 = new_idx

            e1 = self.events[i1]
            e2 = self.events[i2]

            if rel["type"] == "before":
                if not self._check_before(e1, e2, rel.get("within")):
                    return False

            if rel["type"] == "overlaps":
                if not EventGraph._is_overlap(e1, e2):
                    return False

        return True

    @staticmethod
    def _check_before(e1, e2, within):

        t1 = e1.get("t_end")
        t2 = e2.get("t_start")

        if not isinstance(t1, (int, float)) or not isinstance(t2, (int, float)):
            return False

        if t1 > t2:
            return False

        if within is None:
            return True

        return (t2 - t1) <= float(within)



# ============================================================
# GENERIC RULE ENGINE
# ============================================================



def _to_namespace(obj):

    if isinstance(obj, dict):
        return SimpleNamespace(
            **{
                str(k): _to_namespace(v)
                for k, v in obj.items()
            }
        )

    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]

    return obj

def _normalize_expr(expr: str) -> str:
    return (
        expr.replace("&&", " and ")
            .replace("||", " or ")
            .replace(" true", " True")
            .replace(" false", " False")
            .replace("== true", "== True")
            .replace("== false", "== False")
    )

def eval_value(expr: str, ctx: Dict[str, Any]):
    if not expr:
        return None

    expr = _normalize_expr(expr)

    try:
        # only allow string keys
        safe_ctx = {
            str(k): _to_namespace(v)
            for k, v in ctx.items()
            if isinstance(k, str)
        }

        locals_dict = dict(safe_ctx)

        locals_dict["min"] = min
        locals_dict["max"] = max
        locals_dict["abs"] = abs
        locals_dict["math"] = math

        return eval(
            expr,
            {"__builtins__": {}},
            locals_dict
        )

    except AttributeError:
        return 0.0    

    except Exception as e:
        print("\n[EVAL_VALUE ERROR]")
        print("expr =", expr)
        print("error =", repr(e))

        try:
            print("ctx keys =", list(ctx.keys()))
        except Exception:
            pass

        return None

def eval_expr(expr: str, ctx: Dict[str, Any]) -> bool:
    if not expr:
        return False

    expr = _normalize_expr(expr)

    try:
        # only allow string keys
        safe_ctx = {
            str(k): _to_namespace(v)
            for k, v in ctx.items()
            if isinstance(k, str)
        }

        locals_dict = dict(safe_ctx)

        locals_dict["min"] = min
        locals_dict["max"] = max
        locals_dict["abs"] = abs
        locals_dict["math"] = math


        bad_keys = [
            (k, type(k))
            for k in ctx.keys()
            if not isinstance(k, str)
        ]

        if bad_keys:
            print("[BAD CTX KEYS]", bad_keys)


        result = eval(
            expr,
            {"__builtins__": {}},
            locals_dict
        )

        return bool(result)
    
    except AttributeError:
        return False

    except Exception as e:
        print("\n[EVAL_EXPR ERROR]")
        print("expr =", expr)
        print("error =", repr(e))

        try:
            print("ctx keys =", list(ctx.keys()))
        except Exception:
            pass

        return False


# ------------------------------------------------------------
# Resolve dynamic values
# ------------------------------------------------------------
def resolve_value(val, ctx):
    if isinstance(val, str):
        if val in ctx:
            return ctx[val]
        return val
    return val


# ------------------------------------------------------------
# GoRules Engine Executor
# ------------------------------------------------------------
class GoRulesEngine:

    def __init__(self, model: Dict[str, Any]):
        self.nodes = model.get("decision", {}).get("nodes", [])

    def evaluate(self, ctx: Dict[str, Any]) -> List:
        results = []

        events = ctx.get("event_timeline", [])

        for event in events:

            actor_id = event.get("participants", ["both"])[0]

            if actor_id == "vehicle1":
                lane_shift_count = (
                    ctx.get(
                        "derived",
                        {}
                    ).get(
                        "v1_lane_change_count",
                        0
                    )
                )
                actor_path_complexity = ctx.get("v1_path_complexity", 0)
                actor_region_count = ctx.get(
                    "v1_path_region_count",
                    0
                )
                actor_connector_transitions = ctx.get(
                    "v1_connector_transitions",
                    0
                )

            elif actor_id == "vehicle2":
                lane_shift_count = (
                    ctx.get(
                        "derived",
                        {}
                    ).get(
                        "v2_lane_change_count",
                        0
                    )
                )
                actor_path_complexity = ctx.get("v2_path_complexity", 0)
                actor_region_count = ctx.get(
                    "v2_path_region_count",
                    0
                )
                actor_connector_transitions = ctx.get(
                    "v2_connector_transitions",
                    0
                )

            else:
                lane_shift_count = 0
                actor_path_complexity = 0
                actor_region_count = 0
                actor_connector_transitions = 0

            local_ctx = {
                **ctx,
                "event": event,
                "actor": {
                    "id": actor_id,
                    "lane_shift_count": lane_shift_count,
                    "path_complexity": actor_path_complexity,
                    "path_region_count": actor_region_count,
                    "connector_transitions": actor_connector_transitions
                },
                **event.get("features", {}),
                **(event.get("interaction") or {})
            }

            # ---------------------------------
            # PASS 1 : expression nodes
            # ---------------------------------
            for node in self.nodes:

                if node.get("type") != "expression":
                    continue

                expr_id = node.get("id")
                expr = node.get("expr")

                if expr_id:
                    result = eval_value(expr, local_ctx)
                    if result is None:
                        result = 0.0
                    local_ctx[expr_id] = result


            # ---------------------------------
            # PASS 2 : decision nodes
            # ---------------------------------
            for node in self.nodes:

                if node.get("type") != "decision":
                    continue

                for cond in node.get("conditions", []):
                    condition = cond.get("if")
                    result = eval_expr(condition, local_ctx)
                    if not result:
                        continue

                    out = cond.get("then", {})

                    conf = (
                        cond.get("confidence")
                        or out.get("confidence")
                        or out.get("interaction", {}).get("confidence")
                        or out.get("behavior", {}).get("confidence")
                    )

                    if isinstance(conf, str):
                        conf = local_ctx.get(conf, conf)

                    try:
                        conf = float(conf)
                    except Exception:
                        conf = 1.0

                    subtype = (
                        out.get("tag")
                        or out.get("interaction", {}).get("type")
                        or out.get("behavior", {}).get("type")
                        or node.get("id")
                    )

                    results.append(
                        RuleEntry(
                            type="gorules_event",
                            actor=actor_id,
                            confidence=conf,
                            parameters={
                                "subtype": subtype,
                                "timeline": out.get("timeline"),
                                "interaction": out.get("interaction"),
                                "behavior": out.get("behavior"),
                                "event_type": event.get("type"),
                                "start_t": event.get("t_start"),
                                "end_t": event.get("t_end"),
                                "region": event.get("region")
                            }
                        )
                    )

        return results

def _derive_rule_inputs(ctx: dict) -> dict:
    """
    Pure feature extraction for GoRules.

    No thresholds.
    No rule decisions.
    """

    meta = ctx.get("meta") or {}
    vehicles = ctx.get("vehicles") or {}
    engine = ctx.get("derived", {}).get("engine", {})

    # --------------------------------------------------
    # Pair timeline metrics
    # --------------------------------------------------
    pair_tl = (
        ctx.get("pairwise", {}).get("timeline")
        or {}
    )

    ttc_values = []
    if isinstance(pair_tl, dict):
        for frame in pair_tl.values():
            ttc = frame.get("ttc_s")
            if not isinstance(
                ttc,
                (int, float)
            ):
                continue
            if not math.isfinite(ttc):
                continue
            if ttc <= 0:
                continue
            if ttc > 60:
                continue
            ttc_values.append(
                float(ttc)
            )

    interaction_topology = (
        ctx.get(
            "interaction_topology",
            {}
        )
    )

    pair_tl = (
        ctx.get(
            "interaction_facts",
            {}
        ).get(
            "timeline",
            {}
        )
    )

    shared_junction = any(
        (
            f.get(
                "lane_relation",
                {}
            ).get(
                "shared_junction",
                False
            )
        )
        for f in pair_tl.values()
    )

    out = {

        "shared_junction":
            shared_junction,
        "shared_lane_fraction":
            float(
                interaction_topology.get(
                    "shared_lane_ratio",
                    0.0
                )
            ),

        "common_merge_count":
            len(
                interaction_topology.get(
                    "common_merge_points",
                    []
                )
            ),

        "common_diverge_count":
            len(
                interaction_topology.get(
                    "common_diverge_points",
                    []
                )
            ),

        "min_ttc":
            min(ttc_values)
            if ttc_values
            else float("inf")
    }


    # --------------------------------------------------
    # Actor metrics
    # --------------------------------------------------
    def route_semantics_connector_count(transitions):

        return sum(
            1 for t in transitions
            if t == "TRANSFER"
        )
    def vehicle_features(actor):

        actor_obj = (
            ctx.get(
                "actors",
                {}
            )
            .get(actor, {})
        )

        route = (
            actor_obj
            .get(
                "route_semantics",
                {}
            )
            .get(
                "route",
                []
            )
        )

        transitions = (
            actor_obj
            .get(
                "route_semantics",
                {}
            )
            .get(
                "transitions",
                []
            )
        )

        topo = (
            actor_obj
            .get(
                "topology_facts",
                {}
            )
        )

        return {

            "path_complexity":
                len(route),

            "connector_transitions":
                route_semantics_connector_count(
                    actor_obj.get(
                        "route_semantics",
                        {}
                    ).get(
                        "transitions",
                        []
                    )
                ),
            "path_region_count":
                len(route),

            "junction_count":
                len(
                    topo.get(
                        "junctions",
                        []
                    )
                ),

            "merge_count":
                len(
                    topo.get(
                        "merge_points",
                        []
                    )
                ),

            "diverge_count":
                len(
                    topo.get(
                        "diverge_points",
                        []
                    )
                )
        }

    for actor, prefix in (("A","v1"), ("B","v2")):
        vf = vehicle_features(actor)

        out[f"{prefix}_path_complexity"] = vf["path_complexity"]
        out[f"{prefix}_connector_transitions"] = vf["connector_transitions"]
        out[f"{prefix}_path_region_count"] = vf["path_region_count"]
        out[f"{prefix}_merge_count"] = (vf["merge_count"])
        out[f"{prefix}_diverge_count"] = (vf["diverge_count"])        

    # --------------------------------------------------
    # Same lane fraction
    # --------------------------------------------------
    out["shared_lane_fraction"] = (
        interaction_topology.get(
            "shared_lane_ratio",
            0.0
        )
    )

    # --------------------------------------------------
    # Distance evolution
    # --------------------------------------------------
    pair_ts = engine.get("pair_time_series", [])

    distance_trend = (
        pair_ts[-1]["dist"] - pair_ts[0]["dist"]
        if len(pair_ts) >= 2
        else 0.0
    )

    out["distance_trend"] = distance_trend
    out["separation_after_event"] = max(distance_trend, 0.0)
    interaction = (
        _derive_interaction_features(
            ctx
        )
    )

    out.update({

        "interaction_sign_flip":
            interaction.get(
                "interaction_sign_flip",
                0
            ),

        "interaction_lateral_cross":
            interaction.get(
                "interaction_lateral_cross",
                0
            ),

        "interaction_convergence":
            interaction.get(
                "interaction_convergence",
                0.0
            ),

        "waiting_duration_s":
            interaction.get(
                "waiting_duration_s",
                0.0
            ),

        "distance_min":
            ctx.get(
                "derived",
                {}
            )
            .get(
                "signals",
                {}
            )
            .get(
                "min_distance",
                float("inf")
            )
    })
    return out


def _derive_interaction_features(
    ctx: dict,
    t_start: float | None = None,
    t_end: float | None = None,
):
    """
    Pure interaction feature extraction.

    NO semantic interpretation.
    NO conflict classification.
    NO waiting classification.

    GoRules owns interpretation.
    """

    engine = ctx.get("derived", {}).get("engine", {})
    pair_ts = engine.get("pair_time_series", [])

    if (
        isinstance(t_start, (int, float))
        and isinstance(t_end, (int, float))
    ):
        pair_ts = [
            p
            for p in pair_ts
            if t_start <= p.get("t", -1) <= t_end
        ]

    result = {
        "interaction_sign_flip": 0,
        "interaction_lateral_cross": 0,
        "interaction_convergence": 0.0,
        "interaction_deceleration": 0.0,
        "waiting_duration_s": 0.0,
        "stopped_duration_s": 0.0,
    }

    current_event = ctx.get("event", {})

    if current_event.get("type") == "waiting":

        feat = current_event.get(
            "features",
            {}
        )

        result["waiting_duration_s"] = (
            feat.get(
                "waiting_duration_s",
                feat.get("duration", 0.0)
            )
        )

        result["stopped_duration_s"] = (
            feat.get(
                "stopped_duration_s",
                result["waiting_duration_s"]
            )
        )

    if len(pair_ts) < 2:
        return result

    ds = [
        p["ds"]
        for p in pair_ts
        if isinstance(p.get("ds"), (int, float))
    ]

    dd = [
        p["dd"]
        for p in pair_ts
        if isinstance(p.get("dd"), (int, float))
    ]

    dist = [
        p["dist"]
        for p in pair_ts
        if isinstance(p.get("dist"), (int, float))
    ]

    closing = [
        p["closing_rate"]
        for p in pair_ts
        if isinstance(
            p.get("closing_rate"),
            (int, float)
        )
    ]

    sign_flip = sum(
        1
        for i in range(1, len(ds))
        if ds[i - 1] * ds[i] < 0
    )

    lateral_cross_threshold = cfg_get(
        ctx,
        "lateral_cross_threshold",
        0.2
    )

    lateral_cross = sum(
        1
        for i in range(1, len(dd))
        if (
            abs(dd[i] - dd[i - 1])
            > lateral_cross_threshold
            and dd[i - 1] * dd[i] <= 0
        )
    )

    convergence_ratio = 0.0

    if len(dist) >= 2:

        convergence_steps = sum(
            1
            for i in range(1, len(dist))
            if dist[i] < dist[i - 1]
        )

        convergence_ratio = (
            convergence_steps
            / max(len(dist) - 1, 1)
        )

    result.update(
        {
            "interaction_sign_flip": sign_flip,
            "interaction_lateral_cross": lateral_cross,
            "interaction_convergence": convergence_ratio,
            "interaction_deceleration":
                float(np.mean(closing))
                if closing
                else 0.0,
        }
    )

    current_event = ctx.get("event", {})

    if current_event.get("type") == "waiting":

        feat = current_event.get(
            "features",
            {}
        )

        result["waiting_duration_s"] = (
            feat.get(
                "waiting_duration_s",
                feat.get("duration", 0.0)
            )
        )

        result["stopped_duration_s"] = (
            feat.get(
                "stopped_duration_s",
                feat.get(
                    "stopped_duration",
                    0.0
                )
            )
        )

    return result



# ============================================================
# MAIN ENTRY
# ============================================================

def suggest_rules_combined(
    *,
    duration: float = 0.0,
    meta=None,
    templates_path="scenario_gorules.json",
    context: dict | None = None,
) -> List[RuleEntry]:

    # ----------------------------------------------------------
    # 1. VALIDATION
    # ----------------------------------------------------------
    if not context or not isinstance(context, dict) or not context.get("ready"):
        raise ValueError("Context not ready for rule evaluation")

    templates = load_templates(templates_path)

    ctx = copy.deepcopy(context)
    ctx["config"] = templates.get("config", {})

    ctx.setdefault("vehicles", {})
    ctx.setdefault("pairwise", {})
    ctx.setdefault("derived", {})

    ctx["duration_seconds"] = float(duration) if duration > 0 else 0.0

    if meta is not None:
        ctx["meta"] = meta

    # ----------------------------------------------------------
    # 2. TOPOLOGY ENRICHMENT
    # ----------------------------------------------------------
    ctx = enrich_with_signal_metrics(ctx)
    engine = ctx.get("derived", {}).get("engine", {})

    print(
        "[DEBUG]",
        "pair_ts=",
        len(engine.get("pair_time_series", [])),
        "lane_changes=",
        len(engine.get("lane_change_events", []))
    )    
    try:
        ctx = enrich_with_topology(ctx)
    except Exception:
        pass

    # ----------------------------------------------------------
    # 3. SIGNAL + FEATURE BUILD (UNCHANGED CORE)
    # ----------------------------------------------------------


    # -------------------------
    # Interaction features
    # -------------------------
    rule_inputs = _derive_rule_inputs(ctx)
    ctx.setdefault("derived", {})["rule_inputs"] = rule_inputs
    ctx.update(rule_inputs)

    interaction = _derive_interaction_features(ctx)
    ctx.setdefault("derived",{})
    ctx["derived"]["interaction_features"] = interaction

    ctx.update(interaction)
    # -------------------------
    # Behavioral probabilities
    # -------------------------
    behavior_features = _compute_behavioral_probabilities(ctx)
    ctx.setdefault(
        "behavior",
        {}
    ).update(behavior_features)
    ctx.update(behavior_features)
 
    # ----------------------------------------------------------
    # 4. EVENT GENERATION (timeline-first)
    # ----------------------------------------------------------
    ctx["events"] = detect_events(ctx)

    ctx["engine_event_fingerprint"] = {
        "event_count":
            len(ctx["events"]),

        "event_types":
            [
                e.get("type")
                for e in ctx["events"]
            ],

        "lane_change_count":
            sum(
                1 for e in ctx["events"]
                if e.get("type") == "lane_change"
            ),

        "waiting_count":
            sum(
                1 for e in ctx["events"]
                if e.get("type") == "waiting"
            ),

        "proximity_count":
            sum(
                1 for e in ctx["events"]
                if e.get("type") == "proximity"
            )
    }

    print(
        "[DEBUG EVENTS]",
        len(ctx["events"])
    )    
    ctx["event_timeline"] = _build_event_timeline(ctx)
    ctx["_canonical_event_timeline"] = copy.deepcopy(ctx["event_timeline"])
    ctx["semantic_event_timeline"] = copy.deepcopy(ctx["event_timeline"])

    # ----------------------------------------------------------
    # 5. FLATTEN EVENTS FOR GORULES  CRITICAL
    # ----------------------------------------------------------
    def _flatten_event(ev):
        return {
            **ev,
            "event": ev,
            "features": ev.get("features", {}) or {},
            "interaction": ev.get("interaction", {}) or {}
        }

    ctx["event_timeline"] = [
        _flatten_event(
            _normalize_event_features(e)
        )
        for e in ctx["event_timeline"]
    ]

    # ----------------------------------------------------------
    # 6. SIGNAL EXPOSURE
    # ----------------------------------------------------------
    signals = ctx.get("derived", {}).get("signals", {})
    engine = ctx.get("derived", {}).get("engine", {})

    ctx.update(signals)
    ctx["pair_time_series"] = engine.get("pair_time_series", [])
    ctx["lane_change_events"] = engine.get("lane_change_events", [])

    # ----------------------------------------------------------
    # 7. INSTABILITY SIGNALS
    # ----------------------------------------------------------
    ctx["derived"]["signals"].update(_compute_instability_signals(ctx))
    ctx.update(ctx["derived"]["signals"])

    # ----------------------------------------------------------
    # 9. EXECUTE GORULES ENGINE
    # ----------------------------------------------------------
    try:
        gorules_engine = GoRulesEngine(templates)
        final = gorules_engine.evaluate(ctx)
    except Exception as e:
        print("[WARN] GoRules engine failed:", e)
        final = []

    # ----------------------------------------------------------
    # TIMELINE
    # ----------------------------------------------------------


    try:
        matcher = TemporalGraphMatcher(ctx)

        sequence_rules = []
        for r in templates.get("sequence_rules", []):
            sequence_rules.extend(matcher.match(r))

        final = final + sequence_rules
    except Exception as e:
        print("[WARN] Temporal matcher failed:", e)


    # ----------------------------------------------------------
    # 10. DEDUP (UNCHANGED CONTRACT)
    # ----------------------------------------------------------
    def _make_hashable(x):
        if isinstance(x, dict):
            return tuple(sorted((k, _make_hashable(v)) for k, v in x.items()))
        if isinstance(x, list):
            return tuple(_make_hashable(v) for v in x)
        if isinstance(x, set):
            return tuple(sorted(_make_hashable(v) for v in x))
        return x

    seen = set()
    unique = []

    for r in final:
        params_hash = _make_hashable(r.parameters or {})
        key = (r.type, r.actor, params_hash)

        if key not in seen:
            seen.add(key)
            unique.append(r)

    # ----------------------------------------------------------
    # 11. CONFLICT RESOLUTION
    # ----------------------------------------------------------
    unique = _resolve_conflicts(unique, templates)

    descriptor = build_scenario_descriptor(
        ctx,
        unique
    )

    ctx.setdefault(
        "derived",
        {}
    )["scenario_descriptor"] = descriptor

    # ----------------------------------------------------------
    # propagate everything required by caller
    # ----------------------------------------------------------
    try:

        context.setdefault(
            "derived",
            {}
        )["scenario_descriptor"] = copy.deepcopy(descriptor)

        context["events"] = copy.deepcopy(
            ctx.get("events", [])
        )

        context["event_timeline"] = copy.deepcopy(
            ctx.get("event_timeline", [])
        )

        context["_canonical_event_timeline"] = copy.deepcopy(
            ctx.get("_canonical_event_timeline", [])
        )

        context["semantic_event_timeline"] = copy.deepcopy(
            ctx.get("semantic_event_timeline", [])
        )

        context["interaction_features"] = copy.deepcopy(
                ctx.get(
                    "derived",
                    {}
                ).get(
                    "interaction_features",
                    {}
                )
            )
        context["behavior"] = copy.deepcopy(
            ctx.get("behavior", {})
        )

        context.setdefault(
            "derived",
            {}
        )["signals"] = copy.deepcopy(
            ctx.get("derived", {}).get("signals", {})
        )

    except Exception as e:
        print(
            "[WARN] Context propagation failed:",
            e
        )

    # ----------------------------------------------------------
    # 12. CLEANUP
    # ----------------------------------------------------------
    ctx.pop("_runtime", None)
    ctx.pop("_feature_cache", None)

    return unique

def expand_region_to_lanes(
        region_id,
        meta):

    if region_id is None:
        return set()

    try:
        rid = int(region_id)
    except Exception:
        return set()

    entry = meta.get(
        str(rid),
        {}
    )

    if not isinstance(entry, dict):
        return {rid}

    rtype = entry.get("type")

    #
    # overlap region
    #
    if rtype == "overlap":

        out = set()

        for m in (
            entry.get("members", [])
            or []
        ):
            try:
                out.add(int(m))
            except Exception:
                pass

        return out or {rid}

    #
    # normal lane region
    #
    if rtype in (
        "lane",
        "connecting_lane"
    ):
        return {rid}

    #
    # fallback
    #
    return {rid}

def _is_lane_change_transition(
    r_from,
    r_to,
    meta
):

    if r_from == r_to:
        return False

    from_lanes = expand_region_to_lanes(
        r_from,
        meta
    )

    to_lanes = expand_region_to_lanes(
        r_to,
        meta
    )

    for u in from_lanes:

        succ = _succ_set(meta, u)
        adj = _get_adjacent(meta, u)

        for v in to_lanes:

            if u == v:
                continue

            #
            # Topological continuation
            #
            if v in succ:
                return False

            #
            # Connector continuation
            #
            mu = meta.get(str(u), {})
            mv = meta.get(str(v), {})

            if (
                mu.get("junction_id") is not None
                and
                mu.get("junction_id")
                ==
                mv.get("junction_id")
            ):
                return False

            #
            # True lateral move
            #
            if v in adj:
                return True

    return False

# ============================================================
# EMBEDDING GENERATION
# ============================================================

EMBEDDING_DIM = 128


def _stable_hash_index(text: str, dim: int = EMBEDDING_DIM):
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h, 16) % dim


def _add_token(vec, token, weight=1.0):
    idx = _stable_hash_index(str(token))
    vec[idx] += float(weight)


def build_scenario_embedding(
    timeline,
    rules,
    topology,
    numeric_features,
    search_index
):
    """
    Deterministic embedding.

    No training required.

    Preserves:
        - event types
        - transitions
        - motifs
        - rule types
        - topology
        - numeric features
    """

    vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # --------------------------------------------------
    # event types
    # --------------------------------------------------

    for evt in timeline:
        etype = evt.get("type")
        if etype:
            _add_token(vec, f"EV:{etype}", 1.0)

    # --------------------------------------------------
    # rule types
    # --------------------------------------------------

    for r in rules:
        subtype = (
            r.parameters or {}
        ).get("subtype")

        if subtype:
            _add_token(vec, f"RULE:{subtype}", 1.5)

    # --------------------------------------------------
    # transitions
    # --------------------------------------------------

    for t in search_index.get("transitions", []):
        _add_token(vec, f"TR:{t}", 2.0)

    # --------------------------------------------------
    # motifs
    # --------------------------------------------------

    for m in search_index.get("motifs", []):
        _add_token(vec, f"MOTIF:{m}", 3.0)

    # --------------------------------------------------
    # topology
    # --------------------------------------------------

    for k, v in topology.items():
        if isinstance(v, bool):
            if v:
                _add_token(vec, f"TOPO:{k}")

        elif v is not None:
            _add_token(vec, f"TOPO:{k}:{v}")

    # --------------------------------------------------
    # numeric features
    # --------------------------------------------------

    numeric_keys = [
        "event_count",
        "rule_count",
        "lane_change_count",
        "waiting_count",
        "proximity_count",
        "min_distance",
        "avg_closing_rate",
        "same_lane_fraction",
    ]

    for key in numeric_keys:
        val = numeric_features.get(key)

        if (
            isinstance(val, (int, float))
            and np.isfinite(val)
        ):
            idx = _stable_hash_index(f"NUM:{key}")
            vec[idx] += float(val)

    # --------------------------------------------------
    # normalize
    # --------------------------------------------------

    norm = np.linalg.norm(vec)

    if norm > 1e-8:
        vec = vec / norm

    return vec.tolist()


def embedding_similarity(v1, v2):
    try:
        a = np.asarray(v1, dtype=np.float32)
        b = np.asarray(v2, dtype=np.float32)

        da = np.linalg.norm(a)
        db = np.linalg.norm(b)

        if da < 1e-8 or db < 1e-8:
            return 0.0

        return float(np.dot(a, b) / (da * db))

    except Exception:
        return 0.0


def build_scenario_descriptor(
    ctx: dict,
    rules: list[RuleEntry]
):
    """
    Retrieval-first scenario descriptor.

    Event semantics are primary.
    Rules, topology and kinematics are secondary.

    Optimized for:
        - semantic retrieval
        - motif matching
        - similarity search
        - future vector indexing
    """

    timeline = sorted(
        ctx.get(
            "_canonical_event_timeline",
            ctx.get("event_timeline", [])
        ),
        key=lambda e: e.get("t_start", 0.0)
    )

    nodes = []
    edges = []

    event_types = []
    rule_types = []

    # ==================================================
    # EVENT NODES
    # ==================================================

    for idx, evt in enumerate(timeline):

        node_id = f"E{idx}"

        t0 = evt.get("t_start")
        t1 = evt.get("t_end")

        duration = None

        if (
            isinstance(t0, (int, float))
            and isinstance(t1, (int, float))
        ):
            duration = float(max(0.0, t1 - t0))

        features = evt.get("features", {}) or {}
        interaction = evt.get("interaction", {}) or {}

        etype = evt.get("type")

        if etype:
            event_types.append(str(etype))

        nodes.append(
            {
                "id": node_id,
                "node_type": "event",
                "event_type": etype,
                "participants": evt.get(
                    "participants", []
                ),
                "t_start": t0,
                "t_end": t1,
                "duration_s": duration,
                "region": evt.get("region"),
                "metrics": {
                    **features,
                    **interaction
                }
            }
        )

    # ==================================================
    # TEMPORAL GRAPH
    # ==================================================

    for i in range(len(timeline) - 1):

        e1 = timeline[i]
        e2 = timeline[i + 1]

        a0 = e1.get("t_start")
        a1 = e1.get("t_end")
        b0 = e2.get("t_start")
        b1 = e2.get("t_end")

        if None in (a0, a1, b0, b1):
            continue

        edge = {
            "from": f"E{i}",
            "to": f"E{i+1}"
        }

        if a1 <= b0:

            edge.update(
                {
                    "relation": "before",
                    "gap_s": float(b0 - a1)
                }
            )

        else:

            overlap = min(a1, b1) - max(a0, b0)

            edge.update(
                {
                    "relation": "overlaps",
                    "overlap_s": float(
                        max(overlap, 0.0)
                    )
                }
            )

        edges.append(edge)

    # ==================================================
    # RULE NODES
    # ==================================================

    indexed_events = []

    for idx, evt in enumerate(timeline):

        s = evt.get("t_start")
        e = evt.get("t_end")

        if (
            isinstance(s, (int, float))
            and isinstance(e, (int, float))
        ):
            indexed_events.append(
                (s, e, idx)
            )

    indexed_events.sort()

    for idx, rule in enumerate(rules):

        params = rule.parameters or {}

        node_id = f"R{idx}"

        start_t = params.get("start_t")
        end_t = params.get("end_t")

        duration = None

        if (
            isinstance(start_t, (int, float))
            and isinstance(end_t, (int, float))
        ):
            duration = float(
                max(
                    0.0,
                    end_t - start_t
                )
            )

        subtype = params.get("subtype")

        if subtype:
            rule_types.append(str(subtype))

        nodes.append(
            {
                "id": node_id,
                "node_type": "rule",
                "rule_type": subtype,
                "actor": rule.actor,
                "confidence": float(
                    getattr(
                        rule,
                        "confidence",
                        1.0
                    )
                ),
                "t_start": start_t,
                "t_end": end_t,
                "duration_s": duration,
                "interaction": params.get(
                    "interaction"
                ),
                "behavior": params.get(
                    "behavior"
                )
            }
        )

        if not (
            isinstance(start_t, (int, float))
            and isinstance(end_t, (int, float))
        ):
            continue

        for evt_start, evt_end, e_idx in indexed_events:

            if evt_start > end_t:
                break

            if evt_end < start_t:
                continue

            edges.append(
                {
                    "from": f"E{e_idx}",
                    "to": node_id,
                    "relation": "supports"
                }
            )

    # ==================================================
    # EVENT ORDER
    # ==================================================

    ordered_event_types = [
        e.get("type")
        for e in timeline
        if e.get("type")
    ]

    signature = "→".join(
        ordered_event_types
    )

    id_to_node = {
        n["id"]: n
        for n in nodes
    }

    # ==================================================
    # TRANSITIONS
    # ==================================================

    transitions = []

    for a, b in zip(
        ordered_event_types[:-1],
        ordered_event_types[1:]
    ):
        transitions.append(
            f"{a}>{b}"
        )

    # ==================================================
    # EVENT -> RULE TRANSITIONS
    # ==================================================

    for edge in edges:

        if edge.get("relation") != "supports":
            continue

        src = id_to_node.get(
            edge["from"]
        )

        dst = id_to_node.get(
            edge["to"]
        )

        if not src or not dst:
            continue

        ev = src.get("event_type")
        rl = dst.get("rule_type")

        if ev and rl:

            transitions.append(
                f"{ev}>{rl}"
            )

    # ==================================================
    # MOTIFS
    # ==================================================

    motifs = []

    motif_map = {
        "lane_change>proximity":
            "LC_PROX",
        "waiting>merge":
            "WAIT_MERGE",
        "waiting>yield":
            "WAIT_YIELD",
        "lane_change>car_following":
            "LC_FOLLOW",
        "lane_change>dangerous_following":
            "LC_DANGER_FOLLOW",
        "proximity>near_miss_longitudinal":
            "PROX_NEAR_MISS",
        "proximity>near_miss_lateral":
            "PROX_NEAR_MISS",
        "proximity>avoidance":
            "PROX_AVOIDANCE"
    }

    for t in transitions:

        m = motif_map.get(t)

        if m:
            motifs.append(m)

    # ==================================================
    # ENGINE EVENT FINGERPRINT
    # ==================================================

    fingerprint = {
        "event_chain":
            ordered_event_types,

        "event_count":
            len(ordered_event_types),

        "lane_change_count":
            ordered_event_types.count(
                "lane_change"
            ),

        "proximity_count":
            ordered_event_types.count(
                "proximity"
            ),

        "waiting_count":
            ordered_event_types.count(
                "waiting"
            )
    }

    # ==================================================
    # TOPOLOGY
    # ==================================================

    it = ctx.get(
        "interaction_topology",
        {}
    )

    topology = {

        "shared_lane_fraction":
            it.get(
                "shared_lane_ratio",
                0.0
            ),

        "shared_junction_count":
            len(
                it.get(
                    "shared_junctions",
                    []
                )
            ),

        "common_merge_count":
            len(
                it.get(
                    "common_merge_points",
                    []
                )
            ),

        "common_diverge_count":
            len(
                it.get(
                    "common_diverge_points",
                    []
                )
            ),

        "shared_route_segments":
            len(
                it.get(
                    "shared_route_segments",
                    []
                )
            )
    }

    # ==================================================
    # NUMERIC FEATURES
    # ==================================================

    signals = (
        ctx.get("derived", {})
           .get("signals", {})
    )

    lane_change_count = (

        ctx.get(
            "derived",
            {}
        ).get(
            "v1_lane_change_count",
            0
        )

        +

        ctx.get(
            "derived",
            {}
        ).get(
            "v2_lane_change_count",
            0
        )
    )

    numeric_features = {

        "duration_s":
            ctx.get(
                "duration_seconds",
                0.0
            ),

        "event_count":
            len(timeline),

        "rule_count":
            len(rules),

        "lane_change_count":
            lane_change_count,

        "waiting_count":
            fingerprint[
                "waiting_count"
            ],

        "proximity_count":
            fingerprint[
                "proximity_count"
            ],

        "v1_avg_speed":
            signals.get(
                "v1_avg_speed"
            ),

        "v2_avg_speed":
            signals.get(
                "v2_avg_speed"
            ),

        "min_distance":
            signals.get(
                "min_distance"
            ),

        "max_distance":
            signals.get(
                "max_distance"
            ),

        "avg_closing_rate":
            signals.get(
                "avg_closing_rate"
            ),

        "min_ttc":
            ctx.get("min_ttc")
    }

    # ==================================================
    # SEMANTIC SIGNATURE
    # ==================================================

    semantic_signature = {

        "event_chain":
            tuple(
                ordered_event_types
            ),

        "event_types":
            sorted(
                set(event_types)
            ),

        "rule_types":
            sorted(
                set(rule_types)
            ),

        "event_bigrams":
            sorted(
                set(transitions)
            ),

        "motifs":
            sorted(
                set(motifs)
            ),

        "fingerprint":
            fingerprint
    }

    # ==================================================
    # WINDOW
    # ==================================================

    window_info = {

        "scenario_id":
            ctx.get(
                "scenario_id",
                "unknown_scenario"
            ),

        "window_id":
            ctx.get(
                "window_id",
                "full_window"
            ),

        "t_start":
            timeline[0].get("t_start")
            if timeline else None,

        "t_end":
            timeline[-1].get("t_end")
            if timeline else None
    }

    # ==================================================
    # DESCRIPTOR
    # ==================================================

    descriptor = {

        "descriptor_version":
            "graph_v5_semantic",

        "signature":
            signature,

        "event_fingerprint":
            fingerprint,

        "window":
            window_info,

        "topology":
            topology,

        "graph": {
            "node_count":
                len(nodes),

            "edge_count":
                len(edges),

            "nodes":
                nodes,

            "edges":
                edges
        },

        "search_index": {

            "event_types":
                semantic_signature[
                    "event_types"
                ],

            "rule_types":
                semantic_signature[
                    "rule_types"
                ],

            "transitions":
                semantic_signature[
                    "event_bigrams"
                ],

            "motifs":
                semantic_signature[
                    "motifs"
                ],

            "numeric_features":
                numeric_features,

            "semantic_signature":
                semantic_signature
        }
    }

    descriptor["embedding"] = (
        build_scenario_embedding(
            timeline=timeline,
            rules=rules,
            topology=topology,
            numeric_features=numeric_features,
            search_index=descriptor[
                "search_index"
            ]
        )
    )

    descriptor["embedding_version"] = (
        "semantic_v2"
    )

    return descriptor

def build_event_sequence(events):

    seq = []

    total_duration = max(
        (e.end_t for e in events),
        default=0.0
    )

    total_duration = max(
        total_duration,
        1e-6
    )

    for e in events:

        seq.append(
            EventFingerprint(
                event_type=e.event_type,

                duration_norm=(
                    (e.end_t - e.start_t)
                    / total_duration
                ),

                actor_count=len(e.actors),

                context_tags=tuple(
                    sorted(
                        e.context.get(
                            "tags",
                            []
                        )
                    )
                )
            )
        )

    return seq

def canonicalize_event_roles(events):
    """
    Make vehicle1/vehicle2 role-independent.

    Example:

        vehicle1 waiting
        vehicle2 lane_change

    and

        vehicle2 waiting
        vehicle1 lane_change

    become identical.

    Returns:
        normalized_events
    """

    if not events:
        return []

    actor_map = {}

    next_actor = 0

    def map_actor(actor):

        nonlocal next_actor

        if actor not in actor_map:

            actor_map[actor] = (
                f"actor{chr(65 + next_actor)}"
            )

            next_actor += 1

        return actor_map[actor]

    normalized = []

    for ev in events:

        parts = list(
            ev.get(
                "participants",
                []
            )
        )

        mapped = sorted(
            map_actor(p)
            for p in parts
        )

        clone = dict(ev)

        clone["participants"] = mapped

        normalized.append(clone)

    return normalized

def build_searchable_event_sequence(events):

    if not events:
        return [], [], {}

    events = canonicalize_event_roles(events)

    def _finite_float(v, default=0.0):
        try:
            v = float(v)

            if not math.isfinite(v):
                return default

            return v

        except Exception:
            return default

    def _event_name(ev):

        value = ev.get("event")

        #
        # normalized timeline
        #

        if isinstance(value, str):
            return value

        #
        # flattened GoRules timeline
        #

        if isinstance(value, dict):

            return str(
                value.get(
                    "type",
                    value.get(
                        "event",
                        "unknown"
                    )
                )
            )

        #
        # raw detector event
        #

        value = ev.get("type")

        if isinstance(value, str):
            return value

        return "unknown"

    prepared = []

    for idx, e in enumerate(events):

        participants = tuple(
            sorted(
                e.get(
                    "participants",
                    []
                )
            )
        )

        actor_tag = (
            ",".join(participants)
            if participants
            else "none"
        )

        event_name = _event_name(e)

        prepared.append({

            "_idx": idx,

            "_start":
                _finite_float(
                    e.get(
                        "start_t",
                        e.get(
                            "t_start",
                            0.0
                        )
                    )
                ),

            "event":
                event_name,

            "event_token":
                f"{actor_tag}:{event_name}",

            "participants":
                list(participants),

            "start_ratio":
                _finite_float(
                    e.get(
                        "start_ratio",
                        0.0
                    )
                ),

            "end_ratio":
                _finite_float(
                    e.get(
                        "end_ratio",
                        0.0
                    )
                ),

            "duration_ratio":
                _finite_float(
                    e.get(
                        "duration_ratio",
                        0.0
                    )
                ),

            "region":
                e.get("region"),

            "from_region":
                e.get("from_region"),

            "to_region":
                e.get("to_region")
        })

    prepared.sort(
        key=lambda x: (
            x["_start"],
            x["_idx"]
        )
    )

    sequence = []
    transitions = []
    stats = {}

    prev = None

    for rec in prepared:

        sequence.append(rec)

        stats.setdefault(
            rec["event"],
            {
                "count": 0
            }
        )

        stats[
            rec["event"]
        ]["count"] += 1

        if prev:

            transitions.append({

                "src":
                    prev["event"],

                "dst":
                    rec["event"],

                "delta_ratio":
                    max(
                        0.0,
                        rec["start_ratio"]
                        -
                        prev["end_ratio"]
                    )
            })

        prev = rec

    return (
        sequence,
        transitions,
        stats
    )