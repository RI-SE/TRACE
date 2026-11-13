# road_infrastructure.py

import json
import numpy as np
import ast
import matplotlib.pyplot as plt


# ============================================================
# Region
# ============================================================
class Region:
    def __init__(self, region_id, data):
        self.id = region_id
        self.type = data.get("type")
        self.attributes = data

        self.raw_polygon = data.get("polygon", [])
        self.centerline = data.get("centerline", None)

        self.polygon = self._parse_polygon(self.raw_polygon)

    # ---------------- Parsing ----------------
    def _parse_polygon(self, poly):
        pts = []
        for p in poly:
            try:
                if isinstance(p, str):
                    p = ast.literal_eval(p)

                pts.append([float(p[0]), float(p[1])])
            except Exception:
                continue  # skip bad points

        return np.array(pts) if pts else None

    # ---------------- Geometry ----------------
    def compute_centerline(self):
        """Midpoint-based centerline approximation."""
        pts = self.polygon
        if pts is None or len(pts) < 4:
            return None

        n = len(pts)
        half = n // 2

        left = pts[:half]
        right = pts[-half:][::-1]

        m = min(len(left), len(right))
        center = (left[:m] + right[:m]) / 2.0

        return center

    def ensure_centerline(self, overwrite=False):
        if self.centerline is not None and not overwrite:
            return

        cl = self.compute_centerline()
        if cl is not None:
            self.centerline = cl.tolist()
            self.attributes["centerline"] = self.centerline


# ============================================================
# RoadInfrastructure
# ============================================================
class RoadInfrastructure:
    def __init__(self, json_path=None):
        self.json_path = json_path
        self.regions = {}
        self._regions_int = {}

        # Optional integration with your GUI world
        self.mask = None
        self.meta = None
        self.H = None
        self.Hinv = None
        self._lookup_cache = {}
        self._cache_enabled = True        
        self._cl_world_cache = {}

        if json_path:
            self.load(json_path)


    def enable_cache(self, enabled: bool = True):
        self._cache_enabled = enabled

    def clear_cache(self):
        self._lookup_cache.clear()
        self._cl_world_cache.clear()

    # ---------------- Load / Save ----------------
    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)

        if "regions" in data:
            data = data["regions"]

        if not isinstance(data, dict):
            raise ValueError("Invalid road JSON format")

        self.regions.clear()
        self._regions_int.clear()

        for key, val in data.items():
            if not isinstance(val, dict):
                continue

            region = Region(key, val)
            self.regions[key] = region

            try:
                self._regions_int[int(key)] = region
            except Exception:
                pass

        if not self.regions:
            raise ValueError("No valid regions loaded")

    def save(self, path):
        out = {rid: r.attributes for rid, r in self.regions.items()}
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def set_homography(self, H, Hinv):
        """
        Attach homography matrices.

        Args:
            H: 3x3 (pixel -> world)
            Hinv: 3x3 (world -> pixel)
        """
        import numpy as np

        self.H = np.asarray(H, dtype=float)
        self.Hinv = np.asarray(Hinv, dtype=float)
        self.clear_cache()

    def get_centerline_world(self, region_id: int):
        if region_id in self._cl_world_cache:
            return self._cl_world_cache[region_id]

        cl_px = self.get_centerline(region_id)
        if cl_px is None or len(cl_px) < 2:
            self._cl_world_cache[region_id] = None
            return None

        cl_world = []
        for px, py in cl_px:
            wp = self.pixel_to_world(px, py)
            if wp:
                cl_world.append(wp)

        if len(cl_world) < 2:
            self._cl_world_cache[region_id] = None
            return None

        cl_world = np.asarray(cl_world, dtype=float)
        self._cl_world_cache[region_id] = cl_world
        return cl_world

    def project_pixel_to_frenet(self, region_id: int, x: float, y: float):
        """
        Pixel → world → Frenet (single call).
        """
        wp = self.pixel_to_world(x, y)
        if not wp:
            return None

        return self.project_world_to_frenet(region_id, wp[0], wp[1])

    def project_world_to_frenet(self, region_id: int, x: float, y: float):
        if region_id is None or self.H is None:
            return None

        # --- Cache key ---
        key = ("wf", region_id, round(x, 3), round(y, 3))
        if self._cache_enabled and key in self._lookup_cache:
            return self._lookup_cache[key]

        cl = self.get_centerline_world(region_id)
        if cl is None or len(cl) < 2:
            return None

        # --- Arc length along centerline ---
        seglens = np.linalg.norm(cl[1:] - cl[:-1], axis=1)
        s_arr = np.concatenate([[0.0], np.cumsum(seglens)])

        p = np.array([x, y], dtype=float)

        best = None
        best_dist = float("inf")

        # --- Project onto each segment ---
        for i in range(len(cl) - 1):
            a = cl[i]
            b = cl[i + 1]
            ab = b - a

            denom = np.dot(ab, ab)
            if denom < 1e-9:
                continue

            t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)

            proj = a + t * ab
            vec = p - proj
            dist = np.linalg.norm(vec)

            if dist < best_dist:
                # Signed lateral offset via 2D cross product
                cross = ab[0] * vec[1] - ab[1] * vec[0]
                sign = np.sign(cross)

                s_val = s_arr[i] + t * (s_arr[i + 1] - s_arr[i])

                best = {
                    "s": float(s_val),
                    "d": float(sign * dist),
                    "segment_index": i,
                }
                best_dist = dist

        if best is None or best_dist > 20.0:
            best = None

        # --- Cache store ---
        if self._cache_enabled:
            self._lookup_cache[key] = best

        return best

    def world_to_pixel(self, X: float, Y: float):
        if self.Hinv is None:
            return None

        try:
            px = self.Hinv[0, 0]*X + self.Hinv[0, 1]*Y + self.Hinv[0, 2]
            py = self.Hinv[1, 0]*X + self.Hinv[1, 1]*Y + self.Hinv[1, 2]
            w  = self.Hinv[2, 0]*X + self.Hinv[2, 1]*Y + self.Hinv[2, 2]

            if abs(w) < 1e-9:
                return None

            return (px / w, py / w)

        except Exception:
            return None

    def compute_centerline_arclength(self, cl: np.ndarray):
        if cl is None or len(cl) < 2:
            return None

        seglens = np.linalg.norm(cl[1:] - cl[:-1], axis=1)
        s = np.concatenate([[0.0], np.cumsum(seglens)])
        return s

    def project_to_frenet(self, region_id: int, x: float, y: float):
        """
        Returns:
            {
                "s": longitudinal position (meters along centerline)
                "d": lateral offset (meters, signed)
                "segment_index": int
            }
        """

        cl = self.get_centerline(region_id)
        if cl is None or len(cl) < 2:
            return None

        s_arr = self.compute_centerline_arclength(cl)
        if s_arr is None:
            return None

        p = np.array([x, y])

        best = None
        best_dist = float("inf")

        for i in range(len(cl) - 1):
            a = cl[i]
            b = cl[i + 1]
            ab = b - a

            denom = np.dot(ab, ab)
            if denom < 1e-9:
                continue

            t = np.dot(p - a, ab) / denom
            t = np.clip(t, 0.0, 1.0)

            proj = a + t * ab
            vec = p - proj
            dist = np.linalg.norm(vec)

            if dist < best_dist:
                # signed lateral offset (2D cross product sign)
                cross = ab[0]*vec[1] - ab[1]*vec[0]
                sign = np.sign(cross)

                s_val = s_arr[i] + t * (s_arr[i+1] - s_arr[i])

                best = {
                    "s": float(s_val),
                    "d": float(sign * dist),
                    "segment_index": i,
                }
                best_dist = dist

        return best

    def compute_frenet_velocity(self, prev_sd, curr_sd, dt: float):
        if not prev_sd or not curr_sd or dt <= 0:
            return None

        ds = (curr_sd["s"] - prev_sd["s"]) / dt
        dd = (curr_sd["d"] - prev_sd["d"]) / dt

        return {
            "ds": float(ds),   # longitudinal velocity
            "dd": float(dd),   # lateral velocity
        }

    def classify_motion_sd(self, sd, vel):
        if not sd or not vel:
            return {}

        ds = vel["ds"]
        dd = vel["dd"]

        return {
            "moving_forward": ds > 0.1,
            "moving_backward": ds < -0.1,

            "lateral_left": dd > 0.1,
            "lateral_right": dd < -0.1,

            "lane_centered": abs(sd["d"]) < 0.5,
            "off_center": abs(sd["d"]) >= 0.5,
        }

    def pixel_to_world(self, x: float, y: float):
        if self.H is None:
            return None

        try:
            wx = self.H[0, 0]*x + self.H[0, 1]*y + self.H[0, 2]
            wy = self.H[1, 0]*x + self.H[1, 1]*y + self.H[1, 2]
            w  = self.H[2, 0]*x + self.H[2, 1]*y + self.H[2, 2]

            if abs(w) < 1e-9:
                return None

            return (wx / w, wy / w)

        except Exception:
            return None

    def query_from_world(self, world_pt):
        if world_pt is None:
            return {
                "region_id": None,
                "road_id": None,
                "lane_id": None,
                "meta": {}
            }

        px = self.world_to_pixel(world_pt.x, world_pt.y)
        if px is None:
            return {
                "region_id": None,
                "road_id": None,
                "lane_id": None,
                "meta": {}
            }

        x, y = px

        return self.region_lookup(x, y)

    def _flatten_overlaps(self, path, meta):
        """
        Replace overlap regions in a path with their member lanes.
        """

        if not path or not meta:
            return path or []

        flat = []

        for rid in path:
            try:
                e = meta.get(str(rid), {})
            except Exception:
                flat.append(rid)
                continue

            # --- OVERLAP ---
            if isinstance(e, dict) and e.get("type") == "overlap":

                members = set()

                if "members" in e:
                    for m in e.get("members", []):
                        try:
                            members.add(int(m))
                        except Exception:
                            continue

                elif "lanes" in e:
                    for l in e.get("lanes", []):
                        try:
                            if "mask_id" in l:
                                members.add(int(l["mask_id"]))
                        except Exception:
                            continue

                for m in sorted(members):
                    if not flat or flat[-1] != m:
                        flat.append(m)

            # --- NORMAL ---
            else:
                if not flat or flat[-1] != rid:
                    flat.append(rid)

        return flat

    def resolve_region(self, region_id: int):
        if region_id is None:
            return None

        if self.meta is None:
            return region_id

        try:
            flat = self._flatten_overlaps([region_id], self.meta)
            return flat[0] if flat else region_id
        except Exception:
            return region_id

    # ---------------- Integration hooks ----------------
    def attach_mask_meta(self, mask=None, meta=None):
        if mask is None or meta is None:
            raise ValueError("Mask and meta must be provided")

        if not isinstance(mask, np.ndarray):
            raise TypeError("mask must be numpy array")

        if not isinstance(meta, dict):
            raise TypeError("meta must be dict")

        self.mask = mask
        self.meta = meta
        self.clear_cache()

    # ---------------- Query ----------------

    def get_region(self, region_id: int):
        return self._regions_int.get(region_id)

    def get_regions_by_type(self, region_type):
        return [r for r in self.regions.values() if r.type == region_type]

    # ---------------- Centerline generation ----------------
    def populate_centerlines(self, region_type=None, overwrite=False):
        """
        region_type:
            None → all regions
            str → single type
            list/set → multiple types
        """

        if isinstance(region_type, str):
            allowed = {region_type}
        elif isinstance(region_type, (list, set, tuple)):
            allowed = set(region_type)
        else:
            allowed = None  # all

        for region in self.regions.values():

            if allowed and region.type not in allowed:
                continue

            region.ensure_centerline(overwrite=overwrite)

    def get_centerline(self, region_id: int):
        reg = self.get_region(region_id)
        if not reg or not reg.centerline:
            return None
        return np.asarray(reg.centerline)

    def nearest_centerline_point(self, region_id: int, x: float, y: float):
        cl = self.get_centerline(region_id)
        if cl is None or len(cl) < 2:
            return None

        pts = cl
        dists = np.linalg.norm(pts - np.array([x, y]), axis=1)
        idx = np.argmin(dists)

        return {
            "point": pts[idx],
            "index": int(idx),
            "distance": float(dists[idx])
        }

    def project_onto_centerline(self, region_id: int, x: float, y: float):
        cl = self.get_centerline(region_id)
        if cl is None or len(cl) < 2:
            return None

        p = np.array([x, y])
        best = None
        best_dist = float("inf")

        for i in range(len(cl) - 1):
            a = cl[i]
            b = cl[i + 1]
            ab = b - a
            t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-9)
            t = np.clip(t, 0.0, 1.0)

            proj = a + t * ab
            dist = np.linalg.norm(p - proj)

            if dist < best_dist:
                best_dist = dist
                best = {
                    "point": proj,
                    "segment_index": i,
                    "t": float(t),
                    "distance": float(dist)
                }

        return best


    # ---------------- Plotting ----------------
    def plot(self, region_type="lane", show_centerline=True):
        plt.figure(figsize=(10, 10))

        for region in self.regions.values():

            if region_type and region.type != region_type:
                continue

            if region.polygon is None:
                continue

            pts = region.polygon

            # color logic aligned with GUI semantics
            if region.attributes.get("is_connecting_road", False):
                color = "orange"
            else:
                color = "blue"

            # polygon
            plt.fill(pts[:, 0], pts[:, 1],
                     alpha=0.3, color=color)

            # centerline
            if show_centerline and region.centerline:
                cl = np.array(region.centerline)
                plt.plot(
                    cl[:, 0], cl[:, 1],
                    linestyle="--", color="black", linewidth=1
                )

        plt.title("Road Infrastructure")
        plt.axis("equal")
        plt.grid(True)
        plt.show()

    def _expand_overlap_ids(self, ids):
        if not ids or self.meta is None:
            return ids or []

        expanded = []

        for rid in ids:
            try:
                m = self.meta.get(str(rid), {})
            except Exception:
                expanded.append(rid)
                continue

            # ✅ If overlap → expand
            if isinstance(m, dict) and m.get("type") == "overlap":
                members = []

                if "members" in m:
                    members = m["members"]

                elif "lanes" in m:  # fallback support
                    members = [l.get("mask_id") for l in m.get("lanes", [])]

                for mem in members:
                    try:
                        mem = int(mem)
                        if mem not in expanded:
                            expanded.append(mem)
                    except Exception:
                        continue
            else:
                if rid not in expanded:
                    expanded.append(rid)

        return expanded

    def region_lookup(self, x: int, y: int):
        if self.mask is None or self.meta is None:
            return {
                "region_id": None,
                "road_id": None,
                "lane_id": None,
                "meta": {}
            }

        h, w = self.mask.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return {
                "region_id": None,
                "road_id": None,
                "lane_id": None,
                "meta": {}
            }

        key_xy = (x, y)

        # ✅ CACHE HIT
        if self._cache_enabled and key_xy in self._lookup_cache:
            return self._lookup_cache[key_xy]

        rid = int(self.mask[y, x])

        if rid == 0:
            result = {
                "region_id": None,
                "road_id": None,
                "lane_id": None,
                "meta": {}
            }
        else:
            key = str(rid)
            meta = self.meta.get(key, {})

            road_id = None
            lane_id = None

            for k in ("roadId", "road_id", "road"):
                if k in meta and meta[k] not in (None, "", []):
                    road_id = meta[k]
                    break

            for k in ("laneId", "lane_id", "lane"):
                if k in meta and meta[k] not in (None, "", []):
                    lane_id = meta[k]
                    break

            try:
                if isinstance(road_id, str) and road_id.isdigit():
                    road_id = int(road_id)
            except Exception:
                pass

            resolved = rid
           
            if resolved not in self._regions_int:
                resolved = None


            result = {
                "region_id": resolved,
                "road_id": road_id,
                "lane_id": lane_id,
                "meta": meta
            }

        if self._cache_enabled:
            self._lookup_cache[key_xy] = result

        return result


    def is_valid_drivable_region(self, region_id: int) -> bool:
        if region_id is None:
            return False

        reg = self.get_region(region_id)
        if not reg:
            return False

        return reg.type in ("lane", "connecting_lane")

    def compute_lane_relation(self, r1: int, r2: int) -> dict:
        """
        Pure topology-based relationship between two regions.
        """

        if r1 is None or r2 is None:
            return {}

        c1 = self.get_region_context(r1) or {}
        c2 = self.get_region_context(r2) or {}

        t1 = c1.get("topology", {})
        t2 = c2.get("topology", {})

        relation = {
            "same_lane": (
                t1.get("lane_id") is not None and
                t1.get("lane_id") == t2.get("lane_id")
            ),

            "adjacent": (
                r1 in t2.get("adjacent", []) or
                r2 in t1.get("adjacent", []) or
                r1 in t2.get("adjacent_geometric", []) or
                r2 in t1.get("adjacent_geometric", [])
            ),

            "merge_relation": (
                r1 in t2.get("merges_with", []) or
                r2 in t1.get("merges_with", [])
            ),

            "successor_relation": (
                r1 in t2.get("successors", []) or
                r2 in t1.get("successors", [])
            ),

            "shared_junction": bool(
                set(t1.get("shares_upcoming_junction_with", [])) &
                set(t2.get("shares_upcoming_junction_with", []))
            )
        }

        return relation

    def get_region_context(self, region_id: int) -> dict:
        """
        Unified map-level context for a region.
        No vehicle/time dependency.
        """


        if region_id is None:
            return {}

        if self.meta is None:
            return {"region_id": region_id, "topology": {}, "junction": {}, "possibility": {}}


        meta = self.meta.get(str(region_id), {}) or {}

        raw_successors = meta.get("successors", []) or []
        raw_predecessors = meta.get("predecessors", []) or []
        raw_adjacent = meta.get("adjacent", []) or []
        topo = {
            "lane_type": meta.get("lane_type"),
            "is_connecting_road": meta.get("is_connecting_road", False),

            "direct_successors": meta.get("direct_successors", []) or [],
            "direct_predecessors": meta.get("direct_predecessors", []) or [],

            "successors": self._expand_overlap_ids(raw_successors),
            "predecessors": self._expand_overlap_ids(raw_predecessors),

            "adjacent": self._expand_overlap_ids(raw_adjacent),
            "adjacent_geometric": meta.get("adjacent_geometric", []) or [],

            "merges_with": meta.get("merges_with", []) or [],

            "shares_upcoming_junction_with": meta.get("shares_upcoming_junction_with", []) or [],
        }

        junction = {
            "upcoming_junction_ids": meta.get("upcoming_junction_ids", []) or [],
            "previous_junction_ids": meta.get("previous_junction_ids", []) or [],

            "distance_to_next_junction": meta.get("distance_to_next_junction_m"),
            "distance_to_prev_junction": meta.get("distance_to_prev_junction_m"),
        }

        possibility = {
            "can_continue": len(topo["successors"]) > 0,
            "can_merge": len(topo["merges_with"]) > 0,
            "can_change_lane": (
                len(topo["adjacent"]) > 0 or
                len(topo["adjacent_geometric"]) > 0
            ),
            "approaching_junction": (
                junction["distance_to_next_junction"] is not None and
                junction["distance_to_next_junction"] < 20.0
            ),
            "in_junction": junction["distance_to_next_junction"] == 0.0,
        }

        return {
            "region_id": region_id,
            "topology": topo,
            "junction": junction,
            "possibility": possibility,
        }

    def compute_pair_metrics(self, p1: dict, p2: dict):
        """
        Geometry + motion metrics between two objects at one frame.
        Inputs:
        p1, p2: {
            "xy": (x,y),
            "vx": float,
            "vy": float,
            "roadId": ...,
            "laneId": ...
        }
        """

        (x1, y1), (x2, y2) = p1["xy"], p2["xy"]

        dx, dy = x2 - x1, y2 - y1
        dist = float(np.hypot(dx, dy))

        vx_rel = p2["vx"] - p1["vx"]
        vy_rel = p2["vy"] - p1["vy"]

        inv_d = 1.0 / (dist + 1e-9)
        closing_speed = vx_rel * (dx * inv_d) + vy_rel * (dy * inv_d)

        ttc = float("inf")
        if closing_speed < 0:
            ttc = dist / (-closing_speed + 1e-6)

        same_lane = (
            p1.get("roadId") is not None and
            p1.get("roadId") == p2.get("roadId") and
            p1.get("laneId") == p2.get("laneId")
        )

        return dist, ttc, same_lane
    
    def group_consecutive_frames(self, frames, min_len=1):
        out = []
        if not frames:
            return out

        frames = sorted(frames)

        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
                continue

            if prev - start + 1 >= min_len:
                out.append((start, prev))

            start = prev = f

        if prev - start + 1 >= min_len:
            out.append((start, prev))

        return out    

    # ---------------- Utility ----------------
    def debug_summary(self):
        print(f"Total regions: {len(self.regions)}")

        types = {}
        for r in self.regions.values():
            types[r.type] = types.get(r.type, 0) + 1

        print("By type:", types)