# lane_map_viewer_gui.py
import json
import os 
import gc
import tempfile
import math
import time
from typing import List, Tuple
import cv2
import numpy as np
import copy
from scipy.spatial import cKDTree
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF
from PyQt6.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainterPath, QPolygonF
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QComboBox, QMessageBox, QSplitter, QSizePolicy,
    QProgressDialog, QApplication, QDoubleSpinBox, QSpinBox, QTextEdit, QListWidget, QListWidgetItem
)

from scenario_engine import (
    build_engine_insight_context,
    build_searchable_event_sequence,
    build_scenario_descriptor,
    suggest_rules_combined
)
from road_infrastructure import RoadInfrastructure
from signal_engine import ContextBuilder, SignalConfig, normalize_context, validate_context, PixelPoint, WorldPoint, VehicleTrajectory


# ===============================================================
# Utility loaders
# ===============================================================

def _fix_timeline_keys(tl):
    return {int(k): v for k, v in tl.items()}

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

def to_json_safe(obj):
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_json_safe(v) for k, v in vars(obj).items()}
    return obj

def load_openlabel(path: str):
    with open(path, encoding="utf-8") as f:
        frames = json.load(f).get("openlabel", {}).get("frames", {})

    frames_dict = {}
    traj_map = {}
    for fkey, fval in frames.items():
        try:
            fr = int(fkey)
            objs = fval.get("objects") or {}
        except Exception:
            continue
        rows = []
        for oid, oval in objs.items():
            rbbox = oval.get("object_data", {}).get("rbbox") or []
            shape = next((e.get("val") for e in rbbox if e.get("name") == "shape"), None)
            if not shape or len(shape) < 5:
                continue
            try:
                cx, cy, w, h, ang = map(float, shape[:5])
            except Exception:
                continue
            rows.append([cx, cy, w, h, ang, oid])
            traj_map.setdefault(oid, []).append((fr, cx, cy))
        if rows:
            frames_dict[fr] = rows
    if not frames_dict:
        return {}, {}, 0, 0
    fids = sorted(frames_dict)
    return frames_dict, traj_map, fids[0], fids[-1]


class LaneCanvas(QGraphicsView):
    clicked = pyqtSignal(int, int)
    region_hovered = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMouseTracking(True)
        self.mask, self.meta = None, None
        self.pixmap_item = None
        self.centerline_items = []
        self.mode = "manual"
        self.active_vehicle = None
        self.dragging_wp = None
        self._hover_traj_hit = None
        self._hit_radius_px = 10
        self.manual_waypoints = {"V1": [], "V2": []}
        self.wp_items = {"V1": [], "V2": []}
        self._viewer_cache = None
    # ======================================================
    # Data loading
    # ======================================================

    def load_mask(self, path: str):
        self.mask = np.load(path) if path.endswith(".npy") else self._read_img(path)

        rgb = self._colorize(self.mask)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

        if self.pixmap_item:
            self.scene.removeItem(self.pixmap_item)

        self.pixmap_item = self.scene.addPixmap(QPixmap.fromImage(qimg))
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def _read_img(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not read mask: {path}")
        return img[:, :, 0] if img.ndim == 3 else img

    def load_meta(self, path: str):
        with open(path, encoding="utf-8") as f:
            self.meta = json.load(f)
        return self.meta

    def _colorize(self, mask):
        uniq = np.unique(mask)

        rng = np.random.default_rng(1234)

        max_id = int(uniq.max())

        lut = np.zeros((max_id + 1, 3), dtype=np.uint8)

        for u in uniq:
            if u == 0:
                continue

            lut[int(u)] = rng.integers(
                80,
                255,
                size=3,
                dtype=np.uint8
            )

        return lut[mask]

    # ======================================================
    # Drawing helpers
    # ======================================================

    def draw_centerlines(self, road):
        self._clear_items(self.centerline_items)

        if not road:
            return

        for reg in road.regions.values():
            if reg.type not in ("lane", "connecting_lane") or not reg.centerline:
                continue

            pts = np.asarray(reg.centerline)
            if len(pts) < 2:
                continue

            path = QPainterPath()
            path.moveTo(*pts[0])
            for p in pts[1:]:
                path.lineTo(*p)

            item = self.scene.addPath(path, QPen(QColor(255, 0, 0, 180), 2))
            item.setZValue(40)
            self.centerline_items.append(item)

    def draw_trajectory(self, pts, color):
        if not pts:
            return []

        path = QPainterPath()
        path.moveTo(*pts[0])
        for p in pts[1:]:
            path.lineTo(*p)

        poly = QGraphicsPathItem(path)
        poly.setPen(QPen(color, 3))
        poly.setZValue(50)
        self.scene.addItem(poly)

        r = 5
        start = self.scene.addEllipse(*self._circle(pts[0], r), QPen(color), QBrush(color))
        end = self.scene.addEllipse(*self._circle(pts[-1], r), QPen(color), QBrush(color))

        for it in (start, end):
            it.setZValue(51)

        return [poly, start, end]

    def _circle(self, pt, r):
        x, y = pt
        return x - r, y - r, 2 * r, 2 * r

    def _clear_items(self, items):
        for it in items:
            self.scene.removeItem(it)
        items.clear()

    # ======================================================
    # Waypoints
    # ======================================================

    def set_mode(self, mode: str):
        self.mode = mode

    def enable_waypoints(self, vehicle: str):
        self.mode = "waypoints"
        self.active_vehicle = vehicle

    def clear_waypoints(self, vehicle=None):
        for v in ([vehicle] if vehicle else ("V1", "V2")):
            self._clear_items(self.wp_items[v])
            self.manual_waypoints[v].clear()

    def _draw_waypoints(self, vehicle):
        pts = self.manual_waypoints[vehicle]
        items = self.wp_items[vehicle]
        color = QColor("green") if vehicle == "V1" else QColor("cyan")

        while len(items) > len(pts):
            self.scene.removeItem(items.pop())

        for i, (x, y) in enumerate(pts):
            if i < len(items):
                items[i].setRect(x - 4, y - 4, 8, 8)
            else:
                dot = self.scene.addEllipse(x - 4, y - 4, 8, 8,
                                            QPen(color, 2), QBrush(color))
                dot.setZValue(60)
                items.append(dot)

    # ======================================================
    # Interaction helpers
    # ======================================================

    def in_image_bounds(self, x, y):
        return self.mask is not None and 0 <= x < self.mask.shape[1] and 0 <= y < self.mask.shape[0]


    def _detect_hover_on_trajectory(self, x, y):
        viewer = self._find_viewer()

        if not viewer:
            return None

        tree = getattr(viewer, "_hover_tree", None)
        index = getattr(viewer, "_hover_index", None)

        if tree is None or not index:
            return None

        dist, pos = tree.query(
            [float(x), float(y)],
            k=1
        )

        if dist > self._hit_radius_px:
            return None

        vid, idx = index[pos]

        return (
            vid,
            idx,
            float(dist)
        )


    # ======================================================
    # Mouse events
    # ======================================================

    def wheelEvent(self, e):
        s = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        self.scale(s, s)

    def mousePressEvent(self, ev):
        x, y = map(int, (self.mapToScene(ev.pos()).x(), self.mapToScene(ev.pos()).y()))

        if self.mode == "waypoints" and self.active_vehicle:
            pts = self.manual_waypoints[self.active_vehicle]

            for i, (wx, wy) in enumerate(pts):
                if abs(x - wx) <= 8 and abs(y - wy) <= 8:
                    self.dragging_wp = (self.active_vehicle, i)
                    return

            pts.append((x, y))
            self._draw_waypoints(self.active_vehicle)
            return

        self.clicked.emit(x, y)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        x, y = map(int, (self.mapToScene(ev.pos()).x(), self.mapToScene(ev.pos()).y()))

        viewer = self._find_viewer()
        rid, md = None, {}

        if viewer and getattr(viewer, "road", None):
            try:
                res = viewer.road.region_lookup(x, y) or {}
            except Exception:
                res = {}

            md = res.get("meta", {}) or {}
            rid = res.get("region_id")
            if hasattr(viewer, "update_hover_coords"):
                viewer.update_hover_coords(x, y)

        self._hover_traj_hit = self._detect_hover_on_trajectory(x, y)
        self.region_hovered.emit(rid, md)

        if self.dragging_wp:
            v, idx = self.dragging_wp
            self.manual_waypoints[v][idx] = (x, y)
            self._draw_waypoints(v)
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.dragging_wp = None
        super().mouseReleaseEvent(ev)

    def _find_viewer(self):
        if self._viewer_cache is not None:
            return self._viewer_cache

        viewer = self.parent()

        while viewer and not hasattr(viewer, "road"):
            viewer = viewer.parent()

        self._viewer_cache = viewer
        return viewer


# ======================================================================
# Main Viewer
# ======================================================================
class LaneMapViewer(QWidget):
    polyline_committed = pyqtSignal(str, str)
    mask_loaded = pyqtSignal(object)
    meta_loaded = pyqtSignal(object)
    xodr_loaded = pyqtSignal(str)
    map_context_ready = pyqtSignal(dict)
    duration_changed = pyqtSignal(float)
    xodr_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Map Viewer")
        self._init_state()
        self._init_shared_controls() 
        self._build_main_layout()
        self.view.region_hovered.connect(self.update_hover_info)
        self.update_status_panel()
        self.resize(1600, 900)
        self.showMaximized()
        QTimer.singleShot(0, lambda: self._on_mode_changed(self.mode_box.currentText()))

    def _init_state(self):
        self._project = {
            "mask": None,
            "layout": None,
            "homography": None,
            "xodr": None,
            "openlabel": None
        }

        self._mask_path = None
        self._layout_path = None
        self._homog_path = None
        self._xodr_path = None
        self._openlabel_path = None
        self.homography = None
        self.road = None
        self.signal_config = SignalConfig()
        self.fps = 0.0
        self.num_frames = 0
        self.sim_duration_s = None
        self._hover_px = None
        self._hover_wp = None
        self.current_wp_vehicle = None
        self.active_mode = "manual"
        self.current_pick_mode = "V1 Initial"
        self._traj_map_manual = {}
        self._traj_map_json = {}
        self._world_polylines_manual = {"V1": None, "V2": None}
        self._world_polylines_json = {"V1": None, "V2": None}
        self._engine_context = None
        self._manual_dirty = False
        self._json_dirty = False       
        self._last_similarity_results = [] 
        self._similarity_items = []
        self._similarity_candidate = None        
        self._vehicle_context_cache = {}
        self._openlabel_vehicle_ctx_cache = {}
        self._builder_traj_cache = {}
        self._semantic_lane_roles = {}
        self._centerline_geom_cache = {}

        if self.is_manual_mode:
            self._manual_traj_done = {"V1": False, "V2": False}

        self._traj_items = {
            "manual": {"V1": [], "V2": []},
            "json": {"V1": [], "V2": []},
        }
        self._map_context_manual = {"ready": False}
        self._map_context_json = {"ready": False}
        self.openlabel_frames = None
        self.openlabel_traj_map = None
        self.openlabel_fmin = 0
        self.openlabel_fmax = 0
        self._precomp = {
            "per_frame": {},
            "pair_windows": []
        }
        self.PRECOMP_DIST_THRESH_PX = 80.0
        self.PRECOMP_TTC_THRESH_FRAMES = 3.0
        self.PRECOMP_MIN_CONSEC_FRAMES = 10
        self.PRECOMP_GRID_CELL_PX = 96
        self.PRECOMP_GRID_NEIGHBOR_RADIUS = 1
        self.PRECOMP_MAX_CELL_PAIR_COMBOS = 2000
        self.scenario_gui = None

    @property
    def is_manual_mode(self):
        return self.active_mode in ("manual", "waypoints")

    def clear_similarity_preview(self):

        if not self.view:
            return
        scene = self.view.scene
        for item in self._similarity_items:
            try:
                scene.removeItem(item)
            except Exception:
                pass
        self._similarity_items.clear()
        self._similarity_candidate = None

    def _draw_preview_polyline(
        self,
        points,
        color,
        width=3
    ):
        if len(points) < 2:
            return []
        poly = QPolygonF(
            [
                QPointF(x, y)
                for x, y in points
            ]
        )
        item = QGraphicsPathItem()
        path = QPainterPath()
        path.moveTo(
            poly[0]
        )
        for p in poly[1:]:
            path.lineTo(p)
        item.setPath(path)
        pen = QPen(
            QColor(color)
        )
        pen.setWidth(width)
        item.setPen(pen)
        self.view.scene.addItem(
            item
        )
        return [item]

    def _on_similarity_selected(self):

        items = (
            self.similar_list.selectedItems()
        )

        if not items:
            return

        idx = items[0].data(
            Qt.ItemDataRole.UserRole
        )

        if (
            idx is None
            or idx >= len(
                self._last_similarity_results
            )
        ):
            return

        result = (
            self._last_similarity_results[idx]
        )

        self.preview_similarity_match(
            result["track"]
        )

    def preview_similarity_match(
        self,
        track
    ):

        self.clear_similarity_preview()

        actors = track.get(
            "actor_ids",
            []
        )

        if len(actors) < 2:

            self.log_warn(
                "Candidate has no actor pair."
            )

            return

        start_frame = track.get(
            "start_frame",
            track.get("first_frame")
        )

        end_frame = track.get(
            "end_frame",
            track.get("last_frame")
        )


        if (
            start_frame is None
            or end_frame is None
        ):

            self.log_warn(
                "Candidate contains no frame range."
            )

            return

        traj_map = getattr(
            self,
            "openlabel_traj_map",
            None
        )

        if not traj_map:

            self.log_warn(
                "OpenLABEL trajectories not loaded."
            )

            return

        try:

            v1 = str(actors[0])
            v2 = str(actors[1])

            if v1 not in traj_map:

                self.log_warn(
                    f"Vehicle {v1} not found."
                )

                return

            if v2 not in traj_map:

                self.log_warn(
                    f"Vehicle {v2} not found."
                )

                return

            traj1 = [
                (x, y)

                for f, x, y
                in traj_map[v1]

                if start_frame <= f <= end_frame
            ]

            traj2 = [
                (x, y)

                for f, x, y
                in traj_map[v2]

                if start_frame <= f <= end_frame
            ]

            self.log_info(
                f"Preview candidate "
                f"{v1}-{v2} "
                f"frames={start_frame}-{end_frame} "
                f"pts=({len(traj1)},{len(traj2)})"
            )

            if len(traj1) < 2:

                self.log_warn(
                    f"{v1} preview trajectory too short."
                )

                return

            if len(traj2) < 2:

                self.log_warn(
                    f"{v2} preview trajectory too short."
                )

                return

            self._similarity_items.extend(
                self._draw_preview_polyline(
                    traj1,
                    "#FFD600",
                    4
                )
            )

            self._similarity_items.extend(
                self._draw_preview_polyline(
                    traj2,
                    "#FF00FF",
                    4
                )
            )

            self._similarity_candidate = track

        except Exception as e:

            self.log_warn(
                f"Preview failed: {e}"
            )

    def _resolve_to_lane(self, rid, wp_obj, last_lane=None):
        if rid is None or not self.road:
            return None
        rc = self.road.get_region_context(rid)
        if not rc:
            return None
        rtype = rc.get("type")
        if rtype in ("lane", "connecting_lane"):
            return rid
        if rtype != "overlap":
            return None
        members = rc.get("members") or []
        if not members:
            return None

        candidates = []
        if wp_obj:
            for m in members:
                rc_m = self.road.get_region_context(m) or {}
                if rc_m.get("type") not in ("lane", "connecting_lane"):
                    continue

                try:
                    proj = self.road.project_world_to_frenet(m, wp_obj.x, wp_obj.y)
                except Exception:
                    proj = None

                if not self._valid_proj(proj):
                    continue

                s = proj.get("s")
                d = proj.get("d")

                if (
                    isinstance(s, (int, float)) and
                    isinstance(d, (int, float)) and
                    not math.isnan(s) and
                    not math.isnan(d)
                ):
                    score = 0.0
                    if last_lane == m:
                        score += 10.0
                    score += -abs(d)

                    candidates.append((m, score))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        if wp_obj:
            px, py = wp_obj.x, wp_obj.y

            best = None
            best_dist = float("inf")

            for m in members:
                reg = self.road.regions.get(m)
                if not reg or reg.type not in ("lane", "connecting_lane"):
                    continue

                centerline = getattr(reg, "centerline", None)
                if not centerline or len(centerline) < 2:
                    continue

                pts = np.asarray(centerline)
                dists = np.linalg.norm(pts - np.array([px, py]), axis=1)
                dmin = float(np.min(dists))

                if dmin < best_dist:
                    best_dist = dmin
                    best = m

            if best is not None:
                return best

        return members[0]

    def _clear_edit_selection(self):
        self.current_wp_vehicle = None
        self.view.active_vehicle = None

        self.ctrl_v1_btn.setStyleSheet("")
        self.ctrl_v2_btn.setStyleSheet("")

    def _normalize_event(self, ev):

        return {
            "type": str(ev.get("type", "unknown")),

            "t_start": int(
                ev.get(
                    "t_start",
                    ev.get("frame_start", 0)
                )
            ),

            "t_end": int(
                ev.get(
                    "t_end",
                    ev.get("frame_end", 0)
                )
            ),

            "participants": sorted(
                [
                    str(x)
                    for x in ev.get(
                        "participants",
                        []
                    )
                ]
            )
        }


    def _extract_sequence_tracks(
        self,
        event_timeline,
        v1_id,
        v2_id
    ):

        tracks = []

        if not event_timeline:
            return tracks

        normalized = [
            self._normalize_event(ev)
            for ev in event_timeline
        ]

        normalized.sort(
            key=lambda e: e["t_start"]
        )

        tracks.append({
            "scope": "pair",

            "actor_ids": [
                str(v1_id),
                str(v2_id)
            ],

            "events": normalized
        })

        for vid in [str(v1_id), str(v2_id)]:

            veh_events = []

            for ev in normalized:

                parts = ev.get(
                    "participants",
                    []
                )

                if len(parts) == 1 and vid in parts:
                    veh_events.append(ev)

            if veh_events:

                tracks.append({
                    "scope": "vehicle",
                    "actor_ids": [vid],
                    "events": veh_events
                })

        return tracks

    def _traj_to_builder_format(self, traj):
        out = []
        last_lane = None
        last_s_local = None
        global_s = 0.0

        for frame_id, x, y in traj:
            wp = self.road.pixel_to_world(x, y) if self.road else None
            if wp:
                world = {"x": float(wp[0]), "y": float(wp[1])}
                wp_obj = WorldPoint(wp[0], wp[1])
            else:
                world = {"x": float("nan"), "y": float("nan")}
                wp_obj = None
            raw_rid = None
            rid = None
            if self.road:
                try:
                    res = self.road.region_lookup(int(x), int(y)) or {}
                    raw_rid = res.get("region_id")
                    rid = raw_rid
                    if rid is not None and wp_obj:
                        rid_resolved = self._resolve_to_lane(rid, wp_obj, last_lane)
                        if rid_resolved is not None:
                            rid = rid_resolved
                except Exception:
                    pass
            proj = None

            if self.road and wp and rid is not None:
                try:
                    proj = self.road.project_world_to_frenet(rid, wp[0], wp[1])
                except Exception:
                    proj = None
            if (
                (not proj or not isinstance(proj.get("s"), (int, float)) or math.isnan(proj.get("s")))
                and raw_rid is not None
                and wp_obj
            ):
                rid_resolved = self._resolve_to_lane(raw_rid, wp_obj, last_lane)

                if rid_resolved is not None:
                    try:
                        proj2 = self._project_to_centerline(rid_resolved, wp)

                        if not self._valid_proj(proj2):
                            if last_lane:
                                proj2 = self._project_to_centerline(last_lane, wp)

                        if self._valid_proj(proj2):
                            proj = proj2
                            rid = rid_resolved
                    except Exception:
                        pass
            if (
                (not proj or not isinstance(proj.get("s"), (int, float)) or math.isnan(proj.get("s")))
                and last_lane is not None
                and wp
            ):
                try:
                    proj2 = self.road.project_world_to_frenet(
                        last_lane, wp[0], wp[1]
                    )

                    if proj2:
                        proj = proj2
                        rid = last_lane
                except Exception:
                    pass

            sd = {"s": float("nan"), "d": float("nan")}

            if self._valid_proj(proj):
                s_local = proj.get("s")
                d = proj.get("d")

                if (
                    isinstance(s_local, (int, float)) and
                    isinstance(d, (int, float)) and
                    not math.isnan(s_local) and
                    not math.isnan(d)
                ):

                    if last_lane == rid and last_s_local is not None:
                        ds = s_local - last_s_local

                        if abs(ds) < 5.0:
                            global_s += max(ds, 0.0)
                        else:
                            global_s += max(s_local, 0.0)
                    else:
                        if last_s_local is not None:
                            global_s += max(s_local, 0.0)

                    sd = {
                        "s": float(global_s),
                        "d": float(d)
                    }
                    last_s_local = s_local
                    last_lane = rid

            out.append((
                frame_id,
                {
                    "world": world,
                    "region_id": rid,
                    "sd": sd,

                    "topology": (
                        self._semantic_lane_roles.get(
                            str(rid),
                            ()
                        )
                        if rid is not None
                        else ()
                    )
                }
            ))

        return out

    def _build_semantic_lane_roles(self):

        self._semantic_lane_roles = {}

        if not self.road or not getattr(self.road, "meta", None):
            return

        meta = self.road.meta

        for rid, info in meta.items():

            if not isinstance(info, dict):
                continue

            rtype = str(
                info.get("type", "")
            )

            if rtype not in (
                "lane",
                "connecting_lane"
            ):
                continue

            succ = info.get(
                "successors",
                []
            ) or []

            pred = info.get(
                "predecessors",
                []
            ) or []

            role = []

            if info.get(
                "is_connecting_road",
                False
            ):
                role.append(
                    "CONNECTOR"
                )

            if len(pred) == 0:
                role.append(
                    "ENTRY"
                )

            if len(succ) == 0:
                role.append(
                    "EXIT"
                )

            if (
                len(pred) > 0
                and
                len(succ) > 0
                and
                not info.get(
                    "is_connecting_road",
                    False
                )
            ):
                role.append(
                    "FLOW"
                )

            if len(succ) >= 2:
                role.append(
                    "SPLIT"
                )

            if len(pred) >= 2:
                role.append(
                    "MERGE"
                )
            junction_id = info.get(
                "junction_id"
            )

            if junction_id is not None:
                role.append(
                    f"JUNCTION_{junction_id}"
                )

            self._semantic_lane_roles[
                str(rid)
            ] = tuple(role)
    def _get_lane_semantic_debug(self, lane_id):

        roles = self._semantic_lane_roles.get(
            str(lane_id),
            ()
        )

        rc = None

        try:
            rc = self.road.get_region_context(
                lane_id
            )
        except Exception:
            pass

        if not rc:
            return {}

        return {
            "roles": list(roles),
            "successors": rc.get("successors", []),
            "predecessors": rc.get("predecessors", []),
            "adjacent": rc.get("adjacent", [])
        }
    def _get_centerline_geom(self, lane_id):

        cached = self._centerline_geom_cache.get(lane_id)
        if cached is not None:
            return cached

        reg = self.road.regions.get(lane_id)
        if not reg:
            return None

        cl = getattr(reg, "centerline", None)
        if not cl or len(cl) < 2:
            return None

        pts = np.asarray(cl, dtype=np.float64)

        p1 = pts[:-1]
        p2 = pts[1:]

        v = p2 - p1

        seg_len = np.linalg.norm(v, axis=1)

        valid = seg_len > 1e-6

        cumulative = np.zeros(len(seg_len) + 1)
        cumulative[1:] = np.cumsum(seg_len)

        result = {
            "pts": pts,
            "p1": p1,
            "p2": p2,
            "v": v,
            "seg_len": seg_len,
            "valid": valid,
            "cumulative": cumulative
        }

        self._centerline_geom_cache[lane_id] = result

        return result

    def _valid_proj(self, proj):
        if not proj:
            return False

        s = proj.get("s")
        d = proj.get("d")

        return (
            isinstance(s, (int, float)) and
            isinstance(d, (int, float)) and
            not math.isnan(s) and
            not math.isnan(d)
        )


    def _project_to_centerline(self, lane_id, wp):

        geom = self._get_centerline_geom(lane_id)

        if not geom:
            return None

        px = float(wp[0])
        py = float(wp[1])

        best_s = None
        best_d = None
        best_dist = float("inf")

        q = np.array([px, py], dtype=np.float64)

        p1_arr = geom["p1"]
        v_arr = geom["v"]
        seg_len_arr = geom["seg_len"]
        valid_arr = geom["valid"]
        cumulative = geom["cumulative"]

        for i in range(len(v_arr)):

            if not valid_arr[i]:
                continue

            p1 = p1_arr[i]
            v = v_arr[i]
            seg_len = seg_len_arr[i]

            w = q - p1

            t = np.dot(w, v) / (seg_len * seg_len)

            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            proj = p1 + t * v

            diff = q - proj
            dist = np.hypot(diff[0], diff[1])

            if dist < best_dist:

                best_dist = dist

                best_s = cumulative[i] + (t * seg_len)

                normal = np.array(
                    [-v[1], v[0]],
                    dtype=np.float64
                )

                sign = np.sign(
                    np.dot(w, normal)
                )

                best_d = sign * dist

        if best_s is None:
            return None

        return {
            "s": float(best_s),
            "d": float(best_d)
        }


    
    # ===================== NEW MAIN LAYOUT =====================

    def _get_hover_vehicle_insight(
            self,
            vid,
            idx):

        ctx = (
            self._map_context_manual
            if self.is_manual_mode
            else self._map_context_json
        )

        actor = (
            "A"
            if vid == "V1"
            else "B"
        )

        timeline = (
            ctx.get("actors", {})
            .get(actor, {})
            .get("timeline", {})
        )

        frame = timeline.get(idx)

        if not frame:
            return None

        sd = frame.get("sd", {})

        return {
            "s": sd.get("s"),
            "d": sd.get("d"),
            "t": idx,
            "region": frame.get("region_id")
        }

    def _get_hover_pair_insight(self, idx):
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json

        timeline = (
            ctx.get(
                "interaction_facts",
                {}
            )
            .get(
                "timeline",
                {}
            )
        )
        p = timeline.get(idx)
        if not p:
            return None

        return {
            "dist": p.get("distance"),
            "ds": p.get("delta_s"),
            "dd": p.get("delta_d"),
        }


    def _init_shared_controls(self):
        self.global_start_box = QComboBox()
        self.global_end_box = QComboBox()
        self.v1_id_box = QComboBox()
        self.v2_id_box = QComboBox()
        self.json_status = QLabel("No data loaded")
        self.global_start_box.currentTextChanged.connect(self._on_json_frame_range_changed)
        self.global_end_box.currentTextChanged.connect(self._on_json_frame_range_changed)
        self.wp_duration = QDoubleSpinBox()
        self.wp_duration.setRange(0.1, 60)
        self.wp_duration.setValue(3.0)
        self.wp_fps = QSpinBox()
        self.wp_fps.setRange(1, 240)
        self.wp_fps.setValue(30)
        self.wp_duration.valueChanged.connect(self._on_manual_timing_changed)
        self.wp_fps.valueChanged.connect(self._on_manual_timing_changed)

    def _update_waypoint_visibility(self):
        is_manual = self.is_manual_mode
        for v in ("V1", "V2"):
            for item in self.view.wp_items[v]:
                item.setVisible(is_manual)

       
    def _on_mode_changed(self, txt):

        previous_mode = self.active_mode

        self.active_mode = (
            "manual"
            if txt.lower() == "manual"
            else "json"
        )

        is_manual = self.is_manual_mode

        self._clear_edit_selection()

        self.view.dragging_wp = None
        self.view.set_mode(self.active_mode)

        self.json_status.setVisible(not is_manual)

        self.ctrl_v1_btn.setVisible(is_manual)
        self.ctrl_v2_btn.setVisible(is_manual)

        self.ctrl_v1_box.setVisible(not is_manual)
        self.ctrl_v2_box.setVisible(not is_manual)

        self.ctrl_start.setEnabled(not is_manual)
        self.ctrl_end.setEnabled(not is_manual)

        self.ctrl_generate.setEnabled(is_manual)

        self.ctrl_clear.setVisible(is_manual)

        self._hide_all_trajectories()
        self._show_active_mode_trajectories()

        self._update_waypoint_visibility()

        self.update_status_panel()

        self.log_info(
            f"Mode changed: "
            f"{previous_mode.upper()} -> {self.active_mode.upper()}"
        )

        self._emit_active_context()

    def _build_top_toolbar(self):
        row = QHBoxLayout()

        self.btn_new_proj = QPushButton("New")
        self.btn_open_proj = QPushButton("Open")
        self.btn_save_proj = QPushButton("Save")

        self.btn_load_mask = QPushButton("Mask")
        self.btn_load_meta = QPushButton("Layout")
        self.btn_load_homog = QPushButton("H")
        self.btn_load_xodr = QPushButton("xODR")

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Manual", "JSON"])
        self.mode_box.currentTextChanged.connect(self._on_mode_changed)

        self.btn_new_proj.clicked.connect(self.new_project)
        self.btn_open_proj.clicked.connect(self.open_project)
        self.btn_save_proj.clicked.connect(self.save_project)

        self.btn_load_mask.clicked.connect(self.load_mask)
        self.btn_load_meta.clicked.connect(
            lambda: self.load_meta_from_path(
                self._load_with_dialog("Layout", "JSON (*.json)")
            )
        )
        self.btn_load_homog.clicked.connect(self.load_homography_json)
        self.btn_load_xodr.clicked.connect(self.load_xodr)

        for b in [
            self.btn_new_proj, self.btn_open_proj, self.btn_save_proj,
            self.btn_load_mask, self.btn_load_meta,
            self.btn_load_homog, self.btn_load_xodr
        ]:
            row.addWidget(b)

        row.addWidget(QLabel("Mode:"))
        row.addWidget(self.mode_box)

        row.addStretch()

        return row

    def _build_main_layout(self):
        root = QVBoxLayout(self)
        root.addLayout(self._build_top_toolbar())
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.view = LaneCanvas(self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.panel_controls = self._build_unified_panel()
        self.bottom_layout.addWidget(self.panel_controls)
        self._build_info_panel()
        splitter.addWidget(self.view)
        splitter.addWidget(self.bottom_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        QTimer.singleShot(0, lambda: splitter.setSizes([900, 200]))
        root.addWidget(splitter)
        return root

    def _build_unified_panel(self):

        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --------------------------------------------------
        # Vehicle row
        # --------------------------------------------------

        veh_row = QHBoxLayout()

        self.ctrl_v1_btn = QPushButton("✏️")
        self.ctrl_v2_btn = QPushButton("✏️")

        self.ctrl_v1_box = self.v1_id_box
        self.ctrl_v2_box = self.v2_id_box

        self.ctrl_v1_btn.clicked.connect(
            lambda: self._set_wp_mode("V1")
        )
        self.ctrl_v2_btn.clicked.connect(
            lambda: self._set_wp_mode("V2")
        )

        veh_row.addWidget(QLabel("V1"))
        veh_row.addWidget(self.ctrl_v1_btn)
        veh_row.addWidget(self.ctrl_v1_box)

        veh_row.addSpacing(20)

        veh_row.addWidget(QLabel("V2"))
        veh_row.addWidget(self.ctrl_v2_btn)
        veh_row.addWidget(self.ctrl_v2_box)

        veh_row.addStretch()

        layout.addLayout(veh_row)

        # --------------------------------------------------
        # Time row
        # --------------------------------------------------

        time_row = QHBoxLayout()

        self.ctrl_start = self.global_start_box
        self.ctrl_end = self.global_end_box

        self.ctrl_duration = self.wp_duration
        self.ctrl_fps = self.wp_fps

        time_row.addWidget(QLabel("Start"))
        time_row.addWidget(self.ctrl_start)

        time_row.addWidget(QLabel("End"))
        time_row.addWidget(self.ctrl_end)

        time_row.addSpacing(15)

        time_row.addWidget(QLabel("Dur"))
        time_row.addWidget(self.ctrl_duration)

        time_row.addWidget(QLabel("FPS"))
        time_row.addWidget(self.ctrl_fps)

        time_row.addStretch()

        layout.addLayout(time_row)

        # --------------------------------------------------
        # Actions row
        # --------------------------------------------------

        act_row = QHBoxLayout()

        self.ctrl_generate = QPushButton("Generate")
        self.ctrl_apply = QPushButton("Apply")
        self.ctrl_clear = QPushButton("Clear")
        self.ctrl_build_timeline = QPushButton("Build/Refresh DB Index")

        self.ctrl_find_similar = QPushButton("Find Similar")
        self.ctrl_load_db = QPushButton("Load Semantic DB")

        act_row.addWidget(self.ctrl_generate)
        act_row.addWidget(self.ctrl_apply)
        act_row.addWidget(self.ctrl_clear)
        act_row.addWidget(self.ctrl_build_timeline)
        act_row.addSpacing(20)

        act_row.addWidget(self.ctrl_find_similar)

        act_row.addStretch()

        layout.addLayout(act_row)

        # --------------------------------------------------
        # Status row
        # --------------------------------------------------
        self.similar_list = QListWidget()
        self.similar_list.setMinimumHeight(220)

        self.similar_list.itemSelectionChanged.connect(
            self._on_similarity_selected
        )

        layout.addWidget(
            QLabel("Similarity Matches")
        )

        layout.addWidget(
            self.similar_list
        )        

        status_row = QHBoxLayout()

        status_row.addWidget(self.json_status)

        self.similarity_status = QLabel(
            "No similarity search executed"
        )

        status_row.addWidget(self.similarity_status)

        status_row.addStretch()

        layout.addLayout(status_row)

        # --------------------------------------------------
        # Wiring
        # --------------------------------------------------

        self.ctrl_generate.clicked.connect(
            self._on_generate_clicked
        )

        self.ctrl_apply.clicked.connect(
            self._on_apply_clicked
        )

        self.ctrl_clear.clicked.connect(
            self._on_clear_clicked
        )

        self.ctrl_find_similar.clicked.connect(
            self.find_similar_scenarios
        )

        self.ctrl_build_timeline.clicked.connect(
            self.build_semantic_timeline_database
        )
        self.ctrl_load_db.clicked.connect(
            self.load_semantic_database_dialog
        )        

        return panel


    def _on_clear_clicked(self):
        self._invalidate_active_scenario("clear_clicked")
        if self.is_manual_mode:
            self._clear_all_waypoints()
            self._manual_traj_done = {"V1": False, "V2": False}
            self._traj_map_manual.clear()
            self._vehicle_context_cache.clear()
            self._world_polylines_manual = {"V1": None, "V2": None}
        else:
            self.clear_all_trajectories()
            self._traj_map_json.clear()
            self._world_polylines_json = {"V1": None, "V2": None}

        self._emit_active_context()
        self.update_status_panel()
        self.log_info("Cleared current mode data.")

    def _rebuild_context_builder(self):
        if self.road:
            self._ctx_builder = ContextBuilder(
                self.road,
                self.signal_config
            )

    def _build_vehicle_contexts(self):
        traj_map = self._get_active_traj_map()

        if not traj_map:
            return None, "missing_trajectories"

        v1_traj = traj_map.get("V1")
        v2_traj = traj_map.get("V2")

        if not v1_traj or not v2_traj:
            return None, "incomplete_vehicles"

        self._rebuild_context_builder()
        try:
            traj1 = self._traj_to_builder_format(v1_traj)
            traj2 = self._traj_to_builder_format(v2_traj)

            ctx_a = self._ctx_builder.build_vehicle_context(traj1)
            ctx_b = self._ctx_builder.build_vehicle_context(traj2)

        except Exception as e:
            return None, f"vehicle_ctx_failed: {e}"

        return (ctx_a, ctx_b), None

    def _empty_context(self):

        return {

            "kind":
                "map_scenario_context",

            "ready":
                False,

            "state":
                "init",

            "error":
                None,

            "actors": {},

            "interaction_facts": {},

            "interaction_topology": {},

            "meta": {}
        }

    def _invalidate_active_scenario(self, reason=""):
        """
        Active scenario changed.

        Any engine-derived products are now invalid.
        """

        self._engine_context = None

        if self.scenario_gui:
            try:
                if hasattr(self.scenario_gui, "reset_vehicle_contexts"):
                    self.scenario_gui.reset_vehicle_contexts()
            except Exception:
                pass

        if reason:
            print(f"[LaneMapViewer] Scenario invalidated: {reason}")

    def apply_manual_trajectory(self):
        for v in ("V1", "V2"):
            traj = self._traj_map_manual.get(v, [])
            if len(traj) < 2:
                return self.log_warn(f"{v} trajectory too short.")

            world = self._world_polylines_manual.get(v)
            if not world or len(world) < 2:
                return self.log_warn(f"{v} world trajectory invalid.")
        self._engine_context = None
        self.emit_all_vehicle_map_contexts()
        self.log_info("Manual trajectories applied")

    def _build_info_panel(self):
        row = QHBoxLayout()

        self.status_label = QLabel()
        self.status_label.setFixedHeight(60)  

        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(80)  

        self.log_view = QTextEdit()
        self.log_view.setFixedHeight(80)

        row.addWidget(self.status_label, 2)
        row.addWidget(self.info_label, 3)
        row.addWidget(self.log_view, 2)

        self.bottom_layout.addLayout(row)

    def _debug_save_context(self, ctx, tag="auto"):
        try:
            os.makedirs("debug_contexts", exist_ok=True)

            ts = int(time.time())
            path = f"debug_contexts/context_{tag}_{ts}.json"

            clean = to_json_safe(ctx)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)

            print(f"[DEBUG] Context saved → {path}")
        except Exception as e:
            print(f"[DEBUG] Save failed: {e}")



    def _on_manual_timing_changed(self):

        if not self.is_manual_mode:
            return

        if not any(self._manual_traj_done.values()):
            return

        self._manual_dirty = True

        self._invalidate_active_scenario(
            "manual_timing_changed"
        )

        self.log_warn(
            "Manual timing changed. "
            "Press Generate to rebuild trajectories."
        )

        self.update_status_panel()


    def compute_and_emit_duration(self):
        if self.fps <= 0 or self.num_frames <= 0:
            self.sim_duration_s = None
            return None

        duration = float(max(self.num_frames - 1, 1)) / max(self.fps, 1.0)
        self.sim_duration_s = duration
        self.duration_changed.emit(duration)
        return duration

    def _rebuild_hover_index(self):
        self._hover_index = []
        self._hover_tree = None

        traj_map = self._get_active_traj_map()

        points = []
        meta = []

        for vid, traj in traj_map.items():
            for idx, (_, x, y) in enumerate(traj):
                points.append((float(x), float(y)))
                meta.append((vid, idx))

        if points:
            self._hover_tree = cKDTree(points)
            self._hover_index = meta

    def _get_active_traj_map(self):
        if self.is_manual_mode:
            return self._traj_map_manual
        elif self.active_mode == "json":
            return self._traj_map_json
        return {}

    def _rebuild_vehicle_context_cache(self):
        """
        Cache vehicle trajectory lookups.

        Safe optimization:
        only derived data, no behavioural changes.
        """

        self._vehicle_context_cache.clear()

        traj_map = self._get_active_traj_map()

        for vid, traj in traj_map.items():

            frame_lookup = {}
            frames = []

            for fr, x, y in traj:
                fr_i = int(fr)

                frame_lookup[fr_i] = (
                    float(x),
                    float(y)
                )

                frames.append(fr_i)

            if frames:
                self._vehicle_context_cache[vid] = {
                    "frame_lookup": frame_lookup,
                    "frames": frames,
                    "frame_set": set(frames),
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "length": len(frames)
                }

    def _get_active_store(self):
        return {
            "traj_map": self._traj_map_manual if self.is_manual_mode else self._traj_map_json,
            "world": self._world_polylines_manual if self.is_manual_mode else self._world_polylines_json,
            "context": self._map_context_manual if self.is_manual_mode else self._map_context_json,
            "items": self._traj_items.setdefault(self.active_mode, {"V1": [], "V2": []})
        }


    def _get_valid_duration(self):
        if self.is_manual_mode:
            return float(self.wp_duration.value())
        if self.sim_duration_s and self.sim_duration_s > 0:
            return float(self.sim_duration_s)
        if self.fps > 0 and self.num_frames > 0:
            return float(max(self.num_frames - 1, 1)) / float(self.fps)
        return 0.1

    def _build_normalize_validate_emit(self):

        ctx = self.build_context_atomic()

        try:

            ctx = normalize_context(ctx)

            validate_context(ctx)

            ctx["ready"] = True

        except Exception as e:

            import traceback
            traceback.print_exc()

            print(
                "[CONTEXT VALIDATION FAILED]",
                type(e).__name__,
                str(e)
            )

            ctx["ready"] = False
            ctx["error"] = str(e)

        if self.is_manual_mode:
            self._map_context_manual = ctx
        else:
            self._map_context_json = ctx

        return ctx


    def build_context_atomic(self):
        ctx = self._empty_context()

        if not self.road:
            return self._fail_ctx(ctx, "missing_road")

        vehicles_ctx, err = self._build_vehicle_contexts()
        if err:
            return self._fail_ctx(ctx, err)

        ctx_a, ctx_b = vehicles_ctx

        try:
            k1 = set(map(int, ctx_a["timeline"].keys()))
            k2 = set(map(int, ctx_b["timeline"].keys()))
            common = sorted(k1 & k2)

            if not common:
                return self._fail_ctx(ctx, "no_common_frames")

            ctx_a["timeline"] = {k: ctx_a["timeline"].get(k) for k in common}
            ctx_b["timeline"] = {k: ctx_b["timeline"].get(k) for k in common}

            _dbg("aligned V1 size", len(ctx_a["timeline"]))
            _dbg("aligned V2 size", len(ctx_b["timeline"]))

            pair = self._ctx_builder.build_pairwise(ctx_a, ctx_b)

            _dbg("pair timeline size BEFORE fill", len(pair.get("timeline", {})))

            pair_tl = pair.get("timeline", {})
            pair["timeline"] = {
                k: pair_tl.get(k, {
                    "distance": float("nan"),
                    "delta_s": float("nan"),
                    "delta_d": float("nan"),
                })
                for k in common
            }

        except Exception as e:
            return self._fail_ctx(ctx, f"pairwise_failed: {e}")

        if not pair or not pair.get("timeline"):
            return self._fail_ctx(ctx, "empty_pairwise")
        
        return self._build_pair_scenario_context(
            actor_a_id= self.v1_id_box.currentText() or "V1",
            actor_b_id= self.v2_id_box.currentText() or "V2",
            ctx_a=ctx_a,
            ctx_b=ctx_b,
            pair=pair,
            start_frame=common[0],
            end_frame=common[-1]
        )
    
    def build_semantic_context(self, ctx):
        """
        Enrich a map-based context with scenario-engine events and produce a
        normalized event timeline for indexing.

        Timing policy:
        - Event times are retained in their source unit.
        - Ratios are normalized using the observed event/timeline extent.
        - Ratios are clamped to [0, 1].
        - Raw start/end values remain available as start_t/end_t.

        Location policy:
        - Prefer the location attached directly to an event.
        - Fall back to features, interaction, and metrics.
        - Preserve from/to region for transition events.
        """
        ctx = ctx.copy()
        if not isinstance(ctx, dict) or not ctx:
            return {}

        ctx["ready"] = True

        def _finite_float(value, default=None):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default

            if not math.isfinite(value):
                return default

            return value

        def _first_present(*values):
            for value in values:
                if value is not None:
                    return value
            return None

        def _event_name(event):
            value = _first_present(
                event.get("type"),
                event.get("event"),
                event.get("name"),
                event.get("label"),
            )

            if isinstance(value, dict):
                value = _first_present(
                    value.get("type"),
                    value.get("event"),
                    value.get("name"),
                    value.get("label"),
                )

            return str(value) if value is not None else "unknown"

        duration = _finite_float(
            ctx.get("duration_seconds"),
            1.0,
        )
        duration = max(duration, 1e-6)

        # --------------------------------------------------
        # Engine enrichment
        # --------------------------------------------------

        ctx = build_engine_insight_context(
            context=ctx,
            duration=duration
        )

        rules = suggest_rules_combined(
            duration=duration,
            meta=ctx.get("meta"),
            context=ctx
        )

        timeline = (
            ctx.get(
                "event_timeline",
                []
            )
        )

        # --------------------------------------------------
        # Event detection
        # --------------------------------------------------


        valid_events = []

        for idx, event in enumerate(timeline):
            if not isinstance(event, dict):
                continue

            t0 = _finite_float(
                _first_present(
                    event.get("t_start"),
                    event.get("start_t"),
                    event.get("frame_start"),
                )
            )

            t1 = _finite_float(
                _first_present(
                    event.get("t_end"),
                    event.get("end_t"),
                    event.get("frame_end"),
                    t0,
                )
            )

            if t0 is None:
                continue

            if t1 is None:
                t1 = t0

            if t1 < t0:
                t0, t1 = t1, t0

            valid_events.append(
                (idx, event, t0, t1)
            )

        if not valid_events:
            ctx["event_timeline"] = []
            ctx["event_transitions"] = []
            ctx["event_stats"] = {}
            return ctx

        # Event detectors currently use a mixture of seconds and sample/frame
        # indices. Normalize against the actual observed extent to avoid ratios
        # above 1.0.
        min_time = min(item[2] for item in valid_events)
        max_time = max(item[3] for item in valid_events)
        observed_span = max_time - min_time

        if observed_span <= 1e-9:
            observed_span = duration

        observed_span = max(observed_span, 1e-6)

        normalized = []

        for event_id, event, t0, t1 in valid_events:
            metrics = event.get("metrics")
            if not isinstance(metrics, dict):
                metrics = {}

            features = event.get("features")
            if not isinstance(features, dict):
                features = {}

            interaction = event.get("interaction")
            if not isinstance(interaction, dict):
                interaction = {}

            location = event.get("location")
            if not isinstance(location, dict):
                location = {}

            participants = event.get("participants", [])
            if isinstance(participants, str):
                participants = [participants]
            elif not isinstance(participants, (list, tuple, set)):
                participants = []

            region = _first_present(
                event.get("region"),
                event.get("region_id"),
                location.get("region"),
                location.get("region_id"),
                features.get("region"),
                features.get("region_id"),
                interaction.get("region"),
                interaction.get("region_id"),
                metrics.get("region"),
                metrics.get("region_id"),
            )

            from_region = _first_present(
                event.get("from_region"),
                event.get("from_lane"),
                location.get("from_region"),
                location.get("from_lane"),
                features.get("from_region"),
                features.get("from_lane"),
                interaction.get("from_region"),
                interaction.get("from_lane"),
                metrics.get("from_region"),
                metrics.get("from_lane"),
            )

            to_region = _first_present(
                event.get("to_region"),
                event.get("to_lane"),
                location.get("to_region"),
                location.get("to_lane"),
                features.get("to_region"),
                features.get("to_lane"),
                interaction.get("to_region"),
                interaction.get("to_lane"),
                metrics.get("to_region"),
                metrics.get("to_lane"),
            )

            # A lane transition often has only from/to lane. Use the destination
            # as its primary location, while preserving both endpoints.
            if region is None:
                region = _first_present(
                    to_region,
                    from_region,
                )

            duration_event = max(0.0, t1 - t0)

            start_ratio = (t0 - min_time) / observed_span
            end_ratio = (t1 - min_time) / observed_span
            duration_ratio = duration_event / observed_span

            start_ratio = min(1.0, max(0.0, start_ratio))
            end_ratio = min(1.0, max(start_ratio, end_ratio))
            duration_ratio = min(1.0, max(0.0, duration_ratio))

            normalized.append(
                {
                    "event_id": int(event_id),
                    "event": _event_name(event),
                    "start_t": t0,
                    "end_t": t1,
                    "duration": duration_event,
                    "start_ratio": start_ratio,
                    "end_ratio": end_ratio,
                    "duration_ratio": duration_ratio,
                    "participants": sorted(
                        str(value)
                        for value in participants
                        if value is not None
                    ),
                    "region": region,
                    "from_region": from_region,
                    "to_region": to_region,
                    "features": features,
                    "interaction": interaction,
                    "metrics": metrics,
                }
            )

        normalized.sort(
            key=lambda item: (
                item["start_t"],
                item["end_t"],
                item["event"],
                item["event_id"],
            )
        )

        # --------------------------------------------------
        # Temporal transitions
        # --------------------------------------------------

        transitions = []

        for previous, current in zip(
            normalized[:-1],
            normalized[1:],
        ):
            delta_t = current["start_t"] - previous["end_t"]

            transitions.append(
                {
                    "src": previous["event"],
                    "dst": current["event"],
                    "src_event_id": previous["event_id"],
                    "dst_event_id": current["event_id"],
                    "delta_t": delta_t,
                    "delta_ratio": max(
                        0.0,
                        delta_t / observed_span,
                    ),
                    "overlaps": delta_t < 0.0,
                }
            )

        # --------------------------------------------------
        # Per-event statistics
        # --------------------------------------------------

        stats = {}

        for event in normalized:
            bucket = stats.setdefault(
                event["event"],
                {
                    "count": 0,
                    "durations": [],
                    "start_ratios": [],
                    "regions": {},
                },
            )

            bucket["count"] += 1
            bucket["durations"].append(event["duration"])
            bucket["start_ratios"].append(event["start_ratio"])

            region = event.get("region")
            if region is not None:
                key = str(region)
                bucket["regions"][key] = (
                    bucket["regions"].get(key, 0) + 1
                )

        for bucket in stats.values():
            durations = bucket.pop("durations")
            start_ratios = bucket.pop("start_ratios")

            bucket["mean_duration"] = (
                float(np.mean(durations))
                if durations
                else 0.0
            )
            bucket["max_duration"] = (
                float(np.max(durations))
                if durations
                else 0.0
            )
            bucket["mean_start_ratio"] = (
                float(np.mean(start_ratios))
                if start_ratios
                else 0.0
            )

        signals = (
            ctx.get("derived", {})
            .get("signals", {})
        )

        ctx["event_time_origin"] = min_time
        ctx["event_time_span"] = observed_span
        ctx["event_timeline"] = normalized
        ctx["event_transitions"] = transitions
        ctx["event_stats"] = stats
        if "scenario_descriptor" not in ctx.get("derived", {}):
            ctx.setdefault("derived", {})
            ctx["derived"]["scenario_descriptor"] = build_scenario_descriptor(
                ctx,
                rules
            )
        return ctx

    def _fail_ctx(
            self,
            ctx,
            err):

        ctx.update({

            "ready":
                False,

            "state":
                "failed",

            "error":
                err,

            "actors": {},

            "interaction_facts": {},

            "interaction_topology": {}
        })

        return ctx

    def _on_generate_clicked(self):
        if not self.is_manual_mode:
            return

        self.current_wp_vehicle = "V1"
        self._generate_wp_trajectory()

        self.current_wp_vehicle = "V2"
        self._generate_wp_trajectory()


    def _on_apply_clicked(self):
        if self.is_manual_mode:
            self.apply_manual_trajectory()
        else:
            self.apply_loaded_trajectories()
        print(
            "CTX READY:",
            self._map_context_json.get("ready")
        )

        print(
            "A timeline size:",
            len(
                self._map_context_json
                .get("actors", {})
                .get("A", {})
                .get("timeline", {})
            )
        )

        print(
            "B timeline size:",
            len(
                self._map_context_json
                .get("actors", {})
                .get("B", {})
                .get("timeline", {})
            )
        )

        print(
            "Interaction timeline size:",
            len(
                self._map_context_json
                .get(
                    "interaction_facts",
                    {}
                )
                .get(
                    "timeline",
                    {}
                )
            )
        )   

    def update_hover_coords(self, px: int, py: int):
        self._hover_px = (px, py)
        if self.road:
            wp = self.road.pixel_to_world(px, py)
            self._hover_wp = WorldPoint(*wp) if wp else None
        else:
            self._hover_wp = None

        self.update_status_panel()


    def new_project(self):
        self._invalidate_active_scenario("new_project")
        self._project = {"mask": None, "layout": None, "homography": None, "xodr": None, "openlabel": None}
        self._mask_path = self._layout_path = self._homog_path = self._xodr_path = self._openlabel_path = None

        self.log_info("New Project: Browse files for your project (you can skip any).")

        path, _ = QFileDialog.getOpenFileName(self, "Select Mask", "", "Mask Files (*.png *.tif *.tiff *.npy)")
        mask = path or None

        if mask:
            self.load_mask_from_path(mask)

        layout = self._load_with_dialog("Select Layout JSON", "JSON Files (*.json)")
        if layout:
            self.load_meta_from_path(layout)

        homog = self._load_with_dialog("Select Homography JSON", "JSON Files (*.json)")
        if homog:
            self.load_homography_from_path(homog)

        xodr = self._load_with_dialog("Select OpenDRIVE (xODR)", "OpenDRIVE (*.xodr)")
        if xodr:
            self.load_xodr_from_path(xodr)

        openlabel_path = self._load_with_dialog("Select OpenLABEL JSON (optional)", "JSON Files (*.json)")
        if openlabel_path:
            self.load_openlabel_from_path(openlabel_path)

        ret = QMessageBox.question(
            self, "Save Project",
            "Save this project (paths) to a .json file now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if ret == QMessageBox.StandardButton.Yes:
            self.save_project()

        self.update_status_panel()

    def open_project(self):
        self._invalidate_active_scenario("open_project")
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "Project (*.trace)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                proj = json.load(f)
        except Exception as e:
            self.log_warn("Open Project" + f"Failed to read project: {e}")
            return

        self._project = {k: proj.get(k) for k in ("mask","layout","homography","xodr","openlabel")}

        if self._project.get("mask"):
            self.load_mask_from_path(self._project["mask"])
        if self._project.get("layout"):
            self.load_meta_from_path(self._project["layout"])
        if self._project.get("homography"):
            self.load_homography_from_path(self._project["homography"])
        if self._project.get("xodr"):
            self.load_xodr_from_path(self._project["xodr"])
        if self._project.get("openlabel"):
            self.load_openlabel_from_path(self._project["openlabel"])

        self.log_info("Project: Project loaded.")
        self.update_status_panel()

    def save_project(self):
        proj = {
            "mask": self._mask_path,
            "layout": self._layout_path,
            "homography": self._homog_path,
            "xodr": self._xodr_path,
            "openlabel": self._openlabel_path
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.trace", "Project (*.trace)")
        if not path:
            return
        if not path.lower().endswith(".trace"):
            path += ".trace"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(proj, f, ensure_ascii=False, indent=2)
            self.log_info(f"Project: Project saved to {path}")
        except Exception:
            self.log_warn("Project: Failed to save project.")


    def _set_wp_mode(self, vehicle):
        if not self.is_manual_mode:
            return

        self.current_wp_vehicle = vehicle
        self.view.enable_waypoints(vehicle)

        self.ctrl_v1_btn.setStyleSheet("")
        self.ctrl_v2_btn.setStyleSheet("")

        if vehicle == "V1":
            self.ctrl_v1_btn.setStyleSheet(
                "background:#4CAF50;color:white;font-weight:bold;"
            )
        else:
            self.ctrl_v2_btn.setStyleSheet(
                "background:#4CAF50;color:white;font-weight:bold;"
            )

        self.log_info(f"Editing {vehicle} waypoints")
        self.update_status_panel()



    def _clear_all_waypoints(self):

        self.view.clear_waypoints()
        self.clear_all_trajectories()

        self.current_wp_vehicle = None
        self.view.active_vehicle = None

        self.update_status_panel()
        self.log_info("Waypoints and trajectories cleared.")


    def emit_all_vehicle_map_contexts(self):
        traj_map = self._get_active_traj_map()

        if not traj_map:
            print("[LaneMapViewer] No active trajectories.")
            return

        if not (self.road and self.view.meta and self.homography):
            print("[LaneMapViewer] Missing required map components.")
            return

        print(
            f"[LaneMapViewer] Emitting contexts mode={self.active_mode}"
        )

        ctx = self._build_normalize_validate_emit()
        self._engine_context = None
        self.map_context_ready.emit(ctx)

        if ctx.get("ready"):
            print("[LaneMapViewer] Emitted VALID context")
        else:
            print("[LaneMapViewer] Emitted INVALID context")


    def _emit_active_context(self):
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json

        if not ctx:
            return

        if not ctx.get("ready"):
            print("[Context] Skipped emit (not ready)")
            return

        self.map_context_ready.emit(ctx)

        print("[Context] emitted")


    def receive_engine_context(self, ctx: dict):
        if not isinstance(ctx, dict):
            print("[LaneMapViewer] Ignored invalid engine context")
            return

        if ctx.get("kind") != "engine_context":
            print(
                "[LaneMapViewer] "
                "Ignored non-engine context"
            )
            return
        self._engine_context = dict(ctx)
        print("[LaneMapViewer]  Engine context received")
        try:
            self._debug_save_context(ctx, tag="engine_enriched")
        except Exception:
            pass

    def _on_json_frame_range_changed(self):
        if self.active_mode != "json":
            return

        try:
            fstart = int(self.global_start_box.currentText())
            fend = int(self.global_end_box.currentText())
        except Exception:
            return

        if fstart > fend:
            fstart, fend = fend, fstart

        self.num_frames = max(1, fend - fstart + 1)

        try:
            self.fps = float(self.wp_fps.value())
        except Exception:
            self.fps = 30.0
        if self._traj_map_json:

            self._json_dirty = True

            self._invalidate_active_scenario(
                "json_frame_range_changed"
            )

            self.log_warn(
                "Frame range changed. "
                "Press Apply to reload trajectories."
            )

        self.update_vehicle_ids_for_duration()
        self.compute_and_emit_duration()
        self.update_status_panel()

    
    def _generate_wp_trajectory(self):
        v = self.current_wp_vehicle
        if not v:
            return self.log_warn("Select V1 or V2 first.")

        pts = self.view.manual_waypoints.get(v, [])
        if len(pts) < 2:
            return self.log_warn("Need at least 2 waypoints.")

        fps = int(self.wp_fps.value())
        duration = float(self.wp_duration.value())

        self.fps = fps
        self.num_frames = max(2, int(duration * fps))
        self.compute_and_emit_duration()

        traj = self._resample_waypoints_time_based(pts, duration, fps)
        self._manual_dirty = False
        px_pts = [(x, y) for _, x, y in traj]

        world_pts = self.px_to_world_polyline(px_pts)
        if not world_pts:
            return self.log_warn("World conversion failed.")

        if v not in ("V1", "V2"):
            return

        store = self._get_active_store()

        store["traj_map"][v] = traj
        store["world"][v] = world_pts
        self._rebuild_vehicle_context_cache()

        color = QColor("green") if v == "V1" else QColor("cyan")

        self.clear_traj_items(store["items"].get(v, []))
        store["items"][v] = self.view.draw_trajectory(px_pts, color)

        self._manual_traj_done[v] = True
        self._rebuild_hover_index()

        self._update_waypoint_visibility()
        self.log_info(f"{v} trajectory generated ({len(px_pts)} frames)")

        self.update_status_panel()
        self.polyline_committed.emit(v, json.dumps({
            "px": px_pts,
            "world": world_pts
        }))


    def _try_init_road(self):
        if self.view.mask is None or self.view.meta is None:
            return

        if not self.road:
            self.road = RoadInfrastructure(self._layout_path)

        self.road.attach_mask_meta(
            mask=self.view.mask,
            meta=self.view.meta
        )
        self._rebuild_context_builder()       

    def _attach_homography_to_road(self):
        if self.road and self.homography:
            self.road.set_homography(
                self.homography["H"],
                self.homography["Hinv"]
            )

    def load_mask_from_path(self, path: str):
        try:
            self.view.load_mask(path)
            self._mask_path = path
            self._project["mask"] = path
            if self.view.mask is not None:
                self.mask_loaded.emit(self.view.mask)
            self._try_init_road()
            self._attach_homography_to_road()             
        except Exception as e:
            self.log_warn(f"Failed to load mask: {e}")

    def load_meta_from_path(self, path: str):
        try:
            meta = self.view.load_meta(path)  
            self.view.meta = meta
            try:
                self.road = RoadInfrastructure(path)
                if self.view.meta is None:
                    self.log_warn("Meta failed to load")
                    return
                self.road.attach_mask_meta(
                    mask=self.view.mask,
                    meta=self.view.meta
                )
                self.road.populate_centerlines(region_type=["lane", "connecting_lane"])
                self._centerline_geom_cache.clear()
                self._build_semantic_lane_roles()
                self.log_info("RoadInfrastructure initialized")
            except Exception as e:
                self.log_warn(f"RoadInfrastructure failed: {e}")

            self._layout_path = path
            self._project["layout"] = path
            if self.view.meta is not None:
                self.meta_loaded.emit(self.view.meta)

            if not self.road or not self.road.regions:
                self.log_warn("RoadInfrastructure has no regions")
                return

            if self.road:
                self.view.draw_centerlines(self.road)

            self._attach_homography_to_road()  # in case homography was loaded before layout
            self._rebuild_context_builder()
        except Exception as e:
            self.log_warn(f"Failed to load layout: {e}")

    def load_homography_from_path(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            H = data.get("transformation_matrix")
            Hinv = data.get("inverse_matrix")

            if H is None or Hinv is None:
                raise ValueError("Homography JSON must contain transformation_matrix and inverse_matrix")

            H = np.asarray(H, dtype=float)
            Hinv = np.asarray(Hinv, dtype=float)

            if H.shape != (3, 3) or Hinv.shape != (3, 3):
                raise ValueError("Homography matrices must be 3×3")
            if not (np.isfinite(H).all() and np.isfinite(Hinv).all()):
                raise ValueError("Homography contains invalid values")
           
            self.homography = {
                "H": H,
                "Hinv": Hinv,
                "raw": data,
            }

            self._homog_path = path
            self._project["homography"] = path

            self.log_info(f"Homography loaded successfully from {path}")
            if self.road and self.homography:
                self.road.set_homography(
                    self.homography["H"],
                    self.homography["Hinv"]
                )            

        except Exception as e:
            self.homography = None
            self.log_warn(f"Failed to load homography: {e}")


    def load_xodr_from_path(self, path: str):
        self._xodr_path = path
        self._project["xodr"] = path

        self.xodr_loaded.emit(path)
        self.xodr_changed.emit(path)

        self.log_info(f"OpenDRIVE: {path}")

    def load_openlabel_from_path(self, path: str):
        try:
            frames, traj_map, fmin, fmax = load_openlabel(path)
            self.openlabel_frames = frames
            self.openlabel_traj_map = traj_map
            self._builder_traj_cache.clear()
            self._openlabel_vehicle_ctx_cache.clear()
            self.openlabel_fmin = fmin
            self.openlabel_fmax = fmax

            self.num_frames = max(0, self.openlabel_fmax - self.openlabel_fmin + 1)
            try:
                self.fps = float(self.wp_fps.value())
            except Exception:
                self.fps = 30.0
            self.compute_and_emit_duration()

            self._openlabel_path = path
            self._project["openlabel"] = path
            fr_list = [str(f) for f in range(fmin, fmax+1)]
            self.global_start_box.blockSignals(True)
            self.global_end_box.blockSignals(True)
            self.global_start_box.clear(); self.global_start_box.addItems(fr_list)
            self.global_end_box.clear(); self.global_end_box.addItems(fr_list)
            self.global_start_box.setCurrentText(str(fmin))
            self.global_end_box.setCurrentText(str(fmax))
            self.global_start_box.blockSignals(False)
            self.global_end_box.blockSignals(False)
            self.update_vehicle_ids_for_duration()
            QTimer.singleShot(0, self.update_vehicle_ids_for_duration)
        except Exception as e:
            self.log_warn(f"Failed to load OpenLABEL: {e}")

    def _hide_all_trajectories(self):
        for mode in self._traj_items.values():
            for veh_items in mode.values():
                for it in veh_items:
                    if it:
                        it.setVisible(False)


    def _show_active_mode_trajectories(self):
        mode_items = self._traj_items.get(self.active_mode, {})

        for veh_items in mode_items.values():
            for it in veh_items:
                if it:
                    it.setVisible(True)
  
    def _resample_waypoints_time_based(self, pts_px, duration_s, fps):
        """
        Convert waypoint polyline into OpenLABEL-like trajectory:
        [(frame, x_px, y_px), ...]
        """
        if len(pts_px) < 2:
            return []

        total_frames = max(2, int(round(duration_s * fps)))

        arr = np.asarray(pts_px, dtype=float)
        seglen = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total_len = cum[-1]

        if total_len < 1e-6:
            frames = [(i, arr[0][0], arr[0][1]) for i in range(total_frames)]
            return frames

        target = np.linspace(0, total_len, total_frames)
        xs = np.interp(target, cum, arr[:, 0])
        ys = np.interp(target, cum, arr[:, 1])

        return [(i, float(xs[i]), float(ys[i])) for i in range(total_frames)]


    def _group_consecutive(self, frames: List[int], min_len: int) -> List[Tuple[int,int]]:
        """Group sorted frame indices into [start,end] windows with min length."""
        out = []
        if not frames:
            return out
        start = prev = frames[0]
        for f in frames[1:]:
            if f == prev + 1:
                prev = f
                continue
            if (prev - start + 1) >= min_len:
                out.append((start, prev))
            start = prev = f
        if (prev - start + 1) >= min_len:
            out.append((start, prev))
        return out

    def load_homography_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Homography JSON", "", "JSON Files (*.json)"
        )
        if not path:
            return
        self.load_homography_from_path(path)

   
    def _load_with_dialog(self, title, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path or None


    def load_mask(self):
        path = self._load_with_dialog(
            "Open Mask",
            "Mask Files (*.png *.tif *.tiff *.npy)"
        )
        if path:
            self.load_mask_from_path(path)


    def load_xodr(self):
        path,_ = QFileDialog.getOpenFileName(self, "Open xODR","", "OpenDRIVE (*.xodr)")
        if not path:
            return
        self._xodr_path = path
        self._project["xodr"] = path
        self.xodr_loaded.emit(path)
        self.log_info(f"xODR Loaded Using: {path}")

    # ---------------------------- Status/Hover ---------------------------

    def log_info(self, text: str):
        self._append_log("INFO", text)

    def log_warn(self, text: str):
        self._append_log("WARN", text)

    def log_error(self, text: str):
        self._append_log("ERROR", text)

    def _append_log(self, level: str, text: str):
        if not hasattr(self, "log_view"):
            return
        self.log_view.append(f"[{level}] {text}")
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())


    def update_status_panel(self):

        if self.is_manual_mode:

            ready = (
                self._manual_traj_done["V1"]
                and self._manual_traj_done["V2"]
            )

            state = (
                "REGENERATE REQUIRED"
                if self._manual_dirty
                else (
                    "READY"
                    if ready
                    else "INCOMPLETE"
                )
            )

            txt = (
                f"<b>MANUAL</b> | "
                f"State: <b>{state}</b><br>"
                f"Duration: {self.wp_duration.value():.1f}s | "
                f"FPS: {self.wp_fps.value()}<br>"
                f"V1 Frames: "
                f"{len(self._traj_map_manual.get('V1', []))} | "
                f"V2 Frames: "
                f"{len(self._traj_map_manual.get('V2', []))}"
            )

        else:

            state = (
                "APPLY REQUIRED"
                if self._json_dirty
                else (
                    "READY"
                    if self._traj_map_json
                    else "NO SCENARIO"
                )
            )

            txt = (
                f"<b>JSON</b> | "
                f"State: <b>{state}</b><br>"
                f"Frames: "
                f"{self.global_start_box.currentText()} → "
                f"{self.global_end_box.currentText()}<br>"
                f"V1={self.v1_id_box.currentText()} | "
                f"V2={self.v2_id_box.currentText()}"
            )

        self.status_label.setText(txt)


    def update_hover_info(self, rid, md):
        if rid is None:
            line1 = "Outside image"
        elif rid == 0:
            line1 = "Non-drivable"
        elif not md:
            line1 = f"Region {rid} (no metadata)"
        else:
            t = md.get("type", "unknown")

            if t in ("lane", "connecting_lane"):
                semantic = self._get_lane_semantic_debug(
                    rid
                )

                sem_txt = ",".join(
                    semantic.get("roles", [])
                )

                if sem_txt:
                    sem_txt = f" | SEM:{sem_txt}"
                else:
                    sem_txt = ""                
                adj = md.get("adjacent", [])
                suc = md.get("successors", [])
                pre = md.get("predecessors", [])

                line1 = (
                    f"R{rid} | {t}"
                    f"{sem_txt} | "
                    f"Adjacent:{adj if adj else '-'} | "
                    f"Successors:{suc if suc else '-'} | "
                    f"Predecessors:{pre if pre else '-'}"
                )

            elif t == "overlap":
                members = md.get("members", [])
                line1 = (
                    f"R{rid} | overlap | "
                    f"Members:{members if members else '-'}"
                )

            else:
                line1 = f"R{rid} | {t}"

        # ----------------------------
        # LINE 2 — COORDINATES
        # ----------------------------
        px_txt = "px: -"
        world_txt = "w: -"
        sd_txt = "s:- d:-"

        if self._hover_px:
            x, y = self._hover_px
            px_txt = f"px:({x},{y})"

        if self._hover_wp:
            wp = self._hover_wp
            world_txt = f"w:({wp.x:.2f},{wp.y:.2f})"

        if self._hover_wp and self.road and rid not in (None, 0):
            try:
                sd = self.road.project_world_to_frenet(
                    rid, self._hover_wp.x, self._hover_wp.y
                )
                if sd:
                    sd_txt = f"s:{sd['s']:.2f} d:{sd['d']:.2f}"
            except Exception:
                pass

        line2 = f"{px_txt} | {world_txt} | {sd_txt}"
        line3 = ""

        hit = getattr(self.view, "_hover_traj_hit", None)

        if hit:
            vid, idx, dist_px = hit

            v_ins = self._get_hover_vehicle_insight(vid, idx)
            pair_ins = self._get_hover_pair_insight(idx)

            parts = [f"[{vid} @ t={idx}]"]

            if v_ins:
                parts.append(f"R:{v_ins['region']}")
                parts.append(f"s:{v_ins['s']:.2f}" if isinstance(v_ins.get("s"), (int, float)) else "s:-")
                parts.append(f"d:{v_ins['d']:.2f}" if isinstance(v_ins.get("d"), (int, float)) else "d:-")


            if pair_ins:
                if isinstance(pair_ins.get("dist"), (int, float)):
                    parts.append(f"dist:{pair_ins['dist']:.2f}")
                else:
                    parts.append("dist:-")

                if isinstance(pair_ins.get("dd"), (int, float)):
                    parts.append(f"Δd:{pair_ins['dd']:.2f}")
                else:
                    parts.append("Δd:-")


            parts.append(f"px_dist:{dist_px:.1f}")

            line3 = " | ".join(parts)

        text_full = f"{line1}\n{line2}"

        if line3:
            text_full += f"\n{line3}"

        self.info_label.setText("<pre>" + text_full + "</pre>")

    def clear_traj_items(self, items):
        for it in items:
            if it:
                self.view.scene.removeItem(it)


    def clear_all_trajectories(self):
        store = self._get_active_store()

        for v in ("V1", "V2"):
            self.clear_traj_items(store["items"].get(v, []))
            store["items"][v] = []

    def px_to_world_polyline(self, pts_px):
        if not (self.road and self.homography and pts_px):
            return None

        world = []

        for pt in pts_px:
            if len(pt) == 3:
                _, x, y = pt
            else:
                x, y = pt
            try:
                wp = self.road.pixel_to_world(x, y)
                if wp and len(wp) >= 2:
                    if np.isfinite(wp[0]) and np.isfinite(wp[1]):
                        world.append((float(wp[0]), float(wp[1])))
            except Exception as e:
                print(f"[px_to_world] Failed: {e}")

        return world if len(world) >= 2 else None


    def _ensure_start_le_end(self):
        try:
            s = int(self.global_start_box.currentText())
            e = int(self.global_end_box.currentText())
        except:
            return
        if s > e:
            self.global_start_box.blockSignals(True)
            self.global_end_box.blockSignals(True)
            self.global_start_box.setCurrentText(str(e))
            self.global_end_box.setCurrentText(str(s))
            self.global_start_box.blockSignals(False)
            self.global_end_box.blockSignals(False)


    def update_vehicle_ids_for_duration(self):
        if not self.openlabel_traj_map:
            return

        self._ensure_start_le_end()

        try:
            fstart = int(self.global_start_box.currentText())
            fend = int(self.global_end_box.currentText())
        except Exception:
            return

        require_both = True

        valid_ids = []

        for oid, traj in self.openlabel_traj_map.items():
            frames = {fr for (fr, _, _) in traj}

            if require_both:
                if fstart in frames and fend in frames:
                    valid_ids.append(oid)
            else:
                if any(fstart <= fr <= fend for fr in frames):
                    valid_ids.append(oid)

        valid_ids = sorted(valid_ids, key=lambda s: str(s))
        prev_v1 = self.v1_id_box.currentText()
        prev_v2 = self.v2_id_box.currentText()

        self.v1_id_box.blockSignals(True)
        self.v2_id_box.blockSignals(True)

        self.v1_id_box.clear()
        self.v2_id_box.clear()

        self.v1_id_box.addItems(valid_ids)
        self.v2_id_box.addItems(valid_ids)

        if prev_v1 in valid_ids:
            self.v1_id_box.setCurrentText(prev_v1)
        if prev_v2 in valid_ids:
            self.v2_id_box.setCurrentText(prev_v2)

        if (
            self.v1_id_box.currentText() == self.v2_id_box.currentText()
            and len(valid_ids) > 1
        ):
            self.v2_id_box.setCurrentIndex(1)

        self.v1_id_box.blockSignals(False)
        self.v2_id_box.blockSignals(False)
        self.json_status.setText(f"{len(valid_ids)} valid vehicles in range")
        self.update_status_panel()        


    def apply_loaded_trajectories(self):
        if self.active_mode != "json" or not self.openlabel_traj_map:
            return self.log_warn("Load JSON first.")

        if not self.road:
            return self.log_warn("Load layout (RoadInfrastructure) first.")

        if not self.homography:
            return self.log_warn("Load homography first.")

        try:
            fstart = int(self.global_start_box.currentText())
            fend = int(self.global_end_box.currentText())
        except:
            return self.log_warn("Invalid frame selection.")

        self._map_context_json = self._empty_context()
        self._engine_context = None

        if fstart > fend:
            fstart, fend = fend, fstart


        v1_id = self.v1_id_box.currentText()
        v2_id = self.v2_id_box.currentText()

        if not v1_id or not v2_id or v1_id == v2_id:
            return self.log_warn("Select two different vehicles.")

        px1 = [(fr, x, y) for (fr, x, y) in self.openlabel_traj_map[v1_id] if fstart <= fr <= fend]
        px2 = [(fr, x, y) for (fr, x, y) in self.openlabel_traj_map[v2_id] if fstart <= fr <= fend]

        if len(px1) < 2 or len(px2) < 2:
            return self.log_warn("Too few trajectory points.")

        w1 = self.px_to_world_polyline(px1)
        w2 = self.px_to_world_polyline(px2)

        if not w1 or not w2:
            return self.log_warn("World conversion failed (check homography + mask coverage).")

        store = self._get_active_store()

        traj1_dict = {fr: (x, y) for (fr, x, y) in self.openlabel_traj_map[v1_id]}
        traj2_dict = {fr: (x, y) for (fr, x, y) in self.openlabel_traj_map[v2_id]}

        frames = list(range(fstart, fend + 1))

        traj_v1 = []
        traj_v2 = []

        for fr in frames:

            if fr in traj1_dict and fr in traj2_dict:

                x1, y1 = traj1_dict[fr]
                x2, y2 = traj2_dict[fr]

                traj_v1.append((fr, x1, y1))
                traj_v2.append((fr, x2, y2))

        try:
            store["traj_map"].update({
                "V1": traj_v1,
                "V2": traj_v2
            })

            store["world"].update({
                "V1": w1,
                "V2": w2
            })
            self._rebuild_vehicle_context_cache()

            for v, pts, color in [
                ("V1", px1, QColor("green")),
                ("V2", px2, QColor("cyan"))
            ]:
                pts_xy = [(x, y) for (_, x, y) in pts]

                self.clear_traj_items(store["items"].get(v, []))

                store["items"][v] = self.view.draw_trajectory(pts_xy, color)


            if self.scenario_gui and hasattr(self.scenario_gui, "reset_vehicle_contexts"):
                self.scenario_gui.reset_vehicle_contexts()
            self._json_dirty = False
            self.compute_and_emit_duration()
            self._rebuild_hover_index()
            self.emit_all_vehicle_map_contexts()

            self.log_info(f"Applied trajectories: V1={v1_id}, V2={v2_id}")

        except Exception as e:
            self.log_error(f"Apply failed: {e}")

    def _filter_by_timing(
        self,
        seq,
        event_name,
        min_dur=None,
        max_dur=None
    ):

        result = []

        for e in seq:

            if e["event"] != event_name:
                continue

            d = e["duration_ratio"]

            if (
                min_dur is not None
                and d < min_dur
            ):
                continue

            if (
                max_dur is not None
                and d > max_dur
            ):
                continue

            result.append(e)

        return result

    def _candidate_signature(
            self,
            track):

        routes = track.get(
            "route_signatures",
            []
        )

        topo = " | ".join(
            "→".join(r)
            for r in routes
        )

        seq = " -> ".join(

            ev.get(
                "event_token",
                ev.get(
                    "event",
                    "?"
                )
            )

            for ev in track.get(
                "event_sequence",
                []
            )
        )

        return (
            f"{topo}\n{seq}"
        )

    def _format_track_event_line(self, track):
        """
        One-line event timeline for the candidate list.

        lane_change@0.5s(0.2s) 12>13 | +0.4s waiting@0.9s(2.3s)
        | +4.9s proximity@5.8s(1.6s) | +1.5s waiting@7.3s(4.9s)

        @t    event start (engine time, seconds)
        (x s) event duration (recovered: duration_ratio * span)
        +x s  start-to-start gap to the previous event
        a>b   from_region > to_region (lane_change events)
        """
        seq = [
            e for e in (track.get("event_sequence", []) or [])
            if isinstance(e, dict)
        ]

        if not seq:
            return "no events"

        # Ratios are normalized against the observed span, which the
        # track does not store. Recover it exactly:
        #   span = (t_j - t_i) / (start_ratio_j - start_ratio_i)
        span = None
        for i in range(len(seq)):
            ri = seq[i].get("start_ratio")
            ti = seq[i].get("_start")
            if not (isinstance(ri, (int, float)) and isinstance(ti, (int, float))):
                continue
            for j in range(i + 1, len(seq)):
                rj = seq[j].get("start_ratio")
                tj = seq[j].get("_start")
                if not (isinstance(rj, (int, float)) and isinstance(tj, (int, float))):
                    continue
                dr = rj - ri
                if dr > 1e-9:
                    span = (tj - ti) / dr
                    break
            if span is not None:
                break

        parts = []
        prev_t = None

        for e in seq:
            t0 = e.get("_start")
            if not isinstance(t0, (int, float)):
                continue

            name = str(e.get("event", "?"))

            dur = e.get("duration_ratio")
            if isinstance(dur, (int, float)) and span:
                dur_txt = f"({dur * span:.1f}s)"
            elif isinstance(dur, (int, float)):
                dur_txt = f"({dur * 100:.0f}%)"
            else:
                dur_txt = ""

            region_txt = ""
            fr = e.get("from_region")
            to = e.get("to_region")
            if fr is not None and to is not None:
                region_txt = f" {fr}>{to}"

            token = f"{name}@{t0:.1f}s{dur_txt}{region_txt}"

            if prev_t is not None:
                token = f"+{t0 - prev_t:.1f}s {token}"

            parts.append(token)
            prev_t = t0

        return " | ".join(parts)

    def find_similar_scenarios(self):
        """
        If 'Use precomputed index' is enabled and a precompute exists:
        - If NO base scenario is present (no rules): propose candidate (v1, v2, [start,end]) windows from index.
        - If a base scenario exists: restrict the expensive similarity evaluation to candidate windows only.
        Otherwise, fallback to the original exhaustive search (unchanged behavior).
        """
        use_pre = hasattr(self, "chk_use_precomp") and self.chk_use_precomp and self.chk_use_precomp.isChecked()
        have_index = bool(getattr(self, "_precomp", None) and self._precomp.get("pair_windows"))

        # ---------- Case A: Use precomp suggestions without a base scenario ----------
        base_ctx = (
            getattr(
                self.scenario_gui,
                "_engine_context",
                None
            )
            or {}
        )

        query_sequence = self._build_query_sequence()

        if not query_sequence:
            wins = self._precomp.get("pair_windows",[])
            if not wins:
                self.log_info("No candidate windows in index. Try Precompute again.")
                return

            # Score windows: prioritize lane consistency, then short TTC, then short distance
            def wscore(w):
                s_lane = w.get("shared_lane_fraction", 0.0)
                inv_ttc = 0.0 if not (w.get("min_ttc", float("inf")) > 0) else 1.0 / w.get("min_ttc", float("inf"))
                inv_dist = 0.0 if not (w.get("min_dist", 1e9) > 0) else 1.0 / w.get("min_dist", 1e9)
                return (0.5 * s_lane) + (0.3 * inv_ttc) + (0.2 * inv_dist)

            top = sorted(wins, key=wscore, reverse=True)[:50]
            lines = []
            for w in top:
                mt = w['min_ttc']
                ttc_s = "inf" if mt == float("inf") else f"{mt:.2f}"
                lines.append(
                    f"V1={w['v1']}  V2={w['v2']}  frames=[{w['start']}, {w['end']}], "
                    f"min_dist={w['min_dist']:.1f}, min_ttc={ttc_s}, same_lane={w['shared_lane_fraction']:.2f}"
                )
            msg = "Candidate (v1, v2, [start,end]) windows from index:\n\n" + "\n".join(lines)
            self.log_info("Suggested Candidate Windows" + msg)
            return

        # ---------- Guardrails (mode-aware) ----------
        #
        if self.active_mode == "json":
            if not self.openlabel_traj_map or not self.openlabel_frames:
                self.log_warn("Load OpenLABEL JSON first.")
                return
        if not self.scenario_gui:
            self.log_warn(
                "Scenario GUI not connected."
            )
            return

        # --- Target signature from ScenarioRuleGUI (base) ---
        base_ctx = (
            getattr(self.scenario_gui, "_engine_context", None)
            or {}
        )

        try:
            fstart_cur = int(self.global_start_box.currentText())
            fend_cur = int(self.global_end_box.currentText())
            if fstart_cur > fend_cur:
                fstart_cur, fend_cur = fend_cur, fstart_cur
        except Exception:
            fstart_cur = None
            fend_cur = None


        # ---------- Case B: Base scenario present & precomp enabled: evaluate only candidate windows ----------
        db = getattr(
            self,
            "semantic_db",
            None
        )

        if db is None:

            db_path = getattr(
                self,
                "_loaded_db_path",
                None
            )

            if (
                db_path
                and
                os.path.exists(db_path)
            ):
                try:
                    self.load_semantic_database(
                        db_path
                    )
                    db = self.semantic_db
                except Exception:
                    db = None

        if db is None:

            self.log_warn(
                "Build an Event Timeline first."
            )

            return

        query_sequence = self._build_query_sequence()

        if not query_sequence:
            if self.is_manual_mode:
                self.log_warn(
                    "Manual scenario has no event sequence. "
                    "Generate V1+V2 trajectories and press Apply first."
                )
            else:
                self.log_warn(
                    "Current scenario has no event sequence."
                )
            return


        tracks = db.get(
            "tracks",
            []
        )

        print("\n=== QUERY SEQUENCE ===")
        print("len =", len(query_sequence))

        for e in query_sequence:
            print(
                e["event"],
                e["participants"]
            )

        candidate_ids = (
            self._candidate_track_indices(
                query_sequence
            )
        )
        print(
            "candidate_ids =",
            len(candidate_ids)
        )
        query_events = {
            e["event"]
            for e in query_sequence
        }

        results = []

        for idx in candidate_ids:

            track = tracks[idx]

            cand_sequence = track.get(
                "event_sequence",
                []
            )

            if not cand_sequence:
                continue

            fp = set(
                track.get(
                    "event_fingerprint",
                    []
                )
            )

            if not fp:
                # pair tracks do not store event_fingerprint -
                # derive it from the sequence (raw names, matching
                # the query side)
                fp = {
                    e["event"]
                    for e in cand_sequence
                    if isinstance(e, dict)
                }

            if (
                len(
                    query_events & fp
                )
                < 1
            ):
                continue

            topology_score = 0.0

            try:

                descriptor = (
                    self.scenario_gui
                    ._engine_context
                    .get("derived", {})
                    .get("scenario_descriptor", {})
                )

                q_sig = (
                    descriptor
                    .get("search_index", {})
                    .get("semantic_signature", {})
                )

                c_sig = (
                    track.get("semantic_signature", {})
                )

                q_routes = [
                    list(x)
                    for x in q_sig.get(
                        "route_signatures",
                        []
                    )
                ]

                c_routes = [
                    list(x)
                    for x in c_sig.get(
                        "route_signatures",
                        []
                    )
                ]

                route_scores = []

                for qr in q_routes:

                    best = 0.0

                    for cr in c_routes:

                        best = max(
                            best,
                            self._sequence_similarity(
                                qr,
                                cr
                            )
                        )

                    route_scores.append(best)

                topology_score = (
                    sum(route_scores)
                    /
                    max(len(route_scores), 1)
                )

            except Exception:
                topology_score = 0.0            

            event_score = (
                self._event_similarity(
                    query_sequence,
                    cand_sequence
                )
            )

            score = (
                0.60 * event_score
                +
                0.40 * topology_score
            )

            if score < 0.15:
                continue

            results.append({

                "score": score,

                "scope": track.get(
                    "scope"
                ),

                "track": track
            })

        unique = {}

        for r in results:

            track = r["track"]
            route_sig = tuple(

                tuple(r)

                for r in track.get(
                    "route_signatures",
                    []
                )
            )

            sig = (

                route_sig,

                tuple(
                    e["event_token"]
                    for e in track.get(
                        "event_sequence",
                        []
                    )
                )
            )

            if (
                sig not in unique
                or
                r["score"]
                >
                unique[sig]["score"]
            ):
                unique[sig] = r

        results = list(unique.values())

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        self._last_similarity_results = results[:50]

        self.similar_list.clear()

        for idx, r in enumerate(
            self._last_similarity_results,
            start=1
        ):

            track = r["track"]

            seq = self._candidate_signature(
                track
            )

            it = track.get(
                "interaction_topology",
                {}
            )

            actor_ids = track.get("actor_ids", []) or []

            if track.get("scope") == "pair" and len(actor_ids) >= 2:
                actors_txt = f"A={actor_ids[0]} B={actor_ids[1]}"
            elif actor_ids:
                actors_txt = f"A={actor_ids[0]} (vehicle)"
            else:
                actors_txt = "A=- B=-"

            # vehicle tracks store first_frame/last_frame,
            # pair tracks store start_frame/end_frame
            fr_start = track.get(
                "start_frame",
                track.get("first_frame")
            )
            fr_end = track.get(
                "end_frame",
                track.get("last_frame")
            )

            item_text = (
                f"{idx}. {r['score']:.3f} | "
                f"{actors_txt} | "
                f"J={it.get('shared_junction_count', 0)} "
                f"M={len(it.get('common_merge_points', []))} "
                f"D={len(it.get('common_diverge_points', []))} | "
                f"fr {fr_start}-{fr_end} "
                f"({track.get('frame_count', '?')}f)\n"
                f"{seq}\n"
                f"{self._format_track_event_line(track)}"
            )

            item = QListWidgetItem(
                item_text
            )

            item.setData(
                Qt.ItemDataRole.UserRole,
                idx - 1
            )

            self.similar_list.addItem(
                item
            )

        self.log_info(
            f"{len(self._last_similarity_results)} "
            f"candidate matches found."
        )
        return



    def load_semantic_database(self, path):

        with open(path, "r", encoding="utf-8") as f:
            self.semantic_db = json.load(f)

        tracks = self.semantic_db.get(
            "tracks",
            []
        )

        event_index = {}
        transition_index = {}

        for idx, track in enumerate(tracks):
            seq = track.get(
                "event_sequence",
                []
            )

            events = [
                e.get(
                    "event",
                    "?"
                )
                for e in seq
                if isinstance(e, dict)
            ]

            for ev in set(events):

                event_index.setdefault(
                    ev,
                    []
                ).append(idx)

            for a, b in zip(
                events[:-1],
                events[1:]
            ):

                transition_index.setdefault(
                    f"{a}>{b}",
                    []
                ).append(idx)

        self.semantic_db["_event_index"] = event_index
        self.semantic_db["_transition_index"] = transition_index

        self._loaded_db_path = path

        self.log_info(
            f"Loaded event database "
            f"({len(tracks)} tracks)"
        )

    def load_semantic_database_dialog(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Event Sequence DB",
            "",
            "JSON (*.json)"
        )

        if not path:
            return

        self.load_semantic_database(path)

    def _sequence_similarity(
        self,
        seq_a,
        seq_b
    ):

        if not seq_a or not seq_b:
            return 0.0

        m = len(seq_a)
        n = len(seq_b)

        prev = [0] * (n + 1)

        for i in range(m):

            curr = [0] * (n + 1)

            ai = seq_a[i]

            for j in range(n):

                if ai == seq_b[j]:

                    curr[j + 1] = (
                        prev[j] + 1
                    )

                else:

                    curr[j + 1] = max(
                        curr[j],
                        prev[j + 1]
                    )

            prev = curr

        lcs = prev[n]

        return (
            2.0 * lcs
            /
            (m + n)
        )

    def _event_similarity(self, query, candidate):

        if not query or not candidate:
            return 0.0

        q = [
            e["event"]
            for e in query
        ]

        c = [
            e["event"]
            for e in candidate
        ]

        lcs_score = self._sequence_similarity(q, c)

        q_set = set(q)
        c_set = set(c)

        jaccard = (
            len(q_set & c_set)
            /
            max(len(q_set | c_set), 1)
        )

        query_is_subsequence = (
            lcs_score >=
            len(q) / max(len(c), len(q))
        )

        subsequence_bonus = (
            1.0 if query_is_subsequence else 0.0
        )

        return (
            0.50 * lcs_score
            + 0.20 * jaccard
            + 0.30 * subsequence_bonus
        )


    def _build_query_sequence(self):

        if (
            not self.scenario_gui
            or
            not getattr(
                self.scenario_gui,
                "_engine_context",
                None
            )
        ):
            print("NO ENGINE CONTEXT")
            return []

        ctx = self.scenario_gui._engine_context

        print(
            "timeline size =",
            len(
                ctx.get(
                    "event_timeline",
                    []
                )
            )
        )

        events = (
            ctx.get("event_timeline")
            or []
        )

        if not events:
            print("NO EVENTS")
            return []

        seq, _, _ = (
            build_searchable_event_sequence(
                events
            )
        )

        print(
            "query events:",
            [e["event"] for e in seq]
        )

        return seq


    def _candidate_track_indices(
        self,
        query_sequence
    ):

        db = getattr(
            self,
            "semantic_db",
            None
        )

        if not db:
            return []

        event_index = db.get(
            "_event_index",
            {}
        )

        transition_index = db.get(
            "_transition_index",
            {}
        )

        scores = {}

        #
        # Index convention (must match load_semantic_database):
        #   _event_index      -> keyed by RAW event name ("waiting")
        #   _transition_index -> keyed by token pairs
        #                        ("actorA:waiting>actorA,actorB:proximity")
        #
        query_events = {
            e["event"]
            for e in query_sequence
        }

        query_transitions = {
            f"{a['event_token']}>{b['event_token']}"
            for a, b in zip(
                query_sequence[:-1],
                query_sequence[1:]
            )
        }

        for ev in query_events:

            matches = event_index.get(
                ev,
                []
            )

            print(
                ev,
                len(matches)
            )
            rarity = max(
                1.0,
                len(matches)
            )

            weight = 100.0 / rarity

            for idx in matches:

                scores[idx] = (
                    scores.get(idx, 0.0)
                    + weight
                )

        for tr in query_transitions:

            matches = transition_index.get(
                tr,
                []
            )

            rarity = max(
                1.0,
                len(matches)
            )

            weight = 200.0 / rarity

            for idx in matches:

                scores[idx] = (
                    scores.get(idx, 0.0)
                    + weight
                )

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            idx
            for idx, score
            in ranked
            if score > 0
        ][:200]

    def _ctx_events(self, ctx):

        return [
            {
                "type": e.get("type"),
                "t_start": e.get("t_start"),
                "t_end": e.get("t_end")
            }
            for e in (ctx.get("event_timeline") or [])
        ]    

    def _write_track(self, f, track, first_track):

        if not first_track:
            f.write(",\n")

        json.dump(
            track,
            f,
            ensure_ascii=False
        )

        return False    


    def _get_cached_builder_traj(self, oid, traj):
        cached = self._builder_traj_cache.get(oid)

        if cached is not None:
            return cached

        result = self._traj_to_builder_format(traj)

        self._builder_traj_cache[oid] = result

        return result

    def _build_route_semantics(
            self,
            timeline):

        route = []
        transitions = []

        prev_lane = None

        for frame_id in sorted(timeline):

            sample = timeline.get(frame_id)

            if not sample:
                continue

            lane = sample.get(
                "region_id"
            )

            if lane is None:
                continue

            lane = str(lane)

            if prev_lane is None:

                route.append(lane)
                prev_lane = lane
                continue

            if lane == prev_lane:
                continue

            meta_prev = self.road.meta.get(
                prev_lane,
                {}
            )

            succ = {
                str(x)
                for x in meta_prev.get(
                    "successors",
                    []
                )
            }

            adj = {
                str(x)
                for x in meta_prev.get(
                    "adjacent",
                    []
                )
            }

            if lane in succ:

                transitions.append(
                    "DIVERGE"
                    if len(succ) > 1
                    else "FOLLOW"
                )

            elif lane in adj:

                transitions.append(
                    "LANE_CHANGE"
                )

            else:

                transitions.append(
                    "TRANSFER"
                )

            route.append(lane)

            prev_lane = lane

        return {
            "route": route,
            "transitions": transitions
        }

    def _build_topology_facts(self, ctx_a, ctx_b):
        meta = getattr(self.road, "meta", {}) or {}

        def actor_signature(ctx):
            timeline = ctx.get("timeline", {})

            route = []
            prev = None

            for _, sample in sorted(timeline.items()):
                lane = sample.get("region_id")
                if lane is None:
                    continue

                lane = str(lane)

                if lane == prev:
                    continue

                route.append(lane)
                prev = lane

            topology_sequence = []
            merge_points = []
            diverge_points = []

            for lane in route:
                m = meta.get(str(lane), {}) or {}

                preds = m.get("predecessors", []) or []
                succs = m.get("successors", []) or []

                if len(preds) > 1:
                    topology_sequence.append("CONVERGE")
                    if lane not in merge_points:
                        merge_points.append(lane)

                if len(succs) > 1:
                    topology_sequence.append("DIVERGE")
                    if lane not in diverge_points:
                        diverge_points.append(lane)

                if m.get("junction_id") is not None:
                    topology_sequence.append("JUNCTION")

                if m.get("is_connecting_road", False):
                    topology_sequence.append("CONNECTOR")

            compressed = []

            for token in topology_sequence:
                if not compressed or compressed[-1] != token:
                    compressed.append(token)

            return {
                "topology_sequence": compressed,
                "route_length": len(route),
                "merge_points": merge_points,
                "diverge_points": diverge_points
            }

        a = actor_signature(ctx_a)
        b = actor_signature(ctx_b)

        shared_topology = len(
            set(a["topology_sequence"])
            &
            set(b["topology_sequence"])
        )

        # Common merge/diverge POINTS: lanes that are merge/diverge
        # lanes (per meta topology) AND are traversed by both actors.
        b_merges = set(b["merge_points"])
        b_diverges = set(b["diverge_points"])

        common_merge = [
            lane for lane in a["merge_points"] if lane in b_merges
        ]
        common_diverge = [
            lane for lane in a["diverge_points"] if lane in b_diverges
        ]

        return (
            {
                "A": a,
                "B": b
            },
            {
                "topology_sequence_A": a["topology_sequence"],
                "topology_sequence_B": b["topology_sequence"],
                "shared_topology_tokens": shared_topology,
                "topology_complexity":
                    len(a["topology_sequence"])
                    +
                    len(b["topology_sequence"]),
                "common_merge_points": common_merge,
                "common_diverge_points": common_diverge
            }
        )


    def _build_interaction_facts(
            self,
            pair):

        return {
            "timeline":
                pair.get(
                    "timeline",
                    {}
                ),

            "summary": {
                "distance_min":
                    pair.get(
                        "distance_min"
                    ),

                "distance_max":
                    pair.get(
                        "distance_max"
                    ),

                "ttc_min":
                    pair.get(
                        "ttc_min"
                    )
            }
        }

    def _build_pair_scenario_context(
            self,
            actor_a_id,
            actor_b_id,
            ctx_a,
            ctx_b,
            pair,
            start_frame,
            end_frame):

        route_a = (
            ctx_a.get(
                "route_semantics"
            )
            or
            self._build_route_semantics(
                ctx_a["timeline"]
            )
        )

        route_b = (
            ctx_b.get(
                "route_semantics"
            )
            or
            self._build_route_semantics(
                ctx_b["timeline"]
            )
        )

        actor_topology, interaction_topology = (
            self._build_topology_facts(
                ctx_a,
                ctx_b
            )
        )
        pair_timeline = (
            pair.get(
                "timeline",
                {}
            )
        )

        shared_frames = []

        for frame_id, sample in pair_timeline.items():

            lane_rel = sample.get(
                "lane_relation",
                {}
            )

            if lane_rel.get(
                "shared_junction",
                False
            ):
                shared_frames.append(
                    int(frame_id)
                )

        interaction_topology[
            "shared_junction_frames"
        ] = shared_frames

        interaction_topology[
            "shared_junction_count"
        ] = len(shared_frames)
        return {

            "kind":
                "map_scenario_context",

            "version":
                "1.0",

            "context_version":
                "map_scenario_context_v1",

            "ready":
                False,

            "scope": {

                "type":
                    "pair_window",

                "start_frame":
                    int(start_frame),

                "end_frame":
                    int(end_frame)
            },

            "actors": {

                "A": {

                    "id":
                        str(actor_a_id),

                    "timeline":
                        ctx_a["timeline"],

                    "route_semantics":
                        route_a,

                    "topology_facts":
                        actor_topology["A"]
                },

                "B": {

                    "id":
                        str(actor_b_id),

                    "timeline":
                        ctx_b["timeline"],

                    "route_semantics":
                        route_b,

                    "topology_facts":
                        actor_topology["B"]
                }
            },

            "interaction_facts":
                self._build_interaction_facts(
                    pair
                ),

            "interaction_topology":
                interaction_topology,

            "duration_seconds":
                float(
                    self._get_valid_duration()
                    or 1.0
                ),

            "meta":
                getattr(
                    self.road,
                    "meta",
                    {}
                ) or {}
        }

    def build_semantic_timeline_database(self):

        if not self.openlabel_traj_map:
            return self.log_warn("Load OpenLABEL first.")

        if not self.road:
            return self.log_warn("Load layout first.")

        if not self.homography:
            return self.log_warn("Load homography first.")

        try:

            save_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Event Sequence Database",
                "event_sequence_db.json",
                "JSON (*.json)"
            )

            if not save_path:
                return

            ids = sorted(
                self.openlabel_traj_map,
                key=str
            )

            # --------------------------------------------------
            # Vehicle cache
            # --------------------------------------------------

            vehicle_cache = {}
            
            for oid in ids:

                traj = self.openlabel_traj_map.get(oid)

                if not traj:
                    continue

                frame_map = {
                    fr: (x, y)
                    for fr, x, y in traj
                }

                frames = sorted(frame_map)

                vehicle_cache[oid] = {
                    "traj": traj,
                    "frame_map": frame_map,
                    "frame_keys": set(frames),
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "frame_count": len(frames)
                }

            # --------------------------------------------------
            # Vehicle contexts + semantic cache
            # --------------------------------------------------

            vehicle_semantic_cache = {}
            vehicle_ctx_cache = self._openlabel_vehicle_ctx_cache

            progress_ctx = QProgressDialog(
                "Precomputing vehicle semantics...",
                "Cancel",
                0,
                len(ids),
                self
            )

            for i, oid in enumerate(ids):

                if i % 20 == 0:
                    progress_ctx.setValue(i)
                    QApplication.processEvents()

                if progress_ctx.wasCanceled():
                    return

                try:

                    if oid not in vehicle_ctx_cache:

                        builder_traj = (
                            self._traj_to_builder_format(
                                vehicle_cache[oid]["traj"]
                            )
                        )

                        vehicle_ctx_cache[oid] = (
                            self._ctx_builder.build_vehicle_context(
                                builder_traj
                            )
                        )

                    veh_ctx = vehicle_ctx_cache[oid]

                    semantic_ctx = (
                        self._build_pair_scenario_context(
                            actor_a_id=oid,
                            actor_b_id=oid,
                            ctx_a=veh_ctx,
                            ctx_b=veh_ctx,
                            pair={"timeline": {}},
                            start_frame=vehicle_cache[oid]["first_frame"],
                            end_frame=vehicle_cache[oid]["last_frame"]
                        )
                    )

                    semantic_ctx["ready"] = True

                    semantic_ctx = (
                        self.build_semantic_context(
                            semantic_ctx
                        )
                    )

                    vehicle_semantic_cache[oid] = {
                        "context": semantic_ctx,
                        "descriptor":
                            semantic_ctx
                            .get("derived", {})
                            .get(
                                "scenario_descriptor",
                                {}
                            ),
                        "events":
                            semantic_ctx.get(
                                "event_timeline",
                                []
                            )
                    }

                except Exception as e:

                    print(
                        f"[CTX CACHE] {oid}: {e}"
                    )

            progress_ctx.close()

            total_pairs = len(ids) * (len(ids) - 1) // 2

            progress = QProgressDialog(
                "Building Event Database...",
                "Cancel",
                0,
                max(total_pairs, 1),
                self
            )

            tracks_written = 0
            first_track = True
            counter = 0

            with open(save_path, "w", encoding="utf-8") as out:

                out.write(
                    '{"version":6,"type":"event_sequence_database","tracks":[\n'
                )

                # ==================================================
                # Vehicle tracks
                # ==================================================

                for oid, veh in vehicle_cache.items():
                    try:
                        sem = vehicle_semantic_cache.get(oid)
                        if sem is None:
                            continue
                        # Reuse the precomputed semantic context.
                        ctx_single = sem["context"]
                        seen = set()
                        events = []

                        for ev in ctx_single.get(
                            "event_timeline",
                            []
                        ):
                            participants = ev.get(
                                "participants",
                                []
                            )

                            # remove synthetic pair events
                            if len(participants) > 1:
                                continue

                            ev_name = ev.get("event")

                            # deduplicate mirrored self-events
                            key = (
                                ev_name,
                                round(
                                    float(ev.get("start_t", 0.0)),
                                    6
                                ),
                                round(
                                    float(ev.get("end_t", 0.0)),
                                    6
                                ),
                                ev.get("region"),
                                ev.get("from_region"),
                                ev.get("to_region")
                            )
                            if key in seen:
                                continue
                            seen.add(key)
                            events.append(ev)

                        event_sequence, event_transitions, stats = (
                            build_searchable_event_sequence(
                                events
                            )
                        )
                        descriptor = sem["descriptor"]
                        track = {
                            "scope": "vehicle",

                            "actor_ids": [
                                str(oid)
                            ],

                            "first_frame":
                                veh["first_frame"],

                            "last_frame":
                                veh["last_frame"],

                            "frame_count":
                                veh["frame_count"],

                            "event_sequence":
                                event_sequence,

                            "event_transitions":
                                event_transitions,

                            "event_stats":
                                stats,

                            "event_fingerprint":
                                sorted(
                                    {
                                        e["event"]
                                        for e in event_sequence
                                    }
                                ),
                            "rule_types":
                                sorted(
                                    set(
                                        descriptor
                                        .get(
                                            "search_index",
                                            {}
                                        )
                                        .get(
                                            "rule_types",
                                            []
                                        )
                                    )
                                ),

                            "descriptor":
                                descriptor,                                
                            "route_signatures": [
                                tuple(
                                    ctx_single["actors"]["A"]
                                    ["route_semantics"]
                                    ["transitions"]
                                )
                            ],

                            "topology_facts":
                                ctx_single["actors"]["A"]
                                ["topology_facts"],                                
                        }

                        first_track = self._write_track(
                            out,
                            track,
                            first_track
                        )

                        tracks_written += 1

                    except Exception as e:

                        print(
                            f"[VEHICLE INDEX] "
                            f"{oid}: {e}"
                        )

                # ==================================================
                # Pair tracks
                # ==================================================

                for i, v1_id in enumerate(ids):

                    v1 = vehicle_cache.get(v1_id)

                    if not v1:
                        continue

                    for v2_id in ids[i + 1:]:

                        counter += 1

                        if counter % 1000 == 0:
                            progress.setValue(counter)
                            QApplication.processEvents()

                        if progress.wasCanceled():
                            return

                        v2 = vehicle_cache.get(v2_id)

                        if not v2:
                            continue

                        start_frame = max(
                            v1["first_frame"],
                            v2["first_frame"]
                        )

                        end_frame = min(
                            v1["last_frame"],
                            v2["last_frame"]
                        )

                        if end_frame <= start_frame:
                            continue

                        if end_frame - start_frame < 50:
                            continue

                        start_frame = max(
                            v1["first_frame"],
                            v2["first_frame"]
                        )

                        end_frame = min(
                            v1["last_frame"],
                            v2["last_frame"]
                        )

                        if end_frame <= start_frame:
                            continue

                        common_count = (
                            end_frame -
                            start_frame +
                            1
                        )

                        if common_count < 50:
                            continue

                        # --------------------------------------
                        # Cheap spatial rejection
                        # --------------------------------------

                        samples = (
                            start_frame,
                            (start_frame + end_frame) // 2,
                            end_frame
                        )

                        min_dist = float("inf")

                        for fr in samples:

                            p1 = v1["frame_map"].get(fr)
                            p2 = v2["frame_map"].get(fr)

                            if not p1 or not p2:
                                continue

                            min_dist = min(
                                min_dist,
                                math.hypot(
                                    p2[0] - p1[0],
                                    p2[1] - p1[1]
                                )
                            )

                        if min_dist > 150:
                            continue

                        try:

                            ctx_a = vehicle_ctx_cache[v1_id]
                            ctx_b = vehicle_ctx_cache[v2_id]
                            veh1_events = (
                                vehicle_semantic_cache
                                [v1_id]
                                ["events"]
                            )

                            veh2_events = (
                                vehicle_semantic_cache
                                [v2_id]
                                ["events"]
                            )

                            if (
                                len(veh1_events) == 0
                                and
                                len(veh2_events) == 0
                            ):
                                continue
                            tl1 = {
                                k: v
                                for k, v in ctx_a["timeline"].items()
                                if start_frame <= k <= end_frame
                            }

                            tl2 = {
                                k: v
                                for k, v in ctx_b["timeline"].items()
                                if start_frame <= k <= end_frame
                            }

                            common = set(tl1) & set(tl2)

                            if len(common) < 50:
                                continue

                            tl1 = {k: tl1[k] for k in common}
                            tl2 = {k: tl2[k] for k in common}

                            pair = self._ctx_builder.build_pairwise(
                                {"timeline": tl1},
                                {"timeline": tl2}
                            )

                            duration = (
                                end_frame - start_frame
                            ) / max(self.fps, 1.0)

                            ctx = self._build_pair_scenario_context(
                                actor_a_id=v1_id,
                                actor_b_id=v2_id,
                                ctx_a={"timeline": tl1},
                                ctx_b={"timeline": tl2},
                                pair=pair,
                                start_frame=start_frame,
                                end_frame=end_frame
                            )

                            ctx["ready"] = True

                            ctx = (
                                self.build_semantic_context(
                                    ctx
                                )
                            )

                            events = (
                                ctx.get("event_timeline")
                                or []
                            )

                            event_sequence, transitions, stats = (
                                build_searchable_event_sequence(
                                    events
                                )
                            )
                            interaction_topology = (
                                ctx.get(
                                    "interaction_topology",
                                    {}
                                )
                            )
                            if (
                                not event_sequence
                                and
                                not interaction_topology.get(
                                    "common_merge_points"
                                )
                                and
                                not interaction_topology.get(
                                    "common_diverge_points"
                                )
                            ):
                                continue

                            actors = ctx.get(
                                "actors",
                                {}
                            )

                            actor_a_topology = (
                                actors.get(
                                    "A",
                                    {}
                                )
                                .get(
                                    "route_semantics",
                                    {}
                                )
                                .get(
                                    "transitions",
                                    []
                                )
                            )

                            actor_b_topology = (
                                actors.get(
                                    "B",
                                    {}
                                )
                                .get(
                                    "route_semantics",
                                    {}
                                )
                                .get(
                                    "transitions",
                                    []
                                )
                            )

                            track = {
                                "scope": "pair",
                                "actor_ids": [
                                    str(v1_id),
                                    str(v2_id)
                                ],
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "frame_count": len(common),
                                "interaction_topology":
                                    interaction_topology,

                                "route_signatures":
                                    sorted([
                                        tuple(actor_a_topology),
                                        tuple(actor_b_topology)
                                    ]),

                                "topology_transitions": {
                                    "A": actor_a_topology,
                                    "B": actor_b_topology
                                },

                                "event_sequence":
                                    event_sequence,

                                "event_transitions":
                                    transitions,

                                "event_stats":
                                    stats,

                                "event_fingerprint":
                                    sorted(
                                        set(
                                            e["event"]
                                            for e in event_sequence
                                        )
                                    ),
                                "semantic_signature":
                                    ctx.get(
                                        "derived",
                                        {}
                                    )
                                    .get(
                                        "scenario_descriptor",
                                        {}
                                    )
                                    .get(
                                        "search_index",
                                        {}
                                    )
                                    .get(
                                        "semantic_signature",
                                        {}
                                    ),                                    
                            }


                            first_track = self._write_track(
                                out,
                                track,
                                first_track
                            )

                            tracks_written += 1

                            if tracks_written % 500 == 0:
                                out.flush()

                        except Exception as e:

                            print(
                                f"[INDEX] "
                                f"{v1_id}-{v2_id}: {e}"
                            )

                out.write("\n]}")

            progress.close()

            self.load_semantic_database(
                save_path
            )

            self.log_info(
                f"Database saved and loaded.\n"
                f"Tracks: {tracks_written}\n"
                f"File: {save_path}"
            )

        except Exception as e:

            self.log_error(
                f"Database generation failed: {e}"
            )