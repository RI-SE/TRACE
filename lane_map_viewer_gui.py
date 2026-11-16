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
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import (
    QImage, QPixmap, QColor, QPen, QBrush, QPainterPath
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsPathItem, QComboBox, QMessageBox, QSplitter, QSizePolicy,
    QProgressDialog, QApplication, QDoubleSpinBox, QSpinBox, QTextEdit
)

from scenario_engine import suggest_rules_combined, build_engine_insight_context, enrich_with_signal_metrics, enrich_with_topology, detect_events, _build_event_timeline
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

        hover_index = getattr(viewer, "_hover_index", None)

        if not hover_index:
            return None

        best = None
        best_dist = float("inf")

        for vid, idx, px, py in hover_index:
            d = (px - x) ** 2 + (py - y) ** 2

            if d < best_dist:
                best_dist = d
                best = (vid, idx, d ** 0.5)

        return best if best and best[2] <= self._hit_radius_px else None


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

        if self.is_manual_mode:
            self._manual_traj_done = {"V1": False, "V2": False}

        self._traj_items = {
            "manual": {"V1": [], "V2": []},
            "json": {"V1": [], "V2": []},
        }
        self._context_manual = {"ready": False}
        self._context_json = {"ready": False}
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
                    "sd": sd
                }
            ))

        return out

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
        reg = self.road.regions.get(lane_id)
        if not reg:
            return None

        cl = getattr(reg, "centerline", None)
        if not cl or len(cl) < 2:
            return None

        pts = np.asarray(cl)
        px, py = wp[0], wp[1]

        best_s = None
        best_d = None
        best_dist = float("inf")
        accum_s = 0.0

        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i + 1]

            v = p2 - p1
            w = np.array([px, py]) - p1

            seg_len = np.linalg.norm(v)
            if seg_len < 1e-6:
                continue

            t = np.clip(np.dot(w, v) / (seg_len ** 2), 0.0, 1.0)
            proj = p1 + t * v

            dist = np.linalg.norm(np.array([px, py]) - proj)

            if dist < best_dist:
                best_dist = dist
                best_s = accum_s + t * seg_len
                normal = np.array([-v[1], v[0]])
                sign = np.sign(np.dot(w, normal))
                best_d = sign * dist

            accum_s += seg_len

        if best_s is None:
            return None

        return {"s": best_s, "d": best_d}

    
    # ===================== NEW MAIN LAYOUT =====================

    def _get_hover_vehicle_insight(self, vid, idx):
        ctx = self._context_manual if self.is_manual_mode else self._context_json

        if not ctx.get("ready"):
            return None

        veh = ctx.get("vehicles", {}).get(vid, {})
        timeline = veh.get("timeline", {})

        frame = timeline.get(idx)
        if not frame:
            return None

        sd = frame.get("sd") or {}

        return {
            "s": sd.get("s"),
            "d": sd.get("d"),
            "t": idx,
            "region": frame.get("region_id")
        }

    def _get_hover_pair_insight(self, idx):
        ctx = self._context_manual if self.is_manual_mode else self._context_json

        pair = ctx.get("pair", {})
        timeline = pair.get("timeline", {})

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
        self.ctrl_build_timeline = QPushButton("Build Event Timeline")

        self.ctrl_find_similar = QPushButton("Find Similar")
        self.ctrl_precompute = QPushButton("Precompute Index")
        self.ctrl_load_db = QPushButton("Load Semantic DB")

        act_row.addWidget(self.ctrl_generate)
        act_row.addWidget(self.ctrl_apply)
        act_row.addWidget(self.ctrl_clear)
        act_row.addWidget(self.ctrl_build_timeline)
        act_row.addSpacing(20)

        act_row.addWidget(self.ctrl_precompute)
        act_row.addWidget(self.ctrl_find_similar)

        act_row.addStretch()

        layout.addLayout(act_row)

        # --------------------------------------------------
        # Status row
        # --------------------------------------------------

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

            ctx1 = self._ctx_builder.build_vehicle_context(traj1)
            ctx2 = self._ctx_builder.build_vehicle_context(traj2)

        except Exception as e:
            return None, f"vehicle_ctx_failed: {e}"

        return (ctx1, ctx2), None

    def _empty_context(self):
        return {
            "ready": False,
            "state": "init",
            "error": None,
            "vehicles": {},
            "pair": {},
            "meta": {},
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

        traj_map = self._get_active_traj_map()

        for vid, traj in traj_map.items():
            for idx, (_, x, y) in enumerate(traj):
                self._hover_index.append(
                    (vid, idx, float(x), float(y))
                )

    def _get_active_traj_map(self):
        if self.is_manual_mode:
            return self._traj_map_manual
        elif self.active_mode == "json":
            return self._traj_map_json
        return {}

    def _get_active_store(self):
        return {
            "traj_map": self._traj_map_manual if self.is_manual_mode else self._traj_map_json,
            "world": self._world_polylines_manual if self.is_manual_mode else self._world_polylines_json,
            "context": self._context_manual if self.is_manual_mode else self._context_json,
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

        if not ctx.get("vehicles") or "V1" not in ctx["vehicles"]:
            ctx["ready"] = False
            return ctx
        try:
            ctx = normalize_context(ctx)
            ctx["vehicles"]["V1"]["timeline"] = _fix_timeline_keys(
                ctx["vehicles"]["V1"]["timeline"]
            )
            ctx["vehicles"]["V2"]["timeline"] = _fix_timeline_keys(
                ctx["vehicles"]["V2"]["timeline"]
            )
            ctx["pair"]["timeline"] = _fix_timeline_keys(
                ctx["pair"]["timeline"]
            )
            validate_context(ctx)
            ctx["ready"] = True

        except Exception as e:
            ctx["ready"] = False
            ctx["error"] = str(e)

        if self.is_manual_mode:
            self._context_manual = ctx
        else:
            self._context_json = ctx

        return ctx


    def build_context_atomic(self):
        ctx = self._empty_context()

        if not self.road:
            return self._fail_ctx(ctx, "missing_road")

        vehicles_ctx, err = self._build_vehicle_contexts()
        if err:
            return self._fail_ctx(ctx, err)

        ctx1, ctx2 = vehicles_ctx

        try:
            k1 = set(map(int, ctx1["timeline"].keys()))
            k2 = set(map(int, ctx2["timeline"].keys()))
            common = sorted(k1 & k2)

            if not common:
                return self._fail_ctx(ctx, "no_common_frames")

            ctx1["timeline"] = {k: ctx1["timeline"].get(k) for k in common}
            ctx2["timeline"] = {k: ctx2["timeline"].get(k) for k in common}

            _dbg("aligned V1 size", len(ctx1["timeline"]))
            _dbg("aligned V2 size", len(ctx2["timeline"]))

            pair = self._ctx_builder.build_pairwise(ctx1, ctx2)

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

        ctx.update({
            "ready": False,
            "state": "pair_ready",

            "vehicles": {
                "V1": {"timeline": ctx1.get("timeline", {})},
                "V2": {"timeline": ctx2.get("timeline", {})}
            },

            "pair": pair,
            "duration_seconds": float(self._get_valid_duration() or 1.0),
            "context_version": "timeline_v2",

            "meta": getattr(self.road, "meta", {}) or {},
        })

        return ctx

    def build_semantic_context(self, ctx):

        if not ctx:
            return {}

        ctx["ready"] = True

        duration = float(
            ctx.get("duration_seconds", 1.0)
        )

        ctx = build_engine_insight_context(
            context=ctx,
            duration=duration
        )

        ctx = enrich_with_topology(ctx)

        ctx = enrich_with_signal_metrics(ctx)

        engine = (
            ctx.get("derived", {})
            .get("engine", {})
        )

        if not engine.get("v1_sd") or not engine.get("v2_sd"):
            return ctx

        ctx["events"] = detect_events(ctx)

        ctx["event_timeline"] = (
            _build_event_timeline(ctx)
            or []
        )

        return ctx

    def _fail_ctx(self, ctx, err):
        ctx.update({
            "ready": False,
            "state": "failed",
            "error": err,
            "vehicles": {
                "V1": {"timeline": {}},
                "V2": {"timeline": {}},
            },
            "pair": {"timeline": {}},
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
        print("CTX READY:", self._context_json.get("ready"))
        print("V1 timeline size:", len(self._context_json.get("vehicles", {}).get("V1", {}).get("timeline", {})))
        print("Pair timeline size:", len(self._context_json.get("pair", {}).get("timeline", {})))            

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
        ctx = self._context_manual if self.is_manual_mode else self._context_json

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

        if "vehicles" not in ctx or "derived" not in ctx:
            print("[LaneMapViewer] Incomplete engine context")
            return
        self._engine_context = dict(ctx)
        self._last_engine_context = dict(ctx)
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
                adj = md.get("adjacent", [])
                suc = md.get("successors", [])
                pre = md.get("predecessors", [])

                line1 = (
                    f"R{rid} | {t} | "
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

        self._context_json = self._empty_context()
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
                s_lane = w.get("same_lane_ratio", 0.0)
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
                    f"min_dist={w['min_dist']:.1f}, min_ttc={ttc_s}, same_lane={w['same_lane_ratio']:.2f}"
                )
            msg = "Candidate (v1, v2, [start,end]) windows from index:\n\n" + "\n".join(lines)
            self.log_info("Suggested Candidate Windows" + msg)
            return

        # ---------- Guardrails for JSON / base scenario path ----------
        if self.active_mode != "json":
            self.log_warn("Switch to JSON mode to use candidate finder.")
            return
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


        # --- Reference window length from GUI (as before) ---
        try:
            fstart_cur = int(self.global_start_box.currentText())
            fend_cur = int(self.global_end_box.currentText())
        except:
            self.log_warn("Invalid frames", "Select a valid global frame range.")
            return
        if fstart_cur > fend_cur:
            fstart_cur, fend_cur = fend_cur, fstart_cur

        # ---------- Case B: Base scenario present & precomp enabled: evaluate only candidate windows ----------
        db = getattr(
            self,
            "semantic_db",
            None
        )

        if not db:
            self.log_warn(
                "Load event sequence DB first."
            )
            return

        query_sequence = self._build_query_sequence()

        if not query_sequence:
            self.log_warn(
                "Current scenario has no event sequence."
            )
            return

        results = []

        for track in db.get("tracks",[]):
            cand_sequence = track.get(
                "rule_signature",
                []
            )

            seq_score = self._sequence_similarity(
                query_sequence,
                cand_sequence
            )
            if seq_score < 0.15:
                continue

            results.append({

                "score": seq_score,

                "scope": track.get(
                    "scope"
                ),

                "track": track
            })

        unique = {}

        for r in results:
            sig = (
                tuple(
                    r["track"].get(
                        "rule_signature",
                        []
                    )
                ),
                r["track"].get(
                    "path_relationship"
                )
            )

            if (
                sig not in unique
                or
                r["score"] > unique[sig]["score"]
            ):
                unique[sig] = r

        results = list(unique.values())

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        top = results[:20]

        lines = []

        for i, r in enumerate(top, start=1):

            track = r["track"]
            seq = " → ".join(
                track.get(
                    "rule_signature",
                    []
                )
            )

            lines.append(
                f"{i}. "
                f"score={r['score']:.3f} | "
                f"path={track.get('path_relationship')} | "
                f"frames={track.get('start_frame')}-"
                f"{track.get('end_frame')} | "
                f"rules={seq}"
            )

        self.log_info(
            "Top Event Sequence Matches\n\n"
            +
            "\n".join(lines)
        )
        return


    def load_semantic_database(self, path):

        with open(path, "r", encoding="utf-8") as f:
            self.semantic_db = json.load(f)

        self.log_info(
            f"Loaded event sequence database "
            f"({len(self.semantic_db.get('tracks', []))} tracks)"
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
        query_sequence,
        candidate_sequence
    ):

        n = len(query_sequence)
        m = len(candidate_sequence)

        if n == 0 or m == 0:
            return 0.0

        dp = [
            [0]*(m+1)
            for _ in range(n+1)
        ]

        for i in range(1, n+1):

            for j in range(1, m+1):

                if (
                    query_sequence[i-1]
                    ==
                    candidate_sequence[j-1]
                ):
                    dp[i][j] = (
                        dp[i-1][j-1] + 1
                    )
                else:
                    dp[i][j] = max(
                        dp[i-1][j],
                        dp[i][j-1]
                    )

        lcs = dp[n][m]

        return lcs / float(max(n, m))

    def _build_query_sequence(self):

        items = []

        for r in getattr(
            self.scenario_gui,
            "_rules",
            []
        ):

            params = getattr(
                r,
                "parameters",
                {}
            ) or {}

            subtype = params.get(
                "subtype"
            )

            if not subtype:
                continue

            items.append(
                (
                    params.get(
                        "start_t",
                        0.0
                    ),
                    str(subtype)
                )
            )

        items.sort(
            key=lambda x: x[0]
        )

        return [
            x[1]
            for x in items
        ]

    def _rules_to_records(self, rules):

        seen = set()
        sigs = []
        records = []

        for r in rules:

            p = getattr(r, "parameters", {}) or {}
            subtype = p.get("subtype")

            if not subtype:
                continue

            key = (
                subtype,
                r.actor,
                p.get("start_t"),
                p.get("end_t")
            )

            if key in seen:
                continue

            seen.add(key)

            sigs.append(str(subtype))

            records.append({
                "type": str(subtype),
                "actor": r.actor,
                "start_t": p.get("start_t"),
                "end_t": p.get("end_t")
            })

        return sigs, records

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
            # Vehicle contexts
            # --------------------------------------------------

            vehicle_ctx_cache = {}

            progress_ctx = QProgressDialog(
                "Precomputing vehicle contexts...",
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

                    vehicle_ctx_cache[oid] = (
                        self._ctx_builder.build_vehicle_context(
                            self._traj_to_builder_format(
                                vehicle_cache[oid]["traj"]
                            )
                        )
                    )

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

                for oid in ids:

                    veh = vehicle_cache.get(oid)
                    veh_ctx = vehicle_ctx_cache.get(oid)

                    if not veh or not veh_ctx:
                        continue

                    try:

                        duration = (
                            veh["last_frame"]
                            - veh["first_frame"]
                        ) / max(self.fps, 1.0)

                        tl = veh_ctx.get("timeline", {})

                        route = []
                        transitions = []

                        prev_region = None

                        for fr in sorted(tl):

                            rid = tl[fr].get("region_id")

                            if rid is None:
                                continue

                            if rid != prev_region:

                                route.append(rid)

                                if prev_region is not None:

                                    transitions.append({
                                        "frame": int(fr),
                                        "from_region": prev_region,
                                        "to_region": rid
                                    })

                                prev_region = rid

                        # -----------------------------
                        # build synthetic 2-vehicle ctx
                        # -----------------------------

                        single_ctx = {

                            "ready": True,

                            "vehicles": {
                                "V1": {
                                    "timeline": tl
                                },
                                "V2": {
                                    "timeline": {}
                                }
                            },

                            "pair": {
                                "timeline": {}
                            },

                            "duration_seconds": duration,

                            "meta": getattr(
                                self.road,
                                "meta",
                                {}
                            ) or {}
                        }

                        try:

                            single_ctx = self.build_semantic_context(
                                single_ctx
                            )

                            rules = suggest_rules_combined(
                                duration=duration,
                                context=single_ctx
                            )

                            sigs = []
                            records = []
                            seen = set()

                            for r in rules:

                                p = (
                                    getattr(
                                        r,
                                        "parameters",
                                        {}
                                    )
                                    or {}
                                )

                                subtype = p.get(
                                    "subtype"
                                )

                                if not subtype:
                                    continue

                                key = (
                                    subtype,
                                    r.actor,
                                    p.get("start_t"),
                                    p.get("end_t")
                                )

                                if key in seen:
                                    continue

                                seen.add(key)

                                sigs.append(
                                    str(subtype)
                                )

                                records.append({

                                    "type":
                                        str(subtype),

                                    "actor":
                                        r.actor,

                                    "start_t":
                                        p.get("start_t"),

                                    "end_t":
                                        p.get("end_t"),

                                    "interaction":
                                        p.get("interaction"),

                                    "behavior":
                                        p.get("behavior"),

                                    "timeline":
                                        p.get("timeline")
                                })


                        except Exception:

                            sigs = []
                            records = []

                        vehicle_events = []

                        for ev in (
                            single_ctx.get(
                                "event_timeline",
                                []
                            )
                            or []
                        ):

                            item = {

                                "type":
                                    ev.get("type"),

                                "t_start":
                                    ev.get("t_start"),

                                "t_end":
                                    ev.get("t_end"),

                                "participants":
                                    ev.get(
                                        "participants",
                                        []
                                    )
                            }

                            metrics = (
                                ev.get("metrics")
                                or {}
                            )

                            if metrics:

                                item["metrics"] = metrics

                                if "from_lane" in metrics:

                                    item[
                                        "from_region"
                                    ] = metrics[
                                        "from_lane"
                                    ]

                                if "to_lane" in metrics:

                                    item[
                                        "to_region"
                                    ] = metrics[
                                        "to_lane"
                                    ]

                            vehicle_events.append(item)

                        track = {

                            "scope": "vehicle",

                            "actor_ids": [
                                str(oid)
                            ],

                            "start_frame":
                                veh["first_frame"],

                            "end_frame":
                                veh["last_frame"],

                            "frame_count":
                                veh["frame_count"],

                            "route":
                                route,

                            "transitions":
                                transitions,

                            "rule_signature":
                                sigs,

                            "rules":
                                records,

                            "events":
                                vehicle_events
                        }

                        first_track = self._write_track(
                            out,
                            track,
                            first_track
                        )

                        tracks_written += 1

                    except Exception as e:

                        print(
                            f"[VEHICLE] {oid}: {e}"
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

                        common = (
                            v1["frame_keys"]
                            &
                            v2["frame_keys"]
                        )

                        if len(common) < 50:
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

                            ctx1 = vehicle_ctx_cache[v1_id]
                            ctx2 = vehicle_ctx_cache[v2_id]

                            tl1 = {
                                k: v
                                for k, v in ctx1["timeline"].items()
                                if k in common
                            }

                            tl2 = {
                                k: v
                                for k, v in ctx2["timeline"].items()
                                if k in common
                            }

                            pair = self._ctx_builder.build_pairwise(
                                {"timeline": tl1},
                                {"timeline": tl2}
                            )

                            duration = (
                                end_frame - start_frame
                            ) / max(self.fps, 1.0)

                            ctx = {
                                "ready": True,
                                "vehicles": {
                                    "V1": {"timeline": tl1},
                                    "V2": {"timeline": tl2}
                                },
                                "pair": pair,
                                "duration_seconds": duration,
                                "meta": getattr(
                                    self.road,
                                    "meta",
                                    {}
                                ) or {}
                            }

                            ctx = self.build_semantic_context(
                                ctx
                            )

                            rules = suggest_rules_combined(
                                duration=duration,
                                context=ctx
                            )

                            sigs, records = (
                                self._rules_to_records(
                                    rules
                                )
                            )

                            if not sigs:
                                continue

                            pair_events = []

                            for ev in (ctx.get("event_timeline") or []):

                                item = {

                                    "type":
                                        ev.get("type"),

                                    "t_start":
                                        ev.get("t_start"),

                                    "t_end":
                                        ev.get("t_end"),

                                    "participants":
                                        ev.get(
                                            "participants",
                                            []
                                        )
                                }

                                metrics = (
                                    ev.get("metrics")
                                    or {}
                                )

                                if metrics:

                                    item["metrics"] = metrics

                                    if "from_lane" in metrics:
                                        item["from_region"] = (
                                            metrics["from_lane"]
                                        )

                                    if "to_lane" in metrics:
                                        item["to_region"] = (
                                            metrics["to_lane"]
                                        )

                                pair_events.append(item)

                            track = {
                                "scope": "pair",
                                "actor_ids": [
                                    str(v1_id),
                                    str(v2_id)
                                ],
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "frame_count": len(common),
                                "path_relationship":
                                    ctx.get(
                                        "path_relationship"
                                    ),
                                "rule_signature": sigs,
                                "rules": records,
                                "events": pair_events
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

            self.log_info(
                f"Database saved.\n"
                f"Tracks: {tracks_written}\n"
                f"File: {save_path}"
            )

        except Exception as e:

            self.log_error(
                f"Database generation failed: {e}"
            )