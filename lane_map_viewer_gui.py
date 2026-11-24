# lane_map_viewer_gui.py
"""Lane map viewer: manual waypoint trajectories + JSON/OpenLABEL trajectories,
map context construction, semantic timeline DB, and similarity search."""
import json
import math
import os
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor, QPen, QBrush, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPathItem, QComboBox, QMessageBox,
    QSplitter, QSizePolicy, QProgressDialog, QApplication, QDoubleSpinBox,
    QSpinBox, QTextEdit, QListWidget, QListWidgetItem, QCheckBox,
)

from scipy.spatial import cKDTree

from scenario_engine import (
    build_engine_insight_context,
    build_searchable_event_sequence,
    build_scenario_descriptor,
    suggest_rules_combined,
    sanitize_for_json 
)
from road_infrastructure import RoadInfrastructure
from signal_engine import (
    ContextBuilder, SignalConfig, normalize_context, validate_context,
    WorldPoint,
)


# ===============================================================
# Utility helpers
# ===============================================================

def to_json_safe(obj):
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
    frames_dict, traj_map = {}, {}
    for fkey, fval in frames.items():
        try:
            fr = int(fkey)
        except (TypeError, ValueError):
            continue
        rows = []
        for oid, oval in (fval.get("objects") or {}).items():
            rbbox = oval.get("object_data", {}).get("rbbox") or []
            shape = next((e.get("val") for e in rbbox if e.get("name") == "shape"), None)
            if not shape or len(shape) < 5:
                continue
            try:
                cx, cy, w, h, ang = map(float, shape[:5])
            except (TypeError, ValueError):
                continue
            rows.append([cx, cy, w, h, ang, oid])
            traj_map.setdefault(oid, []).append((fr, cx, cy))
        if rows:
            frames_dict[fr] = rows
    if not frames_dict:
        return {}, {}, 0, 0
    fids = sorted(frames_dict)
    return frames_dict, traj_map, fids[0], fids[-1]


def polyline_path(pts):
    path = QPainterPath()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    return path


# ===============================================================
# Canvas
# ===============================================================
class LaneCanvas(QGraphicsView):
    clicked = pyqtSignal(int, int)
    region_hovered = pyqtSignal(int, dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setMouseTracking(True)
        self.mask = None
        self.meta = None
        self.pixmap_item = None
        self.centerline_items = []
        self.mode = "manual"
        self.active_vehicle = None
        self.dragging_wp = None
        self._hover_traj_hit = None
        self._hit_radius_px = 10
        self.manual_waypoints = {"V1": [], "V2": []}
        self.wp_items = {"V1": [], "V2": []}
        self.wp_line = {"V1": None, "V2": None}   # U3: live preview polyline per vehicle
        self._viewer_cache = None
        # U1: right-drag panning state
        self._panning = False
        self._pan_last = None
        self._pan_moved = False

    # ---------------- data ----------------
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

    @staticmethod
    def _colorize(mask):
        uniq = np.unique(mask)
        lut = np.zeros((int(uniq.max()) + 1, 3), dtype=np.uint8)
        rng = np.random.default_rng(1234)
        for u in uniq:
            if u:
                lut[int(u)] = rng.integers(80, 255, size=3, dtype=np.uint8)
        return lut[mask]

    # ---------------- drawing ----------------
    def draw_centerlines(self, road):
        self._clear_items(self.centerline_items)
        if not road:
            return
        for reg in road.regions.values():
            if reg.type not in ("lane", "connecting_lane"):
                continue
            cl = reg.centerline
            if not cl or len(cl) < 2:
                continue
            item = self.scene.addPath(polyline_path(cl), QPen(QColor(255, 0, 0, 180), 2))
            item.setZValue(40)
            self.centerline_items.append(item)

    def draw_trajectory(self, pts, color):
        if not pts:
            return []
        poly = QGraphicsPathItem(polyline_path(pts))
        poly.setPen(QPen(color, 3))
        poly.setZValue(50)
        self.scene.addItem(poly)
        items = [poly]
        for p in (pts[0], pts[-1]):
            dot = self.scene.addEllipse(*self._circle(p, 5), QPen(color), QBrush(color))
            dot.setZValue(51)
            items.append(dot)
        return items

    @staticmethod
    def _circle(pt, r):
        x, y = pt
        return x - r, y - r, 2 * r, 2 * r

    def _clear_items(self, items):
        for it in items:
            if it:
                self.scene.removeItem(it)
        items.clear()

    # ---------------- waypoints ----------------
    def set_mode(self, mode: str):
        self.mode = mode

    def enable_waypoints(self, vehicle: str):
        self.mode = "waypoints"
        self.active_vehicle = vehicle
        self.setCursor(Qt.CursorShape.CrossCursor)   # U3: visible mode feedback

    def clear_waypoints(self, vehicle=None):
        for v in ([vehicle] if vehicle else ("V1", "V2")):
            self._clear_items(self.wp_items[v])
            self.manual_waypoints[v].clear()
            if self.wp_line.get(v) is not None:      # U3: remove the live polyline too
                self.scene.removeItem(self.wp_line[v])
                self.wp_line[v] = None

    def _draw_waypoints(self, vehicle):
        pts = self.manual_waypoints[vehicle]
        items = self.wp_items[vehicle]
        color = QColor("green" if vehicle == "V1" else "cyan")
        while len(items) > len(pts):
            self.scene.removeItem(items.pop())
        for i, (x, y) in enumerate(pts):
            if i < len(items):
                items[i].setRect(x - 4, y - 4, 8, 8)
            else:
                dot = self.scene.addEllipse(x - 4, y - 4, 8, 8, QPen(color, 2), QBrush(color))
                dot.setZValue(60)
                items.append(dot)

        # U3: live dashed polyline so click order / start-end is visible before Generate
        line = self.wp_line[vehicle]
        if line is None:
            line = QGraphicsPathItem()
            line.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
            line.setZValue(55)
            self.scene.addItem(line)
            self.wp_line[vehicle] = line
        if len(pts) >= 2:
            line.setPath(polyline_path(pts))
        else:
            line.setPath(QPainterPath())

    # ---------------- interaction ----------------
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
        dist, pos = tree.query([float(x), float(y)], k=1)
        # U15: tree distance is in scene units - convert the pixel radius
        radius = self._hit_radius_px / max(self.transform().m11(), 1e-3)
        if dist > radius:
            return None
        vid, idx = index[pos]
        return vid, idx, float(dist)


    def _scene_int(self, ev):
        p = self.mapToScene(ev.pos())
        return int(p.x()), int(p.y())

    def wheelEvent(self, e):
        if self.pixmap_item is None:
            return
        factor = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        s = self.transform().m11()
        if not (0.05 <= s * factor <= 40.0):         # U1: sane zoom bounds
            return
        before = self.mapToScene(e.position().toPoint())
        self.scale(factor, factor)
        after = self.mapToScene(e.position().toPoint())
        # U1: keep the point under the cursor fixed (was: zoom about top-left corner)
        h, v = self.horizontalScrollBar(), self.verticalScrollBar()
        h.setValue(h.value() + int((after.x() - before.x()) * self.transform().m11()))
        v.setValue(v.value() + int((after.y() - before.y()) * self.transform().m22()))


    def mousePressEvent(self, ev):
        x, y = self._scene_int(ev)
        # U1: right button = pan (a right *click* without movement deletes a waypoint)
        if ev.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_last = ev.position()
            self._pan_moved = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            ev.accept()
            return
        if self.mode == "waypoints" and self.active_vehicle:
            pts = self.manual_waypoints[self.active_vehicle]
            tol = 8 / max(self.transform().m11(), 1e-3)   # U3: 8 px on screen, any zoom
            for i, (wx, wy) in enumerate(pts):
                if abs(x - wx) <= tol and abs(y - wy) <= tol:
                    self.dragging_wp = (self.active_vehicle, i)
                    return
            if not self.in_image_bounds(x, y):             # U3: no waypoints outside the mask
                return
            pts.append((x, y))
            self._draw_waypoints(self.active_vehicle)
            return
        self.clicked.emit(x, y)
        super().mousePressEvent(ev)


    def mouseMoveEvent(self, ev):
        if self._panning:                                  # U1: right-drag pan
            delta = ev.position() - self._pan_last
            self._pan_last = ev.position()
            if abs(int(delta.x())) + abs(int(delta.y())) > 3:
                self._pan_moved = True
            h, v = self.horizontalScrollBar(), self.verticalScrollBar()
            h.setValue(h.value() - int(delta.x()))
            v.setValue(v.value() - int(delta.y()))
            return
        x, y = self._scene_int(ev)
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
            v, i = self.dragging_wp
            self.manual_waypoints[v][i] = (x, y)
            self._draw_waypoints(v)
            return
        super().mouseMoveEvent(ev)


    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            # U3: right *click* (no movement) on a waypoint deletes it
            if (not self._pan_moved and self.mode == "waypoints"
                    and self.active_vehicle):
                x, y = self._scene_int(ev)
                pts = self.manual_waypoints[self.active_vehicle]
                tol = 8 / max(self.transform().m11(), 1e-3)
                for i, (wx, wy) in enumerate(pts):
                    if abs(x - wx) <= tol and abs(y - wy) <= tol:
                        del pts[i]
                        self._draw_waypoints(self.active_vehicle)
                        break
            if self.mode == "waypoints" and self.active_vehicle:
                self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self.unsetCursor()
        self.dragging_wp = None
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        # U3: Esc finishes waypoint editing
        if ev.key() == Qt.Key.Key_Escape and self.mode == "waypoints" and self.active_vehicle:
            viewer = self._find_viewer()
            if viewer is not None and hasattr(viewer, "_clear_edit_selection"):
                viewer._clear_edit_selection()
                self.mode = "manual"
                if hasattr(viewer, "log_info"):
                    viewer.log_info("Waypoint editing finished (Esc).")
                if hasattr(viewer, "update_status_panel"):
                    viewer.update_status_panel()
            self.unsetCursor()
            ev.accept()
            return
        super().keyPressEvent(ev)


    def _find_viewer(self):
        if self._viewer_cache is not None:
            return self._viewer_cache
        viewer = self.parent()
        while viewer and not hasattr(viewer, "road"):
            viewer = viewer.parent()
        self._viewer_cache = viewer
        return viewer


# ===============================================================
# Main Viewer
# ===============================================================
class LaneMapViewer(QWidget):
    polyline_committed = pyqtSignal(str, str)
    mask_loaded = pyqtSignal(object)
    meta_loaded = pyqtSignal(object)
    xodr_loaded = pyqtSignal(str)
    map_context_ready = pyqtSignal(dict)
    duration_changed = pyqtSignal(float)
    xodr_changed = pyqtSignal(str)

    PRECOMP_DIST_THRESH_PX = 80.0
    PRECOMP_TTC_THRESH_FRAMES = 3.0
    PRECOMP_MIN_CONSEC_FRAMES = 10
    PRECOMP_GRID_CELL_PX = 96
    PRECOMP_GRID_NEIGHBOR_RADIUS = 1
    PRECOMP_MAX_CELL_PAIR_COMBOS = 2000

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

    # ===================== state =====================

    def _init_state(self):
        self._project = {k: None for k in ("mask", "layout", "homography", "xodr", "openlabel", "db")}
        self._mask_path = self._layout_path = self._homog_path = self._xodr_path = self._openlabel_path = None
        self.homography = None
        self.road = None
        self._ctx_builder = None
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
        self._manual_traj_done = {"V1": False, "V2": False}
        self._traj_items = {"manual": {"V1": [], "V2": []}, "json": {"V1": [], "V2": []}}
        self._map_context_manual = {"ready": False}
        self._map_context_json = {"ready": False}
        self._hover_index = []
        self._hover_tree = None
        self.openlabel_frames = None
        self.openlabel_traj_map = None
        self.openlabel_fmin = 0
        self.openlabel_fmax = 0
        self._precomp = {"per_frame": {}, "pair_windows": []}
        self.semantic_db = None
        self._loaded_db_path = None
        self.scenario_gui = None
        self._save_debug_contexts = False   # U7: debug file writes are opt-in
        self.wp_line = {"V1": None, "V2": None}   # U3: live preview polyline per vehicle
        self._db_path = None          # event-sequence DB associated with the project
        self._project_path = None     # .trace file this session was opened/saved as
   

    @property
    def is_manual_mode(self):
        return self.active_mode in ("manual", "waypoints")

    # ===================== UI construction =====================

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

    def _build_top_toolbar(self):
        row = QHBoxLayout()
        self.btn_new_proj = QPushButton("New")
        self.btn_new_proj.setToolTip("Start a new project (asks for mask/layout/homography/xODR/OpenLABEL)")
        self.btn_open_proj = QPushButton("Open")
        self.btn_open_proj.setToolTip("Open a saved project (.trace) - paths + manual waypoints + DB")
        self.btn_save_proj = QPushButton("Save")
        self.btn_save_proj.setToolTip("Save current project (file paths + manual waypoints + DB) to .trace")
        self.btn_load_mask = QPushButton("Mask")
        self.btn_load_mask.setToolTip("Load region mask image (.png/.tif/.npy, 0 = non-drivable)")
        self.btn_load_meta = QPushButton("Layout")
        self.btn_load_meta.setToolTip("Load layout JSON (regions, topology, centerlines)")
        self.btn_load_homog = QPushButton("Homo.")
        self.btn_load_homog.setToolTip("Load homography JSON (pixel <-> world transform)")
        self.btn_load_xodr = QPushButton("xODR")
        self.btn_load_xodr.setToolTip("Load OpenDRIVE file (referenced by the exported OpenSCENARIO)")
        self.btn_load_openlabel = QPushButton("OpenLABEL")
        self.btn_load_openlabel.setToolTip("Load an OpenLABEL JSON with tracked objects (needed for JSON mode + DB build)")
        self.btn_load_openlabel.clicked.connect(self.load_openlabel_dialog)
        self.btn_fit_view = QPushButton("Fit")
        self.btn_fit_view.setToolTip("Zoom to fit the whole mask")
        self.btn_fit_view.clicked.connect(self._fit_view)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["Manual", "OpenLABEL (JSON)"])
        self.mode_box.setToolTip("Scenario source: manual waypoints or OpenLABEL tracked objects")
        self.mode_box.currentTextChanged.connect(self._on_mode_changed)

        self.btn_new_proj.clicked.connect(self.new_project)
        self.btn_open_proj.clicked.connect(self.open_project)
        self.btn_save_proj.clicked.connect(self.save_project)
        self.btn_load_mask.clicked.connect(self.load_mask)
        self.btn_load_meta.clicked.connect(
            lambda: self.load_meta_from_path(self._load_with_dialog("Layout", "JSON (*.json)")))
        self.btn_load_homog.clicked.connect(self.load_homography_json)
        self.btn_load_xodr.clicked.connect(self.load_xodr)

        for b in (self.btn_new_proj, self.btn_open_proj, self.btn_save_proj,
                  self.btn_load_mask, self.btn_load_meta, self.btn_load_homog,
                  self.btn_load_xodr, self.btn_load_openlabel, self.btn_fit_view):
            row.addWidget(b)
        row.addWidget(QLabel("Mode:"))
        row.addWidget(self.mode_box)
        row.addStretch()
        return row

    def load_openlabel_dialog(self):
        path = self._load_with_dialog("Open OpenLABEL JSON", "JSON Files (*.json)")
        if path:
            self.load_openlabel_from_path(path)


    def _fit_view(self):
        if self.view.pixmap_item is not None:
            self.view.fitInView(self.view.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


    def _build_main_layout(self):
        root = QVBoxLayout(self)
        root.addLayout(self._build_top_toolbar())
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.view = LaneCanvas(self)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.bottom_layout.addWidget(self._build_unified_panel())
        self._build_info_panel()
        splitter.addWidget(self.view)
        splitter.addWidget(self.bottom_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        QTimer.singleShot(0, lambda: splitter.setSizes([560, 340]))   # U16: was [900, 200]
        root.addWidget(splitter)
        return root

    def _build_unified_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Vehicle row
        veh_row = QHBoxLayout()
        self.ctrl_v1_btn = QPushButton("✏️")
        self.ctrl_v1_btn.setToolTip("Edit V1 waypoints: click on the map, drag to move, right-click a dot to delete, Esc to finish")
        self.ctrl_v2_btn = QPushButton("✏️")
        self.ctrl_v2_btn.setToolTip("Edit V2 waypoints: click on the map, drag to move, right-click a dot to delete, Esc to finish")
        self.ctrl_v1_box = self.v1_id_box
        self.ctrl_v2_box = self.v2_id_box
        self.ctrl_v1_btn.clicked.connect(lambda: self._set_wp_mode("V1"))
        self.ctrl_v2_btn.clicked.connect(lambda: self._set_wp_mode("V2"))
        veh_row.addWidget(QLabel("V1"))
        veh_row.addWidget(self.ctrl_v1_btn)
        veh_row.addWidget(self.ctrl_v1_box)
        veh_row.addSpacing(20)
        veh_row.addWidget(QLabel("V2"))
        veh_row.addWidget(self.ctrl_v2_btn)
        veh_row.addWidget(self.ctrl_v2_box)
        veh_row.addStretch()
        layout.addLayout(veh_row)

        # Time row
        time_row = QHBoxLayout()
        self.ctrl_start = self.global_start_box
        self.ctrl_end = self.global_end_box
        self.ctrl_duration = self.wp_duration
        self.ctrl_fps = self.wp_fps
        self.ctrl_duration.setToolTip("Manual mode: trajectory duration in seconds (uniform arc-length resampling). Disabled in JSON mode - duration comes from the frame range.")
        self.ctrl_fps.setToolTip("Frames per second: manual resampling rate (Manual) / OpenLABEL simulation rate (JSON)")
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

        # Actions row (U19: Load DB before Find Similar - workflow order)
        act_row = QHBoxLayout()
        self.ctrl_generate = QPushButton("Generate")
        self.ctrl_generate.setToolTip("Resample both waypoint sets into frame trajectories (Manual mode)")
        self.ctrl_apply = QPushButton("Apply")
        self.ctrl_apply.setToolTip("Build the scenario context from the current trajectories")
        self.ctrl_clear = QPushButton("Clear")
        self.ctrl_clear.setToolTip("Clear the current mode's trajectories (and waypoints in Manual mode)")
        self.ctrl_build_timeline = QPushButton("Build/Refresh DB Index")
        self.ctrl_build_timeline.setToolTip("Index the loaded OpenLABEL file into an event-sequence database (vehicle + pair tracks)")
        self.ctrl_load_db = QPushButton("Load Semantic DB")
        self.ctrl_load_db.setToolTip("Load a previously built event-sequence database")
        self.ctrl_find_similar = QPushButton("Find Similar")
        self.ctrl_find_similar.setToolTip("Find similar scenarios in the loaded DB (needs a generated base scenario; otherwise lists candidate windows)")
        act_row.addWidget(self.ctrl_generate)
        act_row.addWidget(self.ctrl_apply)
        act_row.addWidget(self.ctrl_clear)
        act_row.addWidget(self.ctrl_build_timeline)
        act_row.addSpacing(20)
        act_row.addWidget(self.ctrl_load_db)
        act_row.addWidget(self.ctrl_find_similar)
        act_row.addStretch()
        layout.addLayout(act_row)

        layout.addWidget(QLabel("Similarity Matches"))
        self.similar_list = QListWidget()
        self.similar_list.setMinimumHeight(120)      # U16: was 220 (broke the 200 px panel)
        self.similar_list.setToolTip("Click a match to preview its trajectories on the map (yellow = A, magenta = B)")
        self.similar_list.itemSelectionChanged.connect(self._on_similarity_selected)
        layout.addWidget(self.similar_list)

        status_row = QHBoxLayout()
        status_row.addWidget(self.json_status)
        self.similarity_status = QLabel("No similarity search executed")
        status_row.addWidget(self.similarity_status)
        status_row.addStretch()
        self.chk_save_debug = QCheckBox("Save debug contexts")     # U7
        self.chk_save_debug.setToolTip("Write every engine context to debug_contexts/ (off by default)")
        status_row.addWidget(self.chk_save_debug)
        layout.addLayout(status_row)

        self.ctrl_generate.clicked.connect(self._on_generate_clicked)
        self.ctrl_apply.clicked.connect(self._on_apply_clicked)
        self.ctrl_clear.clicked.connect(self._on_clear_clicked)
        self.ctrl_find_similar.clicked.connect(self.find_similar_scenarios)
        self.ctrl_build_timeline.clicked.connect(self.build_semantic_timeline_database)
        self.ctrl_load_db.clicked.connect(self.load_semantic_database_dialog)
        self.chk_save_debug.stateChanged.connect(
            lambda state: setattr(self, "_save_debug_contexts", bool(state)))
        self.v1_id_box.currentTextChanged.connect(self._refresh_action_states)
        self.v2_id_box.currentTextChanged.connect(self._refresh_action_states)
        self._refresh_action_states()
        return panel


    def _build_info_panel(self):
        row = QHBoxLayout()
        self.status_label = QLabel()
        self.status_label.setFixedHeight(60)
        self.info_label = QLabel()
        self.info_label.setWordWrap(True)
        self.info_label.setFixedHeight(80)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)          # U8: the log was editable
        self.log_view.setFixedHeight(80)
        log_col = QVBoxLayout()
        log_col.setContentsMargins(0, 0, 0, 0)
        log_row = QHBoxLayout()
        log_row.setContentsMargins(0, 0, 0, 0)
        log_row.addWidget(self.log_view, 5)
        self.btn_clear_log = QPushButton("Clear")   # U8
        self.btn_clear_log.setFixedWidth(44)
        self.btn_clear_log.setToolTip("Clear the log")
        self.btn_clear_log.clicked.connect(self.log_view.clear)
        log_row.addWidget(self.btn_clear_log)
        log_col.addLayout(log_row)
        row.addWidget(self.status_label, 2)
        row.addWidget(self.info_label, 3)
        row.addLayout(log_col, 2)
        self.bottom_layout.addLayout(row)


    # ===================== mode / editing =====================

    def _on_mode_changed(self, txt):
        previous_mode = self.active_mode
        self.active_mode = "manual" if txt.lower() == "manual" else "json"
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
        self._refresh_action_states()              # U12: per-mode enable/visibility
        self._hide_all_trajectories()
        self._show_active_mode_trajectories()
        self._update_waypoint_visibility()
        self.clear_similarity_preview()            # U13: no stale previews across modes
        self.update_status_panel()
        self.log_info(f"Mode changed: {previous_mode.upper()} -> {self.active_mode.upper()}")
        self._emit_active_context()

    def _clear_edit_selection(self):
        self.current_wp_vehicle = None
        self.view.active_vehicle = None
        self.view.dragging_wp = None
        self.view.unsetCursor()                    # U3: restore cursor when editing ends
        self.ctrl_v1_btn.setStyleSheet("")
        self.ctrl_v2_btn.setStyleSheet("")


    def _set_wp_mode(self, vehicle):
        if not self.is_manual_mode:
            return
        self._clear_edit_selection()          # reset previous vehicle + styles
        self.current_wp_vehicle = vehicle
        self.view.enable_waypoints(vehicle)
        (self.ctrl_v1_btn if vehicle == "V1" else self.ctrl_v2_btn).setStyleSheet(
            "background:#4CAF50;color:white;font-weight:bold;")
        self.log_info(f"Editing {vehicle} waypoints")
        self.update_status_panel()

    def _clear_all_waypoints(self):
        self.view.clear_waypoints()
        self.clear_all_trajectories()
        self.current_wp_vehicle = None
        self.view.active_vehicle = None
        self.update_status_panel()
        self.log_info("Waypoints and trajectories cleared.")

    def _update_waypoint_visibility(self):
        for v in ("V1", "V2"):
            for item in self.view.wp_items[v]:
                item.setVisible(self.is_manual_mode)
            if self.wp_line.get(v) is not None:      # U3
                self.wp_line[v].setVisible(self.is_manual_mode)

    def _on_clear_clicked(self):
        self._invalidate_active_scenario("clear_clicked")
        self.clear_similarity_preview()            # U13
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
        self._refresh_action_states()              # U12
        self.log_info("Cleared current mode data.")


    # ===================== generate / apply =====================

    def _on_generate_clicked(self):
        if not self.is_manual_mode:
            return
        for v in ("V1", "V2"):
            self.current_wp_vehicle = v
            self._generate_wp_trajectory()

    def _on_apply_clicked(self):
        if self.is_manual_mode:
            self.apply_manual_trajectory()
        else:
            self.apply_loaded_trajectories()
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json
        actors = ctx.get("actors", {})
        # U9: the Apply result used to go to print() only - now it is visible
        self.log_info(
            f"Context ready={ctx.get('ready')} | "
            f"V1: {len(actors.get('A', {}).get('timeline', {}))} frames | "
            f"V2: {len(actors.get('B', {}).get('timeline', {}))} frames | "
            f"pair: {len(ctx.get('interaction_facts', {}).get('timeline', {}))} frames"
            + (f" | error: {ctx.get('error')}" if ctx.get("error") else "")
        )


    def apply_manual_trajectory(self):
        if self._manual_dirty:
            # U2: stale trajectories + new duration = silently wrong speeds/times
            return self.log_warn("Timing changed - press Generate first (stale trajectories would be applied).")
        for v in ("V1", "V2"):
            if len(self._traj_map_manual.get(v, [])) < 2:
                return self.log_warn(f"{v} trajectory too short.")
            world = self._world_polylines_manual.get(v)
            if not world or len(world) < 2:
                return self.log_warn(f"{v} world trajectory invalid.")
        self._engine_context = None
        self.emit_all_vehicle_map_contexts()
        self.log_info("Manual trajectories applied")


    def _generate_wp_trajectory(self):
        v = self.current_wp_vehicle
        if not v or v not in ("V1", "V2"):
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
        store = self._get_active_store()
        store["traj_map"][v] = traj
        store["world"][v] = world_pts
        self._rebuild_vehicle_context_cache()
        color = QColor("green" if v == "V1" else "cyan")
        self.clear_traj_items(store["items"].get(v, []))
        store["items"][v] = self.view.draw_trajectory(px_pts, color)
        self._manual_traj_done[v] = True
        self._rebuild_hover_index()
        self._update_waypoint_visibility()
        self.log_info(f"{v} trajectory generated ({len(px_pts)} frames)")
        self.update_status_panel()
        self.polyline_committed.emit(v, json.dumps({"px": px_pts, "world": world_pts}))
        self._refresh_action_states()              # U12


    def _resample_waypoints_time_based(self, pts_px, duration_s, fps):
        """Waypoint polyline -> [(frame, x_px, y_px), ...], uniform in arc length."""
        if len(pts_px) < 2:
            return []
        total_frames = max(2, int(round(duration_s * fps)))
        arr = np.asarray(pts_px, dtype=float)
        seglen = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seglen)])
        total_len = cum[-1]
        if total_len < 1e-6:
            return [(i, arr[0][0], arr[0][1]) for i in range(total_frames)]
        target = np.linspace(0, total_len, total_frames)
        xs = np.interp(target, cum, arr[:, 0])
        ys = np.interp(target, cum, arr[:, 1])
        return [(i, float(xs[i]), float(ys[i])) for i in range(total_frames)]

    def _on_manual_timing_changed(self):
        if self.is_manual_mode:
            if not any(self._manual_traj_done.values()):
                return
            self._manual_dirty = True
            self._invalidate_active_scenario("manual_timing_changed")
            self.log_warn("Manual timing changed. Press Generate to rebuild trajectories.")
        else:
            # U4: FPS used to take effect only when the frame range changed again
            self.fps = float(self.wp_fps.value())
            if self._traj_map_json:
                self._json_dirty = True
                self._invalidate_active_scenario("json_fps_changed")
                self.log_warn("FPS changed. Press Apply to reload trajectories.")
            self.compute_and_emit_duration()
        self.update_status_panel()


    def _on_json_frame_range_changed(self):
        if self.active_mode != "json":
            return
        try:
            fstart = int(self.global_start_box.currentText())
            fend = int(self.global_end_box.currentText())
        except ValueError:
            return
        if fstart > fend:
            fstart, fend = fend, fstart
        self.num_frames = max(1, fend - fstart + 1)
        self.fps = float(self.wp_fps.value())
        if self._traj_map_json:
            self._json_dirty = True
            self._invalidate_active_scenario("json_frame_range_changed")
            self.log_warn("Frame range changed. Press Apply to reload trajectories.")
        self.update_vehicle_ids_for_duration()
        self.compute_and_emit_duration()
        self.update_status_panel()

    # ===================== stores / caches =====================

    def _get_active_traj_map(self):
        if self.is_manual_mode:
            return self._traj_map_manual
        if self.active_mode == "json":
            return self._traj_map_json
        return {}

    def _get_active_store(self):
        return {
            "traj_map": self._traj_map_manual if self.is_manual_mode else self._traj_map_json,
            "world": self._world_polylines_manual if self.is_manual_mode else self._world_polylines_json,
            "context": self._map_context_manual if self.is_manual_mode else self._map_context_json,
            "items": self._traj_items.setdefault(self.active_mode, {"V1": [], "V2": []}),
        }

    def _get_valid_duration(self):
        if self.is_manual_mode:
            return float(self.wp_duration.value())
        if self.sim_duration_s and self.sim_duration_s > 0:
            return float(self.sim_duration_s)
        if self.fps > 0 and self.num_frames > 0:
            return float(max(self.num_frames - 1, 1)) / float(self.fps)
        return 0.1

    def compute_and_emit_duration(self):
        if self.fps <= 0 or self.num_frames <= 0:
            self.sim_duration_s = None
            return None
        duration = float(max(self.num_frames - 1, 1)) / max(self.fps, 1.0)
        self.sim_duration_s = duration
        self.duration_changed.emit(duration)
        return duration

    def _rebuild_vehicle_context_cache(self):
        self._vehicle_context_cache.clear()
        for vid, traj in self._get_active_traj_map().items():
            frames = [int(fr) for fr, _, _ in traj]
            if not frames:
                continue
            self._vehicle_context_cache[vid] = {
                "frame_lookup": {int(fr): (float(x), float(y)) for fr, x, y in traj},
                "frames": frames,
                "frame_set": set(frames),
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "length": len(frames),
            }

    def _rebuild_hover_index(self):
        self._hover_index = []
        self._hover_tree = None
        points, meta = [], []
        for vid, traj in self._get_active_traj_map().items():
            for fr, x, y in traj:
                points.append((float(x), float(y)))
                meta.append((vid, int(fr)))  # frame id, so hover lookups match timeline keys
        if points:
            self._hover_tree = cKDTree(points)
            self._hover_index = meta

    def _get_hover_vehicle_insight(self, vid, frame):
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json
        actor = "A" if vid == "V1" else "B"
        sample = ctx.get("actors", {}).get(actor, {}).get("timeline", {}).get(frame)
        if not sample:
            return None
        sd = sample.get("sd", {})
        return {"s": sd.get("s"), "d": sd.get("d"), "t": frame, "region": sample.get("region_id")}

    def _get_hover_pair_insight(self, frame):
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json
        p = ctx.get("interaction_facts", {}).get("timeline", {}).get(frame)
        if not p:
            return None
        return {"dist": p.get("distance"), "ds": p.get("delta_s"), "dd": p.get("delta_d")}

    # ===================== component loading =====================

    def _try_init_road(self):
        if self.view.mask is None or self.view.meta is None:
            return
        if not self.road:
            self.road = RoadInfrastructure(self._layout_path)
        self.road.attach_mask_meta(mask=self.view.mask, meta=self.view.meta)
        self._attach_homography_to_road()
        self._rebuild_context_builder()

    def _attach_homography_to_road(self):
        if self.road and self.homography:
            self.road.set_homography(self.homography["H"], self.homography["Hinv"])

    def _rebuild_context_builder(self):
        if self.road:
            self._ctx_builder = ContextBuilder(self.road, self.signal_config)

    def load_mask_from_path(self, path: str):
        try:
            self.view.load_mask(path)
            self._mask_path = path
            self._project["mask"] = path
            if self.view.mask is not None:
                self.mask_loaded.emit(self.view.mask)
            self._try_init_road()
            # U5: centerlines were only drawn when the layout loaded; if the mask
            # loads AFTER the layout they now appear as well.
            if self.road is not None:
                self.view.draw_centerlines(self.road)
            self._refresh_action_states()            # U12
        except Exception as e:
            self.log_warn(f"Failed to load mask: {e}")


    def load_meta_from_path(self, path: str):
        try:
            meta = self.view.load_meta(path)
        except Exception as e:
            self.log_warn(f"Failed to load layout: {e}")
            return
        try:
            self.road = RoadInfrastructure(path)
            self.road.attach_mask_meta(mask=self.view.mask, meta=meta)
            self.road.populate_centerlines(region_type=["lane", "connecting_lane"])
            self._centerline_geom_cache.clear()
            self._build_semantic_lane_roles()
            self.view.draw_centerlines(self.road)
            self.log_info("RoadInfrastructure initialized")
        except Exception as e:
            self.log_warn(f"RoadInfrastructure failed: {e}")
            if not self.road or not self.road.regions:
                self.log_warn("RoadInfrastructure has no regions")
                self._refresh_action_states()        # U12
                return
        self._layout_path = path
        self._project["layout"] = path
        if meta is not None:
            self.meta_loaded.emit(meta)
        self._attach_homography_to_road()
        self._rebuild_context_builder()
        self._refresh_action_states()                # U12


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
                raise ValueError("Homography matrices must be 3x3")
            if not (np.isfinite(H).all() and np.isfinite(Hinv).all()):
                raise ValueError("Homography contains invalid values")
            self.homography = {"H": H, "Hinv": Hinv, "raw": data}
            self._homog_path = path
            self._project["homography"] = path
            self.log_info(f"Homography loaded successfully from {path}")
        except Exception as e:
            self.homography = None
            self.log_warn(f"Failed to load homography: {e}")
            self._refresh_action_states()            # U12
            return
        self._attach_homography_to_road()
        self._refresh_action_states()                # U12


    def load_xodr_from_path(self, path: str):
        self._xodr_path = path
        self._project["xodr"] = path
        self.xodr_loaded.emit(path)
        self.xodr_changed.emit(path)
        self.log_info(f"OpenDRIVE: {path}")

    def load_openlabel_from_path(self, path: str):
        try:
            frames, traj_map, fmin, fmax = load_openlabel(path)
        except Exception as e:
            self.log_warn(f"Failed to load OpenLABEL: {e}")
            return
        self.openlabel_frames = frames
        self.openlabel_traj_map = traj_map
        self._builder_traj_cache.clear()
        self._openlabel_vehicle_ctx_cache.clear()
        self.openlabel_fmin = fmin
        self.openlabel_fmax = fmax
        self.num_frames = max(0, fmax - fmin + 1)
        self.fps = float(self.wp_fps.value())
        self.compute_and_emit_duration()
        self._openlabel_path = path
        self._project["openlabel"] = path
        fr_list = [str(f) for f in range(fmin, fmax + 1)]
        for box in (self.global_start_box, self.global_end_box):
            box.blockSignals(True)
            box.clear()
            box.addItems(fr_list)
        self.global_start_box.setCurrentText(str(fmin))
        self.global_end_box.setCurrentText(str(fmax))
        for box in (self.global_start_box, self.global_end_box):
            box.blockSignals(False)
        self.update_vehicle_ids_for_duration()
        QTimer.singleShot(0, self.update_vehicle_ids_for_duration)
        self._refresh_action_states()                # U12


    def new_project(self):
        # U12: confirm before destroying work
        has_work = (
            any(self.view.manual_waypoints.values())
            or bool(self._traj_map_manual)
            or bool(self._traj_map_json)
        )
        if has_work:
            ret = QMessageBox.question(
                self, "New Project",
                "Start a new project? Current waypoints, trajectories and context will be lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        self._invalidate_active_scenario("new_project")
        self._project = {k: None for k in self._project}
        self._mask_path = self._layout_path = self._homog_path = self._xodr_path = self._openlabel_path = None
        self._db_path = None
        self._project_path = None
        # U12/U21: full state reset (previously the canvas kept old waypoints/trajectories)
        self.view.clear_waypoints()
        self._manual_traj_done = {"V1": False, "V2": False}
        self._traj_map_manual.clear()
        self._traj_map_json.clear()
        self._vehicle_context_cache.clear()
        self._world_polylines_manual = {"V1": None, "V2": None}
        self._world_polylines_json = {"V1": None, "V2": None}
        for mode_items in self._traj_items.values():
            for v in ("V1", "V2"):
                self.clear_traj_items(mode_items.get(v, []))
                mode_items[v] = []
        self._map_context_manual = {"ready": False}
        self._map_context_json = {"ready": False}
        self._engine_context = None
        self.clear_similarity_preview()
        self._refresh_action_states()
        self.log_info("New Project: Browse files for your project (you can skip any).")
        mask = self._load_with_dialog("Select Mask", "Mask Files (*.png *.tif *.tiff *.npy)")
        if mask:
            self.load_mask_from_path(mask)
        for title, filt, loader in (
            ("Select Layout JSON", "JSON Files (*.json)", self.load_meta_from_path),
            ("Select Homography JSON", "JSON Files (*.json)", self.load_homography_from_path),
            ("Select OpenDRIVE (xODR)", "OpenDRIVE (*.xodr)", self.load_xodr_from_path),
            ("Select OpenLABEL JSON (optional)", "JSON Files (*.json)", self.load_openlabel_from_path),
        ):
            p = self._load_with_dialog(title, filt)
            if p:
                loader(p)
        ret = QMessageBox.question(
            self, "Save Project",
            "Save this project (paths) to a .json file now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
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
            self.log_warn(f"Failed to read project: {e}")
            return
        self._project = {k: proj.get(k) for k in ("mask", "layout", "homography", "xodr", "openlabel", "db")}
        # resolve relative paths against the project file, not the CWD
        base = os.path.dirname(os.path.abspath(path))
        for k, p in self._project.items():
            if p and not os.path.isabs(p):
                self._project[k] = os.path.join(base, p)
        self._project_path = path
        self._db_path = self._project.get("db")

        loaders = {
            "mask": self.load_mask_from_path,
            "layout": self.load_meta_from_path,
            "homography": self.load_homography_from_path,
            "xodr": self.load_xodr_from_path,
            "openlabel": self.load_openlabel_from_path,
        }
        missing = []
        for key, loader in loaders.items():
            p = self._project.get(key)
            if not p:
                continue
            if not os.path.exists(p):
                missing.append(key)
                self.log_warn(f"Project file missing: {key} -> {p}")
                continue
            loader(p)

        # restore manual waypoints
        wps = proj.get("waypoints") or {}
        for v in ("V1", "V2"):
            try:
                pts = [(float(x), float(y)) for x, y in (wps.get(v) or [])]
            except (TypeError, ValueError):
                pts = []
            self.view.manual_waypoints[v] = pts
            if pts:
                self.view._draw_waypoints(v)
        if any(wps.values()):
            self.log_info(f"Restored waypoints: V1={len(wps.get('V1', []))}, "
                          f"V2={len(wps.get('V2', []))} - press Generate to rebuild trajectories")

        # auto-load the project's event-sequence DB (no rebuild needed)
        if self._db_path:
            if os.path.exists(self._db_path):
                try:
                    self.load_semantic_database(self._db_path)
                except Exception as e:
                    self.log_warn(f"Project DB could not be loaded: {e}")
            else:
                self.log_warn(f"Project DB file missing: {self._db_path} - build the index again")

        if missing:
            QMessageBox.warning(
                self, "Project",
                f"Project loaded with missing files: {', '.join(missing)}")
        else:
            self.log_info("Project loaded.")
        self._refresh_action_states()
        self.update_status_panel()


    def _sync_project_file(self):
        """Rewrite the .trace file (paths + waypoints + DB) when one is known,
        so the next Open Project restores the index without rebuilding."""
        if not self._project_path or not os.path.exists(self._project_path):
            return
        proj = {
            "mask": self._mask_path,
            "layout": self._layout_path,
            "homography": self._homog_path,
            "xodr": self._xodr_path,
            "openlabel": self._openlabel_path,
            "db": self._db_path,
            "waypoints": {
                v: [[float(x), float(y)] for x, y in self.view.manual_waypoints[v]]
                for v in ("V1", "V2")
            },
        }
        try:
            with open(self._project_path, "w", encoding="utf-8") as f:
                json.dump(proj, f, ensure_ascii=False, indent=2)
            self.log_info(f"Project updated: {self._project_path}")
        except Exception as e:
            self.log_warn(f"Could not update project file: {e}")


    def save_project(self):
        proj = {
            "mask": self._mask_path,
            "layout": self._layout_path,
            "homography": self._homog_path,
            "xodr": self._xodr_path,
            "openlabel": self._openlabel_path,
            "db": self._db_path,
            "waypoints": {
                v: [[float(x), float(y)] for x, y in self.view.manual_waypoints[v]]
                for v in ("V1", "V2")
            },
        }
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "project.trace", "Project (*.trace)")
        if not path:
            return
        if not path.lower().endswith(".trace"):
            path += ".trace"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(proj, f, ensure_ascii=False, indent=2)
            self._project_path = path
            self.log_info(f"Project saved to {path}")
        except Exception:
            self.log_warn("Failed to save project.")



    def load_mask(self):
        path = self._load_with_dialog("Open Mask", "Mask Files (*.png *.tif *.tiff *.npy)")
        if path:
            self.load_mask_from_path(path)

    def load_homography_json(self):
        path = self._load_with_dialog("Open Homography JSON", "JSON Files (*.json)")
        if path:
            self.load_homography_from_path(path)

    def load_xodr(self):
        path = self._load_with_dialog("Open xODR", "OpenDRIVE (*.xodr)")
        if path:
            self.load_xodr_from_path(path)

    def _load_with_dialog(self, title, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter_str)
        return path or None

    # ===================== trajectory display =====================

    def _hide_all_trajectories(self):
        for mode in self._traj_items.values():
            for veh_items in mode.values():
                for it in veh_items:
                    if it:
                        it.setVisible(False)

    def _show_active_mode_trajectories(self):
        for veh_items in self._traj_items.get(self.active_mode, {}).values():
            for it in veh_items:
                if it:
                    it.setVisible(True)

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
            x, y = pt[-2], pt[-1]  # accepts (x, y) or (frame, x, y)
            try:
                wp = self.road.pixel_to_world(x, y)
                if wp and len(wp) >= 2 and np.isfinite(wp[0]) and np.isfinite(wp[1]):
                    world.append((float(wp[0]), float(wp[1])))
            except Exception:
                pass
        return world if len(world) >= 2 else None

    # ===================== map context construction =====================

    def _empty_context(self):
        return {
            "kind": "map_scenario_context",
            "ready": False,
            "state": "init",
            "error": None,
            "actors": {},
            "interaction_facts": {},
            "interaction_topology": {},
            "meta": {},
        }

    def _fail_ctx(self, ctx, err):
        ctx.update(ready=False, state="failed", error=err, actors={},
                   interaction_facts={}, interaction_topology={})
        return ctx

    def _build_vehicle_contexts(self):
        traj_map = self._get_active_traj_map()
        v1_traj = traj_map.get("V1")
        v2_traj = traj_map.get("V2")
        if not traj_map:
            return None, "missing_trajectories"
        if not v1_traj or not v2_traj:
            return None, "incomplete_vehicles"
        self._rebuild_context_builder()
        try:
            traj1 = self._traj_to_builder_format(v1_traj)
            traj2 = self._traj_to_builder_format(v2_traj)
            return (self._ctx_builder.build_vehicle_context(traj1),
                    self._ctx_builder.build_vehicle_context(traj2)), None
        except Exception as e:
            return None, f"vehicle_ctx_failed: {e}"

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
            raw_rid = rid = None
            if self.road:
                try:
                    res = self.road.region_lookup(int(x), int(y)) or {}
                    raw_rid = rid = res.get("region_id")
                    if rid is not None and wp_obj:
                        resolved = self._resolve_to_lane(rid, wp_obj, last_lane)
                        if resolved is not None:
                            rid = resolved
                except Exception:
                    pass
            proj = None
            if self.road and wp and rid is not None:
                try:
                    proj = self.road.project_world_to_frenet(rid, wp[0], wp[1])
                except Exception:
                    proj = None
            if not self._valid_proj(proj) and raw_rid is not None and wp_obj:
                resolved = self._resolve_to_lane(raw_rid, wp_obj, last_lane)
                if resolved is not None:
                    try:
                        proj2 = self._project_to_centerline(resolved, wp)
                        if not self._valid_proj(proj2) and last_lane:
                            proj2 = self._project_to_centerline(last_lane, wp)
                        if self._valid_proj(proj2):
                            proj, rid = proj2, resolved
                    except Exception:
                        pass
            if not self._valid_proj(proj) and last_lane is not None and wp:
                try:
                    proj2 = self.road.project_world_to_frenet(last_lane, wp[0], wp[1])
                    if proj2:
                        proj, rid = proj2, last_lane
                except Exception:
                    pass
            sd = {"s": float("nan"), "d": float("nan")}
            if self._valid_proj(proj):
                s_local, d = proj.get("s"), proj.get("d")
                if last_lane == rid and last_s_local is not None:
                    ds = s_local - last_s_local
                    global_s += max(ds, 0.0) if abs(ds) < 5.0 else max(s_local, 0.0)
                elif last_s_local is not None:
                    global_s += max(s_local, 0.0)
                sd = {"s": float(global_s), "d": float(d)}
                last_s_local = s_local
                last_lane = rid
            out.append((frame_id, {
                "world": world,
                "region_id": rid,
                "sd": sd,
                "topology": self._semantic_lane_roles.get(str(rid), ()) if rid is not None else (),
            }))
        return out

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
        for m in members:
            rc_m = self.road.get_region_context(m) or {}
            if rc_m.get("type") not in ("lane", "connecting_lane"):
                continue
            try:
                proj = self.road.project_world_to_frenet(m, wp_obj.x, wp_obj.y) if wp_obj else None
            except Exception:
                proj = None
            if not self._valid_proj(proj):
                continue
            score = (10.0 if last_lane == m else 0.0) - abs(proj["d"])
            candidates.append((m, score))
        if candidates:
            candidates.sort(key=lambda t: t[1], reverse=True)
            return candidates[0][0]
        if wp_obj:
            best, best_dist = None, float("inf")
            for m in members:
                reg = self.road.regions.get(m)
                if not reg or reg.type not in ("lane", "connecting_lane"):
                    continue
                centerline = getattr(reg, "centerline", None)
                if not centerline or len(centerline) < 2:
                    continue
                pts = np.asarray(centerline)
                dmin = float(np.min(np.linalg.norm(pts - np.array([wp_obj.x, wp_obj.y]), axis=1)))
                if dmin < best_dist:
                    best, best_dist = m, dmin
            if best is not None:
                return best
        return members[0]

    def _build_semantic_lane_roles(self):
        self._semantic_lane_roles = {}
        meta = getattr(self.road, "meta", None) if self.road else None
        if not meta:
            return
        for rid, info in meta.items():
            if not isinstance(info, dict) or info.get("type") not in ("lane", "connecting_lane"):
                continue
            succ = info.get("successors", []) or []
            pred = info.get("predecessors", []) or []
            connecting = info.get("is_connecting_road", False)
            role = []
            if connecting:
                role.append("CONNECTOR")
            if not pred:
                role.append("ENTRY")
            if not succ:
                role.append("EXIT")
            if pred and succ and not connecting:
                role.append("FLOW")
            if len(succ) >= 2:
                role.append("SPLIT")
            if len(pred) >= 2:
                role.append("MERGE")
            junction_id = info.get("junction_id")
            if junction_id is not None:
                role.append(f"JUNCTION_{junction_id}")
            self._semantic_lane_roles[str(rid)] = tuple(role)

    def _get_lane_semantic_debug(self, lane_id):
        rc = None
        try:
            rc = self.road.get_region_context(lane_id)
        except Exception:
            pass
        if not rc:
            return {}
        return {
            "roles": list(self._semantic_lane_roles.get(str(lane_id), ())),
            "successors": rc.get("successors", []),
            "predecessors": rc.get("predecessors", []),
            "adjacent": rc.get("adjacent", []),
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
        p1, p2 = pts[:-1], pts[1:]
        v = p2 - p1
        seg_len = np.linalg.norm(v, axis=1)
        result = {
            "pts": pts, "p1": p1, "p2": p2, "v": v, "seg_len": seg_len,
            "valid": seg_len > 1e-6,
            "cumulative": np.concatenate([[0.0], np.cumsum(seg_len)]),
        }
        self._centerline_geom_cache[lane_id] = result
        return result

    @staticmethod
    def _valid_proj(proj):
        if not proj:
            return False
        s, d = proj.get("s"), proj.get("d")
        return (isinstance(s, (int, float)) and isinstance(d, (int, float))
                and not math.isnan(s) and not math.isnan(d))

    def _project_to_centerline(self, lane_id, wp):
        geom = self._get_centerline_geom(lane_id)
        if not geom:
            return None
        q = np.asarray(wp, dtype=np.float64)
        p1_arr, v_arr, seg_len_arr, valid_arr, cumulative = (
            geom["p1"], geom["v"], geom["seg_len"], geom["valid"], geom["cumulative"])
        best_s = best_d = None
        best_dist = float("inf")
        for i in range(len(v_arr)):
            if not valid_arr[i]:
                continue
            p1, v, seg_len = p1_arr[i], v_arr[i], seg_len_arr[i]
            w = q - p1
            t = float(np.dot(w, v)) / (seg_len * seg_len)
            t = min(max(t, 0.0), 1.0)
            diff = q - (p1 + t * v)
            dist = float(np.hypot(diff[0], diff[1]))
            if dist < best_dist:
                best_dist = dist
                best_s = cumulative[i] + t * seg_len
                sign = np.sign(np.dot(w, np.array([-v[1], v[0]], dtype=np.float64)))
                best_d = sign * dist
        if best_s is None:
            return None
        return {"s": float(best_s), "d": float(best_d)}

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
            ctx_a["timeline"] = {k: ctx_a["timeline"][k] for k in common}
            ctx_b["timeline"] = {k: ctx_b["timeline"][k] for k in common}
            pair = self._ctx_builder.build_pairwise(ctx_a, ctx_b)
            pair_tl = pair.get("timeline", {})
            pair["timeline"] = {
                k: pair_tl.get(k, {"distance": float("nan"), "delta_s": float("nan"),
                                   "delta_d": float("nan")})
                for k in common
            }
        except Exception as e:
            return self._fail_ctx(ctx, f"pairwise_failed: {e}")
        if not pair or not pair.get("timeline"):
            return self._fail_ctx(ctx, "empty_pairwise")
        return self._build_pair_scenario_context(
            actor_a_id=self.v1_id_box.currentText() or "V1",
            actor_b_id=self.v2_id_box.currentText() or "V2",
            ctx_a=ctx_a, ctx_b=ctx_b, pair=pair,
            start_frame=common[0], end_frame=common[-1],
        )

    def _build_normalize_validate_emit(self):
        ctx = self.build_context_atomic()
        try:
            ctx = normalize_context(ctx)
            validate_context(ctx)
            ctx["ready"] = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("[CONTEXT VALIDATION FAILED]", type(e).__name__, str(e))
            ctx["ready"] = False
            ctx["error"] = str(e)
        if self.is_manual_mode:
            self._map_context_manual = ctx
        else:
            self._map_context_json = ctx
        return ctx

    def build_semantic_context(self, ctx):
        """
        Enrich a map-based context with scenario-engine events and produce a
        normalized event timeline for indexing.

        - Event times keep their source unit; ratios are normalized against the
          observed event extent and clamped to [0, 1].
        - Location: prefer the event's own region, fall back to
          features/interaction/metrics; transition events keep from/to region
          and use the destination as primary region.
        """
        if not isinstance(ctx, dict) or not ctx:
            return {}
        ctx = ctx.copy()
        ctx["ready"] = True

        def finite(value, default=None):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            return value if math.isfinite(value) else default

        def first(*values):
            return next((v for v in values if v is not None), None)

        def event_name(event):
            value = first(event.get("type"), event.get("event"),
                          event.get("name"), event.get("label"))
            if isinstance(value, dict):
                value = first(value.get("type"), value.get("event"),
                              value.get("name"), value.get("label"))
            return str(value) if value is not None else "unknown"

        duration = max(finite(ctx.get("duration_seconds"), 1.0), 1e-6)
        ctx = build_engine_insight_context(context=ctx, duration=duration)
        rules = suggest_rules_combined(duration=duration, meta=ctx.get("meta"), context=ctx)

        valid_events = []
        for idx, event in enumerate(ctx.get("event_timeline", [])):
            if not isinstance(event, dict):
                continue
            t0 = finite(first(event.get("t_start"), event.get("start_t"), event.get("frame_start")))
            if t0 is None:
                continue
            t1 = finite(first(event.get("t_end"), event.get("end_t"), event.get("frame_end"), t0))
            if t1 is None:
                t1 = t0
            if t1 < t0:
                t0, t1 = t1, t0
            valid_events.append((idx, event, t0, t1))

        if not valid_events:
            ctx.update(event_timeline=[], event_transitions=[], event_stats={})
            return ctx

        min_time = min(e[2] for e in valid_events)
        max_time = max(e[3] for e in valid_events)
        span = max_time - min_time
        if span <= 1e-9:
            span = duration
        span = max(span, 1e-6)

        loc_srcs = (None, "location", "features", "interaction", "metrics")

        def region_of(event, keys):
            sources = [event] + [event.get(k) for k in loc_srcs[1:] if isinstance(event.get(k), dict)]
            return first(*(d.get(k) for d in sources for k in keys))

        normalized = []
        for event_id, event, t0, t1 in valid_events:
            metrics = event.get("metrics") if isinstance(event.get("metrics"), dict) else {}
            features = event.get("features") if isinstance(event.get("features"), dict) else {}
            interaction = event.get("interaction") if isinstance(event.get("interaction"), dict) else {}

            participants = event.get("participants", [])
            if isinstance(participants, str):
                participants = [participants]
            elif not isinstance(participants, (list, tuple, set)):
                participants = []

            region = region_of(event, ("region", "region_id"))
            from_region = region_of(event, ("from_region", "from_lane"))
            to_region = region_of(event, ("to_region", "to_lane"))
            if region is None:
                region = first(to_region, from_region)

            dur = max(0.0, t1 - t0)
            start_ratio = min(1.0, max(0.0, (t0 - min_time) / span))
            end_ratio = min(1.0, max(start_ratio, (t1 - min_time) / span))
            duration_ratio = min(1.0, max(0.0, dur / span))

            normalized.append({
                "event_id": int(event_id),
                "event": event_name(event),
                "start_t": t0,
                "end_t": t1,
                "duration": dur,
                "start_ratio": start_ratio,
                "end_ratio": end_ratio,
                "duration_ratio": duration_ratio,
                "participants": sorted(str(v) for v in participants if v is not None),
                "region": region,
                "from_region": from_region,
                "to_region": to_region,
                "features": features,
                "interaction": interaction,
                "metrics": metrics,
            })

        normalized.sort(key=lambda e: (e["start_t"], e["end_t"], e["event"], e["event_id"]))

        transitions = []
        for previous, current in zip(normalized[:-1], normalized[1:]):
            delta_t = current["start_t"] - previous["end_t"]
            transitions.append({
                "src": previous["event"],
                "dst": current["event"],
                "src_event_id": previous["event_id"],
                "dst_event_id": current["event_id"],
                "delta_t": delta_t,
                "delta_ratio": max(0.0, delta_t / span),
                "overlaps": delta_t < 0.0,
            })

        stats = {}
        for event in normalized:
            bucket = stats.setdefault(event["event"],
                                      {"count": 0, "durations": [], "start_ratios": [], "regions": {}})
            bucket["count"] += 1
            bucket["durations"].append(event["duration"])
            bucket["start_ratios"].append(event["start_ratio"])
            if event.get("region") is not None:
                key = str(event["region"])
                bucket["regions"][key] = bucket["regions"].get(key, 0) + 1
        for bucket in stats.values():
            durations = bucket.pop("durations")
            start_ratios = bucket.pop("start_ratios")
            bucket["mean_duration"] = float(np.mean(durations)) if durations else 0.0
            bucket["max_duration"] = float(np.max(durations)) if durations else 0.0
            bucket["mean_start_ratio"] = float(np.mean(start_ratios)) if start_ratios else 0.0

        ctx["event_time_origin"] = min_time
        ctx["event_time_span"] = span
        ctx["event_timeline"] = normalized
        ctx["event_transitions"] = transitions
        ctx["event_stats"] = stats
        if "scenario_descriptor" not in ctx.get("derived", {}):
            ctx.setdefault("derived", {})["scenario_descriptor"] = build_scenario_descriptor(ctx, rules)
        return ctx

    def _stamp_frame_times(self, ctx):
        """Give every actor frame a real 't' in seconds.

        Without this, scenario_engine.extract_sd_sequences falls back to
        t = frame_index * default_frame_dt (0.1 s), which only matches real
        time at 10 FPS - event times, durations and speeds were all scaled
        by that assumption.
        """
        fps = float(self.fps or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            return ctx
        for actor in ("A", "B"):
            tl = (ctx.get("actors", {}).get(actor, {}) or {}).get("timeline", {})
            if not tl:
                continue
            try:
                k0 = min(int(k) for k in tl)
            except ValueError:
                continue
            for k, frame in tl.items():
                if isinstance(frame, dict):
                    frame["t"] = (int(k) - k0) / fps
        return ctx


    def _build_pair_scenario_context(self, actor_a_id, actor_b_id, ctx_a, ctx_b, pair,
                                     start_frame, end_frame):
        route_a = ctx_a.get("route_semantics") or self._build_route_semantics(ctx_a["timeline"])
        route_b = ctx_b.get("route_semantics") or self._build_route_semantics(ctx_b["timeline"])
        actor_topology, interaction_topology = self._build_topology_facts(ctx_a, ctx_b)
        shared_frames = [
            int(frame_id)
            for frame_id, sample in pair.get("timeline", {}).items()
            if sample.get("lane_relation", {}).get("shared_junction", False)
        ]
        interaction_topology["shared_junction_frames"] = shared_frames
        interaction_topology["shared_junction_count"] = len(shared_frames)

        # H2: expose the same-lane fraction for rule inputs + signature.
        ratio = pair.get("shared_lane_ratio")
        if isinstance(ratio, (int, float)) and math.isfinite(ratio):
            interaction_topology["shared_lane_ratio"] = float(ratio)

        ctx = {
            "kind": "map_scenario_context",
            "version": "1.0",
            "context_version": "map_scenario_context_v1",
            "ready": False,
            "mode": self.active_mode,   # M8: MainApp routes on ctx["mode"]
            "scope": {"type": "pair_window", "start_frame": int(start_frame),
                      "end_frame": int(end_frame)},
            "actors": {
                "A": {"id": str(actor_a_id), "timeline": ctx_a["timeline"],
                      "route_semantics": route_a, "topology_facts": actor_topology["A"]},
                "B": {"id": str(actor_b_id), "timeline": ctx_b["timeline"],
                      "route_semantics": route_b, "topology_facts": actor_topology["B"]},
            },
            "interaction_facts": {
                "timeline": pair.get("timeline", {}),
                "summary": {
                    "distance_min": pair.get("distance_min"),
                    "distance_max": pair.get("distance_max"),
                    "ttc_min": pair.get("ttc_min"),
                },
            },
            "interaction_topology": interaction_topology,
            "duration_seconds": float(self._get_valid_duration() or 1.0),
            "meta": getattr(self.road, "meta", {}) or {},
        }
        self._stamp_frame_times(ctx)
        return ctx


    def _build_route_semantics(self, timeline):
        route, transitions, prev_lane = [], [], None
        for frame_id in sorted(timeline):
            sample = timeline.get(frame_id)
            if not sample:
                continue
            lane = sample.get("region_id")
            if lane is None:
                continue
            lane = str(lane)
            if lane == prev_lane:
                continue
            if prev_lane is not None:
                meta_prev = self.road.meta.get(prev_lane, {}) if self.road else {}
                succ = {str(x) for x in meta_prev.get("successors", [])}
                adj = {str(x) for x in meta_prev.get("adjacent", [])}
                if lane in succ:
                    transitions.append("DIVERGE" if len(succ) > 1 else "FOLLOW")
                elif lane in adj:
                    transitions.append("LANE_CHANGE")
                else:
                    transitions.append("TRANSFER")
            route.append(lane)
            prev_lane = lane
        return {"route": route, "transitions": transitions}

    def _build_topology_facts(self, ctx_a, ctx_b):
        meta = getattr(self.road, "meta", {}) or {}

        def actor_signature(ctx):
            route, prev = [], None
            for _, sample in sorted(ctx.get("timeline", {}).items()):
                lane = sample.get("region_id")
                if lane is None:
                    continue
                lane = str(lane)
                if lane == prev:
                    continue
                route.append(lane)
                prev = lane
            topology_sequence, merge_points, diverge_points = [], [], []
            for lane in route:
                m = meta.get(lane, {}) or {}
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
                "diverge_points": diverge_points,
            }

        a = actor_signature(ctx_a)
        b = actor_signature(ctx_b)
        b_merges, b_diverges = set(b["merge_points"]), set(b["diverge_points"])
        return (
            {"A": a, "B": b},
            {
                "topology_sequence_A": a["topology_sequence"],
                "topology_sequence_B": b["topology_sequence"],
                "shared_topology_tokens": len(set(a["topology_sequence"]) & set(b["topology_sequence"])),
                "topology_complexity": len(a["topology_sequence"]) + len(b["topology_sequence"]),
                "common_merge_points": [l for l in a["merge_points"] if l in b_merges],
                "common_diverge_points": [l for l in a["diverge_points"] if l in b_diverges],
            },
        )

    # ===================== context emit / receive =====================

    def emit_all_vehicle_map_contexts(self):
        if not self._get_active_traj_map():
            self.log_warn("No active trajectories.")
            return
        if not (self.road and self.view.meta and self.homography):
            self.log_warn("Missing required map components (mask/layout/homography).")
            return
        self.log_info(f"Building map context (mode={self.active_mode}) ...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)   # U10
        try:
            ctx = self._build_normalize_validate_emit()
        finally:
            QApplication.restoreOverrideCursor()
        self._engine_context = None
        self.map_context_ready.emit(ctx)
        self.log_info(
            f"Emitted {'VALID' if ctx.get('ready') else 'INVALID'} context"
            + (f" ({ctx.get('error')})" if ctx.get("error") else ""))


    def _emit_active_context(self):
        ctx = self._map_context_manual if self.is_manual_mode else self._map_context_json
        if not ctx or not ctx.get("ready"):
            self.log_info("Context emit skipped (not ready)")
            return
        self.map_context_ready.emit(ctx)
        self.log_info("Active context re-emitted")


    def receive_engine_context(self, ctx: dict):
        if not isinstance(ctx, dict) or ctx.get("kind") != "engine_context":
            self.log_warn("Ignored invalid engine context")
            return
        self._engine_context = dict(ctx)
        self.log_info("Engine context received")
        if getattr(self, "_save_debug_contexts", False):   # U7: opt-in only
            self._debug_save_context(ctx, tag="engine_enriched")


    def _invalidate_active_scenario(self, reason=""):
        """Active scenario changed; engine-derived products are now invalid."""
        self._engine_context = None
        if self.scenario_gui and hasattr(self.scenario_gui, "reset_vehicle_contexts"):
            try:
                self.scenario_gui.reset_vehicle_contexts()
            except Exception:
                pass
        if reason:
            print(f"[LaneMapViewer] Scenario invalidated: {reason}")

    def _debug_save_context(self, ctx, tag="auto"):
        try:
            os.makedirs("debug_contexts", exist_ok=True)
            path = f"debug_contexts/context_{tag}_{int(time.time())}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(to_json_safe(ctx), f, indent=2)
            print(f"[DEBUG] Context saved -> {path}")
        except Exception as e:
            print(f"[DEBUG] Save failed: {e}")

    # ===================== JSON mode: apply =====================

    def _ensure_start_le_end(self):
        try:
            s = int(self.global_start_box.currentText())
            e = int(self.global_end_box.currentText())
        except ValueError:
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
        except ValueError:
            return
        valid_ids = []
        for oid, traj in self.openlabel_traj_map.items():
            frames = {fr for fr, _, _ in traj}
            if fstart in frames and fend in frames:
                valid_ids.append(oid)
        valid_ids.sort(key=str)
        prev_v1, prev_v2 = self.v1_id_box.currentText(), self.v2_id_box.currentText()
        for box in (self.v1_id_box, self.v2_id_box):
            box.blockSignals(True)
            box.clear()
            box.addItems(valid_ids)
        if prev_v1 in valid_ids:
            self.v1_id_box.setCurrentText(prev_v1)
        if prev_v2 in valid_ids:
            self.v2_id_box.setCurrentText(prev_v2)
        if self.v1_id_box.currentText() == self.v2_id_box.currentText() and len(valid_ids) > 1:
            self.v2_id_box.setCurrentIndex(1)
        for box in (self.v1_id_box, self.v2_id_box):
            box.blockSignals(False)
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
        except ValueError:
            return self.log_warn("Invalid frame selection.")
        self._map_context_json = self._empty_context()
        self._engine_context = None
        self.clear_similarity_preview()          # U13
        if fstart > fend:
            fstart, fend = fend, fstart
        v1_id, v2_id = self.v1_id_box.currentText(), self.v2_id_box.currentText()
        if not v1_id or not v2_id or v1_id == v2_id:
            return self.log_warn("Select two different vehicles.")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)   # U10
        try:
            px1 = [(fr, x, y) for fr, x, y in self.openlabel_traj_map[v1_id] if fstart <= fr <= fend]
            px2 = [(fr, x, y) for fr, x, y in self.openlabel_traj_map[v2_id] if fstart <= fr <= fend]
            if len(px1) < 2 or len(px2) < 2:
                return self.log_warn("Too few trajectory points.")
            w1, w2 = self.px_to_world_polyline(px1), self.px_to_world_polyline(px2)
            if not w1 or not w2:
                return self.log_warn("World conversion failed (check homography + mask coverage).")

            try:
                traj1_dict = {fr: (x, y) for fr, x, y in self.openlabel_traj_map[v1_id]}
                traj2_dict = {fr: (x, y) for fr, x, y in self.openlabel_traj_map[v2_id]}
                traj_v1, traj_v2 = [], []
                for fr in range(fstart, fend + 1):
                    if fr in traj1_dict and fr in traj2_dict:
                        traj_v1.append((fr, *traj1_dict[fr]))
                        traj_v2.append((fr, *traj2_dict[fr]))

                store = self._get_active_store()
                store["traj_map"].update({"V1": traj_v1, "V2": traj_v2})
                store["world"].update({"V1": w1, "V2": w2})
                self._rebuild_vehicle_context_cache()

                for v, pts, color in (("V1", px1, QColor("green")), ("V2", px2, QColor("cyan"))):
                    pts_xy = [(x, y) for _, x, y in pts]
                    self.clear_traj_items(store["items"].get(v, []))
                    store["items"][v] = self.view.draw_trajectory(pts_xy, color)

                if self.scenario_gui and hasattr(self.scenario_gui, "reset_vehicle_contexts"):
                    self.scenario_gui.reset_vehicle_contexts()
                self._json_dirty = False
                self.compute_and_emit_duration()
                self._rebuild_hover_index()
                self.emit_all_vehicle_map_contexts()
                self.log_info(f"Applied trajectories: V1={v1_id}, V2={v2_id} "
                              f"({len(traj_v1)} common frames, {fstart}-{fend})")   # U11
            except Exception as e:
                self.log_error(f"Apply failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()
            self._refresh_action_states()         # U12


    # ===================== status / hover info =====================

    def log_info(self, text: str):
        self._append_log("INFO", text)

    def log_warn(self, text: str):
        self._append_log("WARN", text)

    def log_error(self, text: str):
        self._append_log("ERROR", text)

    def _append_log(self, level: str, text: str):
        if not hasattr(self, "log_view"):
            return
        sb = self.log_view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 4       # U8: don't yank the scroll position
        color = {"WARN": "#b9770e", "ERROR": "#c0392b"}.get(level, "#2c3e50")
        self.log_view.append(
            f'<span style="color:#7f8c8d">{time.strftime("%H:%M:%S")}</span> '
            f'<span style="color:{color};font-weight:bold">[{level}]</span> {text}'
        )
        if at_bottom:
            sb.setValue(sb.maximum())

    def _refresh_action_states(self):
        """State-based button enabling - preconditions visible without clicking.

        JSON mode: Apply is NOT gated on _traj_map_json - that store is only
        filled BY Apply itself (chicken-and-egg). Its real preconditions are
        OpenLABEL loaded + map components + two distinct vehicles.
        """
        is_manual = self.is_manual_mode
        has_map = bool(self.road and self.view.meta is not None and self.homography)
        has_traj = bool(self._get_active_traj_map())
        self.ctrl_generate.setVisible(is_manual)
        self.ctrl_generate.setEnabled(is_manual and has_map)
        if is_manual:
            self.ctrl_apply.setEnabled(has_traj)
        else:
            v1 = self.v1_id_box.currentText()
            v2 = self.v2_id_box.currentText()
            self.ctrl_apply.setEnabled(
                bool(self.openlabel_traj_map) and has_map
                and bool(v1) and bool(v2) and v1 != v2)
        self.ctrl_clear.setVisible(True)
        self.ctrl_duration.setEnabled(is_manual)
        self.ctrl_build_timeline.setEnabled(
            (not is_manual) and bool(self.openlabel_traj_map) and has_map)
        self.ctrl_find_similar.setEnabled(
            self.semantic_db is not None or bool(self._precomp.get("pair_windows", [])))
        self.json_status.setVisible(not is_manual)



    def update_status_panel(self):
        if self.is_manual_mode:
            ready = self._manual_traj_done["V1"] and self._manual_traj_done["V2"]
            state = "REGENERATE REQUIRED" if self._manual_dirty else ("READY" if ready else "INCOMPLETE")
            txt = (f"<b>MANUAL</b> | State: <b>{state}</b><br>"
                   f"Duration: {self.wp_duration.value():.1f}s | FPS: {self.wp_fps.value()}<br>"
                   f"V1 Frames: {len(self._traj_map_manual.get('V1', []))} | "
                   f"V2 Frames: {len(self._traj_map_manual.get('V2', []))}")
        else:
            state = "APPLY REQUIRED" if self._json_dirty else ("READY" if self._traj_map_json else "NO SCENARIO")
            txt = (f"<b>JSON</b> | State: <b>{state}</b><br>"
                   f"Frames: {self.global_start_box.currentText()} -> {self.global_end_box.currentText()}<br>"
                   f"V1={self.v1_id_box.currentText()} | V2={self.v2_id_box.currentText()}")
        self.status_label.setText(txt)

    def update_hover_coords(self, px: int, py: int):
        self._hover_px = (px, py)
        if self.road:
            wp = self.road.pixel_to_world(px, py)
            self._hover_wp = WorldPoint(*wp) if wp else None
        else:
            self._hover_wp = None
        self.update_status_panel()

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
                semantic = self._get_lane_semantic_debug(rid)
                sem_txt = ",".join(semantic.get("roles", []))
                sem_txt = f" | SEM:{sem_txt}" if sem_txt else ""
                adj, suc, pre = md.get("adjacent", []), md.get("successors", []), md.get("predecessors", [])
                line1 = (f"R{rid} | {t}{sem_txt} | "
                         f"Adjacent:{adj if adj else '-'} | "
                         f"Successors:{suc if suc else '-'} | "
                         f"Predecessors:{pre if pre else '-'}")
            elif t == "overlap":
                members = md.get("members", [])
                line1 = f"R{rid} | overlap | Members:{members if members else '-'}"
            else:
                line1 = f"R{rid} | {t}"

        if self._hover_px:
            x, y = self._hover_px
            px_txt = f"px:({x},{y})"
        else:
            px_txt = "px: -"
        world_txt = (f"w:({self._hover_wp.x:.2f},{self._hover_wp.y:.2f})"
                     if self._hover_wp else "w: -")
        sd_txt = "s:- d:-"
        if self._hover_wp and self.road and rid not in (None, 0):
            try:
                sd = self.road.project_world_to_frenet(rid, self._hover_wp.x, self._hover_wp.y)
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
                parts.append(f"dist:{pair_ins['dist']:.2f}" if isinstance(pair_ins.get("dist"), (int, float)) else "dist:-")
                parts.append(f"Δd:{pair_ins['dd']:.2f}" if isinstance(pair_ins.get("dd"), (int, float)) else "Δd:-")
            parts.append(f"px_dist:{dist_px:.1f}")
            line3 = " | ".join(parts)

        text = f"{line1}\n{line2}"
        if line3:
            text += f"\n{line3}"
        self.info_label.setText("<pre>" + text + "</pre>")

    # ===================== similarity preview =====================

    def clear_similarity_preview(self):
        if self.view is None:
            return
        for item in self._similarity_items:
            try:
                self.view.scene.removeItem(item)
            except Exception:
                pass
        self._similarity_items.clear()
        self._similarity_candidate = None

    def _draw_preview_polyline(self, points, color, width=3):
        if len(points) < 2:
            return []
        item = QGraphicsPathItem(polyline_path(points))
        pen = QPen(QColor(color))
        pen.setWidth(width)
        item.setPen(pen)
        self.view.scene.addItem(item)
        return [item]

    def _on_similarity_selected(self):
        items = self.similar_list.selectedItems()
        if not items:
            return
        idx = items[0].data(Qt.ItemDataRole.UserRole)
        if idx is None or idx >= len(self._last_similarity_results):
            return
        self.preview_similarity_match(self._last_similarity_results[idx]["track"])

    def preview_similarity_match(self, track):
        self.clear_similarity_preview()
        actors = track.get("actor_ids", [])
        if len(actors) < 2:
            return self.log_warn("Candidate has no actor pair.")
        start_frame = track.get("start_frame", track.get("first_frame"))
        end_frame = track.get("end_frame", track.get("last_frame"))
        if start_frame is None or end_frame is None:
            return self.log_warn("Candidate contains no frame range.")
        traj_map = self.openlabel_traj_map
        if not traj_map:
            return self.log_warn("OpenLABEL trajectories not loaded.")
        v1, v2 = str(actors[0]), str(actors[1])
        if v1 not in traj_map or v2 not in traj_map:
            return self.log_warn(f"Vehicle {v1 if v1 not in traj_map else v2} not found.")
        try:
            traj1 = [(x, y) for f, x, y in traj_map[v1] if start_frame <= f <= end_frame]
            traj2 = [(x, y) for f, x, y in traj_map[v2] if start_frame <= f <= end_frame]
            self.log_info(f"Preview candidate {v1}-{v2} frames={start_frame}-{end_frame} "
                          f"pts=({len(traj1)},{len(traj2)})")
            if len(traj1) < 2:
                return self.log_warn(f"{v1} preview trajectory too short.")
            if len(traj2) < 2:
                return self.log_warn(f"{v2} preview trajectory too short.")
            self._similarity_items.extend(self._draw_preview_polyline(traj1, "#FFD600", 4))
            self._similarity_items.extend(self._draw_preview_polyline(traj2, "#FF00FF", 4))
            self._similarity_candidate = track
        except Exception as e:
            self.log_warn(f"Preview failed: {e}")

    # ===================== similarity search =====================

    def find_similar_scenarios(self):
        """
        Without a base scenario: suggest candidate (v1, v2, [start,end]) windows
        from the precomputed index. With a base scenario: score candidate tracks
        by event + topology similarity.
        """
        query_sequence = self._build_query_sequence()

        if not query_sequence:
            wins = self._precomp.get("pair_windows", [])
            if not wins:
                self.similarity_status.setText("No candidate windows in index")   # U6
                return self.log_info("No candidate windows in index. Build the DB first.")

            def wscore(w):
                s_lane = w.get("shared_lane_fraction", 0.0)
                inv_ttc = 0.0 if not (w.get("min_ttc", float("inf")) > 0) else 1.0 / w["min_ttc"]
                inv_dist = 0.0 if not (w.get("min_dist", 1e9) > 0) else 1.0 / w["min_dist"]
                return 0.5 * s_lane + 0.3 * inv_ttc + 0.2 * inv_dist

            top = sorted(wins, key=wscore, reverse=True)[:50]
            lines = []
            for w in top:
                mt = w["min_ttc"]
                ttc_s = "inf" if mt == float("inf") else f"{mt:.2f}"
                lines.append(f"V1={w['v1']}  V2={w['v2']}  frames=[{w['start']}, {w['end']}], "
                             f"min_dist={w['min_dist']:.1f}, min_ttc={ttc_s}, "
                             f"same_lane={w['shared_lane_fraction']:.2f}")
            msg = "Candidate (v1, v2, [start,end]) windows from index:\n\n" + "\n".join(lines)
            self.similarity_status.setText(f"{len(top)} candidate windows (no base scenario)")  # U6
            self.log_info("Suggested Candidate Windows" + msg)
            return

        # Guardrails
        if self.active_mode == "json" and (not self.openlabel_traj_map or not self.openlabel_frames):
            return self.log_warn("Load OpenLABEL JSON first.")
        if not self.scenario_gui:
            return self.log_warn("Scenario GUI not connected.")

        db = self.semantic_db
        if db is None and self._loaded_db_path and os.path.exists(self._loaded_db_path):
            try:
                self.load_semantic_database(self._loaded_db_path)
                db = self.semantic_db
            except Exception:
                db = None
        if db is None:
            self.similarity_status.setText("No semantic DB loaded")   # U6
            return self.log_warn("Build an Event Timeline first.")

        tracks = db.get("tracks", [])
        print("\n=== QUERY SEQUENCE ===")
        print("len =", len(query_sequence))
        for e in query_sequence:
            print(e["event"], e["participants"])

        candidate_ids = self._candidate_track_indices(query_sequence)
        print("candidate_ids =", len(candidate_ids))
        query_events = {e["event"] for e in query_sequence}

        results = []
        for idx in candidate_ids:
            track = tracks[idx]
            cand_sequence = track.get("event_sequence", [])
            if not cand_sequence:
                continue
            fp = set(track.get("event_fingerprint", []))
            if not fp:  # pair tracks: derive from raw names (matches query side)
                fp = {e["event"] for e in cand_sequence if isinstance(e, dict)}
            if len(query_events & fp) < 1:
                continue
            topology_score = self._topology_score(track)
            event_score = self._event_similarity(query_sequence, cand_sequence)
            score = 0.60 * event_score + 0.40 * topology_score
            if score < 0.15:
                continue
            results.append({"score": score, "scope": track.get("scope"), "track": track})

        unique = {}
        for r in results:
            track = r["track"]
            sig = (
                tuple(tuple(x) for x in track.get("route_signatures", [])),
                tuple(e["event_token"] for e in track.get("event_sequence", [])),
            )
            if sig not in unique or r["score"] > unique[sig]["score"]:
                unique[sig] = r
        results = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
        self._last_similarity_results = results[:50]
        self._populate_similarity_list()
        if self._last_similarity_results:
            self.similarity_status.setText(
                f"{len(self._last_similarity_results)} matches "
                f"(top score {self._last_similarity_results[0]['score']:.3f})")   # U6
            self.log_info(f"{len(self._last_similarity_results)} candidate matches found.")
        else:
            self.similarity_status.setText("No matches (score < 0.15)")           # U6
            self.log_info("No candidate matches found (all scores < 0.15).")


    def _topology_score(self, track):
        try:
            descriptor = ((self.scenario_gui._engine_context or {}).get("derived", {})
                          .get("scenario_descriptor", {}))
            q_sig = descriptor.get("search_index", {}).get("semantic_signature", {})
            c_sig = track.get("semantic_signature", {})
            q_routes = [list(x) for x in q_sig.get("route_signatures", [])]
            c_routes = [list(x) for x in c_sig.get("route_signatures", [])]
            route_scores = [
                max((self._sequence_similarity(qr, cr) for cr in c_routes), default=0.0)
                for qr in q_routes
            ]
            return sum(route_scores) / max(len(route_scores), 1)
        except Exception:
            return 0.0

    def _populate_similarity_list(self):
        self.similar_list.clear()
        for idx, r in enumerate(self._last_similarity_results, start=1):
            track = r["track"]
            it = track.get("interaction_topology", {})
            actor_ids = track.get("actor_ids", []) or []
            if track.get("scope") == "pair" and len(actor_ids) >= 2:
                actors_txt = f"A={actor_ids[0]} B={actor_ids[1]}"
            elif actor_ids:
                actors_txt = f"A={actor_ids[0]} (vehicle)"
            else:
                actors_txt = "A=- B=-"
            fr_start = track.get("start_frame", track.get("first_frame"))
            fr_end = track.get("end_frame", track.get("last_frame"))
            item_text = (
                f"{idx}. {r['score']:.3f} | {actors_txt} | "
                f"J={it.get('shared_junction_count', 0)} "
                f"M={len(it.get('common_merge_points', []))} "
                f"D={len(it.get('common_diverge_points', []))} | "
                f"fr {fr_start}-{fr_end} ({track.get('frame_count', '?')}f)\n"
                f"{self._candidate_signature(track)}\n"
                f"{self._format_track_event_line(track)}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, idx - 1)
            self.similar_list.addItem(item)

    def _candidate_signature(self, track):
        routes = track.get("route_signatures", [])
        topo = " | ".join("→".join(r) for r in routes)
        seq = " -> ".join(
            ev.get("event_token", ev.get("event", "?"))
            for ev in track.get("event_sequence", [])
        )
        return f"{topo}\n{seq}"

    def _format_track_event_line(self, track):
        """
        One-line event timeline:
        lane_change@0.5s(0.2s) 12>13 | +0.4s waiting@0.9s(2.3s) | ...
        @t event start (s) | (x s) duration (ratio x recovered span)
        +x s start-to-start gap | a>b from_region > to_region
        """
        seq = [e for e in (track.get("event_sequence", []) or []) if isinstance(e, dict)]
        if not seq:
            return "no events"
        # Recover observed span: span = (t_j - t_i) / (start_ratio_j - start_ratio_i)
        span = None
        for i in range(len(seq)):
            ri, ti = seq[i].get("start_ratio"), seq[i].get("_start")
            if not (isinstance(ri, (int, float)) and isinstance(ti, (int, float))):
                continue
            for j in range(i + 1, len(seq)):
                rj, tj = seq[j].get("start_ratio"), seq[j].get("_start")
                if not (isinstance(rj, (int, float)) and isinstance(tj, (int, float))):
                    continue
                if rj - ri > 1e-9:
                    span = (tj - ti) / (rj - ri)
                    break
            if span is not None:
                break
        parts, prev_t = [], None
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
            fr, to = e.get("from_region"), e.get("to_region")
            region_txt = f" {fr}>{to}" if fr is not None and to is not None else ""
            token = f"{name}@{t0:.1f}s{dur_txt}{region_txt}"
            if prev_t is not None:
                token = f"+{t0 - prev_t:.1f}s {token}"
            parts.append(token)
            prev_t = t0
        return " | ".join(parts)

    # ===================== semantic DB =====================

    def load_semantic_database(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self.semantic_db = json.load(f)
        tracks = self.semantic_db.get("tracks", [])
        event_index, transition_index = {}, {}
        for idx, track in enumerate(tracks):
            events = [e.get("event", "?") for e in track.get("event_sequence", [])
                      if isinstance(e, dict)]
            for ev in set(events):
                event_index.setdefault(ev, []).append(idx)
            for a, b in zip(events[:-1], events[1:]):
                transition_index.setdefault(f"{a}>{b}", []).append(idx)
        self.semantic_db["_event_index"] = event_index
        self.semantic_db["_transition_index"] = transition_index
        self._loaded_db_path = path
        self._db_path = path
        self._project["db"] = path
        self._sync_project_file()
        self.similarity_status.setText(f"DB: {len(tracks)} tracks")
        self.log_info(f"Loaded event database ({len(tracks)} tracks)")
        self._refresh_action_states()
                         # U12


    def load_semantic_database_dialog(self):
        path = self._load_with_dialog("Load Event Sequence DB", "JSON (*.json)")
        if path:
            try:
                self.load_semantic_database(path)
            except Exception as e:   # U11: a corrupt file used to surface as
                self.log_error(f"Failed to load event database: {e}")     # "Build an Event Timeline first."


    def build_semantic_timeline_database(self):
        if not self.openlabel_traj_map:
            return self.log_warn("Load OpenLABEL first.")
        if not self.road:
            return self.log_warn("Load layout first.")
        if not self.homography:
            return self.log_warn("Load homography first.")

        # default to the DB the project already uses, so re-building overwrites it
        default_name = (os.path.basename(self._db_path)
                        if self._db_path and os.path.exists(self._db_path)
                        else "event_sequence_db.json")
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Event Sequence Database", default_name, "JSON (*.json)")
        if not save_path:
            return
        active_dialog = None

        def _abort(reason):
            """U10: close progress, remove the partial file, say why."""
            nonlocal active_dialog
            if active_dialog is not None:
                try:
                    active_dialog.close()
                except Exception:
                    pass
                active_dialog = None
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
            except OSError:
                pass
            self.log_warn(f"Database build aborted ({reason}) - partial file removed.")

        try:
            ids = sorted(self.openlabel_traj_map, key=str)

            # --- vehicle cache ---
            vehicle_cache = {}
            for oid in ids:
                traj = self.openlabel_traj_map.get(oid)
                if not traj:
                    continue
                frame_map = {fr: (x, y) for fr, x, y in traj}
                frames = sorted(frame_map)
                vehicle_cache[oid] = {
                    "traj": traj,
                    "frame_map": frame_map,
                    "first_frame": frames[0],
                    "last_frame": frames[-1],
                    "frame_count": len(frames),
                }

            # --- vehicle contexts + semantics ---
            vehicle_semantic_cache = {}
            vehicle_ctx_cache = self._openlabel_vehicle_ctx_cache
            progress_ctx = QProgressDialog("Precomputing vehicle semantics...", "Cancel",
                                           0, max(len(ids), 1), self)
            active_dialog = progress_ctx
            for i, oid in enumerate(ids):
                if i % 20 == 0:
                    progress_ctx.setValue(i)
                    QApplication.processEvents()
                if progress_ctx.wasCanceled():
                    _abort("cancelled")
                    return
                try:
                    if oid not in vehicle_ctx_cache:
                        vehicle_ctx_cache[oid] = self._ctx_builder.build_vehicle_context(
                            self._traj_to_builder_format(vehicle_cache[oid]["traj"]))
                    veh_ctx = vehicle_ctx_cache[oid]
                    semantic_ctx = self._build_pair_scenario_context(
                        actor_a_id=oid, actor_b_id=oid,
                        ctx_a=veh_ctx, ctx_b=veh_ctx, pair={"timeline": {}},
                        start_frame=vehicle_cache[oid]["first_frame"],
                        end_frame=vehicle_cache[oid]["last_frame"],
                    )
                    semantic_ctx["ready"] = True
                    semantic_ctx = self.build_semantic_context(semantic_ctx)
                    vehicle_semantic_cache[oid] = {
                        "context": semantic_ctx,
                        "descriptor": semantic_ctx.get("derived", {}).get("scenario_descriptor", {}),
                        "events": semantic_ctx.get("event_timeline", []),
                    }
                except Exception as e:
                    print(f"[CTX CACHE] {oid}: {e}")
            progress_ctx.close()
            active_dialog = None

            total_pairs = len(ids) * (len(ids) - 1) // 2
            progress = QProgressDialog("Building Event Database...", "Cancel",
                                       0, max(total_pairs, 1), self)
            active_dialog = progress
            tracks_written, first_track, counter = 0, True, 0

            def write_track(track):
                nonlocal first_track, tracks_written
                if not first_track:
                    out.write(",\n")
                json.dump(track, out, ensure_ascii=False)
                first_track = False
                tracks_written += 1
                if tracks_written % 500 == 0:
                    out.flush()

            with open(save_path, "w", encoding="utf-8") as out:
                out.write('{"version":6,"type":"event_sequence_database","tracks":[\n')

                # --- vehicle tracks ---
                for oid, veh in vehicle_cache.items():
                    sem = vehicle_semantic_cache.get(oid)
                    if sem is None:
                        continue
                    try:
                        ctx_single = sem["context"]
                        seen, events = set(), []
                        for ev in ctx_single.get("event_timeline", []):
                            participants = ev.get("participants", [])
                            if len(participants) > 1:  # drop synthetic pair events
                                continue
                            key = (ev.get("event"),
                                   round(float(ev.get("start_t", 0.0)), 6),
                                   round(float(ev.get("end_t", 0.0)), 6),
                                   ev.get("region"), ev.get("from_region"), ev.get("to_region"))
                            if key in seen:  # dedup mirrored self-events
                                continue
                            seen.add(key)
                            events.append(ev)
                        event_sequence, event_transitions, stats = build_searchable_event_sequence(events)
                        descriptor = sem["descriptor"]
                        write_track({
                            "scope": "vehicle",
                            "actor_ids": [str(oid)],
                            "first_frame": veh["first_frame"],
                            "last_frame": veh["last_frame"],
                            "frame_count": veh["frame_count"],
                            "event_sequence": event_sequence,
                            "event_transitions": event_transitions,
                            "event_stats": stats,
                            "event_fingerprint": sorted({e["event"] for e in event_sequence}),
                            "rule_types": sorted(set(descriptor.get("search_index", {})
                                                     .get("rule_types", []))),
                            "descriptor": descriptor,
                            "route_signatures": [tuple(ctx_single["actors"]["A"]
                                                       ["route_semantics"]["transitions"])],
                            "topology_facts": ctx_single["actors"]["A"]["topology_facts"],
                        })
                    except Exception as e:
                        print(f"[VEHICLE INDEX] {oid}: {e}")

                # --- pair tracks ---
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
                            _abort("cancelled")
                            return
                        v2 = vehicle_cache.get(v2_id)
                        if not v2:
                            continue
                        start_frame = max(v1["first_frame"], v2["first_frame"])
                        end_frame = min(v1["last_frame"], v2["last_frame"])
                        if end_frame - start_frame < 50:  # need >= 50 common frames
                            continue
                        if self._pair_spatial_rejection(v1, v2, start_frame, end_frame):
                            continue
                        try:
                            ctx_a, ctx_b = vehicle_ctx_cache[v1_id], vehicle_ctx_cache[v2_id]
                            if (not vehicle_semantic_cache[v1_id]["events"]
                                    and not vehicle_semantic_cache[v2_id]["events"]):
                                continue
                            tl1 = {k: v for k, v in ctx_a["timeline"].items()
                                   if start_frame <= k <= end_frame}
                            tl2 = {k: v for k, v in ctx_b["timeline"].items()
                                   if start_frame <= k <= end_frame}
                            common = set(tl1) & set(tl2)
                            if len(common) < 50:
                                continue
                            tl1 = {k: tl1[k] for k in common}
                            tl2 = {k: tl2[k] for k in common}
                            pair = self._ctx_builder.build_pairwise(
                                {"timeline": tl1}, {"timeline": tl2})
                            ctx = self._build_pair_scenario_context(
                                actor_a_id=v1_id, actor_b_id=v2_id,
                                ctx_a={"timeline": tl1}, ctx_b={"timeline": tl2},
                                pair=pair, start_frame=start_frame, end_frame=end_frame)
                            ctx["ready"] = True
                            ctx = self.build_semantic_context(ctx)
                            events = ctx.get("event_timeline") or []
                            event_sequence, transitions, stats = build_searchable_event_sequence(events)
                            interaction_topology = ctx.get("interaction_topology", {})
                            if (not event_sequence
                                    and not interaction_topology.get("common_merge_points")
                                    and not interaction_topology.get("common_diverge_points")):
                                continue
                            actors = ctx.get("actors", {})
                            actor_a_topology = actors.get("A", {}).get("route_semantics", {}).get("transitions", [])
                            actor_b_topology = actors.get("B", {}).get("route_semantics", {}).get("transitions", [])
                            write_track({
                                "scope": "pair",
                                "actor_ids": [str(v1_id), str(v2_id)],
                                "start_frame": start_frame,
                                "end_frame": end_frame,
                                "frame_count": len(common),
                                "interaction_topology": interaction_topology,
                                "route_signatures": sorted([tuple(actor_a_topology),
                                                            tuple(actor_b_topology)]),
                                "topology_transitions": {"A": actor_a_topology,
                                                         "B": actor_b_topology},
                                "event_sequence": event_sequence,
                                "event_transitions": transitions,
                                "event_stats": stats,
                                "event_fingerprint": sorted({e["event"] for e in event_sequence}),
                                "semantic_signature": (ctx.get("derived", {})
                                                       .get("scenario_descriptor", {})
                                                       .get("search_index", {})
                                                       .get("semantic_signature", {})),
                            })
                        except Exception as e:
                            print(f"[INDEX] {v1_id}-{v2_id}: {e}")

                out.write("\n]}")
            progress.close()
            active_dialog = None
            self.load_semantic_database(save_path)   # records _db_path + syncs the .trace
            if not self._project_path:
                self.log_info("Tip: save the project (.trace) to remember this DB for next time.")
            self.log_info(f"Database saved and loaded.\nTracks: {tracks_written}\nFile: {save_path}")
        except Exception as e:
            self.log_error(f"Database generation failed: {e}")
            _abort("error")


    def _pair_spatial_rejection(self, v1, v2, start_frame, end_frame):
        """Cheap prefilter: reject pairs far apart at 3 sampled frames."""
        min_dist = float("inf")
        for fr in (start_frame, (start_frame + end_frame) // 2, end_frame):
            p1 = v1["frame_map"].get(fr)
            p2 = v2["frame_map"].get(fr)
            if not p1 or not p2:
                continue
            min_dist = min(min_dist, math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return min_dist > 150

    # ===================== similarity math =====================

    def _sequence_similarity(self, seq_a, seq_b):
        """LCS-based similarity in [0, 1]."""
        if not seq_a or not seq_b:
            return 0.0
        m, n = len(seq_a), len(seq_b)
        prev = [0] * (n + 1)
        for i in range(m):
            curr = [0] * (n + 1)
            ai = seq_a[i]
            for j in range(n):
                curr[j + 1] = prev[j] + 1 if ai == seq_b[j] else max(curr[j], prev[j + 1])
            prev = curr
        return 2.0 * prev[n] / (m + n)

    def _event_similarity(self, query, candidate):
        if not query or not candidate:
            return 0.0
        q = [e["event"] for e in query]
        c = [e["event"] for e in candidate]
        lcs_score = self._sequence_similarity(q, c)
        q_set, c_set = set(q), set(c)
        jaccard = len(q_set & c_set) / max(len(q_set | c_set), 1)
        subsequence_bonus = 1.0 if lcs_score >= len(q) / max(len(c), len(q)) else 0.0
        return 0.50 * lcs_score + 0.20 * jaccard + 0.30 * subsequence_bonus

    def _build_query_sequence(self):
        ctx = getattr(self.scenario_gui, "_engine_context", None) if self.scenario_gui else None
        if not ctx:
            print("NO ENGINE CONTEXT")
            return []
        events = ctx.get("event_timeline") or []
        if not events:
            print("NO EVENTS")
            return []
        seq, _, _ = build_searchable_event_sequence(events)
        print("query events:", [e["event"] for e in seq])
        return seq

    def _candidate_track_indices(self, query_sequence):
        db = self.semantic_db
        if not db:
            return []
        # Index convention (see load_semantic_database):
        #   _event_index      -> keyed by RAW event name ("waiting")
        #   _transition_index -> keyed by token pairs ("a:waiting>b:proximity")
        event_index = db.get("_event_index", {})
        transition_index = db.get("_transition_index", {})
        query_events = {e["event"] for e in query_sequence}
        query_transitions = {
            f"{a['event_token']}>{b['event_token']}"
            for a, b in zip(query_sequence[:-1], query_sequence[1:])
        }
        scores = {}
        for ev in query_events:
            matches = event_index.get(ev, [])
            print(ev, len(matches))
            weight = 100.0 / max(1.0, len(matches))
            for idx in matches:
                scores[idx] = scores.get(idx, 0.0) + weight
        for tr in query_transitions:
            matches = transition_index.get(tr, [])
            weight = 200.0 / max(1.0, len(matches))
            for idx in matches:
                scores[idx] = scores.get(idx, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in ranked if score > 0][:200]