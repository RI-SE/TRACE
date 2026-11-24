# scenario_rule_gui.py
import copy
import json
import math
import traceback

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QCheckBox,
)

from explanation_generator import RuleExplanationGenerator
from lstm_single_bridge import LSTMSingleBridge, load_single_model, load_paths_json
from OpenScenarioExporter import OpenScenarioExporter
from scenario_engine import (
    SafeJSONEncoder,
    ScenarioRules,
    build_engine_insight_context,
    suggest_rules_combined,
)
from signal_engine import validate_context


class ScenarioRuleGUI(QWidget):
    engine_context_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scenario Rule Editor (v2)")
        self.mask = None
        self.meta = None
        self.xodr_path = None
        self.v1_pixel_polyline = None
        self.v2_pixel_polyline = None
        self.v1_world_polyline = None
        self.v2_world_polyline = None
        self._rules = []
        self._map_context = None
        self._engine_context = None
        self._last_context = None
        self._explainer = RuleExplanationGenerator()
        self.duration_seconds: float | None = None
        self.paths = []
        self.single_model = None
        self.bridge = None
        self.norms = None
        self.device = "cpu"
        self._build_ui()

    # ---------------- UI ----------------

    def _build_ui(self):
        lay = QVBoxLayout(self)

        lay.addWidget(QLabel("Scenario Name"))
        self.edit_name = QLineEdit()
        lay.addWidget(self.edit_name)

        lay.addWidget(QLabel("Duration (seconds)"))
        self.lbl_duration = QLabel("—")
        lay.addWidget(self.lbl_duration)

        lay.addWidget(QLabel("Notes"))
        self.edit_notes = QTextEdit()
        lay.addWidget(self.edit_notes)

        gen_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Rule Suggestions")
        self.btn_generate.clicked.connect(self.generate_rules)
        gen_row.addWidget(self.btn_generate)
        self.btn_explain = QPushButton("Explain Rules")
        self.btn_explain.setEnabled(False)
        self.btn_explain.clicked.connect(self.explain_rules)
        gen_row.addWidget(self.btn_explain)
        lay.addLayout(gen_row)

        io_row = QHBoxLayout()
        self.btn_load_paths = QPushButton("Load Paths JSON")
        self.btn_load_paths.clicked.connect(self.on_load_paths)
        io_row.addWidget(self.btn_load_paths)
        self.btn_load_model = QPushButton("Load SINGLE Model (.pth)")
        self.btn_load_model.clicked.connect(self.on_load_model)
        io_row.addWidget(self.btn_load_model)
        lay.addLayout(io_row)

        self.rule_table = QTableWidget(0, 5)
        self.rule_table.setHorizontalHeaderLabels(["WHO", "WHAT", "WHEN", "WHERE", "DETAILS"])
        self.rule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.rule_table)

        scope_row = QHBoxLayout()
        self.chk_explain_only_fired = QCheckBox("Explain only fired rules")
        scope_row.addWidget(self.chk_explain_only_fired)
        lay.addLayout(scope_row)

        self.insights_table = QTableWidget(0, 2)
        self.insights_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.insights_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(QLabel("Insights"))
        lay.addWidget(self.insights_table)

        self.btn_save = QPushButton("Save Scenario JSON")
        self.btn_save.clicked.connect(self.save_json)
        lay.addWidget(self.btn_save)
        self.btn_export_osc = QPushButton("Export OpenSCENARIO (.xosc)")
        self.btn_export_osc.clicked.connect(self.export_osc)
        lay.addWidget(self.btn_export_osc)
        self.btn_export_insights = QPushButton("Export Insights (this scenario)")
        self.btn_export_insights.clicked.connect(self.export_insights)
        lay.addWidget(self.btn_export_insights)

    # ---------------- Public setters (MainApp) ----------------

    def set_duration_seconds(self, duration_s: float):
        try:
            self.duration_seconds = float(duration_s)
            self.lbl_duration.setText(f"{self.duration_seconds:.2f}")
        except Exception:
            self.duration_seconds = None
            self.lbl_duration.setText("—")

    def set_mask_meta(self, mask, meta):
        self.mask = mask
        self.meta = meta

    def set_xodr_path(self, xodr_path: str):
        self.xodr_path = xodr_path

    def set_pixel_polyline(self, vehicle: str, pixel_json: str):
        poly = json.loads(pixel_json) if pixel_json else None
        if vehicle == "V1":
            self.v1_pixel_polyline = poly
        elif vehicle == "V2":
            self.v2_pixel_polyline = poly

    def set_world_polyline(self, vehicle: str, world_json: str):
        poly = None
        if world_json:
            try:
                data = json.loads(world_json)
                poly = data["world"] if isinstance(data, dict) and "world" in data else data
            except Exception:
                poly = None
        if vehicle == "V1":
            self.v1_world_polyline = poly
        elif vehicle == "V2":
            self.v2_world_polyline = poly

    def _actors_to_vehicle_view(self, ctx):
        actors = ctx.get("actors", {})
        return {"V1": actors.get("A", {}), "V2": actors.get("B", {})}

    # ---------------- Context ingestion ----------------

    def _convert_to_rule_engine_context(self, ctx):
        duration = self.duration_seconds or ctx.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise RuntimeError("Invalid duration")
        base_ctx = copy.deepcopy(ctx)
        base_ctx["ready"] = True
        base_ctx["duration_seconds"] = float(duration)
        enriched = build_engine_insight_context(context=base_ctx, duration=float(duration))
        enriched["kind"] = "engine_context"
        enriched["version"] = "1.0"
        return enriched

    def ingest_map_context(self, ctx: dict):
        if not ctx or ctx.get("kind") != "map_scenario_context" or ctx.get("ready") is not True:
            return
        validate_context(ctx)
        actors = ctx.get("actors", {})
        for actor in ("A", "B"):
            if not actors.get(actor, {}).get("timeline", {}):
                QMessageBox.warning(self, "Context Error", f"Actor {actor} timeline empty")
                return
        if "interaction_facts" not in ctx:
            QMessageBox.warning(self, "Context Error", "Missing interaction facts")
            return

        self._map_context = ctx
        ctx_duration = ctx.get("duration_seconds")
        if isinstance(ctx_duration, (int, float)):
            self.set_duration_seconds(ctx_duration)
        try:
            self._engine_context = self._convert_to_rule_engine_context(ctx)
        except Exception as e:
            QMessageBox.critical(self, "Engine Context Error", str(e))
            return
        self._last_context = self._engine_context
        self._populate_insights_table(self._engine_context)

    # ---------------- Model / paths loading ----------------

    def on_load_paths(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select paths JSON", "", "JSON (*.json)")
        if not path:
            return
        try:
            if load_paths_json is not None:
                self.paths = load_paths_json(path)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.paths = data.get("paths", data)
            QMessageBox.information(self, "Paths", f"Loaded {len(self.paths)} paths.")
        except Exception as e:
            QMessageBox.critical(self, "Paths", f"Failed to load: {e}")

    def on_load_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select SINGLE model .pth", "", "PyTorch (*.pth *.pt)")
        if not p:
            return
        try:
            if load_single_model is None:
                QMessageBox.warning(self, "Model",
                                    "Please include gui.model_loader.load_single_model in your project.")
                return
            model, norms = load_single_model(p, device=self.device)
            self.single_model = model
            self.norms = norms or {}
            if LSTMSingleBridge is not None:
                self.bridge = LSTMSingleBridge(
                    model=self.single_model,
                    device=self.device,
                    norms={
                        "lane_half_width_px": self.norms.get("lane_half_width_px", 40.0),
                        "GLOBAL_DDOT": self.norms.get("GLOBAL_DDOT", 0.2),
                        "GLOBAL_PSIDOT": self.norms.get("GLOBAL_PSIDOT", 0.1),
                    },
                    target_len=self.norms.get("target_len", 160),
                )
            else:
                QMessageBox.warning(self, "Bridge",
                                    "training.lstm_single_bridge not found. Bridge features disabled.")
            QMessageBox.information(self, "Model", "Loaded SINGLE model.")
        except Exception as e:
            QMessageBox.critical(self, "Model", f"Failed: {e}")

    # ---------------- Rule generation ----------------

    def generate_rules(self):
        if not self._map_context:
            QMessageBox.warning(self, "Missing Context", "No map context received from viewer.")
            return
        ctx = self._engine_context
        if not ctx:
            QMessageBox.warning(self, "Missing Context", "Engine context not initialized.")
            return
        if "actors" not in ctx or "derived" not in ctx:
            QMessageBox.critical(self, "Context Error", "Engine context corrupted.")
            return

        try:
            duration = (self.duration_seconds if self.duration_seconds is not None
                        else ctx.get("duration_seconds"))
            if duration is None:
                raise RuntimeError("Scenario duration unavailable.")

            raw_rules = suggest_rules_combined(context=ctx, duration=float(duration))
            seen, unique = set(), []
            for r in raw_rules:
                p = getattr(r, "parameters", {}) or {}
                key = (r.type, p.get("subtype"),
                       round(p.get("start_t", -1) or -1, 2),
                       round(p.get("end_t", -1) or -1, 2),
                       r.actor)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            self._rules = self._attach_event_timing(unique, ctx)
        except Exception as e:
            print("\n===== RULE GENERATION ERROR =====")
            traceback.print_exc()
            QMessageBox.critical(self, "Rule Generation Failed", f"{type(e).__name__}: {e}")
            return

        def rule_time_key(r):
            p = getattr(r, "parameters", {}) or {}
            t = p.get("start_t")
            if isinstance(t, (int, float)) and math.isfinite(t):
                return (0, float(t))
            frame = p.get("frame")
            if isinstance(frame, int) and frame >= 0:
                v1 = ctx.get("derived", {}).get("engine", {}).get("v1_sd", [])
                if 0 <= frame < len(v1) and isinstance(v1[frame].get("t"), (int, float)):
                    return (1, float(v1[frame]["t"]))
            return (2, float("inf"))

        self.rule_table.setRowCount(0)
        for r in sorted(self._rules, key=rule_time_key):
            for who, what, when, where, details in self._format_rule_summary(r, ctx):
                row = self.rule_table.rowCount()
                self.rule_table.insertRow(row)
                for col, text in enumerate((who, what, when, where, details)):
                    self.rule_table.setItem(row, col, QTableWidgetItem(str(text)))

        self._engine_context = ctx
        self.engine_context_ready.emit(ctx)
        self._populate_insights_table(ctx)
        self.setWindowTitle(f"TRACE Scenario — {self._derive_scenario_label(ctx)}")
        self.btn_explain.setEnabled(bool(self._engine_context))
        QMessageBox.information(self, "Done", "Rules generated successfully.")

    def _attach_event_timing(self, rules, ctx):
        """Snap each rule to a matching timeline event (type + actor); fill
        start_t/end_t/region (and lane_change from/to) from the best event."""
        semantic_map = {
            "dangerous_following": "proximity",
            "critical_proximity": "proximity",
            "near_miss": "proximity",
            "aggressive_approach": "proximity",
        }
        events = ctx.get("event_timeline", [])
        duration = ctx.get("duration_seconds", 0.0)
        engine = ctx.get("derived", {}).get("engine", {})
        v1_sd = engine.get("v1_sd", [])

        def _actor_matches(rule, event):
            r_actor = getattr(rule, "actor", "both")
            ev_parts = event.get("participants", [])
            if not ev_parts or r_actor == "both":
                return True
            return r_actor in ev_parts

        def _types_match(rule, event):
            rtype = getattr(rule, "type", None)
            subtype = (rule.parameters or {}).get("subtype")
            ev_type = event.get("type")
            return (subtype == ev_type or rtype == ev_type
                    or semantic_map.get(subtype) == ev_type)

        for r in rules:
            params = getattr(r, "parameters", {}) or {}
            subtype = params.get("subtype")
            best_event, best_score = None, float("inf")
            for e in events:
                if not _types_match(r, e) or not _actor_matches(r, e):
                    continue
                t = e.get("t_start")
                if not isinstance(t, (int, float)):
                    continue
                if subtype in semantic_map:
                    dist_val = e.get("metrics", {}).get("dist")
                    score = dist_val if isinstance(dist_val, (int, float)) else float("inf")
                elif subtype == "lane_change":
                    target_t = params.get("start_t") or (duration * 0.5 if duration else t)
                    score = abs(t - target_t)
                else:
                    score = abs(t - duration * 0.5)
                if score < best_score:
                    best_score, best_event = score, e

            if best_event is not None:
                params["start_t"] = best_event.get("t_start")
                params["end_t"] = best_event.get("t_end")
                region = best_event.get("region")
                if region is None and v1_sd and isinstance(params.get("start_t"), (int, float)):
                    idx = min(range(len(v1_sd)),
                              key=lambda i: abs(v1_sd[i]["t"] - params["start_t"]))
                    region = v1_sd[idx].get("region_id")
                params["region"] = region if region is not None else "unknown"
                if best_event.get("type") == "lane_change":
                    metrics = best_event.get("metrics", {})
                    params.setdefault("from_lane", metrics.get("from_lane"))
                    params.setdefault("to_lane", metrics.get("to_lane"))
                continue

            # No matching event: coarse fallbacks
            frame = engine.get("interaction_frame")
            if subtype in ("car_following", "avoidance", "overtake"):
                mid = duration * 0.5
                params["start_t"] = mid
                params["end_t"] = mid
            if isinstance(frame, int) and frame >= 0 and v1_sd:
                if 0 <= frame < len(v1_sd):
                    params.setdefault("region", v1_sd[frame].get("region_id"))
                params.setdefault("frame", frame)
            else:
                params.setdefault("frame", None)
            params.setdefault("start_t", None)
            params.setdefault("end_t", None)
        return rules

    def _derive_scenario_label(self, ctx):
        topo = ctx.get("interaction_topology", {})
        return (f"M{len(topo.get('common_merge_points', []))}"
                f"_D{len(topo.get('common_diverge_points', []))}"
                f"_J{topo.get('shared_junction_count', 0)}")

    def _format_rule_summary(self, rule, ctx):
        actor = getattr(rule, "actor", "both")
        params = self._resolve_actor_placeholders(getattr(rule, "parameters", {}) or {}, actor)

        # WHO
        who = "vehicle1 + vehicle2" if actor == "both" else actor

        # WHAT
        event_type = params.get("event_type")
        subtype = params.get("subtype")
        interaction = params.get("interaction") or {}
        behavior = params.get("behavior") or {}
        what = (subtype or event_type or interaction.get("type")
                or behavior.get("type") or getattr(rule, "type", "event"))

        # WHEN
        start_t = params.get("start_t")
        end_t = params.get("end_t")
        if isinstance(start_t, (int, float)):
            start_t = float(start_t)
            if isinstance(end_t, (int, float)) and abs(end_t - start_t) > 1e-3:
                when = f"{round(start_t, 2)}s → {round(end_t, 2)}s"
            else:
                when = f"{round(start_t, 2)}s"
        else:
            when = "-"

        # WHERE (fallback: nearest V1 sample at start_t)
        region = params.get("region")
        if region is None and isinstance(start_t, (int, float)):
            v1 = ctx.get("derived", {}).get("engine", {}).get("v1_sd", [])
            if v1:
                idx = min(range(len(v1)), key=lambda i: abs(v1[i]["t"] - start_t))
                region = v1[idx].get("region_id")
        where = str(region) if region is not None else "-"

        # DETAILS
        parts = []
        if subtype and subtype != what:
            parts.append(f"class={subtype}")
        timeline = params.get("timeline", {})
        if isinstance(timeline, dict):
            if timeline.get("axis"):
                parts.append(f"axis={timeline['axis']}")
            if timeline.get("pattern"):
                parts.append(f"pattern={timeline['pattern']}")
            intensity = timeline.get("intensity")
            if isinstance(intensity, (int, float)):
                parts.append(f"intensity={round(intensity, 3)}")
        for src in (interaction, behavior):
            if isinstance(src, dict):
                for k, v in src.items():
                    if k == "type":
                        continue
                    parts.append(f"{k}={round(v, 3) if isinstance(v, float) else v}")
        skip = {"frame", "subtype", "event_type", "start_t", "end_t",
                "region", "timeline", "interaction", "behavior"}
        for k, v in params.items():
            if k in skip:
                continue
            parts.append(f"{k}={round(v, 3) if isinstance(v, float) else v}")

        details = ", ".join(parts) if parts else "-"
        return [(who, what, when, where, details)]

    def _resolve_actor_placeholders(self, params, actor):
        if not isinstance(params, dict):
            return params
        out = {}
        for k, v in params.items():
            if isinstance(v, list):
                out[k] = [actor if x == "actor.id" else x for x in v]
            else:
                out[k] = actor if v == "actor.id" else v
        return out

    # ---------------- Insights table ----------------

    def _populate_insights_table(self, ctx: dict | None):
        self.insights_table.setRowCount(0)
        if not ctx:
            return

        def add(name, val):
            if val is None:
                return
            if isinstance(val, float):
                if not math.isfinite(val):
                    return
                val = round(val, 3)
            row = self.insights_table.rowCount()
            self.insights_table.insertRow(row)
            self.insights_table.setItem(row, 0, QTableWidgetItem(str(name)))
            self.insights_table.setItem(row, 1, QTableWidgetItem(str(val)))

        derived = ctx.get("derived", {}) or {}
        signals = derived.get("signals", {}) or {}

        # 1. Core signals + topology
        for k in ("min_distance", "max_distance", "avg_closing_rate"):
            add(k, signals.get(k))
        it = ctx.get("interaction_topology", {})
        add("shared_lane_fraction", (derived.get("rule_inputs") or {}).get("shared_lane_fraction"))
        add("shared_junction_count", it.get("shared_junction_count", 0))
        add("common_merge_count", len(it.get("common_merge_points", [])))
        add("common_diverge_count", len(it.get("common_diverge_points", [])))

        # 2. Probabilistic / rule inputs (top-level exposure)
        prob_keys = ("shared_junction", "shared_lane_fraction", "common_merge_count",
                     "common_diverge_count", "min_ttc", "distance_min",
                     "interaction_convergence", "interaction_lateral_cross",
                     "interaction_sign_flip", "waiting_duration_s")
        for k in prob_keys:
            if k in ctx:
                add(k, ctx[k])

        # 3. Interaction features + prefixed rule inputs
        for k, v in derived.get("interaction_features", {}).items():
            add(k, v)
        rule_inputs = derived.get("rule_inputs", {})
        for k in prob_keys:
            if k in rule_inputs:
                add(f"rule_{k}", rule_inputs[k])

        # 4. Timeline event stats
        events = ctx.get("event_timeline", [])
        add("event_count", len(events))
        type_counts = {}
        for e in events:
            t = e.get("type")
            if t:
                type_counts[t] = type_counts.get(t, 0) + 1
        for k, v in type_counts.items():
            add(f"events_{k}", v)

        # 5. Rule output summary
        if hasattr(self, "_rules"):
            subtype_counts = {}
            for r in self._rules:
                st = (getattr(r, "parameters", {}) or {}).get("subtype")
                if st:
                    subtype_counts[st] = subtype_counts.get(st, 0) + 1
            for k, v in subtype_counts.items():
                add(f"rules_{k}", v)

        # 6. Scenario descriptor
        for actor in ("A", "B"):
            route_sem = ctx.get("actors", {}).get(actor, {}).get("route_semantics", {})
            add(f"{actor}_route_regions", len(route_sem.get("route", [])))
            add(f"{actor}_route_transitions", len(route_sem.get("transitions", [])))
        descriptor = derived.get("scenario_descriptor") or {}
        if descriptor:
            add("descriptor_version", descriptor.get("descriptor_version"))
            add("scenario_signature", descriptor.get("signature"))
            graph = descriptor.get("graph") or {}
            add("descriptor_nodes", graph.get("node_count"))
            add("descriptor_edges", graph.get("edge_count"))
            nodes = graph.get("nodes", [])
            add("descriptor_event_nodes", sum(1 for n in nodes if n.get("node_type") == "event"))
            add("descriptor_rule_nodes", sum(1 for n in nodes if n.get("node_type") == "rule"))
            search_index = descriptor.get("search_index") or {}
            add("descriptor_event_types", ", ".join(search_index.get("event_types", [])))
            add("descriptor_rule_types", ", ".join(search_index.get("rule_types", [])))
            add("descriptor_motifs", ", ".join(search_index.get("motifs", [])))
            add("descriptor_transitions", ", ".join(search_index.get("transitions", [])[:10]))

        # 7. Safety summary
        if ctx.get("prob_near_miss"):
            add("risk_level", "HIGH" if ctx["prob_near_miss"] > 0.6 else "MEDIUM")

    # ---------------- Semantic artifacts (external API) ----------------

    def generate_semantic_artifacts(self, map_ctx):
        engine_ctx = self._convert_to_rule_engine_context(map_ctx)
        rules = suggest_rules_combined(
            duration=float(engine_ctx.get("duration_seconds", 0.0)),
            context=engine_ctx,
        )
        return {
            "engine_context": engine_ctx,
            "rules": rules,
            "descriptor": engine_ctx.get("derived", {}).get("scenario_descriptor", {}),
            "timeline": (engine_ctx.get("_canonical_event_timeline")
                         or engine_ctx.get("event_timeline", [])),
        }

    # ---------------- Explain / save / export ----------------

    def explain_rules(self):
        if not self._engine_context:
            QMessageBox.information(self, "Explain Rules",
                                    "No scenario context yet. Apply a scenario first.")
            return
        payload = {
            "derived_rules": self._rules,
            "context": self._engine_context or {},
            "options": {"only_fired": self.chk_explain_only_fired.isChecked()},
        }
        try:
            text = self._explainer.explain(payload)
            QMessageBox.information(self, "Scenario Explanation", text)
        except Exception as ex:
            QMessageBox.warning(self, "Explain Rules", f"Could not generate explanation: {ex}")

    def save_json(self):
        if not self._rules:
            QMessageBox.warning(self, "No Rules", "Generate rules first.")
            return
        name = self.edit_name.text().strip() or "scenario"
        if not self.duration_seconds:
            QMessageBox.warning(self, "Missing Duration", "Duration not provided by viewer.")
            return
        vehicles = self._actors_to_vehicle_view(self._engine_context)
        scen = ScenarioRules(
            scenario_name=name,
            duration_seconds=self.duration_seconds,
            vehicle1=vehicles["V1"],
            vehicle2=vehicles["V2"],
            derived_rules=self._rules,
            notes=self.edit_notes.toPlainText(),
        )
        path, _ = QFileDialog.getSaveFileName(self, "Save Scenario JSON", "scenario.json", "JSON (*.json)")
        if not path:
            return
        scen.save_json(path)
        QMessageBox.information(self, "Saved", f"Scenario saved to {path}")

    def _timeline_to_polyline(self, timeline):
        pts = []
        for k in sorted(timeline.keys()):
            wp = timeline[k].get("world")
            if isinstance(wp, dict):
                pts.append((float(wp.get("x", 0.0)), float(wp.get("y", 0.0))))
            elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                pts.append((float(wp[0]), float(wp[1])))
            elif pts:
                pts.append(pts[-1])
        return pts

    def avg_speed(self, poly):
        dist = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:]))
        return max(0.1, dist / max(self.duration_seconds or 1.0, 1e-3))

    def _resample_polyline(self, polyline, step=0.5):
        if not polyline or len(polyline) < 2:
            return polyline
        pts = np.array(polyline)
        dist = np.insert(np.cumsum(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))), 0, 0)
        new_pts = []
        for d in np.arange(0, dist[-1], step):
            i = max(0, min(np.searchsorted(dist, d) - 1, len(pts) - 2))
            den = dist[i + 1] - dist[i]
            t = 0.0 if abs(den) < 1e-9 else (d - dist[i]) / den
            p = pts[i] * (1 - t) + pts[i + 1] * t
            new_pts.append((float(p[0]), float(p[1])))
        return new_pts

    # ---------------- OpenSCENARIO export helpers ----------------

    @staticmethod
    def _read_centerline(meta: dict, rid: int):
        if meta is None or rid is None:
            return None
        e = meta.get(str(rid), {})
        if not isinstance(e, dict):
            return None
        for key in ("centerline", "center_points", "center", "polyline"):
            raw = e.get(key)
            if not isinstance(raw, list) or len(raw) < 2:
                continue
            pts = []
            for p in raw:
                try:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append((float(p[0]), float(p[1])))
                    elif isinstance(p, dict) and "x" in p and "y" in p:
                        pts.append((float(p["x"]), float(p["y"])))
                except (TypeError, ValueError):
                    pass
            if len(pts) >= 2:
                return pts
        return None

    @staticmethod
    def _project_s_on_polyline(pt, poly):
        """Arc-length of the projection of pt onto poly (0.0 if degenerate)."""
        if not poly or len(poly) < 2 or pt is None:
            return 0.0
        px, py = float(pt[0]), float(pt[1])
        cum = [0.0]
        for i in range(1, len(poly)):
            cum.append(cum[-1] + math.hypot(poly[i][0] - poly[i - 1][0],
                                            poly[i][1] - poly[i - 1][1]))
        best_i, best_t, best_d2 = 0, 0.0, float("inf")
        for i in range(len(poly) - 1):
            x1, y1 = poly[i]
            vx, vy = poly[i + 1][0] - x1, poly[i + 1][1] - y1
            wx, wy = px - x1, py - y1
            v2 = vx * vx + vy * vy
            t = 0.0 if v2 <= 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / v2))
            d2 = (px - (x1 + t * vx)) ** 2 + (py - (y1 + t * vy)) ** 2
            if d2 < best_d2:
                best_i, best_t, best_d2 = i, t, d2
        return cum[best_i] + best_t * (cum[best_i + 1] - cum[best_i])

    def _is_lane_or_connector(self, e: dict) -> bool:
        if not isinstance(e, dict):
            return False
        t = e.get("type")
        if t in ("lane", "connecting_lane"):
            return True
        if e.get("is_connecting_road") is True:
            return True
        return e.get("lane_type") in ("driving", "shoulder", "biking", "bus")

    def _lookup_road_lane(self, e: dict, rid=None):
        if not isinstance(e, dict):
            return None, None
        roadId = next((e[k] for k in ("roadId", "road_id", "road")
                       if k in e and e[k] not in (None, "", [])), None)
        laneId = next((e[k] for k in ("laneId", "lane_id", "lane")
                       if k in e and e[k] not in (None, "", [])), None)
        if isinstance(roadId, str) and roadId.strip().isdigit():
            roadId = int(roadId.strip())
        if roadId is None and rid is not None:
            print(f"[WARN] Cannot map region {rid} to road/lane")
        return roadId, laneId

    def _build_route_waypoints(self, vlabel: str, ctx: dict, polyline: list | None):
        path_regions = (ctx or {}).get(f"{vlabel}_path_regions", [])
        if not path_regions:
            veh = self._actors_to_vehicle_view(ctx)["V1" if vlabel == "v1" else "V2"]
            path_regions = [f.get("region_id") for f in veh.get("timeline", {}).values()
                            if f.get("region_id") is not None]
        meta = self.meta or {}
        waypoints, widx, s_fallback = [], 0, 0.5
        for rid in path_regions:
            e = meta.get(str(rid), {})
            if not self._is_lane_or_connector(e):
                continue
            roadId, laneId = self._lookup_road_lane(e, rid)
            if roadId is None:
                continue
            center = self._read_centerline(meta, rid)
            if not center:
                print(f"[WARN] No centerline for region {rid}")
            pt = polyline[widx] if polyline and 0 <= widx < len(polyline) else None
            if pt and center:
                try:
                    s_m = max(0.5, float(self._project_s_on_polyline(pt, center)))
                except (TypeError, ValueError):
                    s_m = s_fallback
            else:
                s_m = s_fallback
            waypoints.append({
                "roadId": roadId,
                "laneId": laneId,
                "s": s_m,
                "offset": 0.0,
                "s_trustworthy": False,
            })
            if polyline:
                widx += max(1, len(polyline) // max(1, len(path_regions)))
            else:
                widx += 1
            s_fallback += 1.0
        return waypoints

    def lane_seed_from_route(self, route):
        if not route:
            return None
        return {
            "roadId": route[0].get("roadId"),
            "laneId": route[0].get("laneId"),
            "s": max(0.5, float(route[0].get("s", 0.5))),
            "offset": 0.0,
        }

    def export_osc(self):
        if not self._engine_context:
            QMessageBox.warning(self, "Context Error", "Generate rules first.")
            return
        ctx = self._engine_context
        view = self._actors_to_vehicle_view(ctx)
        v1, v2 = view["V1"], view["V2"]
        if not v1 or not v2:
            QMessageBox.warning(self, "Export Error", "Vehicles missing.")
            return
        duration = float(self.duration_seconds or 5.0)
        v1_poly = v1.get("trajectory") or self._timeline_to_polyline(v1.get("timeline", {}))
        v2_poly = v2.get("trajectory") or self._timeline_to_polyline(v2.get("timeline", {}))
        if len(v1_poly) < 2 or len(v2_poly) < 2:
            QMessageBox.warning(self, "Export Error", "Trajectories too short.")
            return
        route_v1 = self._build_route_waypoints("v1", ctx, v1_poly)
        route_v2 = self._build_route_waypoints("v2", ctx, v2_poly)
        if not route_v1:
            QMessageBox.warning(self, "Export Error", "V1 route invalid.")
            return
        if not route_v2:
            QMessageBox.warning(self, "Export Error", "V2 route invalid.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save OpenSCENARIO", "scenario.xosc",
                                              "OpenSCENARIO (*.xosc)")
        if not path:
            return
        if not path.lower().endswith(".xosc"):
            path += ".xosc"

        scen = ScenarioRules(
            scenario_name=self.edit_name.text().strip() or "scenario",
            duration_seconds=duration,
            vehicle1=v1,
            vehicle2=v2,
            derived_rules=self._rules or [],
            notes=self.edit_notes.toPlainText(),
        )
        scen.event_timeline = (ctx.get("_canonical_event_timeline")
                               or ctx.get("event_timeline") or [])
        scen.events = ctx.get("events") or []
        scen.insights = ctx.get("derived", {}).get("signals", {})

        exporter = OpenScenarioExporter(
            scenario_rules=scen,
            world_polyline_v1=self._resample_polyline(v1_poly, step=1.0),
            world_polyline_v2=self._resample_polyline(v2_poly, step=1.0),
            xodr_path=self.xodr_path,
            spawn_position_type="lane",
            route_v1=route_v1,
            route_v2=route_v2,
            v1_init_speed=self.avg_speed(v1_poly),
            v2_init_speed=self.avg_speed(v2_poly),
            v1_lane_seed=self.lane_seed_from_route(route_v1),
            v2_lane_seed=self.lane_seed_from_route(route_v2),
            meta=self.meta,
        )
        try:
            exporter.save(path)
            QMessageBox.information(self, "Export Successful", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    def export_insights(self):
        if not self._engine_context:
            QMessageBox.information(self, "Export Insights",
                                    "Generate rules first so context exists.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Insights JSON",
                                              "scenario_insights.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                # SafeJSONEncoder: engine context contains inf/nan and numpy types.
                json.dump(self._engine_context, f, indent=2, ensure_ascii=False,
                          cls=SafeJSONEncoder)
            QMessageBox.information(self, "Export Insights", f"Saved insights to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Insights", f"Failed to write insights: {e}")
