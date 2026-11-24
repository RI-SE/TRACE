# OpenScenarioExporter.py
import os
import math
import xml.etree.ElementTree as ET


def _finite_float(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _to_int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


class OpenScenarioExporter:

    # Key variants the layout meta may use for the OpenDRIVE mapping
    # (must stay in sync with ScenarioRuleGUI._lookup_road_lane).
    ROAD_KEYS = ("roadId", "road_id", "road")
    LANE_KEYS = ("laneId", "lane_id", "lane")

    def __init__(self, scenario_rules, **kwargs):
        self.rules = scenario_rules
        self._xosc_dir = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.meta = kwargs.get("meta", {}) or {}
        self.world_polyline_v1 = kwargs.get("world_polyline_v1", []) or []
        self.world_polyline_v2 = kwargs.get("world_polyline_v2", []) or []

        self.v1_lane_changes = kwargs.get("v1_lane_changes", []) or []
        self.v2_lane_changes = kwargs.get("v2_lane_changes", []) or []

        self.derived_rules = getattr(scenario_rules, "derived_rules", []) or []
        self.event_timeline = getattr(scenario_rules, "event_timeline", []) or []
        self.detected_events = getattr(scenario_rules, "events", []) or []
        self.insights = getattr(scenario_rules, "insights", {}) or {}

        # Sanitized scalars (safe defaults instead of AttributeError/crashes)
        self.v1_init_speed = _finite_float(kwargs.get("v1_init_speed"), 10.0)
        self.v2_init_speed = _finite_float(kwargs.get("v2_init_speed"), 10.0)
        self._duration = _finite_float(
            getattr(scenario_rules, "duration_seconds", None), 10.0
        )

    # ============================================================
    # BASIC STRUCTURE
    # ============================================================
    def header(self):
        return ET.Element(
            "FileHeader",
            revMajor="1",
            revMinor="0",
            description=str(getattr(self.rules, "scenario_name", "") or ""),
            author="AutoExporter",
        )

    def road_network(self):
        rn = ET.Element("RoadNetwork")
        xodr = getattr(self, "xodr_path", None)
        if not xodr:
            print("[OpenScenarioExporter] WARNING: no xodr_path - "
                  "esmini will have no road network")
            return rn

        # esmini resolves LogicFile relative to the .xosc location:
        # relative path when possible, absolute otherwise.
        path = str(xodr)
        if self._xosc_dir and os.path.isabs(path):
            try:
                path = os.path.relpath(path, self._xosc_dir)
            except ValueError:
                pass  # e.g. different drive letter - keep absolute

        ET.SubElement(rn, "LogicFile", filepath=path)
        return rn

    def entities(self):
        ents = ET.Element("Entities")

        for name in ["vehicle1", "vehicle2"]:
            so = ET.SubElement(ents, "ScenarioObject", name=name)
            veh = ET.SubElement(so, "Vehicle", name=name, vehicleCategory="car")

            ET.SubElement(veh, "Performance",
                          maxSpeed="50",
                          maxAcceleration="8",
                          maxDeceleration="10")

            bb = ET.SubElement(veh, "BoundingBox")
            ET.SubElement(bb, "Center", x="1.5", y="0", z="0.9")
            ET.SubElement(bb, "Dimensions", width="2.0", length="4.5", height="1.5")

            # controller (required by esmini for driver actions)
            ctrl = ET.SubElement(so, "ObjectController")
            ET.SubElement(ctrl, "Controller", name="DefaultDriver")

        return ents

    # ============================================================
    # INIT (lane spawn, world-position fallback)
    # ============================================================
    def init_actions(self):
        init = ET.Element("Init")
        acts = ET.SubElement(init, "Actions")

        def setup(name, seed, speed):
            priv = ET.SubElement(acts, "Private", entityRef=name)

            pa = ET.SubElement(priv, "PrivateAction")
            tp = ET.SubElement(pa, "TeleportAction")
            pos = ET.SubElement(tp, "Position")

            spawned = False

            if (
                isinstance(seed, dict)
                and seed.get("roadId") is not None
                and _to_int(seed.get("laneId")) is not None
                and _to_int(seed.get("laneId")) != 0
            ):
                ET.SubElement(
                    pos,
                    "LanePosition",
                    roadId=str(seed["roadId"]),
                    laneId=str(_to_int(seed["laneId"])),
                    s=str(max(2.0, _finite_float(seed.get("s"), 2.0))),
                    offset="0",
                )
                spawned = True

            if not spawned:
                poly = (self.world_polyline_v1 if name == "vehicle1"
                        else self.world_polyline_v2)

                if poly and len(poly) > 1:
                    x, y = poly[0]
                    x2, y2 = poly[1]
                    heading = math.atan2(y2 - y, x2 - x)
                    ET.SubElement(
                        pos,
                        "WorldPosition",
                        x=str(x), y=str(y), z="0", h=str(heading),
                    )
                else:
                    print(f"[OpenScenarioExporter] WARNING: no spawn "
                          f"reference for {name} - teleporting to (0,0)")
                    ET.SubElement(pos, "WorldPosition", x="0", y="0", z="0")

            pa2 = ET.SubElement(priv, "PrivateAction")
            lo = ET.SubElement(pa2, "LongitudinalAction")
            sp = ET.SubElement(lo, "SpeedAction")

            ET.SubElement(
                sp,
                "SpeedActionDynamics",
                dynamicsShape="step",
                dynamicsDimension="time",
                value="0",
            )
            tgt = ET.SubElement(sp, "SpeedActionTarget")
            ET.SubElement(tgt, "AbsoluteTargetSpeed", value=str(speed))

        setup("vehicle1", getattr(self, "v1_lane_seed", None), self.v1_init_speed)
        setup("vehicle2", getattr(self, "v2_lane_seed", None), self.v2_init_speed)

        return init

    # ============================================================
    # TRIGGERS
    # ============================================================
    def _time_trigger(self, t):
        # Condition REQUIRES conditionEdge + delay in OpenSCENARIO 1.0
        trig = ET.Element("StartTrigger")
        cg = ET.SubElement(trig, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", conditionEdge="none", delay="0")
        bv = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(
            bv,
            "SimulationTimeCondition",
            value=str(max(0.1, _finite_float(t, 0.1))),
            rule="greaterThan",
        )
        return trig

    # ============================================================
    # EVENT BUILDERS
    # ============================================================
    def _lane_change_event(self, idx, ch, actor):
        lane_id = _to_int(ch.get("laneId"))

        if lane_id is None or lane_id == 0:
            return None

        ev = ET.Element(
            "Event",
            name=f"{actor}_lane_change_{idx}",
            priority="overwrite",
        )
        ev.append(self._time_trigger(ch.get("time")))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")
        lat = ET.SubElement(pa, "LateralAction")
        lc = ET.SubElement(lat, "LaneChangeAction")

        ET.SubElement(
            lc,
            "LaneChangeActionDynamics",
            dynamicsShape="linear",
            dynamicsDimension="time",
            value="2",
        )
        tgt = ET.SubElement(lc, "LaneChangeTarget")
        ET.SubElement(tgt, "AbsoluteTargetLane", value=str(lane_id))

        return ev

    def _speed_event(self, speed, t, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_speed_hold_base",
            priority="overwrite",
        )
        ev.append(self._time_trigger(t))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")
        lo = ET.SubElement(pa, "LongitudinalAction")
        sp = ET.SubElement(lo, "SpeedAction")

        # re-assert base speed over a short ramp (Init already set it)
        ET.SubElement(
            sp,
            "SpeedActionDynamics",
            dynamicsShape="linear",
            dynamicsDimension="time",
            value="5",
        )
        tgt = ET.SubElement(sp, "SpeedActionTarget")
        ET.SubElement(tgt, "AbsoluteTargetSpeed", value=str(speed))

        return ev

    def _stop_event(self, t, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_stop",
            priority="overwrite",
        )
        ev.append(self._time_trigger(t))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")
        lo = ET.SubElement(pa, "LongitudinalAction")
        sp = ET.SubElement(lo, "SpeedAction")

        ET.SubElement(
            sp,
            "SpeedActionDynamics",
            dynamicsShape="linear",
            dynamicsDimension="time",
            value="1.0",
        )
        tgt = ET.SubElement(sp, "SpeedActionTarget")
        ET.SubElement(tgt, "AbsoluteTargetSpeed", value="0")

        return ev

    def _lane_keep_event(self, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_lane_keep",
            priority="parallel",
            maximumExecutionCount="1",
        )
        ev.append(self._time_trigger(0.2))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")
        lat = ET.SubElement(pa, "LateralAction")
        lk = ET.SubElement(lat, "LaneKeepingAction")

        # engage immediately at the trigger (old value "10000" meant the
        # action would only start at t=10000s)
        ET.SubElement(
            lk,
            "LaneKeepingActionDynamics",
            dynamicsShape="step",
            dynamicsDimension="time",
            value="0",
        )

        return ev

    # ============================================================
    # TIMELINE -> LANE CHANGES
    # ============================================================
    def _meta_node(self, region_id):
        if not isinstance(self.meta, dict) or region_id is None:
            return None
        node = self.meta.get(str(region_id))
        if node is None:
            node = self.meta.get(region_id)
        return node if isinstance(node, dict) else None

    @classmethod
    def _first_key(cls, node, keys):
        for k in keys:
            v = node.get(k)
            if v not in (None, "", []):
                return v
        return None

    def _region_position(self, region_id):
        node = self._meta_node(region_id)
        if node is None:
            return None

        road_id = self._first_key(node, self.ROAD_KEYS)
        lane_id = _to_int(self._first_key(node, self.LANE_KEYS))

        if road_id is None or lane_id is None or lane_id == 0:
            return None

        return {"roadId": str(road_id), "laneId": lane_id}

    def _region_to_laneid(self, region_id):
        pos = self._region_position(region_id)
        return pos["laneId"] if pos else None

    def _timeline_time_scale(self):
        """Engine event times are synthetic (index * default_frame_dt);
        duration_seconds is real time - rescale the ratio."""
        timeline = self.event_timeline if isinstance(self.event_timeline, list) else []
        event_end = max(
            (
                _finite_float(evt.get("t_end", evt.get("t_start", 0.0)), 0.0) or 0.0
                for evt in timeline
                if isinstance(evt, dict)
            ),
            default=0.0,
        )
        if self._duration > 0.0 and event_end > self._duration:
            return self._duration / event_end
        return 1.0

    def _timeline_time(self, value, default=0.2):
        return max(0.1, _finite_float(value, default) * self._timeline_time_scale())

    def _timeline_lane_changes(self, actor):
        changes = []

        for evt in self.event_timeline:
            if not isinstance(evt, dict):
                continue
            if str(evt.get("type", "")).lower() != "lane_change":
                continue

            participants = evt.get("participants", [])
            if isinstance(participants, str):
                participants = [participants]
            if actor not in participants:
                continue

            metrics = evt.get("metrics", {}) or {}
            from_region = metrics.get("from_lane")
            to_region = metrics.get("to_lane")

            source = self._region_position(from_region)
            destination = self._region_position(to_region)

            if source is None or destination is None:
                continue

            # LaneChangeAction cannot switch roads
            if source["roadId"] != destination["roadId"]:
                continue
            if source["laneId"] == destination["laneId"]:
                continue

            changes.append({
                "laneId": destination["laneId"],
                "time": self._timeline_time(evt.get("t_start")),
                "duration": max(
                    0.5,
                    self._timeline_time(evt.get("t_end"))
                    - self._timeline_time(evt.get("t_start")),
                ),
                "fromRegion": from_region,
                "toRegion": to_region,
                "roadId": destination["roadId"],
            })

        changes.sort(key=lambda item: item["time"])
        return changes

    # ============================================================
    # BUILD EVENTS
    # ============================================================
    def build_events(self, actor):
        events = []

        timeline_changes = self._timeline_lane_changes(actor)

        # timeline-derived changes take precedence over static lists
        changes = [] if timeline_changes else getattr(
            self, f"{actor}_lane_changes", []
        )

        base_speed = (self.v1_init_speed if actor == "vehicle1"
                      else self.v2_init_speed)

        events.append(self._speed_event(base_speed, 0.2, actor))

        for i, ch in enumerate(timeline_changes + changes):
            ev = self._lane_change_event(i, ch, actor)
            if ev is not None:
                events.append(ev)

        # stop near scenario end
        events.append(self._stop_event(max(0.1, self._duration - 0.5), actor))

        events.sort(
            key=lambda e: float(
                e.find(".//SimulationTimeCondition").attrib.get("value", "999")
            )
        )

        return events

    def maneuver_group(self, actor):
        mg = ET.Element(
            "ManeuverGroup",
            name=f"{actor}_group",
            maximumExecutionCount="1",
        )
        actors = ET.SubElement(mg, "Actors", selectTriggeringEntities="false")
        ET.SubElement(actors, "EntityRef", entityRef=actor)

        man = ET.SubElement(mg, "Maneuver", name=f"{actor}_maneuver")

        if not getattr(self, f"{actor}_lane_changes", []):
            man.append(self._lane_keep_event(actor))

        for e in self.build_events(actor):
            man.append(e)

        return mg

    # ============================================================
    # PARAMETERS
    # ============================================================
    def parameter_declarations(self):
        pd = ET.Element("ParameterDeclarations")

        if not isinstance(self.insights, dict):
            return pd

        for k, v in self.insights.items():
            if isinstance(v, bool):
                ptype, value = "string", str(v)
            elif isinstance(v, str):
                ptype, value = "string", v
            elif isinstance(v, (int, float)):
                f = _finite_float(v)
                if f is None:
                    continue  # inf/nan are not valid OpenSCENARIO doubles
                ptype, value = "double", str(f)
            else:
                continue  # skip non-scalars (lists/dicts/None)

            name = "".join(
                c if (c.isalnum() or c == "_") else "_" for c in str(k)
            )
            if not name:
                continue

            ET.SubElement(
                pd,
                "ParameterDeclaration",
                name=name,
                parameterType=ptype,
                value=value,
            )

        return pd

    # ============================================================
    # STORYBOARD
    # ============================================================
    def storyboard(self):
        sb = ET.Element("Storyboard")
        sb.append(self.init_actions())

        story = ET.SubElement(sb, "Story", name="MainStory")
        act = ET.SubElement(story, "Act", name="MainAct")

        start = ET.SubElement(act, "StartTrigger")
        cg = ET.SubElement(start, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", conditionEdge="none", delay="0")
        bv = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(
            bv,
            "SimulationTimeCondition",
            value="0.1",
            rule="greaterThan",
        )

        act.append(self.maneuver_group("vehicle1"))
        act.append(self.maneuver_group("vehicle2"))

        # NOTE: conditionEdge is REQUIRED on Condition (was missing here,
        # making the file non-conformant)
        stop = ET.SubElement(sb, "StopTrigger")
        cg2 = ET.SubElement(stop, "ConditionGroup")
        cond2 = ET.SubElement(cg2, "Condition", conditionEdge="none", delay="0")
        bv2 = ET.SubElement(cond2, "ByValueCondition")
        ET.SubElement(
            bv2,
            "SimulationTimeCondition",
            value=str(self._duration),
            rule="greaterThan",
        )

        return sb

    # ============================================================
    # SAVE
    # ============================================================
    def save(self, path):
        self._xosc_dir = os.path.dirname(os.path.abspath(path))

        root = ET.Element("OpenSCENARIO")
        root.append(self.header())

        pd = self.parameter_declarations()
        if len(pd):
            root.append(pd)

        root.append(self.road_network())
        root.append(self.entities())
        root.append(self.storyboard())

        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
