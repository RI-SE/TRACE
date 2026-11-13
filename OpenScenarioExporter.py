import xml.etree.ElementTree as ET
import math


class OpenScenarioExporter:

    def __init__(self, scenario_rules, **kwargs):
        self.rules = scenario_rules

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.meta = kwargs.get("meta", {}) or {}

        self.world_polyline_v1 = kwargs.get("world_polyline_v1", [])
        self.world_polyline_v2 = kwargs.get("world_polyline_v2", [])

        self.v1_lane_changes = getattr(self, "v1_lane_changes", []) or []
        self.v2_lane_changes = getattr(self, "v2_lane_changes", []) or []
        self.derived_rules = getattr(
            scenario_rules,
            "derived_rules",
            []
        ) or []

        self.event_timeline = getattr(
            scenario_rules,
            "event_timeline",
            []
        ) or []

        self.detected_events = getattr(
            scenario_rules,
            "events",
            []
        ) or []

        self.insights = getattr(
            scenario_rules,
            "insights",
            {}
        ) or {}
    # ============================================================
    # BASIC STRUCTURE
    # ============================================================
    def header(self):
        return ET.Element(
            "FileHeader",
            revMajor="1",
            revMinor="0",
            description=self.rules.scenario_name,
            author="AutoExporter"
        )

    def road_network(self):
        rn = ET.Element("RoadNetwork")
        if getattr(self, "xodr_path", None):
            ET.SubElement(rn, "LogicFile", filepath=str(self.xodr_path).split("/")[-1])
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

            # ✅ controller (safe for esmini)
            ctrl = ET.SubElement(so, "ObjectController")
            ET.SubElement(ctrl, "Controller", name="DefaultDriver")

        return ents



    # ============================================================
    # INIT (FIXED SPAWN)
    # ============================================================
    def init_actions(self):
        init = ET.Element("Init")
        acts = ET.SubElement(init, "Actions")

        def setup(name, seed, speed):
            priv = ET.SubElement(acts, "Private", entityRef=name)

            pa = ET.SubElement(priv, "PrivateAction")
            tp = ET.SubElement(pa, "TeleportAction")
            pos = ET.SubElement(tp, "Position")

            # ✅ PRIMARY: lane spawn
            if seed and seed.get("roadId") is not None and seed.get("laneId") is not None:
                ET.SubElement(
                    pos,
                    "LanePosition",
                    roadId=str(seed["roadId"]),
                    laneId=str(seed["laneId"]),
                    s=str(max(2.0, seed.get("s", 2.0))),   # ✅ safer s
                    offset="0"
                )

            # ✅ FALLBACK: world position
            else:
                poly = self.world_polyline_v1 if name == "vehicle1" else self.world_polyline_v2

                if poly and len(poly) > 1:
                    x, y = poly[0]
                    x2, y2 = poly[1]

                    heading = math.atan2(y2 - y, x2 - x)

                    ET.SubElement(
                        pos,
                        "WorldPosition",
                        x=str(x),
                        y=str(y),
                        z="0",
                        h=str(heading)
                    )
                else:
                    ET.SubElement(pos, "WorldPosition", x="0", y="0", z="0")

            # ✅ initial speed
            pa2 = ET.SubElement(priv, "PrivateAction")
            lo = ET.SubElement(pa2, "LongitudinalAction")
            sp = ET.SubElement(lo, "SpeedAction")

            ET.SubElement(
                sp,
                "SpeedActionDynamics",
                dynamicsShape="step",
                dynamicsDimension="time",
                value="0"
            )

            tgt = ET.SubElement(sp, "SpeedActionTarget")
            ET.SubElement(
                tgt,
                "AbsoluteTargetSpeed",
                value=str(speed)
            )

        setup("vehicle1", getattr(self, "v1_lane_seed", None), self.v1_init_speed)
        setup("vehicle2", getattr(self, "v2_lane_seed", None), self.v2_init_speed)

        return init

    # ============================================================
    # TIME TRIGGER
    # ============================================================
    def _time_trigger(self, t):
        trig = ET.Element("StartTrigger")
        cg = ET.SubElement(trig, "ConditionGroup")

        cond = ET.SubElement(cg, "Condition", conditionEdge="none", delay="0")
        bv = ET.SubElement(cond, "ByValueCondition")

        ET.SubElement(
            bv,
            "SimulationTimeCondition",
            value=str(max(0.1, float(t))),
            rule="greaterThan"
        )

        return trig

    # ============================================================
    # EVENTS
    # ============================================================
    def _lane_change_event(self, idx, ch, actor):
        laneId = ch.get("laneId")
        t = ch.get("time")

        if laneId is None or int(laneId) == 0:
            return None

        ev = ET.Element(
            "Event",
            name=f"{actor}_lane_change_{idx}",
            priority="overwrite"
        )

        ev.append(self._time_trigger(t))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")
        lat = ET.SubElement(pa, "LateralAction")

        lc = ET.SubElement(lat, "LaneChangeAction")

        ET.SubElement(
            lc,
            "LaneChangeActionDynamics",
            dynamicsShape="linear",
            dynamicsDimension="time",
            value="2"
        )

        tgt = ET.SubElement(lc, "LaneChangeTarget")
        ET.SubElement(tgt, "AbsoluteTargetLane", value=str(laneId))

        return ev

    def _speed_event(self, speed, t, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_speed_hold_base",
            priority="overwrite"
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
            value="10000"
        )

        tgt = ET.SubElement(sp, "SpeedActionTarget")

        ET.SubElement(
            tgt,
            "AbsoluteTargetSpeed",
            value=str(speed)
        )

        return ev

    # ✅ NEW STOP EVENT
    def _stop_event(self, t, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_stop",
            priority="overwrite"
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
            value="1.0"
        )

        tgt = ET.SubElement(sp, "SpeedActionTarget")

        ET.SubElement(
            tgt,
            "AbsoluteTargetSpeed",
            value="0"
        )

        return ev

    def _lane_keep_event(self, actor):
        ev = ET.Element(
            "Event",
            name=f"{actor}_lane_keep",
            priority="parallel",
            maximumExecutionCount="1"
        )

        ev.append(self._time_trigger(0.2))

        act = ET.SubElement(ev, "Action")
        pa = ET.SubElement(act, "PrivateAction")

        lat = ET.SubElement(pa, "LateralAction")
        lk = ET.SubElement(lat, "LaneKeepingAction")

        ET.SubElement(
            lk,
            "LaneKeepingActionDynamics",
            dynamicsShape="step",
            dynamicsDimension="time",
            value="10000"
        )

        return ev

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

            # LaneChangeAction cannot choose another road.
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

    def _region_to_laneid(
        self,
        region_id
    ):
        try:
            meta = self.meta

            if isinstance(
                meta,
                dict
            ):
                node = meta.get(
                    str(region_id)
                )

                if not node:
                    node = meta.get(
                        region_id
                    )

                if node:
                    lane_id = node.get(
                        "lane_id"
                    )

                    if lane_id is not None:
                        return int(lane_id)

        except Exception:
            pass

        return None

    def _timeline_time_scale(self):
        duration = float(getattr(self.rules, "duration_seconds", 0.0) or 0.0)

        timeline = self.event_timeline if isinstance(self.event_timeline, list) else []
        event_end = max(
            (
                float(evt.get("t_end", evt.get("t_start", 0.0)) or 0.0)
                for evt in timeline
                if isinstance(evt, dict)
            ),
            default=0.0,
        )

        if duration > 0.0 and event_end > duration:
            return duration / event_end

        return 1.0


    def _timeline_time(self, value, default=0.2):
        try:
            raw_time = float(value)
        except (TypeError, ValueError):
            raw_time = default

        return max(0.1, raw_time * self._timeline_time_scale())

    def _region_position(self, region_id):
        if not isinstance(self.meta, dict):
            return None

        node = self.meta.get(str(region_id))
        if node is None:
            node = self.meta.get(region_id)

        if not isinstance(node, dict):
            return None

        road_id = node.get("road_id")
        lane_id = node.get("lane_id")

        if road_id is None or lane_id is None:
            return None

        try:
            lane_id = int(lane_id)
        except (TypeError, ValueError):
            return None

        if lane_id == 0:
            return None

        return {
            "roadId": str(road_id),
            "laneId": lane_id,
        }

    # ============================================================
    # BUILD EVENTS (FIXED)
    # ============================================================
    def build_events(self, actor):
        events = []
        changes = getattr(
            self,
            f"{actor}_lane_changes",
            []
        )

        #
        # Prefer timeline lane changes
        #
        timeline_changes = (
            self._timeline_lane_changes(
                actor
            )
        )

        if timeline_changes:
            changes = []

        base_speed = self.v1_init_speed if actor == "vehicle1" else self.v2_init_speed

        # ✅ single speed controller
        events.append(self._speed_event(base_speed, 0.2, actor))

        for i, tc in enumerate(
            timeline_changes
        ):

            lane_id = tc["laneId"]

            if lane_id is None:
                continue

            ev = self._lane_change_event(
                i,
                {
                    "laneId": lane_id,
                    "time": tc["time"]
                },
                actor
            )

            if ev:
                events.append(ev)

        for i, ch in enumerate(changes):
            ev = self._lane_change_event(i, ch, actor)
            if ev:
                events.append(ev)

        # ✅ stop near scenario end
        duration = getattr(self.rules, "duration_seconds", 10.0)
        events.append(self._stop_event(max(0.1, duration - 0.5), actor))

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
            maximumExecutionCount="1"
        )
        actors = ET.SubElement(mg, "Actors", selectTriggeringEntities="false")
        ET.SubElement(actors, "EntityRef", entityRef=actor)

        man = ET.SubElement(
            mg,
            "Maneuver",
            name=f"{actor}_maneuver"
        )

        if not getattr(self, f"{actor}_lane_changes", []):
            man.append(self._lane_keep_event(actor))

        for e in self.build_events(actor):
            man.append(e)

        return mg


    def parameter_declarations(self):

        pd = ET.Element(
            "ParameterDeclarations"
        )

        if not isinstance(
            self.insights,
            dict
        ):
            return pd

        for k, v in self.insights.items():

            safe_name = (
                str(k)
                .replace(" ", "_")
                .replace("-", "_")
                .replace("(", "")
                .replace(")", "")
            )

            ET.SubElement(
                pd,
                "ParameterDeclaration",
                name=safe_name,
                parameterType=(
                    "double"
                    if isinstance(v, (int, float))
                    else "string"
                ),
                value=str(v)
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
            rule="greaterThan"
        )

        act.append(self.maneuver_group("vehicle1"))
        act.append(self.maneuver_group("vehicle2"))

        stop = ET.SubElement(sb, "StopTrigger")
        cg2 = ET.SubElement(stop, "ConditionGroup")
        cond2 = ET.SubElement(cg2, "Condition", delay="0")
        bv2 = ET.SubElement(cond2, "ByValueCondition")

        ET.SubElement(
            bv2,
            "SimulationTimeCondition",
            value=str(self.rules.duration_seconds),
            rule="greaterThan"
        )

        return sb

    # ============================================================
    # SAVE
    # ============================================================
    def save(self, path):
        root = ET.Element("OpenSCENARIO")

        root.append(self.header())
        pd = self.parameter_declarations()
        if len(pd):
            root.append(pd)
        root.append(self.road_network())
        root.append(self.entities())
        root.append(self.storyboard())

        ET.ElementTree(root).write(
            path,
            encoding="utf-8",
            xml_declaration=True
        )
