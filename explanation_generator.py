class RuleExplanationGenerator:

    # =========================================================
    # UTIL
    # =========================================================
    def _deduplicate(self, lines):
        seen, out = set(), []
        for l in lines:
            k = l.strip().lower()
            if k not in seen:
                seen.add(k)
                out.append(l)
        return out

    def _poly_len(self, poly):
        return sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(poly, poly[1:]))

    # =========================================================
    # ENTRY
    # =========================================================
    def explain(self, payload):
        if isinstance(payload, dict):
            rules = payload.get("derived_rules", [])
            ctx = payload.get("context", {}) or {}
            opts = payload.get("options", {}) or {}
        else:
            rules, ctx, opts = payload or [], {}, {}

        rules = [
            r if isinstance(r, dict) else {
                "type": r.type,
                "parameters": getattr(r, "parameters", {}) or {},
                "actor": getattr(r, "actor", "both")
            }
            for r in rules
        ]

        parts = []
        parts += self._global_summary(ctx)
        parts += self._event_summary(ctx)
        parts += self._rule_timeline(rules, ctx)
        parts += self._insights(ctx)

        return "\n".join(self._deduplicate(parts)) or "No notable interactions detected."

    # =========================================================
    # GLOBAL SUMMARY
    # =========================================================
    def _global_summary(self, ctx):
        out = []

        duration = ctx.get("duration_seconds")

        if duration:
            out.append(f"Scenario duration: {duration:.2f}s.")

        rel = ctx.get("path_relationship")
        if rel:
            out.append(f"Path relationship: {rel}.")

        same_lane = ctx.get("same_lane_fraction")
        if isinstance(same_lane, (int, float)):
            out.append(
                f"Same-lane exposure: {same_lane*100:.0f}% of aligned frames."
            )

        if ctx.get("shared_junction"):
            frac = ctx.get("shared_junction_fraction", 0.0)
            out.append(
                f"Vehicles share a junction for {frac*100:.0f}% of the scenario."
            )

        min_dist = ctx.get("min_distance")
        if isinstance(min_dist, (int, float)) and math.isfinite(min_dist):
            out.append(
                f"Minimum separation: {min_dist:.2f} m."
            )

        min_ttc = ctx.get("min_ttc")
        if isinstance(min_ttc, (int, float)) and math.isfinite(min_ttc):
            out.append(
                f"Minimum TTC: {min_ttc:.2f} s."
            )

        return out

    # =========================================================
    # EVENT SUMMARY
    # =========================================================
    def _event_summary(self, ctx):

        out = []

        events = ctx.get("event_timeline", [])

        if not events:
            return out

        counts = {}

        for e in events:
            t = e.get("type")
            counts[t] = counts.get(t, 0) + 1

        parts = [
            f"{v} {k}"
            for k, v in sorted(counts.items())
        ]

        out.append(
            "Detected events: " + ", ".join(parts) + "."
        )

        return out

    # =========================================================
    # RULE TIMELINE
    # =========================================================
    def _rule_timeline(self, rules, ctx):

        meta = ctx.get("meta", {}) or {}

        groups = {}

        for r in rules:

            params = r.get("parameters", {}) or {}

            actor = r.get("actor", "both")

            start_t = params.get("start_t")
            end_t = params.get("end_t")

            key = (
                actor,
                round(start_t or -1, 2),
                round(end_t or start_t or -1, 2)
            )

            groups.setdefault(
                key,
                {
                    "actor": actor,
                    "start_t": start_t,
                    "end_t": end_t,
                    "region": params.get("region"),
                    "params": params,
                    "tags": []
                }
            )

            subtype = params.get("subtype")

            if subtype:
                groups[key]["tags"].append(subtype)

        rows = []

        actor_map = {
            "vehicle1": "Vehicle 1",
            "vehicle2": "Vehicle 2",
            "both": "Both vehicles"
        }

        for _, g in sorted(
            groups.items(),
            key=lambda x: (
                x[1]["start_t"]
                if isinstance(x[1]["start_t"], (int, float))
                else float("inf")
            )
        ):

            start_t = g["start_t"]
            end_t = g["end_t"]

            actor_txt = actor_map.get(
                g["actor"],
                str(g["actor"])
            )

            region = g["region"]

            road_id = None
            lane_id = None

            if region is not None:
                entry = meta.get(str(region), {})

                if isinstance(entry, dict):
                    road_id = (
                        entry.get("roadId")
                        or entry.get("road_id")
                        or entry.get("road")
                    )

                    lane_id = (
                        entry.get("laneId")
                        or entry.get("lane_id")
                        or entry.get("lane")
                    )

            tags = set(g["tags"])

            # --------------------------------------------------
            # Timing
            # --------------------------------------------------
            if isinstance(start_t, (int, float)):

                if (
                    isinstance(end_t, (int, float))
                    and abs(end_t - start_t) > 0.05
                ):
                    header = f"{start_t:.2f}s–{end_t:.2f}s"
                else:
                    header = f"{start_t:.2f}s"

            else:
                header = "Unknown time"

            # --------------------------------------------------
            # Main maneuver
            # --------------------------------------------------
            maneuver = None

            if (
                "lane_change" in tags
                or "aggressive_lane_change" in tags
                or "cut_in" in tags
            ):
                maneuver = "changes lane"

            elif "merge" in tags:
                maneuver = "merges into traffic"

            elif "yield" in tags:
                maneuver = "yields"

            elif "overtake" in tags:
                maneuver = "overtakes another vehicle"

            elif "avoidance" in tags:
                maneuver = "performs an avoidance manoeuvre"

            elif "connector_navigation" in tags:
                maneuver = "travels through connector lanes"

            elif "complex_route_navigation" in tags:
                maneuver = "navigates a complex route"

            else:
                maneuver = "performs a manoeuvre"

            line = f"{header}: {actor_txt} {maneuver}"

            # --------------------------------------------------
            # Location
            # --------------------------------------------------
            loc = []

            if road_id is not None:
                loc.append(f"road {road_id}")

            if lane_id is not None:
                loc.append(f"lane {lane_id}")

            if region is not None:
                loc.append(f"region {region}")

            if loc:
                line += " (" + ", ".join(loc) + ")"

            # --------------------------------------------------
            # Lane transition
            # --------------------------------------------------
            p = g["params"]

            from_lane = p.get("from_lane")
            to_lane = p.get("to_lane")

            if (
                from_lane is not None
                and to_lane is not None
                and from_lane != to_lane
            ):
                line += f" [{from_lane} → {to_lane}]"

            # --------------------------------------------------
            # Classification summary
            # --------------------------------------------------
            labels = []

            if "aggressive_lane_change" in tags:
                labels.append("aggressive")

            if "cut_in" in tags:
                labels.append("cut-in")

            if "multi_stage_lane_positioning" in tags:
                labels.append("multi-stage")

            if "connector_navigation" in tags:
                labels.append("connector")

            if "complex_route_navigation" in tags:
                labels.append("complex-route")

            if "near_miss" in tags:
                labels.append("near-miss")

            if "near_miss_prob" in tags:
                labels.append("near-miss indicators")

            if "avoidance" in tags:
                labels.append("avoidance")

            if labels:
                line += " — " + ", ".join(labels)

            rows.append(line)

        return rows
    # =========================================================
    # INSIGHTS (NEW VERSION)
    # =========================================================
    def _insights(self, ctx):

        out = []

        rel = ctx.get("path_relationship")

        if rel == "parallel":
            out.append(
                "Vehicles follow independent paths with limited interaction potential."
            )

        elif rel == "overlapping":
            out.append(
                "Vehicles share portions of the same route."
            )

        elif rel == "converging":
            out.append(
                "Vehicles move toward a common route segment."
            )

        elif rel == "crossing":
            out.append(
                "Vehicle paths intersect."
            )

        pnm = ctx.get("prob_near_miss")
        pav = ctx.get("prob_avoidance")
        pov = ctx.get("prob_overtake")

        if isinstance(pnm, (int, float)):
            if pnm > 0.7:
                out.append("Near-miss likelihood is high.")
            elif pnm > 0.4:
                out.append("Near-miss likelihood is moderate.")

        if isinstance(pav, (int, float)) and pav > 0.7:
            out.append("Strong avoidance indicators detected.")

        if isinstance(pov, (int, float)) and pov > 0.7:
            out.append("Overtaking behaviour is probable.")

        if ctx.get("shared_junction"):
            out.append(
                "Interaction occurs within a shared junction environment."
            )

        return out