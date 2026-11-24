# explanation_generator.py
"""
Human explanation of a scenario.

The "SCENARIO SIGNATURE" and "EVENT SEQUENCE" sections mirror exactly
what build_scenario_descriptor indexes for similarity search:

    S{shared_lane}_M{merge}_D{diverge}_J{junction}
    || route V1 | route V2
    || ONE:waiting -> AB:proximity -> ONE:lane_change

so the text an analyst reads is the same semantic sequence the
search matches on.
"""

import math

_EVENT_VERBS = {
    "waiting": "waits (standing still)",
    "lane_change": "changes lane",
    "proximity": "drives in close proximity",
    "merge": "merges into traffic",
    "yield": "yields",
    "overtake": "overtakes another vehicle",
    "avoidance": "performs an avoidance manoeuvre",
}

_MOTIF_NAMES = {
    "WAIT_MERGE": "waiting before merge",
    "WAIT_YIELD": "waiting before yield",
    "PROX_NEAR_MISS": "proximity leading to near miss",
    "PROX_AVOIDANCE": "proximity leading to avoidance",
}


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

    @staticmethod
    def _has_tag(tags, needle):
        """Substring match over tag strings.

        (Old code used set membership, so e.g. 'complex_route_navigation'
        in tags never matched the real subtype
        'trace_complex_route_navigation'.)
        """
        return any(needle in t for t in tags)

    def _descriptor(self, ctx):
        return (ctx.get("derived") or {}).get("scenario_descriptor") or {}

    def _actors_text(self, parts):
        if len(parts) >= 2:
            return "Both vehicles"
        if parts:
            return {
                "vehicle1": "Vehicle 1",
                "vehicle2": "Vehicle 2",
            }.get(str(parts[0]), str(parts[0]))
        return "Vehicle"

    def _shared_junction_fraction(self, ctx):
        it = ctx.get("interaction_topology") or {}
        frames = it.get("shared_junction_frames")
        pair_tl = (ctx.get("interaction_facts") or {}).get("timeline") or {}
        if frames and pair_tl:
            return len(frames) / len(pair_tl)
        return None

    # =========================================================
    # ENTRY
    # =========================================================
    def explain(self, payload):
        if isinstance(payload, dict):
            rules = payload.get("derived_rules", [])
            ctx = payload.get("context", {}) or {}
            opts = payload.get("options", {}) or {}
        elif isinstance(payload, (list, tuple)):
            rules = payload[0] if len(payload) > 0 else []
            ctx = payload[1] if len(payload) > 1 else {}
            opts = payload[2] if len(payload) > 2 else {}
        else:
            rules, ctx, opts = [], {}, {}

        rules = [
            r if isinstance(r, dict) else {
                "type": r.type,
                "parameters": getattr(r, "parameters", {}) or {},
                "actor": getattr(r, "actor", "both")
            }
            for r in rules
        ]

        parts = []
        parts += self._signature_block(ctx)      # NEW: the search key, humanized
        parts += self._event_sequence(ctx)       # NEW: ordered semantic sequence
        parts += self._global_summary(ctx)       # fixed: real field names
        parts += self._event_summary(ctx)        # compact counts
        parts += self._rule_timeline(rules, ctx, opts)  # honors only_fired
        parts += self._insights(ctx)             # fixed: real engine outputs

        return "\n".join(self._deduplicate(parts)) or "No notable interactions detected."

    # =========================================================
    # SEARCH SIGNATURE (NEW)
    # =========================================================
    def _signature_block(self, ctx):
        out = []
        d = self._descriptor(ctx)
        topo = d.get("topology") or {}

        slf = topo.get("shared_lane_fraction")
        m = topo.get("common_merge_count", 0) or 0
        dd = topo.get("common_diverge_count", 0) or 0
        j = topo.get("shared_junction_count", 0) or 0

        out.append("=" * 64)
        out.append("SCENARIO SIGNATURE (what similarity search matches)")
        out.append("=" * 64)

        if isinstance(slf, (int, float)) and math.isfinite(slf):
            out.append(
                f"Topology: S{slf:.2f}_M{m}_D{dd}_J{j}   "
                f"(same-lane {slf * 100:.0f}%, {m} shared merge, "
                f"{dd} shared diverge, {j} shared-junction frames)"
            )
        else:
            out.append(f"Topology: M{m} D{dd} J{j}")

        # Per-actor route chains. (The search index stores these as a
        # sorted set; this view keeps actor identity.)
        actors = ctx.get("actors") or {}
        for label, key in (("V1", "A"), ("V2", "B")):
            route = (
                actors.get(key, {})
                .get("route_semantics", {})
                .get("transitions", [])
            ) or []
            out.append(
                f"Route {label}: "
                + (" -> ".join(str(t) for t in route)
                   if route else "(no resolved lanes)")
            )

        ss = (d.get("search_index") or {}).get("semantic_signature") or {}
        chain = ss.get("event_chain") or []
        out.append(
            "Event chain: " + (" -> ".join(chain) if chain else "(none yet)")
        )

        bigrams = (d.get("search_index") or {}).get("transitions") or []
        if bigrams:
            out.append("Transitions: " + ", ".join(bigrams))

        motifs = (d.get("search_index") or {}).get("motifs") or []
        if motifs:
            out.append(
                "Motifs: "
                + ", ".join(f"{mo} = {_MOTIF_NAMES.get(mo, mo)}" for mo in motifs)
            )

        sig = d.get("signature")
        if sig:
            out.append(f"Raw signature: {sig}")

        return out

    # =========================================================
    # SEMANTIC EVENT SEQUENCE (NEW)
    # =========================================================
    def _event_sequence(self, ctx):
        events = (
            ctx.get("_canonical_event_timeline")
            or ctx.get("event_timeline")
            or []
        )
        events = [
            e for e in events
            if isinstance(e, dict) and e.get("type")
        ]

        out = []
        out.append("-" * 64)
        out.append("EVENT SEQUENCE (semantic, time-ordered)")
        out.append("-" * 64)

        if not events:
            out.append(
                "(no events yet - press 'Generate Rule Suggestions' "
                "to build the event sequence)"
            )
            return out

        # Same ordering and actor grouping as build_scenario_descriptor:
        # sort by t_start, AB: if >= 2 participants else ONE:.
        events = sorted(
            events,
            key=lambda e: (e.get("t_start") or 0.0)
        )

        for i, e in enumerate(events, 1):
            etype = str(e.get("type"))
            parts = e.get("participants") or []
            token = "AB" if len(parts) >= 2 else "ONE"

            desc = f"{self._actors_text(parts)} {_EVENT_VERBS.get(etype, etype)}"

            if etype == "lane_change":
                fr = e.get("from_region")
                to = e.get("to_region")
                if fr is None or to is None:
                    metrics = e.get("metrics") or {}
                    if fr is None:
                        fr = metrics.get("from_lane")
                    if to is None:
                        to = metrics.get("to_lane")
                if fr is not None and to is not None and fr != to:
                    desc += f" ({fr} -> {to})"

            t0 = e.get("t_start")
            t1 = e.get("t_end")
            if isinstance(t0, (int, float)):
                if isinstance(t1, (int, float)) and abs(t1 - t0) > 0.05:
                    time_txt = f"t={t0:.2f}s-{t1:.2f}s"
                else:
                    time_txt = f"t={t0:.2f}s"
            else:
                time_txt = "t=?"

            out.append(
                f"{i:2d}. {time_txt:<22s} {desc:<46s} [{token}:{etype}]"
            )

        return out

    # =========================================================
    # GLOBAL SUMMARY (fixed: reads fields the engine actually emits)
    # =========================================================
    def _global_summary(self, ctx):
        out = []

        duration = ctx.get("duration_seconds")
        if isinstance(duration, (int, float)) and duration > 0:
            out.append(f"Scenario duration: {duration:.2f}s.")

        # real key is shared_lane_fraction (flattened from rule_inputs)
        slf = ctx.get("shared_lane_fraction")
        if isinstance(slf, (int, float)) and math.isfinite(slf):
            out.append(
                f"Same-lane exposure: {slf * 100:.0f}% of aligned frames."
            )

        if ctx.get("shared_junction"):
            frac = self._shared_junction_fraction(ctx)
            if frac is not None:
                out.append(
                    f"Vehicles share a junction for "
                    f"{frac * 100:.0f}% of the scenario."
                )
            else:
                n = (ctx.get("interaction_topology") or {}).get(
                    "shared_junction_count"
                )
                if isinstance(n, int):
                    out.append(
                        f"Vehicles share a junction for {n} frames."
                    )

        min_dist = ctx.get("min_distance")
        if isinstance(min_dist, (int, float)) and math.isfinite(min_dist):
            out.append(f"Minimum separation: {min_dist:.2f} m.")

        min_ttc = ctx.get("min_ttc")
        if isinstance(min_ttc, (int, float)) and math.isfinite(min_ttc):
            out.append(f"Minimum TTC: {min_ttc:.2f} s.")

        return out

    # =========================================================
    # EVENT SUMMARY (compact counts - order is shown above)
    # =========================================================
    def _event_summary(self, ctx):
        events = ctx.get("event_timeline", [])
        if not events:
            return []

        counts = {}
        for e in events:
            t = e.get("type")
            counts[t] = counts.get(t, 0) + 1

        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        return ["Detected events: " + ", ".join(parts) + "."]

    # =========================================================
    # RULE TIMELINE (substring tag matching + only_fired)
    # =========================================================
    def _rule_timeline(self, rules, ctx, opts=None):
        opts = opts or {}
        only_fired = bool(opts.get("only_fired"))

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
            if only_fired and not isinstance(g["start_t"], (int, float)):
                continue

            start_t = g["start_t"]
            end_t = g["end_t"]

            actor_txt = actor_map.get(g["actor"], str(g["actor"]))
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

            # ---- timing ----
            if isinstance(start_t, (int, float)):
                if (
                    isinstance(end_t, (int, float))
                    and abs(end_t - start_t) > 0.05
                ):
                    header = f"{start_t:.2f}s-{end_t:.2f}s"
                else:
                    header = f"{start_t:.2f}s"
            else:
                header = "Unknown time"

            # ---- main maneuver (first match wins; check specific
            #      subtypes before their generic substrings) ----
            if self._has_tag(tags, "aggressive_lane_change"):
                maneuver = "changes lane aggressively"
            elif self._has_tag(tags, "lane_change"):
                maneuver = "changes lane"
            elif self._has_tag(tags, "cut_in"):
                maneuver = "cuts in"
            elif self._has_tag(tags, "merge"):
                maneuver = "merges into traffic"
            elif self._has_tag(tags, "yield"):
                maneuver = "yields"
            elif self._has_tag(tags, "gap"):
                maneuver = "waits for a gap"
            elif self._has_tag(tags, "overtake"):
                maneuver = "overtakes another vehicle"
            elif self._has_tag(tags, "near_miss"):
                maneuver = "passes in a near miss"
            elif self._has_tag(tags, "avoidance"):
                maneuver = "performs an avoidance manoeuvre"
            elif self._has_tag(tags, "proximity"):
                maneuver = "drives in close proximity"
            elif self._has_tag(tags, "junction"):
                maneuver = "crosses a shared junction"
            elif self._has_tag(tags, "unstable"):
                maneuver = "shows unstable lane positioning"
            elif self._has_tag(tags, "multi_stage"):
                maneuver = "repositions across multiple stages"
            elif self._has_tag(tags, "connector"):
                maneuver = "travels through connector lanes"
            elif self._has_tag(tags, "complex"):
                maneuver = "navigates a complex route"
            else:
                maneuver = "performs a manoeuvre"

            line = f"{header}: {actor_txt} {maneuver}"

            # ---- location ----
            loc = []
            if road_id is not None:
                loc.append(f"road {road_id}")
            if lane_id is not None:
                loc.append(f"lane {lane_id}")
            if region is not None:
                loc.append(f"region {region}")
            if loc:
                line += " (" + ", ".join(loc) + ")"

            # ---- lane transition ----
            p = g["params"]
            from_lane = p.get("from_lane")
            to_lane = p.get("to_lane")
            if (
                from_lane is not None
                and to_lane is not None
                and from_lane != to_lane
            ):
                line += f" [{from_lane} -> {to_lane}]"

            # ---- classification summary ----
            labels = []
            if self._has_tag(tags, "aggressive"):
                labels.append("aggressive")
            if self._has_tag(tags, "cut_in"):
                labels.append("cut-in")
            if self._has_tag(tags, "multi_stage"):
                labels.append("multi-stage")
            if self._has_tag(tags, "unstable"):
                labels.append("unstable")
            if self._has_tag(tags, "connector"):
                labels.append("connector")
            if self._has_tag(tags, "complex"):
                labels.append("complex-route")
            if self._has_tag(tags, "near_miss"):
                labels.append("near-miss")
            if self._has_tag(tags, "avoidance"):
                labels.append("avoidance")
            if self._has_tag(tags, "gap"):
                labels.append("gap-based")
            if self._has_tag(tags, "junction"):
                labels.append("junction")

            if labels:
                line += " - " + ", ".join(labels)

            rows.append(line)

        return rows

    # =========================================================
    # INSIGHTS (fixed: only fields the engine actually produces)
    # =========================================================
    def _insights(self, ctx):
        out = []

        slf = ctx.get("shared_lane_fraction")
        if isinstance(slf, (int, float)) and slf >= 0.5:
            out.append(
                "Vehicles spend most of the scenario on the same lane."
            )

        it = ctx.get("interaction_topology") or {}
        m = len(it.get("common_merge_points", []) or [])
        d = len(it.get("common_diverge_points", []) or [])
        if m:
            out.append(
                f"Both routes pass through {m} shared merge point(s)."
            )
        if d:
            out.append(
                f"Both routes pass through {d} shared diverge point(s)."
            )

        if ctx.get("shared_junction"):
            out.append(
                "Interaction occurs within a shared junction environment."
            )

        conv = ctx.get("interaction_convergence")
        if isinstance(conv, (int, float)) and conv > 0.5:
            out.append(
                "Separation shrank in most steps - a converging approach."
            )

        cross = ctx.get("interaction_lateral_cross")
        if isinstance(cross, (int, float)) and cross > 0:
            out.append(
                "Lateral crossing between the vehicles was detected."
            )

        flips = ctx.get("interaction_sign_flip")
        if isinstance(flips, (int, float)) and flips > 0:
            out.append(
                f"Longitudinal ordering flipped {int(flips)} time(s) "
                f"(overtaking manoeuvre(s))."
            )

        wd = ctx.get("waiting_duration_s")
        if isinstance(wd, (int, float)) and wd >= 2.0:
            out.append(f"Standing-still time totals {wd:.1f}s.")

        return out
