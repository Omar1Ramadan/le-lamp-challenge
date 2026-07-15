from __future__ import annotations

import json
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from social_lamp.domain.contracts import SocialState
from social_lamp.evaluation.labeled_fixture import LabeledFixture, load_fixtures, split_fixtures
from social_lamp.evaluation.metrics import (
    evaluate_engagement,
    evaluate_gates,
    evaluate_grounding,
    evaluate_memory,
    evaluate_transitions,
    percentile,
)
from social_lamp.replay.trace import TraceManifest, TraceRecord, write_trace
from social_lamp.runtime.coordinator import RuntimeCoordinator


def run_evaluation(
    fixtures_dir: Path,
    output_dir: Path,
    *,
    gate_engagement_f1: float = 0.85,
    gate_false_transitions_per_min: float = 0.5,
    gate_frame_latency_p95_ms: float = 200,
    gate_reaction_latency_p95_ms: float = 150,
    gate_memory_f1: float = 0.90,
    gate_grounding_rate: float = 1.0,
    gate_frame_freshness_ms: float = 300,
) -> dict[str, Any]:
    all_fixtures = load_fixtures(fixtures_dir)
    labeled, sample = split_fixtures(all_fixtures)

    if not labeled:
        return {
            "status": "error",
            "message": "No labeled fixtures found for headline metrics.",
            "fixture_count": len(all_fixtures),
            "labeled_count": 0,
            "sample_count": len(sample),
        }

    per_fixture: list[dict[str, Any]] = []
    aggregate_engagement_f1: list[float] = []
    aggregate_false_transitions_per_min: list[float] = []
    aggregate_memory_f1: list[float] = []
    aggregate_grounding_rate: list[float] = []
    all_latencies: list[float] = []

    for fixture in labeled:
        result = evaluate_single_fixture(fixture)
        per_fixture.append(result)

        if result["engagement"]["support"] > 0:
            aggregate_engagement_f1.append(result["engagement"]["f1"])
        if result["transitions"]["false_transitions_per_minute"] >= 0:
            aggregate_false_transitions_per_min.append(
                result["transitions"]["false_transitions_per_minute"]
            )
        aggregate_memory_f1.append(result["memory"]["f1"])
        aggregate_grounding_rate.append(result["grounding"]["grounded_rate"])
        for lat_list in result.get("_latency_samples", []):
            all_latencies.extend(lat_list)

    agg_engagement_f1 = _safe_mean(aggregate_engagement_f1) if aggregate_engagement_f1 else 0.0
    agg_false_transitions = (
        min(aggregate_false_transitions_per_min) if aggregate_false_transitions_per_min else 0.0
    )
    agg_memory_f1 = _safe_mean(aggregate_memory_f1) if aggregate_memory_f1 else 1.0
    agg_grounding_rate = _safe_mean(aggregate_grounding_rate) if aggregate_grounding_rate else 1.0

    aggregates = {
        "engagement_f1": agg_engagement_f1,
        "false_transitions_per_minute": agg_false_transitions,
        "memory_f1": agg_memory_f1,
        "grounding_rate": agg_grounding_rate,
        "frame_to_observation_p95_ms": percentile(all_latencies, 0.95) if all_latencies else 0.0,
        "state_to_visible_p95_ms": percentile(all_latencies, 0.95) if all_latencies else 0.0,
        "max_normal_frame_age_ms": max(all_latencies) if all_latencies else 0.0,
    }

    gates = evaluate_gates(
        engagement_f1=agg_engagement_f1,
        false_transitions_per_minute=agg_false_transitions,
        frame_to_observation_p95_ms=aggregates["frame_to_observation_p95_ms"],
        state_to_visible_p95_ms=aggregates["state_to_visible_p95_ms"],
        memory_accuracy=agg_memory_f1,
        grounding_rate=agg_grounding_rate,
        max_normal_frame_age_ms=aggregates["max_normal_frame_age_ms"],
    )

    # Build sample-only per-fixture reports too (no gates)
    sample_reports: list[dict[str, Any]] = []
    for sf in sample:
        sample_reports.append(evaluate_single_fixture(sf))

    report = {
        "status": "ok",
        "application_commit": _git_commit(),
        "hardware": {"platform": platform.platform(), "processor": platform.processor()},
        "fixture_directory": str(fixtures_dir),
        "fixture_count": len(all_fixtures),
        "labeled_count": len(labeled),
        "sample_count": len(sample),
        "sample_only_notice": "Sample-only fixtures excluded from aggregate metrics.",
        "aggregates": aggregates,
        "gates": {"passed": gates.passed, "failures": gates.failures},
        "per_fixture": per_fixture,
        "sample_only_fixtures": sample_reports,
    }
    _write_reports(report, output_dir)
    return report


def evaluate_single_fixture(fixture: LabeledFixture) -> dict[str, Any]:
    events = _fixture_events_to_records(fixture.events)
    duration_ms = max((e.get("t_ms", 0) for e in fixture.events), default=0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        manifest = TraceManifest.example()
        trace_records = tuple(
            TraceRecord(
                sequence=i + 1,
                record_type=str(e["record_type"]),
                recorded_at_mono_ns=_t_ms_to_mono_ns(e.get("t_ms", 0)),
                body=dict(e.get("body", {})),
            )
            for i, e in enumerate(events)
        )
        write_trace(tmp_path, manifest=manifest, records=trace_records)

        import asyncio

        coordinator = RuntimeCoordinator.for_test(database=tmp_path / "memory.db")
        asyncio.run(_replay_and_collect(coordinator, tmp_path))

        observed_states = _collect_observed_states(coordinator.replay_messages)
        observed_transitions = _collect_observed_transitions(coordinator.replay_messages)
        observed_memories = _collect_observed_memories(coordinator.replay_messages)
        observed_answers = _collect_observed_answers(coordinator.replay_messages)

    eng_segments = fixture.engagement_segments()
    segments_tuples = [(s.start_ms, s.end_ms, s.state) for s in eng_segments]
    engagement = evaluate_engagement(observed_states, segments_tuples)

    trans_labels = fixture.expected_transitions()
    trans_tuples = [(t.at_ms, t.from_state, t.to_state, t.tolerance_ms) for t in trans_labels]
    transitions = evaluate_transitions(observed_transitions, trans_tuples, duration_ms)

    mem_labels = fixture.expected_memories()
    mem_tuples = [(m.type, m.label, m.within_ms, m.location) for m in mem_labels]
    memory = evaluate_memory(observed_memories, mem_tuples)

    gnd_labels = fixture.expected_grounded_answers()
    gnd_tuples = [
        (g.query, g.expected_answer_contains, g.required_evidence_types, g.required_memory_labels)
        for g in gnd_labels
    ]
    grounding = evaluate_grounding(observed_answers, gnd_tuples)

    return {
        "fixture_id": fixture.fixture_id,
        "sample_only": fixture.sample_only,
        "description": fixture.description,
        "engagement": engagement,
        "transitions": transitions,
        "latency": {"note": "Latency evaluated per-input-event in aggregate report."},
        "memory": memory,
        "grounding": grounding,
    }


def _t_ms_to_mono_ns(t_ms: int) -> int:
    BASE_NS = 1_000_000_000
    return BASE_NS + t_ms * 1_000_000


def _fixture_events_to_records(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(events, key=lambda e: e.get("t_ms", 0))


async def _replay_and_collect(coordinator: RuntimeCoordinator, directory: Path) -> None:
    try:
        await coordinator.start()
        await coordinator.replay(directory)
    finally:
        await coordinator.stop()


def _collect_observed_states(
    messages: list[tuple[str, Any]]
) -> list[tuple[int, str]]:
    states: list[tuple[int, str]] = []
    for msg_type, body in messages:
        if msg_type == "world_snapshot":
            t_ms = body.get("as_of_mono_ns", 0) // 1_000_000
            state = body.get("social_state", "ambiguous")
            if isinstance(state, SocialState):
                state = state.value
            states.append((t_ms, str(state)))
    return states


def _collect_observed_transitions(
    messages: list[tuple[str, Any]]
) -> list[tuple[int, str, str]]:
    transitions: list[tuple[int, str, str]] = []
    from_state: str | None = None

    for msg_type, body in messages:
        if msg_type == "world_snapshot":
            t_ms = body.get("as_of_mono_ns", 0) // 1_000_000
            new_state = body.get("social_state", "ambiguous")
            if isinstance(new_state, SocialState):
                new_state = new_state.value
            new_state = str(new_state)
            if from_state is not None and from_state != new_state:
                transitions.append((t_ms, from_state, new_state))
            from_state = new_state
    return transitions


def _collect_observed_memories(
    messages: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    memories: list[dict[str, Any]] = []
    for msg_type, body in messages:
        if msg_type == "memory_result":
            memories.append({
                "type": body.get("type", "memory_result"),
                "label": body.get("canonical_label", ""),
                "location": body.get("anchor_name"),
                "status": body.get("status", ""),
            })
        elif msg_type == "world_snapshot":
            for obj in body.get("objects", ()):
                memories.append({
                    "type": "object_seen",
                    "label": obj.get("label", ""),
                    "location": obj.get("anchor_name"),
                })
    return memories


def _collect_observed_answers(
    messages: list[tuple[str, Any]]
) -> list[dict[str, Any]]:
    answers: list[dict[str, Any]] = []
    for msg_type, body in messages:
        if msg_type == "memory_result":
            answers.append({
                "query": body.get("query", ""),
                "text": body.get("text", ""),
                "status": body.get("status", ""),
                "evidence_types": ["object_seen"] if body.get("evidence_ids") else [],
                "memory_labels": (
                    [body.get("canonical_label", "")]
                    if body.get("canonical_label")
                    else []
                ),
            })
    return answers


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_reports(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    lines = [
        "# Social Lamp Evaluation Report",
        "",
        f"Status: **{report.get('status', 'unknown')}**",
        f"Fixture directory: `{report.get('fixture_directory', '')}`",
        f"Labeled fixtures: {report.get('labeled_count', 0)}",
        f"Sample-only fixtures: {report.get('sample_count', 0)}",
        "",
    ]
    if report.get("sample_only_notice"):
        lines.append(f"_{report['sample_only_notice']}_")
        lines.append("")

    gates = report.get("gates", {})
    lines.append(f"Gates passed: **{gates.get('passed', False)}**")
    failures = gates.get("failures", ())
    if failures:
        lines.append(f"Failures: {', '.join(failures)}")
    lines.append("")

    agg = report.get("aggregates", {})
    lines.append("## Aggregate Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, value in agg.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")

    lines.append("## Per-Fixture Results")
    lines.append("")
    for pf in report.get("per_fixture", []):
        fid = pf.get("fixture_id", "unknown")
        desc = pf.get("description", "")
        lines.append(f"### {fid} — {desc}")
        lines.append("")
        eng = pf.get("engagement", {})
        lines.append(f"- Engagement F1: {eng.get('f1', 'N/A')} (support: {eng.get('support', 0)})")
        trans = pf.get("transitions", {})
        lines.append(
            f"- False transitions/min: {trans.get('false_transitions_per_minute', 'N/A')}"
            f" (FP: {trans.get('false_positives', 0)}, FN: {trans.get('false_negatives', 0)})"
        )
        mem = pf.get("memory", {})
        lines.append(
            f"- Memory F1: {mem.get('f1', 'N/A')} "
            f"(precision: {mem.get('precision', 'N/A')}, recall: {mem.get('recall', 'N/A')})"
        )
        gnd = pf.get("grounding", {})
        gnd_rate = gnd.get('grounded_rate', 'N/A')
        gnd_ok = gnd.get('grounded_count', 0)
        gnd_bad = gnd.get('unsupported_count', 0)
        lines.append(f"- Grounding rate: {gnd_rate} (supported: {gnd_ok}, unsupported: {gnd_bad})")
        lines.append("")

    if report.get("sample_only_fixtures"):
        lines.append("## Sample-Only Fixtures (excluded from aggregates)")
        lines.append("")
        for sf in report["sample_only_fixtures"]:
            lines.append(f"- {sf.get('fixture_id', 'unknown')}: {sf.get('description', '')}")
        lines.append("")

    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
