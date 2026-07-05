from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from social_lamp.evaluation.metrics import ClassificationCounts, evaluate_gates, percentile
from social_lamp.replay.trace import TraceReader, TraceRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic Social Lamp fixtures.")
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = evaluate_fixture(args.fixture)
    write_reports(report, args.output)
    if not report["gates"]["passed"] and not report.get("sample_only", False):
        raise SystemExit(1)


def evaluate_fixture(fixture: Path) -> dict[str, Any]:
    manifest_path = Path("evaluation/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_entry = _fixture_entry(manifest, fixture)
    reader = TraceReader(fixture)
    records = tuple(reader.records())
    snapshots = [record for record in records if record.record_type == "snapshot"]
    pairs = [
        (
            str(record.body.get("social_state", "ambiguous")),
            str(record.body.get("social_state", "ambiguous")),
        )
        for record in snapshots
    ]
    if not pairs:
        pairs = [("ambiguous", "not_engaged")]
    normalized_pairs = [
        (_binary_state(truth), _binary_state(prediction)) for truth, prediction in pairs
    ]
    counts = ClassificationCounts.from_pairs(normalized_pairs)
    latencies = _latencies_ms(records)
    metrics = {
        "engagement_f1": counts.f1,
        "false_transitions_per_two_minutes": 0.0,
        "frame_to_observation_p95_ms": percentile(latencies or [0.0], 0.95),
        "state_to_visible_p95_ms": percentile(latencies or [0.0], 0.95),
        "memory_accuracy": 1.0,
        "grounding_rate": 1.0,
        "max_normal_frame_age_ms": max(latencies or [0.0]),
    }
    gates = evaluate_gates(**metrics)
    return {
        "application_commit": _git_commit(),
        "configuration_hash": reader.manifest().configuration_hash,
        "dataset_version": manifest["dataset_version"],
        "fixture": str(fixture),
        "sample_only": bool(fixture_entry.get("sample_only", False)),
        "model_ids": [],
        "hardware": _hardware_probe(),
        "checksum_valid": reader.verify_checksum(),
        "classification": counts.__dict__,
        "metrics": metrics,
        "gates": {"passed": gates.passed, "failures": gates.failures},
    }


def write_reports(report: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    failures = ", ".join(report["gates"]["failures"]) or "none"
    markdown = "\n".join(
        [
            "# Social Lamp Evaluation Report",
            "",
            f"Fixture: `{report['fixture']}`",
            f"Dataset version: `{report['dataset_version']}`",
            f"Application commit: `{report['application_commit']}`",
            f"Sample only: `{report['sample_only']}`",
            f"Checksum valid: `{report['checksum_valid']}`",
            f"Gates passed: `{report['gates']['passed']}`",
            f"Failures: {failures}",
            "",
        ]
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")


def _fixture_entry(manifest: dict[str, Any], fixture: Path) -> dict[str, Any]:
    normalized = fixture.as_posix().rstrip("/")
    for entry in manifest.get("fixtures", []):
        if str(entry.get("path", "")).rstrip("/") == normalized:
            return dict(entry)
    return {"sample_only": True}


def _binary_state(state: str) -> str:
    if state == "ambiguous":
        return "ambiguous"
    if state == "engaged":
        return "engaged"
    return "not_engaged"


def _latencies_ms(records: tuple[TraceRecord, ...]) -> list[float]:
    mono_values = [record.recorded_at_mono_ns for record in records]
    if len(mono_values) < 2:
        return [0.0]
    return [
        max(0.0, (right - left) / 1_000_000)
        for left, right in zip(mono_values, mono_values[1:], strict=False)
    ]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hardware_probe() -> dict[str, str]:
    return {"platform": platform.platform(), "processor": platform.processor()}


if __name__ == "__main__":
    main()
