from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import aiosqlite

from social_lamp.domain.contracts import MemoryResult

MIGRATION_PATH = Path(__file__).with_name("migrations") / "001_initial.sql"


def _now_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical(label: str) -> str:
    return " ".join(label.strip().lower().split())


@dataclass(frozen=True)
class ObservationWrite:
    observation_id: str
    track_id: str
    session_id: str
    observed_at_utc: str
    observed_at_mono_ns: int
    label: str
    label_source: str
    detection_confidence: float
    bbox: tuple[float, float, float, float]
    horizontal_region: str | None
    depth_band: str | None
    anchor_name: str | None
    location_confidence: float
    frame_ref: str | None
    snapshot_path: str | None
    correlation_id: str
    aliases: tuple[str, ...] = ()

    @classmethod
    def example(
        cls,
        label: str,
        horizontal_region: str,
        *,
        observed_at_mono_ns: int = 10,
        observed_at_utc: str = "2026-07-04T12:00:00Z",
        session_id: str = "session-1",
        aliases: tuple[str, ...] = (),
    ) -> Self:
        canonical = _canonical(label)
        return cls(
            observation_id=f"observation-{canonical.replace(' ', '-')}-{observed_at_mono_ns}",
            track_id=f"track-{canonical.replace(' ', '-')}",
            session_id=session_id,
            observed_at_utc=observed_at_utc,
            observed_at_mono_ns=observed_at_mono_ns,
            label=canonical,
            label_source="test",
            detection_confidence=0.9,
            bbox=(0.1, 0.1, 0.2, 0.2),
            horizontal_region=horizontal_region,
            depth_band="foreground",
            anchor_name="desk",
            location_confidence=0.8,
            frame_ref=None,
            snapshot_path=None,
            correlation_id="00000000-0000-7000-8000-000000000001",
            aliases=aliases,
        )


@dataclass(frozen=True)
class PreferenceAuditWrite:
    context: str
    behavior: str
    outcome: str
    previous_score: float
    new_score: float
    correlation_id: str
    created_at_utc: str


class MemoryRepository:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self.fail_after_observation_insert = False

    @classmethod
    async def open(cls, path: Path) -> Self:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(path)
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.executescript(MIGRATION_PATH.read_text(encoding="utf-8"))
        await connection.execute(
            "INSERT OR IGNORE INTO schema_meta(version, migrated_at_utc) VALUES(1, ?)",
            (_now_utc(),),
        )
        await connection.commit()
        return cls(connection)

    async def close(self) -> None:
        await self._connection.close()

    async def count_observations(self) -> int:
        cursor = await self._connection.execute("SELECT COUNT(*) AS count FROM observations")
        row = await cursor.fetchone()
        if row is None:
            return 0
        return int(row["count"])

    async def clear(self) -> None:
        await self._connection.execute("BEGIN")
        try:
            for table in (
                "observation_aliases",
                "last_known_objects",
                "observations",
                "object_tracks",
                "sessions",
                "preference_audit",
                "behavior_preferences",
            ):
                await self._connection.execute(f"DELETE FROM {table}")
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def record_preference_update(
        self,
        *,
        context: str,
        behavior: str,
        outcome: str,
        previous_score: float,
        new_score: float,
        correlation_id: str,
    ) -> None:
        now = _now_utc()
        try:
            await self._connection.execute("BEGIN")
            await self._connection.execute(
                """
                INSERT INTO preference_audit(
                    context_key, behavior_key, outcome, previous_score,
                    new_score, correlation_id, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (context, behavior, outcome, previous_score, new_score, correlation_id, now),
            )
            await self._connection.execute(
                """
                INSERT INTO behavior_preferences(
                    context_key, behavior_key, score, evidence_count, updated_at_utc
                ) VALUES (?, ?, ?, 1, ?)
                ON CONFLICT(context_key, behavior_key) DO UPDATE SET
                    score = excluded.score,
                    evidence_count = behavior_preferences.evidence_count + 1,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (context, behavior, new_score, now),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def preference_score(self, context: str, behavior: str) -> float:
        cursor = await self._connection.execute(
            """
            SELECT score FROM behavior_preferences
            WHERE context_key = ? AND behavior_key = ?
            """,
            (context, behavior),
        )
        row = await cursor.fetchone()
        if row is None:
            return 1.0
        return float(row["score"])

    async def preference_audit(self) -> tuple[PreferenceAuditWrite, ...]:
        cursor = await self._connection.execute(
            """
            SELECT context_key, behavior_key, outcome, previous_score,
                   new_score, correlation_id, created_at_utc
            FROM preference_audit
            ORDER BY audit_id
            """
        )
        rows = await cursor.fetchall()
        return tuple(
            PreferenceAuditWrite(
                context=str(row["context_key"]),
                behavior=str(row["behavior_key"]),
                outcome=str(row["outcome"]),
                previous_score=float(row["previous_score"]),
                new_score=float(row["new_score"]),
                correlation_id=str(row["correlation_id"]),
                created_at_utc=str(row["created_at_utc"]),
            )
            for row in rows
        )

    async def record(self, observation: ObservationWrite) -> str:
        try:
            await self._connection.execute("BEGIN")
            await self._connection.execute(
                """
                INSERT OR IGNORE INTO sessions(
                    session_id, started_at_utc, source_mode, config_hash
                ) VALUES (?, ?, 'runtime', 'local')
                """,
                (observation.session_id, observation.observed_at_utc),
            )
            await self._connection.execute(
                """
                INSERT OR IGNORE INTO object_tracks(
                    track_id, session_id, first_seen_utc, last_seen_utc,
                    current_label, current_label_confidence, active
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    observation.track_id,
                    observation.session_id,
                    observation.observed_at_utc,
                    observation.observed_at_utc,
                    observation.label,
                    observation.detection_confidence,
                ),
            )
            await self._connection.execute(
                """
                UPDATE object_tracks
                SET last_seen_utc = ?, current_label = ?, current_label_confidence = ?
                WHERE track_id = ?
                """,
                (
                    observation.observed_at_utc,
                    observation.label,
                    observation.detection_confidence,
                    observation.track_id,
                ),
            )
            await self._connection.execute(
                """
                INSERT INTO observations(
                    observation_id, track_id, session_id, observed_at_utc,
                    observed_at_mono_ns, label, label_source, detection_confidence,
                    bbox_json, horizontal_region, depth_band, anchor_name,
                    location_confidence, frame_ref, snapshot_path, correlation_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    observation.observation_id,
                    observation.track_id,
                    observation.session_id,
                    observation.observed_at_utc,
                    observation.observed_at_mono_ns,
                    observation.label,
                    observation.label_source,
                    observation.detection_confidence,
                    json.dumps(observation.bbox),
                    observation.horizontal_region,
                    observation.depth_band,
                    observation.anchor_name,
                    observation.location_confidence,
                    observation.frame_ref,
                    observation.snapshot_path,
                    observation.correlation_id,
                ),
            )
            for alias in observation.aliases:
                await self._connection.execute(
                    """
                    INSERT INTO observation_aliases(alias, canonical_label, observation_id)
                    VALUES (?, ?, ?)
                    """,
                    (_canonical(alias), observation.label, observation.observation_id),
                )
            if self.fail_after_observation_insert:
                raise RuntimeError("injected failure")
            await self._connection.execute(
                """
                INSERT INTO last_known_objects(
                    canonical_label, observation_id, updated_at_utc
                ) VALUES (?, ?, ?)
                ON CONFLICT(canonical_label) DO UPDATE SET
                    observation_id = excluded.observation_id,
                    updated_at_utc = excluded.updated_at_utc
                """,
                (observation.label, observation.observation_id, observation.observed_at_utc),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
        return observation.observation_id

    async def find_last_seen(
        self,
        object_label: str,
        *,
        session_scope: str | None = None,
        before_utc: str | None = None,
    ) -> MemoryResult:
        label = _canonical(object_label)
        alias_cursor = await self._connection.execute(
            """
            SELECT canonical_label FROM observation_aliases
            WHERE alias = ?
            ORDER BY observation_id DESC
            LIMIT 2
            """,
            (label,),
        )
        alias_rows = await alias_cursor.fetchall()
        if alias_rows:
            labels = sorted({str(row["canonical_label"]) for row in alias_rows})
            if len(labels) > 1:
                return MemoryResult(status="ambiguous", alternatives=tuple(labels))
            label = labels[0]

        clauses = ["label = ?"]
        params: list[Any] = [label]
        if session_scope is not None:
            clauses.append("session_id = ?")
            params.append(str(session_scope))
        if before_utc is not None:
            clauses.append("observed_at_utc < ?")
            params.append(before_utc)
        cursor = await self._connection.execute(
            f"""
            SELECT * FROM observations
            WHERE {" AND ".join(clauses)}
            ORDER BY observed_at_mono_ns DESC
            LIMIT 1
            """,
            tuple(params),
        )
        row = await cursor.fetchone()
        if row is None:
            return MemoryResult.not_found()
        return MemoryResult(
            status="found",
            canonical_label=str(row["label"]),
            horizontal_region=row["horizontal_region"],
            depth_band=row["depth_band"],
            anchor_name=row["anchor_name"],
            observed_at_utc=row["observed_at_utc"],
            evidence_ids=(str(row["observation_id"]),),
        )
