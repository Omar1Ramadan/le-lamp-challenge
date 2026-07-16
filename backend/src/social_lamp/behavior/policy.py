from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from uuid6 import uuid7

from social_lamp.behavior.preferences import PreferenceModel
from social_lamp.domain.contracts import (
    BehaviorDecision,
    BehaviorIntent,
    SocialState,
    WorldSnapshot,
)


@dataclass(frozen=True)
class AttentionIntent:
    parameters: dict[str, int]


class AttentionSchedule:
    OFFSETS_NS = (5_000_000_000, 15_000_000_000, 35_000_000_000)
    RESTART_COOLDOWN_NS = 60_000_000_000

    def __init__(self, *, disengaged_at_ns: int) -> None:
        self._start = disengaged_at_ns
        self._emitted = 0
        self.exhausted = False
        self.suppression_reason: str | None = None
        self._exhausted_at_ns: int | None = None

    def suppress(self, reason: str, *, mono_ns: int | None = None) -> None:
        self.suppression_reason = reason
        self.exhausted = True
        self._exhausted_at_ns = self._start if mono_ns is None else mono_ns

    def intent_at(self, mono_ns: int) -> AttentionIntent | None:
        if self.exhausted or self._emitted >= len(self.OFFSETS_NS):
            return None
        due = self._start + self.OFFSETS_NS[self._emitted]
        if mono_ns < due:
            return None
        self._emitted += 1
        level = self._emitted
        if level == len(self.OFFSETS_NS):
            self.exhausted = True
            self._exhausted_at_ns = mono_ns
        return AttentionIntent(parameters={"level": level})

    def can_restart_at(self, mono_ns: int) -> bool:
        if not self.exhausted:
            return False
        exhausted_at = self._exhausted_at_ns if self._exhausted_at_ns is not None else self._start
        return mono_ns >= exhausted_at + self.RESTART_COOLDOWN_NS


class BehaviorPolicy:
    PRIORITY: dict[str, int] = {
        "fault_notify": 100,
        "interruption_ack": 90,
        "recall_success": 80,
        "recall_unknown": 75,
        "acknowledge_engagement": 70,
        "attention_seek": 40,
        "idle_settle": 10,
        "return_neutral": 5,
    }

    SOFT_PROMPT_AFTER_IDLE_NS = 30_000_000_000
    STRONG_PROMPT_AFTER_IDLE_NS = 120_000_000_000
    MAX_PROMPTS_PER_10_MIN = 3
    TEN_MIN_NS = 600_000_000_000
    COOLDOWN_AFTER_IGNORE_NS = 180_000_000_000

    STATE_TRANSITION_MAP: dict[SocialState, tuple[str, int]] = {
        SocialState.CANDIDATE: ("orient", 50),
        SocialState.ENGAGED: ("acknowledge_engagement", 70),
        SocialState.DISENGAGED: ("disengage", 60),
        SocialState.SEEKING_ATTENTION: ("attention_seek", 40),
        SocialState.IDLE: ("idle_settle", 10),
    }

    def __init__(
        self, preferences: PreferenceModel | None = None, *, deterministic: bool = False
    ) -> None:
        self._preferences = preferences or PreferenceModel()
        self._deterministic = deterministic
        self._decisions: list[BehaviorDecision] = []
        self._timeline_by_kind: dict[str, str] = {}
        self._attention_prompt_times: list[int] = []
        self._idle_since_mono_ns: int | None = None
        self._suppressed_until_mono_ns: int | None = None
        self._last_decision_mono_ns: int | None = None

    def _make_id(self) -> UUID:
        return uuid7()

    def reset_idle_timer(self, mono_ns: int) -> None:
        self._idle_since_mono_ns = mono_ns

    def clear_idle_timer(self) -> None:
        self._idle_since_mono_ns = None

    def record_active_timeline(self, kind: str, timeline_id: str) -> None:
        self._timeline_by_kind[kind] = timeline_id

    def clear_timeline(self, kind: str) -> None:
        self._timeline_by_kind.pop(kind, None)

    def clear_all_timelines(self) -> None:
        self._timeline_by_kind.clear()

    @property
    def decisions(self) -> list[BehaviorDecision]:
        return list(self._decisions)

    @property
    def idle_since_ns(self) -> int | None:
        return self._idle_since_mono_ns

    def _make_intent(
        self,
        kind: str,
        snapshot: WorldSnapshot,
        *,
        target_person_id: str | None = None,
        parameters: dict[str, Any] | None = None,
        replace_policy: str = "replace_lower_priority",
        cancellation_reason: str | None = None,
        source_event_ids: tuple[str, ...] = (),
    ) -> BehaviorIntent:
        prio = self.PRIORITY.get(kind, 50)
        now = snapshot.as_of_mono_ns
        duration_sec = 2
        return BehaviorIntent(
            intent_id=self._make_id(),
            correlation_id=self._make_id(),
            session_id=snapshot.session_id,
            kind=kind,
            priority=prio,
            created_at_mono_ns=now,
            expires_at_mono_ns=now + duration_sec * 1_000_000_000,
            target_person_id=target_person_id or snapshot.primary_person_id,
            cancellable=True,
            replace_policy=replace_policy,
            suppression_reason=cancellation_reason,
            source_event_ids=source_event_ids,
            parameters=parameters or {},
        )

    def _suppressed(
        self,
        reason: str,
        *,
        intent: BehaviorIntent | None = None,
        active_timeline_id: str | None = None,
        replacement: str | None = None,
    ) -> BehaviorDecision:
        decision = BehaviorDecision(
            intent=intent,
            suppressed=True,
            suppression_reason=reason,
            active_timeline_id=active_timeline_id,
            replacement=replacement,
        )
        self._decisions.append(decision)
        return decision

    def _selected(
        self,
        intent: BehaviorIntent,
        *,
        active_timeline_id: str | None = None,
        replacement: str | None = None,
    ) -> BehaviorDecision:
        decision = BehaviorDecision(
            intent=intent,
            suppressed=False,
            suppression_reason=None,
            active_timeline_id=active_timeline_id,
            replacement=replacement,
        )
        self._decisions.append(decision)
        return decision

    def evaluate(
        self,
        snapshot: WorldSnapshot,
        previous_social_state: SocialState | None = None,
        *,
        mono_ns: int | None = None,
        active_timeline_id: str | None = None,
        active_timeline_kind: str | None = None,
        active_timeline_priority: int = 0,
        active_timeline_cancellable: bool = True,
        has_visible_person: bool = True,
        audio_suppressed: bool = False,
    ) -> BehaviorDecision:
        now = mono_ns if mono_ns is not None else snapshot.as_of_mono_ns
        self._last_decision_mono_ns = now

        did_transition = (
            previous_social_state is not None
            and previous_social_state != snapshot.social_state
        )

        candidate: BehaviorIntent | None = None

        if did_transition:
            transition_kind = self.STATE_TRANSITION_MAP.get(snapshot.social_state)
            if transition_kind is not None:
                kind, _ = transition_kind
                candidate = self._make_intent(kind, snapshot)

        visible_kinds = {"attention_seek", "acknowledge_engagement", "orient"}
        attention_intent = self._check_attention(
            snapshot, now, has_visible_person, audio_suppressed
        )
        if attention_intent is not None:
            attn_prio = self.PRIORITY.get(attention_intent.kind, 0)
            if candidate is not None and attn_prio > self.PRIORITY.get(candidate.kind, 0):
                candidate = attention_intent
            elif candidate is None:
                candidate = attention_intent

        if candidate is None:
            return self._suppressed("no_candidate_intent")

        candidate_prio = self.PRIORITY.get(candidate.kind, 0)

        if audio_suppressed and candidate.kind != "fault_notify":
            return self._suppressed("background_media", intent=candidate)

        if not has_visible_person and candidate.kind in visible_kinds:
            return self._suppressed("no_visible_person", intent=candidate)

        if candidate.kind in self._timeline_by_kind:
            existing_id = self._timeline_by_kind[candidate.kind]
            if candidate.replace_policy == "drop":
                return self._suppressed(
                    "already_active", intent=candidate, active_timeline_id=existing_id
                )
            if candidate.replace_policy == "ignore_if_active":
                return self._suppressed(
                    "same_kind_already_active", intent=candidate, active_timeline_id=existing_id
                )

        if active_timeline_id is not None and active_timeline_kind is not None:
            if not active_timeline_cancellable:
                return self._suppressed(
                    "active_timeline_not_cancellable",
                    intent=candidate,
                    active_timeline_id=active_timeline_id,
                )
            if candidate_prio < active_timeline_priority:
                return self._suppressed(
                    "lower_priority",
                    intent=candidate,
                    active_timeline_id=active_timeline_id,
                )
            same_kind = candidate.kind == active_timeline_kind
            if candidate_prio <= active_timeline_priority and same_kind:
                if candidate.replace_policy != "replace_same_kind":
                    return self._suppressed(
                        "same_kind_equal_priority",
                        intent=candidate,
                        active_timeline_id=active_timeline_id,
                    )

        replacement = None
        if active_timeline_id is not None and active_timeline_kind != candidate.kind:
            replacement = f"replaced_{active_timeline_kind}"

        return self._selected(
            candidate,
            active_timeline_id=active_timeline_id,
            replacement=replacement,
        )

    def _check_attention(
        self,
        snapshot: WorldSnapshot,
        mono_ns: int,
        has_visible_person: bool,
        audio_suppressed: bool,
    ) -> BehaviorIntent | None:
        if snapshot.social_state in (SocialState.ENGAGED, SocialState.CANDIDATE):
            self._idle_since_mono_ns = None
            return None

        if snapshot.social_state == SocialState.SEEKING_ATTENTION:
            return None

        if self._suppressed_until_mono_ns is not None and mono_ns < self._suppressed_until_mono_ns:
            return None

        if self._idle_since_mono_ns is None or self._idle_since_mono_ns > mono_ns:
            self._idle_since_mono_ns = mono_ns
            return None

        elapsed_since_idle = mono_ns - self._idle_since_mono_ns

        if elapsed_since_idle < self.SOFT_PROMPT_AFTER_IDLE_NS:
            return None

        cutoff = mono_ns - self.TEN_MIN_NS
        recent_prompts = [t for t in self._attention_prompt_times if t >= cutoff]
        if len(recent_prompts) >= self.MAX_PROMPTS_PER_10_MIN:
            return None

        if not has_visible_person and elapsed_since_idle < self.STRONG_PROMPT_AFTER_IDLE_NS:
            return None

        level = 1 if elapsed_since_idle < self.STRONG_PROMPT_AFTER_IDLE_NS else 2
        self._attention_prompt_times.append(mono_ns)
        kind = "attention_seek"
        return self._make_intent(
            kind,
            snapshot,
            parameters={"level": level},
        )

    def record_attention_outcome(
        self, outcome: str, *, mono_ns: int | None = None, correlation_id: str | None = None
    ) -> None:
        context = "attention_seek"
        behavior = "motion"
        self._preferences.record(context, behavior, outcome, correlation_id=correlation_id)
        score = self._preferences.score(context, behavior)
        if outcome == "rejected" or outcome == "ignored":
            now = mono_ns if mono_ns is not None else 0
            cooldown = int(self.COOLDOWN_AFTER_IGNORE_NS * (2.0 - score))
            self._suppressed_until_mono_ns = now + cooldown
