from uuid6 import uuid7

from social_lamp.domain.contracts import (
    BehaviorIntent,
    BehaviorTimeline,
    LightKeyframe,
    MotionKeyframe,
    MotionTrack,
)


def _default_light(duration_ms: int) -> tuple[LightKeyframe, ...]:
    midpoint = duration_ms // 2
    return (
        LightKeyframe(offset_ms=0, rgb=(1.0, 0.55, 0.2), brightness=0.2),
        LightKeyframe(offset_ms=midpoint, rgb=(1.0, 0.55, 0.2), brightness=1.0),
        LightKeyframe(offset_ms=duration_ms, rgb=(1.0, 0.55, 0.2), brightness=0.3),
    )


class BehaviorCompositor:
    def compose(self, intent: BehaviorIntent, current_pose: dict[str, float]) -> BehaviorTimeline:
        timeline_id = uuid7()
        priority = intent.priority

        if intent.kind in ("return_neutral", "idle_settle"):
            tracks = _tracks(
                current_pose,
                (("head_yaw", 0.0), ("head_pitch", 0.0), ("base_yaw", 0.0)),
                duration_ms=600,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=600,
                cancellable=True,
                motion_tracks=tracks,
            )
        if intent.kind == "disengage":
            tracks = _tracks(
                current_pose,
                (("base_yaw", -0.35), ("head_yaw", -0.55), ("head_pitch", -0.45)),
                duration_ms=900,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=900,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(0.3, 0.5, 1.0), brightness=0.45),
                    LightKeyframe(offset_ms=450, rgb=(0.3, 0.5, 1.0), brightness=0.9),
                    LightKeyframe(offset_ms=900, rgb=(0.3, 0.5, 1.0), brightness=0.25),
                ),
            )
        if intent.kind in ("attention_seek", "seek_attention"):
            level = intent.parameters.get("level", 1)
            duration = 700 if level == 1 else 900
            intensity = 0.5 if level == 1 else 1.0
            targets = (
                ("base_yaw", 0.35 * intensity),
                ("head_yaw", 0.45 * intensity),
                ("head_pitch", 0.35 * intensity),
            )
            tracks = _tracks(current_pose, targets, duration_ms=duration)
            mid = duration // 2
            rgb_attn = (1.0, 0.25, 0.15)
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=duration,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=rgb_attn, brightness=0.15 * intensity),
                    LightKeyframe(offset_ms=mid, rgb=rgb_attn, brightness=0.7 * intensity),
                    LightKeyframe(offset_ms=duration, rgb=rgb_attn, brightness=0.25 * intensity),
                ),
            )
        if intent.kind in ("orient", "acknowledge_engagement", "acknowledge"):
            duration = 800 if intent.kind == "orient" else 900
            if intent.kind == "orient":
                targets = (("base_yaw", 0.25), ("head_yaw", 0.35), ("head_pitch", 0.35))
                rgb = (0.7, 0.9, 1.0)
            else:
                targets = (("head_yaw", 0.3), ("head_pitch", 0.4), ("base_yaw", 0.15))
                rgb = (0.4, 0.85, 0.5)
            tracks = _tracks(current_pose, targets, duration_ms=duration)
            midpoint = duration // 2
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=duration,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=rgb, brightness=0.2),
                    LightKeyframe(offset_ms=midpoint, rgb=rgb, brightness=0.75),
                    LightKeyframe(offset_ms=duration, rgb=rgb, brightness=0.3),
                ),
            )
        if intent.kind == "recall_success":
            tracks = _tracks(
                current_pose,
                (("head_yaw", 0.2), ("head_pitch", 0.5)),
                duration_ms=600,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=600,
                cancellable=True,
                motion_tracks=tracks,
                light_track=_default_light(600),
            )
        if intent.kind == "recall_unknown":
            tracks = _tracks(
                current_pose,
                (("head_yaw", -0.2), ("head_pitch", -0.3)),
                duration_ms=500,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=500,
                cancellable=True,
                motion_tracks=tracks,
            )
        if intent.kind == "interruption_ack":
            tracks = _tracks(
                current_pose,
                (("head_yaw", 0.4), ("head_pitch", 0.4), ("base_yaw", 0.2)),
                duration_ms=700,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=700,
                cancellable=False,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(1.0, 0.9, 0.3), brightness=0.3),
                    LightKeyframe(offset_ms=350, rgb=(1.0, 0.9, 0.3), brightness=0.9),
                    LightKeyframe(offset_ms=700, rgb=(1.0, 0.9, 0.3), brightness=0.2),
                ),
            )
        if intent.kind == "fault_notify":
            tracks = _tracks(
                current_pose,
                (("head_pitch", -0.5),),
                duration_ms=400,
            )
            return BehaviorTimeline(
                timeline_id=timeline_id,
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=priority,
                duration_ms=400,
                cancellable=False,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(1.0, 0.1, 0.1), brightness=0.5),
                    LightKeyframe(offset_ms=200, rgb=(1.0, 0.1, 0.1), brightness=1.0),
                    LightKeyframe(offset_ms=400, rgb=(1.0, 0.1, 0.1), brightness=0.0),
                ),
            )
        tracks = _tracks(
            current_pose,
            (("head_yaw", 0.45), ("head_pitch", 0.65), ("base_yaw", 0.25)),
            duration_ms=900,
        )
        return BehaviorTimeline(
            timeline_id=timeline_id,
            intent_id=intent.intent_id,
            correlation_id=intent.correlation_id,
            priority=priority,
            duration_ms=900,
            cancellable=True,
            motion_tracks=tracks,
            light_track=_default_light(900),
        )


def _tracks(
    current_pose: dict[str, float], targets: tuple[tuple[str, float], ...], *, duration_ms: int
) -> tuple[MotionTrack, ...]:
    midpoint_ms = duration_ms // 2
    return tuple(
        MotionTrack(
            channel=channel,
            keyframes=(
                MotionKeyframe(offset_ms=0, value=current_pose.get(channel, 0.0)),
                MotionKeyframe(offset_ms=midpoint_ms, value=target),
                MotionKeyframe(offset_ms=duration_ms, value=0.0),
            ),
        )
        for channel, target in targets
    )
