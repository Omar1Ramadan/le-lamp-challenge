from uuid6 import uuid7

from social_lamp.domain.contracts import (
    BehaviorIntent,
    BehaviorTimeline,
    LightKeyframe,
    MotionKeyframe,
    MotionTrack,
)


class BehaviorCompositor:
    def compose(self, intent: BehaviorIntent, current_pose: dict[str, float]) -> BehaviorTimeline:
        if intent.kind == "return_neutral":
            tracks = _tracks(
                current_pose,
                (("head_yaw", 0.0), ("head_pitch", 0.0), ("base_yaw", 0.0)),
                duration_ms=600,
            )
            return BehaviorTimeline(
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
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
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
                duration_ms=900,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(0.3, 0.5, 1.0), brightness=0.45),
                    LightKeyframe(offset_ms=450, rgb=(0.3, 0.5, 1.0), brightness=0.9),
                    LightKeyframe(offset_ms=900, rgb=(0.3, 0.5, 1.0), brightness=0.25),
                ),
            )
        if intent.kind == "seek_attention":
            tracks = _tracks(
                current_pose,
                (("base_yaw", 0.45), ("head_yaw", 0.65), ("head_pitch", 0.55)),
                duration_ms=900,
            )
            return BehaviorTimeline(
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
                duration_ms=900,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(1.0, 0.25, 0.15), brightness=0.25),
                    LightKeyframe(offset_ms=450, rgb=(1.0, 0.25, 0.15), brightness=1.0),
                    LightKeyframe(offset_ms=900, rgb=(1.0, 0.25, 0.15), brightness=0.35),
                ),
            )
        if intent.kind == "orient":
            tracks = _tracks(
                current_pose,
                (("base_yaw", 0.25), ("head_yaw", 0.35), ("head_pitch", 0.35)),
                duration_ms=800,
            )
            return BehaviorTimeline(
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
                duration_ms=800,
                cancellable=True,
                motion_tracks=tracks,
                light_track=(
                    LightKeyframe(offset_ms=0, rgb=(0.7, 0.9, 1.0), brightness=0.2),
                    LightKeyframe(offset_ms=400, rgb=(0.7, 0.9, 1.0), brightness=0.75),
                    LightKeyframe(offset_ms=800, rgb=(0.7, 0.9, 1.0), brightness=0.3),
                ),
            )
        tracks = _tracks(
            current_pose,
            (("head_yaw", 0.45), ("head_pitch", 0.65), ("base_yaw", 0.25)),
            duration_ms=900,
        )
        return BehaviorTimeline(
            timeline_id=uuid7(),
            intent_id=intent.intent_id,
            correlation_id=intent.correlation_id,
            priority=intent.urgency,
            duration_ms=900,
            cancellable=True,
            motion_tracks=tracks,
            light_track=(
                LightKeyframe(offset_ms=0, rgb=(1.0, 0.55, 0.2), brightness=0.2),
                LightKeyframe(offset_ms=450, rgb=(1.0, 0.55, 0.2), brightness=1.0),
                LightKeyframe(offset_ms=900, rgb=(1.0, 0.55, 0.2), brightness=0.3),
            ),
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
