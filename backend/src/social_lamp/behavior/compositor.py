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
        if intent.kind != "acknowledge":
            return BehaviorTimeline(
                timeline_id=uuid7(),
                intent_id=intent.intent_id,
                correlation_id=intent.correlation_id,
                priority=intent.urgency,
                duration_ms=500,
                cancellable=True,
                motion_tracks=(),
            )
        tracks = tuple(
            MotionTrack(
                channel=channel,
                keyframes=(
                    MotionKeyframe(offset_ms=0, value=current_pose.get(channel, 0.0)),
                    MotionKeyframe(offset_ms=350, value=target),
                    MotionKeyframe(offset_ms=700, value=0.0),
                ),
            )
            for channel, target in (("head_yaw", 0.12), ("head_pitch", 0.25))
        )
        return BehaviorTimeline(
            timeline_id=uuid7(),
            intent_id=intent.intent_id,
            correlation_id=intent.correlation_id,
            priority=intent.urgency,
            duration_ms=700,
            cancellable=True,
            motion_tracks=tracks,
            light_track=(
                LightKeyframe(offset_ms=0, rgb=(1.0, 0.55, 0.2), brightness=0.2),
                LightKeyframe(offset_ms=350, rgb=(1.0, 0.55, 0.2), brightness=0.8),
                LightKeyframe(offset_ms=700, rgb=(1.0, 0.55, 0.2), brightness=0.3),
            ),
        )
