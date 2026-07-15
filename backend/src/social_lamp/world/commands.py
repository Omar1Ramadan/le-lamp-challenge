from social_lamp.domain.contracts import (
    AudioMode,
    ComponentHealth,
    ObjectState,
    PersonState,
    SocialState,
)


class Source:
    CAMERA = "camera"
    BROWSER = "browser"
    AUDIO = "audio"
    SYSTEM = "system"


class VisionUpdate:
    __slots__ = (
        "people",
        "social_state",
        "primary_person_id",
        "as_of_mono_ns",
        "source",
        "objects",
        "health_updates",
    )

    def __init__(
        self,
        *,
        people: tuple[PersonState, ...],
        social_state: SocialState,
        primary_person_id: str | None,
        as_of_mono_ns: int,
        source: str,
        objects: tuple[ObjectState, ...] | None = None,
        health_updates: tuple[ComponentHealth, ...] = (),
    ) -> None:
        self.people = people
        self.social_state = social_state
        self.primary_person_id = primary_person_id
        self.as_of_mono_ns = as_of_mono_ns
        self.source = source
        self.objects = objects
        self.health_updates = health_updates


class AudioUpdate:
    __slots__ = (
        "audio_mode",
        "as_of_mono_ns",
        "source",
        "active_speaker_id",
        "health_update",
    )

    def __init__(
        self,
        *,
        audio_mode: AudioMode,
        as_of_mono_ns: int,
        source: str = Source.AUDIO,
        active_speaker_id: str | None = None,
        health_update: ComponentHealth | None = None,
    ) -> None:
        self.audio_mode = audio_mode
        self.as_of_mono_ns = as_of_mono_ns
        self.source = source
        self.active_speaker_id = active_speaker_id
        self.health_update = health_update


class HealthUpdate:
    __slots__ = ("component", "status", "detail", "as_of_mono_ns", "source")

    def __init__(
        self,
        *,
        component: str,
        status: str,
        detail: str | None,
        as_of_mono_ns: int,
        source: str = "health",
    ) -> None:
        self.component = component
        self.status = status
        self.detail = detail
        self.as_of_mono_ns = as_of_mono_ns
        self.source = source
