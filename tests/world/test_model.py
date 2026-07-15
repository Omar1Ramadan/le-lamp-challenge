from uuid import UUID

from social_lamp.domain.clock import FakeClock
from social_lamp.domain.contracts import (
    AudioMode,
    ComponentHealth,
    ObservationEvent,
    ObservationSource,
    PersonState,
    SocialState,
    WorldSnapshot,
)
from social_lamp.world.commands import AudioUpdate, HealthUpdate, Source, VisionUpdate
from social_lamp.world.model import WorldModel


def test_health_update_increments_revision_on_change() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    update = HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=100)
    result = model.apply_health(update)
    assert result is not None
    assert result.revision == 1
    assert ComponentHealth(component="camera", status="ok") in result.health


def test_duplicate_health_does_not_increment_revision() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    first = model.apply_health(
        HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=100)
    )
    second = model.apply_health(
        HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=200)
    )
    assert first is not None
    assert first.revision == 1
    assert second is None


def test_distinct_health_changes_increment_revision() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    first = model.apply_health(
        HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=100)
    )
    second = model.apply_health(
        HealthUpdate(component="camera", status="degraded", detail="no signal", as_of_mono_ns=200)
    )
    assert first is not None and first.revision == 1
    assert second is not None and second.revision == 2
    ch = ComponentHealth(component="camera", status="degraded", detail="no signal")
    assert ch in second.health


def test_stale_health_update_is_ignored() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    model.apply_health(
        HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=200)
    )
    stale = model.apply_health(
        HealthUpdate(component="camera", status="degraded", detail="old", as_of_mono_ns=100)
    )
    assert stale is None


def test_stale_vision_update_is_ignored() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    first = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=200,
            source=Source.CAMERA,
        )
    )
    stale = model.apply_vision(
        VisionUpdate(
            people=(),
            social_state=SocialState.IDLE,
            primary_person_id=None,
            as_of_mono_ns=100,
            source=Source.CAMERA,
        )
    )
    assert first is not None
    assert stale is None


def test_newer_vision_update_is_accepted() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    first = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=100,
            source=Source.CAMERA,
        )
    )
    person2 = PersonState(person_id="p2", engagement_score=0.5, engagement_confidence=0.7)
    second = model.apply_vision(
        VisionUpdate(
            people=(person2,),
            social_state=SocialState.CANDIDATE,
            primary_person_id="p2",
            as_of_mono_ns=200,
            source=Source.CAMERA,
        )
    )
    assert first is not None and first.revision == 1
    assert second is not None and second.revision == 2
    assert second.social_state is SocialState.CANDIDATE


def test_meaningful_vision_change_accepted() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    result = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=100,
            source=Source.CAMERA,
        )
    )
    assert result is not None
    assert result.revision == 1
    assert result.people == (person,)
    assert result.social_state is SocialState.ENGAGED


def test_empty_vision_does_not_clear_newer_positive_people() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    positive = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=200,
            source=Source.CAMERA,
        )
    )
    assert positive is not None and len(positive.people) == 1

    empty = model.apply_vision(
        VisionUpdate(
            people=(),
            social_state=SocialState.IDLE,
            primary_person_id=None,
            as_of_mono_ns=150,
            source=Source.BROWSER,
        )
    )
    assert empty is None


def test_audio_update_preserves_people_and_objects() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    vision_result = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=100,
            source=Source.CAMERA,
        )
    )
    assert vision_result is not None
    audio_result = model.apply_audio(
        AudioUpdate(
            audio_mode=AudioMode.LISTENING,
            as_of_mono_ns=200,
            active_speaker_id=None,
        )
    )
    assert audio_result is not None
    assert audio_result.people == (person,)
    assert audio_result.audio_mode is AudioMode.LISTENING


def test_audio_update_with_speaker_sets_people() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    result = model.apply_audio(
        AudioUpdate(
            audio_mode=AudioMode.LISTENING,
            as_of_mono_ns=100,
            active_speaker_id="person-1",
            health_update=ComponentHealth(component="microphone", status="ok"),
        )
    )
    assert result is not None
    assert result.audio_mode is AudioMode.LISTENING
    assert result.people == (
        PersonState(
            person_id="person-1",
            engagement_score=0.0,
            engagement_confidence=0.0,
            is_active_speaker=True,
        ),
    )
    assert ComponentHealth(component="microphone", status="ok") in result.health


def test_vision_update_preserves_audio() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    model.apply_audio(
        AudioUpdate(
            audio_mode=AudioMode.LISTENING,
            as_of_mono_ns=100,
            active_speaker_id=None,
        )
    )
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    vision_result = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=200,
            source=Source.CAMERA,
        )
    )
    assert vision_result is not None
    assert vision_result.audio_mode is AudioMode.LISTENING
    assert vision_result.people == (person,)


def test_replay_snapshot_replacement() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    model = WorldModel(session_id=session_id, clock=clock)
    snapshot = WorldSnapshot.empty(session_id=session_id, mono_ns=999)
    replaced = model.replace_from_replay(snapshot)
    assert replaced.as_of_mono_ns == 999
    assert model.snapshot.as_of_mono_ns == 999


def test_health_dedupe_different_components_independent() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    cam = model.apply_health(
        HealthUpdate(component="camera", status="ok", detail=None, as_of_mono_ns=100)
    )
    assert cam is not None and cam.revision == 1
    mic = model.apply_health(
        HealthUpdate(component="microphone", status="ok", detail=None, as_of_mono_ns=200)
    )
    assert mic is not None and mic.revision == 2
    assert ComponentHealth(component="camera", status="ok") in mic.health
    assert ComponentHealth(component="microphone", status="ok") in mic.health


def test_vision_update_does_not_change_revision_if_noop() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    person = PersonState(person_id="p1", engagement_score=0.8, engagement_confidence=0.9)
    first = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=100,
            source=Source.CAMERA,
        )
    )
    assert first is not None and first.revision == 1
    duplicate = model.apply_vision(
        VisionUpdate(
            people=(person,),
            social_state=SocialState.ENGAGED,
            primary_person_id="p1",
            as_of_mono_ns=200,
            source=Source.CAMERA,
        )
    )
    assert duplicate is None


def test_stale_audio_update_is_ignored() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    model = WorldModel(session_id=UUID("018f0000-0000-7000-8000-000000000001"), clock=clock)
    model.apply_audio(
        AudioUpdate(audio_mode=AudioMode.LISTENING, as_of_mono_ns=200, active_speaker_id=None)
    )
    stale = model.apply_audio(
        AudioUpdate(audio_mode=AudioMode.SILENT, as_of_mono_ns=100, active_speaker_id=None)
    )
    assert stale is None


def test_apply_method_still_works() -> None:
    clock = FakeClock(100, "2026-07-04T12:00:00Z")
    session_id = UUID("018f0000-0000-7000-8000-000000000001")
    correlation_id = UUID("018f0000-0000-7000-8000-000000000002")
    model = WorldModel(session_id=session_id, clock=clock)
    event = ObservationEvent.create(
        clock=clock,
        session_id=session_id,
        correlation_id=correlation_id,
        source=ObservationSource.SYSTEM,
        kind="social_state_changed",
        confidence=0.9,
        payload={"state": "engaged", "primary_person_id": "person-a"},
    )
    changed = model.apply(event)
    unchanged = model.apply(event)
    assert changed.social_state is SocialState.ENGAGED
    assert changed.revision == 1
    assert unchanged.revision == 1
