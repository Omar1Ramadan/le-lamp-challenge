from social_lamp.audio.analysis import (
    ActiveSpeakerScorer,
    AudioAnalyzer,
    AudioClass,
    MicrophoneHealth,
    SimulatorSpeechInterruption,
    VocalAffectWindow,
    VoiceFrame,
)


def test_speech_starts_after_120_ms_and_ends_after_500_ms_silence() -> None:
    analyzer = AudioAnalyzer(frame_ms=20)
    states = [analyzer.push(VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.9)) for _ in range(6)]
    assert states[-1].speech_active
    for _ in range(24):
        state = analyzer.push(VoiceFrame(False, AudioClass.OTHER, 0.9))
    assert state.speech_active
    state = analyzer.push(VoiceFrame(False, AudioClass.OTHER, 0.9))
    assert not state.speech_active


def test_television_suppresses_unsolicited_sound() -> None:
    analyzer = AudioAnalyzer(frame_ms=20)
    state = analyzer.push(VoiceFrame(True, AudioClass.TELEVISION_MEDIA, 0.85))
    assert state.suppress_unsolicited_sound
    assert state.speaker_id is None


def test_speech_during_simulator_audio_cancels_and_listens() -> None:
    interruption = SimulatorSpeechInterruption()
    analyzer = AudioAnalyzer(frame_ms=20, interruption=interruption)
    analyzer.set_simulator_speaking(True)
    state = analyzer.push(VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.95, speaker_id="person-1"))
    assert state.listen_priority == 90
    assert interruption.cancelled
    assert interruption.reason == "human speech interrupted simulator audio"


def test_anonymous_speaker_below_threshold_is_uncertain() -> None:
    scorer = ActiveSpeakerScorer(threshold=0.65)
    assert (
        scorer.associate(
            {"person-1": {"mouth_correlation": 0.45, "visual_plausibility": 0.5, "continuity": 0.2}}
        )
        is None
    )


def test_affect_confidence_below_point_60_is_discarded() -> None:
    window = VocalAffectWindow(frame_ms=20)
    for _ in range(150):
        window.push(VoiceFrame(True, AudioClass.DIRECT_SPEECH, 0.55, speaker_id="person-1"))
    assert window.observation() is None


def test_missing_microphone_reports_unhealthy_without_device() -> None:
    health = MicrophoneHealth.from_device_available(False)
    assert health.status == "missing"
    assert "microphone" in health.detail
