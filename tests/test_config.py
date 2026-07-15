from social_lamp.config import Settings


def test_settings_load_dotenv_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENABLE_LIVE_CAPTURE", raising=False)
    monkeypatch.delenv("ENABLE_MEDIAPIPE_FACE_LANDMARKER", raising=False)
    monkeypatch.delenv("CAMERA_INDEX", raising=False)
    (tmp_path / ".env").write_text(
        "ENABLE_LIVE_CAPTURE=true\nENABLE_MEDIAPIPE_FACE_LANDMARKER=true\nCAMERA_INDEX=2\n"
    )

    settings = Settings()

    assert settings.enable_live_capture is True
    assert settings.enable_mediapipe_face_landmarker is True
    assert settings.camera_index == 2


def test_mediapipe_face_landmarker_defaults_off() -> None:
    settings = Settings(_env_file=None)

    assert settings.enable_mediapipe_face_landmarker is False


def test_face_detector_mode_defaults_to_auto() -> None:
    settings = Settings(_env_file=None)
    assert settings.face_detector_mode == "auto"


def test_legacy_mediapipe_env_var_sets_mode_to_mediapipe() -> None:
    settings = Settings(_env_file=None, enable_mediapipe_face_landmarker=True)
    assert settings.face_detector_mode == "mediapipe"


def test_explicit_face_detector_mode_takes_priority_over_legacy() -> None:
    settings = Settings(
        _env_file=None,
        enable_mediapipe_face_landmarker=True,
        face_detector_mode="opencv",
    )
    assert settings.face_detector_mode == "opencv"


def test_all_face_detector_modes_are_accepted() -> None:
    for mode in ("auto", "mediapipe", "opencv", "heuristic", "disabled"):
        settings = Settings(_env_file=None, face_detector_mode=mode)
        assert settings.face_detector_mode == mode
