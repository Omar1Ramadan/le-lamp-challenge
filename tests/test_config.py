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
