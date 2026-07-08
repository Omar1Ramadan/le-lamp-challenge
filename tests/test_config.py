from social_lamp.config import Settings


def test_settings_load_dotenv_file(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ENABLE_LIVE_CAPTURE", raising=False)
    monkeypatch.delenv("CAMERA_INDEX", raising=False)
    (tmp_path / ".env").write_text("ENABLE_LIVE_CAPTURE=true\nCAMERA_INDEX=2\n")

    settings = Settings()

    assert settings.enable_live_capture is True
    assert settings.camera_index == 2
