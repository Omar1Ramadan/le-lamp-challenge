from social_lamp.main import create_app


def test_application_has_expected_title() -> None:
    app = create_app()
    assert app.title == "Simulated Social Lamp"
