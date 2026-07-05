from pathlib import Path


def test_delivery_files_cover_required_operations() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    privacy = Path("docs/PRIVACY.md").read_text(encoding="utf-8")
    demo = Path("docs/DEMO.md").read_text(encoding="utf-8")
    limitations = Path("docs/LIMITATIONS.md").read_text(encoding="utf-8")
    assert "uv sync" in readme and "pnpm install" in readme
    assert "offline" in readme.lower() and "replay" in readme.lower()
    assert "seven days" in privacy.lower() and "clear all memory" in privacy.lower()
    assert all(
        step in demo
        for step in (
            "Engagement",
            "Attention seeking",
            "Memory formation",
            "Memory recall",
        )
    )
    assert "monocular" in limitations.lower() and "session-only" in limitations.lower()
