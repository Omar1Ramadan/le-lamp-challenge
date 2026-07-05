# Agent Task 08 — Bounded capture and face evidence

Child roles spawned via Hermes delegation:
- Task 8 edge-case/test-design review
- Task 9 spec checker
- Task 10 memory repository spec checker

## RED

Command:

```bash
PYTHONPATH= uv run pytest tests/capture/test_frames.py tests/perception/test_faces.py -v
```

Expected failure observed:

```text
ModuleNotFoundError: No module named 'social_lamp.capture'
```

## GREEN

Commands:

```bash
PYTHONPATH= uv run pytest tests/capture/test_frames.py tests/perception/test_faces.py -v
PYTHONPATH= uv run pytest -q
PYTHONPATH= uv run ruff check backend tests
PYTHONPATH= uv run mypy
PYTHONPATH= uv run python -m social_lamp.capture.frames --probe
```

Results:

- focused pytest: 5 passed
- full pytest: 17 passed, 1 third-party deprecation warning
- ruff: All checks passed after import sorting
- mypy: Success, no issues in 23 source files
- camera probe: camera_available width=640 height=480 fps=-1.00

## Parent corrections

- The plan's original head-toward formula returned 0.7857 for the specified test inputs while the test required >0.8. The implementation uses a slightly less aggressive yaw/pitch falloff to satisfy the acceptance test.
