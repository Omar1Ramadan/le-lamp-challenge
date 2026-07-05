import numpy as np
from social_lamp.capture.frames import LatestFrameBuffer


def test_latest_frame_replaces_oldest_without_queueing() -> None:
    buffer = LatestFrameBuffer(capacity=3)
    for index in range(5):
        buffer.put(np.full((2, 2, 3), index, dtype=np.uint8), mono_ns=index)
    latest = buffer.latest()
    assert latest is not None
    assert latest.mono_ns == 4
    assert int(latest.image[0, 0, 0]) == 4
    assert buffer.dropped == 2


def test_latest_frame_is_copied_on_write() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    buffer = LatestFrameBuffer(capacity=1)
    buffer.put(image, mono_ns=1)
    image[0, 0, 0] = 255
    latest = buffer.latest()
    assert latest is not None
    assert int(latest.image[0, 0, 0]) == 0
