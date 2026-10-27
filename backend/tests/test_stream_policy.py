from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.stream_policy import StreamPolicy


@pytest.fixture
def stream_policy() -> StreamPolicy:
    return StreamPolicy(
        SimpleNamespace(
            rtsp_allowed_networks=("10.20.0.0/16", "127.0.0.1/32", "2001:db8::/32"),
            rtsp_allowed_ports=(554, 8554, 322),
        )
    )


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://10.20.1.2/camera",
        "rtsp://viewer:p%40ss%3Aword@10.20.1.2:554/stream/one",
        "rtsp://127.0.0.1:8554/demo/retail",
        "rtsps://[2001:db8::123]/camera",
    ],
)
def test_explicit_rtsp_destinations_are_allowed(stream_policy, url):
    stream_policy.validate(SecretStr(url))


@pytest.mark.parametrize(
    "url",
    [
        "http://10.20.1.2/camera",
        "file:///etc/passwd",
        "rtsp://camera.example/stream",
        "rtsp://localhost/stream",
        "rtsp://2130706433/stream",
        "rtsp://0177.0.0.1/stream",
        "rtsp://127.0.0.2:8554/stream",
        "rtsp://169.254.169.254/latest/meta-data",
        "rtsp://0.0.0.0/stream",
        "rtsp://224.0.0.1/stream",
        "rtsp://[::ffff:127.0.0.1]/stream",
        "rtsp://[fe80::1%25eth0]/stream",
        "rtsp://10.20.1.2:22/stream",
        "rtsp://10.20.1.2:0/stream",
        "rtsp://10.20.1.2:65536/stream",
        "rtsp://10.20.1.2:invalid/stream",
        "rtsp://user:secret@10.20.1.2/stream#option",
        "rtsp://user:secret@10.20.1.2/stream?redirect=http://example.com",
        "rtsp://user:secret@10.20.1.2/stream\n-option",
        "rtsp://user:secret@10.20.1.2/stream%0A-option",
        "rtsp://user:secret@10.20.1.2/stream%GG",
        "rtsp://user:secret@10.20.1.2/stream\\option",
        "rtsp://user@10.20.1.2/stream",
        "rtsp://user:secret@10.20.1.2@10.20.1.3/stream",
        "rtsp://10.20.1.2/stréam",
    ],
)
def test_ssrf_and_ambiguous_rtsp_urls_are_rejected(stream_policy, url):
    with pytest.raises(ValueError) as result:
        stream_policy.validate(SecretStr(url))
    assert str(result.value) == "Stream destination is not allowed"
    assert "secret" not in str(result.value)


def test_no_destination_is_allowed_without_explicit_configuration():
    policy = StreamPolicy(SimpleNamespace(rtsp_allowed_networks=(), rtsp_allowed_ports=(554,)))
    with pytest.raises(ValueError):
        policy.validate("rtsp://127.0.0.1/stream")
