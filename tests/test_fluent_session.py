"""Helpers of ansys/fluent_session.py that need NO Fluent install.

These run in every normal pytest run; the real launch is tests/test_fluent_smoke.py
(RUN_FLUENT=1 only).
"""

import socket
import threading
import time

import pytest

import ansys.case0_inputs  # noqa: F401  (our modules import alongside PyFluent's namespace)
from ansys.fluent_session import (read_server_address, split_address, transcript_tail,
                                  wait_for_port)


def test_server_info_needs_address_and_password(tmp_path):
    f = tmp_path / "serverinfo-x.txt"
    assert read_server_address(f) is None                      # not written yet
    f.write_text("127.0.0.1:51740\n", encoding="utf-8")
    assert read_server_address(f) is None                      # half written
    f.write_text("127.0.0.1:51740\nsecret\n", encoding="utf-8")
    assert read_server_address(f) == "127.0.0.1:51740"         # password never returned


def test_split_address():
    assert split_address("127.0.0.1:51740") == ("127.0.0.1", 51740)
    assert split_address("[::1]:6000") == ("::1", 6000)
    with pytest.raises(ValueError):
        split_address("unix:/tmp/fluent.sock")


def test_wait_for_port_returns_once_the_server_listens():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                                   # port free: nothing listens yet
    assert not wait_for_port("127.0.0.1", port, timeout_s=0.6, interval_s=0.2)

    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def listen_later():                            # like Fluent: file first, port later
        time.sleep(0.5)
        server.bind(("127.0.0.1", port))
        server.listen()

    threading.Thread(target=listen_later, daemon=True).start()
    t0 = time.monotonic()
    try:
        assert wait_for_port("127.0.0.1", port, timeout_s=10, interval_s=0.1)
        assert time.monotonic() - t0 >= 0.4
    finally:
        server.close()


def test_wait_for_port_stops_when_the_server_process_is_gone():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    t0 = time.monotonic()
    assert not wait_for_port("127.0.0.1", port, timeout_s=30, interval_s=0.1,
                             alive=lambda: False)
    assert time.monotonic() - t0 < 5


def test_transcript_tail(tmp_path):
    assert "no Fluent transcript" in transcript_tail(tmp_path)
    (tmp_path / "fluent-1.trn").write_text("a\n\nb\nShutdown Complete.\n", encoding="utf-8")
    tail = transcript_tail(tmp_path, n_lines=2)
    assert tail.startswith("fluent-1.trn") and "b" in tail and "Shutdown Complete." in tail
