"""Start and stop Ansys Fluent from Python (PyFluent) and confirm the link works.

Run directly:   python -m ansys.fluent_session
Run as a test:  set RUN_FLUENT=1, then python -m pytest -m fluent -s

Nothing here produces CFD results. It only proves that Python can start the
local Fluent install, talk to it, and shut it down, so later pieces can script
the Case 0 benchmark instead of clicking through the GUI.

Why launch and connect are separate steps
-----------------------------------------
``pyfluent.launch_fluent`` connects as soon as Fluent has written its
server-info file. Its first gRPC call has no retry, so if the gRPC server is
not listening yet -- e.g. on a busy laptop (the Fluent transcript reported a
system load of 23.6) -- the call fails with "connection refused" and Fluent
keeps running until its idle guard shuts it down. Here we:

1. build PyFluent's own launch command (``dry_run=True``) and start it with
   the same subprocess settings PyFluent uses;
2. wait for the server-info file, then wait until the port ACCEPTS a TCP
   connection;
3. connect with ``connect_to_fluent(server_info_file_name=...)`` and disable
   the idle-timeout guard, exactly as PyFluent's launcher does.

If any step fails, a ``FluentLaunchError`` explains what was seen (address,
whether the port ever opened, whether Fluent is still running, transcript
tail), and only the Fluent instance started here is stopped -- with the
``cleanup-fluent-*.bat`` script that this Fluent session wrote, or by ending
this process tree.

Note: the project folder ``ansys/`` has NO ``__init__.py`` on purpose. It is a
namespace-package portion, so ``ansys.fluent.core`` (PyFluent) and our
``ansys.case0_inputs`` / ``ansys.fluent_session`` can both be imported.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


class FluentLaunchError(RuntimeError):
    """Fluent could not be started or connected; the message says what was seen."""


@dataclass
class FluentCheck:
    pyfluent_version: str
    fluent_version: str
    startup_s: float
    shutdown_ok: bool
    address: str = ""
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Small helpers (no Fluent needed; unit-tested in tests/test_fluent_session.py)
# --------------------------------------------------------------------------

def read_server_address(server_info_file: Path) -> str | None:
    """First line of a Fluent server-info file ('ip:port' or 'unix:...'), or None.

    The second line (the session password) is never returned or printed.
    """
    try:
        lines = Path(server_info_file).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if len(lines) < 2 or not lines[0].strip() or not lines[1].strip():
        return None                      # not (completely) written yet
    return lines[0].strip()


def split_address(address: str) -> tuple[str, int]:
    """'127.0.0.1:51740' -> ('127.0.0.1', 51740)."""
    host, _, port = address.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"not an ip:port address: {address!r}")
    return host.strip("[]"), int(port)


def wait_for_port(host: str, port: int, timeout_s: float, interval_s: float = 0.5,
                  alive=lambda: True) -> bool:
    """True as soon as host:port accepts a TCP connection; False on timeout or
    when ``alive()`` reports that the server process has ended."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=interval_s):
                return True
        except OSError:
            if not alive():
                return False
            time.sleep(interval_s)
    return False


def transcript_tail(folder: Path, n_lines: int = 15, since: float = 0.0) -> str:
    """Last lines of the newest Fluent transcript (*.trn) in ``folder``."""
    trns = [p for p in Path(folder).glob("*.trn") if p.stat().st_mtime >= since]
    if not trns:
        return "(no Fluent transcript found)"
    newest = max(trns, key=lambda p: p.stat().st_mtime)
    lines = [ln for ln in newest.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    return f"{newest.name}:\n    " + "\n    ".join(lines[-n_lines:])


# --------------------------------------------------------------------------
# Launch / connect / stop
# --------------------------------------------------------------------------

def _import_pyfluent():
    try:
        import ansys.fluent.core as pyfluent  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the PC
        raise RuntimeError(
            "PyFluent is not installed in this environment. "
            "Run: python -m pip install -r requirements-ansys.txt"
        ) from exc
    return pyfluent


def _popen_kwargs(cwd: Path) -> dict:
    """Same subprocess settings PyFluent uses for a local Fluent (Windows/Linux)."""
    env = os.environ.copy()
    env["REMOTING_THROW_LAST_TUI_ERROR"] = "1"
    env["REMOTING_THROW_LAST_SETTINGS_ERROR"] = "1"
    kwargs = dict(cwd=str(cwd), env=env, shell=True)
    if os.name == "nt":
        kwargs["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                   | subprocess.CREATE_NO_WINDOW)
    else:
        kwargs["start_new_session"] = True
    return kwargs


def stop_started_fluent(process: subprocess.Popen | None, cwd: Path, since: float) -> str:
    """Stop only the Fluent started here: its own cleanup script, else its process tree."""
    done = []
    for bat in Path(cwd).glob("cleanup-fluent-*.bat"):
        if bat.stat().st_mtime >= since:
            subprocess.run(["cmd", "/c", str(bat)], capture_output=True, timeout=60)
            done.append(f"ran {bat.name}")
    if process is not None and process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=60)
        else:
            process.kill()
        done.append(f"ended process tree {process.pid}")
    return ", ".join(done) or "nothing left to stop"


@dataclass
class LaunchedFluent:
    session: object            # PyFluent solver session
    process: subprocess.Popen
    cwd: Path
    started_at: float
    address: str
    notes: list[str]


def launch_solver(cwd: Path, processor_count: int = 1, start_timeout_s: float = 300.0,
                  port_timeout_s: float = 120.0) -> LaunchedFluent:
    """Start Fluent (solver, no GUI) in ``cwd`` and connect once its port is open."""
    pyfluent = _import_pyfluent()
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    mode = getattr(getattr(pyfluent, "FluentMode", None), "SOLVER", "solver")
    cmd, sifile = pyfluent.launch_fluent(
        mode=mode, ui_mode="no_gui", processor_count=processor_count, cwd=str(cwd),
        start_timeout=int(start_timeout_s), dry_run=True)
    sif = Path(sifile)
    sif.unlink(missing_ok=True)
    notes: list[str] = []
    started = time.time()
    process = subprocess.Popen(cmd, **_popen_kwargs(cwd))

    def fail(reason: str) -> FluentLaunchError:
        stopped = stop_started_fluent(process, cwd, started)
        return FluentLaunchError(
            f"{reason}\n  launch command: {cmd}\n  Fluent process still running: "
            f"{process.poll() is None}\n  stopped: {stopped}\n  transcript "
            f"{transcript_tail(cwd, since=started)}")

    # 1. server-info file
    address = None
    deadline = time.monotonic() + start_timeout_s
    while time.monotonic() < deadline:
        address = read_server_address(sif)
        if address:
            break
        # The launcher may exit while Fluent runs on; only give up early if it has
        # ended AND this Fluent session never wrote its cleanup script.
        if (process.poll() is not None and time.time() - started > 60
                and not _fluent_children_alive(cwd, started)):
            raise fail(f"Fluent ended before writing its server-info file "
                       f"(launcher exit code {process.returncode})")
        time.sleep(1.0)
    if not address:
        raise fail(f"no server-info file after {start_timeout_s:.0f} s")
    notes.append(f"server-info written after {time.time() - started:.0f} s")

    # 2. port really open (Fluent may write the file before gRPC listens)
    if not address.startswith("unix:"):
        host, port = split_address(address)
        t_port = time.time()
        if not wait_for_port(host, port, port_timeout_s):
            raise fail(f"Fluent reported {host}:{port} but nothing accepted connections "
                       f"there for {port_timeout_s:.0f} s (check that no firewall / "
                       f"security tool blocks local connections of fluent.exe)")
        notes.append(f"port open after a further {time.time() - t_port:.1f} s")

    # 3. connect and switch off the idle-timeout guard (as launch_fluent does)
    try:
        session = pyfluent.connect_to_fluent(server_info_file_name=str(sif),
                                             cleanup_on_exit=True)
    except Exception as exc:  # noqa: BLE001 - report everything we know
        raise fail(f"port {address} is open but PyFluent could not connect: "
                   f"{type(exc).__name__}: {exc}") from exc
    try:
        idle_min = session.preferences.General.IdleTimeout()
        session.application_runtime.set_idle_timeout(idle_min * 60)
    except Exception as exc:  # noqa: BLE001 - not fatal for a smoke test
        notes.append(f"idle-timeout guard not reset ({type(exc).__name__}); Fluent will "
                     f"stop by itself after {int(start_timeout_s / 60) + 2} min idle")
    finally:
        sif.unlink(missing_ok=True)
    return LaunchedFluent(session, process, cwd, started, address, notes)


def _fluent_children_alive(cwd: Path, since: float) -> bool:
    """On Windows the shell may exit while Fluent runs on: a fresh cleanup script
    in ``cwd`` shows that this Fluent session is alive."""
    return any(b.stat().st_mtime >= since for b in Path(cwd).glob("cleanup-fluent-*.bat"))


def check_connection(cwd: Path) -> FluentCheck:
    """Start Fluent, read its version, shut it down; report what happened."""
    pyfluent = _import_pyfluent()
    t0 = time.perf_counter()
    launched = launch_solver(cwd)
    startup = time.perf_counter() - t0
    solver = launched.session
    version = "unknown"
    try:
        get_version = getattr(solver, "get_fluent_version", None)
        version = str(get_version()) if get_version else "unknown"
    finally:
        shutdown_ok = True
        try:
            solver.exit()
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            shutdown_ok = False
            launched.notes.append(f"exit failed: {type(exc).__name__}: {exc}")
        time.sleep(2.0)
        leftover = stop_started_fluent(launched.process, launched.cwd, launched.started_at)
        if leftover != "nothing left to stop":
            launched.notes.append(f"after exit: {leftover}")
    return FluentCheck(
        pyfluent_version=getattr(pyfluent, "__version__", "unknown"),
        fluent_version=version,
        startup_s=round(startup, 1),
        shutdown_ok=shutdown_ok,
        address=launched.address,
        notes=launched.notes,
    )


if __name__ == "__main__":
    out = Path(__file__).parent / "_fluent_runs" / "connection_check"
    result = check_connection(out)
    print(f"PyFluent version : {result.pyfluent_version}")
    print(f"Fluent version   : {result.fluent_version}")
    print(f"Address          : {result.address}")
    print(f"Startup time     : {result.startup_s} s")
    print(f"Clean shutdown   : {result.shutdown_ok}")
    for note in result.notes:
        print(f"  - {note}")
