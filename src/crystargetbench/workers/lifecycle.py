"""Stop a session process group when its supervising CTB process disappears."""
import os
import signal
import threading


def start_parent_watchdog(expected_parent=None, poll_seconds=0.1):
    """Linux/POSIX parent reparenting guard; no shell, prctl, or model imports.

    Popen uses start_new_session=True. Killing that group removes descendants
    as well as this worker when the supervisor dies without normal cancellation.
    The check is also immediate, covering parent death during worker startup.
    """
    raw = expected_parent if expected_parent is not None else os.environ.get("CTB_PARENT_PID")
    if raw is None:
        raise ValueError("CTB_PARENT_PID supervisor identity is required")
    expected = int(raw)
    if expected < 2 or os.getpgrp() != os.getpid():
        raise ValueError("worker requires an isolated process group and explicit parent")
    stopped = threading.Event()
    def check():
        if os.getppid() != expected:
            os.killpg(os.getpgrp(), signal.SIGKILL)
    check()
    def watch():
        while not stopped.wait(poll_seconds):
            check()
    thread = threading.Thread(target=watch, daemon=True, name="ctb-parent-watchdog")
    thread.start()
    return stopped
