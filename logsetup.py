"""Structured logging: one rotating file per app (APP_DIR/logs), one logger per module.

    from logsetup import get_logger
    log = get_logger("worker")            # -> logger "chessoverlay.worker"

`setup_logging()` is called once from main.py. Status messages for the GUI travel through the
logger "chessoverlay.status"; `CallbackHandler` forwards them to the GUI and its level *is* the
status level filter (Settings.status_level), while the file always gets everything.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Callable

from config import APP_DIR

ROOT = "chessoverlay"
LOG_DIR = APP_DIR / "logs"
LOG_FILE = "chess_overlay.log"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s [%(threadName)s] %(message)s"
STATUS_LOGGER = ROOT + ".status"

_configured = False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}")


def parse_level(value, default: int = logging.INFO) -> int:
    if isinstance(value, int):
        return value
    lvl = logging.getLevelName(str(value or "").strip().upper())
    return lvl if isinstance(lvl, int) else default


class CallbackHandler(logging.Handler):
    """Forwards the formatted message of every record that passes `level` to `callback`."""

    def __init__(self, callback: Callable[[str], None], level: int = logging.INFO):
        super().__init__(level)
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._callback(record.getMessage())
        except Exception:                      # a broken GUI callback must never kill the pipeline
            self.handleError(record)


def setup_logging(level=None, log_dir: Path | None = None) -> Path | None:
    """Idempotent. Returns the log file path (None if only console logging could be set up)."""
    global _configured
    root = logging.getLogger(ROOT)
    if _configured:
        return getattr(root, "_log_path", None)
    lvl = parse_level(level or os.environ.get("CHESS_OVERLAY_LOG") or "INFO")
    root.setLevel(logging.DEBUG)               # handlers decide what is shown; the file keeps DEBUG at lvl
    root.propagate = False
    fmt = logging.Formatter(FORMAT)
    path = None
    try:
        d = Path(log_dir) if log_dir else LOG_DIR
        d.mkdir(parents=True, exist_ok=True)
        path = d / LOG_FILE
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        fh.setLevel(lvl)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        path = None                            # read-only profile etc.: console only
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    root._log_path = path                      # type: ignore[attr-defined]
    _configured = True

    def _thread_hook(args):                    # uncaught exception in any thread -> log file
        get_logger("threads").critical("Uncaught exception in thread %s", getattr(args.thread, "name", "?"),
                                       exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    def _main_hook(exc_type, exc, tb):
        get_logger("main").critical("Uncaught exception", exc_info=(exc_type, exc, tb))
        sys.__excepthook__(exc_type, exc, tb)

    threading.excepthook = _thread_hook
    sys.excepthook = _main_hook
    get_logger("logsetup").info("Logging to %s (level %s)", path, logging.getLevelName(lvl))
    return path


def reset_for_tests() -> None:
    """Tear down handlers (selftest only)."""
    global _configured
    root = logging.getLogger(ROOT)
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    _configured = False
