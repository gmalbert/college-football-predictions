"""Run the API with environment-driven settings.

    python -m api                       # 127.0.0.1:8000, one worker
    TAILGATE_WORKERS=4 python -m api    # four workers
    TAILGATE_API_KEY=... python -m api  # require a key on /api/*

Equivalent to ``uvicorn api.main:app`` but reads host, port, worker count and
log level from the ``TAILGATE_*`` variables documented in ``api/settings.py``.

A note on workers: the artifact cache lives in process memory, so every worker
builds and holds its own copy.  ``TAILGATE_WORKERS=4`` therefore quadruples both
the resident memory and the startup warm-up work.  If that is too expensive, set
``TAILGATE_WARM_CACHE=false`` and let each worker populate its cache on demand.
"""
from __future__ import annotations

import sys

import uvicorn

from api.settings import settings


def main() -> int:
    for note in settings.warnings():
        print(f"[config] {note}", file=sys.stderr)
    print(f"[config] {settings.describe()}", file=sys.stderr)

    uvicorn.run(
        # The import string (not the app object) is required for workers > 1.
        "api.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
