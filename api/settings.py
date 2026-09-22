"""Environment-driven configuration for the API.

Everything the server needs to know about its runtime comes from here so the
deployment can be configured without editing code.  All variables use the
``TAILGATE_`` prefix to avoid colliding with the pipeline's ``CFBD_*`` secrets.

======================  ==========================  ==========================
Variable                Default                     Meaning
======================  ==========================  ==========================
``TAILGATE_HOST``       ``127.0.0.1``               Bind address.
``TAILGATE_PORT``       ``8000``                    Bind port.
``TAILGATE_WORKERS``    ``1``                       Uvicorn worker processes.
``TAILGATE_LOG_LEVEL``  ``warning``                 Uvicorn log level.
``TAILGATE_CORS_ORIGINS`` (empty)                   Comma-separated browser
                                                    origins allowed to call the
                                                    API cross-origin. Empty
                                                    disables CORS entirely,
                                                    which is correct when
                                                    FastAPI serves the SPA.
``TAILGATE_API_KEY``    (empty)                     When set, every ``/api/*``
                                                    request except
                                                    ``/api/health`` must send
                                                    ``X-API-Key`` or
                                                    ``Authorization: Bearer``.
``TAILGATE_ADMIN_TOKEN`` (empty)                    Enables
                                                    ``POST /api/cache/clear``.
                                                    When empty the endpoint
                                                    returns 404.
``TAILGATE_WARM_CACHE`` ``true``                    Prime caches at startup.
``TAILGATE_LOGO_WIDTH`` ``400``                     Width ``/api/logo`` is
                                                    downscaled to.
======================  ==========================  ==========================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from os import environ
from typing import Mapping

__all__ = ["Settings", "load_settings", "settings", "TRUE_VALUES"]

TRUE_VALUES = {"1", "true", "yes", "on", "y", "t"}
FALSE_VALUES = {"0", "false", "no", "off", "n", "f", ""}

DEFAULT_DEV_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)


def _as_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return default


def _as_int(
    raw: str | None, default: int, *, minimum: int = 1, maximum: int | None = None
) -> int:
    """Parse an integer, falling back to ``default`` when it is out of range.

    Out-of-range values are treated as misconfiguration rather than clamped:
    a port of ``-5`` should leave the default in place, not silently become 1.
    """
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    if value < minimum:
        return default
    if maximum is not None and value > maximum:
        return default
    return value


def _as_origins(raw: str | None) -> tuple[str, ...]:
    """Parse a comma-separated origin list.

    An empty or unset value yields no origins, which disables the CORS
    middleware — the safe default when the SPA is served by this same app.
    """
    if not raw:
        return ()
    return tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 1
    log_level: str = "warning"
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    api_key: str | None = None
    admin_token: str | None = None
    warm_cache: bool = True
    logo_width: int = 400

    @property
    def cors_enabled(self) -> bool:
        return bool(self.cors_origins)

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def admin_enabled(self) -> bool:
        return bool(self.admin_token)

    def describe(self) -> dict:
        """A redacted summary, safe to log or return from /api/health."""
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "cors_origins": list(self.cors_origins),
            "auth": "api-key" if self.auth_enabled else "none",
            "admin_endpoints": "enabled" if self.admin_enabled else "disabled",
            "warm_cache": self.warm_cache,
        }

    def warnings(self) -> list[str]:
        """Deployment footguns worth surfacing at boot."""
        notes: list[str] = []
        if self.workers > 1 and self.warm_cache:
            notes.append(
                f"TAILGATE_WORKERS={self.workers} with warm cache: each worker holds its own "
                "copy of the artifact cache, so memory and startup cost scale with the worker "
                "count. Set TAILGATE_WARM_CACHE=false and let the workers warm on demand if "
                "that is too expensive."
            )
        if self.workers > 1 and self.admin_enabled is False and self.auth_enabled is False:
            notes.append(
                "No API key configured. The API is read-only, but POST /api/cache/clear is "
                "disabled and every endpoint is public — put authentication in front of it if "
                "it is reachable from an untrusted network."
            )
        if not self.admin_enabled:
            notes.append(
                "TAILGATE_ADMIN_TOKEN is unset, so POST /api/cache/clear returns 404."
            )
        return notes


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build ``Settings`` from a mapping (defaults to ``os.environ``)."""
    source = environ if env is None else env
    origins = _as_origins(source.get("TAILGATE_CORS_ORIGINS"))
    return Settings(
        host=source.get("TAILGATE_HOST", "127.0.0.1").strip() or "127.0.0.1",
        port=_as_int(source.get("TAILGATE_PORT"), 8000, minimum=1, maximum=65535),
        workers=_as_int(source.get("TAILGATE_WORKERS"), 1, maximum=64),
        log_level=(source.get("TAILGATE_LOG_LEVEL", "warning").strip() or "warning"),
        cors_origins=origins,
        api_key=(source.get("TAILGATE_API_KEY") or "").strip() or None,
        admin_token=(source.get("TAILGATE_ADMIN_TOKEN") or "").strip() or None,
        warm_cache=_as_bool(source.get("TAILGATE_WARM_CACHE"), True),
        logo_width=_as_int(source.get("TAILGATE_LOGO_WIDTH"), 400, minimum=16),
    )


settings = load_settings()
