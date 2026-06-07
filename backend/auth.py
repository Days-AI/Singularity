"""Optional JWT auth dependency.

The dashboard does not yet send a token, so auth is DISABLED by default
(AUTH_ENABLED=false) and `require_auth` is a no-op. When enabled, it verifies a
Supabase-issued JWT (HS256 with the project's secret) on protected routes.
"""
from __future__ import annotations

import logging

from fastapi import Header, HTTPException

from config import get_settings

logger = logging.getLogger("singularity.auth")


async def require_auth(authorization: str | None = Header(default=None)) -> dict | None:
    settings = get_settings()
    if not settings.auth_enabled:
        return None

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    token = authorization.split(" ", 1)[1].strip()
    try:
        # jwt is a transitive dep of supabase (PyJWT). Imported lazily so the
        # backend boots even if auth is off and the lib is absent.
        import jwt  # type: ignore

        claims = jwt.decode(
            token,
            settings.app_secret_key,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )
        return claims
    except Exception as exc:  # noqa: BLE001
        logger.warning("JWT verification failed: %s", exc)
        raise HTTPException(status_code=401, detail="invalid token") from exc
