# Auth — Google ID token + guest sessions (httpOnly cookie)

from __future__ import annotations

import os
from typing import Annotated

from fastapi import Cookie, Depends, HTTPException, Request, Response

from database import (
    claim_guest_data,
    create_guest_session_bundle,
    create_session,
    delete_session,
    get_user_by_session,
    upsert_google_user,
)

SESSION_COOKIE = "genrag_session"


def google_client_id() -> str:
    return (os.getenv("GOOGLE_CLIENT_ID") or "").strip()


def _cookie_secure() -> bool:
    explicit = os.getenv("COOKIE_SECURE")
    if explicit is not None:
        return explicit.lower() not in ("0", "false", "no")
    # HTTPS platforms (Render sets RENDER=true)
    return bool(os.getenv("RENDER"))


def _set_session_cookie(response: Response, session_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        max_age=30 * 24 * 60 * 60,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(SESSION_COOKIE, path="/")


def start_guest_session(response: Response) -> dict:
    user, session_id = create_guest_session_bundle()
    _set_session_cookie(response, session_id)
    return _public_user(user)


def start_google_session(
    response: Response,
    id_token: str,
    merge_guest: bool = True,
    guest_cookie: str | None = None,
) -> dict:
    client_id = google_client_id()
    if not client_id:
        raise HTTPException(
            status_code=503,
            detail="GOOGLE_CLIENT_ID is not configured on the server.",
        )
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        info = google_id_token.verify_oauth2_token(
            id_token,
            google_requests.Request(),
            client_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"Invalid Google token: {exc}") from exc

    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Google token missing subject.")

    user = upsert_google_user(
        google_sub=sub,
        email=info.get("email"),
        name=info.get("name"),
        picture=info.get("picture"),
    )

    if merge_guest and guest_cookie:
        guest = get_user_by_session(guest_cookie)
        if guest and guest.get("is_guest") and guest["id"] != user["id"]:
            claim_guest_data(guest["id"], user["id"])
            delete_session(guest_cookie)

    session_id = create_session(user["id"])
    _set_session_cookie(response, session_id)
    return _public_user(user)


def logout_session(response: Response, session_id: str | None) -> None:
    if session_id:
        delete_session(session_id)
    clear_session_cookie(response)


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user.get("email"),
        "name": user.get("name") or ("Guest" if user.get("is_guest") else "User"),
        "picture": user.get("picture"),
        "is_guest": bool(user.get("is_guest")),
    }


def get_optional_user(
    genrag_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> dict | None:
    if not genrag_session:
        return None
    user = get_user_by_session(genrag_session)
    return _public_user(user) if user else None


def require_user(
    genrag_session: Annotated[str | None, Cookie(alias=SESSION_COOKIE)] = None,
) -> dict:
    user = get_optional_user(genrag_session)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Sign in or continue as guest to use GenRAG.",
        )
    return user


def auth_config() -> dict:
    return {
        "google_client_id": google_client_id(),
        "google_enabled": bool(google_client_id()),
    }
