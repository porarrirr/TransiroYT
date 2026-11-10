"""Fallbacks for YouTube's sign-in / "not a bot" check.

yt-dlp's default player clients (visionos, web as of 2026.08) can be challenged
with "Sign in to confirm you're not a bot" even when the user is signed in to
YouTube in a browser: the request comes from the app, not from that browser.
The embedded web player (web_embedded) currently answers without that challenge
and still exposes caption tracks and audio-only HTTPS formats, so it is used
only after the default clients fail. Cookies selected in the app are kept for
both attempts.
"""
from __future__ import annotations

import re

FALLBACK_CLIENT = "web_embedded"

_SIGN_IN = re.compile(
    r"sign in to confirm you(?:'|’)re not a bot|confirm you(?:'|’)re not a bot|please sign in",
    re.IGNORECASE,
)

SIGN_IN_MESSAGE = (
    "YouTubeがログイン確認を求めています。設定の「Cookieの取得元」で使用中のブラウザーを選んで"
    "保存するか、時間をおいて再試行してください。"
)


def needs_sign_in(error) -> bool:
    """True only for yt-dlp's authentication challenge, not for every error."""
    return bool(_SIGN_IN.search(str(error)))
